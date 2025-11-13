"""
External Factors Management Service.
Handles external factor data that can influence forecasts.
"""

import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import logging

from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, NotFoundException
from app.schemas.forecasting import ExternalFactorCreate, ExternalFactorUpdate

logger = logging.getLogger(__name__)


class ExternalFactorsService:
    """Service for external factors operations."""

    @staticmethod
    def create_factor(
        tenant_id: str,
        database_name: str,
        request: ExternalFactorCreate,
        user_email: str
    ) -> Dict[str, Any]:
        """Create a new external factor record."""
        factor_id = str(uuid.uuid4())
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        INSERT INTO external_factors
                        (factor_id, tenant_id, date, factor_name, factor_value,
                         unit, source, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        factor_id,
                        tenant_id,
                        request.date,
                        request.factor_name,
                        request.factor_value,
                        request.unit,
                        request.source,
                        user_email
                    ))

                    conn.commit()
                    logger.info(f"External factor created: {factor_id}")

                    return ExternalFactorsService.get_factor(
                        tenant_id, database_name, factor_id
                    )

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to create external factor: {str(e)}")
            raise DatabaseException(f"Failed to create external factor: {str(e)}")

    @staticmethod
    def get_factor(
        tenant_id: str,
        database_name: str,
        factor_id: str
    ) -> Dict[str, Any]:
        """Get external factor by ID."""
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT factor_id, tenant_id, date, factor_name, factor_value,
                               unit, source, created_at, updated_at, created_by, updated_by
                        FROM external_factors
                        WHERE factor_id = %s AND tenant_id = %s AND deleted_at IS NULL
                    """, (factor_id, tenant_id))

                    result = cursor.fetchone()
                    if not result:
                        raise NotFoundException("External Factor", factor_id)

                    return {
                        "factor_id": str(result[0]),
                        "tenant_id": str(result[1]),
                        "date": result[2].isoformat() if result[2] else None,
                        "factor_name": result[3],
                        "factor_value": float(result[4]),
                        "unit": result[5],
                        "source": result[6],
                        "created_at": result[7].isoformat() if result[7] else None,
                        "updated_at": result[8].isoformat() if result[8] else None,
                        "created_by": result[9],
                        "updated_by": result[10]
                    }

                finally:
                    cursor.close()

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to get external factor: {str(e)}")
            raise DatabaseException(f"Failed to get external factor: {str(e)}")

    @staticmethod
    def update_factor(
        tenant_id: str,
        database_name: str,
        factor_id: str,
        request: ExternalFactorUpdate,
        user_email: str
    ) -> Dict[str, Any]:
        """Update external factor."""
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check factor exists
                    cursor.execute("""
                        SELECT factor_id FROM external_factors
                        WHERE factor_id = %s AND tenant_id = %s AND deleted_at IS NULL
                    """, (factor_id, tenant_id))
                    
                    if not cursor.fetchone():
                        raise NotFoundException("External Factor", factor_id)

                    # Build update query
                    update_fields = []
                    params = []

                    if request.factor_value is not None:
                        update_fields.append("factor_value = %s")
                        params.append(request.factor_value)

                    if request.unit is not None:
                        update_fields.append("unit = %s")
                        params.append(request.unit)

                    if request.source is not None:
                        update_fields.append("source = %s")
                        params.append(request.source)

                    if not update_fields:
                        return ExternalFactorsService.get_factor(
                            tenant_id, database_name, factor_id
                        )

                    update_fields.extend(["updated_at = %s", "updated_by = %s"])
                    params.extend([datetime.utcnow(), user_email, factor_id, tenant_id])

                    query = f"""
                        UPDATE external_factors
                        SET {', '.join(update_fields)}
                        WHERE factor_id = %s AND tenant_id = %s
                    """

                    cursor.execute(query, params)
                    conn.commit()
                    logger.info(f"External factor updated: {factor_id}")

                    return ExternalFactorsService.get_factor(
                        tenant_id, database_name, factor_id
                    )

                finally:
                    cursor.close()

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to update external factor: {str(e)}")
            raise DatabaseException(f"Failed to update external factor: {str(e)}")

    @staticmethod
    def list_factors(
        tenant_id: str,
        database_name: str,
        factor_name: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List external factors with optional filters."""
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Build WHERE clause
                    where_clauses = ["tenant_id = %s", "deleted_at IS NULL"]
                    params = [tenant_id]

                    if factor_name:
                        where_clauses.append("factor_name = %s")
                        params.append(factor_name)

                    if date_from:
                        where_clauses.append("date >= %s")
                        params.append(date_from)

                    if date_to:
                        where_clauses.append("date <= %s")
                        params.append(date_to)

                    where_sql = " AND ".join(where_clauses)

                    # Get total count
                    cursor.execute(
                        f"SELECT COUNT(*) FROM external_factors WHERE {where_sql}",
                        params
                    )
                    total_count = cursor.fetchone()[0]

                    # Get paginated results
                    offset = (page - 1) * page_size
                    cursor.execute(f"""
                        SELECT factor_id, date, factor_name, factor_value, unit,
                               source, created_at, created_by
                        FROM external_factors
                        WHERE {where_sql}
                        ORDER BY date DESC, factor_name
                        LIMIT %s OFFSET %s
                    """, params + [page_size, offset])

                    factors = []
                    for row in cursor.fetchall():
                        factors.append({
                            "factor_id": str(row[0]),
                            "date": row[1].isoformat() if row[1] else None,
                            "factor_name": row[2],
                            "factor_value": float(row[3]),
                            "unit": row[4],
                            "source": row[5],
                            "created_at": row[6].isoformat() if row[6] else None,
                            "created_by": row[7]
                        })

                    return factors, total_count

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to list external factors: {str(e)}")
            raise DatabaseException(f"Failed to list external factors: {str(e)}")

    @staticmethod
    def delete_factor(
        tenant_id: str,
        database_name: str,
        factor_id: str
    ) -> bool:
        """Soft delete an external factor."""
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        UPDATE external_factors
                        SET deleted_at = %s
                        WHERE factor_id = %s AND tenant_id = %s AND deleted_at IS NULL
                    """, (datetime.utcnow(), factor_id, tenant_id))

                    if cursor.rowcount == 0:
                        raise NotFoundException("External Factor", factor_id)

                    conn.commit()
                    logger.info(f"External factor deleted: {factor_id}")
                    return True

                finally:
                    cursor.close()

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete external factor: {str(e)}")
            raise DatabaseException(f"Failed to delete external factor: {str(e)}")

    @staticmethod
    def get_factors_for_period(
        tenant_id: str,
        database_name: str,
        start_date: date,
        end_date: date,
        factor_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all external factors for a specific time period.
        Useful for forecast execution.
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    where_clauses = [
                        "tenant_id = %s",
                        "deleted_at IS NULL",
                        "date >= %s",
                        "date <= %s"
                    ]
                    params = [tenant_id, start_date, end_date]

                    if factor_names:
                        placeholders = ', '.join(['%s'] * len(factor_names))
                        where_clauses.append(f"factor_name IN ({placeholders})")
                        params.extend(factor_names)

                    where_sql = " AND ".join(where_clauses)

                    cursor.execute(f"""
                        SELECT factor_id, date, factor_name, factor_value, unit, source
                        FROM external_factors
                        WHERE {where_sql}
                        ORDER BY date, factor_name
                    """, params)

                    factors = []
                    for row in cursor.fetchall():
                        factors.append({
                            "factor_id": str(row[0]),
                            "date": row[1].isoformat() if row[1] else None,
                            "factor_name": row[2],
                            "factor_value": float(row[3]),
                            "unit": row[4],
                            "source": row[5]
                        })

                    return factors

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to get factors for period: {str(e)}")
            raise DatabaseException(f"Failed to get factors for period: {str(e)}")