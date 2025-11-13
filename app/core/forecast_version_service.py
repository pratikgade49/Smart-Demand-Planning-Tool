"""
Forecast Version Management Service.
Handles version creation, updates, and activation.
"""

import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from app.core.database import get_db_manager
from app.core.exceptions import (
    DatabaseException,
    ValidationException,
    NotFoundException,
    ConflictException
)
from app.schemas.forecasting import ForecastVersionCreate, ForecastVersionUpdate

logger = logging.getLogger(__name__)


class ForecastVersionService:
    """Service for forecast version operations."""

    @staticmethod
    def create_version(
        tenant_id: str,
        database_name: str,
        request: ForecastVersionCreate,
        user_email: str
    ) -> Dict[str, Any]:
        """Create a new forecast version."""
        version_id = str(uuid.uuid4())
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check if version name already exists for this tenant
                    cursor.execute("""
                        SELECT version_id FROM forecast_versions
                        WHERE tenant_id = %s AND version_name = %s
                    """, (tenant_id, request.version_name))
                    
                    if cursor.fetchone():
                        raise ConflictException(
                            f"Version name '{request.version_name}' already exists"
                        )

                    # If setting as active, deactivate other versions of same type
                    if request.is_active:
                        cursor.execute("""
                            UPDATE forecast_versions
                            SET is_active = FALSE, updated_at = %s, updated_by = %s
                            WHERE tenant_id = %s AND version_type = %s AND is_active = TRUE
                        """, (datetime.utcnow(), user_email, tenant_id, request.version_type))

                    # Insert new version
                    cursor.execute("""
                        INSERT INTO forecast_versions
                        (version_id, tenant_id, version_name, version_type, 
                         is_active, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        version_id,
                        tenant_id,
                        request.version_name,
                        request.version_type,
                        request.is_active,
                        user_email
                    ))

                    conn.commit()
                    logger.info(f"Forecast version created: {version_id}")

                    # Fetch and return created version
                    return ForecastVersionService.get_version(
                        tenant_id, database_name, version_id
                    )

                finally:
                    cursor.close()

        except (ConflictException, ValidationException):
            raise
        except Exception as e:
            logger.error(f"Failed to create forecast version: {str(e)}")
            raise DatabaseException(f"Failed to create version: {str(e)}")

    @staticmethod
    def get_version(
        tenant_id: str,
        database_name: str,
        version_id: str
    ) -> Dict[str, Any]:
        """Get forecast version by ID."""
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT version_id, tenant_id, version_name, version_type,
                               is_active, created_at, updated_at, created_by, updated_by
                        FROM forecast_versions
                        WHERE version_id = %s AND tenant_id = %s
                    """, (version_id, tenant_id))

                    result = cursor.fetchone()
                    if not result:
                        raise NotFoundException("Forecast Version", version_id)

                    return {
                        "version_id": str(result[0]),
                        "tenant_id": str(result[1]),
                        "version_name": result[2],
                        "version_type": result[3],
                        "is_active": result[4],
                        "created_at": result[5].isoformat() if result[5] else None,
                        "updated_at": result[6].isoformat() if result[6] else None,
                        "created_by": result[7],
                        "updated_by": result[8]
                    }

                finally:
                    cursor.close()

        except NotFoundException:
            raise
        except Exception as e:
            logger.error(f"Failed to get forecast version: {str(e)}")
            raise DatabaseException(f"Failed to get version: {str(e)}")

    @staticmethod
    def update_version(
        tenant_id: str,
        database_name: str,
        version_id: str,
        request: ForecastVersionUpdate,
        user_email: str
    ) -> Dict[str, Any]:
        """Update forecast version."""
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check version exists
                    cursor.execute("""
                        SELECT version_type, version_name 
                        FROM forecast_versions
                        WHERE version_id = %s AND tenant_id = %s
                    """, (version_id, tenant_id))
                    
                    result = cursor.fetchone()
                    if not result:
                        raise NotFoundException("Forecast Version", version_id)
                    
                    version_type, current_name = result

                    # Check if new name conflicts
                    if request.version_name and request.version_name != current_name:
                        cursor.execute("""
                            SELECT version_id FROM forecast_versions
                            WHERE tenant_id = %s AND version_name = %s AND version_id != %s
                        """, (tenant_id, request.version_name, version_id))
                        
                        if cursor.fetchone():
                            raise ConflictException(
                                f"Version name '{request.version_name}' already exists"
                            )

                    # If activating, deactivate other versions of same type
                    if request.is_active is True:
                        cursor.execute("""
                            UPDATE forecast_versions
                            SET is_active = FALSE, updated_at = %s, updated_by = %s
                            WHERE tenant_id = %s AND version_type = %s 
                            AND is_active = TRUE AND version_id != %s
                        """, (datetime.utcnow(), user_email, tenant_id, version_type, version_id))

                    # Build update query
                    update_fields = []
                    params = []

                    if request.version_name is not None:
                        update_fields.append("version_name = %s")
                        params.append(request.version_name)

                    if request.is_active is not None:
                        update_fields.append("is_active = %s")
                        params.append(request.is_active)

                    if not update_fields:
                        # No updates requested, just return current version
                        return ForecastVersionService.get_version(
                            tenant_id, database_name, version_id
                        )

                    update_fields.extend(["updated_at = %s", "updated_by = %s"])
                    params.extend([datetime.utcnow(), user_email, version_id, tenant_id])

                    query = f"""
                        UPDATE forecast_versions
                        SET {', '.join(update_fields)}
                        WHERE version_id = %s AND tenant_id = %s
                    """

                    cursor.execute(query, params)
                    conn.commit()
                    logger.info(f"Forecast version updated: {version_id}")

                    # Fetch and return updated version
                    return ForecastVersionService.get_version(
                        tenant_id, database_name, version_id
                    )

                finally:
                    cursor.close()

        except (NotFoundException, ConflictException):
            raise
        except Exception as e:
            logger.error(f"Failed to update forecast version: {str(e)}")
            raise DatabaseException(f"Failed to update version: {str(e)}")

    @staticmethod
    def list_versions(
        tenant_id: str,
        database_name: str,
        version_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        page: int = 1,
        page_size: int = 50
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List forecast versions with optional filters."""
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Build WHERE clause
                    where_clauses = ["tenant_id = %s"]
                    params = [tenant_id]

                    if version_type:
                        where_clauses.append("version_type = %s")
                        params.append(version_type)

                    if is_active is not None:
                        where_clauses.append("is_active = %s")
                        params.append(is_active)

                    where_sql = " AND ".join(where_clauses)

                    # Get total count
                    cursor.execute(
                        f"SELECT COUNT(*) FROM forecast_versions WHERE {where_sql}",
                        params
                    )
                    total_count = cursor.fetchone()[0]

                    # Get paginated results
                    offset = (page - 1) * page_size
                    cursor.execute(f"""
                        SELECT version_id, version_name, version_type, is_active,
                               created_at, created_by, updated_at, updated_by
                        FROM forecast_versions
                        WHERE {where_sql}
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """, params + [page_size, offset])

                    versions = []
                    for row in cursor.fetchall():
                        versions.append({
                            "version_id": str(row[0]),
                            "version_name": row[1],
                            "version_type": row[2],
                            "is_active": row[3],
                            "created_at": row[4].isoformat() if row[4] else None,
                            "created_by": row[5],
                            "updated_at": row[6].isoformat() if row[6] else None,
                            "updated_by": row[7]
                        })

                    return versions, total_count

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to list forecast versions: {str(e)}")
            raise DatabaseException(f"Failed to list versions: {str(e)}")

    @staticmethod
    def delete_version(
        tenant_id: str,
        database_name: str,
        version_id: str
    ) -> bool:
        """
        Delete a forecast version.
        Note: This will cascade delete all associated forecast runs and results.
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check if version exists and has associated runs
                    cursor.execute("""
                        SELECT COUNT(*) FROM forecast_runs
                        WHERE version_id = %s
                    """, (version_id,))
                    
                    run_count = cursor.fetchone()[0]
                    
                    if run_count > 0:
                        raise ValidationException(
                            f"Cannot delete version with {run_count} associated forecast runs"
                        )

                    # Delete version
                    cursor.execute("""
                        DELETE FROM forecast_versions
                        WHERE version_id = %s AND tenant_id = %s
                    """, (version_id, tenant_id))

                    if cursor.rowcount == 0:
                        raise NotFoundException("Forecast Version", version_id)

                    conn.commit()
                    logger.info(f"Forecast version deleted: {version_id}")
                    return True

                finally:
                    cursor.close()

        except (NotFoundException, ValidationException):
            raise
        except Exception as e:
            logger.error(f"Failed to delete forecast version: {str(e)}")
            raise DatabaseException(f"Failed to delete version: {str(e)}")