"""
Generic Dashboard Data Service Methods.
Provides table-agnostic methods for saving, updating, and copying data across tables.
"""

import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4

from psycopg2 import sql

from app.core.database import get_db_manager
from app.core.exceptions import AppException, DatabaseException, ValidationException, NotFoundException
from app.core.sales_data_service import SalesDataService
from app.core.dynamic_table_service import DynamicTableService

logger = logging.getLogger(__name__)


class GenericDashboardService:
    """Generic methods for dashboard data operations across any table."""

    @staticmethod
    def save_data_row(
        database_name: str,
        table_name: str,
        user_email: str,
        record_id: Optional[str],
        master_data: Optional[Dict[str, Any]],
        plan_date,
        quantity: float,
    ) -> Dict[str, Any]:
        """
        Generic method to save or update a row in any dynamic table.
        
        Args:
            database_name: Tenant database name
            table_name: Normalized table name (e.g., "product_manager", "sales_team")
            user_email: User performing the operation
            record_id: Primary key if updating, None for new insert
            master_data: Master data fields to resolve master_id (for inserts)
            plan_date: Date for the record
            quantity: Quantity value
            
        Returns:
            Dictionary with status and record_id
            
        Raises:
            ValidationException: If parameters invalid
            NotFoundException: If table or master data not found
            DatabaseException: If operation fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Validate table exists
                    if not GenericDashboardService._table_exists(cursor, table_name):
                        raise NotFoundException("Table", table_name)
                    
                    # Get field names
                    date_field, target_field = SalesDataService._get_field_names(cursor)
                    
                    # Get primary key column name
                    id_column = f"{table_name}_id"
                    
                    if record_id:
                        # Update existing record
                        cursor.execute(
                            f"""
                            UPDATE {table_name}
                            SET "{target_field}" = %s,
                                updated_at = CURRENT_TIMESTAMP,
                                updated_by = %s
                            WHERE {id_column} = %s
                            """,
                            (quantity, user_email, record_id)
                        )
                        conn.commit()
                        return {
                            "status": "success",
                            "message": f"Record updated in {table_name}",
                            "record_id": record_id,
                            "table_name": table_name
                        }
                    
                    # Insert new record
                    if master_data is None or plan_date is None:
                        raise ValidationException("master_data and date are required for new records")
                    
                    # Resolve master_id
                    master_id = GenericDashboardService._resolve_master_id(cursor, master_data)
                    
                    # Get UOM and unit price
                    uom = master_data.get("uom") if master_data else None
                    if not uom:
                        uom, unit_price = GenericDashboardService._resolve_sales_info(cursor, master_id)
                    else:
                        _, unit_price = GenericDashboardService._resolve_sales_info(cursor, master_id)
                    
                    cursor.execute(
                        f"""
                        INSERT INTO {table_name}
                        (master_id, "{date_field}", "{target_field}", uom, unit_price, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING {id_column}
                        """,
                        (master_id, plan_date, quantity, uom, unit_price, user_email)
                    )
                    
                    row = cursor.fetchone()
                    conn.commit()
                    
                    return {
                        "status": "success",
                        "message": f"Record created in {table_name}",
                        "record_id": str(row[0]),
                        "table_name": table_name
                    }
                    
                finally:
                    cursor.close()
                    
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Error saving data to {table_name}: {str(e)}")
            raise DatabaseException(f"Failed to save data to {table_name}: {str(e)}")

    @staticmethod
    def save_aggregated_data_rows(
        database_name: str,
        table_name: str,
        user_email: str,
        aggregated_fields: List[str],
        group_data: Dict[str, Any],
        plan_date,
        quantity: float,
    ) -> Dict[str, Any]:
        """
        Generic method to save aggregated data by distributing quantity among group members.
        
        Args:
            database_name: Tenant database name
            table_name: Target table name
            user_email: User performing operation
            aggregated_fields: Fields used for aggregation
            group_data: Group values
            plan_date: Date for the records
            quantity: Total quantity to distribute
            
        Returns:
            Dictionary with status and count of inserted records
            
        Raises:
            ValidationException: If parameters invalid
            NotFoundException: If group not found
            DatabaseException: If operation fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Validate table exists
                    if not GenericDashboardService._table_exists(cursor, table_name):
                        raise NotFoundException("Table", table_name)
                    
                    date_field, target_field = SalesDataService._get_field_names(cursor)
                    
                    # Build WHERE clause for aggregated group
                    group_conditions = []
                    group_params = []
                    for field in aggregated_fields:
                        val = group_data.get(field)
                        if val is None:
                            group_conditions.append(f'm."{field}" IS NULL')
                        else:
                            group_conditions.append(f'm."{field}" = %s')
                            group_params.append(val)
                    
                    group_where = " AND ".join(group_conditions)
                    
                    # Get all master_ids in this group
                    master_id_query = f"""
                        SELECT master_id
                        FROM master_data m
                        WHERE {group_where}
                    """
                    cursor.execute(master_id_query, group_params)
                    master_ids = [row[0] for row in cursor.fetchall()]
                    
                    if not master_ids:
                        raise NotFoundException("Aggregated group", str(group_data))
                    
                    # Calculate ratios from existing table data
                    placeholders = ",".join(["%s"] * len(master_ids))
                    totals_query = f"""
                        SELECT master_id, SUM(CAST("{target_field}" AS DOUBLE PRECISION))
                        FROM {table_name}
                        WHERE master_id IN ({placeholders})
                        GROUP BY master_id
                    """
                    cursor.execute(totals_query, master_ids)
                    
                    totals = {row[0]: float(row[1]) for row in cursor.fetchall()}
                    total_sum = sum(totals.values())
                    
                    # Calculate ratios
                    ratios = {}
                    if total_sum > 0:
                        for master_id in master_ids:
                            ratios[master_id] = totals.get(master_id, 0.0) / total_sum
                    else:
                        equal_ratio = 1.0 / len(master_ids)
                        for master_id in master_ids:
                            ratios[master_id] = equal_ratio
                    
                    # Get UOM and unit price from sales data
                    cursor.execute("""
                        SELECT uom, unit_price FROM sales_data
                        WHERE master_id IN (SELECT master_id FROM master_data LIMIT 1)
                        ORDER BY created_at DESC LIMIT 1
                    """)
                    result = cursor.fetchone()
                    uom = result[0] if result else "UNIT"
                    unit_price = result[1] if result else 0
                    
                    # Insert distributed quantities
                    id_column = f"{table_name}_id"
                    inserted_count = 0
                    
                    for master_id, ratio in ratios.items():
                        distributed_qty = quantity * ratio
                        cursor.execute(
                            f"""
                            INSERT INTO {table_name}
                            ({id_column}, master_id, "{date_field}", "{target_field}", uom, unit_price, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (master_id, "{date_field}") DO UPDATE
                            SET "{target_field}" = %s,
                                updated_at = CURRENT_TIMESTAMP,
                                updated_by = %s
                            """,
                            (
                                str(uuid4()), master_id, plan_date, distributed_qty, uom, unit_price, user_email,
                                distributed_qty, user_email
                            )
                        )
                        inserted_count += 1
                    
                    conn.commit()
                    
                    return {
                        "status": "success",
                        "message": f"Distributed {quantity} across {inserted_count} records in {table_name}",
                        "table_name": table_name,
                        "records_affected": inserted_count
                    }
                    
                finally:
                    cursor.close()
                    
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Error saving aggregated data to {table_name}: {str(e)}")
            raise DatabaseException(f"Failed to save aggregated data: {str(e)}")

    @staticmethod
    def copy_data_between_tables(
        database_name: str,
        source_table: str,
        target_table: str,
        user_email: str,
        filters: List[Any],
        from_date,
        to_date,
    ) -> Dict[str, Any]:
        """
        Generic method to copy data between any two tables.
        
        Args:
            database_name: Tenant database name
            source_table: Source table name
            target_table: Target table name
            user_email: User performing operation
            filters: Master data filters
            from_date: Start date for copy
            to_date: End date for copy
            
        Returns:
            Dictionary with copy status and count
            
        Raises:
            ValidationException: If tables don't exist
            DatabaseException: If operation fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Validate tables exist
                    if not GenericDashboardService._table_exists(cursor, source_table):
                        raise NotFoundException("Table", source_table)
                    if not GenericDashboardService._table_exists(cursor, target_table):
                        raise NotFoundException("Table", target_table)
                    
                    if source_table == target_table:
                        raise ValidationException("Source and target tables cannot be the same")
                    
                    date_field, target_field = SalesDataService._get_field_names(cursor)
                    
                    from_date = from_date or "2025-01-01"
                    to_date = to_date or "2026-12-31"
                    
                    # Build WHERE clause
                    where_conditions = [f's."{date_field}" BETWEEN %s AND %s']
                    where_params = [from_date, to_date]
                    
                    # Process filters if any
                    if filters:
                        for f in filters:
                            if f.values:  # Check if values list is not empty
                                placeholders = ",".join(["%s"] * len(f.values))
                                where_conditions.append(f'm."{f.field_name}" IN ({placeholders})')
                                where_params.extend(f.values)
                    
                    where_clause = " AND ".join(where_conditions)
                    
                    # Get ID columns
                    source_id = f"{source_table}_id"
                    target_id = f"{target_table}_id"
                    
                    # Copy data
                    copy_query = f"""
                        INSERT INTO {target_table} 
                        ({target_id}, master_id, "{date_field}", "{target_field}", uom, unit_price, created_by)
                        SELECT 
                            gen_random_uuid(),
                            s.master_id,
                            s."{date_field}",
                            s."{target_field}",
                            s.uom,
                            s.unit_price,
                            %s
                        FROM {source_table} s
                        JOIN master_data m ON s.master_id = m.master_id
                        WHERE {where_clause}
                        ON CONFLICT (master_id, "{date_field}") DO UPDATE
                        SET "{target_field}" = EXCLUDED."{target_field}",
                            updated_at = CURRENT_TIMESTAMP,
                            updated_by = %s
                    """
                    
                    cursor.execute(copy_query, where_params + [user_email, user_email])
                    copied_count = cursor.rowcount
                    conn.commit()
                    
                    logger.info(f"Copied {copied_count} records from {source_table} to {target_table}")
                    
                    return {
                        "status": "success",
                        "message": f"Copied {copied_count} records from {source_table} to {target_table}",
                        "source_table": source_table,
                        "target_table": target_table,
                        "records_copied": copied_count
                    }
                    
                finally:
                    cursor.close()
                    
        except AppException:
            raise
        except Exception as e:
            logger.error(f"Error copying from {source_table} to {target_table}: {str(e)}")
            raise DatabaseException(f"Failed to copy data between tables: {str(e)}")

    @staticmethod
    def _table_exists(cursor, table_name: str) -> bool:
        """Check if table exists."""
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
        """, (table_name,))
        return bool(cursor.fetchone()[0])

    @staticmethod
    def _resolve_master_id(cursor, master_data: Dict[str, Any]) -> str:
        """Resolve master_id from master_data fields."""
        if not master_data:
            raise ValidationException("master_data is required")
        
        clauses = []
        params = []
        for key, value in master_data.items():
            clauses.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
            params.append(value)
        
        query = sql.SQL("SELECT master_id FROM master_data WHERE {} LIMIT 1").format(
            sql.SQL(" AND ").join(clauses)
        )
        cursor.execute(query, params)
        row = cursor.fetchone()
        if not row:
            raise NotFoundException("Master data", str(master_data))
        return row[0]

    @staticmethod
    def _resolve_sales_info(cursor, master_id: str) -> tuple:
        """Get UOM and unit price from latest sales record."""
        cursor.execute("""
            SELECT uom, unit_price
            FROM sales_data
            WHERE master_id = %s
            ORDER BY created_at DESC LIMIT 1
        """, (master_id,))
        
        sales_row = cursor.fetchone()
        uom = sales_row[0] if sales_row else "UNIT"
        unit_price = sales_row[1] if sales_row else 0
        return uom, unit_price
