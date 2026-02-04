"""
Master Data Service.
Handles retrieval of master data fields and their distinct values for UI dropdowns.
"""

import logging
from typing import Dict, Any, List, Optional
from app.core.database import get_db_manager
from app.core.exceptions import ValidationException, NotFoundException, ConflictException

logger = logging.getLogger(__name__)


class MasterDataService:
    """Service for master data field and value retrieval operations."""

    @staticmethod
    def get_master_data_fields(
        tenant_id: str,
        database_name: str
    ) -> List[Dict[str, Any]]:
        """
        Get all available fields from master_data table for dropdown selection.

        Returns:
            List of field dictionaries with name, data_type, and display info
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get available columns from master_data table
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns
                        WHERE table_name = 'master_data'
                        AND table_schema = 'public'
                        AND column_name NOT IN ('master_id','uom', 'tenant_id', 'created_at',
                                               'created_by', 'updated_at', 'updated_by','deleted_at')
                        ORDER BY column_name
                    """)

                    fields = []
                    for row in cursor.fetchall():
                        column_name, data_type, is_nullable = row
                        fields.append({
                            "field_name": column_name,
                            "data_type": data_type,
                            "is_nullable": is_nullable == 'YES',
                            "display_name": column_name.replace('_', ' ').title()
                        })

                    if not fields:
                        raise NotFoundException("Master Data Fields", "No fields found - ensure field catalogue is finalized")

                    logger.info(f"Retrieved {len(fields)} master data fields for tenant {tenant_id}")
                    return fields

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to get master data fields: {str(e)}")
            raise ValidationException(f"Failed to retrieve master data fields: {str(e)}")

    @staticmethod
    def get_field_values(
        tenant_id: str,
        database_name: str,
        field_name: str,
        filters: Optional[Dict[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get distinct values for a specific field, optionally filtered by other fields.

        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            field_name: The field to get values for
            filters: Optional filters for other fields (e.g., {'product': ['A', 'B']})

        Returns:
            List of distinct values with counts
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Validate field exists
                    MasterDataService._validate_field_exists(cursor, field_name)

                    # Build WHERE clause for filters
                    where_clause = ""
                    params = []
                    if filters:
                        filter_conditions = []
                        for filter_field, filter_values in filters.items():
                            if filter_field != field_name:  # Don't filter on the field we're getting values for
                                # Validate filter field exists
                                MasterDataService._validate_field_exists(cursor, filter_field)

                                if isinstance(filter_values, list) and filter_values:
                                    placeholders = ', '.join(['%s'] * len(filter_values))
                                    filter_conditions.append(f'"{filter_field}" IN ({placeholders})')
                                    params.extend(filter_values)
                                elif filter_values:  # Single value
                                    filter_conditions.append(f'"{filter_field}" = %s')
                                    params.append(filter_values)

                        if filter_conditions:
                            where_clause = "WHERE " + " AND ".join(filter_conditions)

                    # Get distinct values with counts
                    query = f"""
                        SELECT "{field_name}", COUNT(*) as value_count
                        FROM master_data
                        {where_clause}
                        GROUP BY "{field_name}"
                        ORDER BY "{field_name}"
                    """

                    cursor.execute(query, params)

                    values = []
                    for row in cursor.fetchall():
                        value, count = row
                        values.append({
                            "value": value,
                            "count": count,
                            "display_value": str(value) if value is not None else "NULL"
                        })

                    logger.info(f"Retrieved {len(values)} distinct values for field '{field_name}' with filters: {filters}")
                    return values

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to get field values for '{field_name}': {str(e)}")
            raise ValidationException(f"Failed to retrieve values for field '{field_name}': {str(e)}")

    @staticmethod
    def get_multiple_field_values(
        tenant_id: str,
        database_name: str,
        field_selections: Dict[str, List[str]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get distinct values for multiple fields based on selections.

        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            field_selections: Dict mapping field names to lists of selected values
                             e.g., {'product': ['A', 'B'], 'customer': ['X']}

        Returns:
            Dict mapping field names to their value lists with counts
        """
        results = {}

        # Get values for each field, using other fields as filters
        for target_field in field_selections.keys():
            filters = {k: v for k, v in field_selections.items() if k != target_field}
            values = MasterDataService.get_field_values(
                tenant_id, database_name, target_field, filters
            )
            results[target_field] = values

        logger.info(f"Retrieved values for {len(results)} fields with cross-filtering")
        return results

    @staticmethod
    def _validate_field_exists(cursor, field_name: str) -> None:
        """Validate that a field exists in the master_data table."""
        cursor.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'master_data'
            AND table_schema = 'public'
            AND column_name = %s
        """, (field_name,))

        if not cursor.fetchone():
            raise ValidationException(f"Field '{field_name}' does not exist in master_data table")

    @staticmethod
    def validate_master_data_exists(database_name: str) -> bool:
        """
        Check if master_data table exists and has been populated.

        Returns:
            True if master_data table exists and has data
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check if table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables
                            WHERE table_name = 'master_data'
                            AND table_schema = 'public'
                        )
                    """)

                    table_exists = cursor.fetchone()[0]
                    if not table_exists:
                        return False

                    # Check if table has data
                    cursor.execute("SELECT COUNT(*) FROM master_data")
                    count = cursor.fetchone()[0]

                    return count > 0

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to validate master_data existence: {str(e)}")
            return False

    @staticmethod
    def create_master_data_record(
        tenant_id: str,
        database_name: str,
        record: Dict[str, Any],
        user_email: str
    ) -> Dict[str, Any]:
        """
        Create a master data record.

        Raises ConflictException if an identical record already exists.
        """
        if not isinstance(record, dict) or not record:
            raise ValidationException("Record payload is required")

        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Ensure master_data table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables
                            WHERE table_name = 'master_data'
                            AND table_schema = 'public'
                        )
                    """)
                    if not cursor.fetchone()[0]:
                        raise NotFoundException("Master Data", "master_data table not found")

                    # Fetch valid master data columns (exclude system columns)
                    cursor.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'master_data'
                        AND table_schema = 'public'
                        AND column_name NOT IN ('master_id','uom', 'tenant_id', 'created_at',
                                               'created_by', 'updated_at', 'updated_by','deleted_at')
                        ORDER BY ordinal_position
                    """)
                    columns = [row[0] for row in cursor.fetchall()]
                    column_set = set(columns)

                    invalid_fields = [key for key in record.keys() if key not in column_set]
                    if invalid_fields:
                        raise ValidationException(
                            f"Invalid master data fields: {', '.join(invalid_fields)}"
                        )

                    # Normalize record values (treat empty strings as NULL)
                    normalized_record = {
                        key: (value if value not in ("", None) else None)
                        for key, value in record.items()
                    }

                    # Check for existing identical record
                    where_conditions = []
                    params: List[Any] = []
                    for column in columns:
                        value = normalized_record.get(column)
                        if value is None:
                            where_conditions.append(f'"{column}" IS NULL')
                        else:
                            where_conditions.append(f'"{column}" = %s')
                            params.append(value)

                    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                    cursor.execute(
                        f"SELECT master_id FROM master_data WHERE {where_clause} LIMIT 1",
                        params
                    )
                    if cursor.fetchone():
                        raise ConflictException("Record already exists")

                    # Insert new record
                    insert_columns = ['created_by']
                    insert_values = [user_email]
                    for key, value in normalized_record.items():
                        insert_columns.append(f'"{key}"')
                        insert_values.append(value)

                    columns_str = ', '.join(insert_columns)
                    placeholders = ', '.join(['%s'] * len(insert_values))
                    insert_query = (
                        f"INSERT INTO master_data ({columns_str}) "
                        f"VALUES ({placeholders}) RETURNING master_id"
                    )
                    cursor.execute(insert_query, insert_values)
                    master_id = cursor.fetchone()[0]
                    conn.commit()

                    logger.info(f"Created master data record {master_id} for tenant {tenant_id}")
                    return {"master_id": str(master_id)}

                finally:
                    cursor.close()

        except (ValidationException, NotFoundException, ConflictException):
            raise
        except Exception as e:
            logger.error(f"Failed to create master data record: {str(e)}")
            raise ValidationException(f"Failed to create master data record: {str(e)}")

    @staticmethod
    def get_master_data_records(
        tenant_id: str,
        database_name: str,
        filters: Dict[str, List[str]],
        page: int = 1,
        page_size: int = 100,
        sort_by: Optional[str] = None,
        sort_order: str = "asc"
    ) -> tuple:
        """
        Retrieve actual master data records matching the provided filters.

        Args:
            tenant_id: The tenant identifier
            database_name: The database to query
            filters: Dict with field_name: [list of values]
                    Example: {"customer": ["0100000034"], "location": ["LOC001"]}
            page: Page number (1-indexed)
            page_size: Number of records per page
            sort_by: Field to sort by (optional)
            sort_order: Sort order ('asc' or 'desc')

        Returns:
            Tuple of (records list, total count)
        """
        db_manager = get_db_manager()

        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Build WHERE clause from filters
                    where_conditions = []
                    params = []

                    if filters:
                        for field_name, values in filters.items():
                            if values:  # Only add if values exist
                                # Validate filter field exists
                                MasterDataService._validate_field_exists(cursor, field_name)

                                placeholders = ', '.join(['%s'] * len(values))
                                where_conditions.append(f'"{field_name}" IN ({placeholders})')
                                params.extend(values)

                    where_clause = "WHERE " + " AND ".join(where_conditions) if where_conditions else ""

                    # Get only relevant columns (exclude system columns)
                    cursor.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'master_data'
                        AND table_schema = 'public'
                        AND column_name NOT IN ('master_id', 'uom', 'tenant_id', 'created_at',
                                               'created_by', 'updated_at', 'updated_by', 'deleted_at')
                        ORDER BY ordinal_position
                    """)
                    columns = [row[0] for row in cursor.fetchall()]
                    columns_str = ', '.join([f'"{col}"' for col in columns])

                    # Get total count
                    count_query = f"SELECT COUNT(*) FROM master_data {where_clause}"
                    cursor.execute(count_query, params if params else None)
                    total = cursor.fetchone()[0]

                    # Build ORDER BY clause
                    if sort_by and sort_by in columns:
                        order_direction = "DESC" if sort_order.lower() == "desc" else "ASC"
                        order_clause = f'ORDER BY "{sort_by}" {order_direction}'
                    else:
                        order_clause = "ORDER BY master_id ASC"

                    # Calculate offset from page number
                    offset = (page - 1) * page_size

                    # Get paginated records with only relevant columns
                    records_query = f"""
                        SELECT {columns_str} FROM master_data
                        {where_clause}
                        {order_clause}
                        LIMIT %s OFFSET %s
                    """
                    params_with_pagination = params + [page_size, offset]
                    cursor.execute(records_query, params_with_pagination)

                    # Get column names
                    column_names = [desc[0] for desc in cursor.description]

                    # Convert rows to list of dictionaries
                    records = []
                    for row in cursor.fetchall():
                        record = dict(zip(column_names, row))
                        records.append(record)

                    logger.info(f"Retrieved {len(records)} master data records (total: {total}) with filters: {filters}")
                    return records, total

                finally:
                    cursor.close()

        except Exception as e:
            logger.error(f"Failed to get master data records: {str(e)}")
            raise ValidationException(f"Failed to retrieve master data records: {str(e)}")
