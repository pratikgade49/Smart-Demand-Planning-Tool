"""
Master Data Service.
Handles retrieval of master data fields and their distinct values for UI dropdowns.
"""

import logging
from typing import Dict, Any, List, Optional
from app.core.database import get_db_manager
from app.core.exceptions import ValidationException, NotFoundException

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
