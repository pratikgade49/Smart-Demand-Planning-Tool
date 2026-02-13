"""
Dynamic Table Management Service.
Handles creation, deletion, and metadata management of custom planning tables.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from uuid import uuid4

from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, ValidationException

logger = logging.getLogger(__name__)


class DynamicTableService:
    """Service for managing dynamic tables per tenant."""

    @staticmethod
    def normalize_table_name(display_name: str) -> str:
        """
        Normalize display name to valid table name.
        "Product Manager" -> "product_manager"
        "Sales Team" -> "sales_team"
        "Final Consensus Plan" -> "final_consensus_plan"
        
        Args:
            display_name: User-provided table display name
            
        Returns:
            Normalized table name (lowercase, spaces/hyphens to underscores)
        """
        if not display_name or not isinstance(display_name, str):
            raise ValidationException("Display name must be a non-empty string")
        
        # Convert to lowercase
        normalized = display_name.lower()
        
        # Replace spaces and hyphens with underscores
        normalized = re.sub(r'[\s\-]+', '_', normalized)
        
        # Remove any other special characters
        normalized = re.sub(r'[^a-z0-9_]', '', normalized)
        
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        
        # Check length
        if len(normalized) == 0:
            raise ValidationException("Display name must contain at least one alphanumeric character")
        if len(normalized) > 63:  # PostgreSQL identifier limit
            raise ValidationException("Normalized table name exceeds maximum length of 63 characters")
        
        return normalized

    @staticmethod
    def _get_field_names_from_catalogue(cursor) -> tuple:
        """
        Get the date and target field names from field_catalogue_metadata table.
        
        Args:
            cursor: Database cursor
            
        Returns:
            Tuple of (date_field_name, target_field_name) or (None, None) if not found
        """
        try:
            cursor.execute("""
                SELECT date_field_name, target_field_name 
                FROM field_catalogue_metadata 
                LIMIT 1
            """)
            result = cursor.fetchone()
            if result:
                return (result[0], result[1])
        except Exception as e:
            logger.warning(f"Could not get field names from field_catalogue_metadata: {str(e)}")
        return (None, None)

    @staticmethod
    def create_dynamic_table(
        database_name: str,
        display_name: str,
        table_type: str = "custom",
        description: Optional[str] = None,
        date_field_name: Optional[str] = None,
        target_field_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new dynamic table for a tenant.
        
        Args:
            database_name: Tenant's database name
            display_name: User-friendly name (e.g., "Product Manager")
            table_type: Type of table (e.g., "planning", "forecast", "approval", "custom")
            description: Optional table description
            date_field_name: Name of the date field in the table (optional - fetched from field_catalogue if not provided)
            target_field_name: Name of the target/quantity field in the table (optional - fetched from field_catalogue if not provided)
            
        Returns:
            Dictionary with table_name, display_name, and creation details
            
        Raises:
            ValidationException: If table already exists or invalid parameters
            DatabaseException: If database operation fails
        """
        table_name = DynamicTableService.normalize_table_name(display_name)
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Get field names from field_catalogue_metadata if not provided
                    catalogue_date_field, catalogue_target_field = DynamicTableService._get_field_names_from_catalogue(cursor)
                    
                    # Use provided values, or fall back to catalogue values, or use defaults
                    final_date_field = date_field_name or catalogue_date_field or "period"
                    final_target_field = target_field_name or catalogue_target_field or "quantity"
                    
                    # Check if table already exists
                    if DynamicTableService._table_exists(cursor, table_name):
                        raise ValidationException(
                            f"Table '{table_name}' already exists. Choose a different name."
                        )
                    
                    # Create the table
                    id_column = f"{table_name}_id"
                    DynamicTableService._create_table_schema(
                        cursor,
                        table_name,
                        id_column,
                        final_date_field,
                        final_target_field
                    )
                    
                    # Record metadata in dynamic_tables
                    metadata_id = str(uuid4())
                    cursor.execute("""
                        INSERT INTO dynamic_tables 
                        (metadata_id, table_name, display_name, table_type, description, is_mandatory)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (
                        metadata_id,
                        table_name,
                        display_name,
                        table_type,
                        description or "",
                        False  # Dynamic tables are never mandatory
                    ))
                    
                    conn.commit()
                    logger.info(f"Created dynamic table '{table_name}' in database {database_name}")
                    
                    return {
                        "status": "success",
                        "table_name": table_name,
                        "display_name": display_name,
                        "table_type": table_type,
                        "description": description
                    }
                    
                finally:
                    cursor.close()
                    
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to create dynamic table '{table_name}': {str(e)}")
            raise DatabaseException(f"Failed to create dynamic table: {str(e)}")

    @staticmethod
    def _create_table_schema(
        cursor,
        table_name: str,
        id_column: str,
        date_field_name: str,
        target_field_name: str
    ) -> None:
        """
        Create the actual table schema for a dynamic table.
        Same structure as final_plan and product_manager.
        
        Args:
            cursor: Database cursor
            table_name: Normalized table name
            id_column: Primary key column name
            date_field_name: Date field name (e.g., "date")
            target_field_name: Target field name (e.g., "quantity")
        """
        # Default SQL types - these match the types used in sales_data and forecast_data
        date_sql_type = "DATE"
        target_sql_type = "DECIMAL(18, 2)"
        
        # Try to get field types from master_data if it exists
        try:
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'master_data'
                ORDER BY ordinal_position
            """)
            
            columns = cursor.fetchall()
            for col_name, col_type in columns:
                # Map common column types
                if col_name == date_field_name and col_type.upper() in ['DATE', 'TIMESTAMP', 'TIMESTAMP WITHOUT TIME ZONE']:
                    date_sql_type = col_type.upper()
                elif col_name == target_field_name and 'NUMERIC' in col_type.upper():
                    target_sql_type = col_type.upper()
        except Exception:
            # If we can't get from master_data, use defaults
            pass
        
        # Create the table
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {id_column} UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                master_id UUID NOT NULL REFERENCES master_data(master_id),
                "{date_field_name}" {date_sql_type} NOT NULL,
                "{target_field_name}" {target_sql_type} NOT NULL,
                uom VARCHAR(20) NOT NULL,
                unit_price DECIMAL(18, 2),

                -- Disaggregation tracking columns
                type VARCHAR(20) DEFAULT 'manual',
                disaggregation_level VARCHAR(255),
                source_aggregation_level VARCHAR(255),
                source_forecast_run_id UUID,

                -- Audit fields
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by VARCHAR(255) NOT NULL,
                updated_at TIMESTAMP,
                updated_by VARCHAR(255),
                
                -- Unique constraint
                UNIQUE(master_id, "{date_field_name}")
            )
        """
        
        cursor.execute(create_table_sql)
        
        # Create indexes
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_date 
            ON {table_name}("{date_field_name}")
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_master_id 
            ON {table_name}(master_id)
        """)

    @staticmethod
    def _table_exists(cursor, table_name: str) -> bool:
        """
        Check if a table exists in the current database.
        
        Args:
            cursor: Database cursor
            table_name: Table name to check
            
        Returns:
            True if table exists
        """
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            )
        """, (table_name,))
        return bool(cursor.fetchone()[0])

    @staticmethod
    def get_tenant_dynamic_tables(database_name: str) -> List[Dict[str, Any]]:
        """
        Retrieve all dynamic tables configured for a tenant.
        
        Args:
            database_name: Tenant's database name
            
        Returns:
            List of dynamic table metadata dictionaries
            
        Raises:
            DatabaseException: If query fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check if dynamic_tables table exists
                    if not DynamicTableService._table_exists(cursor, "dynamic_tables"):
                        return []
                    
                    cursor.execute("""
                        SELECT 
                            metadata_id,
                            table_name,
                            display_name,
                            table_type,
                            description,
                            is_mandatory,
                            created_at
                        FROM dynamic_tables
                        WHERE is_mandatory = false
                        ORDER BY created_at ASC
                    """)
                    
                    columns = [desc[0] for desc in cursor.description]
                    tables = []
                    for row in cursor.fetchall():
                        tables.append(dict(zip(columns, row)))
                    
                    return tables
                    
                finally:
                    cursor.close()
                    
        except Exception as e:
            logger.error(f"Failed to get dynamic tables for {database_name}: {str(e)}")
            raise DatabaseException(f"Failed to retrieve dynamic tables: {str(e)}")

    @staticmethod
    def delete_dynamic_table(
        database_name: str,
        table_name: str
    ) -> Dict[str, Any]:
        """
        Delete a dynamic table (soft delete via metadata).
        
        Args:
            database_name: Tenant's database name
            table_name: Normalized table name to delete
            
        Returns:
            Confirmation message
            
        Raises:
            ValidationException: If table is mandatory or doesn't exist
            DatabaseException: If operation fails
        """
        db_manager = get_db_manager()
        
        try:
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    # Check if table exists and is not mandatory
                    cursor.execute("""
                        SELECT is_mandatory FROM dynamic_tables
                        WHERE table_name = %s
                    """, (table_name,))
                    
                    result = cursor.fetchone()
                    if not result:
                        raise ValidationException(f"Table '{table_name}' not found")
                    
                    if result[0]:  # is_mandatory
                        raise ValidationException(f"Cannot delete mandatory table '{table_name}'")
                    
                    # Drop the table
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                    
                    # Remove metadata
                    cursor.execute("""
                        DELETE FROM dynamic_tables WHERE table_name = %s
                    """, (table_name,))
                    
                    conn.commit()
                    logger.info(f"Deleted dynamic table '{table_name}' from {database_name}")
                    
                    return {
                        "status": "success",
                        "message": f"Table '{table_name}' deleted successfully",
                        "table_name": table_name
                    }
                    
                finally:
                    cursor.close()
                    
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"Failed to delete dynamic table '{table_name}': {str(e)}")
            raise DatabaseException(f"Failed to delete dynamic table: {str(e)}")
