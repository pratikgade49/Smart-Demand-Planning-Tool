"""
Database models and table creation utilities.
"""

from typing import Dict, List, Any, Optional
import json

class FieldDefinition:
    """Represents a field in the Field Catalogue."""
    
    def __init__(
            self,
            field_name: str,
            data_type: str,
            field_length: Optional[int] = None,
            default_value: Optional[str] = None,
            is_characteristic: bool = False,
            is_unique_key: bool = False,
            parent_field_name: Optional[str] = None,
            is_target_variable: bool = False,
            is_date_field: bool = False,
            description: Optional[str] = None
        ):
            self.field_name = field_name
            self.data_type = data_type
            self.field_length = field_length
            self.default_value = default_value
            self.is_characteristic = is_characteristic
            self.is_unique_key = is_unique_key
            self.parent_field_name = parent_field_name
            self.is_target_variable = is_target_variable
            self.is_date_field = is_date_field
            self.description = description
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field_name": self.field_name,
            "description": self.description,
            "data_type": self.data_type,
            "field_length": self.field_length,
            "default_value": self.default_value,
            "is_characteristic": self.is_characteristic,
            "is_unique_key": self.is_unique_key,
            "parent_field_name": self.parent_field_name,
            "is_target_variable": self.is_target_variable,
            "is_date_field": self.is_date_field
        }

    def get_sql_type(self) -> str:
        """Convert data type to SQL type."""
        data_type_upper = self.data_type.upper()
        if data_type_upper == "CHAR":
            if self.field_length:
                return f"VARCHAR({self.field_length})"
            else:
                return "VARCHAR(255)"
        elif data_type_upper == "NUMERIC":
            return "DECIMAL(18,2)"
        elif data_type_upper == "DATE":
            return "DATE"
        elif data_type_upper == "TIMESTAMP":
            return "TIMESTAMP"
        elif data_type_upper == "BOOLEAN":
            return "BOOLEAN"
        elif data_type_upper == "TEXT":
            return "TEXT"
        else:
            # Default fallback
            return "VARCHAR(255)"


class TableSchemaBuilder:
    """Builds SQL DDL for dynamic tables."""
    
    @staticmethod
    def build_master_data_table(
        tenant_id: str,
        fields: List[FieldDefinition]
    ) -> str:
        """
        Build CREATE TABLE statement for Master Data.
        
        Args:
            tenant_id: Tenant identifier
            fields: List of field definitions from Field Catalogue
            
        Returns:
            CREATE TABLE SQL statement
        """
