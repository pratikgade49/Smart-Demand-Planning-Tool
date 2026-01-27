"""
Master Data schemas for field and value retrieval endpoints.
Used for UI dropdown population with master data fields and their distinct values.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class MasterDataField(BaseModel):
    """Schema for master data field information."""
    field_name: str = Field(..., description="The actual field name in the database")
    data_type: str = Field(..., description="PostgreSQL data type of the field")
    is_nullable: bool = Field(..., description="Whether the field allows NULL values")
    display_name: str = Field(..., description="Human-readable display name")


class FieldValue(BaseModel):
    """Schema for individual field values with metadata."""
    value: Any = Field(..., description="The actual value (can be string, number, etc.)")
    count: int = Field(..., description="Number of occurrences of this value")
    display_value: str = Field(..., description="Display-friendly version of the value")


class FieldValuesRequest(BaseModel):
    """Request schema for getting values of a specific field."""
    field_name: str = Field(..., description="The field to get values for")
    filters: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Optional filters for other fields (e.g., {'product': ['A', 'B']})"
    )


class MultipleFieldValuesRequest(BaseModel):
    """Request schema for getting values of multiple fields with cross-filtering."""
    field_selections: Dict[str, List[str]] = Field(
        ...,
        description="Dictionary mapping field names to lists of selected values for filtering",
        example={
            "product": ["Product A", "Product B"],
            "customer": ["Customer X"]
        }
    )


class MasterDataFieldsResponse(BaseModel):
    """Response schema for available master data fields."""
    fields: List[MasterDataField] = Field(..., description="List of available fields")


class FieldValuesResponse(BaseModel):
    """Response schema for field values."""
    field_values: Dict[str, List[Any]] = Field(..., description="Map of field name to its values")


class MultipleFieldValuesResponse(BaseModel):
    """Response schema for multiple field values with cross-filtering."""
    field_values: Dict[str, List[Any]] = Field(
        ...,
        description="Dictionary mapping field names to their value lists"
    )
