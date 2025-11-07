"""
Pydantic schemas for Field Catalogue endpoints.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from enum import Enum

class FieldDataType(str, Enum):
    """Supported field data types."""
    CHAR = "Char"
    NUMERIC = "Numeric"
    DATE = "Date"
    TIMESTAMP = "Timestamp"
    BOOLEAN = "Boolean"
    TEXT = "Text"


# MODIFY FieldCatalogueItemRequest:
class FieldCatalogueItemRequest(BaseModel):
    """Request schema for a single field in Field Catalogue."""
    
    field_name: str = Field(..., min_length=1, max_length=100, description="Field name")
    data_type: FieldDataType = Field(..., description="Data type of the field")
    field_length: Optional[int] = Field(None, ge=1, le=1000, description="Field length for Char type")
    default_value: Optional[str] = Field(None, max_length=255, description="Default value")
    is_characteristic: bool = Field(..., description="Whether this is a characteristic field")
    parent_field_name: Optional[str] = Field(None, description="Parent field name if this is a characteristic")
    
    @field_validator("field_name")
    @classmethod
    def validate_field_name(cls, v: str) -> str:
        """Validate field name format."""
        if not v.replace("_", "").isalnum():
            raise ValueError("Field name can only contain alphanumeric characters and underscores")
        return v.lower()
    
    @field_validator("field_length")
    @classmethod
    def validate_field_length(cls, v: Optional[int], info) -> Optional[int]:
        """Validate field length based on data type."""
        if info.data.get("data_type") == FieldDataType.CHAR and v is None:
            raise ValueError("Field length is required for Char data type")
        return v

class FieldCatalogueItemResponse(FieldCatalogueItemRequest):
    """Response schema for a single field in Field Catalogue."""
    
    field_id: str
    version: int
    created_at: str
    created_by: str
    updated_at: Optional[str] = None
    updated_by: Optional[str] = None

class FieldCatalogueRequest(BaseModel):
    """Request schema for Field Catalogue submission."""
    
    fields: List[FieldCatalogueItemRequest] = Field(..., min_items=1, description="List of fields")
    
    @field_validator("fields")
    @classmethod
    def validate_unique_field_names(cls, v: List[FieldCatalogueItemRequest]) -> List[FieldCatalogueItemRequest]:
        """Ensure field names are unique."""
        field_names = [f.field_name for f in v]
        if len(field_names) != len(set(field_names)):
            raise ValueError("Duplicate field names found")
        return v

class FieldCatalogueResponse(BaseModel):
    """Response schema for Field Catalogue."""
    
    catalogue_id: str
    tenant_id: str
    fields: List[FieldCatalogueItemResponse]
    version: int
    status: str  # DRAFT, FINALIZED, ACTIVE
    created_at: str
    created_by: str

class FieldCatalogueValidationResponse(BaseModel):
    """Response schema for Field Catalogue validation."""
    
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    summary: str
    parent_field_errors: List[str] = []  # New field for parent validation errors
    circular_reference_errors: List[str] = []  # New field for circular reference errors
