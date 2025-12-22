"""
Sales Data API Schemas.
Endpoints for retrieving sales data records with flexible filtering.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import date


class SalesDataFilter(BaseModel):
    """Filter for sales data retrieval."""
    field_name: str = Field(..., description="Master data field name to filter by")
    values: List[Any] = Field(..., description="List of values to filter by")


class SalesDataQueryRequest(BaseModel):
    """Request schema for sales data retrieval."""
    filters: Optional[List[SalesDataFilter]] = Field(
        default=None,
        description="List of filters to apply. Each filter specifies a field and its values."
    )
    from_date: Optional[date] = Field(
        default=None,
        description="Start date for filtering sales data (inclusive)"
    )
    to_date: Optional[date] = Field(
        default=None,
        description="End date for filtering sales data (inclusive)"
    )
    page: int = Field(
        default=1,
        ge=1,
        description="Page number for pagination"
    )
    page_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of records per page (max 1000)"
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="Field to sort by (e.g., 'date', 'quantity')"
    )
    sort_order: str = Field(
        default="asc",
        pattern="^(asc|desc)$",
        description="Sort order: 'asc' or 'desc'"
    )


class SalesDataRecord(BaseModel):
    """Individual sales data record."""
    sales_id: str
    master_id: str
    date: date
    quantity: float
    unit_price: Optional[float] = None
    total_amount: Optional[float] = None
    created_at: str
    updated_at: str
    # Dynamic master data fields will be added here
    master_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Master data fields associated with this sales record"
    )


class SalesDataResponse(BaseModel):
    """Response schema for sales data retrieval."""
    records: List[SalesDataRecord]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


class SalesDataSummary(BaseModel):
    """Summary statistics for sales data."""
    total_records: int
    total_quantity: float
    total_amount: Optional[float] = None
    avg_quantity: float
    avg_price: Optional[float] = None
    date_range: Dict[str, date]
    field_summaries: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Summary statistics for each master data field"
    )
