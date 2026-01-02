"""
Sales Data API Routes.
Endpoints for retrieving sales data records with flexible filtering.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional
from datetime import date
from app.schemas.sales_data import (
    SalesDataQueryRequest,
    SalesDataResponse,
    SalesDataSummary,
    SalesDataFilter
)
from app.core.sales_data_service import SalesDataService
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException
from app.api.dependencies import get_tenant_database
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sales-data", tags=["Sales Data"])


@router.post("/records", response_model=Dict[str, Any])
async def get_sales_data_records(
    request: SalesDataQueryRequest,
    tenant_data: Dict = Depends(get_tenant_database)
):
    """
    Retrieve sales data records with flexible filtering and pagination.

    Supports filtering by any master data fields, date ranges, and pagination.
    Returns sales records joined with their associated master data.
    """
    try:
        result = SalesDataService.get_sales_records_ui(
            database_name=tenant_data["database_name"],
            from_date=request.from_date,
            to_date=request.to_date,
            page=request.page,
            page_size=request.page_size
        )

        return ResponseHandler.success(data=result)

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_sales_data_records: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/summary", response_model=Dict[str, Any])
async def get_sales_data_summary(
    filters: Optional[List[SalesDataFilter]] = None,
    from_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    tenant_data: Dict = Depends(get_tenant_database)
):
    """
    Get summary statistics for sales data with optional filtering.

    Returns aggregated statistics including total records, quantities, amounts,
    and breakdowns by master data fields.
    """
    try:
        # Convert filters to dict format for service
        filter_dicts = None
        if filters:
            filter_dicts = [
                {"field_name": f.field_name, "values": f.values}
                for f in filters
            ]

        result = SalesDataService.get_sales_data_summary(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            filters=filter_dicts,
            from_date=from_date,
            to_date=to_date
        )

        return ResponseHandler.success(data=result.dict())

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_sales_data_summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/records/simple", response_model=Dict[str, Any])
async def get_sales_data_records_simple(
    # Field filters as query parameters
    product: Optional[List[str]] = Query(None, description="Product filter"),
    customer: Optional[List[str]] = Query(None, description="Customer filter"),
    location: Optional[List[str]] = Query(None, description="Location filter"),
    location_region: Optional[List[str]] = Query(None, description="Location region filter"),
    product_group: Optional[List[str]] = Query(None, description="Product group filter"),
    # Date filters
    from_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    # Pagination
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Records per page"),
    # Sorting
    sort_by: Optional[str] = Query(None, description="Sort field"),
    sort_order: str = Query("asc", regex="^(asc|desc)$", description="Sort order"),
    tenant_data: Dict = Depends(get_tenant_database)
):
    """
    Simplified GET endpoint for sales data records.

    Alternative to POST endpoint for simple filtering scenarios.
    Supports common master data fields as query parameters.
    """
    try:
        # Build filters from query parameters
        filters = []
        if product:
            filters.append(SalesDataFilter(field_name="product", values=product))
        if customer:
            filters.append(SalesDataFilter(field_name="customer", values=customer))
        if location:
            filters.append(SalesDataFilter(field_name="location", values=location))
        if location_region:
            filters.append(SalesDataFilter(field_name="location_region", values=location_region))
        if product_group:
            filters.append(SalesDataFilter(field_name="product_group", values=product_group))

        # Create request object
        request = SalesDataQueryRequest(
            filters=filters if filters else None,
            from_date=from_date,
            to_date=to_date,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )

        result = SalesDataService.get_sales_data(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            request=request
        )

        return ResponseHandler.success(data=result.dict())

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_sales_data_records_simple: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

