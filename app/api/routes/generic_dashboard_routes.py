"""
Generic Dashboard API Routes.
New endpoints that work with dynamic tables.
These endpoints work alongside the existing table-specific endpoints.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import date as DateType

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_tenant_database, require_object_access
from app.core.generic_dashboard_service import GenericDashboardService
from app.core.dynamic_table_service import DynamicTableService
from app.core.exceptions import AppException, ValidationException, NotFoundException
from app.core.responses import ResponseHandler
from app.schemas.sales_data import SalesDataFilter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard-Generic"])


# ============================================================================
# Request Models
# ============================================================================

class SaveDataRequest(BaseModel):
    """Generic request to save data to any dynamic table."""
    table_name: str = Field(
        ...,
        description="Normalized table name (e.g., 'product_manager', 'marketing_team')"
    )
    record_id: Optional[str] = Field(
        default=None,
        description="Primary key for updates, None for inserts"
    )
    master_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Master data fields to resolve master_id"
    )
    date: Optional[DateType] = Field(
        default=None,
        description="Record date (YYYY-MM-DD)"
    )
    quantity: float = Field(..., description="Quantity value")


class SaveAggregatedDataRequest(BaseModel):
    """Generic request to save aggregated data across group members."""
    table_name: str = Field(
        ...,
        description="Target table name"
    )
    aggregated_fields: List[str] = Field(
        ...,
        description="Fields used for aggregation (e.g., ['product', 'location'])"
    )
    group_data: Dict[str, Any] = Field(
        ...,
        description="Group values defining the aggregated group"
    )
    date: DateType = Field(..., description="Date for the records (YYYY-MM-DD)")
    quantity: float = Field(..., description="Total quantity to distribute")


class CopyDataRequest(BaseModel):
    """Generic request to copy data between tables."""
    source_table: str = Field(..., description="Source table name")
    target_table: str = Field(..., description="Target table name")
    filters: Optional[List[SalesDataFilter]] = Field(
        default=None,
        description="Master data filters (optional)"
    )
    from_date: Optional[DateType] = Field(
        default=None,
        description="Start date (inclusive)"
    )
    to_date: Optional[DateType] = Field(
        default=None,
        description="End date (inclusive)"
    )


class CreateTableRequest(BaseModel):
    """Request to create a new dynamic table."""
    display_name: str = Field(
        ...,
        description="User-friendly table name (e.g., 'Product Manager')"
    )
    table_type: str = Field(
        default="custom",
        description="Table type: 'planning', 'forecast', 'approval', 'custom'"
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional table description"
    )


class TableListResponse(BaseModel):
    """Response with list of available tables."""
    tables: List[Dict[str, Any]] = Field(..., description="List of table metadata")
    total_count: int = Field(..., description="Total number of tables")


# ============================================================================
# Generic Endpoints
# ============================================================================

@router.post("/save-data", response_model=Dict[str, Any])
async def save_data(
    request: SaveDataRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Save or update a data row in any dynamic table.
    
    **Parameters:**
    - `table_name`: Normalized table name (lowercase, spaces replaced with underscores)
    - `record_id`: Provide to update existing record, omit to insert new
    - `master_data`: Required for new inserts (omit for updates)
    - `date`: Required for new inserts (omit for updates)
    - `quantity`: The quantity value
    
    **Example:**
    ```json
    {
        "table_name": "product_manager",
        "master_data": {"product": "P1"},
        "date": "2026-02-13",
        "quantity": 100
    }
    ```
    """
    try:
        result = GenericDashboardService.save_data_row(
            database_name=tenant_data["database_name"],
            table_name=request.table_name,
            user_email=tenant_data["email"],
            record_id=request.record_id,
            master_data=request.master_data,
            plan_date=request.date,
            quantity=request.quantity,
        )
        return ResponseHandler.success(data=result)
        
    except (ValidationException, NotFoundException) as e:
        status_code = 404 if isinstance(e, NotFoundException) else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in save_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/save-aggregated-data", response_model=Dict[str, Any])
async def save_aggregated_data(
    request: SaveAggregatedDataRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Save aggregated data by distributing quantity among group members.
    
    The quantity is distributed based on existing ratios in the target table.
    
    **Example:**
    ```json
    {
        "table_name": "product_manager",
        "aggregated_fields": ["product", "region"],
        "group_data": {"product": "P1", "region": "North"},
        "date": "2026-02-13",
        "quantity": 500
    }
    ```
    """
    try:
        result = GenericDashboardService.save_aggregated_data_rows(
            database_name=tenant_data["database_name"],
            table_name=request.table_name,
            user_email=tenant_data["email"],
            aggregated_fields=request.aggregated_fields,
            group_data=request.group_data,
            plan_date=request.date,
            quantity=request.quantity,
        )
        return ResponseHandler.success(data=result)
        
    except (ValidationException, NotFoundException) as e:
        status_code = 404 if isinstance(e, NotFoundException) else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in save_aggregated_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/copy-data", response_model=Dict[str, Any])
async def copy_data(
    request: CopyDataRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Copy data between any two tables.
    
    **Example:**
    ```json
    {
        "source_table": "forecast_data",
        "target_table": "product_manager",
        "from_date": "2026-02-01",
        "to_date": "2026-02-28"
    }
    ```
    """
    try:
        result = GenericDashboardService.copy_data_between_tables(
            database_name=tenant_data["database_name"],
            source_table=request.source_table,
            target_table=request.target_table,
            user_email=tenant_data["email"],
            filters=request.filters or [],
            from_date=request.from_date,
            to_date=request.to_date,
        )
        return ResponseHandler.success(data=result)
        
    except (ValidationException, NotFoundException) as e:
        status_code = 404 if isinstance(e, NotFoundException) else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in copy_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Table Management Endpoints
# ============================================================================

@router.get("/tables", response_model=TableListResponse)
async def list_tables(
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard")),
):
    """
    List all available dynamic tables for the tenant.
    
    Returns both mandatory tables (final_plan) and custom tables.
    """
    try:
        tables = DynamicTableService.get_tenant_dynamic_tables(
            database_name=tenant_data["database_name"]
        )
        
        return TableListResponse(
            tables=tables,
            total_count=len(tables)
        )
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in list_tables: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/tables", response_model=Dict[str, Any])
async def create_table(
    request: CreateTableRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=3)),  # Admin only
):
    """
    Create a new dynamic table for the tenant.
    
    The `display_name` is automatically normalized to a valid table name.
    - Spaces are replaced with underscores
    - All letters are converted to lowercase
    - Special characters are removed
    
    **Example:**
    ```json
    {
        "display_name": "Product Manager",
        "table_type": "planning",
        "description": "Product manager approval level"
    }
    ```
    
    **Result:** Table created as `product_manager` in the database
    """
    try:
        result = DynamicTableService.create_dynamic_table(
            database_name=tenant_data["database_name"],
            display_name=request.display_name,
            table_type=request.table_type,
            description=request.description,
        )
        return ResponseHandler.success(data=result)
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in create_table: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/tables/{table_name}", response_model=Dict[str, Any])
async def delete_table(
    table_name: str,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=3)),  # Admin only
):
    """
    Delete a dynamic table.
    
    Note: Cannot delete mandatory tables like `final_plan`.
    
    **Parameters:**
    - `table_name`: Normalized table name (e.g., 'product_manager')
    """
    try:
        result = DynamicTableService.delete_dynamic_table(
            database_name=tenant_data["database_name"],
            table_name=table_name,
        )
        return ResponseHandler.success(data=result)
        
    except ValidationException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in delete_table: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
