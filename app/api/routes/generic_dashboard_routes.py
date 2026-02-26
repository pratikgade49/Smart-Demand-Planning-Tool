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

from app.schemas.disaggregation_schema import DisaggregateTableRequest
from app.core.table_disaggregation_service import TableDisaggregationService


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard-Generic"])


# ============================================================================
# Request Models
# ============================================================================

class SaveDataRequest(BaseModel):
    """Request to save/update data in any dynamic table using UPSERT."""
    
    table_name: str = Field(
        ...,
        description="Normalized table name (e.g., 'product_manager', 'marketing_team')"
    )
    master_data: Dict[str, Any] = Field(
        ...,
        description="Master data fields to identify the record (e.g., {'product': 'P1', 'location': 'North'})"
    )
    date: DateType = Field(
        ...,
        description="Record date (YYYY-MM-DD)"
    )
    quantity: float = Field(
        ...,
        description="Quantity value"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "table_name": "product_manager",
                "master_data": {
                    "product": "P1",
                    "location": "North"
                },
                "date": "2026-02-13",
                "quantity": 100
            }
        }





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
    Save or update a data row in any dynamic table using UPSERT.
    
    **Uses master_data + date as the natural unique key.**
    
    - If a record exists for the given master_data combination + date ->  **UPDATE** quantity
    - If no record exists ->  **INSERT** new record
    - Atomic operation using PostgreSQL's ON CONFLICT clause
    
    **Parameters:**
    - `table_name`: Table name (e.g., 'product_manager', 'marketing_team')
    - `master_data`: Fields that identify the master record (e.g., {"product": "P1", "location": "North"})
    - `date`: Record date (YYYY-MM-DD)
    - `quantity`: Quantity value
    
    **Example:**
    ```json
    {
        "table_name": "product_manager",
        "master_data": {
            "product": "P1",
            "location": "North"
        },
        "date": "2026-02-13",
        "quantity": 100
    }
    ```
    
    - First call: Inserts new record with quantity 100
    - Second call with same master_data + date but quantity 150: Updates to 150
    - Call with different date: Inserts new record for that date
    """
    try:
        # Validate required fields
        if not request.master_data:
            raise ValidationException("master_data is required")
        if request.date is None:
            raise ValidationException("date is required")
        
        # UPSERT: Insert new record or update if (master_id, date) exists
        result = GenericDashboardService.upsert_data_row(
            database_name=tenant_data["database_name"],
            table_name=request.table_name,
            user_email=tenant_data["email"],
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

@router.get("/disaggregation-methods", response_model=Dict[str, Any])
async def get_disaggregation_methods(
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard")),
):
    """
    Returns the three supported disaggregation methods and the list of tables
    that can be used as a key figure reference.

    **Response shape:**
    ```json
    {
      "methods": [
        { "value": "own_ratio",   "label": "Own Ratio",          "description": "..." },
        { "value": "key_figure",  "label": "Key Figure",         "description": "..." },
        { "value": "equal",       "label": "Equal Distribution", "description": "..." }
      ],
      "available_key_figure_tables": [
        { "table_name": "sales_data",    "display_name": "Sales History" },
        { "table_name": "forecast_data", "display_name": "Baseline Forecast" },
        { "table_name": "final_plan",    "display_name": "Final Consensus Plan" },
        ...
      ]
    }
    ```
    """
    try:
        dynamic_tables = DynamicTableService.get_tenant_dynamic_tables(
            database_name=tenant_data["database_name"],
            include_mandatory=True,
        )

        core_tables = [
            {"table_name": "sales_data",    "display_name": "Sales History"},
            {"table_name": "forecast_data", "display_name": "Baseline Forecast"},
        ]

        dynamic_names = {t["table_name"] for t in dynamic_tables}
        key_figure_tables = [t for t in core_tables if t["table_name"] not in dynamic_names]
        key_figure_tables += [
            {"table_name": t["table_name"], "display_name": t["display_name"]}
            for t in dynamic_tables
        ]

        return ResponseHandler.success(
            data={
                "methods": [
                    {
                        "value": "own_ratio",
                        "label": "Own Ratio",
                        "description": (
                            "Ratios are calculated from the target table's existing data. "
                            "The new total is redistributed proportionally."
                        ),
                    },
                    {
                        "value": "key_figure",
                        "label": "Key Figure",
                        "description": (
                            "Ratios are derived from a different reference table "
                            "(e.g. Sales History or another plan). "
                            "Select the reference table via key_figure_table."
                        ),
                    },
                    {
                        "value": "equal",
                        "label": "Equal Distribution",
                        "description": (
                            "The total is divided equally across all lower-level members. "
                            "No historical data is needed."
                        ),
                    },
                ],
                "available_key_figure_tables": key_figure_tables,
            }
        )

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error in get_disaggregation_methods: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ═══════════════════════════════════════════════════════════════════════════════
# POST /dashboard/disaggregate-table
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/disaggregate-table", response_model=Dict[str, Any])
async def disaggregate_table(
    request: DisaggregateTableRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Disaggregate an aggregated quantity into lower-level member records
    using one of three distribution methods.

    ---

    ### Two modes

    **Single-date mode** — disaggregate one specific cell:
    ```json
    {
      "target_table": "final_plan",
      "master_data":  { "product": "P1" },
      "quantity":     1200,
      "date":         "2026-03-01",
      "method":       "own_ratio"
    }
    ```

    **Date-range mode** — disaggregate every period that has data in the
    source table within the range (no `quantity` needed — totals are read
    from the table automatically):
    ```json
    {
      "target_table": "final_plan",
      "master_data":  { "product": "P1" },
      "date_from":    "2026-01-01",
      "date_to":      "2026-12-31",
      "method":       "own_ratio"
    }
    ```

    ---

    ### Methods

    | method       | Ratio source                          | key_figure_table |
    |--------------|---------------------------------------|------------------|
    | `own_ratio`  | Target table's own existing data      | Not required     |
    | `key_figure` | A separate reference table you choose | **Required**     |
    | `equal`      | No data needed — equal split          | Not required     |

    ---

    ### Notes
    - **Zero quantity** is allowed and will write `0` to every member row.
    - If the ratio source table has no data, the service **falls back to
      equal distribution** automatically and logs a warning.
    - In date-range mode the **source table** used to read period totals is:
        - `own_ratio` / `equal` → `target_table`
        - `key_figure` → `key_figure_table`
    - `ratio_date_from` / `ratio_date_to` control the **look-back window**
      for ratio calculation and are independent of `date_from` / `date_to`.

    ---

    ### Response
    ```json
    {
      "status": "success",
      "method": "own_ratio",
      "target_table": "final_plan",
      "mode": "range",
      "date_from": "2026-01-01",
      "date_to": "2026-12-31",
      "periods_processed": 12,
      "members_count": 3,
      "records_upserted": 36,
      "periods": [
        {
          "date": "2026-01-01",
          "total_quantity": 900,
          "records_upserted": 3,
          "distribution": [
            { "master_id": "...", "ratio": 0.5,  "quantity": 450 },
            { "master_id": "...", "ratio": 0.3,  "quantity": 270 },
            { "master_id": "...", "ratio": 0.2,  "quantity": 180 }
          ]
        },
        ...
      ]
    }
    ```
    """
    try:
        logger.info(
            f"Disaggregate-table request: table={request.target_table}, "
            f"method={request.method}, master_data={request.master_data}, "
            f"mode={'range' if request.date_from else 'single'}"
        )

        result = TableDisaggregationService.disaggregate_table_data(
            database_name=tenant_data["database_name"],
            request=request,
            user_email=tenant_data["email"],
        )

        return ResponseHandler.success(data=result)

    except (ValidationException, NotFoundException) as e:
        status_code = 404 if isinstance(e, NotFoundException) else 400
        raise HTTPException(status_code=status_code, detail=str(e))
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Error in disaggregate_table: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


