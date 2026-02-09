"""
Dashboard API Routes.
Endpoints for aggregated sales, forecast, and final plan data.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import date as DateType
from io import BytesIO

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.dependencies import get_tenant_database, require_object_access
from app.core.dashboard_service import DashboardService
from app.core.exceptions import AppException, ValidationException, NotFoundException
from app.core.responses import ResponseHandler
from app.schemas.sales_data import (
    SalesDataQueryRequest,
    AggregatedDataQueryRequest,
    SalesDataFilter,
)
from app.core.master_data_service import MasterDataService
from openpyxl import Workbook

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


class SaveFinalPlanRequest(BaseModel):
    final_plan_id: Optional[str] = Field(
        default=None,
        description="Final plan primary key for updates",
    )
    master_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Master data fields to resolve master_id",
    )
    date: Optional[DateType] = Field(
        default=None,
        description="Final plan date (YYYY-MM-DD)",
    )
    quantity: float = Field(..., description="Final plan quantity")


class SaveProductManagerRequest(BaseModel):
    product_manager_id: Optional[str] = Field(
        default=None,
        description="Product manager primary key for updates",
    )
    master_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Master data fields to resolve master_id",
    )
    date: Optional[DateType] = Field(
        default=None,
        description="Product manager date (YYYY-MM-DD)",
    )
    quantity: float = Field(..., description="Product manager quantity")


class CopyForecastToFinalPlanRequest(BaseModel):
    filters: Optional[List[SalesDataFilter]] = Field(
        default=None,
        description="List of master data filters. Empty or omitted means all.",
    )
    from_date: Optional[DateType] = Field(
        default=None,
        description="Start date (inclusive) to copy forecast data.",
    )
    to_date: Optional[DateType] = Field(
        default=None,
        description="End date (inclusive) to copy forecast data.",
    )


class CopyDashboardDataRequest(BaseModel):
    copy_from: str = Field(
        ...,
        description="Source data type: baseline_forecast or product_manager",
    )
    copy_to: str = Field(
        ...,
        description="Target data type: product_manager or final_consensus_plan",
    )
    filters: Optional[List[SalesDataFilter]] = Field(
        default=None,
        description="List of master data filters. Empty or omitted means all.",
    )
    from_date: Optional[DateType] = Field(
        default=None,
        description="Start date (inclusive) to copy data.",
    )
    to_date: Optional[DateType] = Field(
        default=None,
        description="End date (inclusive) to copy data.",
    )


class SaveAggregatedProductManagerRequest(BaseModel):
    aggregated_fields: List[str] = Field(
        ...,
        description="List of field names used for aggregation (e.g., ['product', 'location'])",
    )
    group_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary of field values defining the aggregated group (e.g., {'product': 'P1', 'location': 'L1'})",
    )
    date: DateType = Field(..., description="Date for the aggregated quantity (YYYY-MM-DD)")
    quantity: float = Field(..., description="Aggregated quantity to distribute among group members")


class SaveAggregatedFinalPlanRequest(BaseModel):
    aggregated_fields: List[str] = Field(
        ...,
        description="List of field names used for aggregation (e.g., ['product', 'location'])",
    )
    group_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary of field values defining the aggregated group (e.g., {'product': 'P1', 'location': 'L1'})",
    )
    date: DateType = Field(..., description="Date for the aggregated quantity (YYYY-MM-DD)")
    quantity: float = Field(..., description="Aggregated quantity to distribute among group members")


@router.post("/alldata", response_model=Dict[str, Any])
async def get_dashboard_all_data(
    request: SalesDataQueryRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard")),
):
    """
    Retrieve paginated master data and their related sales, forecast, and final
    plan records within the requested date range.
    """
    try:
        result = DashboardService.get_all_data_ui(
            database_name=tenant_data["database_name"],
            request=request,
        )

        return ResponseHandler.list_response(
            data=result["records"],
            page=request.page,
            page_size=request.page_size,
            total_count=result["total_count"],
        )

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_dashboard_all_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/aggregated-data", response_model=Dict[str, Any])
async def get_dashboard_aggregated_data(
    request: AggregatedDataQueryRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard")),
):
    """
    Retrieve aggregated master data and their related sales, forecast, and final
    plan records within the requested date range.
    """
    try:
        result = DashboardService.get_aggregated_data_ui(
            database_name=tenant_data["database_name"],
            request=request,
        )

        return ResponseHandler.list_response(
            data=result["records"],
            page=request.page,
            page_size=request.page_size,
            total_count=result["total_count"],
        )

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_dashboard_aggregated_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/save-final-plan", response_model=Dict[str, Any])
async def save_final_plan(
    request: SaveFinalPlanRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Save or update final plan data.

    - If final_plan_id is provided, updates quantity for that record.
    - Otherwise inserts a new record using master_data, date, and quantity.
    """
    try:
        final_plan_id = request.final_plan_id.strip() if request.final_plan_id else ""
        if final_plan_id:
            result = DashboardService.save_final_plan(
                database_name=tenant_data["database_name"],
                user_email=tenant_data["email"],
                final_plan_id=final_plan_id,
                master_data=None,
                plan_date=None,
                quantity=request.quantity,
            )
            return ResponseHandler.success(data=result)

        if not request.master_data:
            raise ValidationException("master_data is required")
        if request.date is None:
            raise ValidationException("date is required")

        result = DashboardService.save_final_plan(
            database_name=tenant_data["database_name"],
            user_email=tenant_data["email"],
            final_plan_id=None,
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
        logger.error(f"Unexpected error in save_final_plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/save-product-manager", response_model=Dict[str, Any])
async def save_product_manager(
    request: SaveProductManagerRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Save or update product manager data.

    - If product_manager_id is provided, updates quantity for that record.
    - Otherwise inserts a new record using master_data, date, and quantity.
    """
    try:
        product_manager_id = (
            request.product_manager_id.strip()
            if request.product_manager_id
            else ""
        )
        if product_manager_id:
            result = DashboardService.save_product_manager(
                database_name=tenant_data["database_name"],
                user_email=tenant_data["email"],
                product_manager_id=product_manager_id,
                master_data=None,
                plan_date=None,
                quantity=request.quantity,
            )
            return ResponseHandler.success(data=result)

        if not request.master_data:
            raise ValidationException("master_data is required")
        if request.date is None:
            raise ValidationException("date is required")

        result = DashboardService.save_product_manager(
            database_name=tenant_data["database_name"],
            user_email=tenant_data["email"],
            product_manager_id=None,
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
        logger.error(f"Unexpected error in save_product_manager: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/copy-forecast-to-final-plan", response_model=Dict[str, Any])
async def copy_forecast_to_final_plan(
    request: CopyForecastToFinalPlanRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Copy forecast_data records into final_plan with optional filters and date range.
    Existing final_plan rows with the same master_id and date are overwritten.
    """
    try:
        result = DashboardService.copy_forecast_to_final_plan(
            database_name=tenant_data["database_name"],
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
        logger.error(
            f"Unexpected error in copy_forecast_to_final_plan: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/copy-dashboard-data", response_model=Dict[str, Any])
async def copy_dashboard_data(
    request: CopyDashboardDataRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Copy data between dashboard tables based on requested source and target.
    Supported:
    - baseline_forecast -> product_manager
    - product_manager -> final_consensus_plan
    """
    try:
        result = DashboardService.copy_dashboard_data(
            database_name=tenant_data["database_name"],
            user_email=tenant_data["email"],
            copy_from=request.copy_from,
            copy_to=request.copy_to,
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
        logger.error(f"Unexpected error in copy_dashboard_data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/save-aggregated-product-manager", response_model=Dict[str, Any])
async def save_aggregated_product_manager(
    request: SaveAggregatedProductManagerRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Save aggregated product manager data by distributing the quantity among group members
    using historical sales distribution ratios.

    The aggregated quantity is distributed proportionally based on historical sales data
    for the group defined by aggregated_fields and group_data.
    """
    try:
        result = DashboardService.save_aggregated_product_manager(
            database_name=tenant_data["database_name"],
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
        logger.error(f"Unexpected error in save_aggregated_product_manager: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/save-aggregated-final-plan", response_model=Dict[str, Any])
async def save_aggregated_final_plan(
    request: SaveAggregatedFinalPlanRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard", min_role_id=2)),
):
    """
    Save aggregated final plan data by distributing the quantity among group members
    using historical sales distribution ratios.

    The aggregated quantity is distributed proportionally based on historical sales data
    for the group defined by aggregated_fields and group_data.
    """
    try:
        result = DashboardService.save_aggregated_final_plan(
            database_name=tenant_data["database_name"],
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
        logger.error(f"Unexpected error in save_aggregated_final_plan: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/export-xlsx")
async def export_dashboard_xlsx(
    request: SalesDataQueryRequest,
    tenant_data: Dict = Depends(get_tenant_database),
    _: Dict = Depends(require_object_access("Dashboard")),
):
    """
    Export dashboard data as an Excel file (no pagination).
    """
    try:
        result = DashboardService.get_all_data_export(
            database_name=tenant_data["database_name"],
            request=request,
        )
        records = result["records"]

        fields = MasterDataService.get_master_data_fields(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
        )
        field_names = [field["field_name"] for field in fields]
        field_headers = [
            field.get("display_name") or field["field_name"] for field in fields
        ]

        start_date = request.from_date or DateType(2025, 1, 1)
        end_date = request.to_date or DateType(2026, 3, 31)
        months = []
        cursor = DateType(start_date.year, start_date.month, 1)
        end_month = DateType(end_date.year, end_date.month, 1)
        while cursor <= end_month:
            months.append(cursor)
            if cursor.month == 12:
                cursor = DateType(cursor.year + 1, 1, 1)
            else:
                cursor = DateType(cursor.year, cursor.month + 1, 1)

        month_headers = [m.strftime("%b %Y") for m in months]
        month_keys = [f"{m.year}-{m.month:02d}" for m in months]

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Dashboard"

        header = ["S.No", "Type", *field_headers, "UOM", *month_headers]
        sheet.append(header)

        def summarize_values(values: List[Dict[str, Any]]) -> Dict[str, float]:
            totals: Dict[str, float] = {}
            for entry in values:
                date_value = entry.get("date")
                if not date_value:
                    continue
                try:
                    parsed = DateType.fromisoformat(str(date_value))
                except ValueError:
                    continue
                key = f"{parsed.year}-{parsed.month:02d}"
                quantity = entry.get("Quantity", entry.get("quantity"))
                if quantity is None:
                    continue
                totals[key] = totals.get(key, 0.0) + float(quantity)
            return totals

        def resolve_uom(values: List[Dict[str, Any]]) -> str:
            for entry in values:
                uom = entry.get("UOM") or entry.get("uom")
                if uom:
                    return str(uom)
            return ""

        row_index = 1
        for record in records:
            master_data = record.get("master_data") or {}
            master_values = [
                str(master_data.get(field_name, "")) for field_name in field_names
            ]

            for label, key in (
                ("Sales history", "sales_data"),
                ("Baseline forecast", "forecast_data"),
                ("Product manager", "product_manager"),
                ("Final consensus plan", "final_plan"),
            ):
                values = record.get(key) or []
                totals = summarize_values(values)
                uom_value = resolve_uom(values)
                row = [
                    row_index,
                    label,
                    *master_values,
                    uom_value,
                    *[
                        totals.get(month_key, "")
                        if month_key in totals
                        else ""
                        for month_key in month_keys
                    ],
                ]
                sheet.append(row)
                row_index += 1

        output = BytesIO()
        workbook.save(output)
        output.seek(0)

        headers = {
            "Content-Disposition": "attachment; filename=dashboard_export.xlsx"
        }
        return StreamingResponse(
            output,
            media_type=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ),
            headers=headers,
        )

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in export_dashboard_xlsx: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
