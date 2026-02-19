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

from app.api.dependencies import get_tenant_database, require_object_access
from app.core.dashboard_service import DashboardService
from app.core.dynamic_table_service import DynamicTableService
from app.core.exceptions import AppException
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
        
        # Get all dynamic tables to include in export
        dynamic_tables = DynamicTableService.get_tenant_dynamic_tables(
            database_name=tenant_data["database_name"]
        )
        
        # Build the list of data series to export
        data_series = [
            ("Sales history", "sales_data"),
            ("Baseline forecast", "forecast_data")
        ]
        
        # Add all dynamic tables (this includes mandatory ones like final_plan)
        for table in dynamic_tables:
            # Avoid duplication if already in list
            if table['table_name'] not in [s[1] for s in data_series]:
                data_series.append((table['display_name'], table['table_name']))

        for record in records:
            master_data = record.get("master_data") or {}
            master_values = [
                str(master_data.get(field_name, "")) for field_name in field_names
            ]

            for label, key in data_series:
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
