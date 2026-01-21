"""
Master Data API Routes.
Endpoints for retrieving master data fields and their distinct values for UI dropdowns.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List
from app.schemas.master_data import (
    MasterDataFieldsResponse,
    FieldValuesRequest,
    FieldValuesResponse,
    MultipleFieldValuesRequest,
    MultipleFieldValuesResponse
)
from app.core.master_data_service import MasterDataService
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException
from app.api.dependencies import get_user_database, require_object_access
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/master-data", tags=["Master Data"])


@router.get("/fields", response_model=Dict[str, Any])
async def get_master_data_fields(
    tenant_data: Dict = Depends(get_user_database),
    _: Dict = Depends(require_object_access("Master Data"))
):
    """
    Get all available fields from master_data table for dropdown selection.

    Returns a list of fields that can be used in UI dropdowns for field selection.
    """
    try:
        # Check if master data exists
        if not MasterDataService.validate_master_data_exists(tenant_data["database_name"]):
            raise HTTPException(
                status_code=404,
                detail="Master data not found. Please ensure field catalogue is finalized and data is uploaded."
            )

        fields = MasterDataService.get_master_data_fields(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"]
        )

        response_data = MasterDataFieldsResponse(fields=fields)
        return ResponseHandler.success(data=response_data.dict())

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_master_data_fields: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/field-values", response_model=Dict[str, Any])
async def get_field_values(
    request: FieldValuesRequest,
    tenant_data: Dict = Depends(get_user_database),
    _: Dict = Depends(require_object_access("Master Data"))
):
    """
    Get distinct values for a specific field, optionally filtered by other fields.

    Used for populating dropdown values after a field is selected.
    Supports filtering based on selections from other fields.
    """
    try:
        # Check if master data exists
        if not MasterDataService.validate_master_data_exists(tenant_data["database_name"]):
            raise HTTPException(
                status_code=404,
                detail="Master data not found. Please ensure field catalogue is finalized and data is uploaded."
            )

        values = MasterDataService.get_field_values(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            field_name=request.field_name,
            filters=request.filters
        )

        response_data = FieldValuesResponse(
            field_name=request.field_name,
            values=values,
            total_count=len(values)
        )
        return ResponseHandler.success(data=response_data.dict())

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_field_values: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/multiple-field-values", response_model=Dict[str, Any])
async def get_multiple_field_values(
    request: MultipleFieldValuesRequest,
    tenant_data: Dict = Depends(get_user_database),
    _: Dict = Depends(require_object_access("Master Data"))
):
    """
    Get distinct values for multiple fields with cross-filtering.

    Allows UI to request values for multiple fields simultaneously,
    where each field's values are filtered based on selections from other fields.
    Supports the JSON input format: {'product': [list of products], 'customer': [list of customers]}
    """
    try:
        # Check if master data exists
        if not MasterDataService.validate_master_data_exists(tenant_data["database_name"]):
            raise HTTPException(
                status_code=404,
                detail="Master data not found. Please ensure field catalogue is finalized and data is uploaded."
            )

        field_values = MasterDataService.get_multiple_field_values(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            field_selections=request.field_selections
        )

        response_data = MultipleFieldValuesResponse(field_values=field_values)
        return ResponseHandler.success(data=response_data.dict())

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_multiple_field_values: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/field-values/{field_name}", response_model=Dict[str, Any])
async def get_field_values_simple(
    field_name: str,
    tenant_data: Dict = Depends(get_user_database),
    _: Dict = Depends(require_object_access("Master Data")),
    filters: str = Query(None, description="JSON string of filters for other fields")
):
    """
    Simplified GET endpoint for field values.

    Alternative to POST endpoint for simple cases.
    Filters should be provided as JSON string in query parameter.
    """
    try:
        # Check if master data exists
        if not MasterDataService.validate_master_data_exists(tenant_data["database_name"]):
            raise HTTPException(
                status_code=404,
                detail="Master data not found. Please ensure field catalogue is finalized and data is uploaded."
            )

        # Parse filters if provided
        parsed_filters = None
        if filters:
            try:
                import json
                parsed_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid filters JSON format"
                )

        values = MasterDataService.get_field_values(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            field_name=field_name,
            filters=parsed_filters
        )

        response_data = FieldValuesResponse(
            field_name=field_name,
            values=values,
            total_count=len(values)
        )
        return ResponseHandler.success(data=response_data.dict())

    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_field_values_simple: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
