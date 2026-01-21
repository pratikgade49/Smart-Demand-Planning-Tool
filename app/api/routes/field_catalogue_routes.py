"""
Field Catalogue API Routes.
Endpoints for field catalogue management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any
from app.schemas.field_catalogue import (
    FieldCatalogueRequest,
    FieldCatalogueResponse
)
from app.core.field_catalogue_service import FieldCatalogueService
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException
from app.api.dependencies import get_user_database, get_current_user, require_object_access
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/field-catalogue", tags=["Field Catalogue"])


@router.post("", response_model=Dict[str, Any])
async def create_field_catalogue(
    request: FieldCatalogueRequest,
    tenant_data: Dict = Depends(get_user_database),
    _: Dict = Depends(require_object_access("Planning Attributes"))
):
    """
    Create a new field catalogue in DRAFT status.
    Validates that parent fields exist and no circular references.
    """
    try:
        # Pre-validate parent fields
        parent_errors = []
        for field in request.fields:
            if field.is_characteristic and field.parent_field_name:
                if not FieldCatalogueService.validate_parent_field_exists(
                    request.fields, 
                    field.parent_field_name
                ):
                    parent_errors.append(
                        f"Parent field '{field.parent_field_name}' not found for characteristic '{field.field_name}'"
                    )
        
        if parent_errors:
            raise HTTPException(
                status_code=400, 
                detail={"errors": parent_errors}
            )
        
        # Check for circular references
        circular_errors = FieldCatalogueService.detect_circular_references(request.fields)
        if circular_errors:
            raise HTTPException(
                status_code=400, 
                detail={"errors": circular_errors}
            )
        
        result = FieldCatalogueService.create_field_catalogue(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            request=request,
            user_email=tenant_data["email"]
        )
        return ResponseHandler.success(data=result, status_code=201)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in create_field_catalogue: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.post("/{catalogue_id}/finalize", response_model=Dict[str, Any])
async def finalize_field_catalogue(
    catalogue_id: str,
    tenant_data: Dict = Depends(get_current_user),
    _: Dict = Depends(require_object_access("Planning Attributes"))
):
    """
    Finalize a field catalogue and create master data table.
    This action is irreversible and creates the database schema.
    """
    try:
        result = FieldCatalogueService.finalize_field_catalogue(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            catalogue_id=catalogue_id,
            user_email=tenant_data["email"]
        )
        return ResponseHandler.success(data=result)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in finalize_field_catalogue: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{catalogue_id}", response_model=Dict[str, Any])
async def get_field_catalogue(
    catalogue_id: str,
    tenant_data: Dict = Depends(get_current_user),
    _: Dict = Depends(require_object_access("Planning Attributes"))
):
    """Get field catalogue details by ID."""
    try:
        result = FieldCatalogueService.get_field_catalogue(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            catalogue_id=catalogue_id
        )
        return ResponseHandler.success(data=result)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in get_field_catalogue: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("", response_model=Dict[str, Any])
async def list_field_catalogues(
    tenant_data: Dict = Depends(get_current_user),
    _: Dict = Depends(require_object_access("Planning Attributes")),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Records per page")
):
    """List all field catalogues for the tenant."""
    try:
        catalogues, total_count = FieldCatalogueService.list_field_catalogues(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            page=page,
            page_size=page_size
        )
        
        return ResponseHandler.list_response(
            data=catalogues,
            page=page,
            page_size=page_size,
            total_count=total_count
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in list_field_catalogues: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")