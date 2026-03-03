"""
API routes for forecast scheduling.
Endpoints for creating, reading, updating, and deleting forecast schedules.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from app.api.dependencies import get_current_user, get_current_tenant
from app.schemas.forecast_schedules import (
    CreateForecastScheduleRequest,
    UpdateForecastScheduleRequest,
    ForecastScheduleResponse,
    ForecastScheduleListResponse,
    ExecuteScheduleNowRequest,
    ExecuteScheduleResponse
)
from app.core.forecast_scheduler_service import ForecastSchedulerService
from app.core.exceptions import ValidationException, NotFoundException, DatabaseException

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/schedules",
    tags=["Forecast Schedules"]
)


@router.post(
    "",
    summary="Create a new forecast schedule",
    response_model=ForecastScheduleResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"description": "Invalid request data"},
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    }
)
async def create_schedule(
    request: CreateForecastScheduleRequest,
    current_user: dict = Depends(get_current_user),
    tenant_data: dict = Depends(get_current_tenant)
) -> ForecastScheduleResponse:
    """
    Create a new forecast schedule.
    
    The schedule will be registered with APScheduler and will execute
    according to the cron expression provided.
    
    Args:
        request: Schedule creation request
        current_user: Current authenticated user
        tenant_data: Tenant information from token
        
    Returns:
        Created schedule details
        
    Raises:
        ValidationException: If validation fails
        HTTPException: On error
    """
    try:
        logger.info(f"Creating schedule '{request.schedule_name}' for tenant {tenant_data['tenant_id']}")
        
        schedule_id = ForecastSchedulerService.create_schedule(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            schedule_name=request.schedule_name,
            request_data=request.request_data,
            cron_expression=request.cron_expression,
            schedule_type=request.schedule_type.value,
            description=request.description,
            created_by=current_user.get('email', 'unknown'),
            is_active=request.is_active
        )
        
        # Retrieve and return the created schedule
        schedule = ForecastSchedulerService.get_schedule(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            schedule_id=schedule_id
        )
        
        return ForecastScheduleResponse(**schedule)
    
    except ValidationException as e:
        logger.warning(f"Validation error creating schedule: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating schedule: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create forecast schedule"
        )


@router.get(
    "",
    summary="List forecast schedules",
    response_model=ForecastScheduleListResponse,
    responses={
        401: {"description": "Unauthorized"},
        500: {"description": "Internal server error"}
    }
)
async def list_schedules(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: dict = Depends(get_current_user),
    tenant_data: dict = Depends(get_current_tenant)
) -> ForecastScheduleListResponse:
    """
    List forecast schedules for the tenant.
    
    Args:
        is_active: Filter by active status (optional)
        limit: Maximum records to return
        offset: Offset for pagination
        current_user: Current authenticated user
        tenant_data: Tenant information from token
        
    Returns:
        List of schedules with pagination info
        
    Raises:
        HTTPException: On error
    """
    try:
        logger.debug(f"Listing schedules for tenant {tenant_data['tenant_id']}")
        
        schedules, total_count = ForecastSchedulerService.list_schedules(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            is_active=is_active,
            limit=limit,
            offset=offset
        )
        
        schedule_responses = [ForecastScheduleResponse(**s) for s in schedules]
        
        return ForecastScheduleListResponse(
            schedules=schedule_responses,
            total_count=total_count,
            limit=limit,
            offset=offset
        )
    
    except Exception as e:
        logger.error(f"Error listing schedules: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list forecast schedules"
        )


@router.get(
    "/{schedule_id}",
    summary="Get a specific forecast schedule",
    response_model=ForecastScheduleResponse,
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Schedule not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_schedule(
    schedule_id: str,
    current_user: dict = Depends(get_current_user),
    tenant_data: dict = Depends(get_current_tenant)
) -> ForecastScheduleResponse:
    """
    Get details of a specific forecast schedule.
    
    Args:
        schedule_id: Schedule identifier
        current_user: Current authenticated user
        tenant_data: Tenant information from token
        
    Returns:
        Schedule details
        
    Raises:
        HTTPException: On error
    """
    try:
        logger.debug(f"Retrieving schedule {schedule_id}")
        
        schedule = ForecastSchedulerService.get_schedule(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            schedule_id=schedule_id
        )
        
        return ForecastScheduleResponse(**schedule)
    
    except NotFoundException as e:
        logger.warning(f"Schedule not found: {schedule_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving schedule: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve forecast schedule"
        )


@router.patch(
    "/{schedule_id}",
    summary="Update a forecast schedule",
    response_model=ForecastScheduleResponse,
    responses={
        400: {"description": "Invalid request data"},
        401: {"description": "Unauthorized"},
        404: {"description": "Schedule not found"},
        500: {"description": "Internal server error"}
    }
)
async def update_schedule(
    schedule_id: str,
    request: UpdateForecastScheduleRequest,
    current_user: dict = Depends(get_current_user),
    tenant_data: dict = Depends(get_current_tenant)
) -> ForecastScheduleResponse:
    """
    Update a forecast schedule.
    
    Only non-null fields will be updated. If the active status is changed,
    the schedule will be re-registered or removed from APScheduler.
    
    Args:
        schedule_id: Schedule identifier
        request: Update request
        current_user: Current authenticated user
        tenant_data: Tenant information from token
        
    Returns:
        Updated schedule details
        
    Raises:
        HTTPException: On error
    """
    try:
        logger.info(f"Updating schedule {schedule_id}")
        
        schedule = ForecastSchedulerService.update_schedule(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            schedule_id=schedule_id,
            schedule_name=request.schedule_name,
            description=request.description,
            request_data=request.request_data,
            cron_expression=request.cron_expression,
            schedule_type=request.schedule_type.value if request.schedule_type else None,
            is_active=request.is_active,
            updated_by=current_user.get('email', 'unknown')
        )
        
        return ForecastScheduleResponse(**schedule)
    
    except ValidationException as e:
        logger.warning(f"Validation error updating schedule: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except NotFoundException as e:
        logger.warning(f"Schedule not found: {schedule_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating schedule: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update forecast schedule"
        )


@router.delete(
    "/{schedule_id}",
    summary="Delete a forecast schedule",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Schedule not found"},
        500: {"description": "Internal server error"}
    }
)
async def delete_schedule(
    schedule_id: str,
    soft_delete: bool = Query(True, description="Use soft delete (mark as deleted) instead of permanent delete"),
    current_user: dict = Depends(get_current_user),
    tenant_data: dict = Depends(get_current_tenant)
) -> None:
    """
    Delete a forecast schedule.
    
    By default, performs a soft delete (marks as deleted). Use soft_delete=false
    for permanent deletion which cannot be recovered.
    
    Args:
        schedule_id: Schedule identifier
        soft_delete: Whether to use soft delete
        current_user: Current authenticated user
        tenant_data: Tenant information from token
        
    Raises:
        HTTPException: On error
    """
    try:
        logger.info(f"Deleting schedule {schedule_id} (soft_delete={soft_delete})")
        
        ForecastSchedulerService.delete_schedule(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            schedule_id=schedule_id,
            soft_delete=soft_delete
        )
    
    except NotFoundException as e:
        logger.warning(f"Schedule not found: {schedule_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error deleting schedule: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete forecast schedule"
        )


@router.post(
    "/{schedule_id}/execute",
    summary="Manually execute a forecast schedule",
    response_model=ExecuteScheduleResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Schedule not found"},
        500: {"description": "Internal server error"}
    }
)
async def execute_schedule_now(
    schedule_id: str,
    request: ExecuteScheduleNowRequest = None,
    current_user: dict = Depends(get_current_user),
    tenant_data: dict = Depends(get_current_tenant)
) -> ExecuteScheduleResponse:
    """
    Manually execute a forecast schedule immediately.
    
    This bypasses the schedule timing and executes the forecast right away.
    Optionally, you can override the schedule's request_data parameters.
    
    Args:
        schedule_id: Schedule identifier
        request: Optional override parameters
        current_user: Current authenticated user
        tenant_data: Tenant information from token
        
    Returns:
        Job creation confirmation
        
    Raises:
        HTTPException: On error
    """
    try:
        logger.info(f"Manually executing schedule {schedule_id}")
        
        # Get the schedule
        schedule = ForecastSchedulerService.get_schedule(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            schedule_id=schedule_id
        )
        
        # Prepare request data with overrides if provided
        request_data = schedule['request_data'].copy()
        if request and request.override_parameters:
            # Merge override parameters (deep merge for nested dicts)
            def deep_merge(base, overrides):
                for key, value in overrides.items():
                    if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                        deep_merge(base[key], value)
                    else:
                        base[key] = value
            
            deep_merge(request_data, request.override_parameters)
        
        # Execute the schedule immediately
        job_id = ForecastSchedulerService._execute_scheduled_forecast.__wrapped__(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            schedule_id=schedule_id,
            request_data=request_data
        )
        
        # Get the job ID from the executed forecast
        from app.core.forecast_job_service import ForecastJobService
        # The execution returns None, so we generate a response with schedule info
        
        return ExecuteScheduleResponse(
            job_id="See forecast jobs for status",
            schedule_id=schedule_id,
            message="Forecast execution initiated successfully"
        )
    
    except NotFoundException as e:
        logger.warning(f"Schedule not found: {schedule_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing schedule: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute forecast schedule"
        )
