"""
Pydantic schemas for forecast scheduling.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class ScheduleTypeEnum(str, Enum):
    """Supported schedule types."""
    ONCE = "once"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ScheduleRunStatusEnum(str, Enum):
    """Schedule run status."""
    PENDING = "Pending"
    IN_PROGRESS = "In-Progress"
    COMPLETED = "Completed"
    COMPLETED_WITH_ERRORS = "Completed with Errors"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class CreateForecastScheduleRequest(BaseModel):
    """Request to create a new forecast schedule."""
    
    schedule_name: str = Field(
        ..., 
        min_length=1, 
        max_length=255,
        description="Name of the forecast schedule"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional description of the schedule"
    )
    request_data: Dict[str, Any] = Field(
        ...,
        description="Complete forecast configuration (algorithms, filters, etc.)"
    )
    cron_expression: str = Field(
        ...,
        description="Cron format expression (e.g., '0 2 * * 1' for Monday at 2 AM)"
    )
    schedule_type: ScheduleTypeEnum = Field(
        ...,
        description="Type of schedule (once, hourly, daily, weekly, monthly, custom)"
    )
    is_active: bool = Field(
        default=True,
        description="Whether the schedule is active"
    )
    
    @validator('cron_expression')
    def validate_cron(cls, v):
        """Validate cron expression format."""
        parts = v.strip().split()
        if len(parts) != 5:
            raise ValueError("Cron expression must have 5 fields (minute hour day month weekday)")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "schedule_name": "Weekly Monday Forecast",
                "description": "Runs every Monday at 2 AM",
                "request_data": {
                    "forecast_filters": {
                        "aggregation_level": "product",
                        "interval": "MONTHLY"
                    },
                    "algorithms": [
                        {
                            "algorithm_id": 1,
                            "execution_order": 1
                        }
                    ],
                    "forecast_start": "2024-01-01",
                    "forecast_end": "2024-12-31"
                },
                "cron_expression": "0 2 * * 1",
                "schedule_type": "weekly",
                "is_active": True
            }
        }


class UpdateForecastScheduleRequest(BaseModel):
    """Request to update a forecast schedule."""
    
    schedule_name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255,
        description="Name of the forecast schedule"
    )
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Description of the schedule"
    )
    request_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Forecast configuration"
    )
    cron_expression: Optional[str] = Field(
        None,
        description="Cron format expression"
    )
    schedule_type: Optional[ScheduleTypeEnum] = Field(
        None,
        description="Type of schedule"
    )
    is_active: Optional[bool] = Field(
        None,
        description="Whether the schedule is active"
    )
    
    @validator('cron_expression')
    def validate_cron(cls, v):
        """Validate cron expression format."""
        if v is not None:
            parts = v.strip().split()
            if len(parts) != 5:
                raise ValueError("Cron expression must have 5 fields (minute hour day month weekday)")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "is_active": False
            }
        }


class ForecastScheduleResponse(BaseModel):
    """Response model for a forecast schedule."""
    
    schedule_id: str = Field(..., description="Unique schedule identifier")
    schedule_name: str = Field(..., description="Name of the schedule")
    description: Optional[str] = Field(None, description="Schedule description")
    request_data: Dict[str, Any] = Field(..., description="Forecast configuration")
    cron_expression: str = Field(..., description="Cron format schedule expression")
    schedule_type: str = Field(..., description="Type of schedule")
    is_active: bool = Field(..., description="Whether schedule is active")
    last_run: Optional[datetime] = Field(None, description="Last execution timestamp")
    next_run: Optional[datetime] = Field(None, description="Next scheduled execution")
    last_job_id: Optional[str] = Field(None, description="Last executed job ID")
    last_run_status: Optional[str] = Field(None, description="Status of last run")
    last_run_error: Optional[str] = Field(None, description="Error message from last run")
    execution_count: int = Field(default=0, description="Total number of executions")
    created_at: datetime = Field(..., description="Creation timestamp")
    created_by: str = Field(..., description="User who created the schedule")
    updated_at: datetime = Field(..., description="Last update timestamp")
    updated_by: str = Field(..., description="User who last updated the schedule")
    
    class Config:
        from_attributes = True
        schema_extra = {
            "example": {
                "schedule_id": "550e8400-e29b-41d4-a716-446655440000",
                "schedule_name": "Weekly Monday Forecast",
                "description": "Runs every Monday at 2 AM",
                "request_data": {},
                "cron_expression": "0 2 * * 1",
                "schedule_type": "weekly",
                "is_active": True,
                "last_run": "2024-01-08T02:00:00",
                "next_run": "2024-01-15T02:00:00",
                "last_job_id": "job-uuid",
                "last_run_status": "Completed",
                "last_run_error": None,
                "execution_count": 5,
                "created_at": "2024-01-01T10:00:00",
                "created_by": "admin@example.com",
                "updated_at": "2024-01-08T02:00:00",
                "updated_by": "scheduler"
            }
        }


class ForecastScheduleListResponse(BaseModel):
    """Response model for listing forecast schedules."""
    
    schedules: List[ForecastScheduleResponse] = Field(..., description="List of schedules")
    total_count: int = Field(..., description="Total number of schedules")
    limit: int = Field(..., description="Query limit")
    offset: int = Field(..., description="Query offset")
    
    class Config:
        schema_extra = {
            "example": {
                "schedules": [],
                "total_count": 0,
                "limit": 100,
                "offset": 0
            }
        }


class ExecuteScheduleNowRequest(BaseModel):
    """Request to manually execute a schedule."""
    
    override_parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional parameters to override the schedule's request_data"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "override_parameters": {
                    "forecast_filters": {
                        "aggregation_level": "category"
                    }
                }
            }
        }


class ExecuteScheduleResponse(BaseModel):
    """Response for manually executing a schedule."""
    
    job_id: str = Field(..., description="Created forecast job ID")
    schedule_id: str = Field(..., description="Schedule ID")
    message: str = Field(..., description="Status message")
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "job-uuid",
                "schedule_id": "schedule-uuid",
                "message": "Forecast execution initiated successfully"
            }
        }
