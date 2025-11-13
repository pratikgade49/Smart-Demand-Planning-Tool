"""
Pydantic schemas for forecasting operations.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from uuid import UUID

# Algorithm Schemas
class AlgorithmBase(BaseModel):
    algorithm_name: str
    default_parameters: Dict[str, Any] = Field(..., description="JSON object with algorithm parameters")
    algorithm_type: str = Field(..., pattern="^(ML|Statistic|Hybrid)$")
    description: Optional[str] = None

class AlgorithmResponse(AlgorithmBase):
    algorithm_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Forecast Version Schemas
class ForecastVersionBase(BaseModel):
    version_name: str
    version_type: str = Field(..., pattern="^(Baseline|Simulation|Final)$")
    is_active: Optional[bool] = False

class ForecastVersionCreate(ForecastVersionBase):
    pass

class ForecastVersionUpdate(BaseModel):
    version_name: Optional[str] = None
    is_active: Optional[bool] = None

class ForecastVersionResponse(ForecastVersionBase):
    version_id: UUID
    tenant_id: UUID
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    class Config:
        from_attributes = True

# External Factors Schemas
class ExternalFactorBase(BaseModel):
    date: date
    factor_name: str
    # Use float without pydantic's decimal_places constraint (not applicable to float)
    factor_value: float = Field(...)
    unit: Optional[str] = None
    source: Optional[str] = None

class ExternalFactorCreate(ExternalFactorBase):
    pass

class ExternalFactorUpdate(BaseModel):
    factor_value: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None

class ExternalFactorResponse(ExternalFactorBase):
    factor_id: UUID
    tenant_id: UUID
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    class Config:
        from_attributes = True

# Forecast Algorithm Mapping Schemas
class AlgorithmMappingBase(BaseModel):
    algorithm_id: int
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="Override default parameters")
    execution_order: int = Field(default=1, ge=1)

class AlgorithmMappingCreate(AlgorithmMappingBase):
    pass

class AlgorithmMappingResponse(AlgorithmMappingBase):
    mapping_id: UUID
    forecast_run_id: UUID
    algorithm_name: str
    execution_status: str
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        from_attributes = True

# Forecast Run Schemas
class ForecastRunBase(BaseModel):
    version_id: UUID
    forecast_start: date
    forecast_end: date
    forecast_filters: Optional[Dict[str, Any]] = None
    run_percentage_frequency: Optional[int] = Field(default=10, ge=1, le=100)

class ForecastRunCreate(ForecastRunBase):
    algorithms: List[AlgorithmMappingCreate] = Field(..., description="List of algorithms to run")

class ForecastRunResponse(ForecastRunBase):
    forecast_run_id: UUID
    tenant_id: UUID
    run_status: str
    run_progress: int
    total_records: int
    processed_records: int
    failed_records: int
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    algorithms: Optional[List[AlgorithmMappingResponse]] = None

    class Config:
        from_attributes = True

# Forecast Results Schemas
class ForecastResultBase(BaseModel):
    forecast_date: date
    # Use float without pydantic's decimal_places constraint (not applicable to float)
    forecast_quantity: float = Field(...)
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    confidence_level: Optional[str] = None
    accuracy_metric: Optional[float] = None
    metric_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ForecastResultResponse(ForecastResultBase):
    result_id: UUID
    forecast_run_id: UUID
    version_id: UUID
    algorithm_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Forecast Results Query Schemas
class ForecastResultsFilter(BaseModel):
    forecast_run_id: Optional[UUID] = None
    version_id: Optional[UUID] = None
    algorithm_id: Optional[int] = None
    forecast_date_start: Optional[date] = None
    forecast_date_end: Optional[date] = None

class ForecastResultsPaginated(BaseModel):
    total: int
    skip: int
    limit: int
    results: List[ForecastResultResponse]

# Audit Log Schemas
class ForecastAuditLogResponse(BaseModel):
    audit_id: UUID
    forecast_run_id: UUID
    action: str
    entity_type: Optional[str] = None
    entity_id: Optional[UUID] = None
    performed_by: Optional[str] = None
    performed_at: datetime
    details: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

# Status Check Schemas
class ForecastRunStatusResponse(BaseModel):
    forecast_run_id: UUID
    run_status: str
    run_progress: int
    total_records: int
    processed_records: int
    failed_records: int
    error_message: Optional[str] = None
    updated_at: datetime
