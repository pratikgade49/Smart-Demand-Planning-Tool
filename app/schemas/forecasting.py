from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

# Parameter Schema Models
class ParameterDefinition(BaseModel):
    name: str
    type: str = Field(..., pattern="^(int|float|string|list|bool)$")
    description: str
    required: bool = False
    default_value: Any = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    list_item_type: Optional[str] = None  # For list parameters

class AlgorithmParameterSchema(BaseModel):
    algorithm_id: int
    algorithm_name: str
    parameters: List[ParameterDefinition]
    description: Optional[str] = None

class ParameterValidationResult(BaseModel):
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []

# Forecasting Schema Models
class ForecastVersionCreate(BaseModel):
    version_name: str = Field(..., min_length=1, max_length=100)
    version_type: str = Field(..., pattern="^(Baseline|Simulation|Final)$")
    is_active: bool = Field(False)

class ForecastVersionUpdate(BaseModel):
    version_name: Optional[str] = Field(None, min_length=1, max_length=100)
    is_active: Optional[bool] = None

class ExternalFactorCreate(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    factor_name: str = Field(..., min_length=1, max_length=100)
    factor_value: float
    unit: Optional[str] = Field(None, max_length=50)
    source: Optional[str] = Field(None, max_length=255)

class ExternalFactorUpdate(BaseModel):
    factor_value: Optional[float] = None
    unit: Optional[str] = Field(None, max_length=50)
    source: Optional[str] = Field(None, max_length=255)

class AlgorithmMapping(BaseModel):
    algorithm_id: int
    execution_order: int
    custom_parameters: Optional[Dict[str, Any]] = None

class ForecastRunCreate(BaseModel):
    version_id: str
    forecast_filters: Optional[Dict[str, Any]] = None
    forecast_start: str = Field(..., description="Start date in YYYY-MM-DD format")
    forecast_end: str = Field(..., description="End date in YYYY-MM-DD format")
    history_start: Optional[str] = Field(None, description="Start date of historic data to use (YYYY-MM-DD)")
    history_end: Optional[str] = Field(None, description="End date of historic data to use (YYYY-MM-DD)")

    algorithms: List[AlgorithmMapping] = Field(..., min_items=1)


class ForecastRunResponse(BaseModel):
    forecast_run_id: str
    tenant_id: str
    version_id: str
    forecast_filters: Optional[Dict[str, Any]] = None
    forecast_start: str
    forecast_end: str
    history_start: Optional[str] = None
    history_end: Optional[str] = None
    run_status: str
    run_progress: int
    total_records: Optional[int] = None
    processed_records: Optional[int] = None
    failed_records: Optional[int] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    created_by: str
    algorithms: List[Dict[str, Any]] = []

class AlgorithmMappingResponse(BaseModel):
    mapping_id: str
    forecast_run_id: str
    algorithm_id: int
    algorithm_name: str
    custom_parameters: Optional[Dict[str, Any]] = None
    execution_order: int
    execution_status: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: Optional[str] = None

class ForecastResultResponse(BaseModel):
    result_id: str
    forecast_run_id: str
    version_id: str
    mapping_id: str
    algorithm_id: int
    forecast_date: str
    forecast_quantity: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    confidence_level: Optional[float] = None
    accuracy_metric: Optional[float] = None
    metric_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    created_by: str
