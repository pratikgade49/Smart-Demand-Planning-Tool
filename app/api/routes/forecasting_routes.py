"""
Forecasting API Routes.
Endpoints for forecast run management and execution.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
import pandas as pd
from psycopg2.extras import Json
from app.schemas.forecasting import (
    ForecastRunCreate,
    ForecastVersionCreate,
    ForecastVersionUpdate,
    ExternalFactorCreate,
    ExternalFactorUpdate
)
from app.core.database import get_db_manager
from app.core.forecast_execution_service import ForecastExecutionService
from app.core.forecasting_service import ForecastingService
from app.core.forecast_version_service import ForecastVersionService
from app.core.external_factors_service import ExternalFactorsService
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException, ValidationException
from app.core.algorithm_parameters import AlgorithmParametersService
from app.api.dependencies import get_current_tenant
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecasting", tags=["Forecasting"])


def _get_algorithm_name_by_id(algorithm_id: int) -> str:
    """Map algorithm_id to algorithm name for execution."""
    algorithm_map = {
        1: "arima",
        2: "linear_regression",
        3: "polynomial_regression",
        4: "exponential_smoothing",
        5: "exponential_smoothing",
        6: "holt_winters",
        7: "prophet",
        8: "lstm",
        9: "xgboost",
        10: "svr",
        11: "knn",
        12: "gaussian_process",
        13: "mlp_neural_network",
        14: "simple_moving_average",
        15: "seasonal_decomposition",
        16: "moving_average",
        17: "sarima",
        999: "best_fit"
    }
    
    if algorithm_id not in algorithm_map:
        raise ValueError(f"Unknown algorithm_id: {algorithm_id}. Valid IDs are: {', '.join(map(str, algorithm_map.keys()))}")
    
    return algorithm_map[algorithm_id]


# ============================================================================
# FORECAST VERSION MANAGEMENT
# ============================================================================

class AlgorithmConfig(BaseModel):
    algorithm_id: int
    execution_order: int
    custom_parameters: Optional[Dict[str, Any]] = None


class DirectForecastExecutionRequest(BaseModel):
    """Request for direct forecast execution with automatic entity handling."""
    version_id: str
    forecast_filters: Dict[str, Any]
    forecast_start: str = Field(..., description="Start date in YYYY-MM-DD format")
    forecast_end: str = Field(..., description="End date in YYYY-MM-DD format")
    algorithm_id: Optional[int] = Field(default=999, description="Single algorithm ID")
    custom_parameters: Optional[Dict[str, Any]] = None
    algorithms: Optional[List[AlgorithmConfig]] = None
    run_percentage_frequency: Optional[int] = None


@router.post("/execute-forecast", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def execute_forecast_directly(
    request: Request,
    request_data: DirectForecastExecutionRequest,
    tenant_data: Dict = Depends(get_current_tenant)
):

    try:
        from app.core.forecasting_service import ForecastingService
        from app.core.forecast_execution_service import ForecastExecutionService
        from datetime import datetime
        import uuid
        
        logger.info(f"Direct forecast execution requested for tenant {tenant_data['tenant_id']}")
        logger.info(f"Request: {request.method} {request.url.path}")

        # Extract configuration
        aggregation_level = request_data.forecast_filters.get('aggregation_level', 'product')
        interval = request_data.forecast_filters.get('interval', 'MONTHLY')
        selected_factors = request_data.forecast_filters.get('selected_external_factors')
        
        # ✅ Step 1: Detect entity combinations
        agg_columns = ForecastingService._get_aggregation_columns(
            tenant_data["tenant_id"],
            tenant_data["database_name"],
            aggregation_level
        )
        
        entity_combinations = []
        multi_entity = False
        
        # Build entity combinations from filters
        for col in agg_columns:
            if col in request_data.forecast_filters:
                filter_value = request_data.forecast_filters[col]
                
                # Handle list of values
                if isinstance(filter_value, list) and len(filter_value) > 1:
                    multi_entity = True
                    if not entity_combinations:
                        entity_combinations = [{col: val} for val in filter_value]
                    else:
                        # Cross product for multi-dimensional
                        new_combinations = []
                        for combo in entity_combinations:
                            for val in filter_value:
                                new_combo = combo.copy()
                                new_combo[col] = val
                                new_combinations.append(new_combo)
                        entity_combinations = new_combinations
                
                # Handle single value in list
                elif isinstance(filter_value, list) and len(filter_value) == 1:
                    if not entity_combinations:
                        entity_combinations = [{col: filter_value[0]}]
                    else:
                        for combo in entity_combinations:
                            combo[col] = filter_value[0]
                
                # Handle single value
                else:
                    if not entity_combinations:
                        entity_combinations = [{col: filter_value}]
                    else:
                        for combo in entity_combinations:
                            combo[col] = filter_value
        
        # Default to empty filter if no entities specified
        if not entity_combinations:
            entity_combinations = [{}]
        
        logger.info(f"Detected {len(entity_combinations)} entity combination(s) to forecast")
        
        # ✅ Step 2: Execute forecast for each entity
        forecast_runs = []
        total_records = 0
        successful_runs = 0
        failed_runs = 0
        
        for entity_filter in entity_combinations:
            entity_name = '-'.join([f"{k}={v}" for k, v in entity_filter.items()]) if entity_filter else "all"
            
            try:
                logger.info(f"Creating forecast run for entity: {entity_name}")
                
                # Create specific filter for this entity
                entity_specific_filters = request_data.forecast_filters.copy()
                entity_specific_filters.update(entity_filter)

                # ✅ Step 3: Get historical data for this entity (shared across all algorithms)
                logger.info(f"Executing forecast for {entity_name}")
                
                # Get historical data for this specific entity
                historical_data, date_field_name = ForecastingService.prepare_aggregated_data(
                    tenant_id=tenant_data["tenant_id"],
                    database_name=tenant_data["database_name"],
                    aggregation_level=aggregation_level,
                    interval=interval,
                    filters=entity_specific_filters
                )

                logger.info(f"Entity {entity_name}: {len(historical_data)} historical records")
                
                # Load external factors
                external_factors_df = ForecastExecutionService._prepare_external_factors(
                    tenant_id=tenant_data["tenant_id"],
                    database_name=tenant_data["database_name"],
                    selected_factors=selected_factors
                )
                
                # Merge external factors if available
                if not external_factors_df.empty:
                    historical_data[date_field_name] = pd.to_datetime(historical_data[date_field_name], errors='coerce')
                    external_factors_df['date'] = pd.to_datetime(external_factors_df['date'], errors='coerce')

                    historical_data = historical_data.merge(
                        external_factors_df,
                        left_on=date_field_name,
                        right_on='date',
                        how='left'
                    )
                    if 'date' in historical_data.columns:
                        historical_data = historical_data.drop(columns=['date'])
                
                # Calculate forecast periods
                forecast_start_date = datetime.fromisoformat(request_data.forecast_start).date()
                forecast_end_date = datetime.fromisoformat(request_data.forecast_end).date()
                periods = ForecastExecutionService._calculate_periods(
                    forecast_start_date, forecast_end_date, interval
                )
                forecast_dates = ForecastExecutionService._generate_forecast_dates(
                    forecast_start_date, periods, interval
                )
                
                # Check if multiple algorithms are specified
                algorithms_to_execute = []
                if request_data.algorithms and len(request_data.algorithms) > 0:
                    algorithms_to_execute = request_data.algorithms
                elif request_data.algorithm_id is not None:
                    algorithms_to_execute = [AlgorithmConfig(algorithm_id=request_data.algorithm_id, execution_order=1, custom_parameters=request_data.custom_parameters)]
                else:
                    algorithms_to_execute = [AlgorithmConfig(algorithm_id=999, execution_order=1)]
                
                logger.info(f"Executing {len(algorithms_to_execute)} algorithm(s) for entity {entity_name}")
                
                # Execute all specified algorithms and collect results
                db_manager = get_db_manager()
                entity_records_count = 0
                process_log = []
                
                for algo_config in sorted(algorithms_to_execute, key=lambda x: x.execution_order):
                    forecast_run_id = str(uuid.uuid4())
                    algo_mapping_id = str(uuid.uuid4())
                    
                    try:
                        # Create separate forecast run for each entity-algorithm combination
                        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                            cursor = conn.cursor()
                            try:
                                cursor.execute("""
                                    INSERT INTO forecast_runs
                                    (forecast_run_id, tenant_id, version_id, forecast_filters,
                                     forecast_start, forecast_end, run_status, created_by)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                """, (
                                    forecast_run_id,
                                    tenant_data["tenant_id"],
                                    request_data.version_id,
                                    Json(entity_specific_filters),
                                    datetime.fromisoformat(request_data.forecast_start).date(),
                                    datetime.fromisoformat(request_data.forecast_end).date(),
                                    "In-Progress",
                                    tenant_data["email"]
                                ))
                                conn.commit()
                            finally:
                                cursor.close()
                        
                        # Insert algorithm mapping
                        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                            cursor = conn.cursor()
                            try:
                                cursor.execute("""
                                    SELECT algorithm_name FROM algorithms WHERE algorithm_id = %s
                                """, (algo_config.algorithm_id,))
                                result = cursor.fetchone()
                                algo_name = result[0] if result else _get_algorithm_name_by_id(algo_config.algorithm_id)

                                cursor.execute("""
                                    INSERT INTO forecast_algorithms_mapping
                                    (mapping_id, tenant_id, forecast_run_id, algorithm_id,
                                     algorithm_name, custom_parameters, execution_order, created_by)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                                """, (
                                    algo_mapping_id,
                                    tenant_data["tenant_id"],
                                    forecast_run_id,
                                    algo_config.algorithm_id,
                                    algo_name,
                                    Json(algo_config.custom_parameters or {}),
                                    algo_config.execution_order,
                                    tenant_data["email"]
                                ))
                                conn.commit()
                            finally:
                                cursor.close()
                        
                        # Execute the algorithm
                        if algo_config.algorithm_id == 999:
                            forecast_result = ForecastExecutionService.generate_forecast(
                                historical_data=historical_data,
                                config={'interval': interval, 'periods': periods},
                                process_log=process_log
                            )
                            algo_name_for_result = "best_fit"
                            algo_accuracy = forecast_result.get('accuracy')
                            algo_forecast = forecast_result['forecast']
                            algo_metrics = {
                                'mae': forecast_result.get('mae'),
                                'rmse': forecast_result.get('rmse'),
                                'mape': forecast_result.get('mape'),
                                'selected_algorithm': forecast_result['selected_algorithm'],
                                'process_log': forecast_result.get('process_log', []),
                                'algorithms_evaluated': len(forecast_result.get('all_algorithms', []))
                            }
                        else:
                            algorithm_name_result = ForecastExecutionService._run_algorithm_safe(
                                algorithm_name=_get_algorithm_name_by_id(algo_config.algorithm_id),
                                data=historical_data.copy(),
                                periods=periods,
                                target_column='total_quantity'
                            )
                            
                            if algorithm_name_result['accuracy'] == 0:
                                logger.warning(f"Algorithm {algo_config.algorithm_id} failed, skipping")
                                continue
                            
                            algo_name_for_result = algorithm_name_result['algorithm']
                            algo_accuracy = algorithm_name_result.get('accuracy')
                            algo_forecast = algorithm_name_result['forecast']
                            algo_metrics = {
                                'mae': algorithm_name_result.get('mae'),
                                'rmse': algorithm_name_result.get('rmse'),
                                'mape': algorithm_name_result.get('mape')
                            }
                        
                        # Store results for this algorithm
                        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                            cursor = conn.cursor()
                            try:
                                cursor.execute("""
                                    UPDATE forecast_algorithms_mapping
                                    SET custom_parameters = %s,
                                        execution_status = 'Completed',
                                        completed_at = %s
                                    WHERE mapping_id = %s
                                """, (
                                    Json({
                                        'algorithm_name': algo_name_for_result,
                                        'entity_filter': entity_filter,
                                        **algo_metrics
                                    }),
                                    datetime.utcnow(),
                                    algo_mapping_id
                                ))
                                
                                for forecast_date, forecast_value in zip(forecast_dates, algo_forecast):
                                    result_id = str(uuid.uuid4())
                                    forecast_value_float = min(999999.9999, max(0, float(forecast_value)))
                                    forecast_value_float = round(forecast_value_float, 4)
                                    accuracy_metric = round(algo_accuracy, 2) if algo_accuracy else None
                                    
                                    cursor.execute("""
                                        INSERT INTO forecast_results
                                        (result_id, tenant_id, forecast_run_id, version_id, mapping_id,
                                         algorithm_id, forecast_date, forecast_quantity, accuracy_metric,
                                         metric_type, metadata, created_by)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    """, (
                                        result_id,
                                        tenant_data["tenant_id"],
                                        forecast_run_id,
                                        request_data.version_id,
                                        algo_mapping_id,
                                        algo_config.algorithm_id,
                                        forecast_date,
                                        forecast_value_float,
                                        accuracy_metric,
                                        'Accuracy',
                                        Json({
                                            'algorithm': algo_name_for_result,
                                            **algo_metrics,
                                            'entity_filter': entity_filter
                                        }),
                                        tenant_data["email"]
                                    ))

                                conn.commit()
                            finally:
                                cursor.close()

                        algo_records_count = len(forecast_dates)
                        entity_records_count += algo_records_count
                        total_records += algo_records_count

                        logger.info(f"Entity {entity_name}, Algorithm {algo_name_for_result}: {algo_records_count} forecast records, Accuracy: {algo_accuracy:.2f}%")

                        # Mark forecast run as completed for this entity-algorithm combination
                        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                            cursor = conn.cursor()
                            try:
                                cursor.execute("""
                                    UPDATE forecast_runs
                                    SET run_status = 'Completed',
                                        run_progress = 100,
                                        total_records = %s,
                                        processed_records = %s,
                                        completed_at = %s,
                                        updated_at = %s
                                    WHERE forecast_run_id = %s
                                """, (
                                    algo_records_count,
                                    algo_records_count,
                                    datetime.utcnow(),
                                    datetime.utcnow(),
                                    forecast_run_id
                                ))
                                conn.commit()
                            finally:
                                cursor.close()

                        successful_runs += 1

                        forecast_runs.append({
                            "entity": entity_name,
                            "entity_filter": entity_filter,
                            "algorithm_id": algo_config.algorithm_id,
                            "algorithm_name": algo_name_for_result,
                            "forecast_run_id": forecast_run_id,
                            "status": "Completed",
                            "records": algo_records_count,
                            "accuracy": round(algo_accuracy, 2) if algo_accuracy else None
                        })
                    except Exception as e:
                        logger.error(f"Failed to execute algorithm {algo_config.algorithm_id} for entity {entity_name}: {str(e)}", exc_info=True)
                        failed_runs += 1
                        
                        if 'forecast_run_id' in locals():
                            with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                                cursor = conn.cursor()
                                try:
                                    cursor.execute("""
                                        UPDATE forecast_runs
                                        SET run_status = 'Failed',
                                            error_message = %s,
                                            updated_at = %s
                                        WHERE forecast_run_id = %s
                                    """, (str(e), datetime.utcnow(), forecast_run_id))
                                    conn.commit()
                                finally:
                                    cursor.close()
                        
                        forecast_runs.append({
                            "entity": entity_name,
                            "entity_filter": entity_filter,
                            "algorithm_id": algo_config.algorithm_id,
                            "forecast_run_id": forecast_run_id if 'forecast_run_id' in locals() else None,
                            "status": "Failed",
                            "error": str(e),
                            "records": 0,
                            "accuracy": None
                        })

                logger.info(f"Entity {entity_name} completed with {len(algorithms_to_execute)} algorithm(s): {entity_records_count} forecast records")

            except Exception as e:
                logger.error(f"Failed to forecast entity {entity_name}: {str(e)}", exc_info=True)
                failed_runs += 1

                forecast_runs.append({
                    "entity": entity_name,
                    "entity_filter": entity_filter,
                    "forecast_run_id": forecast_run_id if 'forecast_run_id' in locals() else None,
                    "status": "Failed",
                    "error": str(e),
                    "records": 0
                })

        # ✅ Step 4: Return comprehensive results
        result = {
            "total_entities": len(entity_combinations),
            "forecast_runs": forecast_runs,
            "execution_summary": {
                "total_runs": len(forecast_runs),
                "successful_runs": successful_runs,
                "failed_runs": failed_runs,
                "total_records": total_records
            },
            "filters_used": {
                "aggregation_level": aggregation_level,
                "interval": interval,
                "forecast_period": {
                    "start": request_data.forecast_start,
                    "end": request_data.forecast_end
                }
            }
        }
        
        logger.info(
            f"Direct forecast execution completed: "
            f"{successful_runs}/{len(forecast_runs)} successful, "
            f"{total_records} total records"
        )
        
        return ResponseHandler.success(data=result, status_code=201)
    
    except ValidationException as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except AppException as e:
        logger.error(f"App error: {str(e)}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in direct forecast execution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/versions", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_forecast_version(
    request: ForecastVersionCreate,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Create a new forecast version.
    
    - **version_name**: Unique name for the version
    - **version_type**: Type (Baseline, Simulation, Final)
    - **is_active**: Whether this version is active
    """
    try:
        logger.info(f"Creating forecast version '{request.version_name}' of type '{request.version_type}' for tenant {tenant_data['tenant_id']}")
        result = ForecastVersionService.create_version(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            request=request,
            user_email=tenant_data["email"]
        )
        logger.info(f"Successfully created forecast version with ID: {result['version_id']}")
        return ResponseHandler.success(data=result, status_code=201)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating version: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/versions", response_model=Dict[str, Any])
async def list_forecast_versions(
    tenant_data: Dict = Depends(get_current_tenant),
    version_type: Optional[str] = Query(None, pattern="^(Baseline|Simulation|Final)$"),
    is_active: Optional[bool] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100)
):
    """
    List all forecast versions for the tenant.
    
    - **version_type**: Filter by version type
    - **is_active**: Filter by active status
    """
    try:
        versions, total_count = ForecastVersionService.list_versions(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            version_type=version_type,
            is_active=is_active,
            page=page,
            page_size=page_size
        )
        
        logger.info(f"Retrieved {total_count} forecast versions for tenant {tenant_data['tenant_id']}")
        return ResponseHandler.list_response(
            data=versions,
            page=page,
            page_size=page_size,
            total_count=total_count
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing versions: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/versions/{version_id}", response_model=Dict[str, Any])
async def get_forecast_version(
    version_id: str,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """Get forecast version details by ID."""
    try:
        result = ForecastVersionService.get_version(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            version_id=version_id
        )
        logger.info(f"Retrieved forecast version {version_id} for tenant {tenant_data['tenant_id']}")
        return ResponseHandler.success(data=result)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting version: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/versions/{version_id}", response_model=Dict[str, Any])
async def update_forecast_version(
    version_id: str,
    request: ForecastVersionUpdate,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Update forecast version.
    
    - Can update version_name and is_active status
    - Only one version of each type can be active at a time
    """
    try:
        logger.info(f"Updating forecast version {version_id} for tenant {tenant_data['tenant_id']}")
        result = ForecastVersionService.update_version(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            version_id=version_id,
            request=request,
            user_email=tenant_data["email"]
        )
        logger.info(f"Successfully updated forecast version {version_id}")
        return ResponseHandler.success(data=result)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error updating version: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# EXTERNAL FACTORS MANAGEMENT
# ============================================================================

@router.post("/external-factors", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_external_factor(
    request: ExternalFactorCreate,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Create a new external factor record.
    
    External factors can include:
    - Weather data (temperature, precipitation)
    - Economic indicators (GDP, inflation)
    - Marketing campaigns
    - Holidays and events
    - Any other external variable that may impact demand
    """
    try:
        logger.info(f"Creating external factor '{request.factor_name}' for tenant {tenant_data['tenant_id']}")
        result = ExternalFactorsService.create_factor(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            request=request,
            user_email=tenant_data["email"]
        )
        logger.info(f"Successfully created external factor with ID: {result['factor_id']}")
        return ResponseHandler.success(data=result, status_code=201)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating external factor: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/external-factors", response_model=Dict[str, Any])
async def list_external_factors(
    tenant_data: Dict = Depends(get_current_tenant),
    factor_name: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100)
):
    """
    List external factors with optional filters.
    
    - **factor_name**: Filter by factor name
    - **date_from**: Start date for filtering
    - **date_to**: End date for filtering
    """
    try:
        factors, total_count = ExternalFactorsService.list_factors(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            factor_name=factor_name,
            date_from=date_from,
            date_to=date_to,
            page=page,
            page_size=page_size
        )
        
        logger.info(f"Retrieved {total_count} external factors for tenant {tenant_data['tenant_id']}")
        return ResponseHandler.list_response(
            data=factors,
            page=page,
            page_size=page_size,
            total_count=total_count
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing external factors: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/aggregation-levels", response_model=Dict[str, Any])
async def get_aggregation_levels(
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get available aggregation levels based on tenant's master data structure.
    
    Returns all fields from the finalized field catalogue that can be used
    for aggregation, including suggested single and multi-dimensional combinations.
    
    **Usage**:
    - Use field names from the response to build custom aggregation levels
    - Single dimension: Use any single field name (e.g., "product", "location")
    - Multi dimension: Combine fields with hyphen (e.g., "product-location", "customer-location")
    """
    try:
        result = ForecastingService.get_available_aggregation_levels(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"]
        )
        logger.info(f"Retrieved aggregation levels for tenant {tenant_data['tenant_id']}: {len(result.get('available_fields', []))} fields")
        return ResponseHandler.success(data=result)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting aggregation levels: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")



@router.get("/runs", response_model=Dict[str, Any])
async def list_forecast_runs(
    tenant_data: Dict = Depends(get_current_tenant),
    version_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None, pattern="^(Pending|In-Progress|Completed|Completed with Errors|Failed|Cancelled)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100)
):
    """
    List forecast runs with optional filters.
    
    - **version_id**: Filter by version
    - **status**: Filter by run status
    """
    try:
        runs, total_count = ForecastingService.list_forecast_runs(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            version_id=version_id,
            status=status,
            page=page,
            page_size=page_size
        )
        
        logger.info(f"Retrieved {total_count} forecast runs for tenant {tenant_data['tenant_id']}")
        return ResponseHandler.list_response(
            data=runs,
            page=page,
            page_size=page_size,
            total_count=total_count
        )
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing forecast runs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/runs/{forecast_run_id}", response_model=Dict[str, Any])
async def get_forecast_run(
    forecast_run_id: str,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """Get detailed information about a forecast run."""
    try:
        result = ForecastingService.get_forecast_run(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            forecast_run_id=forecast_run_id
        )
        logger.info(f"Retrieved forecast run {forecast_run_id} for tenant {tenant_data['tenant_id']}")
        return ResponseHandler.success(data=result)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting forecast run: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/runs/{forecast_run_id}/status", response_model=Dict[str, Any])
async def get_forecast_run_status(
    forecast_run_id: str,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get current status of a forecast run.
    
    Returns progress information and statistics.
    """
    try:
        result = ForecastingService.get_forecast_run(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            forecast_run_id=forecast_run_id
        )

        # Return only status-related fields
        status_data = {
            "forecast_run_id": result["forecast_run_id"],
            "run_status": result["run_status"],
            "run_progress": result["run_progress"],
            "total_records": result["total_records"],
            "processed_records": result["processed_records"],
            "failed_records": result["failed_records"],
            "error_message": result["error_message"],
            "updated_at": result["updated_at"]
        }

        logger.info(f"Retrieved status for forecast run {forecast_run_id}: {result['run_status']}")
        return ResponseHandler.success(data=status_data)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting run status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@router.get("/runs/{forecast_run_id}/results", response_model=Dict[str, Any])
async def get_forecast_results(
    forecast_run_id: str,
    tenant_data: Dict = Depends(get_current_tenant),
    algorithm_id: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000)
):
    """
    Get forecast results for a specific run.
    
    - **algorithm_id**: Optional filter by specific algorithm
    - **page**: Page number
    - **page_size**: Results per page
    """
    try:
        from app.core.database import get_db_manager
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                # Build WHERE clause
                where_clauses = ["tenant_id = %s", "forecast_run_id = %s"]
                params = [tenant_data["tenant_id"], forecast_run_id]
                
                if algorithm_id:
                    where_clauses.append("algorithm_id = %s")
                    params.append(algorithm_id)
                
                where_sql = " AND ".join(where_clauses)
                
                # Get total count
                cursor.execute(
                    f"SELECT COUNT(*) FROM forecast_results WHERE {where_sql}",
                    params
                )
                total_count = cursor.fetchone()[0]
                
                # Get paginated results
                offset = (page - 1) * page_size
                cursor.execute(f"""
                    SELECT result_id, algorithm_id, forecast_date, forecast_quantity,
                           confidence_interval_lower, confidence_interval_upper,
                           confidence_level, accuracy_metric, metric_type, metadata,
                           created_at, created_by
                    FROM forecast_results
                    WHERE {where_sql}
                    ORDER BY forecast_date, algorithm_id
                    LIMIT %s OFFSET %s
                """, params + [page_size, offset])
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "result_id": str(row[0]),
                        "algorithm_id": row[1],
                        "forecast_date": row[2].isoformat() if row[2] else None,
                        "forecast_quantity": float(row[3]),
                        "confidence_interval_lower": float(row[4]) if row[4] else None,
                        "confidence_interval_upper": float(row[5]) if row[5] else None,
                        "confidence_level": row[6],
                        "accuracy_metric": float(row[7]) if row[7] else None,
                        "metric_type": row[8],
                        "metadata": row[9],
                        "created_at": row[10].isoformat() if row[10] else None,
                        "created_by": row[11]
                    })
                
                logger.info(f"Retrieved {total_count} forecast results for run {forecast_run_id}")
                return ResponseHandler.list_response(
                    data=results,
                    page=page,
                    page_size=page_size,
                    total_count=total_count
                )
                
            finally:
                cursor.close()
                
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error getting results: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/results/compare", response_model=Dict[str, Any])
async def compare_forecast_results(
    tenant_data: Dict = Depends(get_current_tenant),
    forecast_run_id: str = Query(..., description="Forecast run ID"),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    """
    Compare results across all algorithms for a forecast run.
    
    Returns aggregated comparison data showing each algorithm's performance.
    """
    try:
        from app.core.database import get_db_manager
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                # Build WHERE clause
                where_clauses = ["fr.tenant_id = %s", "fr.forecast_run_id = %s"]
                params = [tenant_data["tenant_id"], forecast_run_id]
                
                if date_from:
                    where_clauses.append("fr.forecast_date >= %s")
                    params.append(date_from)
                
                if date_to:
                    where_clauses.append("fr.forecast_date <= %s")
                    params.append(date_to)
                
                where_sql = " AND ".join(where_clauses)
                
                # Get comparison data
                cursor.execute(f"""
                    SELECT 
                        a.algorithm_id,
                        a.algorithm_name,
                        COUNT(fr.result_id) as result_count,
                        AVG(fr.forecast_quantity) as avg_forecast,
                        MIN(fr.forecast_quantity) as min_forecast,
                        MAX(fr.forecast_quantity) as max_forecast,
                        AVG(fr.accuracy_metric) as avg_accuracy
                    FROM forecast_results fr
                    JOIN algorithms a ON fr.algorithm_id = a.algorithm_id
                    WHERE {where_sql}
                    GROUP BY a.algorithm_id, a.algorithm_name
                    ORDER BY a.algorithm_name
                """, params)
                
                comparison = []
                for row in cursor.fetchall():
                    comparison.append({
                        "algorithm_id": row[0],
                        "algorithm_name": row[1],
                        "result_count": row[2],
                        "avg_forecast": round(float(row[3]), 2) if row[3] else None,
                        "min_forecast": round(float(row[4]), 2) if row[4] else None,
                        "max_forecast": round(float(row[5]), 2) if row[5] else None,
                        "avg_accuracy_metric": round(float(row[6]), 2) if row[6] else None
                    })
                
                logger.info(f"Generated algorithm comparison for forecast run {forecast_run_id}: {len(comparison)} algorithms")
                return ResponseHandler.success(data={
                    "forecast_run_id": forecast_run_id,
                    "algorithm_comparison": comparison
                })
                
            finally:
                cursor.close()
                
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error comparing results: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/results/export/{forecast_run_id}", response_model=Dict[str, Any])
async def export_forecast_results(
    forecast_run_id: str,
    tenant_data: Dict = Depends(get_current_tenant),
    algorithm_id: Optional[int] = Query(None),
    format: str = Query("json", pattern="^(json|csv)$")
):
    """
    Export forecast results in JSON or CSV format.
    
    - **format**: Output format (json or csv)
    - **algorithm_id**: Optional filter by algorithm
    """
    try:
        from app.core.database import get_db_manager
        import json
        
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                where_clauses = ["fr.tenant_id = %s", "fr.forecast_run_id = %s"]
                params = [tenant_data["tenant_id"], forecast_run_id]
                
                if algorithm_id:
                    where_clauses.append("fr.algorithm_id = %s")
                    params.append(algorithm_id)
                
                where_sql = " AND ".join(where_clauses)
                
                cursor.execute(f"""
                    SELECT 
                        fr.forecast_date,
                        a.algorithm_name,
                        fr.forecast_quantity,
                        fr.confidence_interval_lower,
                        fr.confidence_interval_upper,
                        fr.accuracy_metric,
                        fr.metric_type
                    FROM forecast_results fr
                    JOIN algorithms a ON fr.algorithm_id = a.algorithm_id
                    WHERE {where_sql}
                    ORDER BY fr.forecast_date, a.algorithm_name
                """, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "forecast_date": row[0].isoformat() if row[0] else None,
                        "algorithm_name": row[1],
                        "forecast_quantity": float(row[2]),
                        "confidence_interval_lower": float(row[3]) if row[3] else None,
                        "confidence_interval_upper": float(row[4]) if row[4] else None,
                        "accuracy_metric": float(row[5]) if row[5] else None,
                        "metric_type": row[6]
                    })
                
                if format == "csv":
                    # Convert to CSV format
                    import io
                    import csv
                    
                    output = io.StringIO()
                    if results:
                        writer = csv.DictWriter(output, fieldnames=results[0].keys())
                        writer.writeheader()
                        writer.writerows(results)
                    
                    csv_data = output.getvalue()
                    
                    return ResponseHandler.success(data={
                        "format": "csv",
                        "content": csv_data,
                        "record_count": len(results)
                    })
                else:
                    logger.info(f"Exported {len(results)} forecast results in JSON format for run {forecast_run_id}")
                    return ResponseHandler.success(data={
                        "format": "json",
                        "results": results,
                        "record_count": len(results)
                    })
                
            finally:
                cursor.close()
                
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error exporting results: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/algorithms", response_model=Dict[str, Any])
async def list_algorithms(
    tenant_data: Dict = Depends(get_current_tenant),
    algorithm_type: Optional[str] = Query(None, pattern="^(ML|Statistic|Hybrid)$")
):
    """
    List available forecasting algorithms.
    
    - **algorithm_type**: Filter by type (ML, Statistic, Hybrid)
    """
    try:
        from app.core.database import get_db_manager
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                where_clause = ""
                params = []
                
                if algorithm_type:
                    where_clause = "WHERE algorithm_type = %s"
                    params.append(algorithm_type)
                
                cursor.execute(f"""
                    SELECT algorithm_id, algorithm_name, default_parameters,
                           algorithm_type, description, created_at, updated_at
                    FROM algorithms
                    {where_clause}
                    ORDER BY algorithm_name
                """, params)
                
                algorithms = []
                for row in cursor.fetchall():
                    algorithms.append({
                        "algorithm_id": row[0],
                        "algorithm_name": row[1],
                        "default_parameters": row[2],
                        "algorithm_type": row[3],
                        "description": row[4],
                        "created_at": row[5].isoformat() if row[5] else None,
                        "updated_at": row[6].isoformat() if row[6] else None
                    })
                
                logger.info(f"Retrieved {len(algorithms)} algorithms for tenant {tenant_data['tenant_id']}")
                return ResponseHandler.success(data=algorithms)
                
            finally:
                cursor.close()
                
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

