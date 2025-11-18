"""
Forecasting API Routes.
Endpoints for forecast run management and execution.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from psycopg2.extras import Json
from app.schemas.forecasting import (
    ForecastRunCreate,
    ForecastVersionCreate,
    ForecastVersionUpdate,
    ExternalFactorCreate,
    ExternalFactorUpdate
)

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


# ============================================================================
# FORECAST VERSION MANAGEMENT
# ============================================================================

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


# ============================================================================
# FORECAST RUN MANAGEMENT
# ============================================================================

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


@router.post("/runs", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_forecast_run(
    request: ForecastRunCreate,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Create a new forecast run.
    
    **Forecast Filters** (optional):
    ```json
    {
      "aggregation_level": "product-location",
      "interval": "MONTHLY",
      "product": "specific_product_value",
      "location": "specific_location_value"
    }
    ```
    
    **Aggregation Levels** (Dynamic):
    - Use GET /aggregation-levels to see available fields
    - Single dimension: Any field name from your master data
    - Multi dimension: Combine fields with hyphen (e.g., "product-location")
    - You can create ANY combination of fields that exist in your master data
    
    **Intervals**:
    - WEEKLY
    - MONTHLY
    - QUARTERLY
    - YEARLY
    """
    try:
        agg_level = request.forecast_filters.get('aggregation_level', 'product') if request.forecast_filters else 'product'
        interval = request.forecast_filters.get('interval', 'MONTHLY') if request.forecast_filters else 'MONTHLY'
        logger.info(f"Creating forecast run for tenant {tenant_data['tenant_id']} with aggregation level '{agg_level}' and interval '{interval}'")
        result = ForecastingService.create_forecast_run(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            request=request,
            user_email=tenant_data["email"]
        )
        logger.info(f"Successfully created forecast run with ID: {result['forecast_run_id']}")
        return ResponseHandler.success(data=result, status_code=201)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error creating forecast run: {str(e)}")
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
    
@router.post("/runs/{forecast_run_id}/execute", response_model=Dict[str, Any])
async def execute_forecast_run(
    forecast_run_id: str,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Execute a forecast run.
    
    This will:
    1. Fetch historical data based on filters
    2. Execute all mapped algorithms in order
    3. Store forecast results
    4. Update run status
    
    **Note**: This is an asynchronous operation that may take time for large datasets.
    """
    try:
        logger.info(f"Starting forecast execution for run {forecast_run_id} by user {tenant_data['email']}")
        result = ForecastExecutionService.execute_forecast_run(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            forecast_run_id=forecast_run_id,
            user_email=tenant_data["email"]
        )
        logger.info(f"Successfully executed forecast run {forecast_run_id}: {result.get('status', 'Unknown')}")
        return ResponseHandler.success(data=result)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error executing forecast: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/runs/{forecast_run_id}/execute-best-fit", response_model=Dict[str, Any])
async def execute_forecast_best_fit(
    forecast_run_id: str,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Execute a forecast run using Advanced AI/ML auto model (Best Fit algorithm selection).
    
    This will:
    1. Fetch historical data based on filters
    2. Run all available algorithms in parallel
    3. Select the best performing algorithm
    4. Create ensemble from top 3 algorithms
    5. Store forecast results
    6. Update run status
    
    **When to use**: Select this when "Advanced AI/ML auto model" is chosen in the UI.
    
    **Note**: This is an asynchronous operation that may take time for large datasets.
    Automatically selects the best algorithm without manual configuration.
    """
    try:
        from app.core.database import get_db_manager
        
        db_manager = get_db_manager()
        
        # Get forecast run details
        result = ForecastingService.get_forecast_run(
            tenant_data["tenant_id"],
            tenant_data["database_name"],
            forecast_run_id
        )
        
        # Validate run status
        if result['run_status'] not in ['Pending', 'Failed']:
            raise ValidationException(
                f"Cannot execute forecast run with status: {result['run_status']}"
            )
        
        # Update status to In-Progress
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    UPDATE forecast_runs
                    SET run_status = 'In-Progress',
                        started_at = %s,
                        updated_at = %s,
                        updated_by = %s
                    WHERE forecast_run_id = %s
                """, (datetime.utcnow(), datetime.utcnow(), tenant_data["email"], forecast_run_id))
                conn.commit()
            finally:
                cursor.close()
        
        logger.info(f"Starting best-fit forecast execution for run: {forecast_run_id}")

        # Prepare historical data
        filters = result.get('forecast_filters', {})
        aggregation_level = filters.get('aggregation_level', 'product')
        interval = filters.get('interval', 'MONTHLY')

        logger.info(f"Preparing historical data for best-fit analysis: aggregation_level={aggregation_level}, interval={interval}")

        # Get historical data
        historical_data = ForecastingService.prepare_aggregated_data(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            aggregation_level=aggregation_level,
            interval=interval,
            filters=filters
        )

        if historical_data.empty:
            raise ValidationException("No historical data available for forecasting")

        logger.info(f"Prepared {len(historical_data)} historical records for best-fit analysis")
        
        # Calculate number of forecast periods
        forecast_start_date = datetime.fromisoformat(result['forecast_start']).date()
        forecast_end_date = datetime.fromisoformat(result['forecast_end']).date()
        
        periods = ForecastExecutionService._calculate_periods(
            forecast_start_date,
            forecast_end_date,
            interval
        )
        
        # Execute best-fit algorithm selection
        process_log = []
        forecast_result = ForecastExecutionService.generate_forecast(
            historical_data=historical_data,
            config={
                'interval': interval,
                'periods': periods
            },
            process_log=process_log
        )
        
        # Generate forecast dates
        forecast_dates = ForecastExecutionService._generate_forecast_dates(
            forecast_start_date,
            periods,
            interval
        )

        logger.info(f"Generated {len(forecast_dates)} forecast periods from {forecast_start_date} to {forecast_end_date}")

        # Store results with a special algorithm_id for best-fit
        # Use algorithm_id = 999 for best-fit results
        best_fit_algorithm_id = 999

        # Create a best-fit algorithm mapping record
        best_fit_mapping_id = str(uuid.uuid4())

        logger.info(f"Storing best-fit results for algorithm: {forecast_result['selected_algorithm']}")

        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                # First, create the forecast_algorithms_mapping record for best-fit
                cursor.execute("""
                    INSERT INTO forecast_algorithms_mapping
                    (mapping_id, tenant_id, forecast_run_id, algorithm_id, algorithm_name, execution_order,
                     custom_parameters, execution_status, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    best_fit_mapping_id,
                    tenant_data["tenant_id"],
                    forecast_run_id,
                    best_fit_algorithm_id,
                    'Best Fit',
                    1,
                    Json({
                        'algorithms_evaluated': len(forecast_result['all_algorithms']),
                        'selected_algorithm': forecast_result['selected_algorithm']
                    }),
                    'Completed',
                    tenant_data["email"]
                ))

                conn.commit()

            finally:
                cursor.close()

        # Now store the forecast results
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                for forecast_date, forecast_value in zip(forecast_dates, forecast_result['forecast']):
                    result_id = str(uuid.uuid4())
                    forecast_value_float = float(forecast_value)

                    # Validate bounds
                    max_forecast_value = 999999.9999
                    if not (0 <= forecast_value_float <= max_forecast_value):
                        forecast_value_float = min(max_forecast_value, max(0, forecast_value_float))

                    forecast_value_float = round(forecast_value_float, 4)

                    accuracy_metric = round(forecast_result['accuracy'], 2) if forecast_result['accuracy'] else None

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
                        result.get('version_id'),
                        best_fit_mapping_id,  # Use the mapping_id from forecast_algorithms_mapping
                        best_fit_algorithm_id,
                        forecast_date,
                        forecast_value_float,
                        accuracy_metric,
                        'Accuracy',
                        Json({
                            'selected_algorithm': forecast_result['selected_algorithm'],
                            'mae': forecast_result['mae'],
                            'rmse': forecast_result['rmse'],
                            'mape': forecast_result['mape'],
                            'all_algorithms': len(forecast_result['all_algorithms'])
                        }),
                        tenant_data["email"]
                    ))

                conn.commit()

            finally:
                cursor.close()

        # Update forecast run completion
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    UPDATE forecast_runs
                    SET run_status = 'Completed',
                        run_progress = 100,
                        total_records = %s,
                        processed_records = %s,
                        failed_records = 0,
                        completed_at = %s,
                        updated_at = %s,
                        updated_by = %s
                    WHERE forecast_run_id = %s
                """, (
                    len(forecast_dates),
                    len(forecast_dates),
                    datetime.utcnow(),
                    datetime.utcnow(),
                    tenant_data["email"],
                    forecast_run_id
                ))
                conn.commit()
            finally:
                cursor.close()

        logger.info(f"Best-fit forecast run completed: {forecast_run_id} with {len(forecast_dates)} records")
        
        return ResponseHandler.success(data={
            'forecast_run_id': forecast_run_id,
            'status': 'Completed',
            'selected_algorithm': forecast_result['selected_algorithm'],
            'accuracy': forecast_result['accuracy'],
            'mae': forecast_result['mae'],
            'rmse': forecast_result['rmse'],
            'mape': forecast_result['mape'],
            'total_records': len(forecast_dates),
            'processed_records': len(forecast_dates),
            'failed_records': 0,
            'algorithms_evaluated': len(forecast_result['all_algorithms']),
            'process_log': process_log
        })
        
    except ValidationException as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except AppException as e:
        logger.error(f"App error: {str(e)}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error executing best-fit forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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


# ============================================================================
# ALGORITHMS
# ============================================================================

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
    

# Add this to app/api/routes/forecasting_routes.py

@router.post("/validate-filters", response_model=Dict[str, Any])
async def validate_forecast_filters(
    forecast_filters: Dict[str, Any],
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Validate forecast filters and preview what data will be selected.
    
    This endpoint lets you test your filters BEFORE creating a forecast run.
    """
    try:
        from app.core.forecasting_service import ForecastingService
        from app.core.database import get_db_manager
        
        # Extract filter parameters
        aggregation_level = forecast_filters.get('aggregation_level', 'product')
        interval = forecast_filters.get('interval', 'MONTHLY')
        
        # Preview data selection
        df = ForecastingService.prepare_aggregated_data(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            aggregation_level=aggregation_level,
            interval=interval,
            filters=forecast_filters
        )
        
        # Gather statistics
        validation_result = {
            "filters_applied": {k: v for k, v in forecast_filters.items() 
                               if k not in ['aggregation_level', 'interval']},
            "aggregation_level": aggregation_level,
            "interval": interval,
            "data_found": len(df) > 0,
            "total_records": len(df),
            "date_range": {
                "start": df['period'].min().isoformat() if not df.empty else None,
                "end": df['period'].max().isoformat() if not df.empty else None
            },
            "total_quantity_sum": float(df['total_quantity'].sum()) if not df.empty else 0,
            "sample_data": df.head(10).to_dict('records') if not df.empty else []
        }
        
        # Check for specific values in aggregation columns
        if not df.empty:
            agg_columns = ForecastingService._get_aggregation_columns(
                tenant_data["tenant_id"],
                tenant_data["database_name"],
                aggregation_level
            )
            
            unique_values = {}
            for col in agg_columns:
                if col in df.columns:
                    unique_vals = df[col].dropna().unique().tolist()
                    unique_values[col] = {
                        "count": len(unique_vals),
                        "values": unique_vals[:20]  # First 20 unique values
                    }
            
            validation_result["unique_values_in_results"] = unique_values
        
        # Warning if no data
        if df.empty:
            validation_result["warning"] = (
                "No data found with these filters! "
                "Check if the filter values exist in your master data."
            )
        
        logger.info(f"Validated forecast filters for tenant {tenant_data['tenant_id']}: {len(validation_result.get('sample_data', []))} sample records")
        return ResponseHandler.success(data=validation_result)
        
    except Exception as e:
        logger.error(f"Filter validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/master-data-values", response_model=Dict[str, Any])
async def get_available_master_data_values(
    tenant_data: Dict = Depends(get_current_tenant),
    field_name: Optional[str] = Query(None, description="Specific field to query")
):
    """
    Get all available values in master data for filtering.
    
    Use this to see what product codes, locations, etc. actually exist
    in your master data before setting up filters.
    """
    try:
        from app.core.database import get_db_manager
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                # Get all non-system columns from master_data
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'master_data' 
                    AND table_schema = 'public'
                    AND column_name NOT IN ('master_id', 'tenant_id', 'created_at', 
                                           'created_by', 'updated_at', 'updated_by')
                """)
                
                available_fields = [row[0] for row in cursor.fetchall()]
                
                # If specific field requested, get its values
                if field_name:
                    if field_name not in available_fields:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Field '{field_name}' not found in master data"
                        )
                    
                    cursor.execute(f"""
                        SELECT DISTINCT "{field_name}"
                        FROM master_data
                        WHERE tenant_id = %s 
                        AND "{field_name}" IS NOT NULL
                        ORDER BY "{field_name}"
                        LIMIT 1000
                    """, (tenant_data["tenant_id"],))
                    
                    values = [row[0] for row in cursor.fetchall()]
                    
                    logger.info(f"Retrieved {len(values)} unique values for field '{field_name}' in tenant {tenant_data['tenant_id']}")
                    return ResponseHandler.success(data={
                        "field_name": field_name,
                        "total_unique_values": len(values),
                        "values": values
                    })
                
                # Otherwise, get sample values for all fields
                field_values = {}
                for field in available_fields:
                    cursor.execute(f"""
                        SELECT DISTINCT "{field}"
                        FROM master_data
                        WHERE tenant_id = %s 
                        AND "{field}" IS NOT NULL
                        ORDER BY "{field}"
                        LIMIT 20
                    """, (tenant_data["tenant_id"],))
                    
                    values = [row[0] for row in cursor.fetchall()]
                    
                    # Get total count
                    cursor.execute(f"""
                        SELECT COUNT(DISTINCT "{field}")
                        FROM master_data
                        WHERE tenant_id = %s 
                        AND "{field}" IS NOT NULL
                    """, (tenant_data["tenant_id"],))
                    
                    total_count = cursor.fetchone()[0]
                    
                    field_values[field] = {
                        "total_unique_values": total_count,
                        "sample_values": values
                    }
                
                logger.info(f"Retrieved master data values for tenant {tenant_data['tenant_id']}: {len(available_fields)} fields")
                return ResponseHandler.success(data={
                    "available_fields": available_fields,
                    "field_values": field_values,
                    "note": "Showing first 20 values per field. Use field_name query param for full list."
                })
                
            finally:
                cursor.close()
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get master data values: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/verify-aggregation", response_model=Dict[str, Any])
async def verify_aggregation_logic(
    product_code: str = Query(...),
    interval: str = Query("MONTHLY", pattern="^(DAILY|WEEKLY|MONTHLY|QUARTERLY|YEARLY)$"),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Verify that aggregation logic is working correctly.
    Shows both raw query and Python pandas aggregation side-by-side.
    """
    try:
        from app.core.database import get_db_manager
        from app.core.forecasting_service import ForecastingService
        import pandas as pd
        
        db_manager = get_db_manager()
        
        # Method 1: Current aggregation logic
        current_result = ForecastingService.prepare_aggregated_data(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            aggregation_level="product",
            interval=interval,
            filters={"product": product_code}
        )
        
        # Method 2: Raw data without aggregation
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT 
                        s.date,
                        m.product,
                        s.quantity,
                        s.sales_id
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                    ORDER BY s.date
                """, (tenant_data["tenant_id"], product_code))
                
                raw_data = cursor.fetchall()
                
            finally:
                cursor.close()
        
        # Convert to DataFrame
        raw_df = pd.DataFrame(raw_data, columns=['date', 'product', 'quantity', 'sales_id'])
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        
        # Manual aggregation using pandas
        if interval == "MONTHLY":
            raw_df['period'] = raw_df['date'].dt.to_period('M').dt.to_timestamp()
        elif interval == "WEEKLY":
            raw_df['period'] = raw_df['date'].dt.to_period('W').dt.to_timestamp()
        elif interval == "QUARTERLY":
            raw_df['period'] = raw_df['date'].dt.to_period('Q').dt.to_timestamp()
        elif interval == "YEARLY":
            raw_df['period'] = raw_df['date'].dt.to_period('Y').dt.to_timestamp()
        else:
            raw_df['period'] = raw_df['date']
        
        manual_agg = raw_df.groupby('period').agg({
            'quantity': 'sum',
            'sales_id': 'nunique'
        }).reset_index()
        manual_agg.columns = ['period', 'total_quantity', 'unique_transactions']
        manual_agg['period'] = manual_agg['period'].astype(str)
        
        logger.info(f"Verified aggregation logic for product {product_code} with interval {interval}: current={len(current_result)}, manual={len(manual_agg)}")
        return ResponseHandler.success(data={
            "product": product_code,
            "interval": interval,
            "raw_data_sample": raw_df.head(20).to_dict('records'),
            "raw_data_total_records": len(raw_df),
            "current_aggregation": {
                "method": "SQL DATE_TRUNC",
                "records": len(current_result),
                "data": current_result.head(20).to_dict('records') if not current_result.empty else []
            },
            "manual_aggregation": {
                "method": "Python Pandas",
                "records": len(manual_agg),
                "data": manual_agg.to_dict('records')
            },
            "comparison": {
                "match": len(current_result) == len(manual_agg),
                "current_total_qty": float(current_result['total_quantity'].sum()) if not current_result.empty else 0,
                "manual_total_qty": float(manual_agg['total_quantity'].sum())
            }
        })
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
