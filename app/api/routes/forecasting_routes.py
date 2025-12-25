"""
Forecasting API Routes.
Endpoints for forecast run management and execution.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from app.schemas.forecasting import (
    ForecastVersionCreate,
    ForecastVersionUpdate,
    ExternalFactorCreate,
    ExternalFactorUpdate
)



from app.core.resource_monitor import monitor_endpoint, ResourceMonitor
import logging
from app.core.database import get_db_manager
from app.core.forecasting_service import ForecastingService
from app.core.forecast_version_service import ForecastVersionService
from app.core.external_factors_service import ExternalFactorsService
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException, ValidationException
from app.core.algorithm_parameters import AlgorithmParametersService
from app.core.forecast_job_service import ForecastJobService
from app.core.background_forecast_executor import BackgroundForecastExecutor
from app.api.dependencies import get_current_tenant

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecasting", tags=["Forecasting"])


class DirectForecastExecutionRequest(BaseModel):
    """Request for forecast execution with automatic entity handling."""
    version_id: str
    forecast_filters: Dict[str, Any]
    forecast_start: str = Field(..., description="Start date in YYYY-MM-DD format")
    forecast_end: str = Field(..., description="End date in YYYY-MM-DD format")
    algorithm_id: Optional[int] = Field(default=999, description="Single algorithm ID")
    custom_parameters: Optional[Dict[str, Any]] = None
    algorithms: Optional[List[Dict[str, Any]]] = None



@router.post("/execute-forecast-async", response_model=Dict[str, Any], status_code=status.HTTP_202_ACCEPTED)
@monitor_endpoint("Forecast Request Submission", warn_threshold=3.0)  # Add monitoring decorator
async def execute_forecast_async(
    request_data: Dict[str, Any],  # Your request model here
    background_tasks: BackgroundTasks,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Asynchronously execute a forecast in the background with resource monitoring.
    
    This endpoint returns immediately with a job_id. Use the job_id to check 
    the status and retrieve results via the /job-status/{job_id} endpoint.
    
    Returns:
        - job_id: Unique identifier for tracking the forecast execution
        - status: Current job status (will be "pending")
        - created_at: Timestamp when the job was created
        - start_resources: System resources at job creation
    
    Check job status with: GET /api/v1/forecasting/job-status/{job_id}
    """
    try:
        # Get current resources for logging
        current_resources = ResourceMonitor.get_system_resources()
        
        logger.info(
            f"Async forecast execution requested for tenant {tenant_data['tenant_id']}",
            extra={'start_resources': current_resources}
        )
        
        # Create forecast job record
        job_result = ForecastJobService.create_job(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            request_data=request_data,
            user_email=tenant_data['email']
        )
        
        job_id = job_result['job_id']
        
        # Add background task to execute forecast
        background_tasks.add_task(
            BackgroundForecastExecutor.execute_forecast_async,
            job_id=job_id,
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            request_data=request_data,
            tenant_data=tenant_data
        )
        
        logger.info(f"Created async forecast job {job_id}, added to background task queue")
        
        return {
            "status": "success",
            "data": {
                "job_id": job_id,
                "status": "pending",
                "created_at": job_result['created_at'],
                "message": "Forecast execution started in background. Use job_id to check status.",
                "start_resources": current_resources
            }
        }
    
    except Exception as e:
        logger.error(f"Unexpected error in async forecast execution: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




@router.get("/job-status/{job_id}", response_model=Dict[str, Any])
@monitor_endpoint("Forecast Job Status Check", warn_threshold=2.0)
async def get_forecast_job_status(
    job_id: str,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get the status and results of a forecast job with performance metrics.
    
    Returns:
        - job_id: Job identifier
        - status: Current status (pending, running, completed, failed)
        - result: Forecast results (if completed)
        - error: Error message (if failed)
        - performance_metrics: Resource usage and timing information
        - created_at: Job creation timestamp
        - started_at: Job start timestamp
        - completed_at: Job completion timestamp
    """
    try:
        logger.info(f"Checking job status for {job_id}")
        
        job_status = ForecastJobService.get_job_status(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            job_id=job_id
        )
        
        if job_status.get('status') == 'not_found':
            raise HTTPException(status_code=404, detail="Job not found")
        
        return {
            "status": "success",
            "data": job_status
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/job-history", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_forecast_job_history(
    limit: int = Query(50, ge=1, le=200),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get the history of forecast jobs for the current user.
    
    Parameters:
        - limit: Maximum number of jobs to return (1-200, default 50)
    
    Returns:
        - jobs: List of job summaries with their status and timestamps
    """
    try:
        logger.info(f"Fetching forecast job history for {tenant_data['email']}")
        
        jobs = ForecastJobService.get_user_jobs(
            tenant_id=tenant_data['tenant_id'],
            database_name=tenant_data['database_name'],
            user_email=tenant_data['email'],
            limit=limit
        )
        
        return ResponseHandler.success(
            data={"jobs": jobs, "total": len(jobs)},
            status_code=200
        )
    
    except Exception as e:
        logger.error(f"Error getting job history: {str(e)}", exc_info=True)
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
    

@router.get("/metrics/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get performance metrics for forecasting operations.
    
    Returns:
        - current_resources: Current system resource usage
        - performance_summary: Summary of all tracked operations
    """
    from app.core.resource_monitor import performance_tracker
    
    summary = performance_tracker.get_summary()
    current_resources = ResourceMonitor.get_system_resources()
    resource_warnings = ResourceMonitor.check_resource_warnings(current_resources)
    
    return {
        "status": "success",
        "data": {
            "current_resources": current_resources,
            "resource_warnings": resource_warnings,
            "performance_summary": summary
        }
    }