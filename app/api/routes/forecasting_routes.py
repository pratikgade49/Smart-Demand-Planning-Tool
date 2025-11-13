"""
Forecasting API Routes.
Endpoints for forecast run management and execution.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, Optional
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
from app.core.exceptions import AppException
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
        result = ForecastVersionService.create_version(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            request=request,
            user_email=tenant_data["email"]
        )
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
        result = ForecastVersionService.update_version(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            version_id=version_id,
            request=request,
            user_email=tenant_data["email"]
        )
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
        result = ExternalFactorsService.create_factor(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            request=request,
            user_email=tenant_data["email"]
        )
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
        result = ForecastingService.create_forecast_run(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            request=request,
            user_email=tenant_data["email"]
        )
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
        result = ForecastExecutionService.execute_forecast_run(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            forecast_run_id=forecast_run_id,
            user_email=tenant_data["email"]
        )
        return ResponseHandler.success(data=result)
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error executing forecast: {str(e)}")
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
                
                return ResponseHandler.success(data=algorithms)
                
            finally:
                cursor.close()
                
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error listing algorithms: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")