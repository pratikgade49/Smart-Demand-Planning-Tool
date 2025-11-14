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



# Add this to app/api/routes/forecasting_routes.py

# Add this to app/api/routes/forecasting_routes.py

@router.post("/diagnose-data", response_model=Dict[str, Any])
async def diagnose_product_data(
    product_code: str = Query(..., description="Product code to diagnose"),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Deep diagnostic of product data to identify issues.
    
    Checks:
    - Raw sales_data records
    - Timezone issues
    - Duplicate detection
    - Date range verification
    - Transaction counting
    """
    try:
        from app.core.database import get_db_manager
        db_manager = get_db_manager()
        
        diagnostics = {}
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                # 1. Check master_data for this product
                cursor.execute("""
                    SELECT master_id, product, created_at
                    FROM master_data
                    WHERE tenant_id = %s AND product = %s
                    LIMIT 10
                """, (tenant_data["tenant_id"], product_code))
                
                master_records = cursor.fetchall()
                diagnostics["master_data"] = {
                    "found": len(master_records) > 0,
                    "count": len(master_records),
                    "records": [
                        {
                            "master_id": str(r[0]),
                            "product": r[1],
                            "created_at": r[2].isoformat() if r[2] else None
                        } for r in master_records
                    ]
                }
                
                if not master_records:
                    return ResponseHandler.success(data={
                        "error": f"Product '{product_code}' not found in master_data",
                        "diagnostics": diagnostics
                    })
                
                master_ids = [r[0] for r in master_records]  # Keep as UUID objects
                
                # 2. Check raw sales_data records
                cursor.execute("""
                    SELECT 
                        sales_id,
                        master_id,
                        date,
                        quantity,
                        created_at
                    FROM sales_data
                    WHERE tenant_id = %s 
                    AND master_id = ANY(%s::uuid[])
                    ORDER BY date
                    LIMIT 100
                """, (tenant_data["tenant_id"], master_ids))
                
                sales_records = cursor.fetchall()
                diagnostics["raw_sales_data"] = {
                    "total_records": len(sales_records),
                    "sample_records": [
                        {
                            "sales_id": str(r[0]),
                            "master_id": str(r[1]),
                            "date": r[2].isoformat() if r[2] else None,
                            "quantity": float(r[3]),
                            "created_at": r[4].isoformat() if r[4] else None
                        } for r in sales_records[:20]
                    ]
                }
                
                # 3. Check for duplicates
                cursor.execute("""
                    SELECT 
                        date,
                        COUNT(*) as record_count,
                        COUNT(DISTINCT sales_id) as unique_sales_ids,
                        SUM(quantity) as total_quantity
                    FROM sales_data
                    WHERE tenant_id = %s 
                    AND master_id = ANY(%s::uuid[])
                    GROUP BY date
                    HAVING COUNT(*) > 1
                    ORDER BY date
                    LIMIT 20
                """, (tenant_data["tenant_id"], master_ids))
                
                duplicate_dates = cursor.fetchall()
                diagnostics["potential_duplicates"] = {
                    "dates_with_multiple_records": len(duplicate_dates),
                    "details": [
                        {
                            "date": r[0].isoformat() if r[0] else None,
                            "record_count": r[1],
                            "unique_sales_ids": r[2],
                            "total_quantity": float(r[3])
                        } for r in duplicate_dates
                    ]
                }
                
                # 4. Monthly aggregation (raw - no timezone conversion)
                cursor.execute("""
                    SELECT 
                        date,
                        COUNT(*) as transaction_count,
                        COUNT(DISTINCT sales_id) as unique_transactions,
                        SUM(quantity) as total_quantity,
                        MIN(quantity) as min_quantity,
                        MAX(quantity) as max_quantity
                    FROM sales_data
                    WHERE tenant_id = %s 
                    AND master_id = ANY(%s::uuid[])
                    GROUP BY date
                    ORDER BY date
                """, (tenant_data["tenant_id"], master_ids))
                
                daily_data = cursor.fetchall()
                diagnostics["daily_aggregation"] = {
                    "total_days": len(daily_data),
                    "sample": [
                        {
                            "date": r[0].isoformat() if r[0] else None,
                            "transaction_count": r[1],
                            "unique_transactions": r[2],
                            "total_quantity": float(r[3]),
                            "min_quantity": float(r[4]),
                            "max_quantity": float(r[5])
                        } for r in daily_data[:30]
                    ]
                }
                
                # 5. Check timezone storage
                cursor.execute("""
                    SELECT 
                        date,
                        date::timestamp AT TIME ZONE 'UTC' as utc_timestamp,
                        date::timestamp AT TIME ZONE 'Asia/Kolkata' as ist_timestamp
                    FROM sales_data
                    WHERE tenant_id = %s 
                    AND master_id = ANY(%s::uuid[])
                    ORDER BY date
                    LIMIT 10
                """, (tenant_data["tenant_id"], master_ids))
                
                timezone_check = cursor.fetchall()
                diagnostics["timezone_check"] = {
                    "note": "DATE type doesn't store timezone, check if dates are correct",
                    "sample": [
                        {
                            "stored_date": r[0].isoformat() if r[0] else None,
                            "as_utc": r[1].isoformat() if r[1] else None,
                            "as_ist": r[2].isoformat() if r[2] else None
                        } for r in timezone_check
                    ]
                }
                
                # 6. Check DATE_TRUNC behavior
                cursor.execute("""
                    SELECT 
                        DATE_TRUNC('month', s.date) as truncated_month,
                        s.date as original_date,
                        COUNT(*) as count_per_date
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                    GROUP BY DATE_TRUNC('month', s.date), s.date
                    ORDER BY truncated_month, s.date
                    LIMIT 100
                """, (tenant_data["tenant_id"], product_code))
                
                trunc_results = cursor.fetchall()
                diagnostics["date_trunc_behavior"] = {
                    "note": "Shows how DATE_TRUNC groups your dates",
                    "sample": [
                        {
                            "truncated_month": r[0].isoformat() if r[0] else None,
                            "original_date": r[1].isoformat() if r[1] else None,
                            "count": r[2]
                        } for r in trunc_results[:30]
                    ]
                }
                
                # 7. Summary statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT sales_id) as unique_sales_ids,
                        COUNT(DISTINCT date) as unique_dates,
                        MIN(date) as earliest_date,
                        MAX(date) as latest_date,
                        SUM(quantity) as total_quantity,
                        AVG(quantity) as avg_quantity
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                """, (tenant_data["tenant_id"], product_code))
                
                summary = cursor.fetchone()
                diagnostics["summary"] = {
                    "total_records": summary[0],
                    "unique_sales_ids": summary[1],
                    "unique_dates": summary[2],
                    "earliest_date": summary[3].isoformat() if summary[3] else None,
                    "latest_date": summary[4].isoformat() if summary[4] else None,
                    "total_quantity": float(summary[5]) if summary[5] else 0,
                    "avg_quantity": float(summary[6]) if summary[6] else 0,
                    "data_quality_flags": {
                        "has_duplicates": summary[0] > summary[1],
                        "suspicious_if_true": summary[0] != summary[1]
                    }
                }
                
                return ResponseHandler.success(data=diagnostics)
                
            finally:
                cursor.close()
                
    except Exception as e:
        logger.error(f"Diagnostic failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


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

# Add this simpler version to app/api/routes/forecasting_routes.py

@router.get("/simple-diagnose", response_model=Dict[str, Any])
async def simple_diagnose_product(
    product_code: str = Query(..., description="Product code to diagnose"),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Simple diagnostic - just shows raw data for a product.
    """
    try:
        from app.core.database import get_db_manager
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                # 1. Get sample sales data directly
                cursor.execute("""
                    SELECT 
                        s.sales_id,
                        s.date,
                        s.quantity,
                        s.uom,
                        s.unit_price,
                        m.product,
                        s.created_at
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                    ORDER BY s.date
                    LIMIT 100
                """, (tenant_data["tenant_id"], product_code))
                
                sales_data = []
                for row in cursor.fetchall():
                    sales_data.append({
                        "sales_id": str(row[0]),
                        "date": row[1].isoformat() if row[1] else None,
                        "quantity": float(row[2]) if row[2] else 0,
                        "uom": row[3],
                        "unit_price": float(row[4]) if row[4] else None,
                        "product": row[5],
                        "created_at": row[6].isoformat() if row[6] else None
                    })
                
                # 2. Get summary stats
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_records,
                        COUNT(DISTINCT s.sales_id) as unique_sales,
                        COUNT(DISTINCT s.date) as unique_dates,
                        MIN(s.date) as earliest_date,
                        MAX(s.date) as latest_date,
                        SUM(s.quantity) as total_quantity
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                """, (tenant_data["tenant_id"], product_code))
                
                stats = cursor.fetchone()
                
                # 3. Check for date duplicates
                cursor.execute("""
                    SELECT 
                        s.date,
                        COUNT(*) as records_per_date
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                    GROUP BY s.date
                    ORDER BY records_per_date DESC, s.date
                    LIMIT 20
                """, (tenant_data["tenant_id"], product_code))
                
                date_counts = []
                for row in cursor.fetchall():
                    date_counts.append({
                        "date": row[0].isoformat() if row[0] else None,
                        "records_count": row[1]
                    })
                
                # 4. Monthly aggregation
                cursor.execute("""
                    SELECT 
                        DATE_TRUNC('month', s.date)::date as month,
                        COUNT(*) as total_records,
                        COUNT(DISTINCT s.sales_id) as unique_sales,
                        SUM(s.quantity) as total_quantity
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                    GROUP BY DATE_TRUNC('month', s.date)::date
                    ORDER BY month
                """, (tenant_data["tenant_id"], product_code))
                
                monthly_data = []
                for row in cursor.fetchall():
                    monthly_data.append({
                        "month": row[0].isoformat() if row[0] else None,
                        "total_records": row[1],
                        "unique_sales": row[2],
                        "total_quantity": float(row[3]) if row[3] else 0
                    })
                
                return ResponseHandler.success(data={
                    "product": product_code,
                    "summary": {
                        "total_records": stats[0],
                        "unique_sales_ids": stats[1],
                        "unique_dates": stats[2],
                        "earliest_date": stats[3].isoformat() if stats[3] else None,
                        "latest_date": stats[4].isoformat() if stats[4] else None,
                        "total_quantity": float(stats[5]) if stats[5] else 0,
                        "data_quality_flag": "⚠️ DUPLICATES!" if stats[0] > stats[1] else "✅ OK"
                    },
                    "sample_sales_data": sales_data[:20],
                    "dates_with_multiple_records": date_counts,
                    "monthly_aggregation": monthly_data,
                    "analysis": {
                        "has_duplicate_sales_ids": stats[0] > stats[1],
                        "records_per_date_avg": round(stats[0] / stats[2], 2) if stats[2] > 0 else 0,
                        "note": "If records_per_date_avg > 1, you have multiple transactions per date (normal for daily data)"
                    }
                })
                
            finally:
                cursor.close()
                
    except Exception as e:
        logger.error(f"Simple diagnostic failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-56-transactions", response_model=Dict[str, Any])
async def check_56_transaction_mystery(
    product_code: str = Query(..., description="Product code"),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Specifically investigate why every month shows 56 transactions.
    """
    try:
        from app.core.database import get_db_manager
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                # Check if there are exactly 56 unique dates
                cursor.execute("""
                    SELECT 
                        COUNT(DISTINCT s.date) as unique_dates,
                        MIN(s.date) as min_date,
                        MAX(s.date) as max_date
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                """, (tenant_data["tenant_id"], product_code))
                
                date_info = cursor.fetchone()
                
                # Get monthly breakdown
                cursor.execute("""
                    SELECT 
                        DATE_TRUNC('month', s.date)::date as month,
                        COUNT(DISTINCT s.date) as unique_dates_in_month,
                        COUNT(*) as total_records,
                        ARRAY_AGG(DISTINCT s.date ORDER BY s.date) as all_dates
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                    GROUP BY DATE_TRUNC('month', s.date)::date
                    ORDER BY month
                """, (tenant_data["tenant_id"], product_code))
                
                monthly_breakdown = []
                for row in cursor.fetchall():
                    monthly_breakdown.append({
                        "month": row[0].isoformat() if row[0] else None,
                        "unique_dates": row[1],
                        "total_records": row[2],
                        "sample_dates": [d.isoformat() for d in row[3][:5]] if row[3] else []
                    })
                
                # Check if same sales_id appears multiple times
                cursor.execute("""
                    SELECT 
                        s.sales_id,
                        COUNT(*) as occurrence_count,
                        ARRAY_AGG(s.date ORDER BY s.date) as dates
                    FROM sales_data s
                    JOIN master_data m ON s.master_id = m.master_id
                    WHERE s.tenant_id = %s 
                    AND m.product = %s
                    GROUP BY s.sales_id
                    HAVING COUNT(*) > 1
                    LIMIT 10
                """, (tenant_data["tenant_id"], product_code))
                
                duplicate_sales_ids = []
                for row in cursor.fetchall():
                    duplicate_sales_ids.append({
                        "sales_id": str(row[0]),
                        "appears_times": row[1],
                        "dates": [d.isoformat() for d in row[2]] if row[2] else []
                    })
                
                return ResponseHandler.success(data={
                    "product": product_code,
                    "total_unique_dates": date_info[0],
                    "date_range": {
                        "start": date_info[1].isoformat() if date_info[1] else None,
                        "end": date_info[2].isoformat() if date_info[2] else None
                    },
                    "monthly_breakdown": monthly_breakdown,
                    "duplicate_sales_ids_found": len(duplicate_sales_ids),
                    "duplicate_examples": duplicate_sales_ids,
                    "mystery_analysis": {
                        "is_56_related_to_dates": date_info[0] == 56,
                        "explanation": "If unique_dates_in_month is always 56, you likely uploaded 56 dates worth of data for each month period"
                    }
                })
                
            finally:
                cursor.close()
                
    except Exception as e:
        logger.error(f"Check 56 failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
            raw_df['period'] = raw_df['date'].dt.to_period('M')
        elif interval == "WEEKLY":
            raw_df['period'] = raw_df['date'].dt.to_period('W')
        elif interval == "QUARTERLY":
            raw_df['period'] = raw_df['date'].dt.to_period('Q')
        elif interval == "YEARLY":
            raw_df['period'] = raw_df['date'].dt.to_period('Y')
        else:
            raw_df['period'] = raw_df['date']
        
        manual_agg = raw_df.groupby('period').agg({
            'quantity': 'sum',
            'sales_id': 'nunique'
        }).reset_index()
        manual_agg.columns = ['period', 'total_quantity', 'unique_transactions']
        manual_agg['period'] = manual_agg['period'].astype(str)
        
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