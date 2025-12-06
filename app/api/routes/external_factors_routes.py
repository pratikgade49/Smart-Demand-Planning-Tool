"""
External Factors API Routes with FRED Integration.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from typing import Dict, Any, List, Optional
from datetime import date
from pydantic import BaseModel, Field

from app.core.external_factors_service import ExternalFactorsService
from app.core.fred_api_service import FREDAPIService
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException
from app.api.dependencies import get_current_tenant
from app.config import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/external-factors", tags=["External Factors"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class FREDSeriesConfig(BaseModel):
    """Configuration for importing a FRED series."""
    series_id: str = Field(..., description="FRED series ID")
    factor_name: str = Field(..., description="Name for the factor in the system")


class BulkFREDImportRequest(BaseModel):
    """Request for bulk importing from FRED."""
    series_configs: List[FREDSeriesConfig] = Field(..., min_items=1)
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class ForecastFactorsRequest(BaseModel):
    """Request for forecasting future factor values."""
    factor_names: List[str] = Field(..., min_items=1, description="Factor names to forecast")
    forecast_start: str = Field(..., description="Forecast start date (YYYY-MM-DD)")
    forecast_end: str = Field(..., description="Forecast end date (YYYY-MM-DD)")


# ============================================================================
# FRED INTEGRATION ENDPOINTS
# ============================================================================

@router.get("/fred/common-series", response_model=Dict[str, Any])
async def get_fred_common_series(
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get list of commonly used FRED series.
    
    Returns pre-configured list of popular economic indicators.
    """
    try:
        series_list = FREDAPIService.get_common_series_list()
        
        return ResponseHandler.success(data={
            "series": series_list,
            "total_count": len(series_list)
        })
        
    except Exception as e:
        logger.error(f"Failed to get common FRED series: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/fred/search", response_model=Dict[str, Any])
async def search_fred_series(
    search_text: str = Query(..., min_length=2, description="Search keyword"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results"),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Search for FRED series by keyword.
    
    **Examples**:
    - "unemployment" - finds unemployment rate series
    - "GDP" - finds GDP-related series
    - "consumer price" - finds CPI series
    """
    try:
        if not settings.FRED_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="FRED API integration not configured. Please add FRED_API_KEY to settings."
            )
        
        fred_service = FREDAPIService(settings.FRED_API_KEY)
        results = fred_service.search_series(search_text, limit)
        
        return ResponseHandler.success(data={
            "search_text": search_text,
            "results": results,
            "total_count": len(results)
        })
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"FRED search failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/fred/series/{series_id}/info", response_model=Dict[str, Any])
async def get_fred_series_info(
    series_id: str,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get detailed information about a FRED series.
    
    **Parameters**:
    - **series_id**: FRED series ID (e.g., 'GDP', 'CPIAUCSL', 'UNRATE')
    """
    try:
        if not settings.FRED_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="FRED API integration not configured"
            )
        
        fred_service = FREDAPIService(settings.FRED_API_KEY)
        info = fred_service.get_series_info(series_id)
        
        return ResponseHandler.success(data=info)
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Failed to get FRED series info: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/fred/import", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def bulk_import_from_fred(
    request: BulkFREDImportRequest,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Bulk import external factors from FRED API.
    
    **Example Request**:
    ```json
    {
      "series_configs": [
        {"series_id": "GDP", "factor_name": "US GDP"},
        {"series_id": "CPIAUCSL", "factor_name": "Consumer Price Index"},
        {"series_id": "UNRATE", "factor_name": "Unemployment Rate"}
      ],
      "start_date": "2020-01-01",
      "end_date": "2024-01-01"
    }
    ```
    
    **Note**: This will fetch historical data and store it as external factors.
    You can then select these factors when creating forecast runs.
    """
    try:
        if not settings.FRED_API_KEY:
            raise HTTPException(
                status_code=503,
                detail="FRED API integration not configured"
            )
        
        from app.core.external_factors_service import ExternalFactorsService
        from datetime import datetime
        
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d").date()
        
        result = ExternalFactorsService.bulk_import_from_fred(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            series_configs=[config.model_dump() for config in request.series_configs],
            start_date=start_date,
            end_date=end_date,
            user_email=tenant_data["email"],
            fred_api_key=settings.FRED_API_KEY
        )
        
        return ResponseHandler.success(data=result, status_code=201)
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Bulk FRED import failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# FACTOR FORECASTING
# ============================================================================

@router.post("/forecast-future", response_model=Dict[str, Any])
async def forecast_future_factors(
    request: ForecastFactorsRequest,
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Forecast future values for external factors.
    
    **Use Case**: When you want to use external factors in forecast runs that
    extend beyond available historical data, use this endpoint to forecast
    future values first.
    
    **Example Request**:
    ```json
    {
      "factor_names": ["US GDP", "Consumer Price Index"],
      "forecast_start": "2024-02-01",
      "forecast_end": "2024-12-31"
    }
    ```
    
    **Note**: Uses simple linear regression to forecast future values based
    on historical trends.
    """
    try:
        from app.core.external_factors_service import ExternalFactorsService
        from datetime import datetime
        
        forecast_start = datetime.strptime(request.forecast_start, "%Y-%m-%d").date()
        forecast_end = datetime.strptime(request.forecast_end, "%Y-%m-%d").date()
        
        result = ExternalFactorsService.forecast_future_factors(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            factor_names=request.factor_names,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            user_email=tenant_data["email"]
        )
        
        return ResponseHandler.success(data=result)
        
    except AppException as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Factor forecasting failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# FACTOR MANAGEMENT
# ============================================================================

@router.get("/available", response_model=Dict[str, Any])
async def get_available_factors(
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get list of all available external factors for the tenant.
    
    Returns unique factor names with their metadata.
    """
    try:
        from app.core.database import get_db_manager
        
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT 
                        factor_name,
                        MIN(date) as earliest_date,
                        MAX(date) as latest_date,
                        COUNT(*) as data_points,
                        MAX(unit) as unit,
                        MAX(source) as source
                    FROM external_factors
                    WHERE tenant_id = %s AND deleted_at IS NULL
                    GROUP BY factor_name
                    ORDER BY factor_name
                """, (tenant_data["tenant_id"],))
                
                factors = []
                for row in cursor.fetchall():
                    factors.append({
                        "factor_name": row[0],
                        "earliest_date": row[1].isoformat() if row[1] else None,
                        "latest_date": row[2].isoformat() if row[2] else None,
                        "data_points": row[3],
                        "unit": row[4],
                        "source": row[5]
                    })
                
                return ResponseHandler.success(data={
                    "factors": factors,
                    "total_count": len(factors)
                })
                
            finally:
                cursor.close()
                
    except Exception as e:
        logger.error(f"Failed to get available factors: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("", response_model=Dict[str, Any])
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
    
    - **factor_name**: Filter by specific factor
    - **date_from**: Start date filter
    - **date_to**: End date filter
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
        logger.error(f"Failed to list factors: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
