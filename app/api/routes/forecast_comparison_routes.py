"""
Forecast Comparison API Routes.
Endpoints for comparing multiple forecast scenarios.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional

from app.core.forecast_comparison_service import ForecastComparisonService
from app.core.responses import ResponseHandler
from app.core.exceptions import AppException
from app.api.dependencies import get_current_tenant
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/forecasting/compare", tags=["Forecast Comparison"])


@router.get("", response_model=Dict[str, Any])
async def compare_forecasts(
    entity_identifier: str = Query(..., description="Entity identifier (e.g., '10009736' or '1001-Loc1')"),
    aggregation_level: str = Query(..., description="Aggregation level (e.g., 'product', 'product-location')"),
    interval: str = Query(..., pattern="^(DAILY|WEEKLY|MONTHLY|QUARTERLY|YEARLY)$", description="Time interval"),
    forecast_run_ids: Optional[str] = Query(None, description="Comma-separated list of forecast_run_ids to compare"),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Compare forecast scenarios for a specific entity.
    
    **Entity Identifier Format:**
    - Single dimension: `"1001"` (for product only)
    - Multi-dimension: `"1001-Loc1"` (for product-location)
    - Multi-dimension: `"1001-C1-Loc1"` (for product-customer-location)
    
    The entity_identifier must match the aggregation_level structure.
    
    **Query Parameters:**
    - **entity_identifier**: Hyphen-separated entity values (e.g., "1001-Loc1")
    - **aggregation_level**: Aggregation level matching the identifier (e.g., "product-location")
    - **interval**: Time interval for aggregation (MONTHLY, WEEKLY, etc.)
    - **forecast_run_ids**: Optional comma-separated list of specific forecast runs to compare
    
    **Response:**
    - **historical_data**: All available historical data for the entity
    - **available_forecasts**: List of all matching forecast runs with metadata
    - **forecast_data**: Actual forecast values for each run
    - **comparison_matrix**: Pairwise comparison metrics
    
    **Example Usage:**
    ```
    GET /api/v1/forecasting/compare?entity_identifier=10009736&aggregation_level=product&interval=MONTHLY
    GET /api/v1/forecasting/compare?entity_identifier=1001-Loc1&aggregation_level=product-location&interval=MONTHLY&forecast_run_ids=uuid1,uuid2
    ```
    
    This returns all forecasts for Product 1001 at Location Loc1, allowing you to see:
    - Baseline forecast vs Simulation with external factors
    - Different algorithms applied to the same entity
    - Impact of different parameters or versions
    """
    try:
        logger.info(
            f"Comparison request: entity='{entity_identifier}', "
            f"level='{aggregation_level}', interval='{interval}'"
        )
        
        # Parse forecast_run_ids if provided
        run_ids_list = None
        if forecast_run_ids:
            run_ids_list = [rid.strip() for rid in forecast_run_ids.split(',')]
            logger.info(f"Comparing specific runs: {run_ids_list}")
        
        # Execute comparison
        result = ForecastComparisonService.compare_forecasts(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            entity_identifier=entity_identifier,
            aggregation_level=aggregation_level,
            interval=interval,
            forecast_run_ids=run_ids_list
        )
        
        logger.info(
            f"Comparison completed: {len(result['available_forecasts'])} forecasts, "
            f"{len(result['historical_data'])} historical points"
        )
        
        return ResponseHandler.success(data=result)
        
    except AppException as e:
        logger.error(f"Comparison failed: {str(e)}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in comparison: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/summary", response_model=Dict[str, Any])
async def get_comparison_summary(
    entity_identifier: str = Query(..., description="Entity identifier"),
    aggregation_level: str = Query(..., description="Aggregation level"),
    interval: str = Query(..., pattern="^(DAILY|WEEKLY|MONTHLY|QUARTERLY|YEARLY)$"),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Get a summary of available forecasts for comparison without full data.
    
    Use this endpoint to quickly see what forecasts are available for an entity
    before requesting the full comparison data.
    
    **Response includes:**
    - Entity details
    - List of available forecast runs with metadata
    - Count of historical data points
    - But NO actual forecast values (lighter response)
    
    **Example Usage:**
    ```
    GET /api/v1/forecasting/compare/summary?entity_identifier=10009736&aggregation_level=product&interval=MONTHLY
    ```
    """
    try:
        logger.info(f"Summary request for entity '{entity_identifier}'")
        
        # Parse entity
        entity_fields = ForecastComparisonService.parse_entity_identifier(
            entity_identifier, aggregation_level
        )
        
        # Get historical data count only
        historical_data = ForecastComparisonService.get_historical_data(
            tenant_data["tenant_id"],
            tenant_data["database_name"],
            entity_fields,
            interval
        )
        
        # Find matching runs
        matching_runs = ForecastComparisonService.find_matching_forecast_runs(
            tenant_data["tenant_id"],
            tenant_data["database_name"],
            entity_fields,
            aggregation_level,
            interval
        )
        
        if not matching_runs:
            return ResponseHandler.success(data={
                "entity": {
                    "identifier": entity_identifier,
                    "aggregation_level": aggregation_level,
                    "interval": interval,
                    "field_values": entity_fields
                },
                "historical_data_points": 0,
                "available_forecasts_count": 0,
                "available_forecasts": [],
                "message": "No forecasts found for this entity"
            })
        
        # Get metadata for runs
        run_ids = [run['forecast_run_id'] for run in matching_runs]
        forecast_metadata = ForecastComparisonService.get_forecast_metadata(
            tenant_data["tenant_id"],
            tenant_data["database_name"],
            run_ids
        )
        
        # Build summary
        available_forecasts = []
        for run in matching_runs:
            run_id = run['forecast_run_id']
            metadata = forecast_metadata.get(run_id, {})
            
            forecast_name = f"{run['version_name']} - {metadata.get('algorithm_name', 'Unknown')}"
            if metadata.get('external_factors'):
                forecast_name += f" (with {len(metadata['external_factors'])} factors)"
            
            available_forecasts.append({
                "forecast_run_id": run_id,
                "forecast_name": forecast_name,
                "version_name": run['version_name'],
                "algorithm_name": metadata.get('algorithm_name'),
                "external_factors_count": len(metadata.get('external_factors', [])),
                "has_external_factors": len(metadata.get('external_factors', [])) > 0,
                "forecast_period": {
                    "start": metadata.get('forecast_start') or run['forecast_start'],
                    "end": metadata.get('forecast_end') or run['forecast_end']
                },
                "created_at": run['created_at']
            })
        
        summary = {
            "entity": {
                "identifier": entity_identifier,
                "aggregation_level": aggregation_level,
                "interval": interval,
                "field_values": entity_fields
            },
            "historical_data_points": len(historical_data),
            "historical_period": {
                "start": historical_data[0]['date'] if historical_data else None,
                "end": historical_data[-1]['date'] if historical_data else None
            },
            "available_forecasts_count": len(available_forecasts),
            "available_forecasts": available_forecasts
        }
        
        logger.info(f"Summary completed: {len(available_forecasts)} forecasts available")
        return ResponseHandler.success(data=summary)
        
    except AppException as e:
        logger.error(f"Summary failed: {str(e)}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in summary: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/export", response_model=Dict[str, Any])
async def export_comparison(
    entity_identifier: str = Query(..., description="Entity identifier"),
    aggregation_level: str = Query(..., description="Aggregation level"),
    interval: str = Query(..., pattern="^(DAILY|WEEKLY|MONTHLY|QUARTERLY|YEARLY)$"),
    forecast_run_ids: str = Query(..., description="Comma-separated list of forecast_run_ids"),
    format: str = Query("json", pattern="^(json|csv)$", description="Export format"),
    tenant_data: Dict = Depends(get_current_tenant)
):
    """
    Export comparison data in JSON or CSV format.
    
    **Parameters:**
    - **entity_identifier**: Entity to compare
    - **aggregation_level**: Aggregation level
    - **interval**: Time interval
    - **forecast_run_ids**: Comma-separated list of runs to compare (required)
    - **format**: Export format (json or csv)
    
    **CSV Format:**
    - One row per date
    - Columns: date, actual_quantity, forecast_1, forecast_2, ...
    - Easy to import into Excel or other tools
    
    **Example Usage:**
    ```
    POST /api/v1/forecasting/compare/export?entity_identifier=10009736&aggregation_level=product&interval=MONTHLY&forecast_run_ids=uuid1,uuid2&format=csv
    ```
    """
    try:
        logger.info(f"Export request for entity '{entity_identifier}' in format '{format}'")
        
        # Parse forecast run IDs
        run_ids_list = [rid.strip() for rid in forecast_run_ids.split(',')]
        
        if len(run_ids_list) < 1:
            raise HTTPException(
                status_code=400,
                detail="At least one forecast_run_id is required for export"
            )
        
        # Get full comparison data
        result = ForecastComparisonService.compare_forecasts(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            entity_identifier=entity_identifier,
            aggregation_level=aggregation_level,
            interval=interval,
            forecast_run_ids=run_ids_list
        )
        
        if format == "csv":
            # Convert to CSV format
            import io
            import csv
            from collections import defaultdict
            
            output = io.StringIO()
            
            # Build aligned data structure
            all_dates = set()
            historical_map = {}
            forecast_maps = {}
            
            # Collect all dates and values
            for point in result['historical_data']:
                date = point['date']
                all_dates.add(date)
                historical_map[date] = point['actual_quantity']
            
            for run_id, forecast_points in result['forecast_data'].items():
                forecast_maps[run_id] = {}
                for point in forecast_points:
                    date = point['date']
                    all_dates.add(date)
                    forecast_maps[run_id][date] = point['forecast_quantity']
            
            # Sort dates
            sorted_dates = sorted(all_dates)
            
            # Build CSV header
            header = ['date', 'actual_quantity']
            run_names = {}
            for forecast in result['available_forecasts']:
                run_id = forecast['forecast_run_id']
                if run_id in run_ids_list:
                    run_names[run_id] = forecast['forecast_name']
                    header.append(f"forecast_{run_id[:8]}")
            
            writer = csv.writer(output)
            writer.writerow(header)
            
            # Write data rows
            for date in sorted_dates:
                row = [date, historical_map.get(date, '')]
                for run_id in run_ids_list:
                    row.append(forecast_maps.get(run_id, {}).get(date, ''))
                writer.writerow(row)
            
            csv_data = output.getvalue()
            
            return ResponseHandler.success(data={
                "format": "csv",
                "content": csv_data,
                "filename": f"forecast_comparison_{entity_identifier}_{interval}.csv",
                "row_count": len(sorted_dates),
                "forecast_names": run_names
            })
        
        else:
            # JSON format (default)
            return ResponseHandler.success(data={
                "format": "json",
                "content": result,
                "filename": f"forecast_comparison_{entity_identifier}_{interval}.json"
            })
        
    except AppException as e:
        logger.error(f"Export failed: {str(e)}")
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error(f"Unexpected error in export: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")