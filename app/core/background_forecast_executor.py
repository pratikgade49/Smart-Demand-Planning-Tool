"""
Background Task Executor - Handles async forecast execution.
Executes forecasts in background and updates job status.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from app.core.forecast_job_service import ForecastJobService, JobStatus
from app.core.database import get_db_manager
from app.core.forecast_utils import (
    _process_entity_forecast,
    AlgorithmConfig
)
from app.core.forecasting_service import ForecastingService
from app.core.forecast_execution_service import ForecastExecutionService
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class BackgroundForecastExecutor:
    """Executes forecast jobs in the background."""

    @staticmethod
    def execute_forecast_async(
        job_id: str,
        tenant_id: str,
        database_name: str,
        request_data: Dict[str, Any],
        tenant_data: Dict[str, Any]
    ) -> None:
        """
        Execute forecast asynchronously and update job status.
        This method is called by FastAPI's BackgroundTasks.
        
        Args:
            job_id: Unique job identifier
            tenant_id: Tenant identifier
            database_name: Database name for tenant
            request_data: The forecast request data
            tenant_data: Tenant data dictionary
        """
        try:
            # Update status to RUNNING
            ForecastJobService.update_job_status(
                tenant_id=tenant_id,
                database_name=database_name,
                job_id=job_id,
                status=JobStatus.RUNNING
            )
            
            logger.info(f"Starting background forecast execution for job {job_id}")
            
            # Import here to avoid circular imports
            from datetime import date
            
            # Extract configuration
            aggregation_level = request_data.get('forecast_filters', {}).get('aggregation_level', 'product')
            interval = request_data.get('forecast_filters', {}).get('interval', 'MONTHLY')
            selected_factors = request_data.get('forecast_filters', {}).get('selected_external_factors')
            
            # ✅ CHANGED: Get distinct aggregation combinations from filtered data
            # instead of building from filter values
            # This allows forecasts for all combinations where filters apply
            filters_for_data = {k: v for k, v in request_data.get('forecast_filters', {}).items() 
                               if k not in ['aggregation_level', 'interval', 'selected_external_factors']}
            
            entity_combinations = ForecastingService.get_aggregation_combinations(
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level,
                filters=filters_for_data if filters_for_data else None
            )
            
            logger.info(f"Job {job_id}: Detected {len(entity_combinations)} distinct entity combinations from database")
            
            # Execute forecast for each entity
            forecast_runs = []
            total_records = 0
            successful_runs = 0
            failed_runs = 0
            
            # Calculate forecast periods
            forecast_start_date = datetime.fromisoformat(request_data.get('forecast_start')).date()
            forecast_end_date = datetime.fromisoformat(request_data.get('forecast_end')).date()
            periods = ForecastExecutionService._calculate_periods(
                forecast_start_date, forecast_end_date, interval
            )
            forecast_dates = ForecastExecutionService._generate_forecast_dates(
                forecast_start_date, periods, interval
            )
            
            # Determine algorithms to execute
            algorithms_to_execute = []
            if request_data.get('algorithms') and len(request_data.get('algorithms', [])) > 0:
                algorithms_to_execute = [
                    AlgorithmConfig(**algo) for algo in request_data['algorithms']
                ]
            elif request_data.get('algorithm_id') is not None:
                algorithms_to_execute = [
                    AlgorithmConfig(
                        algorithm_id=request_data['algorithm_id'],
                        execution_order=1,
                        custom_parameters=request_data.get('custom_parameters')
                    )
                ]
            else:
                algorithms_to_execute = [AlgorithmConfig(algorithm_id=999, execution_order=1)]
            
            logger.info(f"Job {job_id}: Executing {len(algorithms_to_execute)} algorithm(s)")
            
            # Create a modified request object for processing
            class ModifiedRequest:
                def __init__(self, data):
                    self.forecast_filters = data.get('forecast_filters', {})
                    self.version_id = data.get('version_id')
            
            modified_request = ModifiedRequest(request_data)
            
            # Execute in parallel
            db_manager = get_db_manager()
            max_workers = min(len(entity_combinations), 10)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_entity = {
                    executor.submit(
                        _process_entity_forecast,
                        entity_filter,
                        tenant_data,
                        modified_request,
                        aggregation_level,
                        interval,
                        selected_factors,
                        forecast_start_date,
                        forecast_end_date,
                        periods,
                        forecast_dates,
                        algorithms_to_execute,
                        db_manager
                    ): entity_filter for entity_filter in entity_combinations
                }
                
                for future in as_completed(future_to_entity):
                    entity_filter = future_to_entity[future]
                    try:
                        entity_result = future.result()
                        forecast_runs.extend(entity_result["results"])
                        total_records += entity_result["total_records"]
                        
                        if entity_result["status"] == "Completed":
                            successful_runs += len([r for r in entity_result["results"] if r["status"] == "Completed"])
                            failed_runs += len([r for r in entity_result["results"] if r["status"] == "Failed"])
                        else:
                            failed_runs += len(entity_result["results"])
                    
                    except Exception as e:
                        entity_name = '-'.join([f"{k}={v}" for k, v in entity_filter.items()]) if entity_filter else "all"
                        logger.error(f"Job {job_id}: Failed to process entity {entity_name}: {str(e)}", exc_info=True)
                        failed_runs += 1
                        
                        forecast_runs.append({
                            "entity": entity_name,
                            "entity_filter": entity_filter,
                            "forecast_run_id": None,
                            "status": "Failed",
                            "error": str(e),
                            "records": 0
                        })
            
            # Prepare result
            result = {
                "job_id": job_id,
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
                        "start": request_data.get('forecast_start'),
                        "end": request_data.get('forecast_end')
                    }
                }
            }
            
            # Update job with results
            ForecastJobService.update_job_status(
                tenant_id=tenant_id,
                database_name=database_name,
                job_id=job_id,
                status=JobStatus.COMPLETED,
                result_data=result
            )
            
            logger.info(f"Job {job_id} completed successfully: {successful_runs}/{len(forecast_runs)} runs successful")
        
        except Exception as e:
            logger.error(f"Job {job_id} failed with error: {str(e)}", exc_info=True)
            
            # Update job status to FAILED
            ForecastJobService.update_job_status(
                tenant_id=tenant_id,
                database_name=database_name,
                job_id=job_id,
                status=JobStatus.FAILED,
                error_message=str(e)
            )
