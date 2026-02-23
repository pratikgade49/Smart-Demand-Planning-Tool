"""
Background Task Executor - Handles async forecast execution with resource monitoring.
Executes forecasts in background and updates job status with performance metrics.
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
from app.core.aggregation_service import AggregationService
from app.core.forecast_execution_service import ForecastExecutionService
from app.core.resource_monitor import ResourceMonitor, performance_tracker
from app.core.algorithm_parameters import AlgorithmParametersService
from app.core.exceptions import ValidationException
from app.config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


def _build_entity_label(entity_filter: dict, aggregation_level: str) -> str:
    """Build a label for an entity that matches the label from forecasting_routes._combo_label."""
    dimensions = [dim.strip() for dim in aggregation_level.split('-')]
    parts = []
    for dim in dimensions:
        val = entity_filter.get(dim)
        if val:
            parts.append(f"{dim}={val[0] if isinstance(val, list) else val}")
    return '-'.join(parts) if parts else 'unknown'


class BackgroundForecastExecutor:
    """Executes forecast jobs in the background with resource monitoring."""

    @staticmethod
    def execute_forecast_async(
        job_id: str,
        tenant_id: str,
        database_name: str,
        request_data: Dict[str, Any],
        tenant_data: Dict[str, Any]
    ) -> None:
        """
        Execute forecast asynchronously and update job status with monitoring.
        This method is called by FastAPI's BackgroundTasks.
        
        Args:
            job_id: Unique job identifier
            tenant_id: Tenant identifier
            database_name: Database name for tenant
            request_data: The forecast request data
            tenant_data: Tenant data dictionary
        """
        
        # Use resource monitoring context
        with ResourceMonitor.monitor_operation(
            f"Forecast Job {job_id}",
            warn_threshold=30.0  # Forecast operations can take longer
        ) as monitor_context:
            
            try:
                # Get initial resources
                start_resources = monitor_context['start_resources']
                
                # Update status to RUNNING with resource info
                ForecastJobService.update_job_status(
                    tenant_id=tenant_id,
                    database_name=database_name,
                    job_id=job_id,
                    status=JobStatus.RUNNING,
                    metadata={
                        'start_resources': start_resources,
                        'started_at': datetime.utcnow().isoformat()
                    }
                )
                
                logger.info(
                    f"Starting background forecast execution for job {job_id}",
                    extra={
                        'job_id': job_id,
                        'start_resources': start_resources
                    }
                )
                
                # Extract configuration
                aggregation_level = request_data.get('forecast_filters', {}).get('aggregation_level', 'product')
                interval = request_data.get('forecast_filters', {}).get('interval', 'MONTHLY')
                selected_factors = request_data.get('forecast_filters', {}).get('selected_external_factors')
                
                # Get distinct aggregation combinations from filtered data
                filters_for_data = {k: v for k, v in request_data.get('forecast_filters', {}).items() 
                                   if k not in ['aggregation_level', 'interval', 'selected_external_factors']}
                
                entity_combinations = AggregationService.get_aggregation_combinations(
                    tenant_id=tenant_id,
                    database_name=database_name,
                    aggregation_level=aggregation_level,
                    filters=filters_for_data if filters_for_data else None
                )
                
                logger.info(
                    f"Job {job_id}: Detected {len(entity_combinations)} distinct entity combinations from database"
                )
                
                # Load external factors ONCE for the entire job
                external_factors_df = ForecastExecutionService._prepare_external_factors(
                    tenant_id=tenant_id,
                    database_name=database_name,
                    selected_factors=selected_factors
                )
                logger.info(f"Job {job_id}: Loaded external factors once for all entities")

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

                # Validate algorithm parameters
                for algo_config in algorithms_to_execute:
                    if algo_config.algorithm_id != 999:  # Skip validation for Best Fit
                        validation_result = AlgorithmParametersService.validate_parameters(
                            algorithm_id=algo_config.algorithm_id,
                            custom_parameters=algo_config.custom_parameters
                        )
                        if not validation_result.is_valid:
                            error_msg = f"Invalid parameters for algorithm {algo_config.algorithm_id}: {'; '.join(validation_result.errors)}"
                            logger.error(f"Job {job_id}: {error_msg}")
                            raise ValidationException(error_msg)

                logger.info(f"Job {job_id}: Executing {len(algorithms_to_execute)} algorithm(s)")
                
                # Pre-load external factors once to avoid redundant database calls for each entity
                logger.info(f"Job {job_id}: Pre-loading external factors...")
                external_factors_df = ForecastExecutionService._prepare_external_factors(
                    tenant_id=tenant_id,
                    database_name=database_name,
                    selected_factors=selected_factors
                )

                # Create a modified request object for processing
                class ModifiedRequest:
                    def __init__(self, data):
                        self.forecast_filters = data.get('forecast_filters', {})
                        self.version_id = data.get('version_id')
                        self.history_start = data.get('history_start')
                        self.history_end = data.get('history_end')
                        self.selected_metrics = data.get('selected_metrics', ['mape', 'accuracy'])
                
                modified_request = ModifiedRequest(request_data)
                
                # Get per-entity metrics mapping (if available) for 'auto' metric selection
                per_entity_metrics = request_data.get('selected_metrics_per_entity', None)
                
                # Execute in parallel with monitoring
                db_manager = get_db_manager()
                max_workers = min(len(entity_combinations), settings.NUMBER_OF_THREADS)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_entity = {}
                    for entity_filter in entity_combinations:
                        # Determine metrics for this specific entity
                        if per_entity_metrics:
                            entity_label = _build_entity_label(entity_filter, aggregation_level)
                            entity_metrics = per_entity_metrics.get(entity_label, ['mape', 'accuracy'])
                        else:
                            entity_metrics = request_data.get('selected_metrics', ['mape', 'accuracy'])
                        
                        future = executor.submit(
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
                            db_manager,
                            external_factors_df,
                            entity_metrics
                        )
                        future_to_entity[future] = entity_filter
                    
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
                                # Count as failure if the entire entity processing failed
                                failed_runs += max(1, len(entity_result.get("results", [])))
                        
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
                
                # Get end resources
                end_resources = ResourceMonitor.get_system_resources()
                duration = datetime.utcnow().timestamp() - datetime.fromisoformat(
                    start_resources['timestamp']
                ).timestamp()
                
                # Check for resource warnings
                resource_warnings = ResourceMonitor.check_resource_warnings(end_resources)
                
                # Record performance
                performance_tracker.record_operation(
                    f"Forecast Job",
                    duration,
                    start_resources,
                    end_resources
                )
                
                # Prepare result with monitoring data
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
                    },
                    "performance_metrics": {
                        "duration_seconds": round(duration, 2),
                        "start_resources": start_resources,
                        "end_resources": end_resources,
                        "resource_warnings": resource_warnings,
                        "is_slow": duration > ResourceMonitor.SLOW_REQUEST_THRESHOLD,
                        "is_very_slow": duration > ResourceMonitor.VERY_SLOW_REQUEST_THRESHOLD
                    }
                }
                
                # Determine final status - FAIL if everything failed, otherwise COMPLETE
                final_status = JobStatus.COMPLETED
                error_message = None
                
                if len(entity_combinations) > 0 and successful_runs == 0:
                    final_status = JobStatus.FAILED
                    error_message = f"Forecast failed for all {len(entity_combinations)} entities. Check individual run logs for details."
                elif len(entity_combinations) == 0:
                    final_status = JobStatus.FAILED
                    error_message = "No data found for the selected filters. No forecast generated."

                # Update job with results
                ForecastJobService.update_job_status(
                    tenant_id=tenant_id,
                    database_name=database_name,
                    job_id=job_id,
                    status=final_status,
                    result_data=result,
                    error_message=error_message
                )
                
                logger.info(
                    f"Job {job_id} {final_status.value}: {successful_runs}/{len(forecast_runs)} runs successful",
                    extra={
                        'job_id': job_id,
                        'status': final_status.value,
                        'duration_seconds': round(duration, 2),
                        'end_resources': end_resources,
                        'resource_warnings': resource_warnings
                    }
                )
                
                # Log performance warning if slow
                if duration > ResourceMonitor.VERY_SLOW_REQUEST_THRESHOLD:
                    logger.warning(
                        f"VERY SLOW FORECAST: Job {job_id} took {duration:.2f}s "
                        f"(threshold: {ResourceMonitor.VERY_SLOW_REQUEST_THRESHOLD}s)"
                    )
                elif duration > ResourceMonitor.SLOW_REQUEST_THRESHOLD:
                    logger.warning(
                        f"Slow forecast: Job {job_id} took {duration:.2f}s "
                        f"(threshold: {ResourceMonitor.SLOW_REQUEST_THRESHOLD}s)"
                    )
            
            except Exception as e:
                logger.error(f"Job {job_id} failed with error: {str(e)}", exc_info=True)
                
                # Get resources at failure
                failure_resources = ResourceMonitor.get_system_resources()
                
                # Update job status to FAILED with monitoring data
                ForecastJobService.update_job_status(
                    tenant_id=tenant_id,
                    database_name=database_name,
                    job_id=job_id,
                    status=JobStatus.FAILED,
                    error_message=str(e),
                    metadata={
                        'failure_resources': failure_resources,
                        'failed_at': datetime.utcnow().isoformat()
                    }
                )