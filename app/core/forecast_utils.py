"""
Forecast Utilities - Shared functions and models for forecasting operations.
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import uuid
import pandas as pd
from psycopg2.extras import Json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import logging

logger = logging.getLogger(__name__)


class AlgorithmConfig(BaseModel):
    algorithm_id: int
    execution_order: int
    custom_parameters: Optional[Dict[str, Any]] = None


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


def _process_entity_forecast(
    entity_filter: Dict[str, Any],
    tenant_data: Dict[str, Any],
    request_data,
    aggregation_level: str,
    interval: str,
    selected_factors: Optional[List[str]],
    forecast_start_date: date,
    forecast_end_date: date,
    periods: int,
    forecast_dates: List[date],
    algorithms_to_execute: List[AlgorithmConfig],
    db_manager
) -> Dict[str, Any]:
    """
    Process forecast for a single entity with all its algorithms.
    Returns entity results dictionary.
    """
    from app.core.forecasting_service import ForecastingService
    from app.core.forecast_execution_service import ForecastExecutionService

    entity_name = '-'.join([f"{k}={v}" for k, v in entity_filter.items()]) if entity_filter else "all"

    try:
        logger.info(f"Processing entity: {entity_name}")

        # Create specific filter for this entity
        entity_specific_filters = request_data.forecast_filters.copy()
        entity_specific_filters.update(entity_filter)

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

        # Execute algorithms for this entity
        entity_results = []
        entity_records_count = 0

        # Execute algorithms in parallel within each entity
        def _execute_algorithm_for_entity(algo_config):
            """Execute a single algorithm for the entity and return results."""
            forecast_run_id = str(uuid.uuid4())
            algo_mapping_id = str(uuid.uuid4())

            try:
                # Create forecast run
                with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            INSERT INTO forecast_runs
                            (forecast_run_id, version_id, forecast_filters,
                             forecast_start, forecast_end, run_status, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            forecast_run_id,
                            request_data.version_id,
                            Json(entity_specific_filters),
                            forecast_start_date,
                            forecast_end_date,
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
                            (mapping_id, forecast_run_id, algorithm_id,
                             algorithm_name, custom_parameters, execution_order, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            algo_mapping_id,
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
                        process_log=[]
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
                        return None

                    algo_name_for_result = algorithm_name_result['algorithm']
                    algo_accuracy = algorithm_name_result.get('accuracy')
                    algo_forecast = algorithm_name_result['forecast']
                    algo_metrics = {
                        'mae': algorithm_name_result.get('mae'),
                        'rmse': algorithm_name_result.get('rmse'),
                        'mape': algorithm_name_result.get('mape')
                    }

                # Batch insert forecast results
                batch_data = []
                for forecast_date, forecast_value in zip(forecast_dates, algo_forecast):
                    result_id = str(uuid.uuid4())
                    forecast_value_float = min(999999.9999, max(0, float(forecast_value)))
                    forecast_value_float = round(forecast_value_float, 4)
                    accuracy_metric = round(algo_accuracy, 2) if algo_accuracy else None

                    batch_data.append((
                        result_id,
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

                # Batch insert results
                with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.executemany("""
                            INSERT INTO forecast_results
                            (result_id, forecast_run_id, version_id, mapping_id,
                             algorithm_id, forecast_date, forecast_quantity, accuracy_metric,
                             metric_type, metadata, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, batch_data)
                        conn.commit()
                    finally:
                        cursor.close()

                # Update algorithm mapping
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
                        conn.commit()
                    finally:
                        cursor.close()

                algo_records_count = len(forecast_dates)

                # Mark forecast run as completed
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

                logger.info(f"Entity {entity_name}, Algorithm {algo_name_for_result}: {algo_records_count} forecast records, Accuracy: {algo_accuracy:.2f}%")

                return {
                    "entity": entity_name,
                    "entity_filter": entity_filter,
                    "algorithm_id": algo_config.algorithm_id,
                    "algorithm_name": algo_name_for_result,
                    "forecast_run_id": forecast_run_id,
                    "status": "Completed",
                    "records": algo_records_count,
                    "accuracy": round(algo_accuracy, 2) if algo_accuracy else None
                }

            except Exception as e:
                logger.error(f"Failed to execute algorithm {algo_config.algorithm_id} for entity {entity_name}: {str(e)}", exc_info=True)

                # Mark as failed
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

                return {
                    "entity": entity_name,
                    "entity_filter": entity_filter,
                    "algorithm_id": algo_config.algorithm_id,
                    "forecast_run_id": forecast_run_id if 'forecast_run_id' in locals() else None,
                    "status": "Failed",
                    "error": str(e),
                    "records": 0,
                    "accuracy": None
                }

        # Use ThreadPoolExecutor for parallel algorithm execution within entity
        max_algo_workers = min(len(algorithms_to_execute), os.cpu_count() or 4)
        logger.info(f"Starting parallel execution of {len(algorithms_to_execute)} algorithms for entity {entity_name} with {max_algo_workers} workers")

        with ThreadPoolExecutor(max_workers=max_algo_workers) as algo_executor:
            # Submit all algorithm tasks
            future_to_algo = {
                algo_executor.submit(_execute_algorithm_for_entity, algo_config): algo_config
                for algo_config in sorted(algorithms_to_execute, key=lambda x: x.execution_order)
            }

            # Collect results as they complete
            for future in as_completed(future_to_algo):
                algo_config = future_to_algo[future]
                try:
                    algo_result = future.result()
                    if algo_result:  # Only add successful results
                        entity_results.append(algo_result)
                        entity_records_count += algo_result.get("records", 0)
                except Exception as e:
                    logger.error(f"Algorithm {algo_config.algorithm_id} execution failed: {str(e)}", exc_info=True)
                    entity_results.append({
                        "entity": entity_name,
                        "entity_filter": entity_filter,
                        "algorithm_id": algo_config.algorithm_id,
                        "forecast_run_id": None,
                        "status": "Failed",
                        "error": str(e),
                        "records": 0,
                        "accuracy": None
                    })

        logger.info(f"Entity {entity_name} completed with {len(algorithms_to_execute)} algorithm(s): {entity_records_count} forecast records")

        return {
            "entity_name": entity_name,
            "entity_filter": entity_filter,
            "results": entity_results,
            "total_records": entity_records_count,
            "status": "Completed"
        }

    except Exception as e:
        logger.error(f"Failed to process entity {entity_name}: {str(e)}", exc_info=True)
        return {
            "entity_name": entity_name,
            "entity_filter": entity_filter,
            "results": [],
            "total_records": 0,
            "status": "Failed",
            "error": str(e)
        }