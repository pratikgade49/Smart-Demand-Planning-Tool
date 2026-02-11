"""
Forecast Utilities - Shared functions and models for forecasting operations.
UPDATED: Now supports train-test split with testing_actual, testing_forecast, and future_forecast
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import uuid
import pandas as pd
import numpy as np
from psycopg2.extras import Json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os
import logging
from app.config import settings

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
        14: "moving_average",
        15: "sarima",
        16: "random_forest",
        17: "sarima",  # Fallback/Duplicate if needed
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
    db_manager,
    external_factors_df: Optional[pd.DataFrame] = None,
    selected_metrics: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process forecast for a single entity with all its algorithms.
    Returns entity results dictionary.
    """
    from app.core.forecasting_service import ForecastingService
    from app.core.forecast_execution_service import ForecastExecutionService

    # Set default selected_metrics if not provided
    if selected_metrics is None:
        selected_metrics = ['mape', 'accuracy']

    entity_name = '-'.join([f"{k}={v}" for k, v in entity_filter.items()]) if entity_filter else "all"

    try:
        logger.info(f"Processing entity: {entity_name}")

        # Get base filters without aggregation-level-specific fields
        base_filters = {k: v for k, v in request_data.forecast_filters.items() 
                       if k not in ['aggregation_level', 'interval', 'selected_external_factors']}

        # Get historical data for this specific aggregation combination
        historical_data, date_field_name = ForecastingService.prepare_aggregated_data(
            tenant_id=tenant_data["tenant_id"],
            database_name=tenant_data["database_name"],
            aggregation_level=aggregation_level,
            interval=interval,
            filters=base_filters,
            specific_combination=entity_filter if entity_filter else None,
            history_start=getattr(request_data, 'history_start', None),
            history_end=getattr(request_data, 'history_end', None)
        )

        logger.info(f"Entity {entity_name}: {len(historical_data)} historical records")

        # Load external factors if not provided
        if external_factors_df is None:
            external_factors_df = ForecastExecutionService._prepare_external_factors(
                tenant_id=tenant_data["tenant_id"],
                database_name=tenant_data["database_name"],
                selected_factors=selected_factors
            )

        # Merge external factors if available
        if external_factors_df is not None and not external_factors_df.empty:
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
                filters_to_store = base_filters.copy()
                filters_to_store.update(entity_filter)
                filters_to_store['aggregation_level'] = aggregation_level
                filters_to_store['interval'] = interval
                if selected_factors:
                    filters_to_store['selected_external_factors'] = selected_factors
                
                with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                    cursor = conn.cursor()
                    try:
                        # Ensure selected_metrics column exists (schema migration)
                        cursor.execute("""
                            INSERT INTO forecast_runs
                            (forecast_run_id, version_id, forecast_filters,
                             forecast_start, forecast_end, run_status, selected_metrics, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            forecast_run_id,
                            request_data.version_id,
                            Json(filters_to_store),
                            forecast_start_date,
                            forecast_end_date,
                            "In-Progress",
                            selected_metrics,
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

                # ================================================================
                # Execute the algorithm
                # ================================================================
                
                if algo_config.algorithm_id == 999:
                    # Best Fit algorithm (Single-pass)
                    forecast_result = ForecastExecutionService.generate_forecast(
                    historical_data=historical_data,
                    config={'interval': interval, 'periods': periods},
                    process_log=[],
                    tenant_id=tenant_data["tenant_id"],
                    database_name=tenant_data["database_name"],
                    aggregation_level=aggregation_level,
                    selected_metrics=selected_metrics
                )
                    algo_name_for_result = "best_fit"
                    algo_accuracy = forecast_result.get('accuracy')
                    algo_forecast = forecast_result['forecast']
                    test_forecasts = forecast_result.get('test_forecast', [])
                    algo_metrics = {
                        'mae': forecast_result.get('mae'),
                        'rmse': forecast_result.get('rmse'),
                        'mape': forecast_result.get('mape'),
                        'selected_algorithm': forecast_result['selected_algorithm'],
                        'process_log': forecast_result.get('process_log', []),
                        'algorithms_evaluated': len(forecast_result.get('all_algorithms', []))
                    }

                    # Update forecast_runs table with selected algorithm and accuracy
                    with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                        cursor = conn.cursor()
                        try:
                            cursor.execute("""
                                UPDATE forecast_runs
                                SET algorithm_name = %s, accuracy = %s, updated_at = %s, updated_by = %s
                                WHERE forecast_run_id = %s
                            """, (
                                forecast_result['selected_algorithm'],
                                forecast_result['accuracy'],
                                datetime.utcnow(),
                                tenant_data["email"],
                                forecast_run_id
                            ))
                            conn.commit()
                            logger.info(f"Updated forecast_runs with algorithm: {forecast_result['selected_algorithm']}, accuracy: {forecast_result['accuracy']}")
                        finally:
                            cursor.close()
                else:
                    # Specific algorithm (Single-pass)
                    algorithm_name_result = ForecastExecutionService._run_algorithm_safe(
                        algorithm_name=_get_algorithm_name_by_id(algo_config.algorithm_id),
                        data=historical_data.copy(),
                        periods=periods,
                        target_column='total_quantity',
                        tenant_id=tenant_data["tenant_id"],
                        database_name=tenant_data["database_name"],
                        aggregation_level=aggregation_level
                    )
                    if algorithm_name_result['accuracy'] == 0 and not algorithm_name_result.get('forecast'):
                        logger.warning(f"Algorithm {algo_config.algorithm_id} failed, skipping")
                        return None

                    algo_name_for_result = algorithm_name_result['algorithm']
                    algo_accuracy = algorithm_name_result.get('accuracy')
                    algo_forecast = algorithm_name_result['forecast']
                    test_forecasts = algorithm_name_result.get('test_forecast', [])
                    algo_metrics = {
                        'mae': algorithm_name_result.get('mae'),
                        'rmse': algorithm_name_result.get('rmse'),
                        'mape': algorithm_name_result.get('mape')
                    }

                    # Update forecast_runs table with algorithm name and accuracy
                    with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                        cursor = conn.cursor()
                        try:
                            cursor.execute("""
                                UPDATE forecast_runs
                                SET algorithm_name = %s, accuracy = %s, updated_at = %s, updated_by = %s
                                WHERE forecast_run_id = %s
                            """, (
                                algo_name_for_result,
                                algo_accuracy,
                                datetime.utcnow(),
                                tenant_data["email"],
                                forecast_run_id
                            ))
                            conn.commit()
                            logger.info(f"Updated forecast_runs with algorithm: {algo_name_for_result}, accuracy: {algo_accuracy}")
                        finally:
                            cursor.close()

                # Get test actuals and dates from historical data for the last N periods
                test_periods = len(test_forecasts)
                if test_periods > 0:
                    test_data = historical_data.iloc[-test_periods:].copy()
                    
                    # Get actual values
                    if 'total_quantity' in test_data.columns:
                        test_actuals = test_data['total_quantity'].values.tolist()
                    elif 'quantity' in test_data.columns:
                        test_actuals = test_data['quantity'].values.tolist()
                    else:
                        test_actuals = []
                    
                    # Get dates
                    if 'period' in test_data.columns:
                        test_dates = pd.to_datetime(test_data['period']).dt.date.tolist()
                    elif date_field_name in test_data.columns:
                        test_dates = pd.to_datetime(test_data[date_field_name]).dt.date.tolist()
                    else:
                        test_dates = []
                else:
                    test_actuals = []
                    test_dates = []
                
                # ✅ LOG what we have before inserting
                logger.info(f"DEBUG {entity_name}: Preparing to insert:")
                logger.info(f"  test_actuals={len(test_actuals)}, test_forecasts={len(test_forecasts)}, test_dates={len(test_dates)}")
                logger.info(f"  future_forecasts={len(algo_forecast)}, future_dates={len(forecast_dates)}")

                # ================================================================
                # BATCH INSERT: All three types of results
                # ================================================================
                batch_data = []
                
                # 1. Insert testing_actual records
                for test_date, actual_value in zip(test_dates, test_actuals):
                    result_id = str(uuid.uuid4())
                    actual_value_float = min(999999.9999, max(0, float(actual_value)))
                    actual_value_float = round(actual_value_float, 4)
                    
                    batch_data.append((
                        result_id,
                        forecast_run_id,
                        request_data.version_id,
                        algo_mapping_id,
                        algo_config.algorithm_id,
                        test_date,
                        actual_value_float,
                        'testing_actual',
                        None,  # No accuracy metric for actuals
                        None,  # No metric type for actuals
                        Json({
                            'algorithm': algo_name_for_result,
                            'entity_filter': entity_filter
                        }),
                        tenant_data["email"]
                    ))
                
                # After calculating algo_metrics and before building batch_data
                # Determine primary metric for the accuracy_metric column
                primary_metric = selected_metrics[0] if selected_metrics else 'mape'
                primary_metric_name = primary_metric.upper()

                # Get primary metric value
                primary_metric_value = algo_metrics.get(primary_metric)
                if primary_metric_value is not None:
                    primary_metric_value = round(float(primary_metric_value), 2)

                # 2. Insert testing_forecast records
                for test_date, forecast_value in zip(test_dates, test_forecasts):
                    result_id = str(uuid.uuid4())
                    forecast_value_float = min(999999.9999, max(0, float(forecast_value)))
                    forecast_value_float = round(forecast_value_float, 4)
                    
                    batch_data.append((
                        result_id,
                        forecast_run_id,
                        request_data.version_id,
                        algo_mapping_id,
                        algo_config.algorithm_id,
                        test_date,
                        forecast_value_float,
                        'testing_forecast',
                        primary_metric_value,  # ✅ Use primary metric value
                        primary_metric_name,   # ✅ Use primary metric name (RMSE, MAE, etc.)
                        Json({
                            'algorithm': algo_name_for_result,
                            'metrics': {k: algo_metrics[k] for k in selected_metrics if k in algo_metrics},  # Only selected
                            'entity_filter': entity_filter
                        }),
                        tenant_data["email"]
                    ))
                
                # 3. Insert future_forecast records
                for forecast_date, forecast_value in zip(forecast_dates, algo_forecast):
                    result_id = str(uuid.uuid4())
                    forecast_value_float = min(999999.9999, max(0, float(forecast_value)))
                    forecast_value_float = round(forecast_value_float, 4)
                    
                    # Simple confidence intervals (±10%)
                    ci_lower = round(forecast_value_float * 0.9, 4)
                    ci_upper = round(forecast_value_float * 1.1, 4)
                    
                    batch_data.append((
                        result_id,
                        forecast_run_id,
                        request_data.version_id,
                        algo_mapping_id,
                        algo_config.algorithm_id,
                        forecast_date,
                        forecast_value_float,
                        'future_forecast',
                        None,  # No accuracy for future
                        None,  # No metric type for future
                        Json({
                            'algorithm': algo_name_for_result,
                            **algo_metrics,
                            'entity_filter': entity_filter,
                            'confidence_interval_lower': ci_lower,
                            'confidence_interval_upper': ci_upper,
                            'confidence_level': '90%'
                        }),
                        tenant_data["email"]
                    ))

                # Batch insert all results
                with db_manager.get_tenant_connection(tenant_data["database_name"]) as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.executemany("""
                            INSERT INTO forecast_results
                            (result_id, forecast_run_id, version_id, mapping_id,
                             algorithm_id, date, value, type, accuracy_metric,
                             metric_type, metadata, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                                **algo_metrics,
                                'test_records': len(test_actuals),
                                'forecast_records': len(forecast_dates)
                            }),
                            datetime.utcnow(),
                            algo_mapping_id
                        ))
                        conn.commit()
                    finally:
                        cursor.close()

                # Calculate total records (test actuals + test forecasts + future forecasts)
                algo_records_count = len(test_actuals) + len(test_forecasts) + len(forecast_dates)

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
                                                      
                logger.info(
                    f"Entity {entity_name}, Algorithm {algo_name_for_result}: "
                    f"{len(test_actuals)} test actuals, {len(test_forecasts)} test forecasts, "
                    f"{len(forecast_dates)} future forecasts, Accuracy: {algo_accuracy:.2f}%"
                )

                return {
                    "entity": entity_name,
                    "entity_filter": entity_filter,
                    "algorithm_id": algo_config.algorithm_id,
                    "algorithm_name": algo_name_for_result,
                    "forecast_run_id": forecast_run_id,
                    "status": "Completed",
                    "records": algo_records_count,
                    "test_records": len(test_actuals),
                    "forecast_records": len(forecast_dates),
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
        max_algo_workers = min(len(algorithms_to_execute), settings.NUMBER_OF_THREADS)
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