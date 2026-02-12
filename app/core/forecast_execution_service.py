"""
Forecast Execution Service - Core forecasting engine.
Orchestrates algorithm execution and stores results.
"""

import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import logging
import threading
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
try:
    from prophet import Prophet
except ImportError:
    Prophet = None
from psycopg2.extras import Json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import psutil
import time

from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, ValidationException, NotFoundException
from app.core.forecasting_service import ForecastingService
from app.core.aggregation_service import AggregationService
from app.core.external_factors_service import ExternalFactorsService
from app.config import settings

logger = logging.getLogger(__name__)


class ForecastExecutionService:
    """Service for executing forecast algorithms and storing results."""

    @staticmethod
    def validate_algorithm_parameters(algorithm_id: int, custom_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Centralized parameter validation for all forecasting algorithms.

        Args:
            algorithm_id: The algorithm identifier
            custom_params: Dictionary of custom parameters

        Returns:
            Validated and sanitized parameters dictionary

        Raises:
            ValidationException: If parameters are invalid
        """
        validated_params = custom_params.copy()

        try:
            if algorithm_id == 1:  # ARIMA
                # order: [p, d, q] - integers >= 0
                if 'order' in validated_params:
                    order = validated_params['order']
                    if not isinstance(order, list) or len(order) != 3:
                        raise ValidationException("ARIMA order must be a list of 3 integers")
                    validated_params['order'] = [max(0, min(10, int(x))) for x in order]
                
                # seasonal_order: [P, D, Q, s]
                if 'seasonal_order' in validated_params:
                    s_order = validated_params['seasonal_order']
                    if not isinstance(s_order, list) or len(s_order) != 4:
                        raise ValidationException("ARIMA seasonal_order must be a list of 4 integers")
                    validated_params['seasonal_order'] = [max(0, min(12, int(x))) for x in s_order]

            elif algorithm_id == 2:  # Linear Regression
                if 'fit_intercept' in validated_params:
                    validated_params['fit_intercept'] = bool(validated_params['fit_intercept'])

            elif algorithm_id == 3:  # Polynomial Regression
                # degree: 1-5
                if 'degree' in validated_params:
                    degree = validated_params['degree']
                    validated_params['degree'] = max(1, min(5, int(degree)))
                    logger.info(f"Validated polynomial degree: {validated_params['degree']}")

            elif algorithm_id == 7:  # Prophet
                if 'window' in validated_params:
                    validated_params['window'] = max(1, min(30, int(validated_params['window'])))
                if 'changepoint_prior_scale' in validated_params:
                    validated_params['changepoint_prior_scale'] = max(0.001, min(0.5, float(validated_params['changepoint_prior_scale'])))

            elif algorithm_id == 12:  # Gaussian Process
                if 'alpha' in validated_params:
                    validated_params['alpha'] = max(1e-15, min(1.0, float(validated_params['alpha'])))

            elif algorithm_id == 13:  # MLP Neural Network
                if 'hidden_layers' in validated_params:
                    if validated_params['hidden_layers'] is None:
                        validated_params['hidden_layers'] = [64, 32]
                    elif isinstance(validated_params['hidden_layers'], list):
                        validated_params['hidden_layers'] = [max(1, min(100, int(x))) for x in validated_params['hidden_layers']]
                if 'epochs' in validated_params:
                    validated_params['epochs'] = max(1, min(1000, int(validated_params['epochs'])))
                if 'batch_size' in validated_params:
                    validated_params['batch_size'] = max(1, min(256, int(validated_params['batch_size'])))

            elif algorithm_id == 4:  # Exponential Smoothing
                # alphas: list of floats 0.0-1.0
                if 'alphas' in validated_params:
                    alphas = validated_params['alphas']
                    if isinstance(alphas, list):
                        validated_params['alphas'] = [max(0.0, min(1.0, float(a))) for a in alphas]
                    else:
                        validated_params['alphas'] = [max(0.0, min(1.0, float(alphas)))]
                elif 'alpha' in validated_params:
                    validated_params['alpha'] = max(0.0, min(1.0, float(validated_params['alpha'])))
                else:
                    # Default to more alpha values for better optimization
                    validated_params['alphas'] = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

            elif algorithm_id == 5:  # Enhanced Exponential Smoothing
                # alphas: list of floats 0.0-1.0
                if 'alphas' in validated_params:
                    alphas = validated_params['alphas']
                    if isinstance(alphas, list):
                        validated_params['alphas'] = [max(0.0, min(1.0, float(a))) for a in alphas]
                    else:
                        validated_params['alphas'] = [max(0.0, min(1.0, float(alphas)))]
                else:
                    # Replace existing alpha values with more comprehensive set
                    validated_params['alphas'] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

            elif algorithm_id == 6:  # Holt Winters
                # alpha, beta, gamma: 0.0-1.0
                # season_length: positive integer
                if 'alpha' in validated_params:
                    validated_params['alpha'] = max(0.0, min(1.0, float(validated_params['alpha'])))
                if 'beta' in validated_params:
                    validated_params['beta'] = max(0.0, min(1.0, float(validated_params['beta'])))
                if 'gamma' in validated_params:
                    validated_params['gamma'] = max(0.0, min(1.0, float(validated_params['gamma'])))
                if 'season_length' in validated_params:
                    validated_params['season_length'] = max(2, min(365, int(validated_params['season_length'])))

            elif algorithm_id == 8:  # LSTM
                if 'window' in validated_params:
                    validated_params['window'] = max(1, min(50, int(validated_params['window'])))

            elif algorithm_id == 9:  # XGBoost
                # n_estimators: 10-1000
                # max_depth: 1-20
                # learning_rate: 0.01-1.0
                if 'n_estimators' in validated_params:
                    validated_params['n_estimators'] = max(10, min(1000, int(validated_params['n_estimators'])))
                if 'max_depth' in validated_params:
                    validated_params['max_depth'] = max(1, min(20, int(validated_params['max_depth'])))
                if 'learning_rate' in validated_params:
                    validated_params['learning_rate'] = max(0.01, min(1.0, float(validated_params['learning_rate'])))

            elif algorithm_id == 10:  # SVR
                # C: 0.1-100.0
                # epsilon: 0.01-1.0
                # kernel: valid kernel types
                if 'C' in validated_params:
                    validated_params['C'] = max(0.1, min(100.0, float(validated_params['C'])))
                if 'epsilon' in validated_params:
                    validated_params['epsilon'] = max(0.01, min(1.0, float(validated_params['epsilon'])))
                if 'kernel' in validated_params:
                    valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
                    if validated_params['kernel'] not in valid_kernels:
                        validated_params['kernel'] = 'rbf'

            elif algorithm_id == 11:  # KNN
                # n_neighbors: positive integer
                if 'n_neighbors' in validated_params:
                    validated_params['n_neighbors'] = max(1, min(50, int(validated_params['n_neighbors'])))

            elif algorithm_id == 16:  # Random Forest
                # n_estimators: 10-500
                # max_depth: 1-50 or None
                # min_samples_split: 2-20
                # min_samples_leaf: 1-10
                if 'n_estimators' in validated_params:
                    validated_params['n_estimators'] = max(10, min(500, int(validated_params['n_estimators'])))
                if 'max_depth' in validated_params:
                    if validated_params['max_depth'] is not None:
                        validated_params['max_depth'] = max(1, min(50, int(validated_params['max_depth'])))
                if 'min_samples_split' in validated_params:
                    validated_params['min_samples_split'] = max(2, min(20, int(validated_params['min_samples_split'])))
                if 'min_samples_leaf' in validated_params:
                    validated_params['min_samples_leaf'] = max(1, min(10, int(validated_params['min_samples_leaf'])))

            elif algorithm_id == 14:  # Moving Average
                # window: 1-50
                if 'window' in validated_params:
                    validated_params['window'] = max(1, min(50, int(validated_params['window'])))

            elif algorithm_id == 15:  # SARIMA
                # order: [p, d, q] - integers >= 0
                if 'order' in validated_params:
                    order = validated_params['order']
                    if not isinstance(order, list) or len(order) != 3:
                        raise ValidationException("SARIMA order must be a list of 3 integers")
                    validated_params['order'] = [max(0, min(10, int(x))) for x in order]

                # seasonal_order: [P, D, Q, s]
                if 'seasonal_order' in validated_params:
                    s_order = validated_params['seasonal_order']
                    if not isinstance(s_order, list) or len(s_order) != 4:
                        raise ValidationException("SARIMA seasonal_order must be a list of 4 integers")
                    validated_params['seasonal_order'] = [max(0, min(12, int(x))) for x in s_order]

            # Log validated parameters
            logger.info(f"Validated parameters for algorithm {algorithm_id}: {validated_params}")

            return validated_params

        except (ValueError, TypeError) as e:
            raise ValidationException(f"Invalid parameter format for algorithm {algorithm_id}: {str(e)}")

    @staticmethod
    def _get_system_resources() -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_percent = memory.percent
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': round(memory_mb, 2),
                'memory_percent': memory_percent
            }
        except Exception as e:
            logger.warning(f"Could not get system resources: {str(e)}")
            return {'cpu_percent': 0.0, 'memory_mb': 0.0, 'memory_percent': 0.0}

    @staticmethod
    def _execute_algorithm_parallel(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str,
        algorithm_mapping: Dict[str, Any],
        historical_data: pd.DataFrame,
        external_factors: pd.DataFrame,
        forecast_start: str,
        forecast_end: str,
        interval: str,
        user_email: str,
        selected_metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a single algorithm and store results (parallel version).
        This is a wrapper around _execute_algorithm for parallel execution.
        """
        return ForecastExecutionService._execute_algorithm(
            tenant_id=tenant_id,
            database_name=database_name,
            forecast_run_id=forecast_run_id,
            algorithm_mapping=algorithm_mapping,
            historical_data=historical_data,
            external_factors=external_factors,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            interval=interval,
            user_email=user_email,
            selected_metrics=selected_metrics
        )

    @staticmethod
    def execute_forecast_run(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str,
        user_email: str
    ) -> Dict[str, Any]:
        """
        Execute a forecast run with all mapped algorithms.
        """
        start_time = datetime.utcnow()
        initial_resources = ForecastExecutionService._get_system_resources()
        logger.info(f"Starting forecast execution for run: {forecast_run_id}")
        logger.info(f"Initial system resources - CPU: {initial_resources['cpu_percent']}%, Memory: {initial_resources['memory_mb']}MB ({initial_resources['memory_percent']}%)")
 
        db_manager = get_db_manager()
 
        try:
            # Get forecast run details
            forecast_run = ForecastingService.get_forecast_run(
                tenant_id, database_name, forecast_run_id
            )
           
            logger.info(f"Retrieved forecast run details: version_id={forecast_run.get('version_id')}, start={forecast_run.get('forecast_start')}, end={forecast_run.get('forecast_end')}")
 
            # Validate run status
            if forecast_run['run_status'] not in ['Pending', 'Failed']:
                logger.error(f"Invalid run status for forecast run {forecast_run_id}: {forecast_run['run_status']}")
                raise ValidationException(
                    f"Cannot execute forecast run with status: {forecast_run['run_status']}"
                )
 
            # Update status to In-Progress
            logger.debug(f"Updating forecast run status to 'In-Progress' for run: {forecast_run_id}")
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        UPDATE forecast_runs
                        SET run_status = 'In-Progress',
                            started_at = %s,
                            updated_at = %s,
                            updated_by = %s
                        WHERE forecast_run_id = %s
                    """, (datetime.utcnow(), datetime.utcnow(), user_email, forecast_run_id))
                    conn.commit()
                    logger.info(f"Successfully updated forecast run {forecast_run_id} status to 'In-Progress'")
                finally:
                    cursor.close()
           
            # Prepare historical data
            filters = forecast_run.get('forecast_filters', {})
            aggregation_level = filters.get('aggregation_level', 'product')
            interval = filters.get('interval', 'MONTHLY')
            selected_factors = filters.get('selected_external_factors')
            selected_metrics = forecast_run.get('selected_metrics', ['mape', 'accuracy'])
           
            # Get historical data
            historical_data = AggregationService.prepare_aggregated_data(
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level,
                interval=interval,
                filters=filters
            )
 
            logger.info(f"Prepared {len(historical_data)} historical records")
           
            # ✅ FIXED: Get external factors WITHOUT date filtering
            external_factors = ForecastExecutionService._prepare_external_factors(
                tenant_id,
                database_name,
                selected_factors=selected_factors  # ✅ Only pass selected_factors
            )
           
            # Execute each algorithm in parallel
            algorithms = forecast_run.get('algorithms', [])
            total_records = 0
            processed_records = 0
            failed_records = 0
 
            # Use ThreadPoolExecutor for parallel algorithm execution
            max_workers = min(len(algorithms), settings.NUMBER_OF_THREADS)
            logger.info(f"Starting parallel execution of {len(algorithms)} algorithms with {max_workers} workers")
 
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all algorithm tasks
                future_to_algorithm = {}
 
                for algo in algorithms:
                    # Add version_id to algorithm mapping
                    algo['version_id'] = forecast_run['version_id']
 
                    future = executor.submit(
                        ForecastExecutionService._execute_algorithm_parallel,
                        tenant_id=tenant_id,
                        database_name=database_name,
                        forecast_run_id=forecast_run_id,
                        algorithm_mapping=algo,
                        historical_data=historical_data.copy(),
                        external_factors=external_factors,
                        forecast_start=forecast_run['forecast_start'],
                        forecast_end=forecast_run['forecast_end'],
                        interval=interval,
                        user_email=user_email,
                        selected_metrics=selected_metrics
                    )
                    future_to_algorithm[future] = algo
 
                # Collect results as they complete
                for future in as_completed(future_to_algorithm):
                    algo = future_to_algorithm[future]
                    try:
                        results = future.result()
                        total_records += len(results)
                        processed_records += len(results)
                        logger.info(f"Algorithm {algo['algorithm_name']} completed: {len(results)} results")
 
                    except Exception as e:
                        failed_records += 1
                        error_msg = f"Algorithm {algo['algorithm_name']} failed: {str(e)}"
                        logger.error(error_msg, exc_info=True)
 
                        # Update algorithm mapping status
                        ForecastExecutionService._update_algorithm_status(
                            database_name,
                            algo['mapping_id'],
                            'Failed',
                            error_message=str(e)
                        )
           
            # Update forecast run completion
            if failed_records == 0:
                final_status = 'Completed'
            elif processed_records == 0:
                final_status = 'Failed'
            else:
                # Some succeeded, some failed - still mark as Completed
                final_status = 'Completed'
           
            with db_manager.get_tenant_connection(database_name) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        UPDATE forecast_runs
                        SET run_status = %s,
                            run_progress = 100,
                            total_records = %s,
                            processed_records = %s,
                            failed_records = %s,
                            completed_at = %s,
                            updated_at = %s,
                            updated_by = %s
                        WHERE forecast_run_id = %s
                    """, (
                        final_status,
                        total_records,
                        processed_records,
                        failed_records,
                        datetime.utcnow(),
                        datetime.utcnow(),
                        user_email,
                        forecast_run_id
                    ))
                    conn.commit()
                finally:
                    cursor.close()
           
            logger.info(f"Forecast run completed: {forecast_run_id}")
           
            return {
                'forecast_run_id': forecast_run_id,
                'status': final_status,
                'total_records': total_records,
                'processed_records': processed_records,
                'failed_records': failed_records,
                'algorithms_executed': len(algorithms)
            }
           
        except Exception as e:
            # Update run status to Failed
            try:
                with db_manager.get_tenant_connection(database_name) as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            UPDATE forecast_runs
                            SET run_status = 'Failed',
                                error_message = %s,
                                updated_at = %s,
                                updated_by = %s
                            WHERE forecast_run_id = %s
                        """, (str(e), datetime.utcnow(), user_email, forecast_run_id))
                        conn.commit()
                    finally:
                        cursor.close()
            except:
                pass
           
            logger.error(f"Forecast execution failed: {str(e)}")
            raise DatabaseException(f"Forecast execution failed: {str(e)}")
    @staticmethod
    def _execute_algorithm(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str,
        algorithm_mapping: Dict[str, Any],
        historical_data: pd.DataFrame,
        external_factors: pd.DataFrame,
        forecast_start: str,
        forecast_end: str,
        interval: str,
        user_email: str,
        selected_metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a single algorithm with detailed debugging."""

        # Set default selected_metrics if not provided
        if selected_metrics is None:
            selected_metrics = ['mape', 'accuracy']

        mapping_id = algorithm_mapping['mapping_id']
        algorithm_id = algorithm_mapping['algorithm_id']
        algorithm_name = algorithm_mapping['algorithm_name']
        custom_params = algorithm_mapping.get('custom_parameters') or {}

        try:
            custom_params = ForecastExecutionService.validate_algorithm_parameters(algorithm_id, custom_params)
        except Exception as e:
            logger.error(f"Parameter validation failed for algorithm {algorithm_name}: {str(e)}")
            raise

        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING ALGORITHM: {algorithm_name} (ID: {algorithm_id})")
        logger.info(f"Selected Metrics: {selected_metrics}")
        logger.info(f"{'='*80}")

        ForecastExecutionService._update_algorithm_status(database_name, mapping_id, 'Running')

        try:
            forecast_start_date = datetime.fromisoformat(forecast_start).date()
            forecast_end_date = datetime.fromisoformat(forecast_end).date()

            future_periods = ForecastExecutionService._calculate_periods(
                forecast_start_date,
                forecast_end_date,
                interval
            )

            logger.info(f"Future periods to forecast: {future_periods}")

            # Merge external factors
            if not external_factors.empty:
                logger.info(f"Merging {len(external_factors)} external factor records")
                historical_data = historical_data.merge(
                    external_factors,
                    left_on='period',
                    right_on='date',
                    how='left'
                )

            # Get aggregation level from forecast run
            from app.core.forecasting_service import ForecastingService
            forecast_run = ForecastingService.get_forecast_run(
                tenant_id, database_name, forecast_run_id
            )
            aggregation_level = forecast_run.get('forecast_filters', {}).get('aggregation_level')

            # Train-test split
            train_ratio = 0.85
            n = len(historical_data)
            split_index = max(2, int(n * train_ratio))

            train_data = historical_data.iloc[:split_index].copy()
            test_data = historical_data.iloc[split_index:].copy()
            test_periods = len(test_data)

            logger.info(f"\nDATA SPLIT:")
            logger.info(f"  Total historical data: {n} periods")
            logger.info(f"  Training data: {len(train_data)} periods (indices 0-{split_index-1})")
            logger.info(f"  Test data: {len(test_data)} periods (indices {split_index}-{n-1})")
            logger.info(f"  Future forecast: {future_periods} periods")

            # Get test actuals and dates
            if 'total_quantity' in test_data.columns:
                test_actuals = test_data['total_quantity'].values
            elif 'quantity' in test_data.columns:
                test_actuals = test_data['quantity'].values
            else:
                test_actuals = np.array([])

            test_dates = test_data['period'].tolist() if 'period' in test_data.columns else []

            logger.info(f"\nTEST SET ACTUALS:")
            logger.info(f"  Count: {len(test_actuals)}")
            if len(test_actuals) > 0:
                logger.info(f"  Range: [{test_actuals.min():.0f}, {test_actuals.max():.0f}]")
                logger.info(f"  First 3 values: {test_actuals[:3]}")
                logger.info(f"  Last 3 values: {test_actuals[-3:]}")

            # ====================================================================
            # GENERATE TEST FORECAST
            # ====================================================================
            test_forecast = np.array([])
            test_metrics = {}

            if test_periods > 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"GENERATING TEST FORECAST")
                logger.info(f"{'='*80}")
                logger.info(f"Training on: {len(train_data)} periods")
                logger.info(f"Predicting: {test_periods} future periods")

                # Call algorithm with tenant context
                test_forecast_result, train_metrics = ForecastExecutionService._route_algorithm(
                    algorithm_id=algorithm_id,
                    algorithm_name=algorithm_name,
                    data=train_data,
                    periods=test_periods,
                    custom_params=custom_params,
                    tenant_id=tenant_id,
                    database_name=database_name,
                    aggregation_level=aggregation_level
                )

                test_forecast = np.array(test_forecast_result)

                logger.info(f"\nTEST FORECAST RESULT:")
                logger.info(f"  Type: {type(test_forecast)}")
                logger.info(f"  Length: {len(test_forecast)}")
                logger.info(f"  Range: [{test_forecast.min():.0f}, {test_forecast.max():.0f}]")
                logger.info(f"  First 3 values: {test_forecast[:3]}")
                logger.info(f"  Last 3 values: {test_forecast[-3:]}")

                # ⚠️ CRITICAL CHECK
                if len(test_forecast) != test_periods:
                    logger.error(f"❌ LENGTH MISMATCH! Expected {test_periods}, got {len(test_forecast)}")

                # Calculate test metrics using selected metrics
                if len(test_actuals) > 0 and len(test_forecast) > 0:
                    min_len = min(len(test_actuals), len(test_forecast))
                    test_metrics = ForecastExecutionService.calculate_metrics(
                        test_actuals[:min_len],
                        test_forecast[:min_len],
                        selected_metrics=selected_metrics
                    )
                    logger.info(f"\nTEST SET METRICS:")
                    for metric, value in test_metrics.items():
                        logger.info(f"  {metric.upper()}: {value:.2f}{'%' if metric == 'mape' else ''}")
                else:
                    # Use default metrics if no test data available
                    test_metrics = ForecastExecutionService.calculate_metrics(
                        np.array([1.0]), np.array([1.0]), selected_metrics=selected_metrics
                    ) if selected_metrics else {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 100.0}
                    logger.warning("Using default metrics for test set (no actuals available)")

            # ====================================================================
            # GENERATE FUTURE FORECAST
            # ====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"GENERATING FUTURE FORECAST")
            logger.info(f"{'='*80}")
            logger.info(f"Training on: {len(historical_data)} periods (ALL historical data)")
            logger.info(f"Predicting: {future_periods} future periods")

            # Call algorithm with tenant context
            future_forecast_result, future_metrics = ForecastExecutionService._route_algorithm(
                algorithm_id=algorithm_id,
                algorithm_name=algorithm_name,
                data=historical_data,
                periods=future_periods,
                custom_params=custom_params,
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level
            )

            future_forecast = np.array(future_forecast_result)

            logger.info(f"\nFUTURE FORECAST RESULT:")
            logger.info(f"  Type: {type(future_forecast)}")
            logger.info(f"  Length: {len(future_forecast)}")
            logger.info(f"  Range: [{future_forecast.min():.0f}, {future_forecast.max():.0f}]")
            logger.info(f"  First 3 values: {future_forecast[:3]}")
            logger.info(f"  Last 3 values: {future_forecast[-3:]}")

            # Generate future dates
            future_dates = ForecastExecutionService._generate_forecast_dates(
                forecast_start_date,
                future_periods,
                interval
            )

            # ====================================================================
            # STORE RESULTS
            # ====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"STORING RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"Test dates: {len(test_dates)}")
            logger.info(f"Test actuals: {len(test_actuals)}")
            logger.info(f"Test forecast: {len(test_forecast)}")
            logger.info(f"Future dates: {len(future_dates)}")
            logger.info(f"Future forecast: {len(future_forecast)}")

            results = ForecastExecutionService._store_forecast_results_v2(
                tenant_id=tenant_id,
                database_name=database_name,
                forecast_run_id=forecast_run_id,
                version_id=algorithm_mapping.get('version_id'),
                mapping_id=mapping_id,
                algorithm_id=algorithm_id,
                test_dates=test_dates,
                test_actuals=test_actuals,
                test_forecast=test_forecast,
                test_metrics=test_metrics,
                future_dates=future_dates,
                future_forecast=future_forecast,
                user_email=user_email,
                selected_metrics=selected_metrics
            )

            ForecastExecutionService._update_algorithm_status(
                database_name, mapping_id, 'Completed'
            )

            logger.info(f"\n{'='*80}")
            logger.info(f"ALGORITHM {algorithm_name} COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*80}\n")

            return results

        except Exception as e:
            logger.error(f"\n{'='*80}")
            logger.error(f"ALGORITHM {algorithm_name} FAILED")
            logger.error(f"ERROR: {str(e)}")
            logger.error(f"{'='*80}\n", exc_info=True)
            ForecastExecutionService._update_algorithm_status(
                database_name, mapping_id, 'Failed', str(e)
            )
            raise
        """Execute a single algorithm with detailed debugging."""
        
        mapping_id = algorithm_mapping['mapping_id']
        algorithm_id = algorithm_mapping['algorithm_id']
        algorithm_name = algorithm_mapping['algorithm_name']
        custom_params = algorithm_mapping.get('custom_parameters') or {}

        try:
            custom_params = ForecastExecutionService.validate_algorithm_parameters(algorithm_id, custom_params)
        except Exception as e:
            logger.error(f"Parameter validation failed for algorithm {algorithm_name}: {str(e)}")
            raise

        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING ALGORITHM: {algorithm_name} (ID: {algorithm_id})")
        logger.info(f"{'='*80}")

        ForecastExecutionService._update_algorithm_status(database_name, mapping_id, 'Running')
        
        try:
            forecast_start_date = datetime.fromisoformat(forecast_start).date()
            forecast_end_date = datetime.fromisoformat(forecast_end).date()
            
            future_periods = ForecastExecutionService._calculate_periods(
                forecast_start_date,
                forecast_end_date,
                interval
            )
            
            logger.info(f"Future periods to forecast: {future_periods}")
            
            # Merge external factors
            if not external_factors.empty:
                logger.info(f"Merging {len(external_factors)} external factor records")
                historical_data = historical_data.merge(
                    external_factors,
                    left_on='period',
                    right_on='date',
                    how='left'
                )
            
            # Get aggregation level from forecast run
            from app.core.forecasting_service import ForecastingService
            forecast_run = ForecastingService.get_forecast_run(
                tenant_id, database_name, forecast_run_id
            )
            aggregation_level = forecast_run.get('forecast_filters', {}).get('aggregation_level')
            
            # Train-test split
            train_ratio = 0.85
            n = len(historical_data)
            split_index = max(2, int(n * train_ratio))
            
            train_data = historical_data.iloc[:split_index].copy()
            test_data = historical_data.iloc[split_index:].copy()
            test_periods = len(test_data)
            
            logger.info(f"\nDATA SPLIT:")
            logger.info(f"  Total historical data: {n} periods")
            logger.info(f"  Training data: {len(train_data)} periods (indices 0-{split_index-1})")
            logger.info(f"  Test data: {len(test_data)} periods (indices {split_index}-{n-1})")
            logger.info(f"  Future forecast: {future_periods} periods")
            
            # Get test actuals and dates
            if 'total_quantity' in test_data.columns:
                test_actuals = test_data['total_quantity'].values
            elif 'quantity' in test_data.columns:
                test_actuals = test_data['quantity'].values
            else:
                test_actuals = np.array([])
            
            test_dates = test_data['period'].tolist() if 'period' in test_data.columns else []
            
            logger.info(f"\nTEST SET ACTUALS:")
            logger.info(f"  Count: {len(test_actuals)}")
            if len(test_actuals) > 0:
                logger.info(f"  Range: [{test_actuals.min():.0f}, {test_actuals.max():.0f}]")
                logger.info(f"  First 3 values: {test_actuals[:3]}")
                logger.info(f"  Last 3 values: {test_actuals[-3:]}")
            
            # ====================================================================
            # GENERATE TEST FORECAST
            # ====================================================================
            test_forecast = np.array([])
            test_metrics = {}

            if test_periods > 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"GENERATING TEST FORECAST")
                logger.info(f"{'='*80}")
                logger.info(f"Training on: {len(train_data)} periods")
                logger.info(f"Predicting: {test_periods} future periods")
                
                # Call algorithm with tenant context
                test_forecast_result, train_metrics = ForecastExecutionService._route_algorithm(
                    algorithm_id=algorithm_id,
                    algorithm_name=algorithm_name,
                    data=train_data,
                    periods=test_periods,
                    custom_params=custom_params,
                    tenant_id=tenant_id,
                    database_name=database_name,
                    aggregation_level=aggregation_level
                )
                
                test_forecast = np.array(test_forecast_result)
                
                logger.info(f"\nTEST FORECAST RESULT:")
                logger.info(f"  Type: {type(test_forecast)}")
                logger.info(f"  Length: {len(test_forecast)}")
                logger.info(f"  Range: [{test_forecast.min():.0f}, {test_forecast.max():.0f}]")
                logger.info(f"  First 3 values: {test_forecast[:3]}")
                logger.info(f"  Last 3 values: {test_forecast[-3:]}")
                
                # ⚠️ CRITICAL CHECK
                if len(test_forecast) != test_periods:
                    logger.error(f"❌ LENGTH MISMATCH! Expected {test_periods}, got {len(test_forecast)}")
                
                # Calculate test metrics
                if len(test_actuals) > 0 and len(test_forecast) > 0:
                    min_len = min(len(test_actuals), len(test_forecast))
                    test_metrics = ForecastExecutionService.calculate_metrics(
                        test_actuals[:min_len],
                        test_forecast[:min_len],
                        selected_metrics=selected_metrics
                    )
                    logger.info(f"\nTEST SET METRICS:")
                    for metric, value in test_metrics.items():
                        logger.info(f"  {metric.upper()}: {value:.2f}{'%' if metric == 'mape' else ''}")
                else:
                    # Use default metrics if no test data available
                    test_metrics = ForecastExecutionService.calculate_metrics(
                        np.array([1.0]), np.array([1.0]), selected_metrics=selected_metrics
                    ) if selected_metrics else {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 100.0}
                    logger.warning("Using default metrics for test set (no actuals available)")

            # ====================================================================
            # GENERATE FUTURE FORECAST
            # ====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"GENERATING FUTURE FORECAST")
            logger.info(f"{'='*80}")
            logger.info(f"Training on: {len(historical_data)} periods (ALL historical data)")
            logger.info(f"Predicting: {future_periods} future periods")
            
            # Call algorithm with tenant context
            future_forecast_result, future_metrics = ForecastExecutionService._route_algorithm(
                algorithm_id=algorithm_id,
                algorithm_name=algorithm_name,
                data=historical_data,
                periods=future_periods,
                custom_params=custom_params,
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level
            )
            
            future_forecast = np.array(future_forecast_result)
            
            logger.info(f"\nFUTURE FORECAST RESULT:")
            logger.info(f"  Type: {type(future_forecast)}")
            logger.info(f"  Length: {len(future_forecast)}")
            logger.info(f"  Range: [{future_forecast.min():.0f}, {future_forecast.max():.0f}]")
            logger.info(f"  First 3 values: {future_forecast[:3]}")
            logger.info(f"  Last 3 values: {future_forecast[-3:]}")
            
            # Generate future dates
            future_dates = ForecastExecutionService._generate_forecast_dates(
                forecast_start_date,
                future_periods,
                interval
            )
            
            # ====================================================================
            # STORE RESULTS
            # ====================================================================
            logger.info(f"\n{'='*80}")
            logger.info(f"STORING RESULTS")
            logger.info(f"{'='*80}")
            logger.info(f"Test dates: {len(test_dates)}")
            logger.info(f"Test actuals: {len(test_actuals)}")
            logger.info(f"Test forecast: {len(test_forecast)}")
            logger.info(f"Future dates: {len(future_dates)}")
            logger.info(f"Future forecast: {len(future_forecast)}")
            
            results = ForecastExecutionService._store_forecast_results_v2(
                tenant_id=tenant_id,
                database_name=database_name,
                forecast_run_id=forecast_run_id,
                version_id=algorithm_mapping.get('version_id'),
                mapping_id=mapping_id,
                algorithm_id=algorithm_id,
                test_dates=test_dates,
                test_actuals=test_actuals,
                test_forecast=test_forecast,
                test_metrics=test_metrics,
                future_dates=future_dates,
                future_forecast=future_forecast,
                user_email=user_email
            )
            
            ForecastExecutionService._update_algorithm_status(
                database_name, mapping_id, 'Completed'
            )
            
            logger.info(f"\n{'='*80}")
            logger.info(f"ALGORITHM {algorithm_name} COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*80}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"\n{'='*80}")
            logger.error(f"ALGORITHM {algorithm_name} FAILED")
            logger.error(f"ERROR: {str(e)}")
            logger.error(f"{'='*80}\n", exc_info=True)
            ForecastExecutionService._update_algorithm_status(
                database_name, mapping_id, 'Failed', str(e)
            )
            raise

    @staticmethod
    def _route_algorithm(
        algorithm_id: int,
        algorithm_name: str,
        data: pd.DataFrame,
        periods: int,
        custom_params: Dict[str, Any],
        tenant_id: str = None,
        database_name: str = None,
        aggregation_level: str = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Route to appropriate algorithm based on ID.
        
        Args:
            algorithm_id: Algorithm identifier
            algorithm_name: Algorithm name
            data: Historical data
            periods: Number of periods to forecast
            custom_params: Custom parameters for the algorithm
            tenant_id: Tenant identifier (for dynamic field detection)
            database_name: Database name (for dynamic field detection)
            aggregation_level: Current aggregation level (e.g., "product")
            
        Returns:
            Tuple of (forecast_array, metrics_dict)
        """
        
        # All algorithm methods return (forecast_array, metrics_dict)
        if algorithm_id == 1:  # ARIMA
            forecast, metrics = ForecastExecutionService.arima_forecast(
                data=data,
                periods=periods,
                order=custom_params.get('order', [1, 1, 1])
            )
            return forecast, metrics
        elif algorithm_id == 2:  # Linear Regression
            return ForecastExecutionService.linear_regression_forecast(
                data=data,
                periods=periods
            )
        elif algorithm_id == 3:  # Polynomial Regression
            return ForecastExecutionService.polynomial_regression_forecast(
                data=data,
                periods=periods,
                degree=custom_params.get('degree', 2)
            )
        elif algorithm_id == 4:  # Exponential Smoothing
            return ForecastExecutionService.exponential_smoothing_forecast(
                data=data,
                periods=periods,
                alphas=custom_params.get('alphas', [custom_params.get('alpha', 0.3)])
            )
        elif algorithm_id == 5:  # Enhanced Exponential Smoothing
            return ForecastExecutionService.exponential_smoothing_forecast(
                data=data,
                periods=periods,
                alphas=custom_params.get('alphas', [0.1, 0.3, 0.5])
            )
        elif algorithm_id == 6:  # Holt Winters
            return ForecastExecutionService.holt_winters_forecast(
                data=data,
                periods=periods,
                season_length=custom_params.get('season_length', 12),
                alpha=custom_params.get('alpha', 0.3),
                beta=custom_params.get('beta', 0.1),
                gamma=custom_params.get('gamma', 0.1)
            )
        elif algorithm_id == 7:  # Prophet
            return ForecastExecutionService.prophet_forecast(
                data=data,
                periods=periods,
                seasonality_mode=custom_params.get('seasonality_mode', 'additive'),
                changepoint_prior_scale=custom_params.get('changepoint_prior_scale', 0.05)
            )
        elif algorithm_id == 8:  # LSTM
            return ForecastExecutionService.lstm_forecast(
                data=data,
                periods=periods,
                sequence_length=custom_params.get('sequence_length', 12),
                epochs=custom_params.get('epochs', 50),
                batch_size=custom_params.get('batch_size', 32)
            )
        elif algorithm_id == 9:  # XGBoost
            return ForecastExecutionService.xgboost_forecast(
                data=data,
                periods=periods,
                n_estimators=custom_params.get('n_estimators', 100),
                max_depth=custom_params.get('max_depth', 6),
                learning_rate=custom_params.get('learning_rate', 0.1)
            )
        elif algorithm_id == 10:  # SVR
            return ForecastExecutionService.svr_forecast(
                data=data,
                periods=periods,
                C=custom_params.get('C', 1.0),
                epsilon=custom_params.get('epsilon', 0.1),
                kernel=custom_params.get('kernel', 'rbf')
            )
        elif algorithm_id == 11:  # KNN
            return ForecastExecutionService.knn_forecast(
                data=data,
                periods=periods,
                n_neighbors=custom_params.get('n_neighbors', 5)
            )
        elif algorithm_id == 12:  # Gaussian Process
            return ForecastExecutionService.gaussian_process_forecast(
                data=data,
                periods=periods,
                kernel=custom_params.get('kernel', 'RBF'),
                alpha=custom_params.get('alpha', 1e-6)
            )
        elif algorithm_id == 13:  # MLP Neural Network
            return ForecastExecutionService.mlp_neural_network_forecast(
                data=data,
                periods=periods,
                hidden_layers=custom_params.get('hidden_layers', [64, 32]),
                epochs=custom_params.get('epochs', 100),
                batch_size=custom_params.get('batch_size', 32)
            )
        elif algorithm_id == 16:  # Random Forest
            return ForecastExecutionService.random_forest_forecast(
                data=data,
                periods=periods,
                n_estimators=custom_params.get('n_estimators', 100),
                max_depth=custom_params.get('max_depth', None),
                min_samples_split=custom_params.get('min_samples_split', 2),
                min_samples_leaf=custom_params.get('min_samples_leaf', 1)
            )
        elif algorithm_id == 14:  # Moving Average
            return ForecastExecutionService.moving_average_forecast(
                data=data,
                periods=periods,
                window=custom_params.get('window', 3)
            )
        elif algorithm_id == 15:  # SARIMA
            return ForecastExecutionService.sarima_forecast(
                data=data,
                periods=periods,
                order=custom_params.get('order', [1, 1, 1]),
                seasonal_order=custom_params.get('seasonal_order', [1, 1, 1, 12])
            )
        else:
            # Default to simple moving average
            return ForecastExecutionService.simple_moving_average(
                data=data,
                periods=periods,
                window=custom_params.get('window', 3)
            )

    @staticmethod
    def linear_regression_forecast(data: pd.DataFrame, periods: int) -> Tuple[np.ndarray, Dict[str, float], Optional[Dict[str, Any]]]:
        """Linear regression forecasting"""
        # Prepare quantity data
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")
        
        n = len(y)
        if n < 2:
            raise ValueError("Need at least 2 historical data points")
        
        # Simple linear regression on time index
        x = np.arange(n).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        
        # Generate forecast
        future_x = np.arange(n, n + periods).reshape(-1, 1)
        forecast = model.predict(future_x)
        forecast = np.maximum(forecast, 0)
        
        # Calculate metrics
        predicted = model.predict(x)
        metrics = ForecastExecutionService.calculate_metrics(y, predicted)
        
        # Prepare test forecast (using fitted values for historical dates)
        test_results = {
            'test_forecast': predicted,
            'test_dates': data['period'].tolist() if 'period' in data.columns else []
        }
        
        return forecast, metrics 

    @staticmethod
    def arima_forecast(data: pd.DataFrame, periods: int, order: List[int] = None) -> Tuple[np.ndarray, Dict[str, float], Optional[Dict[str, Any]]]:
        """ARIMA forecasting (simplified implementation)."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            if order is None:
                order = [1, 1, 1]
            
            y = data['total_quantity'].values if 'total_quantity' in data.columns else data['quantity'].values
            
            model = ARIMA(y, order=order)
            fitted = model.fit()
            
            forecast = fitted.forecast(steps=periods)
            forecast = np.maximum(forecast, 0)
            
            predicted = fitted.fittedvalues
            metrics = ForecastExecutionService.calculate_metrics(y[len(y)-len(predicted):], predicted)
            
            # Prepare test forecast
            test_results = {
                'test_forecast': predicted,
                'test_dates': data['period'].iloc[len(y)-len(predicted):].tolist() if 'period' in data.columns else []
            }
            
            return forecast, metrics 
            
        except ImportError:
            logger.warning("statsmodels not available, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)



    """
FINAL FIX: Use linear regression for trend calculation instead of simple differences
This makes the trend robust to spikes and outliers
"""

    @staticmethod
    def polynomial_regression_forecast(data: pd.DataFrame, periods: int, degree: int = 2) -> tuple:
        """
        Polynomial regression with ROBUST trend calculation using linear regression.
        
        KEY FIX: Calculate trend using linear regression on recent data,
        which is much more robust to spikes/outliers than simple differences.
        """
        logger.info(f"=== POLYNOMIAL REGRESSION START ===")
        logger.info(f"Training data: {len(data)} rows, forecasting {periods} FUTURE periods, degree={degree}")
        
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

        n = len(y)
        logger.info(f"Training set size: {n}, values range: [{y.min():.0f}, {y.max():.0f}]")
        
        if n < 2:
            raise ValueError("Need at least 2 historical data points")

        # Cap degree for stability
        if n < 10:
            safe_degree = 1
        elif n < 25:
            safe_degree = min(degree, 2)
        else:
            safe_degree = min(degree, 3)
        
        logger.info(f"Using degree: {safe_degree} (requested: {degree})")
        degree = safe_degree

        # Fit polynomial on training data
        x = np.arange(n)
        coeffs = np.polyfit(x, y, degree)
        poly_func = np.poly1d(coeffs)
        
        logger.info(f"Polynomial fitted with coefficients: {coeffs}")
        
        # Calculate fitted values for metrics
        predicted = poly_func(x)
        predicted = np.maximum(predicted, 0)
        
        logger.info(f"Fitted values on training data: range=[{predicted.min():.0f}, {predicted.max():.0f}]")
        
        # Generate future forecast
        future_x = np.arange(n, n + periods)
        forecast_raw = poly_func(future_x)
        
        logger.info(f"Raw polynomial forecast: range=[{forecast_raw.min():.0f}, {forecast_raw.max():.0f}]")
        
        # ========================================================================
        # CRITICAL FIX: Use linear regression for robust trend estimation
        # ========================================================================
        
        # Calculate recent statistics
        lookback = min(12, n)  # For mean/std calculation
        recent_values = y[-lookback:]
        recent_mean = np.mean(recent_values)
        recent_std = np.std(recent_values)
        
        # Calculate trend using RANSAC (robust to outliers/spikes)
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        
        # Use up to 30 periods for trend calculation
        trend_window = min(n, 30)
        trend_x = np.arange(trend_window).reshape(-1, 1)
        trend_y = y[-trend_window:].reshape(-1, 1)
        
        # RANSAC is robust to outliers - it will ignore spike points
        try:
            ransac = RANSACRegressor(
                LinearRegression(),
                min_samples=max(3, int(trend_window * 0.5)),  # Use at least 50% of data
                residual_threshold=recent_std * 0.5,  # Points beyond 0.5 std are outliers
                random_state=42
            )
            ransac.fit(trend_x, trend_y)
            robust_trend = float(ransac.estimator_.coef_[0][0])
            
            # Log which points were considered inliers
            inlier_mask = ransac.inlier_mask_
            num_inliers = np.sum(inlier_mask) if inlier_mask is not None else trend_window
            logger.info(f"RANSAC used {num_inliers}/{trend_window} points as inliers")
            
        except Exception as e:
            # Fallback to simple linear regression if RANSAC fails
            logger.warning(f"RANSAC failed: {str(e)}, using simple linear regression")
            lr_model = LinearRegression()
            lr_model.fit(trend_x, trend_y)
            robust_trend = float(lr_model.coef_[0][0])
        
        logger.info(f"Recent stats: mean={recent_mean:.0f}, std={recent_std:.0f}")
        logger.info(f"Robust trend (from RANSAC on last {trend_window} periods): {robust_trend:.2f}/period")
        
        # Apply smart dampening
        forecast = []
        for i in range(periods):
            poly_value = forecast_raw[i]
            
            # Conservative projection using robust trend
            conservative_proj = y[-1] + (i + 1) * robust_trend
            
            # Mean reversion target
            mean_target = recent_mean
            
            # Progressive blending strategy
            if i < 3:
                # Near-term: 60% poly, 40% conservative (reduced from 70%)
                damped = poly_value * 0.6 + conservative_proj * 0.4
            elif i < 6:
                # Mid-term: 40% poly, 60% conservative
                damped = poly_value * 0.4 + conservative_proj * 0.6
            else:
                # Long-term: 20% poly, 50% conservative, 30% mean reversion
                damped = poly_value * 0.2 + conservative_proj * 0.5 + mean_target * 0.3
            
            # Apply reasonable bounds
            max_bound = recent_mean * 1.8  # Tightened from 2.0
            min_bound = recent_mean * 0.5  # Loosened from 0.4
            
            # Also respect training data range
            max_bound = min(max_bound, max(y) * 1.3)
            min_bound = max(min_bound, min(y) * 0.6)
            
            damped = np.clip(damped, min_bound, max_bound)
            forecast.append(max(0, damped))
        
        forecast = np.array(forecast)
        
        logger.info(f"Final forecast after dampening: range=[{forecast.min():.0f}, {forecast.max():.0f}]")
        
        # Calculate metrics on training data
        try:
            mae = float(np.mean(np.abs(y - predicted)))
            rmse = float(np.sqrt(np.mean((y - predicted) ** 2)))
            
            mask = y != 0
            if np.sum(mask) > 0:
                mape = float(np.mean(np.abs((y[mask] - predicted[mask]) / y[mask])) * 100)
            else:
                mape = 100.0
            
            mape = min(mape, 100.0)
            accuracy = max(0.0, 100.0 - mape)
            
            metrics = {
                'accuracy': round(accuracy, 2),
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'mape': round(mape, 2)
            }
            
            logger.info(f"Training metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, Accuracy={accuracy:.2f}%")
            
        except Exception as e:
            logger.error(f"Metric calculation failed: {str(e)}", exc_info=True)
            metrics = {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 100.0}
        
        logger.info(f"=== POLYNOMIAL REGRESSION END ===\n")
        
        return forecast, metrics


    @staticmethod
    def exponential_smoothing_forecast(data: pd.DataFrame, periods: int, alphas: list = [0.1, 0.3, 0.5]) -> tuple:
        """Enhanced exponential smoothing with windowed regression approach"""
        try:
            # Validate parameters
            validated_alphas = []
            for alpha in alphas:
                alpha = max(0.0, min(1.0, float(alpha)))
                validated_alphas.append(alpha)
            alphas = validated_alphas if validated_alphas else [0.3]

            logger.info(f"Using exponential smoothing alphas: {alphas}")

            # Get quantity values
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            
            # Handle small datasets
            if n < 3:
                return np.full(periods, y[-1] if len(y) > 0 else 0), {
                    'mape': 100.0, 
                    'mae': np.std(y) if len(y) > 1 else 0, 
                    'rmse': np.std(y) if len(y) > 1 else 0
                }, None

            best_metrics = None
            best_forecast = None
            best_test_results = None

            for alpha in alphas:
                logger.info(f"Running Exponential Smoothing with alpha={alpha}")
                
                # Use windowed regression approach (matching old implementation)
                window = min(5, n - 1)
                X, y_target = [], []

                for i in range(window, n):
                    # Calculate exponentially weighted historical values
                    weights = np.array([alpha * (1 - alpha) ** j for j in range(window)])
                    weights = weights / weights.sum()
                    weighted_history = np.sum(weights * y[i-window:i])

                    # Features: smoothed value + time trend
                    features = [weighted_history, i]
                    X.append(features)
                    y_target.append(y[i])

                if len(X) > 1:
                    X = np.array(X)
                    y_target = np.array(y_target)

                    # Fit linear regression model with smoothed features
                    model = LinearRegression()
                    model.fit(X, y_target)

                    # Generate forecast with rolling predictions
                    forecast = []
                    last_values = y[-window:].copy()

                    for i in range(periods):
                        # Calculate exponentially weighted history
                        weights = np.array([alpha * (1 - alpha) ** j for j in range(len(last_values))])
                        weights = weights / weights.sum()
                        weighted_history = np.sum(weights * last_values)

                        # Create features for prediction
                        features = [weighted_history, n + i]
                        
                        # Predict next value
                        pred = model.predict([features])[0]
                        pred = max(0, pred)  # Ensure non-negative
                        forecast.append(pred)

                        # Update rolling window with new prediction
                        last_values = np.append(last_values[1:], pred)

                    forecast = np.array(forecast)

                    # Calculate metrics on the windowed subset (matching old implementation)
                    predicted = model.predict(X)
                    metrics = ForecastExecutionService.calculate_metrics(y_target, predicted)
                    
                    test_results = {
                        'test_forecast': predicted,
                        'test_dates': data['period'].iloc[window:].tolist() if 'period' in data.columns else []
                    }

                else:
                    # Fallback to simple exponential smoothing for very small datasets
                    smoothed = np.zeros(n)
                    smoothed[0] = y[0]
                    
                    for i in range(1, n):
                        smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i-1]
                    
                    forecast = np.full(periods, smoothed[-1])
                    metrics = ForecastExecutionService.calculate_metrics(y[1:], smoothed[1:])
                    
                    test_results = {
                        'test_forecast': smoothed[1:],
                        'test_dates': data['period'].iloc[1:].tolist() if 'period' in data.columns else []
                    }

                logger.info(f"Alpha={alpha}, RMSE={metrics.get('rmse', 0):.2f}, MAE={metrics.get('mae', 0):.2f}, MAPE={metrics.get('mape', 0):.2f}")

                # Select best model based on RMSE
                if best_metrics is None or metrics.get('rmse', float('inf')) < best_metrics.get('rmse', float('inf')):
                    best_metrics = metrics
                    best_forecast = forecast
                    best_test_results = test_results

            return best_forecast, best_metrics

        except Exception as e:
            logger.warning(f"Exponential smoothing failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)


    @staticmethod
    def holt_winters_forecast(data: pd.DataFrame, periods: int, season_length: int = 12,
                            alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1) -> tuple:
        """
        Holt-Winters with improved bounds and trend dampening.
        """
        logger.info(f"=== HOLT-WINTERS START ===")
        logger.info(f"Training data: {len(data)} rows, forecasting {periods} FUTURE periods, season_length={season_length}")
        
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

        n = len(y)
        logger.info(f"Training set size: {n}, values range: [{y.min():.0f}, {y.max():.0f}]")
        
        min_required = 2 * season_length
        
        if n < min_required:
            logger.warning(f"Insufficient data for Holt-Winters (need {min_required}, have {n}). Using exponential smoothing.")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
        
        # Detect seasonality type
        cv = np.std(y) / np.mean(y) if np.mean(y) > 0 else 0
        use_multiplicative = cv > 0.3 and np.min(y) > 0
        
        logger.info(f"Coefficient of variation: {cv:.3f}, using {'multiplicative' if use_multiplicative else 'additive'}")
        
        # Initialize components
        level = np.mean(y[:season_length])
        
        if n >= 2 * season_length:
            trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
        else:
            trend = 0
        
        logger.info(f"Initial level: {level:.0f}, initial trend: {trend:.2f}")
        
        # Initialize seasonal
        seasonal = np.zeros(season_length)
        num_seasons = n // season_length
        
        for i in range(season_length):
            vals = [y[i + j*season_length] for j in range(num_seasons) if i + j*season_length < n]
            if vals:
                avg = np.mean(vals)
                if use_multiplicative and level > 0:
                    seasonal[i] = avg / level
                else:
                    seasonal[i] = avg - level
        
        # Normalize seasonal
        if use_multiplicative:
            seasonal = seasonal / np.mean(seasonal)
        else:
            seasonal = seasonal - np.mean(seasonal)
        
        logger.info(f"Seasonal components: range=[{seasonal.min():.3f}, {seasonal.max():.3f}]")
        
        # Fit model
        levels = [level]
        trends = [trend]
        seasonals = list(seasonal)
        fitted = []
        
        for i in range(n):
            s = i % season_length
            
            if i == 0:
                if use_multiplicative:
                    f = level * seasonal[s]
                else:
                    f = level + seasonal[s]
                fitted.append(f)
            else:
                prev_level = levels[-1]
                prev_trend = trends[-1]
                
                if use_multiplicative:
                    new_level = alpha * (y[i] / seasonals[s]) + (1 - alpha) * (prev_level + prev_trend)
                    new_trend = beta * (new_level - prev_level) + (1 - beta) * prev_trend
                    seasonals[s] = gamma * (y[i] / new_level) + (1 - gamma) * seasonals[s]
                    f = (prev_level + prev_trend) * seasonals[s]
                else:
                    new_level = alpha * (y[i] - seasonals[s]) + (1 - alpha) * (prev_level + prev_trend)
                    new_trend = beta * (new_level - prev_level) + (1 - beta) * prev_trend
                    seasonals[s] = gamma * (y[i] - new_level) + (1 - gamma) * seasonals[s]
                    f = prev_level + prev_trend + seasonals[s]
                
                levels.append(new_level)
                trends.append(new_trend)
                fitted.append(f)
                
                level = new_level
                trend = new_trend
        
        fitted = np.array(fitted)
        logger.info(f"Fitted values: range=[{fitted.min():.0f}, {fitted.max():.0f}]")
        
        # ========================================================================
        # CRITICAL FIX: Calculate robust trend for dampening
        # ========================================================================
        from sklearn.linear_model import LinearRegression
        
        # Use last season for trend calculation
        lookback = min(season_length, n)
        trend_data = y[-lookback:]
        trend_x = np.arange(lookback).reshape(-1, 1)
        
        lr_model = LinearRegression()
        lr_model.fit(trend_x, trend_data)
        robust_trend = float(lr_model.coef_[0])
        
        logger.info(f"Robust trend from linear regression: {robust_trend:.2f}/period")
        
        # Generate forecast with dampened trend
        forecast = []
        recent_mean = np.mean(y[-season_length:]) if len(y) >= season_length else np.mean(y)
        
        for i in range(periods):
            s = (n + i) % season_length
            
            # Original Holt-Winters prediction
            if use_multiplicative:
                hw_pred = (level + (i + 1) * trend) * seasonals[s]
            else:
                hw_pred = level + (i + 1) * trend + seasonals[s]
            
            # Conservative prediction using robust trend
            conservative_pred = y[-1] + (i + 1) * robust_trend
            
            # Blend HW with conservative (more dampening for longer horizons)
            blend_weight = 0.7 - (0.3 * i / max(periods - 1, 1))  # 70% -> 40%
            blend_weight = max(0.4, blend_weight)
            
            pred = hw_pred * blend_weight + conservative_pred * (1 - blend_weight)
            
            # Apply bounds
            max_bound = recent_mean * 1.6 
            min_bound = recent_mean * 0.5
            pred = np.clip(pred, min_bound, max_bound)
            
            forecast.append(max(0, pred))
        
        forecast = np.array(forecast)
        logger.info(f"Final forecast: range=[{forecast.min():.0f}, {forecast.max():.0f}]")
        
        # Calculate metrics
        try:
            mae = float(np.mean(np.abs(y - fitted)))
            rmse = float(np.sqrt(np.mean((y - fitted) ** 2)))
            
            mask = y != 0
            if np.sum(mask) > 0:
                mape = float(np.mean(np.abs((y[mask] - fitted[mask]) / y[mask])) * 100)
            else:
                mape = 100.0
            
            mape = min(mape, 100.0)
            accuracy = max(0.0, 100.0 - mape)
            
            metrics = {
                'accuracy': round(accuracy, 2),
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'mape': round(mape, 2)
            }
            
            logger.info(f"Training metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%, Accuracy={accuracy:.2f}%")
            
        except Exception as e:
            logger.error(f"Metric calculation failed: {str(e)}", exc_info=True)
            metrics = {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 100.0}
        
        logger.info(f"=== HOLT-WINTERS END ===\n")
        
        return forecast, metrics



    @staticmethod
    def simple_moving_average(data: pd.DataFrame, periods: int, window: int = 3) -> tuple:
        """
        Improved moving average forecasting with trend adjustment.
        Generates a proper forecast sequence by iteratively updating the moving average.
        """
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")
        
        n = len(y)
        window = min(max(3, window), n)
        
        logger.info(f"Moving Average: n={n}, periods={periods}, window={window}")
        
        # Calculate trend from recent data using linear regression
        trend_window = min(window * 2, n, 12)  # Use at most 12 periods for trend
        recent_values = y[-trend_window:]
        x_trend = np.arange(trend_window)
        
        from sklearn.linear_model import LinearRegression
        trend_model = LinearRegression()
        trend_model.fit(x_trend.reshape(-1, 1), recent_values)
        trend_slope = float(trend_model.coef_[0])
        
        # Calculate base moving average
        base_ma = float(np.mean(y[-window:]))
        
        # ✅ CRITICAL FIX: Generate forecast by rolling forward with trend
        # Instead of just extrapolating, we update the moving average as we forecast
        forecast = []
        
        # Start with the last 'window' actual values
        current_values = list(y[-window:])
        
        for i in range(periods):
            # Calculate moving average of current window
            current_ma = np.mean(current_values)
            
            # Apply trend with dampening
            damping = 0.95 ** i  # Exponential decay
            trend_adjustment = trend_slope * damping
            
            # Next forecast = current MA + trend
            next_value = current_ma + trend_adjustment
            
            # Apply reasonable bounds
            y_min, y_max = float(np.min(y)), float(np.max(y))
            min_bound = y_min * 0.5
            max_bound = y_max * 1.5
            next_value = np.clip(next_value, min_bound, max_bound)
            next_value = max(0, next_value)
            
            forecast.append(next_value)
            
            # ✅ KEY: Update the rolling window with the new forecast
            current_values = current_values[1:] + [next_value]
        
        forecast = np.array(forecast)
        
        # Calculate fitted values for metrics (simple moving averages on historical data)
        fitted = []
        for i in range(n):
            start_idx = max(0, i - window + 1)
            fitted.append(np.mean(y[start_idx:i+1]))
        
        fitted = np.array(fitted)
        
        # Calculate metrics (skip warm-up period)
        if n > window:
            metrics = ForecastExecutionService.calculate_metrics(y[window-1:], fitted[window-1:])
        else:
            metrics = ForecastExecutionService.calculate_metrics(y, fitted)
        
        logger.info(f"Moving Average: trend_slope={trend_slope:.2f}, base_ma={base_ma:.0f}")
        logger.info(f"Moving Average forecast: first={forecast[0]:.0f}, last={forecast[-1]:.0f}")
        logger.info(f"Moving Average metrics: MAE={metrics.get('mae', 0):.2f}, MAPE={metrics.get('mape', 0):.2f}%")
        
        return forecast, metrics

    @staticmethod
    def prophet_forecast(data: pd.DataFrame, periods: int, seasonality_mode: str = 'additive', changepoint_prior_scale: float = 0.05) -> Tuple[np.ndarray, Dict[str, float], Optional[Dict[str, Any]]]:
        """Prophet forecasting using Facebook's Prophet library."""
        try:
            if Prophet is None:
                logger.warning("Prophet library not installed, falling back to Exponential Smoothing")
                return ForecastExecutionService.exponential_smoothing_forecast(data, periods)

            # Prepare data for Prophet
            if 'total_quantity' in data.columns:
                y_values = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y_values = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            # Get dates from data if available, otherwise create sequential dates
            if 'test_date' in data.columns:
                dates = pd.to_datetime(data['period'])
            else:
                # Create dates starting from today
                dates = pd.date_range(end=datetime.now(), periods=len(y_values), freq='MS')

            # Create Prophet dataframe
            prophet_data = pd.DataFrame({
                'ds': dates,
                'y': y_values
            })

            n = len(prophet_data)
            if n < 3:
                raise ValueError("Need at least 3 historical data points for Prophet")

            # Initialize and fit Prophet model
            model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                interval_width=0.95,
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality=False
            )
            
            # Suppress Prophet's verbose output
            with pd.option_context('mode.chained_assignment', None):
                model.fit(prophet_data)

            # Create future dataframe
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast_df = model.predict(future)

            # Extract forecast values
            forecast = forecast_df['yhat'].values[-periods:]
            forecast = np.maximum(forecast, 0)  # Ensure non-negative

            # Calculate metrics on historical data
            fitted_values = forecast_df['yhat'].values[:n]
            metrics = ForecastExecutionService.calculate_metrics(y_values, fitted_values)

            # Prepare test forecast
            test_results = {
                'test_forecast': fitted_values,
                'test_dates': data['period'].tolist() if 'period' in data.columns else []
            }

            logger.info(f"Prophet forecast completed with {periods} periods")
            return forecast, metrics 

        except ImportError:
            logger.warning("Prophet library not available, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
        except Exception as e:
            logger.warning(f"Prophet forecasting failed: {str(e)}, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
        
    @staticmethod
    def lstm_forecast(data: pd.DataFrame, periods: int, sequence_length: int = 12, epochs: int = 50, batch_size: int = 32) -> Tuple[np.ndarray, Dict[str, float], Optional[Dict[str, Any]]]:
        """LSTM Neural Network forecasting."""
        try:
            from tensorflow import keras
            from tensorflow.keras.models import Sequential # type: ignore
            from tensorflow.keras.layers import LSTM, Dense # type: ignore
            from tensorflow.keras.optimizers import Adam # type: ignore
            import warnings
            import random
            import numpy as np
            import tensorflow as tf
            
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            warnings.filterwarnings('ignore')

            # Prepare quantity data
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            if n < sequence_length + 1:
                raise ValueError(f"Need at least {sequence_length + 1} historical data points for LSTM")

            # Normalize data
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

            # Create sequences
            X_train = []
            y_train = []

            for i in range(sequence_length, n):
                X_train.append(y_scaled[i-sequence_length:i])
                y_train.append(y_scaled[i])

            X_train = np.array(X_train).reshape(-1, sequence_length, 1)
            y_train = np.array(y_train)

            if len(X_train) < 2:
                raise ValueError("Insufficient data for LSTM training")

            # Build LSTM model
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
                LSTM(50, activation='relu'),
                Dense(25, activation='relu'),
                Dense(1)
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

            # Train model with suppressed output
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                model.fit(
                    X_train, y_train,
                    epochs=min(epochs, 100),  # Cap epochs to prevent long training
                    batch_size=min(batch_size, len(X_train)),
                    verbose=0,
                    validation_split=0.1
                )

            # Generate forecast
            forecast = []
            current_sequence = y_scaled[-sequence_length:].copy()

            for _ in range(periods):
                next_pred = model.predict(
                    current_sequence.reshape(1, sequence_length, 1),
                    verbose=0
                )[0, 0]
                forecast.append(next_pred)
                current_sequence = np.append(current_sequence[1:], next_pred)

            # Inverse transform forecast
            forecast = np.array(forecast).reshape(-1, 1)
            forecast = scaler.inverse_transform(forecast).flatten()
            forecast = np.maximum(forecast, 0)  # Ensure non-negative

            # Calculate metrics on training data
            y_train_pred_scaled = model.predict(X_train, verbose=0).flatten()
            y_train_pred = scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            metrics = ForecastExecutionService.calculate_metrics(y_train_actual, y_train_pred)

            # Prepare test forecast
            test_results = {
                'test_forecast': y_train_pred,
                'test_dates': data['period'].iloc[sequence_length:].tolist() if 'period' in data.columns else []
            }

            logger.info(f"LSTM forecast completed with {periods} periods")
            return forecast, metrics 

        except ImportError:
            logger.warning("TensorFlow/Keras not available, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
        except Exception as e:
            logger.warning(f"LSTM forecasting failed: {str(e)}, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def gaussian_process_forecast(data: pd.DataFrame, periods: int, kernel: str = 'RBF', alpha: float = 1e-6) -> Tuple[np.ndarray, Dict[str, float], Optional[Dict[str, Any]]]:
        """
        Gaussian Process Regression forecasting with uncertainty quantification.
        
        Args:
            data: Historical data with 'quantity' or 'total_quantity' column
            periods: Number of forecast periods
            kernel: Kernel type ('RBF', 'Matern', 'RationalQuadratic')
            alpha: Regularization strength
            
        Returns:
            Tuple of (forecast_array, metrics_dict )
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel as C
            from sklearn.preprocessing import StandardScaler
            
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")
            
            y = y.reshape(-1, 1)
            
            # Create lagged features for time series
            max_lags = min(12, len(y) // 2)
            X_train = []
            y_train = []
            
            for i in range(max_lags, len(y)):
                X_train.append(y[i-max_lags:i].flatten())
                y_train.append(y[i][0])
            
            if len(X_train) < 5:
                logger.warning("Insufficient data for Gaussian Process, using exponential smoothing fallback")
                return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Scale features
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            
            # Scale targets
            scaler_y = StandardScaler()
            y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # Select kernel
            if kernel.upper() == 'MATERN':
                kernel_obj = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=1.5)
            elif kernel.upper() == 'RATIONALQUADRATIC':
                kernel_obj = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=0.1)
            else:  # RBF
                kernel_obj = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            
            # Train Gaussian Process
            gp = GaussianProcessRegressor(
                kernel=kernel_obj,
                alpha=alpha,
                normalize_y=False,
                n_restarts_optimizer=5,
                random_state=42
            )
            gp.fit(X_train_scaled, y_train_scaled)
            
            # Generate forecast
            forecast = []
            recent_lags = X_train[-1].copy()
            
            for _ in range(periods):
                recent_scaled = scaler_X.transform(recent_lags.reshape(1, -1))[0]
                pred_scaled, std = gp.predict(recent_scaled.reshape(1, -1), return_std=True)
                pred = scaler_y.inverse_transform([[pred_scaled[0]]])[0][0]
                forecast.append(pred)
                
                # Update lags
                recent_lags = np.append(recent_lags[1:], pred)
            
            forecast = np.array(forecast)

            # Handle NaN values in forecast
            forecast = np.nan_to_num(forecast, nan=0.0, posinf=0.0, neginf=0.0)

            # Calculate metrics on training data
            test_results = None
            if len(X_train) > 0:
                y_pred_scaled = gp.predict(X_train_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                # Handle NaN in predictions
                y_pred = np.nan_to_num(y_pred, nan=0.0)
                metrics = ForecastExecutionService.calculate_metrics(y_train, y_pred)
                # Handle NaN in metrics
                metrics = {k: (0.0 if np.isnan(v) else v) for k, v in metrics.items()}
                
                # Prepare test forecast results
                test_results = {
                    'test_forecast': y_pred,
                    'test_dates': data['period'].iloc[max_lags:].tolist() if 'period' in data.columns else []
                }
            else:
                metrics = {'mae': 0, 'rmse': 0, 'mape': 0, 'r_squared': 0}

            # Forecast validation removed

            logger.info(f"Gaussian Process forecast completed with {periods} periods")
            return forecast, metrics 
            
        except ImportError:
            logger.warning("Scikit-learn not available, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
        except Exception as e:
            logger.warning(f"Gaussian Process forecasting failed: {str(e)}, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def mlp_neural_network_forecast(data: pd.DataFrame, periods: int, hidden_layers: List[int] = None, epochs: int = 100, batch_size: int = 32) -> Tuple[np.ndarray, Dict[str, float], Optional[Dict[str, Any]]]:
        """
        Multi-layer Perceptron (MLP) Neural Network forecasting.
        
        Args:
            data: Historical data with 'quantity' or 'total_quantity' column
            periods: Number of forecast periods
            hidden_layers: List of hidden layer sizes, e.g., [64, 32]
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Tuple of (forecast_array, metrics_dict )
        """
        if hidden_layers is None:
            hidden_layers = [64, 32]
            
        try:
            from tensorflow import keras
            from tensorflow.keras import layers # type: ignore
            from sklearn.preprocessing import MinMaxScaler
            import random
            import numpy as np
            import tensorflow as tf
            
            random.seed(42)
            np.random.seed(42)
            tf.random.set_seed(42)
            
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")
            
            if len(y) < 20:
                logger.warning("Insufficient data for MLP, using exponential smoothing fallback")
                return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
            
            # Create lagged features
            sequence_length = min(12, len(y) // 3)
            X_train = []
            y_train = []
            
            for i in range(sequence_length, len(y)):
                X_train.append(y[i-sequence_length:i])
                y_train.append(y[i])
            
            if len(X_train) < 5:
                logger.warning("Insufficient sequences for MLP, using exponential smoothing fallback")
                return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler.fit_transform(X_train)
            y_scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            # Build MLP model
            model = keras.Sequential()
            model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(sequence_length,)))
            
            for hidden_size in hidden_layers[1:]:
                model.add(layers.Dense(hidden_size, activation='relu'))
                model.add(layers.Dropout(0.2))
            
            model.add(layers.Dense(1))
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train model (suppress verbose output for clean logging)
            model.fit(
                X_scaled, y_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=0
            )
            
            # Generate forecast
            forecast = []
            recent_data = X_train[-1].copy()
            
            for _ in range(periods):
                recent_scaled = scaler.transform(recent_data.reshape(1, -1))[0]
                pred_scaled = model.predict(recent_scaled.reshape(1, -1), verbose=0)[0][0]
                pred = y_scaler.inverse_transform([[pred_scaled]])[0][0]
                forecast.append(pred)
                
                # Update sequence
                recent_data = np.append(recent_data[1:], pred)
            
            forecast = np.array(forecast)
            
            # Calculate metrics on training data
            y_train_pred_scaled = model.predict(X_scaled, verbose=0).flatten()
            y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
            metrics = ForecastExecutionService.calculate_metrics(y_train, y_train_pred)
            
            # Prepare test forecast results
            test_results = {
                'test_forecast': y_train_pred,
                'test_dates': data['period'].iloc[sequence_length:].tolist() if 'period' in data.columns else []
            }

            # Forecast validation removed
            
            logger.info(f"MLP Neural Network forecast completed with {periods} periods")
            return forecast, metrics 
            
        except ImportError:
            logger.warning("TensorFlow not available, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
        except Exception as e:
            logger.warning(f"MLP Neural Network forecasting failed: {str(e)}, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)



    @staticmethod
    def seasonal_decomposition_forecast(data: pd.DataFrame, periods: int, season_length: int = 12) -> tuple:
        """Seasonal decomposition forecasting"""
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")
        
        if len(y) < 2 * season_length:
            return ForecastExecutionService.linear_regression_forecast(data, periods)
        
        # Simple seasonal decomposition
        trend = np.convolve(y, np.ones(season_length)/season_length, mode='same')
        
        # Calculate seasonal component
        detrended = y - trend
        seasonal_pattern = []
        for i in range(season_length):
            seasonal_values = [detrended[j] for j in range(i, len(detrended), season_length)]
            seasonal_pattern.append(np.mean(seasonal_values))
        
        # Fit polynomial to trend for future values
        x = np.arange(len(trend))
        valid_trend = ~np.isnan(trend)
        
        if np.sum(valid_trend) > 1:
            # Use polyfit instead of linregress
            coeffs = np.polyfit(x[valid_trend], trend[valid_trend], 1)
            slope, intercept = coeffs[0], coeffs[1]
            future_trend = [slope * (len(y) + i) + intercept for i in range(periods)]
        else:
            future_trend = [np.nanmean(trend)] * periods
        
        # Future seasonal component
        future_seasonal = [seasonal_pattern[(len(y) + i) % season_length] for i in range(periods)]
        
        # Combine forecast
        forecast = np.array(future_trend) + np.array(future_seasonal)
        forecast = np.maximum(forecast, 0)
        
        # Calculate metrics
        seasonal_full = np.tile(seasonal_pattern, len(y) // season_length + 1)[:len(y)]
        fitted = trend + seasonal_full
        valid_fitted = ~np.isnan(fitted)
        
        test_results = None
        if np.sum(valid_fitted) > 0:
            metrics = ForecastExecutionService.calculate_metrics(y[valid_fitted], fitted[valid_fitted])
            test_results = {
                'test_forecast': fitted[valid_fitted],
                'test_dates': data['period'].iloc[np.where(valid_fitted)[0]].tolist() if 'period' in data.columns else []
            }
        else:
            metrics = {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}
        
        return forecast, metrics 

    @staticmethod
    def moving_average_forecast(data: pd.DataFrame, periods: int, window: int = 3) -> tuple:
        """
        Improved moving average forecasting with trend adjustment.
        Generates a proper forecast sequence by iteratively updating the moving average.
        """
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")
        
        n = len(y)
        window = min(max(3, window), n)
        
        logger.info(f"Moving Average: n={n}, periods={periods}, window={window}")
        
        # Calculate trend from recent data using linear regression
        trend_window = min(window * 2, n, 12)  # Use at most 12 periods for trend
        recent_values = y[-trend_window:]
        x_trend = np.arange(trend_window)
        
        from sklearn.linear_model import LinearRegression
        trend_model = LinearRegression()
        trend_model.fit(x_trend.reshape(-1, 1), recent_values)
        trend_slope = float(trend_model.coef_[0])
        
        # Calculate base moving average
        base_ma = float(np.mean(y[-window:]))
        
        # ✅ CRITICAL FIX: Generate forecast by rolling forward with trend
        # Instead of just extrapolating, we update the moving average as we forecast
        forecast = []
        
        # Start with the last 'window' actual values
        current_values = list(y[-window:])
        
        for i in range(periods):
            # Calculate moving average of current window
            current_ma = np.mean(current_values)
            
            # Apply trend with dampening
            damping = 0.95 ** i  # Exponential decay
            trend_adjustment = trend_slope * damping
            
            # Next forecast = current MA + trend
            next_value = current_ma + trend_adjustment
            
            # Apply reasonable bounds
            y_min, y_max = float(np.min(y)), float(np.max(y))
            min_bound = y_min * 0.5
            max_bound = y_max * 1.5
            next_value = np.clip(next_value, min_bound, max_bound)
            next_value = max(0, next_value)
            
            forecast.append(next_value)
            
            # ✅ KEY: Update the rolling window with the new forecast
            current_values = current_values[1:] + [next_value]
        
        forecast = np.array(forecast)
        
        # Calculate fitted values for metrics (simple moving averages on historical data)
        fitted = []
        for i in range(n):
            start_idx = max(0, i - window + 1)
            fitted.append(np.mean(y[start_idx:i+1]))
        
        fitted = np.array(fitted)
        
        # Calculate metrics (skip warm-up period)
        if n > window:
            metrics = ForecastExecutionService.calculate_metrics(y[window-1:], fitted[window-1:])
        else:
            metrics = ForecastExecutionService.calculate_metrics(y, fitted)
        
        logger.info(f"Moving Average: trend_slope={trend_slope:.2f}, base_ma={base_ma:.0f}")
        logger.info(f"Moving Average forecast: first={forecast[0]:.0f}, last={forecast[-1]:.0f}")
        logger.info(f"Moving Average metrics: MAE={metrics.get('mae', 0):.2f}, MAPE={metrics.get('mape', 0):.2f}%")
        
        return forecast, metrics

    @staticmethod
    def sarima_forecast(data: pd.DataFrame, periods: int) -> tuple:
        """SARIMA forecasting using statsmodels"""
        from statsmodels.tsa.arima.model import ARIMA
        from pmdarima import auto_arima # type: ignore
        import warnings

        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

        if len(y) < 24:  # Need at least 2 seasons for SARIMA
            return ForecastExecutionService.arima_forecast(data, periods)

        # Determine seasonal period
        seasonal_period = 12
        if len(y) < 2 * seasonal_period:
            seasonal_period = max(4, len(y) // 3)

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # Auto-select SARIMA parameters
                auto_model = auto_arima(
                    y,
                    start_p=0, start_q=0, max_p=3, max_q=3, max_d=2,
                    start_P=0, start_Q=0, max_P=2, max_Q=2, max_D=1,
                    seasonal=True, m=seasonal_period,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    random_state=42,

                    trace=False
                )

                logger.info(f"SARIMA selected model: {auto_model.summary()}")
                logger.info(f"SARIMA seasonal period used: {seasonal_period}")

                forecast = auto_model.predict(n_periods=periods)
                forecast = np.maximum(forecast, 0)

                # Calculate metrics
                fitted = auto_model.predict_in_sample()
                if len(fitted) == len(y):
                    metrics = ForecastExecutionService.calculate_metrics(y, fitted)
                    start_idx = 0
                else:
                    start_idx = len(y) - len(fitted)
                    metrics = ForecastExecutionService.calculate_metrics(y[start_idx:], fitted)

                # Prepare test forecast
                test_results = {
                    'test_forecast': fitted,
                    'test_dates': data['period'].iloc[start_idx:].tolist() if 'period' in data.columns else []
                }

                return forecast, metrics 

        except Exception as e:
            print(f"SARIMA failed: {e}")

        # Fallback to ARIMA
        return ForecastExecutionService.arima_forecast(data, periods)

    @staticmethod
    def _get_external_factor_columns(
        data: pd.DataFrame, 
        tenant_id: str = None, 
        database_name: str = None,
        aggregation_level: str = None
    ) -> List[str]:
        """
        Identify external factor columns in the data using field catalogue metadata.
        
        Args:
            data: DataFrame with historical data and potentially merged external factors
            tenant_id: Tenant identifier (required for dynamic field detection)
            database_name: Database name (required for dynamic field detection)
            aggregation_level: Current aggregation level being used (e.g., "product")
            
        Returns:
            List of external factor column names
        """
        if not tenant_id or not database_name:
            logger.warning("tenant_id and database_name required for dynamic field detection")
            return []
        
        # Standard columns that are NEVER external factors
        standard_columns = {
            'period', 'date', 'total_quantity', 'quantity',
            'transaction_count', 'avg_price', 'uom'
        }
        
        try:
            # Get dynamic field names from metadata
            from app.core.aggregation_service import AggregationService
            target_field, date_field = AggregationService._get_field_names(tenant_id, database_name)
            standard_columns.update({target_field, date_field})
            
            # Get all dimension fields from field catalogue
            dimension_fields = AggregationService._get_dimension_fields(tenant_id, database_name)
            
            # Parse aggregation level to get fields being used for aggregation
            agg_fields = set()
            if aggregation_level:
                agg_fields = set(field.strip() for field in aggregation_level.split('-'))
                logger.info(f"Current aggregation fields: {agg_fields}")
            
            # Build exclusion set: standard columns + dimension fields + aggregation fields
            exclude_columns = standard_columns | set(dimension_fields) | agg_fields
            
            logger.info(f"Excluding columns: {exclude_columns}")
            
            # Find external factors: columns that are NOT in exclusion set
            external_factors = []
            for col in data.columns:
                # Skip if in exclusion list
                if col in exclude_columns:
                    continue
                
                # Skip datetime columns
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    continue
                
                # If column has only 1 unique value, it's likely a filter - skip it
                if data[col].nunique() == 1:
                    logger.info(f"Excluding '{col}' - single unique value (likely a filter)")
                    continue
                
                # If we get here, treat as external factor
                external_factors.append(col)
                logger.info(f"Including '{col}' as external factor")
            
            if not external_factors:
                logger.info("No external factors identified in data")
            else:
                logger.info(f"Identified {len(external_factors)} external factors: {external_factors}")
            
            return external_factors
            
        except Exception as e:
            logger.error(f"Failed to identify external factors: {str(e)}", exc_info=True)
            return []

    @staticmethod
    def _extract_features_with_factors(data: pd.DataFrame, window: int, external_factors: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features including both lagged values and external factors.
        
        Args:
            data: DataFrame with historical data and external factors
            window: Number of lags for time series features
            external_factors: List of external factor column names
            
        Returns:
            Tuple of (X features array, y target array)
        """
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")
        
        X = []
        y_target = []

        # Preprocess external factors: convert ALL columns to numeric
        encoded_columns = {}
        for factor_name in external_factors:
            if factor_name in data.columns:
                col = data[factor_name]
                
                # Handle different data types
                if pd.api.types.is_numeric_dtype(col):
                    # Numeric column: fill NaN with 0
                    encoded_columns[factor_name] = col.fillna(0.0).astype(float).values
                else:
                    # Categorical/String column: use factorize
                    # Fill NaN with a placeholder first
                    col_filled = col.fillna('__MISSING__').astype(str)
                    codes, _ = pd.factorize(col_filled)
                    encoded_columns[factor_name] = codes.astype(float)
                    logger.info(f"Encoded categorical factor '{factor_name}' with {len(set(codes))} unique values")

        # Build feature matrix
        for i in range(window, len(data)):
            lags = y[i-window:i]
            time_idx = i
            features = list(lags) + [time_idx]

            # Add encoded external factors
            for factor_name in external_factors:
                if factor_name in encoded_columns:
                    factor_value = encoded_columns[factor_name][i]
                    features.append(float(factor_value))
                else:
                    features.append(0.0)

            X.append(features)
            y_target.append(y[i])

        X_array = np.array(X, dtype=np.float64)
        y_array = np.array(y_target, dtype=np.float64)
        
        # Final validation: ensure no NaN or Inf
        if np.any(np.isnan(X_array)) or np.any(np.isinf(X_array)):
            logger.warning("NaN/Inf detected in features, replacing with 0")
            X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.any(np.isnan(y_array)) or np.any(np.isinf(y_array)):
            logger.warning("NaN/Inf detected in targets, replacing with 0")
            y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)

        return X_array, y_array

    @staticmethod
    def _prepare_future_features_with_factors(
        recent_lags: List[float], 
        n: int, 
        idx: int, 
        window: int, 
        external_factors: List[str], 
        last_factor_values: Dict[str, float]
    ) -> np.ndarray:
        """
        Prepare features for future prediction including external factors.
        For external factors, use the last known values.
        """
        features = list(recent_lags) + [n + idx]
        
        for factor_name in external_factors:
            if factor_name in last_factor_values:
                features.append(float(last_factor_values[factor_name]))
            else:
                features.append(0.0)
        
        # Ensure all features are valid floats
        features_array = np.array(features, dtype=np.float64)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array

    @staticmethod
    def xgboost_forecast(
        data: pd.DataFrame, 
        periods: int, 
        n_estimators: int = 100, 
        max_depth: int = 6, 
        learning_rate: float = 0.1,
        tenant_id: str = None,
        database_name: str = None,
        aggregation_level: str = None
    ):
        """
        XGBoost forecasting - uses ALL data for training (no internal split).

        Args:
            data: Historical data with 'quantity' column
            periods: Number of periods to forecast
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate

        Returns:
            Tuple of (forecast_array, metrics_dict)
        """
        try:
            # Validate parameters
            n_estimators = max(10, min(1000, n_estimators))
            max_depth = max(1, min(20, max_depth))
            learning_rate = max(0.01, min(1.0, learning_rate))

            logger.info(f"Using XGBoost parameters: n_estimators={n_estimators}, max_depth={max_depth}, learning_rate={learning_rate}")

            # Prepare quantity data
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            if n < 10:
                raise ValueError("Need at least 10 historical data points for XGBoost")

            # Feature engineering: create lag features, time index, and external factors
            window = min(5, n - 1)
            external_factors = ForecastExecutionService._get_external_factor_columns(
                data,
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level
            )

            if external_factors:
                logger.info(f"XGBoost: Using external factors: {external_factors}")
                X, y_target = ForecastExecutionService._extract_features_with_factors(data, window, external_factors)
            else:
                # Build lagged features without external factors
                X = []
                y_target = []
                for i in range(window, n):
                    lags = y[i-window:i]
                    time_idx = i
                    features = list(lags) + [time_idx]
                    X.append(features)
                    y_target.append(y[i])
                X = np.array(X)
                y_target = np.array(y_target)

            if len(X) < 5:
                raise ValueError("Insufficient data for XGBoost training")

            # Train XGBoost model on ALL data (no internal split) - FIXED: Added random_state for reproducible results
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,  # Fixed random state for reproducible results
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X, y_target, verbose=False)

            logger.info("XGBoost: Model training completed with fixed random_state=42")

            # Generate forecast
            forecast = []
            recent_lags = list(y[-window:])
            
            last_factor_values = {}
            if external_factors and len(data) > 0:
                for factor_name in external_factors:
                    if factor_name in data.columns:
                        factor_value_raw = data[factor_name].iloc[-1]
                        if pd.isna(factor_value_raw):
                            val = 0.0
                        else:
                            try:
                                val = float(factor_value_raw)
                            except Exception:
                                try:
                                    codes, uniques = pd.factorize(data[factor_name].fillna('__NA__'))
                                    val = float(codes[-1])
                                except Exception:
                                    val = float(abs(hash(str(factor_value_raw))) % 1000000) / 1000.0
                        last_factor_values[factor_name] = val

            for i in range(periods):
                if external_factors:
                    features = ForecastExecutionService._prepare_future_features_with_factors(
                        recent_lags, n, i, window, external_factors, last_factor_values
                    )
                else:
                    features = np.array(recent_lags + [n + i])
                
                pred = model.predict([features])[0]
                pred = max(0, pred)
                forecast.append(pred)

                # Update lags with new prediction
                recent_lags = recent_lags[1:] + [pred]

            forecast = np.array(forecast)

            # Calculate metrics on training data
            predicted = model.predict(X)
            metrics = ForecastExecutionService.calculate_metrics(y_target, predicted)

            return forecast, metrics

        except ImportError:
            logger.warning("XGBoost not available, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)
        except Exception as e:
            logger.warning(f"XGBoost forecasting failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)


    @staticmethod
    def svr_forecast(data: pd.DataFrame, periods: int, C: float = 1.0, epsilon: float = 0.1, kernel: str = 'rbf',tenant_id: str = None,database_name: str = None,aggregation_level: str = None):
        """
        Support Vector Regression forecasting - uses ALL data for training (no internal split).
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            C: Regularization parameter
            epsilon: Epsilon-tube parameter
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            
        Returns:
            Tuple of (forecast_array, metrics_dict)
        """
        try:
            # Validate parameters
            C = max(0.1, min(100.0, C))
            epsilon = max(0.01, min(1.0, epsilon))
            valid_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            if kernel not in valid_kernels:
                kernel = 'rbf'

            logger.info(f"Using SVR parameters: C={C}, epsilon={epsilon}, kernel={kernel}")

            # Prepare quantity data
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            if n < 10:
                raise ValueError("Need at least 10 historical data points for SVR")

            # Feature engineering: create lag features, time index, and external factors
            window = min(5, n - 1)
            external_factors = ForecastExecutionService._get_external_factor_columns(
                data,
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level
            )
            
            if external_factors:
                logger.info(f"SVR: Using external factors: {external_factors}")
                X, y_target = ForecastExecutionService._extract_features_with_factors(data, window, external_factors)
            else:
                # Build lagged features without external factors
                X = []
                y_target = []
                for i in range(window, n):
                    lags = y[i-window:i]
                    time_idx = i
                    features = list(lags) + [time_idx]
                    X.append(features)
                    y_target.append(y[i])
                X = np.array(X)
                y_target = np.array(y_target)

            if len(X) < 5:
                raise ValueError("Insufficient data for SVR training")

            # Scale features for SVR (required for better performance)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train SVR model on ALL data (no internal split)
            model = SVR(
                kernel=kernel,
                C=C,
                epsilon=epsilon
            )
            model.fit(X_scaled, y_target)

            # Generate forecast
            forecast = []
            recent_lags = list(y[-window:])
            
            last_factor_values = {}
            if external_factors and len(data) > 0:
                for factor_name in external_factors:
                    if factor_name in data.columns:
                        factor_value_raw = data[factor_name].iloc[-1]
                        if pd.isna(factor_value_raw):
                            val = 0.0
                        else:
                            try:
                                val = float(factor_value_raw)
                            except Exception:
                                try:
                                    codes, uniques = pd.factorize(data[factor_name].fillna('__NA__'))
                                    val = float(codes[-1])
                                except Exception:
                                    val = float(abs(hash(str(factor_value_raw))) % 1000000) / 1000.0
                        last_factor_values[factor_name] = val

            for i in range(periods):
                if external_factors:
                    features = ForecastExecutionService._prepare_future_features_with_factors(
                        recent_lags, n, i, window, external_factors, last_factor_values
                    )
                else:
                    features = np.array(recent_lags + [n + i])
                
                features_scaled = scaler.transform([features])[0]
                pred = model.predict([features_scaled])[0]
                pred = max(0, pred)
                forecast.append(pred)

                # Update lags with new prediction
                recent_lags = recent_lags[1:] + [pred]

            forecast = np.array(forecast)

            # Calculate metrics on training data
            predicted = model.predict(X_scaled)
            metrics = ForecastExecutionService.calculate_metrics(y_target, predicted)

            return forecast, metrics

        except Exception as e:
            logger.warning(f"SVR failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def knn_forecast(data: pd.DataFrame, periods: int, n_neighbors: int = 5,tenant_id: str = None,database_name: str = None,aggregation_level: str = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        K-Nearest Neighbors forecasting - uses ALL data for training (no internal split).
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            n_neighbors: Number of neighbors to use
            
        Returns:
            Tuple of (forecast_array, metrics_dict)
        """
        try:
            # Prepare quantity data
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            if n < n_neighbors + 5:
                raise ValueError(f"Need at least {n_neighbors + 5} historical data points for KNN")

            # Feature engineering: create lag features, time index, and external factors
            window = min(5, n - 1)
            external_factors = ForecastExecutionService._get_external_factor_columns(
                data,
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level
            )
            
            if external_factors:
                logger.info(f"KNN: Using external factors: {external_factors}")
                X, y_target = ForecastExecutionService._extract_features_with_factors(data, window, external_factors)
            else:
                # Build lagged features without external factors
                X = []
                y_target = []
                for i in range(window, n):
                    lags = y[i-window:i]
                    time_idx = i
                    features = list(lags) + [time_idx]
                    X.append(features)
                    y_target.append(y[i])
                X = np.array(X)
                y_target = np.array(y_target)

            if len(X) < n_neighbors:
                raise ValueError(f"Insufficient data for KNN training (need at least {n_neighbors} samples)")

            # Scale features for KNN (important for distance metrics)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train KNN model on ALL data (no internal split)
            model = KNeighborsRegressor(
                n_neighbors=min(n_neighbors, len(X)),
                weights='uniform'
            )
            model.fit(X_scaled, y_target)

            # Generate forecast
            forecast = []
            recent_lags = list(y[-window:])
            
            last_factor_values = {}
            if external_factors and len(data) > 0:
                for factor_name in external_factors:
                    if factor_name in data.columns:
                        factor_value_raw = data[factor_name].iloc[-1]
                        if pd.isna(factor_value_raw):
                            val = 0.0
                        else:
                            try:
                                val = float(factor_value_raw)
                            except Exception:
                                try:
                                    codes, uniques = pd.factorize(data[factor_name].fillna('__NA__'))
                                    val = float(codes[-1])
                                except Exception:
                                    val = float(abs(hash(str(factor_value_raw))) % 1000000) / 1000.0
                        last_factor_values[factor_name] = val

            for i in range(periods):
                if external_factors:
                    features = ForecastExecutionService._prepare_future_features_with_factors(
                        recent_lags, n, i, window, external_factors, last_factor_values
                    )
                else:
                    features = np.array(recent_lags + [n + i])
                
                features_scaled = scaler.transform([features])[0]
                pred = model.predict([features_scaled])[0]
                pred = max(0, pred)
                forecast.append(pred)

                # Update lags with new prediction
                recent_lags = recent_lags[1:] + [pred]

            forecast = np.array(forecast)

            # Calculate metrics on training data
            predicted = model.predict(X_scaled)
            metrics = ForecastExecutionService.calculate_metrics(y_target, predicted)

            return forecast, metrics

        except Exception as e:
            logger.warning(f"KNN failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def random_forest_forecast(
        data: pd.DataFrame, 
        periods: int, 
        n_estimators: int = 100, 
        max_depth: Optional[int] = None, 
        min_samples_split: int = 2, 
        min_samples_leaf: int = 1,
        tenant_id: str = None,
        database_name: str = None,
        aggregation_level: str = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Random Forest regression forecasting - uses ALL data for training (no internal split).
        
        Args:
            data: Historical data
            periods: Number of periods to forecast
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            min_samples_leaf: Minimum samples in leaf
            tenant_id: Tenant identifier (for dynamic field detection)
            database_name: Database name (for dynamic field detection)
            aggregation_level: Current aggregation level (e.g., "product")
            
        Returns:
            Tuple of (forecast_array, metrics_dict)
        """
        try:
            n_estimators = max(10, min(500, n_estimators))
            if max_depth is not None:
                max_depth = max(1, min(50, max_depth))
            min_samples_split = max(2, min(20, min_samples_split))
            min_samples_leaf = max(1, min(10, min_samples_leaf))

            logger.info(f"Using Random Forest parameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            if n < 10:
                raise ValueError("Need at least 10 historical data points for Random Forest")

            window = min(5, n - 1)
            external_factors = ForecastExecutionService._get_external_factor_columns(
                data,
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level
            )
            
            if external_factors:
                logger.info(f"Random Forest: Using external factors: {external_factors}")
                X, y_target = ForecastExecutionService._extract_features_with_factors(data, window, external_factors)
            else:
                # Build lagged features without external factors
                X = []
                y_target = []
                for i in range(window, n):
                    lags = y[i-window:i]
                    time_idx = i
                    features = list(lags) + [time_idx]
                    X.append(features)
                    y_target.append(y[i])
                X = np.array(X)
                y_target = np.array(y_target)

            if len(X) < 5:
                raise ValueError("Insufficient data for Random Forest training")

            # Train Random Forest model on ALL data (no internal split)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y_target)

            # Generate forecast
            forecast = []
            recent_lags = list(y[-window:])
            
            last_factor_values = {}
            if external_factors and len(data) > 0:
                for factor_name in external_factors:
                    if factor_name in data.columns:
                        factor_value_raw = data[factor_name].iloc[-1]
                        if pd.isna(factor_value_raw):
                            val = 0.0
                        else:
                            try:
                                val = float(factor_value_raw)
                            except Exception:
                                try:
                                    codes, uniques = pd.factorize(data[factor_name].fillna('__NA__'))
                                    val = float(codes[-1])
                                except Exception:
                                    val = float(abs(hash(str(factor_value_raw))) % 1000000) / 1000.0
                        last_factor_values[factor_name] = val

            for i in range(periods):
                if external_factors:
                    features = ForecastExecutionService._prepare_future_features_with_factors(
                        recent_lags, n, i, window, external_factors, last_factor_values
                    )
                else:
                    features = np.array(recent_lags + [n + i])
                
                pred = model.predict([features])[0]
                pred = max(0, pred)
                forecast.append(pred)

                recent_lags = recent_lags[1:] + [pred]

            forecast = np.array(forecast)

            # Calculate metrics on training data
            predicted = model.predict(X)
            metrics = ForecastExecutionService.calculate_metrics(y_target, predicted)

            return forecast, metrics

        except Exception as e:
            logger.warning(f"Random Forest failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)
    
    @staticmethod
    def calculate_metrics(actual: Any, predicted: Any, selected_metrics: List[str] = None) -> Dict[str, float]:
        """
        Calculate selected accuracy metrics with extensive error handling and logging.

        Args:
            actual: Actual values array
            predicted: Predicted values array
            selected_metrics: List of metrics to calculate. Options: ['mae', 'rmse', 'mape', 'accuracy']
                             If None, calculates all metrics (backward compatibility)

        Returns:
            Dictionary with calculated metrics
        """
        try:
            # Ensure they are numpy arrays for shape access and calculations
            actual = np.asarray(actual)
            predicted = np.asarray(predicted)

            logger.debug(f"calculate_metrics called: actual shape={actual.shape}, predicted shape={predicted.shape}")

            # Ensure same length
            min_len = min(len(actual), len(predicted))
            if len(actual) != len(predicted):
                logger.warning(f"Length mismatch: truncating to {min_len}")
                actual = actual[:min_len]
                predicted = predicted[:min_len]

            # Ensure float type
            actual = actual.astype(float)
            predicted = predicted.astype(float)

            # Check for NaN/Inf
            if np.any(np.isnan(actual)) or np.any(np.isnan(predicted)):
                logger.error("NaN values detected in input arrays")
                return {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 100.0}

            if np.any(np.isinf(actual)) or np.any(np.isinf(predicted)):
                logger.error("Inf values detected in input arrays")
                return {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 100.0}

            # Default to all metrics if not specified
            if selected_metrics is None:
                selected_metrics = ['mae', 'rmse', 'mape', 'accuracy']

            result = {}
            actual_mean = float(np.mean(actual)) if len(actual) > 0 else 0

            # Calculate MAE if requested
            if 'mae' in selected_metrics:
                mae = float(np.mean(np.abs(actual - predicted)))
                result['mae'] = round(mae, 2)
                # Calculate normalized MAE accuracy
                if actual_mean > 0:
                    result['mae_accuracy'] = round(max(0.0, 100.0 - (mae / actual_mean * 100)), 2)
                else:
                    result['mae_accuracy'] = 0.0

            # Calculate RMSE if requested
            if 'rmse' in selected_metrics:
                rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
                result['rmse'] = round(rmse, 2)
                # Calculate normalized RMSE accuracy
                if actual_mean > 0:
                    result['rmse_accuracy'] = round(max(0.0, 100.0 - (rmse / actual_mean * 100)), 2)
                else:
                    result['rmse_accuracy'] = 0.0

            # Calculate MAPE if requested
            if 'mape' in selected_metrics or 'accuracy' in selected_metrics:
                mask = actual != 0
                num_nonzero = int(np.sum(mask))

                logger.debug(f"Non-zero actuals: {num_nonzero}/{len(actual)}")

                if num_nonzero > 0:
                    pct_errors = np.abs((actual[mask] - predicted[mask]) / actual[mask])
                    mape = float(np.mean(pct_errors) * 100)
                    logger.debug(f"MAPE calculated from {num_nonzero} non-zero values: {mape:.2f}%")
                else:
                    mape = 100.0
                    logger.warning("All actual values are zero - MAPE set to 100%")

                # Cap MAPE
                mape = min(float(mape), 100.0)
                result['mape'] = round(mape, 2)

            # Calculate accuracy if requested (derived from MAPE)
            if 'accuracy' in selected_metrics:
                if 'mape' not in result:
                    # Calculate MAPE if not already calculated
                    mask = actual != 0
                    if np.sum(mask) > 0:
                        pct_errors = np.abs((actual[mask] - predicted[mask]) / actual[mask])
                        mape = float(np.mean(pct_errors) * 100)
                    else:
                        mape = 100.0
                    mape = min(float(mape), 100.0)

                accuracy = max(0.0, 100.0 - result.get('mape', mape))
                result['accuracy'] = round(accuracy, 2)

            logger.debug(f"Calculated metrics: {result}")

            return result

        except Exception as e:
            logger.error(f"Exception in calculate_metrics: {str(e)}", exc_info=True)
            return {'accuracy': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mape': 100.0}

    @staticmethod
    def _prepare_external_factors(
        tenant_id: str,
        database_name: str,
        selected_factors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Prepare external factors data with optional factor selection.
        
        FIXED: No longer filters by date range - fetches ALL available dates
        so that factors can be properly merged with both historical data 
        (for training) and forecast period data (for prediction).
        
        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            selected_factors: List of factor names to include (None/empty = no factors)
            
        Returns:
            DataFrame with factors pivoted by name, containing ALL available dates
        """
        try:
            # Use enhanced service to get factors WITHOUT date filtering
            df_pivot = ExternalFactorsService.get_factors_for_forecast_run(
                tenant_id=tenant_id,
                database_name=database_name,
                selected_factors=selected_factors,
                start_date=None,  # ✅ FIXED: No date filtering
                end_date=None     # ✅ FIXED: No date filtering
            )

            if not df_pivot.empty:
                logger.info(
                    f"Loaded external factors for all available dates: "
                    f"{len(df_pivot)} records, "
                    f"{len(df_pivot.columns) - 1} factors, "
                    f"date range: {df_pivot['date'].min()} to {df_pivot['date'].max()}"
                )
            else:
                logger.info("No external factors loaded (none selected or no data available)")

            return df_pivot

        except Exception as e:
            logger.warning(f"Could not load external factors: {str(e)}")
            return pd.DataFrame()


    @staticmethod
    def _calculate_periods(start_date: date, end_date: date, interval: str) -> int:
        """Calculate number of forecast periods."""
        if interval == 'WEEKLY':
            return (end_date - start_date).days // 7
        elif interval == 'MONTHLY':
            return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1
        elif interval == 'QUARTERLY':
            months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
            return months // 3 + 1
        elif interval == 'YEARLY':
            return end_date.year - start_date.year + 1
        else:
            return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1

    @staticmethod
    def _generate_forecast_dates(start_date: date, periods: int, interval: str) -> List[date]:
        """Generate list of forecast dates."""
        dates = []
        current = start_date
        
        for _ in range(periods):
            dates.append(current)
            
            if interval == 'WEEKLY':
                current += relativedelta(weeks=1)
            elif interval == 'MONTHLY':
                current += relativedelta(months=1)
            elif interval == 'QUARTERLY':
                current += relativedelta(months=3)
            elif interval == 'YEARLY':
                current += relativedelta(years=1)
            else:
                current += relativedelta(months=1)
        
        return dates

    @staticmethod
    def _store_forecast_results_v2(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str,
        version_id: str,
        mapping_id: str,
        algorithm_id: int,
        test_dates: List[date],
        test_actuals: np.ndarray,
        test_forecast: np.ndarray,
        test_metrics: Dict[str, float],
        future_dates: List[date],
        future_forecast: np.ndarray,
        user_email: str,
        selected_metrics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Store forecast results with three types:
        1. testing_actual - actual values from test period
        2. testing_forecast - predicted values for test period
        3. future_forecast - predictions for future period
        """
        from app.core.database import get_db_manager
        from psycopg2.extras import Json
        import uuid
        
        db_manager = get_db_manager()
        results = []
        
        # Ensure arrays are numpy arrays and handle NaN
        test_actuals = np.nan_to_num(np.asarray(test_actuals, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        test_forecast = np.nan_to_num(np.asarray(test_forecast, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        future_forecast = np.nan_to_num(np.asarray(future_forecast, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip values to reasonable bounds
        max_value = 99999999.999999
        test_actuals = np.clip(test_actuals, 0, max_value)
        test_forecast = np.clip(test_forecast, 0, max_value)
        future_forecast = np.clip(future_forecast, 0, max_value)
        
        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                # ====================================================================
                # 1. Store testing_actual records
                # ====================================================================
                for test_date, actual_value in zip(test_dates, test_actuals):
                    result_id = str(uuid.uuid4())
                    actual_float = round(float(actual_value), 4)
                    
                    cursor.execute("""
                        INSERT INTO forecast_results
                        (result_id, forecast_run_id, version_id, mapping_id, algorithm_id,
                         date, value, type, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        result_id,
                        forecast_run_id,
                        version_id,
                        mapping_id,
                        algorithm_id,
                        test_date,
                        actual_float,
                        'testing_actual',
                        user_email
                    ))
                    
                    results.append({
                        'result_id': result_id,
                        'date': test_date.isoformat() if isinstance(test_date, date) else str(test_date),
                        'value': actual_float,
                        'type': 'testing_actual'
                    })
                
                # ====================================================================
                # 2. Store testing_forecast records (with accuracy metrics)
                # ====================================================================
                accuracy_metric = 0.0
                primary_metric = selected_metrics[0] if selected_metrics else 'accuracy'
                
                if test_metrics:
                    if primary_metric == 'mape':
                        accuracy_metric = 100.0 - test_metrics.get('mape', 0)
                    elif primary_metric == 'mae':
                        accuracy_metric = test_metrics.get('mae_accuracy', 0)
                    elif primary_metric == 'rmse':
                        accuracy_metric = test_metrics.get('rmse_accuracy', 0)
                    else:
                        accuracy_metric = test_metrics.get('accuracy', 0)
                        
                accuracy_metric = min(round(float(accuracy_metric), 2), 999.99)
                
                for test_date, forecast_value in zip(test_dates, test_forecast):
                    result_id = str(uuid.uuid4())
                    forecast_float = round(float(forecast_value), 4)
                    
                    cursor.execute("""
                        INSERT INTO forecast_results
                        (result_id, forecast_run_id, version_id, mapping_id, algorithm_id,
                         date, value, type, accuracy_metric, metric_type, metadata, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        result_id,
                        forecast_run_id,
                        version_id,
                        mapping_id,
                        algorithm_id,
                        test_date,
                        forecast_float,
                        'testing_forecast',
                        accuracy_metric,
                        'MAPE',
                        Json({'test_metrics': test_metrics}),
                        user_email
                    ))
                    
                    results.append({
                        'result_id': result_id,
                        'date': test_date.isoformat() if isinstance(test_date, date) else str(test_date),
                        'value': forecast_float,
                        'type': 'testing_forecast',
                        'accuracy_metric': accuracy_metric
                    })
                
                # ====================================================================
                # 3. Store future_forecast records (with confidence intervals)
                # ====================================================================
                for future_date, forecast_value in zip(future_dates, future_forecast):
                    result_id = str(uuid.uuid4())
                    forecast_float = round(float(forecast_value), 4)
                    
                    # Calculate simple confidence intervals (±10% for demonstration)
                    ci_lower = round(forecast_float * 0.9, 4)
                    ci_upper = round(forecast_float * 1.1, 4)
                    
                    cursor.execute("""
                        INSERT INTO forecast_results
                        (result_id, forecast_run_id, version_id, mapping_id, algorithm_id,
                         date, value, type, confidence_interval_lower, confidence_interval_upper,
                         confidence_level, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        result_id,
                        forecast_run_id,
                        version_id,
                        mapping_id,
                        algorithm_id,
                        future_date,
                        forecast_float,
                        'future_forecast',
                        ci_lower,
                        ci_upper,
                        '90%',
                        user_email
                    ))
                    
                    results.append({
                        'result_id': result_id,
                        'date': future_date.isoformat() if isinstance(future_date, date) else str(future_date),
                        'value': forecast_float,
                        'type': 'future_forecast',
                        'confidence_interval_lower': ci_lower,
                        'confidence_interval_upper': ci_upper
                    })
                
                conn.commit()
                logger.info(
                    f"Stored forecast results: {len(test_actuals)} testing_actual, "
                    f"{len(test_forecast)} testing_forecast, {len(future_forecast)} future_forecast"
                )
                
            finally:
                cursor.close()
        
        return results
    @staticmethod
    def _update_algorithm_status(
        database_name: str,
        mapping_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> None:
        """Update algorithm execution status."""
        db_manager = get_db_manager()
        
        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                if status == 'Running':
                    cursor.execute("""
                        UPDATE forecast_algorithms_mapping
                        SET execution_status = %s,
                            started_at = %s,
                            updated_at = %s
                        WHERE mapping_id = %s
                    """, (status, datetime.utcnow(), datetime.utcnow(), mapping_id))
                elif status in ['Completed', 'Failed']:
                    cursor.execute("""
                        UPDATE forecast_algorithms_mapping
                        SET execution_status = %s,
                            completed_at = %s,
                            error_message = %s,
                            updated_at = %s
                        WHERE mapping_id = %s
                    """, (status, datetime.utcnow(), error_message, datetime.utcnow(), mapping_id))
                
                conn.commit()
            finally:
                cursor.close()

    @staticmethod
    def generate_forecast(
        historical_data: pd.DataFrame,
        config: Dict[str, Any],
        process_log: List[str] = None,
        tenant_id: str = None,
        database_name: str = None,
        aggregation_level: str = None,
        selected_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate forecast using best_fit algorithm selection.
        Runs multiple algorithms in parallel and selects the best performing one.
        
        Args:
            historical_data: Historical data with quantity column
            config: Configuration dict with interval and other settings
            process_log: Optional list to append process logs
            tenant_id: Tenant identifier (for dynamic field detection)
            database_name: Database name (for dynamic field detection)
            aggregation_level: Current aggregation level (e.g., "product")
            selected_metrics: List of metrics to use for comparison
            
        Returns:
            Dictionary with forecast results including selected algorithm, metrics, and predictions
        """
        if process_log is None:
            process_log = []
        
        if selected_metrics is None:
            selected_metrics = ['mape', 'accuracy']
            
        primary_metric = selected_metrics[0] if selected_metrics else 'accuracy'
        # Define if a higher value is better for the primary metric
        higher_is_better = primary_metric in ['accuracy']
        
        process_log.append("Loading data for forecasting...")
        
        if len(historical_data) < 2:
            raise ValueError("Insufficient data for forecasting")
        
        process_log.append(f"Data loaded: {len(historical_data)} records")
        process_log.append(f"Running best fit algorithm selection based on {primary_metric}...")
        
        # Define all available algorithms
        available_algorithms = [
            "linear_regression",
            "polynomial_regression",
            "exponential_smoothing",
            "holt_winters",
            "arima",
            "prophet",
            "lstm",
            "xgboost",
            "svr",
            "knn",
            "gaussian_process",
            "mlp_neural_network",
            "simple_moving_average",
            "seasonal_decomposition",
            "moving_average",
            "sarima",
            "random_forest"
        ]
        
        algorithm_results = []
        best_model = None
        best_algorithm = None
        best_metrics = None
        
        # Use ThreadPoolExecutor for parallel execution
        max_workers = min(len(available_algorithms), settings.NUMBER_OF_THREADS)
        process_log.append(f"Starting parallel execution with {max_workers} workers for {len(available_algorithms)} algorithms...")
        
        # Determine forecast periods
        periods = config.get('periods', 12)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all algorithm tasks
            future_to_algorithm = {}
            
            for algorithm in available_algorithms:
                future = executor.submit(
                    ForecastExecutionService._run_algorithm_safe,
                    algorithm,
                    historical_data.copy(),
                    periods,
                    'total_quantity',
                    tenant_id,
                    database_name,
                    aggregation_level,
                    selected_metrics=selected_metrics
                )
                future_to_algorithm[future] = algorithm
            
            # Collect results as they complete
            for future in as_completed(future_to_algorithm):
                algorithm_name = future_to_algorithm[future]
                try:
                    result = future.result()
                    algorithm_results.append(result)
                    
                    metric_value = result.get(primary_metric, 0)
                    process_log.append(
                        f"✅ Algorithm {algorithm_name} completed with {primary_metric}: {metric_value:.2f}{'%' if primary_metric in ['accuracy', 'mape'] else ''}"
                    )
                    
                    # Track best performing algorithm based on selected metric
                    is_better = False
                    if best_metrics is None:
                        is_better = True
                    else:
                        current_best_val = best_metrics.get(primary_metric)
                        if higher_is_better:
                            if metric_value > current_best_val:
                                is_better = True
                        else:
                            if metric_value < current_best_val:
                                is_better = True
                    
                    if is_better:
                        best_metrics = {
                            'accuracy': result.get('accuracy', 0),
                            'mae': result.get('mae', 0),
                            'mae_accuracy': result.get('mae_accuracy', 0),
                            'rmse': result.get('rmse', 0),
                            'rmse_accuracy': result.get('rmse_accuracy', 0),
                            'mape': result.get('mape', 100)
                        }
                        best_model = result
                        best_algorithm = algorithm_name
                        
                except Exception as exc:
                    process_log.append(f"❌ Algorithm {algorithm_name} failed: {str(exc)}")
                    logger.warning(f"Algorithm {algorithm_name} failed: {str(exc)}")
        
        # Filter out failed results
        successful_results = [res for res in algorithm_results if res.get(primary_metric) is not None and res.get('forecast')]

        if not successful_results:
            fallback_results = [res for res in algorithm_results if res.get('forecast')]
            if fallback_results:
                successful_results = fallback_results
                process_log.append(f"No algorithm produced {primary_metric}; using algorithms with non-empty forecasts as fallback")
            else:
                process_log.append("All algorithms failed to produce forecasts; using simple moving average fallback")
                sma_forecast, sma_metrics = ForecastExecutionService.simple_moving_average(historical_data, periods=periods)
                return {
                    'selected_algorithm': 'simple_moving_average (fallback)',
                    'accuracy': sma_metrics.get('accuracy', 0),
                    'mae': sma_metrics.get('mae', 0),
                    'rmse': sma_metrics.get('rmse', 0),
                    'mape': sma_metrics.get('mape'),
                    'forecast': sma_forecast.tolist() if isinstance(sma_forecast, np.ndarray) else sma_forecast,
                    'all_algorithms': algorithm_results,
                    'process_log': process_log
                }
        
        process_log.append(
            f"Parallel execution completed. {len(successful_results)} algorithms succeeded, "
            f"{len(algorithm_results) - len(successful_results)} failed."
        )
        
        # Ensemble: average forecast of top 3 algorithms by selected metric
        if higher_is_better:
            top3 = sorted(successful_results, key=lambda x: -x.get(primary_metric, 0))[:3]
        else:
            top3 = sorted(successful_results, key=lambda x: x.get(primary_metric, float('inf')))[:3]
        
        if len(top3) >= 2:
            process_log.append(f"Creating ensemble from top {len(top3)} algorithms...")
            
            # Average the forecast values from top 3
            ensemble_forecast = []
            for i in range(len(top3[0]['forecast'])):
                values = [algo['forecast'][i] for algo in top3 if i < len(algo['forecast'])]
                if values:
                    ensemble_forecast.append(np.mean(values))
            
            # Average the test forecast values from top 3
            ensemble_test_forecast = []
            if top3[0].get('test_forecast'):
                for i in range(len(top3[0]['test_forecast'])):
                    values = [algo['test_forecast'][i] for algo in top3 if i < len(algo.get('test_forecast', []))]
                    if values:
                        ensemble_test_forecast.append(np.mean(values))

            # Average the metrics
            ensemble_result = {
                'algorithm': 'Ensemble (Top 3 Avg)',
                'forecast': ensemble_forecast,
                'test_forecast': ensemble_test_forecast,
                'accuracy': np.mean([algo.get('accuracy', 0) for algo in top3]),
                'mae': np.mean([algo.get('mae', 0) for algo in top3]),
                'mae_accuracy': np.mean([algo.get('mae_accuracy', 0) for algo in top3]),
                'rmse': np.mean([algo.get('rmse', 0) for algo in top3]),
                'rmse_accuracy': np.mean([algo.get('rmse_accuracy', 0) for algo in top3]),
                'mape': np.mean([algo.get('mape', 0) for algo in top3])
            }
            algorithm_results.append(ensemble_result)
            
            # Update best model if ensemble is better
            is_ensemble_better = False
            ensemble_val = ensemble_result.get(primary_metric)
            current_best_val = best_metrics.get(primary_metric)
            
            if ensemble_val is not None and current_best_val is not None:
                if higher_is_better:
                    if ensemble_val > current_best_val:
                        is_ensemble_better = True
                else:
                    if ensemble_val < current_best_val:
                        is_ensemble_better = True
            
            if is_ensemble_better:
                best_model = ensemble_result
                best_algorithm = 'ensemble'
                best_metrics = {
                    'accuracy': ensemble_result.get('accuracy', 0),
                    'mae': ensemble_result.get('mae', 0),
                    'mae_accuracy': ensemble_result.get('mae_accuracy', 0),
                    'rmse': ensemble_result.get('rmse', 0),
                    'rmse_accuracy': ensemble_result.get('rmse_accuracy', 0),
                    'mape': ensemble_result.get('mape', 0)
                }
        
        process_log.append(f"Best algorithm selected: {best_algorithm} ({primary_metric}: {best_metrics.get(primary_metric):.2f})")
        
        # Determine summary accuracy percentage for DB storage
        if primary_metric == 'mape':
            summary_value = 100.0 - best_metrics.get('mape', 0)
        elif primary_metric == 'mae':
            summary_value = best_metrics.get('mae_accuracy', 0)
        elif primary_metric == 'rmse':
            summary_value = best_metrics.get('rmse_accuracy', 0)
        else:
            summary_value = best_metrics.get('accuracy', 0)
            
        return {
            'selected_algorithm': f"{best_algorithm} (Best Fit)",
            'accuracy': summary_value,
            'mae': best_metrics.get('mae', 0),
            'rmse': best_metrics.get('rmse', 0),
            'mape': best_metrics.get('mape'),
            'forecast': best_model['forecast'],
            'test_forecast': best_model.get('test_forecast', []),
            'all_algorithms': algorithm_results,
            'process_log': process_log
        }
    
    @staticmethod
    def _split_train_test_dataframe(
        data: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if data is None or data.empty:
            return data, pd.DataFrame()
        n = len(data)
        if n < 4:
            return data, pd.DataFrame()
        split_index = max(2, int(n * train_ratio))
        if split_index >= n:
            split_index = n - 1
        return data.iloc[:split_index].copy(), data.iloc[split_index:].copy()
    
    @staticmethod
    def _run_algorithm_safe(
        algorithm_name: str,
        data: pd.DataFrame,
        periods: int,
        target_column: str = 'total_quantity',
        tenant_id: str = None,
        database_name: str = None,
        aggregation_level: str = None,
        selected_metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Safely run an algorithm and return results.

        Args:
            algorithm_name: Name of the algorithm to run
            data: Historical data (must have target_column)
            periods: Number of periods to forecast
            target_column: Name of the target column to forecast
            tenant_id: Tenant identifier (for dynamic field detection)
            database_name: Database name (for dynamic field detection)
            aggregation_level: Current aggregation level (e.g., "product")
            selected_metrics: List of metrics to calculate

        Returns:
            Dictionary with algorithm results
        """
        if selected_metrics is None:
            selected_metrics = ['mape', 'accuracy']

        # Ensure target_column is properly set
        if not target_column:
            target_column = 'total_quantity'

        try:
            # ✅ CRITICAL: Ensure no duplicate columns before proceeding
            # Duplicate columns (like multiple 'total_quantity' or 'period') can crash algorithms
            if data.columns.duplicated().any():
                logger.info(f"Algorithm {algorithm_name}: Found duplicate columns {data.columns[data.columns.duplicated()].unique().tolist()}, keeping only first occurrence")
                data = data.loc[:, ~data.columns.duplicated()].copy()

            # Debug: Log data information
            logger.info(f"Algorithm {algorithm_name}: Forecast periods: {periods}")
            logger.info(f"Algorithm {algorithm_name}: Data shape: {data.shape}")

            # Ensure we have target_column (rename if needed)
            if 'quantity' in data.columns and target_column not in data.columns:
                data = data.rename(columns={'quantity': target_column})
            elif 'quantity' in data.columns and target_column in data.columns:
                # If both exist, drop the old target_column first to avoid duplicates
                data = data.drop(columns=[target_column])
                data = data.rename(columns={'quantity': target_column})
            
            algorithm_map = {
                "linear_regression": ForecastExecutionService.linear_regression_forecast,
                "polynomial_regression": ForecastExecutionService.polynomial_regression_forecast,
                "exponential_smoothing": ForecastExecutionService.exponential_smoothing_forecast,
                "holt_winters": ForecastExecutionService.holt_winters_forecast,
                "arima": ForecastExecutionService.arima_forecast,
                "prophet": ForecastExecutionService.prophet_forecast,
                "lstm": ForecastExecutionService.lstm_forecast,
                "xgboost": ForecastExecutionService.xgboost_forecast,
                "svr": ForecastExecutionService.svr_forecast,
                "knn": ForecastExecutionService.knn_forecast,
                "gaussian_process": ForecastExecutionService.gaussian_process_forecast,
                "mlp_neural_network": ForecastExecutionService.mlp_neural_network_forecast,
                "simple_moving_average": ForecastExecutionService.simple_moving_average,
                "seasonal_decomposition": ForecastExecutionService.seasonal_decomposition_forecast,
                "moving_average": ForecastExecutionService.moving_average_forecast,
                "sarima": ForecastExecutionService.sarima_forecast,
                "random_forest": ForecastExecutionService.random_forest_forecast
            }
            algorithm_func = algorithm_map.get(algorithm_name)
            if algorithm_func is None:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            def execute(input_data: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, Dict[str, float]]:
                """Execute algorithm and ensure it returns exactly 2 values"""
                target_data = input_data if input_data is not None else data
                if target_data is None or target_data.empty:
                    target_data = data
                
                # Call the algorithm with tenant context for ML algorithms
                if algorithm_name in ['xgboost', 'svr', 'knn', 'random_forest']:
                    result = algorithm_func(
                        target_data, 
                        max(1, horizon),
                        tenant_id=tenant_id,
                        database_name=database_name,
                        aggregation_level=aggregation_level
                    )
                else:
                    result = algorithm_func(target_data, max(1, horizon))
                
                # Ensure we have exactly 2 return values
                if isinstance(result, tuple):
                    if len(result) == 2:
                        forecast, metrics = result
                        return forecast, metrics
                    else:
                        raise ValueError(f"Algorithm {algorithm_name} returned {len(result)} values, expected 2")
                else:
                    raise ValueError(f"Algorithm {algorithm_name} did not return a tuple")
            
            # Split data for evaluation
            train_df, test_df = ForecastExecutionService._split_train_test_dataframe(data)
            eval_data = train_df if train_df is not None and not train_df.empty else data
            eval_periods = len(test_df)
            
            metrics: Dict[str, float]
            test_forecast_list: List[float] = []
            
            # Evaluate on test data if available
            if eval_periods > 0:
                eval_forecast, eval_metrics = execute(eval_data, eval_periods)
                actual_eval = test_df[target_column].values
                predicted_eval = np.asarray(eval_forecast)[:len(actual_eval)]
                metrics = ForecastExecutionService.calculate_metrics(
                    actual_eval, 
                    predicted_eval,
                    selected_metrics=selected_metrics
                )
                test_forecast_list = predicted_eval.tolist()
            else:
                # Fallback: use training metrics
                fallback_horizon = min(max(1, periods), max(1, len(eval_data)))
                _, metrics = execute(eval_data, fallback_horizon)
            
            # Generate final forecast on full data
            forecast, _ = execute(data, periods)
            
            # Convert forecast to list
            forecast_list = forecast.tolist() if isinstance(forecast, np.ndarray) else forecast
            
            return {
                'algorithm': algorithm_name,
                'forecast': forecast_list,
                'test_forecast': test_forecast_list,
                **metrics
            }
            
        except Exception as e:
            logger.error(f"Error running algorithm {algorithm_name}: {str(e)}", exc_info=True)
            # Return failed result
            return {
                'algorithm': algorithm_name,
                'forecast': [],
                'accuracy': 0,
                'mae': 999.0,
                'rmse': 999.0,
                'mape': 100.0,
                'error': str(e)
            }