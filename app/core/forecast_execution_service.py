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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from psycopg2.extras import Json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, ValidationException, NotFoundException
from app.core.forecasting_service import ForecastingService
from app.core.external_factors_service import ExternalFactorsService

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
                    validated_params['order'] = [max(0, int(x)) for x in order]

            elif algorithm_id == 3:  # Polynomial Regression
                # degree: 1-5
                if 'degree' in validated_params:
                    degree = validated_params['degree']
                    validated_params['degree'] = max(1, min(5, int(degree)))
                    logger.info(f"Validated polynomial degree: {validated_params['degree']}")

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

            elif algorithm_id == 5:  # Enhanced Exponential Smoothing
                # alphas: list of floats 0.0-1.0
                if 'alphas' in validated_params:
                    alphas = validated_params['alphas']
                    if isinstance(alphas, list):
                        validated_params['alphas'] = [max(0.0, min(1.0, float(a))) for a in alphas]
                    else:
                        validated_params['alphas'] = [max(0.0, min(1.0, float(alphas)))]

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
                    validated_params['season_length'] = max(2, int(validated_params['season_length']))

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
                    validated_params['n_neighbors'] = max(1, int(validated_params['n_neighbors']))

            # Log validated parameters
            logger.info(f"Validated parameters for algorithm {algorithm_id}: {validated_params}")

            return validated_params

        except (ValueError, TypeError) as e:
            raise ValidationException(f"Invalid parameter format for algorithm {algorithm_id}: {str(e)}")

    @staticmethod
    def execute_forecast_run(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str,
        user_email: str
    ) -> Dict[str, Any]:
        """
        Execute a forecast run with all mapped algorithms.

        Args:
            tenant_id: Tenant identifier
            database_name: Tenant's database name
            forecast_run_id: Forecast run identifier
            user_email: User executing the forecast

        Returns:
            Execution summary
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting forecast execution for run: {forecast_run_id}, tenant: {tenant_id}, database: {database_name}, user: {user_email}")

        db_manager = get_db_manager()

        try:
            # Get forecast run details
            logger.debug(f"Retrieving forecast run details for run: {forecast_run_id}")
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
            
            # Get historical data
            historical_data = ForecastingService.prepare_aggregated_data(
                tenant_id=tenant_id,
                database_name=database_name,
                aggregation_level=aggregation_level,
                interval=interval,
                filters=filters
            )
            
            if historical_data.empty:
                raise ValidationException("No historical data available for forecasting")
            
            logger.info(f"Prepared {len(historical_data)} historical records")
            
            # Get external factors
            external_factors = ForecastExecutionService._prepare_external_factors(
                tenant_id,
                database_name,
                forecast_run['forecast_start'],
                forecast_run['forecast_end']
            )
            
            # Execute each algorithm
            algorithms = forecast_run.get('algorithms', [])
            total_records = 0
            processed_records = 0
            failed_records = 0
            
            for algo in sorted(algorithms, key=lambda x: x['execution_order']):
                try:
                    logger.info(f"Executing algorithm: {algo['algorithm_name']}")
                    
                    # Add version_id to algorithm mapping
                    algo['version_id'] = forecast_run['version_id']
                    
                    # Execute algorithm
                    results = ForecastExecutionService._execute_algorithm(
                        tenant_id=tenant_id,
                        database_name=database_name,
                        forecast_run_id=forecast_run_id,
                        algorithm_mapping=algo,
                        historical_data=historical_data.copy(),
                        external_factors=external_factors,
                        forecast_start=forecast_run['forecast_start'],
                        forecast_end=forecast_run['forecast_end'],
                        interval=interval,
                        user_email=user_email
                    )
                    
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
        user_email: str
    ) -> List[Dict[str, Any]]:
        """Execute a single algorithm and store results."""

        mapping_id = algorithm_mapping['mapping_id']
        algorithm_id = algorithm_mapping['algorithm_id']
        algorithm_name = algorithm_mapping['algorithm_name']
        custom_params = algorithm_mapping.get('custom_parameters') or {}  # Handle None

        # Validate custom parameters
        try:
            custom_params = ForecastExecutionService.validate_algorithm_parameters(algorithm_id, custom_params)
        except ValidationException as e:
            logger.error(f"Parameter validation failed for algorithm {algorithm_name}: {str(e)}")
            raise

        start_time = datetime.utcnow()
        logger.info(f"Starting algorithm execution: {algorithm_name} (ID: {algorithm_id}, Mapping: {mapping_id}) for forecast run: {forecast_run_id}")
        logger.debug(f"Validated algorithm parameters: {custom_params}")

        # Update algorithm status to Running
        ForecastExecutionService._update_algorithm_status(
            database_name, mapping_id, 'Running'
        )
        
        try:
            # Convert dates
            forecast_start_date = datetime.fromisoformat(forecast_start).date()
            forecast_end_date = datetime.fromisoformat(forecast_end).date()
            
            # Calculate number of periods
            periods = ForecastExecutionService._calculate_periods(
                forecast_start_date,
                forecast_end_date,
                interval
            )
            
            # Merge external factors into historical data
            if not external_factors.empty:
                historical_data = historical_data.merge(
                    external_factors,
                    left_on='period',
                    right_on='date',
                    how='left'
                )
            
            # Route to appropriate algorithm based on algorithm_id
            if algorithm_id == 1:  # ARIMA
                forecast, metrics = ForecastExecutionService.arima_forecast(
                    data=historical_data,
                    periods=periods,
                    order=custom_params.get('order', [1, 1, 1])
                )
            elif algorithm_id == 2:  # Linear Regression
                forecast, metrics = ForecastExecutionService.linear_regression_forecast(
                    data=historical_data,
                    periods=periods
                )
            elif algorithm_id == 3:  # Polynomial Regression
                forecast, metrics = ForecastExecutionService.polynomial_regression_forecast(
                    data=historical_data,
                    periods=periods,
                    degree=custom_params.get('degree', 2)
                )
            elif algorithm_id == 4:  # Exponential Smoothing
                forecast, metrics = ForecastExecutionService.exponential_smoothing_forecast(
                    data=historical_data,
                    periods=periods,
                    alphas=custom_params.get('alphas', [custom_params.get('alpha', 0.3)])
                )
            elif algorithm_id == 5:  # Enhanced Exponential Smoothing
                forecast, metrics = ForecastExecutionService.exponential_smoothing_forecast(
                    data=historical_data,
                    periods=periods,
                    alphas=custom_params.get('alphas', [0.1, 0.3, 0.5])
                )
            elif algorithm_id == 6:  # Holt Winters
                forecast, metrics = ForecastExecutionService.holt_winters_forecast(
                    data=historical_data,
                    periods=periods,
                    season_length=custom_params.get('season_length', 12),
                    alpha=custom_params.get('alpha', 0.3),
                    beta=custom_params.get('beta', 0.1),
                    gamma=custom_params.get('gamma', 0.1)
                )
            elif algorithm_id == 7:  # Prophet
                # Placeholder for Prophet implementation
                forecast, metrics = ForecastExecutionService.simple_moving_average(
                    data=historical_data,
                    periods=periods,
                    window=custom_params.get('window', 3)
                )
            elif algorithm_id == 8:  # LSTM Neural Network
                # Placeholder for LSTM implementation
                forecast, metrics = ForecastExecutionService.simple_moving_average(
                    data=historical_data,
                    periods=periods,
                    window=custom_params.get('window', 3)
                )
            elif algorithm_id == 9:  # XGBoost
                forecast, metrics = ForecastExecutionService.xgboost_forecast(
                    data=historical_data,
                    periods=periods,
                    n_estimators=custom_params.get('n_estimators', 100),
                    max_depth=custom_params.get('max_depth', 6),
                    learning_rate=custom_params.get('learning_rate', 0.1)
                )
            elif algorithm_id == 10:  # SVR
                forecast, metrics = ForecastExecutionService.svr_forecast(
                    data=historical_data,
                    periods=periods,
                    C=custom_params.get('C', 1.0),
                    epsilon=custom_params.get('epsilon', 0.1),
                    kernel=custom_params.get('kernel', 'rbf')
                )
            elif algorithm_id == 11:  # KNN
                forecast, metrics = ForecastExecutionService.knn_forecast(
                    data=historical_data,
                    periods=periods,
                    n_neighbors=custom_params.get('n_neighbors', 5)
                )
            else:
                # Default to simple moving average
                forecast, metrics = ForecastExecutionService.simple_moving_average(
                    data=historical_data,
                    periods=periods,
                    window=custom_params.get('window', 3)
                )
            
            # Generate forecast dates
            forecast_dates = ForecastExecutionService._generate_forecast_dates(
                forecast_start_date,
                periods,
                interval
            )
            
            # Store results
            results = ForecastExecutionService._store_forecast_results(
                tenant_id=tenant_id,
                database_name=database_name,
                forecast_run_id=forecast_run_id,
                version_id=algorithm_mapping.get('version_id'),
                mapping_id=mapping_id,
                algorithm_id=algorithm_id,
                forecast_dates=forecast_dates,
                forecast_values=forecast,
                metrics=metrics,
                user_email=user_email
            )
            
            # Update algorithm status to Completed
            ForecastExecutionService._update_algorithm_status(
                database_name, mapping_id, 'Completed'
            )
            
            return results
            
        except Exception as e:
            ForecastExecutionService._update_algorithm_status(
                database_name, mapping_id, 'Failed', str(e)
            )
            raise

    @staticmethod
    def linear_regression_forecast(data: pd.DataFrame, periods: int) -> Tuple[np.ndarray, Dict[str, float]]:
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
        
        return forecast, metrics

    @staticmethod
    def arima_forecast(data: pd.DataFrame, periods: int, order: List[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
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
            
            return forecast, metrics
            
        except ImportError:
            logger.warning("statsmodels not available, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)



    @staticmethod
    def polynomial_regression_forecast(data: pd.DataFrame, periods: int, degree: int = 2) -> tuple:
        """Polynomial regression forecasting"""
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

        n = len(y)
        if n < 2:
            raise ValueError("Need at least 2 historical data points")

        # Validate degree parameter
        if degree < 1 or degree > 5:
            logger.warning(f"Invalid degree {degree}, using default degree 2")
            degree = 2

        # Ensure degree doesn't exceed available data points
        degree = min(degree, n - 1)

        logger.info(f"Using polynomial degree: {degree}")

        coeffs = np.polyfit(np.arange(n), y, degree)
        poly_func = np.poly1d(coeffs)
        future_x = np.arange(n, n + periods)
        forecast = poly_func(future_x)
        forecast = np.maximum(forecast, 0)
        predicted = poly_func(np.arange(n))
        metrics = ForecastExecutionService.calculate_metrics(y, predicted)

        return forecast, metrics

    @staticmethod
    def exponential_smoothing_forecast(data: pd.DataFrame, periods: int, alphas: list = [0.1, 0.3, 0.5]) -> tuple:
        """Exponential smoothing forecasting"""
        try:
            # Validate parameters
            validated_alphas = []
            for alpha in alphas:
                alpha = max(0.0, min(1.0, float(alpha)))
                validated_alphas.append(alpha)
            alphas = validated_alphas if validated_alphas else [0.3]

            logger.info(f"Using exponential smoothing alphas: {alphas}")

            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            if n < 3:
                return np.full(periods, y[-1] if len(y) > 0 else 0), {'mape': 100.0, 'mae': np.std(y), 'rmse': np.std(y)}

            best_metrics = None
            best_forecast = None

            for alpha in alphas:
                # Traditional exponential smoothing
                smoothed = pd.Series(y).ewm(alpha=alpha).mean().values
                forecast = np.full(periods, smoothed[-1])
                metrics = ForecastExecutionService.calculate_metrics(y[1:], smoothed[1:])

                if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                    best_metrics = metrics
                    best_forecast = forecast

            return np.array(best_forecast), best_metrics

        except Exception as e:
            logger.warning(f"Exponential smoothing failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def holt_winters_forecast(data: pd.DataFrame, periods: int, season_length: int = 12, alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.1) -> tuple:
        """Holt-Winters exponential smoothing"""
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

        n = len(y)
        if n < 2 * season_length:
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)

        # Validate parameters
        alpha = max(0.0, min(1.0, alpha))
        beta = max(0.0, min(1.0, beta))
        gamma = max(0.0, min(1.0, gamma))

        logger.info(f"Using Holt-Winters parameters: alpha={alpha}, beta={beta}, gamma={gamma}, season_length={season_length}")

        # Traditional Holt-Winters
        alpha, beta, gamma = alpha, beta, gamma
        
        level = np.mean(y[:season_length])
        trend = (np.mean(y[season_length:2*season_length]) - np.mean(y[:season_length])) / season_length
        seasonal = y[:season_length] - level
        
        levels = [level]
        trends = [trend]
        seasonals = list(seasonal)
        fitted = []
        
        for i in range(len(y)):
            if i == 0:
                fitted.append(level + trend + seasonal[i % season_length])
            else:
                level = alpha * (y[i] - seasonals[i % season_length]) + (1 - alpha) * (levels[-1] + trends[-1])
                trend = beta * (level - levels[-1]) + (1 - beta) * trends[-1]
                if len(seasonals) > i:
                    seasonals[i % season_length] = gamma * (y[i] - level) + (1 - gamma) * seasonals[i % season_length]
                
                levels.append(level)
                trends.append(trend)
                fitted.append(level + trend + seasonals[i % season_length])
        
        forecast = []
        for i in range(periods):
            forecast_value = level + (i + 1) * trend + seasonals[(len(y) + i) % season_length]
            forecast.append(max(0, forecast_value))
        
        metrics = ForecastExecutionService.calculate_metrics(y, fitted)
        
        return np.array(forecast), metrics

    @staticmethod
    def simple_moving_average(data: pd.DataFrame, periods: int, window: int = 3) -> Tuple[np.ndarray, Dict[str, float]]:
        """Simple Moving Average forecast."""
        y = data['total_quantity'].values if 'total_quantity' in data.columns else data['quantity'].values

        # Calculate moving average
        if len(y) < window:
            avg = np.mean(y)
        else:
            avg = np.mean(y[-window:])

        forecast = np.array([avg] * periods)

        # Simple metrics
        predicted = np.array([np.mean(y[max(0, i-window):i+1]) for i in range(len(y))])
        metrics = ForecastExecutionService.calculate_metrics(y, predicted)

        return forecast, metrics

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
        
        if np.sum(valid_fitted) > 0:
            metrics = ForecastExecutionService.calculate_metrics(y[valid_fitted], fitted[valid_fitted])
        else:
            metrics = {'accuracy': 50.0, 'mae': np.std(y), 'rmse': np.std(y)}
        
        return forecast, metrics

    @staticmethod
    def moving_average_forecast(data: pd.DataFrame, periods: int, window: int = 3) -> tuple:
        """Moving average forecasting"""
        if 'total_quantity' in data.columns:
            y = data['total_quantity'].values
        elif 'quantity' in data.columns:
            y = data['quantity'].values
        else:
            raise ValueError("Data must contain 'quantity' or 'total_quantity' column")
        
        window = min(window, len(y))
        
        # Calculate moving averages
        moving_avg = []
        for i in range(len(y)):
            start_idx = max(0, i - window + 1)
            moving_avg.append(np.mean(y[start_idx:i+1]))
        
        # Forecast using last moving average
        last_avg = np.mean(y[-window:])
        forecast = np.full(periods, last_avg)
        
        # Calculate metrics
        metrics = ForecastExecutionService.calculate_metrics(y[window-1:], moving_avg[window-1:])
        
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
                    start_p=0, start_q=0, max_p=2, max_q=2, max_d=1,
                    start_P=0, start_Q=0, max_P=2, max_Q=2, max_D=1,
                    seasonal=True, m=seasonal_period,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )

                forecast = auto_model.predict(n_periods=periods)
                forecast = np.maximum(forecast, 0)

                # Calculate metrics
                fitted = auto_model.fittedvalues()
                if len(fitted) == len(y):
                    metrics = ForecastExecutionService.calculate_metrics(y, fitted)
                else:
                    start_idx = len(y) - len(fitted)
                    metrics = ForecastExecutionService.calculate_metrics(y[start_idx:], fitted)

                return forecast, metrics

        except Exception as e:
            print(f"SARIMA failed: {e}")

        # Fallback to ARIMA
        return ForecastExecutionService.arima_forecast(data, periods)

    @staticmethod
    def xgboost_forecast(data: pd.DataFrame, periods: int, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        XGBoost forecasting with feature engineering.

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
            if n < 5:  # Need minimum data for XGBoost
                raise ValueError("Need at least 5 historical data points for XGBoost")

            # Feature engineering: create lag features and time index
            window = min(5, n - 1)

            # Build lagged features
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

            if len(X) < 2:
                raise ValueError("Insufficient data for XGBoost training")

            # Train XGBoost model
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y_target)

            # Generate forecast
            forecast = []
            recent_lags = list(y[-window:])

            for i in range(periods):
                features = recent_lags + [n + i]
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

    @staticmethod
    def svr_forecast(data: pd.DataFrame, periods: int, C: float = 1.0, epsilon: float = 0.1, kernel: str = 'rbf') -> Tuple[np.ndarray, Dict[str, float]]:
        """Support Vector Regression forecasting"""
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
            if n < 5:  # Need minimum data for SVR
                raise ValueError("Need at least 5 historical data points for SVR")

            # Feature engineering: create lag features and time index
            window = min(5, n - 1)

            # Build lagged features
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

            if len(X) < 2:
                raise ValueError("Insufficient data for SVR training")

            # Train SVR model
            model = SVR(
                kernel=kernel,
                C=C,
                epsilon=epsilon
            )
            model.fit(X, y_target)

            # Generate forecast
            forecast = []
            recent_lags = list(y[-window:])

            for i in range(periods):
                features = recent_lags + [n + i]
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

        except Exception as e:
            logger.warning(f"SVR failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def knn_forecast(data: pd.DataFrame, periods: int, n_neighbors: int = 5) -> Tuple[np.ndarray, Dict[str, float]]:
        """K-Nearest Neighbors forecasting"""
        try:
            # Prepare quantity data
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            if n < n_neighbors + 2:  # Need minimum data for KNN
                raise ValueError(f"Need at least {n_neighbors + 2} historical data points for KNN")

            # Feature engineering: create lag features and time index
            window = min(5, n - 1)

            # Build lagged features
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

            # Train KNN model
            model = KNeighborsRegressor(
                n_neighbors=min(n_neighbors, len(X)),
                weights='uniform'
            )
            model.fit(X, y_target)

            # Generate forecast
            forecast = []
            recent_lags = list(y[-window:])

            for i in range(periods):
                features = recent_lags + [n + i]
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

        except Exception as e:
            logger.warning(f"KNN failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        # Ensure arrays are numpy arrays
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        
        # Create mask for valid values
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {'mape': None, 'mae': None, 'rmse': None, 'r_squared': None}
        
        # MAPE
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
        
        # MAE
        mae = np.mean(np.abs(actual - predicted))
        
        # RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mape': round(float(mape), 2),
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'r_squared': round(float(r_squared), 4)
        }

    @staticmethod
    def _prepare_external_factors(
        tenant_id: str,
        database_name: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Prepare external factors data."""
        try:
            start = datetime.fromisoformat(start_date).date()
            end = datetime.fromisoformat(end_date).date()

            factors = ExternalFactorsService.get_factors_for_period(
                tenant_id, database_name, start, end
            )

            if not factors:
                return pd.DataFrame()

            df = pd.DataFrame(factors)
            df['date'] = pd.to_datetime(df['date'])

            # Pivot to wide format
            df_pivot = df.pivot_table(
                index='date',
                columns='factor_name',
                values='factor_value',
                aggfunc='mean'
            ).reset_index()

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
    def _store_forecast_results(
        tenant_id: str,
        database_name: str,
        forecast_run_id: str,
        version_id: str,
        mapping_id: str,
        algorithm_id: int,
        forecast_dates: List[date],
        forecast_values: np.ndarray,
        metrics: Dict[str, float],
        user_email: str
    ) -> List[Dict[str, Any]]:
        """Store forecast results in database."""
        db_manager = get_db_manager()
        results = []
        
        # Ensure forecast_values is numpy array
        forecast_values = np.asarray(forecast_values, dtype=np.float64)
        
        # Validate and clip values to fit DECIMAL(18,4) bounds
        # Max value for DECIMAL(18,4) is 9999999999999.9999
        # But for forecast_quantity, we clip to reasonable business values
        max_forecast_value = 999999.9999
        forecast_values = np.clip(forecast_values, 0, max_forecast_value)
        
        with db_manager.get_tenant_connection(database_name) as conn:
            cursor = conn.cursor()
            try:
                for forecast_date, forecast_value in zip(forecast_dates, forecast_values):
                    result_id = str(uuid.uuid4())
                    
                    # Convert to Python native types and validate
                    forecast_value_float = float(forecast_value)
                    
                    # Double-check bounds
                    if not (0 <= forecast_value_float <= max_forecast_value):
                        logger.warning(
                            f"Forecast value {forecast_value_float} out of bounds, "
                            f"clipping to [{0}, {max_forecast_value}]"
                        )
                        forecast_value_float = np.clip(forecast_value_float, 0, max_forecast_value)
                    
                    # Round to 4 decimal places to match DECIMAL(18,4)
                    forecast_value_float = round(forecast_value_float, 4)
                    
                    # Extract metric value safely
                    accuracy_metric = None
                    if metrics and 'mape' in metrics and metrics['mape'] is not None:
                        accuracy_metric = float(metrics['mape'])
                        # MAPE is DECIMAL(5,2) - max value 999.99
                        if accuracy_metric > 999.99:
                            accuracy_metric = 999.99
                        accuracy_metric = round(accuracy_metric, 2)
                    
                    cursor.execute("""
                        INSERT INTO forecast_results
                        (result_id, tenant_id, forecast_run_id, version_id, mapping_id,
                         algorithm_id, forecast_date, forecast_quantity, accuracy_metric,
                         metric_type, metadata, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        result_id,
                        tenant_id,
                        forecast_run_id,
                        version_id,
                        mapping_id,
                        algorithm_id,
                        forecast_date,
                        forecast_value_float,
                        accuracy_metric,
                        'MAPE',
                        Json({'metrics': metrics}),  # Use Json() wrapper
                        user_email
                    ))
                    
                    results.append({
                        'result_id': result_id,
                        'forecast_date': forecast_date.isoformat(),
                        'forecast_quantity': forecast_value_float
                    })
                
                conn.commit()
                
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
        process_log: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate forecast using best_fit algorithm selection.
        Runs multiple algorithms in parallel and selects the best performing one.
        
        Args:
            historical_data: Historical data with quantity column
            config: Configuration dict with interval and other settings
            process_log: Optional list to append process logs
            
        Returns:
            Dictionary with forecast results including selected algorithm, metrics, and predictions
        """
        if process_log is None:
            process_log = []
        
        process_log.append("Loading data for forecasting...")
        
        if len(historical_data) < 2:
            raise ValueError("Insufficient data for forecasting")
        
        process_log.append(f"Data loaded: {len(historical_data)} records")
        process_log.append("Running best fit algorithm selection...")
        
        # Define all available algorithms
        available_algorithms = [
            "linear_regression",
            "polynomial_regression",
            "exponential_smoothing",
            "holt_winters",
            "arima",
            "xgboost",
            "svr",
            "knn",
            "simple_moving_average",
            "seasonal_decomposition",
            "moving_average",
            "sarima"
        ]
        
        algorithm_results = []
        best_model = None
        best_algorithm = None
        best_metrics = None
        
        # Use ThreadPoolExecutor for parallel execution
        max_workers = min(len(available_algorithms), os.cpu_count() or 4)
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
                    periods
                )
                future_to_algorithm[future] = algorithm
            
            # Collect results as they complete
            for future in as_completed(future_to_algorithm):
                algorithm_name = future_to_algorithm[future]
                try:
                    result = future.result()
                    algorithm_results.append(result)
                    
                    process_log.append(
                        f"✅ Algorithm {algorithm_name} completed with accuracy: {result['accuracy']:.2f}%"
                    )
                    
                    # Track best performing algorithm
                    if best_metrics is None or result['accuracy'] > best_metrics['accuracy']:
                        best_metrics = {
                            'accuracy': result['accuracy'],
                            'mae': result['mae'],
                            'rmse': result['rmse'],
                            'mape': result.get('mape')
                        }
                        best_model = result
                        best_algorithm = algorithm_name
                        
                except Exception as exc:
                    process_log.append(f"❌ Algorithm {algorithm_name} failed: {str(exc)}")
                    logger.warning(f"Algorithm {algorithm_name} failed: {str(exc)}")
        
        # Filter out failed results
        successful_results = [res for res in algorithm_results if res.get('accuracy', 0) > 0]
        
        if not successful_results:
            raise ValueError("All algorithms failed to produce valid results")
        
        process_log.append(
            f"Parallel execution completed. {len(successful_results)} algorithms succeeded, "
            f"{len(algorithm_results) - len(successful_results)} failed."
        )
        
        # Ensemble: average forecast of top 3 algorithms by accuracy
        top3 = sorted(successful_results, key=lambda x: -x['accuracy'])[:3]
        
        if len(top3) >= 2:
            process_log.append(f"Creating ensemble from top {len(top3)} algorithms...")
            
            # Average the forecast values from top 3
            ensemble_forecast = []
            for i in range(len(top3[0]['forecast'])):
                values = [algo['forecast'][i] for algo in top3 if i < len(algo['forecast'])]
                if values:
                    ensemble_forecast.append(np.mean(values))
            
            # Average the metrics
            ensemble_result = {
                'algorithm': 'Ensemble (Top 3 Avg)',
                'forecast': ensemble_forecast,
                'accuracy': np.mean([algo['accuracy'] for algo in top3]),
                'mae': np.mean([algo['mae'] for algo in top3]),
                'rmse': np.mean([algo['rmse'] for algo in top3]),
                'mape': np.mean([algo.get('mape', 0) for algo in top3])
            }
            algorithm_results.append(ensemble_result)
            
            # Update best model if ensemble is better
            if ensemble_result['accuracy'] > best_metrics['accuracy']:
                best_model = ensemble_result
                best_algorithm = 'ensemble'
                best_metrics = {
                    'accuracy': ensemble_result['accuracy'],
                    'mae': ensemble_result['mae'],
                    'rmse': ensemble_result['rmse'],
                    'mape': ensemble_result['mape']
                }
        
        process_log.append(f"Best algorithm selected: {best_algorithm} (Accuracy: {best_metrics['accuracy']:.2f}%)")
        
        return {
            'selected_algorithm': f"{best_algorithm} (Best Fit)",
            'accuracy': best_metrics['accuracy'],
            'mae': best_metrics['mae'],
            'rmse': best_metrics['rmse'],
            'mape': best_metrics.get('mape'),
            'forecast': best_model['forecast'],
            'all_algorithms': algorithm_results,
            'process_log': process_log
        }
    
    @staticmethod
    def _run_algorithm_safe(
        algorithm_name: str,
        data: pd.DataFrame,
        periods: int
    ) -> Dict[str, Any]:
        """
        Safely run an algorithm and return results.
        
        Args:
            algorithm_name: Name of the algorithm to run
            data: Historical data (must have 'total_quantity' column)
            periods: Number of periods to forecast
            
        Returns:
            Dictionary with algorithm results
        """
        try:
            # Ensure we have total_quantity column (rename if needed)
            if 'quantity' in data.columns and 'total_quantity' not in data.columns:
                data = data.rename(columns={'quantity': 'total_quantity'})
            
            if algorithm_name == "linear_regression":
                forecast, metrics = ForecastExecutionService.linear_regression_forecast(data, periods)
            elif algorithm_name == "polynomial_regression":
                forecast, metrics = ForecastExecutionService.polynomial_regression_forecast(data, periods)
            elif algorithm_name == "exponential_smoothing":
                forecast, metrics = ForecastExecutionService.exponential_smoothing_forecast(data, periods)
            elif algorithm_name == "holt_winters":
                forecast, metrics = ForecastExecutionService.holt_winters_forecast(data, periods)
            elif algorithm_name == "arima":
                forecast, metrics = ForecastExecutionService.arima_forecast(data, periods)
            elif algorithm_name == "xgboost":
                forecast, metrics = ForecastExecutionService.xgboost_forecast(data, periods)
            elif algorithm_name == "svr":
                forecast, metrics = ForecastExecutionService.svr_forecast(data, periods)
            elif algorithm_name == "knn":
                forecast, metrics = ForecastExecutionService.knn_forecast(data, periods)
            elif algorithm_name == "simple_moving_average":
                forecast, metrics = ForecastExecutionService.simple_moving_average(data, periods)
            elif algorithm_name == "seasonal_decomposition":
                forecast, metrics = ForecastExecutionService.seasonal_decomposition_forecast(data, periods)
            elif algorithm_name == "moving_average":
                forecast, metrics = ForecastExecutionService.moving_average_forecast(data, periods)
            elif algorithm_name == "sarima":
                forecast, metrics = ForecastExecutionService.sarima_forecast(data, periods)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            # Convert forecast to list
            forecast_list = forecast.tolist() if isinstance(forecast, np.ndarray) else forecast
            
            # Extract accuracy metric (use MAPE if available, otherwise calculate from R-squared)
            accuracy = metrics.get('mape')
            if accuracy is None:
                # Convert R-squared to accuracy percentage
                r_squared = metrics.get('r_squared', 0)
                accuracy = max(0, min(100, (r_squared * 100)))
            else:
                # MAPE is error, so accuracy = 100 - MAPE
                accuracy = max(0, min(100, (100 - accuracy)))
            
            return {
                'algorithm': algorithm_name,
                'forecast': forecast_list,
                'accuracy': accuracy,
                'mae': metrics.get('mae', 0),
                'rmse': metrics.get('rmse', 0),
                'mape': metrics.get('mape', 0)
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
                'mape': 100.0
            }