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
from sklearn.linear_model import LinearRegression
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

            elif algorithm_id == 14:  # Random Forest
                # n_estimators: 10-500
                # max_depth: 1-50 or None
                # min_samples_split: 2-20
                # min_samples_leaf: 1-10
                if 'n_estimators' in validated_params:
                    validated_params['n_estimators'] = max(10, min(500, int(validated_params['n_estimators'])))
                if 'max_depth' in validated_params:
                    if validated_params['max_depth'] is None:
                        validated_params['max_depth'] = None
                    else:
                        validated_params['max_depth'] = max(1, min(50, int(validated_params['max_depth'])))
                if 'min_samples_split' in validated_params:
                    validated_params['min_samples_split'] = max(2, min(20, int(validated_params['min_samples_split'])))
                if 'min_samples_leaf' in validated_params:
                    validated_params['min_samples_leaf'] = max(1, min(10, int(validated_params['min_samples_leaf'])))

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
        user_email: str
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
            user_email=user_email
        )

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
        custom_params = algorithm_mapping.get('custom_parameters') or {}

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
            
            # ✅ IMPROVED: Merge external factors with better logging
            if not external_factors.empty:
                logger.info(
                    f"Merging external factors with historical data for {algorithm_name}"
                )
                logger.debug(f"Historical data date range: {historical_data['period'].min()} to {historical_data['period'].max()}")
                logger.debug(f"External factors date range: {external_factors['date'].min()} to {external_factors['date'].max()}")
                logger.debug(f"External factor columns: {list(external_factors.columns)}")
                
                # Merge on date
                historical_data = historical_data.merge(
                    external_factors,
                    left_on='period',
                    right_on='date',
                    how='left'
                )
                
                # Check merge success
                factor_columns = [col for col in external_factors.columns if col != 'date']
                non_null_count = historical_data[factor_columns].notna().sum().sum()
                total_cells = len(historical_data) * len(factor_columns)
                merge_success_rate = (non_null_count / total_cells * 100) if total_cells > 0 else 0
                
                logger.info(
                    f"External factors merge: {non_null_count}/{total_cells} cells matched "
                    f"({merge_success_rate:.1f}% success rate)"
                )
                
                if merge_success_rate < 50:
                    logger.warning(
                        f"Low merge success rate for external factors! "
                        f"Check that factor dates overlap with historical data dates."
                    )
            else:
                logger.info(f"No external factors to merge for {algorithm_name}")
            
            # Route to appropriate algorithm
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
                forecast, metrics = ForecastExecutionService.prophet_forecast(
                    data=historical_data,
                    periods=periods,
                    seasonality_mode=custom_params.get('seasonality_mode', 'additive'),
                    changepoint_prior_scale=custom_params.get('changepoint_prior_scale', 0.05)
                )
            elif algorithm_id == 8:  # LSTM Neural Network
                forecast, metrics = ForecastExecutionService.lstm_forecast(
                    data=historical_data,
                    periods=periods,
                    sequence_length=custom_params.get('sequence_length', 12),
                    epochs=custom_params.get('epochs', 50),
                    batch_size=custom_params.get('batch_size', 32)
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
            elif algorithm_id == 12:  # Gaussian Process
                forecast, metrics = ForecastExecutionService.gaussian_process_forecast(
                    data=historical_data,
                    periods=periods,
                    kernel=custom_params.get('kernel', 'RBF'),
                    alpha=custom_params.get('alpha', 1e-6)
                )
            elif algorithm_id == 13:  # Neural Network (MLP)
                forecast, metrics = ForecastExecutionService.mlp_neural_network_forecast(
                    data=historical_data,
                    periods=periods,
                    hidden_layers=custom_params.get('hidden_layers', [64, 32]),
                    epochs=custom_params.get('epochs', 100),
                    batch_size=custom_params.get('batch_size', 32)
                )
            elif algorithm_id == 14:  # Random Forest
                forecast, metrics = ForecastExecutionService.random_forest_forecast(
                    data=historical_data,
                    periods=periods,
                    n_estimators=custom_params.get('n_estimators', 100),
                    max_depth=custom_params.get('max_depth', None),
                    min_samples_split=custom_params.get('min_samples_split', 2),
                    min_samples_leaf=custom_params.get('min_samples_leaf', 1)
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
            
            # # Validate forecast quality
            # Forecast quality validation removed

            # if is_constant:
            #     logger.warning(f"Algorithm {algorithm_name} produced constant forecast - all values are {forecast[0]:.2f}")

            # # Log without Unicode characters to avoid Windows console encoding issues
            # logger.info(f"Forecast validation: {variance_msg.replace(chr(10003), '[PASS]').replace(chr(9888), '[WARN]')}")

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
                }

            best_metrics = None
            best_forecast = None

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

                else:
                    # Fallback to simple exponential smoothing for very small datasets
                    smoothed = np.zeros(n)
                    smoothed[0] = y[0]
                    
                    for i in range(1, n):
                        smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i-1]
                    
                    forecast = np.full(periods, smoothed[-1])
                    metrics = ForecastExecutionService.calculate_metrics(y[1:], smoothed[1:])

                logger.info(f"Alpha={alpha}, RMSE={metrics.get('rmse', 0):.2f}, MAE={metrics.get('mae', 0):.2f}, MAPE={metrics.get('mape', 0):.2f}")

                # Select best model based on RMSE
                if best_metrics is None or metrics.get('rmse', float('inf')) < best_metrics.get('rmse', float('inf')):
                    best_metrics = metrics
                    best_forecast = forecast

            return best_forecast, best_metrics

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
    def prophet_forecast(data: pd.DataFrame, periods: int, seasonality_mode: str = 'additive', changepoint_prior_scale: float = 0.05) -> Tuple[np.ndarray, Dict[str, float]]:
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
            if 'period' in data.columns:
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

            logger.info(f"Prophet forecast completed with {periods} periods")
            return forecast, metrics

        except ImportError:
            logger.warning("Prophet library not available, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
        except Exception as e:
            logger.warning(f"Prophet forecasting failed: {str(e)}, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def lstm_forecast(data: pd.DataFrame, periods: int, sequence_length: int = 12, epochs: int = 50, batch_size: int = 32) -> Tuple[np.ndarray, Dict[str, float]]:
        """LSTM Neural Network forecasting."""
        try:
            from tensorflow import keras
            from tensorflow.keras.models import Sequential # type: ignore
            from tensorflow.keras.layers import LSTM, Dense # type: ignore
            from tensorflow.keras.optimizers import Adam # type: ignore
            import warnings
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
            y_train_pred = model.predict(X_train, verbose=0).flatten()
            y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            metrics = ForecastExecutionService.calculate_metrics(y_train_actual, y_train_pred)

            logger.info(f"LSTM forecast completed with {periods} periods")
            return forecast, metrics

        except ImportError:
            logger.warning("TensorFlow/Keras not available, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)
        except Exception as e:
            logger.warning(f"LSTM forecasting failed: {str(e)}, falling back to Exponential Smoothing")
            return ForecastExecutionService.exponential_smoothing_forecast(data, periods)

    @staticmethod
    def gaussian_process_forecast(data: pd.DataFrame, periods: int, kernel: str = 'RBF', alpha: float = 1e-6) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Gaussian Process Regression forecasting with uncertainty quantification.
        
        Args:
            data: Historical data with 'quantity' or 'total_quantity' column
            periods: Number of forecast periods
            kernel: Kernel type ('RBF', 'Matern', 'RationalQuadratic')
            alpha: Regularization strength
            
        Returns:
            Tuple of (forecast_array, metrics_dict)
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

            # Calculate metrics on test data
            if len(X_train) > 0:
                y_pred_scaled = gp.predict(X_train_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                # Handle NaN in predictions
                y_pred = np.nan_to_num(y_pred, nan=0.0)
                metrics = ForecastExecutionService.calculate_metrics(y_train, y_pred)
                # Handle NaN in metrics
                metrics = {k: (0.0 if np.isnan(v) else v) for k, v in metrics.items()}
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
    def mlp_neural_network_forecast(data: pd.DataFrame, periods: int, hidden_layers: List[int] = None, epochs: int = 100, batch_size: int = 32) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Multi-layer Perceptron (MLP) Neural Network forecasting.
        
        Args:
            data: Historical data with 'quantity' or 'total_quantity' column
            periods: Number of forecast periods
            hidden_layers: List of hidden layer sizes, e.g., [64, 32]
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Tuple of (forecast_array, metrics_dict)
        """
        if hidden_layers is None:
            hidden_layers = [64, 32]
            
        try:
            from tensorflow import keras
            from tensorflow.keras import layers # type: ignore
            from sklearn.preprocessing import MinMaxScaler
            
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
    def _get_external_factor_columns(data: pd.DataFrame) -> List[str]:
        """
        Identify external factor columns in the data.
        External factors are columns not in the standard set (period, total_quantity, quantity, date).
        Also excludes aggregation columns and calculated fields.
        
        Args:
            data: DataFrame with historical data and potentially merged external factors
            
        Returns:
            List of external factor column names
        """
        # Standard columns from sales data and aggregation
        standard_columns = {
            'period', 'total_quantity', 'quantity', 'date',
            'transaction_count', 'avg_price', 'uom'
        }
        
        # Exclude any column that could be a master data aggregation field
        # These would typically be string/categorical columns used for grouping
        external_factors = []
        for col in data.columns:
            if col not in standard_columns:
                # Check if it's likely a categorical aggregation column
                # (typically has few unique values relative to data size)
                if col in data.columns:
                    dtype = data[col].dtype
                    # If it's a string/object column with few unique values, it's likely an aggregation field
                    if dtype == 'object' or dtype.name == 'category':
                        unique_ratio = data[col].nunique() / len(data) if len(data) > 0 else 0
                        if unique_ratio < 0.1:  # Less than 10% unique values = aggregation field
                            continue
                # Otherwise, treat as external factor
                external_factors.append(col)
        
        return external_factors

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

        # Preprocess external factors: factorize categorical columns to numeric codes
        encoded_columns = {}
        for factor_name in external_factors:
            if factor_name in data.columns:
                col = data[factor_name].fillna('__NA__')
                if not np.issubdtype(col.dtype, np.number):
                    codes, uniques = pd.factorize(col)
                    # store encoded series as float
                    encoded_columns[factor_name] = codes.astype(float)
                else:
                    # numeric column: convert NaN to 0.0 and keep
                    encoded_columns[factor_name] = col.astype(float).fillna(0.0).values

        for i in range(window, len(data)):
            lags = y[i-window:i]
            time_idx = i
            features = list(lags) + [time_idx]

            for factor_name in external_factors:
                if factor_name in data.columns:
                    # use precomputed encoded value when available
                    if factor_name in encoded_columns:
                        factor_value = encoded_columns[factor_name][i]
                    else:
                        factor_value = data[factor_name].iloc[i]
                        if pd.isna(factor_value):
                            factor_value = 0.0
                        else:
                            try:
                                factor_value = float(factor_value)
                            except Exception:
                                # last resort: use hash-based deterministic encoding
                                factor_value = float(abs(hash(str(factor_value))) % 1000000) / 1000.0

                    features.append(float(factor_value))

            X.append(features)
            y_target.append(y[i])

        return np.array(X), np.array(y_target)

    @staticmethod
    def _prepare_future_features_with_factors(recent_lags: List[float], n: int, idx: int, window: int, external_factors: List[str], last_factor_values: Dict[str, float]) -> np.ndarray:
        """
        Prepare features for future prediction including external factors.
        For external factors, use the last known values.
        
        Args:
            recent_lags: Recent lagged values
            n: Current data length
            idx: Forecast index
            window: Window size for lags
            external_factors: List of external factor column names
            last_factor_values: Dictionary of last known external factor values
            
        Returns:
            Feature array for prediction
        """
        features = recent_lags + [n + idx]
        
        for factor_name in external_factors:
            if factor_name in last_factor_values:
                features.append(last_factor_values[factor_name])
            else:
                features.append(0.0)
        
        return np.array(features)

    @staticmethod
    def xgboost_forecast(data: pd.DataFrame, periods: int, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        XGBoost forecasting with time series cross-validation to prevent data leakage.

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
            if n < 10:  # Need more data for proper cross-validation
                raise ValueError("Need at least 10 historical data points for XGBoost with cross-validation")

            # Feature engineering: create lag features, time index, and external factors
            window = min(5, n - 1)
            external_factors = ForecastExecutionService._get_external_factor_columns(data)
            
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

            # Time series cross-validation: use expanding window
            train_size = max(5, len(X) - periods)
            X_train = X[:train_size]
            y_train = y_target[:train_size]
            X_test = X[train_size:]
            y_test = y_target[train_size:]

            logger.info(f"XGBoost time series split: train_size={len(X_train)}, test_size={len(X_test)}")

            # Train XGBoost model on training data only
            model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X_train, y_train, verbose=False)

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
                                # Try to factorize entire column and pick numeric code for last value
                                try:
                                    codes, uniques = pd.factorize(data[factor_name].fillna('__NA__'))
                                    val = float(codes[-1])
                                except Exception:
                                    # fallback deterministic encoding
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

            # Calculate metrics on test data (to reflect real performance)
            if len(X_test) > 0:
                predicted_test = model.predict(X_test)
                metrics = ForecastExecutionService.calculate_metrics(y_test, predicted_test)
            else:
                # Fallback to training metrics if no test data
                predicted = model.predict(X_train)
                metrics = ForecastExecutionService.calculate_metrics(y_train, predicted)

            # Forecast validation removed

            return forecast, metrics

        except ImportError:
            logger.warning("XGBoost not available, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def svr_forecast(data: pd.DataFrame, periods: int, C: float = 1.0, epsilon: float = 0.1, kernel: str = 'rbf') -> Tuple[np.ndarray, Dict[str, float]]:
        """Support Vector Regression forecasting with time series cross-validation."""
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
            if n < 10:  # Need more data for proper cross-validation
                raise ValueError("Need at least 10 historical data points for SVR with cross-validation")

            # Feature engineering: create lag features, time index, and external factors
            window = min(5, n - 1)
            external_factors = ForecastExecutionService._get_external_factor_columns(data)
            
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

            # Time series cross-validation: use expanding window
            train_size = max(5, len(X) - periods)
            X_train = X[:train_size]
            y_train = y_target[:train_size]
            X_test = X[train_size:]
            y_test = y_target[train_size:]

            logger.info(f"SVR time series split: train_size={len(X_train)}, test_size={len(X_test)}")

            # Scale features for SVR (required for better performance)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Train SVR model on training data only
            model = SVR(
                kernel=kernel,
                C=C,
                epsilon=epsilon
            )
            model.fit(X_train_scaled, y_train)

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

            # Calculate metrics on test data
            if len(X_test) > 0:
                X_test_scaled = scaler.transform(X_test)
                predicted_test = model.predict(X_test_scaled)
                metrics = ForecastExecutionService.calculate_metrics(y_test, predicted_test)
            else:
                # Fallback to training metrics if no test data
                predicted = model.predict(X_train_scaled)
                metrics = ForecastExecutionService.calculate_metrics(y_train, predicted)

            return forecast, metrics

        except Exception as e:
            logger.warning(f"SVR failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def knn_forecast(data: pd.DataFrame, periods: int, n_neighbors: int = 5) -> Tuple[np.ndarray, Dict[str, float]]:
        """K-Nearest Neighbors forecasting with time series cross-validation."""
        try:
            # Prepare quantity data
            if 'total_quantity' in data.columns:
                y = data['total_quantity'].values
            elif 'quantity' in data.columns:
                y = data['quantity'].values
            else:
                raise ValueError("Data must contain 'quantity' or 'total_quantity' column")

            n = len(y)
            if n < n_neighbors + 5:  # Need more data for proper cross-validation
                raise ValueError(f"Need at least {n_neighbors + 5} historical data points for KNN with cross-validation")

            # Feature engineering: create lag features, time index, and external factors
            window = min(5, n - 1)
            external_factors = ForecastExecutionService._get_external_factor_columns(data)
            
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

            # Time series cross-validation: use expanding window
            train_size = max(n_neighbors, len(X) - periods)
            X_train = X[:train_size]
            y_train = y_target[:train_size]
            X_test = X[train_size:]
            y_test = y_target[train_size:]

            logger.info(f"KNN time series split: train_size={len(X_train)}, test_size={len(X_test)}")

            # Scale features for KNN (important for distance metrics)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Train KNN model on training data only
            model = KNeighborsRegressor(
                n_neighbors=min(n_neighbors, len(X_train)),
                weights='uniform'
            )
            model.fit(X_train_scaled, y_train)

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

            # Calculate metrics on test data
            if len(X_test) > 0:
                X_test_scaled = scaler.transform(X_test)
                predicted_test = model.predict(X_test_scaled)
                metrics = ForecastExecutionService.calculate_metrics(y_test, predicted_test)
            else:
                # Fallback to training metrics if no test data
                predicted = model.predict(X_train_scaled)
                metrics = ForecastExecutionService.calculate_metrics(y_train, predicted)

            return forecast, metrics

        except Exception as e:
            logger.warning(f"KNN failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def random_forest_forecast(data: pd.DataFrame, periods: int, n_estimators: int = 100, max_depth: Optional[int] = None, min_samples_split: int = 2, min_samples_leaf: int = 1) -> Tuple[np.ndarray, Dict[str, float]]:
        """Random Forest regression forecasting with hyperparameter tuning."""
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
                raise ValueError("Need at least 10 historical data points for Random Forest with cross-validation")

            window = min(5, n - 1)
            external_factors = ForecastExecutionService._get_external_factor_columns(data)
            
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

            train_size = max(5, len(X) - periods)
            X_train = X[:train_size]
            y_train = y_target[:train_size]
            X_test = X[train_size:]
            y_test = y_target[train_size:]

            logger.info(f"Random Forest time series split: train_size={len(X_train)}, test_size={len(X_test)}")

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            forecast = []
            recent_lags = list(y[-window:])
            
            last_factor_values = {}
            if external_factors and len(data) > 0:
                for factor_name in external_factors:
                    if factor_name in data.columns:
                        factor_value = data[factor_name].iloc[-1]
                        last_factor_values[factor_name] = 0.0 if pd.isna(factor_value) else float(factor_value)

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

            if len(X_test) > 0:
                predicted_test = model.predict(X_test)
                metrics = ForecastExecutionService.calculate_metrics(y_test, predicted_test)
            else:
                predicted = model.predict(X_train)
                metrics = ForecastExecutionService.calculate_metrics(y_train, predicted)

            # Forecast validation removed

            return forecast, metrics

        except Exception as e:
            logger.warning(f"Random Forest failed: {str(e)}, falling back to Linear Regression")
            return ForecastExecutionService.linear_regression_forecast(data, periods)

    @staticmethod
    def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))

        # Calculate accuracy as percentage
        mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, 1))) * 100
        accuracy = max(0, 100 - mape)

        return {
            'accuracy': min(accuracy, 99.9),
            'mae': mae,
            'rmse': rmse
        }

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

        # Handle NaN and infinite values
        forecast_values = np.nan_to_num(forecast_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Validate and clip values to fit DECIMAL(18,4) bounds
        # Max value for DECIMAL(18,4) is 9999999999999.9999
        # But for forecast_quantity, we clip to reasonable business values
        max_forecast_value = 99999999.999999
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
                    accuracy_metric = 0.0
                    if metrics and 'accuracy' in metrics and metrics['accuracy'] is not None:
                        accuracy_metric = float(metrics['accuracy'])
                        # Accuracy is DECIMAL(5,2) - max value 999.99
                        if accuracy_metric > 999.99:
                            accuracy_metric = 999.99
                        accuracy_metric = round(accuracy_metric, 2)
                    
                    cursor.execute("""
                        INSERT INTO forecast_results
                        (result_id, forecast_run_id, version_id, mapping_id,
                         algorithm_id, forecast_date, forecast_quantity, accuracy_metric,
                         metric_type, metadata, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        result_id,
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
        
        # Filter out failed results: prefer positive accuracy and non-empty forecasts
        successful_results = [res for res in algorithm_results if res.get('accuracy', 0) > 0 and res.get('forecast')]

        # If none have positive accuracy, fall back to any algorithm that produced a non-empty forecast
        if not successful_results:
            fallback_results = [res for res in algorithm_results if res.get('forecast')]
            if fallback_results:
                successful_results = fallback_results
                process_log.append("No algorithm produced positive accuracy; using algorithms with non-empty forecasts as fallback")
            else:
                # Final fallback: use a simple moving average forecast
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
        target_column: str = 'total_quantity'
    ) -> Dict[str, Any]:
        """
        Safely run an algorithm and return results.

        Args:
            algorithm_name: Name of the algorithm to run
            data: Historical data (must have target_column)
            periods: Number of periods to forecast
            target_column: Name of the target column to forecast

        Returns:
            Dictionary with algorithm results
        """
        # Ensure target_column is properly set
        if not target_column:
            target_column = 'total_quantity'

        try:
            # Debug: Log data information
            logger.info(f"Algorithm {algorithm_name}: Forecast periods: {periods}")
            logger.info(f"Algorithm {algorithm_name}: Data shape: {data.shape}")
            logger.info(f"Algorithm {algorithm_name}: Data columns: {list(data.columns)}")
            logger.info(f"Algorithm {algorithm_name}: Data dtypes: {data.dtypes.to_dict()}")
            logger.info(f"Algorithm {algorithm_name}: First 5 rows:\n{data.head()}")
            logger.info(f"Algorithm {algorithm_name}: Last 5 rows:\n{data.tail()}")
            logger.info(f"Algorithm {algorithm_name}: Data summary:\n{data.describe()}")

            # Check for null values
            null_counts = data.isnull().sum()
            if null_counts.sum() > 0:
                logger.info(f"Algorithm {algorithm_name}: Null value counts:\n{null_counts}")

            # Ensure we have target_column (rename if needed)
            if 'quantity' in data.columns and target_column not in data.columns:
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
                "sarima": ForecastExecutionService.sarima_forecast
            }
            algorithm_func = algorithm_map.get(algorithm_name)
            if algorithm_func is None:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
            
            def execute(input_data: pd.DataFrame, horizon: int) -> Tuple[np.ndarray, Dict[str, float]]:
                target_data = input_data if input_data is not None else data
                if target_data is None or target_data.empty:
                    target_data = data
                return algorithm_func(target_data, max(1, horizon))
            
            train_df, test_df = ForecastExecutionService._split_train_test_dataframe(data)
            eval_data = train_df if train_df is not None and not train_df.empty else data
            eval_periods = len(test_df)
            metrics: Dict[str, float]
            if eval_periods > 0:
                eval_forecast, _ = execute(eval_data, eval_periods)
                actual_eval = test_df[target_column].values
                predicted_eval = np.asarray(eval_forecast)[:len(actual_eval)]
                metrics = ForecastExecutionService.calculate_metrics(actual_eval, predicted_eval)
            else:
                fallback_horizon = min(max(1, periods), max(1, len(eval_data)))
                _, metrics = execute(eval_data, fallback_horizon)
            
            forecast, _ = execute(data, periods)
            
            # Convert forecast to list
            forecast_list = forecast.tolist() if isinstance(forecast, np.ndarray) else forecast
            
            # Forecast quality validation removed
            
            # Extract accuracy metric (already calculated in calculate_metrics)
            accuracy = metrics.get('accuracy', 0)
            
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