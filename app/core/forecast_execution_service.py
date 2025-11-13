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
from psycopg2.extras import Json

from app.core.database import get_db_manager
from app.core.exceptions import DatabaseException, ValidationException, NotFoundException
from app.core.forecasting_service import ForecastingService
from app.core.external_factors_service import ExternalFactorsService

logger = logging.getLogger(__name__)


class ForecastExecutionService:
    """Service for executing forecast algorithms and storing results."""

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
        db_manager = get_db_manager()
        
        try:
            # Get forecast run details
            forecast_run = ForecastingService.get_forecast_run(
                tenant_id, database_name, forecast_run_id
            )
            
            # Validate run status
            if forecast_run['run_status'] not in ['Pending', 'Failed']:
                raise ValidationException(
                    f"Cannot execute forecast run with status: {forecast_run['run_status']}"
                )
            
            # Update status to In-Progress
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
                finally:
                    cursor.close()
            
            logger.info(f"Starting forecast execution for run: {forecast_run_id}")
            
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
            
            # Route to appropriate algorithm
            if algorithm_name.lower() == 'linear regression':
                forecast, metrics = ForecastExecutionService.linear_regression_forecast(
                    data=historical_data,
                    periods=periods
                )
            elif algorithm_name.lower() == 'arima':
                forecast, metrics = ForecastExecutionService.arima_forecast(
                    data=historical_data,
                    periods=periods,
                    order=custom_params.get('order', [1, 1, 1])
                )
            elif algorithm_name.lower() == 'exponential smoothing':
                forecast, metrics = ForecastExecutionService.exponential_smoothing_forecast(
                    data=historical_data,
                    periods=periods,
                    alpha=custom_params.get('alpha', 0.3)
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
        """
        Linear regression forecasting with feature engineering.
        
        Args:
            data: Historical data with 'quantity' column and optional external factors
            periods: Number of periods to forecast
            
        Returns:
            Tuple of (forecast_array, metrics_dict)
        """
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
        
        # Feature engineering: create lag features and time index
        window = min(5, n - 1)
        
        # Get external factor columns
        excluded_cols = ['date', 'quantity', 'total_quantity', 'period', 'product', 
                        'customer', 'location', 'transaction_count', 'avg_price']
        external_factor_cols = [col for col in data.columns if col not in excluded_cols]
        
        logger.debug(f"External factor columns: {external_factor_cols}")
        
        if window < 1:
            # Not enough data for feature engineering, fallback to simple time-based regression
            x = np.arange(n).reshape(-1, 1)
            if external_factor_cols:
                x = np.hstack([x, data[external_factor_cols].fillna(0).values])
            
            model = LinearRegression()
            model.fit(x, y)
            
            future_x = np.arange(n, n + periods).reshape(-1, 1)
            if external_factor_cols:
                # Use last known external factor values
                last_factors = data[external_factor_cols].fillna(0).iloc[-1].values
                future_factors = np.tile(last_factors, (periods, 1))
                future_x = np.hstack([future_x, future_factors])
            
            forecast = model.predict(future_x)
            forecast = np.maximum(forecast, 0)
            
            predicted = model.predict(x)
            metrics = ForecastExecutionService.calculate_metrics(y, predicted)
            
            return forecast, metrics
        
        # Build lagged features
        X = []
        y_target = []
        
        for i in range(window, n):
            lags = y[i-window:i]
            time_idx = i
            features = list(lags) + [time_idx]
            
            if external_factor_cols:
                features.extend(data[external_factor_cols].fillna(0).iloc[i].values)
            
            X.append(features)
            y_target.append(y[i])
        
        X = np.array(X)
        y_target = np.array(y_target)
        
        logger.debug(f"Feature engineered data: X shape {X.shape}, y shape {y_target.shape}")
        
        # Train model
        model = LinearRegression()
        model.fit(X, y_target)
        
        # Generate forecast
        forecast = []
        recent_lags = list(y[-window:])
        
        for i in range(periods):
            features = recent_lags + [n + i]
            
            if external_factor_cols:
                last_factors = data[external_factor_cols].fillna(0).iloc[-1].values
                features.extend(last_factors)
            
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
    def exponential_smoothing_forecast(data: pd.DataFrame, periods: int, alpha: float = 0.3) -> Tuple[np.ndarray, Dict[str, float]]:
        """Exponential Smoothing forecasting."""
        y = data['total_quantity'].values if 'total_quantity' in data.columns else data['quantity'].values
        
        # Simple exponential smoothing
        forecast = []
        smoothed = [y[0]]
        
        for i in range(1, len(y)):
            smoothed.append(alpha * y[i] + (1 - alpha) * smoothed[i-1])
        
        # Forecast
        last_smoothed = smoothed[-1]
        forecast = [last_smoothed] * periods
        forecast = np.array(forecast)
        
        metrics = ForecastExecutionService.calculate_metrics(y[1:], smoothed[1:])
        
        return forecast, metrics

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