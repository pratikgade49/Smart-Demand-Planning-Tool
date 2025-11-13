"""
Base class for forecasting algorithms.
All algorithm implementations must inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import logging

logger = logging.getLogger(__name__)


class ForecastAlgorithm(ABC):
    """Abstract base class for forecasting algorithms."""
    
    def __init__(
        self,
        algorithm_name: str,
        parameters: Dict[str, Any],
        forecast_start: date,
        forecast_end: date,
        interval: str = "MONTHLY"
    ):
        """
        Initialize algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            parameters: Algorithm-specific parameters
            forecast_start: Start date for forecast
            forecast_end: End date for forecast
            interval: Forecast interval (WEEKLY, MONTHLY, QUARTERLY, YEARLY)
        """
        self.algorithm_name = algorithm_name
        self.parameters = parameters
        self.forecast_start = forecast_start
        self.forecast_end = forecast_end
        self.interval = interval
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(
        self,
        historical_data: pd.DataFrame,
        external_factors: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Train the forecasting model.
        
        Args:
            historical_data: Historical data with columns [period, total_quantity]
            external_factors: Optional external factors data
        """
        pass
    
    @abstractmethod
    def predict(self) -> pd.DataFrame:
        """
        Generate forecast predictions.
        
        Returns:
            DataFrame with columns [forecast_date, forecast_quantity, 
                                   confidence_interval_lower, confidence_interval_upper]
        """
        pass
    
    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data format."""
        required_columns = ['period', 'total_quantity']
        missing_cols = set(required_columns) - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("Historical data cannot be empty")
        
        if df['total_quantity'].isna().all():
            raise ValueError("All quantity values are null")
    
    def generate_forecast_periods(self) -> List[date]:
        """Generate list of forecast periods based on interval."""
        periods = []
        current = self.forecast_start
        
        while current <= self.forecast_end:
            periods.append(current)
            
            if self.interval == "WEEKLY":
                current += timedelta(weeks=1)
            elif self.interval == "MONTHLY":
                current += relativedelta(months=1)
            elif self.interval == "QUARTERLY":
                current += relativedelta(months=3)
            elif self.interval == "YEARLY":
                current += relativedelta(years=1)
            else:
                current += relativedelta(months=1)  # Default to monthly
        
        return periods
    
    def calculate_confidence_intervals(
        self,
        predictions: np.ndarray,
        std_error: float,
        confidence_level: float = 0.95
    ) -> tuple:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions: Array of predicted values
            std_error: Standard error of predictions
            confidence_level: Confidence level (default 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        from scipy import stats
        
        # Z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        margin = z_score * std_error
        lower_bounds = predictions - margin
        upper_bounds = predictions + margin
        
        # Ensure non-negative forecasts
        lower_bounds = np.maximum(lower_bounds, 0)
        
        return lower_bounds, upper_bounds
    
    def calculate_accuracy_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with MAPE, MAE, RMSE, R-squared
        """
        # Remove NaN values
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0:
            return {
                "mape": None,
                "mae": None,
                "rmse": None,
                "r_squared": None
            }
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100
        
        # Mean Absolute Error
        mae = np.mean(np.abs(actual - predicted))
        
        # Root Mean Square Error
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "mape": round(float(mape), 2),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r_squared": round(float(r_squared), 4)
        }
    
    def prepare_time_series(
        self,
        df: pd.DataFrame,
        date_column: str = 'period',
        value_column: str = 'total_quantity'
    ) -> pd.Series:
        """
        Prepare time series data for modeling.
        
        Args:
            df: Input DataFrame
            date_column: Name of date column
            value_column: Name of value column
            
        Returns:
            Time series with DatetimeIndex
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        df = df.set_index(date_column)
        
        # Handle missing values
        series = df[value_column].fillna(method='ffill').fillna(method='bfill')
        
        return series
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata for storage."""
        return {
            "algorithm_name": self.algorithm_name,
            "parameters": self.parameters,
            "interval": self.interval,
            "forecast_start": self.forecast_start.isoformat(),
            "forecast_end": self.forecast_end.isoformat(),
            "is_fitted": self.is_fitted
        }