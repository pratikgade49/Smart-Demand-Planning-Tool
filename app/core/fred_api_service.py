"""
FRED API Integration Service.
Fetches economic data from Federal Reserve Economic Data (FRED).
"""

import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, date
import pandas as pd
import logging
from app.core.exceptions import ValidationException, DatabaseException

logger = logging.getLogger(__name__)


class FREDAPIService:
    """Service for integrating with FRED API."""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Common FRED series IDs
    COMMON_SERIES = {
        "GDP": {"id": "GDP", "name": "Gross Domestic Product", "unit": "Billions of Dollars"},
        "CPI": {"id": "CPIAUCSL", "name": "Consumer Price Index", "unit": "Index 1982-1984=100"},
        "UNEMPLOYMENT": {"id": "UNRATE", "name": "Unemployment Rate", "unit": "Percent"},
        "INTEREST_RATE": {"id": "FEDFUNDS", "name": "Federal Funds Rate", "unit": "Percent"},
        "INFLATION": {"id": "FPCPITOTLZGUSA", "name": "Inflation Rate", "unit": "Percent"},
        "RETAIL_SALES": {"id": "RSXFS", "name": "Retail Sales", "unit": "Millions of Dollars"},
        "INDUSTRIAL_PRODUCTION": {"id": "INDPRO", "name": "Industrial Production Index", "unit": "Index 2017=100"},
        "HOUSING_STARTS": {"id": "HOUST", "name": "Housing Starts", "unit": "Thousands of Units"},
        "CONSUMER_SENTIMENT": {"id": "UMCSENT", "name": "Consumer Sentiment Index", "unit": "Index 1966:Q1=100"}
    }
    
    def __init__(self, api_key: str):
        """
        Initialize FRED API service.
        
        Args:
            api_key: FRED API key (get from https://fred.stlouisfed.org/docs/api/api_key.html)
        """
        self.api_key = api_key
        if not api_key:
            raise ValidationException("FRED API key is required")
    
    def get_series_data(
        self,
        series_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Fetch time series data from FRED.
        
        Args:
            series_id: FRED series ID (e.g., 'GDP', 'CPIAUCSL')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with date and value columns
        """
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json"
            }
            
            if start_date:
                params["observation_start"] = start_date.strftime("%Y-%m-%d")
            if end_date:
                params["observation_end"] = end_date.strftime("%Y-%m-%d")
            
            response = requests.get(
                f"{self.BASE_URL}/series/observations",
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ValidationException(
                    f"FRED API error: {response.status_code} - {response.text}"
                )
            
            data = response.json()
            
            if "observations" not in data:
                raise ValidationException("No data returned from FRED API")
            
            observations = data["observations"]
            
            # Convert to DataFrame
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Remove missing values (marked as '.')
            df = df[df['value'].notna()]
            
            logger.info(f"Fetched {len(df)} observations for series {series_id}")
            return df[['date', 'value']]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {str(e)}")
            raise DatabaseException(f"Failed to fetch data from FRED: {str(e)}")
    
    def get_series_info(self, series_id: str) -> Dict[str, Any]:
        """
        Get metadata about a FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Dictionary with series metadata
        """
        try:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json"
            }
            
            response = requests.get(
                f"{self.BASE_URL}/series",
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ValidationException(
                    f"FRED API error: {response.status_code} - {response.text}"
                )
            
            data = response.json()
            
            if "seriess" not in data or len(data["seriess"]) == 0:
                raise ValidationException(f"Series {series_id} not found")
            
            series_info = data["seriess"][0]
            
            return {
                "series_id": series_info["id"],
                "title": series_info["title"],
                "units": series_info["units"],
                "frequency": series_info["frequency"],
                "seasonal_adjustment": series_info.get("seasonal_adjustment", "Not Seasonally Adjusted"),
                "last_updated": series_info.get("last_updated"),
                "observation_start": series_info.get("observation_start"),
                "observation_end": series_info.get("observation_end")
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {str(e)}")
            raise DatabaseException(f"Failed to fetch series info from FRED: {str(e)}")
    
    def search_series(self, search_text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for FRED series by keyword.
        
        Args:
            search_text: Search keyword
            limit: Maximum number of results
            
        Returns:
            List of series metadata
        """
        try:
            params = {
                "search_text": search_text,
                "api_key": self.api_key,
                "file_type": "json",
                "limit": limit
            }
            
            response = requests.get(
                f"{self.BASE_URL}/series/search",
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                raise ValidationException(
                    f"FRED API error: {response.status_code} - {response.text}"
                )
            
            data = response.json()
            
            if "seriess" not in data:
                return []
            
            results = []
            for series in data["seriess"]:
                results.append({
                    "series_id": series["id"],
                    "title": series["title"],
                    "units": series.get("units", ""),
                    "frequency": series.get("frequency", ""),
                    "observation_start": series.get("observation_start"),
                    "observation_end": series.get("observation_end")
                })
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {str(e)}")
            raise DatabaseException(f"Failed to search FRED series: {str(e)}")
    
    def forecast_future_values(
        self,
        series_id: str,
        historical_data: pd.DataFrame,
        periods: int
    ) -> pd.DataFrame:
        """
        Forecast future values for an external factor using simple methods.
        
        Args:
            series_id: FRED series ID
            historical_data: Historical data DataFrame with 'date' and 'value'
            periods: Number of periods to forecast
            
        Returns:
            DataFrame with forecasted dates and values
        """
        try:
            from sklearn.linear_model import LinearRegression
            import numpy as np
            
            if len(historical_data) < 3:
                raise ValidationException("Insufficient historical data for forecasting")
            
            # Sort by date
            df = historical_data.sort_values('date').copy()
            
            # Create time index
            df['time_index'] = range(len(df))
            
            # Fit linear regression
            X = df['time_index'].values.reshape(-1, 1)
            y = df['value'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast future values
            last_index = df['time_index'].max()
            future_indices = np.arange(last_index + 1, last_index + periods + 1).reshape(-1, 1)
            future_values = model.predict(future_indices)
            
            # Generate future dates
            last_date = df['date'].max()
            freq = self._infer_frequency(df['date'])
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=periods,
                freq=freq
            )
            
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'value': future_values,
                'is_forecast': True
            })
            
            logger.info(f"Forecasted {periods} future values for series {series_id}")
            return forecast_df
            
        except Exception as e:
            logger.error(f"Failed to forecast future values: {str(e)}")
            raise ValidationException(f"Forecasting failed: {str(e)}")
    
    def _infer_frequency(self, dates: pd.Series) -> str:
        """
        Infer frequency from date series.
        
        Args:
            dates: Series of dates
            
        Returns:
            Frequency string for pandas
        """
        if len(dates) < 2:
            return 'D'  # Default to daily
        
        # Calculate average difference in days
        diff_days = (dates.max() - dates.min()).days / (len(dates) - 1)
        
        if diff_days <= 1.5:
            return 'D'  # Daily
        elif diff_days <= 7.5:
            return 'W'  # Weekly
        elif diff_days <= 31:
            return 'MS'  # Monthly
        elif diff_days <= 92:
            return 'QS'  # Quarterly
        else:
            return 'YS'  # Yearly
    
    @classmethod
    def get_common_series_list(cls) -> List[Dict[str, Any]]:
        """
        Get list of commonly used FRED series.
        
        Returns:
            List of series metadata
        """
        return [
            {
                "key": key,
                "series_id": value["id"],
                "name": value["name"],
                "unit": value["unit"]
            }
            for key, value in cls.COMMON_SERIES.items()
        ]