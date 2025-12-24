"""
Monitoring Configuration.
Add this to your app/config.py or create app/monitoring_config.py
"""

from pydantic_settings import BaseSettings
from typing import Optional


class MonitoringSettings(BaseSettings):
    """Configuration for resource monitoring and performance tracking."""
    
    # Resource warning thresholds
    CPU_WARNING_THRESHOLD: float = 90.0  # CPU usage percentage
    MEMORY_WARNING_THRESHOLD: float = 90.0  # Memory usage percentage
    DISK_WARNING_THRESHOLD: float = 90.0  # Disk usage percentage
    
    # Performance thresholds
    SLOW_REQUEST_THRESHOLD: float = 10.0  # Seconds for general APIs
    VERY_SLOW_REQUEST_THRESHOLD: float = 40.0  # Seconds for critical slowness
    FORECAST_SLOW_THRESHOLD: float = 40.0  # Seconds for forecast operations
    
    # Monitoring features
    ENABLE_RESOURCE_MONITORING: bool = True
    ENABLE_PERFORMANCE_TRACKING: bool = True
    ENABLE_REQUEST_LOGGING: bool = True
    
    # Monitoring exclusions
    MONITORING_EXCLUDE_PATHS: list = [
        '/docs',
        '/openapi.json',
        '/redoc',
        '/health',
        '/api/v1/metrics'
    ]
    
    # Alert settings
    ENABLE_RESOURCE_ALERTS: bool = True
    ENABLE_SLOW_REQUEST_ALERTS: bool = True
    ALERT_EMAIL: Optional[str] = None  # Email for critical alerts
    
    # Logging configuration
    LOG_RESOURCE_METRICS: bool = True
    LOG_PERFORMANCE_SUMMARY_INTERVAL: int = 3600  # Log summary every hour (seconds)
    
    class Config:
        env_prefix = "MONITORING_"
        case_sensitive = True


# Create settings instance
monitoring_settings = MonitoringSettings()


# Export thresholds for easy access
CPU_WARNING_THRESHOLD = monitoring_settings.CPU_WARNING_THRESHOLD
MEMORY_WARNING_THRESHOLD = monitoring_settings.MEMORY_WARNING_THRESHOLD
DISK_WARNING_THRESHOLD = monitoring_settings.DISK_WARNING_THRESHOLD
SLOW_REQUEST_THRESHOLD = monitoring_settings.SLOW_REQUEST_THRESHOLD
VERY_SLOW_REQUEST_THRESHOLD = monitoring_settings.VERY_SLOW_REQUEST_THRESHOLD
FORECAST_SLOW_THRESHOLD = monitoring_settings.FORECAST_SLOW_THRESHOLD