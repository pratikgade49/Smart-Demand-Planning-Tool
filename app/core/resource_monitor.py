"""
Resource Monitoring Service.
Monitors system resources, request performance, and provides warnings for slow operations.
"""

import psutil
import time
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager
import asyncio

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources and request performance."""
    
    # Thresholds for warnings
    CPU_WARNING_THRESHOLD = 80.0  # CPU usage percentage
    MEMORY_WARNING_THRESHOLD = 85.0  # Memory usage percentage
    DISK_WARNING_THRESHOLD = 90.0  # Disk usage percentage
    SLOW_REQUEST_THRESHOLD = 5.0  # Seconds
    VERY_SLOW_REQUEST_THRESHOLD = 30.0  # Seconds
    
    @staticmethod
    def get_system_resources() -> Dict[str, Any]:
        """
        Get current system resource usage.
        
        Returns:
            Dictionary with CPU, memory, and disk usage metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_gb = disk.used / (1024 * 1024 * 1024)
            disk_available_gb = disk.free / (1024 * 1024 * 1024)
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            process_cpu = process.cpu_percent(interval=0.1)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu': {
                    'percent': round(cpu_percent, 2),
                    'count': cpu_count,
                    'frequency_mhz': round(cpu_freq.current, 2) if cpu_freq else None,
                    'warning': cpu_percent > ResourceMonitor.CPU_WARNING_THRESHOLD
                },
                'memory': {
                    'percent': round(memory.percent, 2),
                    'used_mb': round(memory_mb, 2),
                    'available_mb': round(memory_available_mb, 2),
                    'total_mb': round(memory.total / (1024 * 1024), 2),
                    'warning': memory.percent > ResourceMonitor.MEMORY_WARNING_THRESHOLD
                },
                'disk': {
                    'percent': round(disk.percent, 2),
                    'used_gb': round(disk_gb, 2),
                    'available_gb': round(disk_available_gb, 2),
                    'total_gb': round(disk.total / (1024 * 1024 * 1024), 2),
                    'warning': disk.percent > ResourceMonitor.DISK_WARNING_THRESHOLD
                },
                'process': {
                    'memory_mb': round(process_memory, 2),
                    'cpu_percent': round(process_cpu, 2)
                }
            }
        except Exception as e:
            logger.warning(f"Could not get system resources: {str(e)}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'cpu': {'percent': 0.0, 'warning': False},
                'memory': {'percent': 0.0, 'warning': False},
                'disk': {'percent': 0.0, 'warning': False}
            }
    
    @staticmethod
    def check_resource_warnings(resources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if any resources exceed warning thresholds.
        
        Args:
            resources: Resource metrics from get_system_resources()
            
        Returns:
            Dictionary with warning status and messages
        """
        warnings = []
        critical = False
        
        # Check CPU
        if resources['cpu'].get('warning', False):
            warnings.append(
                f"High CPU usage: {resources['cpu']['percent']}% "
                f"(threshold: {ResourceMonitor.CPU_WARNING_THRESHOLD}%)"
            )
            if resources['cpu']['percent'] > 95:
                critical = True
        
        # Check Memory
        if resources['memory'].get('warning', False):
            warnings.append(
                f"High memory usage: {resources['memory']['percent']}% "
                f"(threshold: {ResourceMonitor.MEMORY_WARNING_THRESHOLD}%)"
            )
            if resources['memory']['percent'] > 95:
                critical = True
        
        # Check Disk
        if resources['disk'].get('warning', False):
            warnings.append(
                f"High disk usage: {resources['disk']['percent']}% "
                f"(threshold: {ResourceMonitor.DISK_WARNING_THRESHOLD}%)"
            )
            if resources['disk']['percent'] > 95:
                critical = True
        
        return {
            'has_warnings': len(warnings) > 0,
            'is_critical': critical,
            'warnings': warnings,
            'timestamp': resources['timestamp']
        }
    
    @staticmethod
    def log_resource_warnings(resources: Dict[str, Any], context: str = ""):
        """
        Log resource warnings if thresholds are exceeded.
        
        Args:
            resources: Resource metrics
            context: Context string for logging (e.g., "Forecast Start", "Forecast End")
        """
        warnings = ResourceMonitor.check_resource_warnings(resources)
        
        if warnings['has_warnings']:
            level = logging.CRITICAL if warnings['is_critical'] else logging.WARNING
            logger.log(
                level,
                f"{context} - Resource Warning: {', '.join(warnings['warnings'])}",
                extra={'resources': resources, 'context': context}
            )
    
    @staticmethod
    @contextmanager
    def monitor_operation(operation_name: str, warn_threshold: Optional[float] = None):
        """
        Context manager to monitor an operation's resource usage and duration.
        
        Usage:
            with ResourceMonitor.monitor_operation("Forecast Execution"):
                # Your code here
                pass
        
        Args:
            operation_name: Name of the operation being monitored
            warn_threshold: Custom threshold in seconds for slow operation warning
        """
        if warn_threshold is None:
            warn_threshold = ResourceMonitor.SLOW_REQUEST_THRESHOLD
        
        # Get resources before operation
        start_time = time.time()
        start_resources = ResourceMonitor.get_system_resources()
        
        logger.info(
            f"{operation_name} - Started",
            extra={
                'operation': operation_name,
                'start_resources': start_resources
            }
        )
        
        # Log warnings for starting resources
        ResourceMonitor.log_resource_warnings(start_resources, f"{operation_name} Start")
        
        try:
            yield {
                'start_time': start_time,
                'start_resources': start_resources
            }
        finally:
            # Get resources after operation
            end_time = time.time()
            duration = end_time - start_time
            end_resources = ResourceMonitor.get_system_resources()
            
            # Calculate resource deltas
            cpu_delta = end_resources['cpu']['percent'] - start_resources['cpu']['percent']
            memory_delta = end_resources['memory']['percent'] - start_resources['memory']['percent']
            
            # Determine if operation was slow
            is_slow = duration > warn_threshold
            is_very_slow = duration > ResourceMonitor.VERY_SLOW_REQUEST_THRESHOLD
            
            # Log completion
            log_level = logging.WARNING if is_slow else logging.INFO
            if is_very_slow:
                log_level = logging.WARNING
            
            logger.log(
                log_level,
                f"{operation_name} - Completed in {duration:.2f}s",
                extra={
                    'operation': operation_name,
                    'duration_seconds': round(duration, 2),
                    'is_slow': is_slow,
                    'is_very_slow': is_very_slow,
                    'start_resources': start_resources,
                    'end_resources': end_resources,
                    'resource_deltas': {
                        'cpu_percent': round(cpu_delta, 2),
                        'memory_percent': round(memory_delta, 2)
                    }
                }
            )
            
            # Log warnings for ending resources
            ResourceMonitor.log_resource_warnings(end_resources, f"{operation_name} End")
            
            # Log slow operation warning
            if is_very_slow:
                logger.warning(
                    f"VERY SLOW OPERATION: {operation_name} took {duration:.2f}s "
                    f"(threshold: {ResourceMonitor.VERY_SLOW_REQUEST_THRESHOLD}s)"
                )
            elif is_slow:
                logger.warning(
                    f"Slow operation: {operation_name} took {duration:.2f}s "
                    f"(threshold: {warn_threshold}s)"
                )


def monitor_endpoint(operation_name: Optional[str] = None, warn_threshold: Optional[float] = None):
    """
    Decorator to monitor API endpoint performance and resources.
    
    Usage:
        @monitor_endpoint("Forecast Execution", warn_threshold=10.0)
        async def execute_forecast(...):
            ...
    
    Args:
        operation_name: Name of the operation (defaults to function name)
        warn_threshold: Custom threshold in seconds for slow operation warning
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with ResourceMonitor.monitor_operation(op_name, warn_threshold):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with ResourceMonitor.monitor_operation(op_name, warn_threshold):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class PerformanceTracker:
    """Track performance metrics across operations."""
    
    def __init__(self):
        self.operations: Dict[str, Dict[str, Any]] = {}
    
    def record_operation(
        self,
        operation_name: str,
        duration: float,
        start_resources: Dict[str, Any],
        end_resources: Dict[str, Any]
    ):
        """Record an operation's performance metrics."""
        if operation_name not in self.operations:
            self.operations[operation_name] = {
                'count': 0,
                'total_duration': 0.0,
                'min_duration': float('inf'),
                'max_duration': 0.0,
                'slow_count': 0,
                'very_slow_count': 0
            }
        
        op = self.operations[operation_name]
        op['count'] += 1
        op['total_duration'] += duration
        op['min_duration'] = min(op['min_duration'], duration)
        op['max_duration'] = max(op['max_duration'], duration)
        
        if duration > ResourceMonitor.VERY_SLOW_REQUEST_THRESHOLD:
            op['very_slow_count'] += 1
        elif duration > ResourceMonitor.SLOW_REQUEST_THRESHOLD:
            op['slow_count'] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary for all tracked operations."""
        summary = {}
        
        for op_name, metrics in self.operations.items():
            if metrics['count'] > 0:
                avg_duration = metrics['total_duration'] / metrics['count']
                summary[op_name] = {
                    'total_executions': metrics['count'],
                    'average_duration_seconds': round(avg_duration, 2),
                    'min_duration_seconds': round(metrics['min_duration'], 2),
                    'max_duration_seconds': round(metrics['max_duration'], 2),
                    'slow_operations': metrics['slow_count'],
                    'very_slow_operations': metrics['very_slow_count'],
                    'slow_percentage': round(
                        (metrics['slow_count'] + metrics['very_slow_count']) / metrics['count'] * 100,
                        2
                    )
                }
        
        return summary


# Global performance tracker
performance_tracker = PerformanceTracker()