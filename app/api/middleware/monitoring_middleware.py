"""
API Monitoring Middleware.
Automatically monitors all API requests with resource tracking and performance metrics.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import time
import logging
from typing import Callable
from app.core.resource_monitor import ResourceMonitor, performance_tracker

logger = logging.getLogger(__name__)


class ResourceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware to monitor resource usage and performance for all API requests.
    
    Automatically tracks:
    - Request duration
    - CPU usage before/after
    - Memory usage before/after
    - Warnings for slow requests
    - Warnings for high resource usage
    """
    
    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: list = None,
        slow_request_threshold: float = 5.0
    ):
        """
        Initialize monitoring middleware.
        
        Args:
            app: FastAPI application
            exclude_paths: List of path prefixes to exclude from monitoring
            slow_request_threshold: Threshold in seconds for slow request warnings
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or ['/docs', '/openapi.json', '/redoc', '/health']
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with resource monitoring.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint handler
            
        Returns:
            HTTP response with monitoring headers
        """
        # Skip monitoring for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Get start resources
        start_time = time.time()
        start_resources = ResourceMonitor.get_system_resources()
        
        # Log request start with resources
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                'method': request.method,
                'path': request.url.path,
                'client': request.client.host if request.client else None,
                'start_resources': start_resources
            }
        )
        
        # Log resource warnings at start
        ResourceMonitor.log_resource_warnings(
            start_resources,
            f"Request Start: {request.method} {request.url.path}"
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error with resources
            end_resources = ResourceMonitor.get_system_resources()
            duration = time.time() - start_time
            
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    'method': request.method,
                    'path': request.url.path,
                    'duration_seconds': round(duration, 2),
                    'error': str(e),
                    'end_resources': end_resources
                },
                exc_info=True
            )
            raise
        
        # Get end resources
        end_time = time.time()
        duration = end_time - start_time
        end_resources = ResourceMonitor.get_system_resources()
        
        # Calculate resource deltas
        cpu_delta = end_resources['cpu']['percent'] - start_resources['cpu']['percent']
        memory_delta = end_resources['memory']['percent'] - start_resources['memory']['percent']
        
        # Determine if request was slow
        is_slow = duration > self.slow_request_threshold
        is_very_slow = duration > ResourceMonitor.VERY_SLOW_REQUEST_THRESHOLD
        
        # Record performance
        operation_name = f"{request.method} {request.url.path}"
        performance_tracker.record_operation(
            operation_name,
            duration,
            start_resources,
            end_resources
        )
        
        # Log request completion with appropriate level
        log_level = logging.WARNING if is_slow else logging.INFO
        if is_very_slow:
            log_level = logging.WARNING
        
        logger.log(
            log_level,
            f"Request completed: {request.method} {request.url.path} - "
            f"{response.status_code} in {duration:.2f}s",
            extra={
                'method': request.method,
                'path': request.url.path,
                'status_code': response.status_code,
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
        
        # Log resource warnings at end
        ResourceMonitor.log_resource_warnings(
            end_resources,
            f"Request End: {request.method} {request.url.path}"
        )
        
        # Log slow request warnings
        if is_very_slow:
            logger.warning(
                f"VERY SLOW REQUEST: {request.method} {request.url.path} took {duration:.2f}s "
                f"(threshold: {ResourceMonitor.VERY_SLOW_REQUEST_THRESHOLD}s)"
            )
        elif is_slow:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} took {duration:.2f}s "
                f"(threshold: {self.slow_request_threshold}s)"
            )
        
        # Add monitoring headers to response
        response.headers['X-Request-Duration'] = f"{duration:.3f}"
        response.headers['X-CPU-Usage'] = f"{end_resources['cpu']['percent']:.1f}"
        response.headers['X-Memory-Usage'] = f"{end_resources['memory']['percent']:.1f}"
        
        if is_slow:
            response.headers['X-Performance-Warning'] = 'slow-request'
        if end_resources['cpu'].get('warning') or end_resources['memory'].get('warning'):
            response.headers['X-Resource-Warning'] = 'high-usage'
        
        return response


class PerformanceMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to expose performance metrics endpoint.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and handle metrics endpoint."""
        
        # Handle metrics endpoint
        if request.url.path == '/api/v1/metrics/performance':
            from fastapi.responses import JSONResponse
            
            summary = performance_tracker.get_summary()
            current_resources = ResourceMonitor.get_system_resources()
            
            return JSONResponse({
                'status': 'success',
                'data': {
                    'current_resources': current_resources,
                    'performance_summary': summary
                }
            })
        
        return await call_next(request)