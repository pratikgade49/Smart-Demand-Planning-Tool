"""
Main FastAPI Application with Enhanced Logging.
Entry point for the Smart Demand Planning Tool API.
"""

from fastapi import FastAPI, Request, status # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import JSONResponse # type: ignore
from contextlib import asynccontextmanager
import time
from datetime import datetime

from app.config import settings
from app.core.database import get_db_manager
from app.core.exceptions import AppException
from app.core.logging_config import (
    setup_logging,
    get_logger,
    log_operation_start,
    log_operation_end,
    log_api_request
)

from app.api.routes import auth_routes, field_catalogue_routes, upload_routes, forecasting_routes
from app.api.routes import external_factors_routes
from app.api.routes import forecast_comparison_routes
from app.api.routes import sap_ibp_routes
from app.api.routes import sales_data_routes
from app.api.routes import dashboard_routes
from app.api.routes import master_data_routes
from app.api.middleware.monitoring_middleware import ResourceMonitoringMiddleware
from app.api.middleware.request_audit_middleware import RequestAuditMiddleware



# Setup logging before anything else
setup_logging(
    log_level="DEBUG" if settings.DEBUG else "INFO",
    enable_file_logging=True,
    enable_performance_logging=settings.DEBUG
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("=" * 80)
    logger.info("Starting Smart Demand Planning Tool API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"Version: {settings.APP_VERSION}")
    logger.info("=" * 80)
    
    # Log initial performance metrics
    logger.perf.log_performance_snapshot("Application Startup")
    
    # Initialize database connection pool
    try:
        log_operation_start(logger, "database_initialization")
        db_manager = get_db_manager()
        pool_status = db_manager.get_pool_status()
        logger.info(f"Database connection pool initialized: {pool_status}")
        log_operation_end(logger, "database_initialization", success=True)
    except Exception as e:
        logger.critical(f"Failed to initialize database: {str(e)}", exc_info=True)
        log_operation_end(logger, "database_initialization", success=False, error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("Shutting down Smart Demand Planning Tool API")
    logger.perf.log_performance_snapshot("Application Shutdown")
    
    try:
        log_operation_start(logger, "database_shutdown")
        db_manager = get_db_manager()
        db_manager.close_pool()
        logger.info("Database connections closed successfully")
        log_operation_end(logger, "database_shutdown", success=True)
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)
        log_operation_end(logger, "database_shutdown", success=False, error=str(e))
    
    logger.info("Shutdown complete")
    logger.info("=" * 80)


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-tenant demand planning tool with dynamic field catalogue",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(ResourceMonitoringMiddleware)
app.add_middleware(
    RequestAuditMiddleware,
    exclude_paths=[
        "/api/docs",
        "/api/openapi.json",
        "/api/redoc",
        "/api/v1/metrics",
    ],
)

logger.info("CORS middleware configured")


# Request timing and logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing information."""
    # Start timing
    start_time = time.time()
    request.state.timestamp = datetime.utcnow().isoformat()
    
    # Log incoming request
    logger.debug(
        f"Incoming request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_host": request.client.host if request.client else "unknown",
            "query_params": dict(request.query_params)
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        log_api_request(
            logger,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms
        )
        
        # Log performance metrics for slow requests
        if duration_ms > 1000:  # Slow request threshold: 1 second
            logger.warning(
                f"Slow request detected: {request.method} {request.url.path} - {duration_ms:.2f}ms",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": duration_ms,
                    "slow_request": True
                }
            )
            logger.perf.log_performance_snapshot(f"Slow Request: {request.url.path}")
        
        # Add custom headers
        response.headers["X-Request-ID"] = request.state.timestamp
        response.headers["X-Process-Time"] = str(duration_ms)
        
        return response
        
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            f"Request failed: {request.method} {request.url.path} - {str(e)}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration_ms,
                "error": str(e)
            },
            exc_info=True
        )
        raise


# Global exception handler
@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    """Handle custom application exceptions."""
    logger.warning(
        f"Application exception: {exc.error_code} - {exc.message}",
        extra={
            "error_code": exc.error_code,
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details
            },
            "metadata": {
                "timestamp": request.state.timestamp if hasattr(request.state, 'timestamp') else None,
                "status_code": exc.status_code
            }
        }
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )
    
    # Log performance snapshot on critical errors
    logger.perf.log_performance_snapshot("Unhandled Exception")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred" if not settings.DEBUG else str(exc),
                "details": {}
            },
            "metadata": {
                "timestamp": request.state.timestamp if hasattr(request.state, 'timestamp') else None,
                "status_code": 500
            }
        }
    )


# Then register the forecasting router with other routers
logger.info("Registering API routes...")
app.include_router(auth_routes.router, prefix="/api/v1")
logger.debug("Registered auth routes")
app.include_router(field_catalogue_routes.router, prefix="/api/v1")
logger.debug("Registered field catalogue routes")
app.include_router(upload_routes.router, prefix="/api/v1")
logger.debug("Registered upload routes")
app.include_router(forecasting_routes.router, prefix="/api/v1")  # NEW LINE
logger.debug("Registered forecasting routes")  # NEW LINE
app.include_router(external_factors_routes.router, prefix="/api/v1")
logger.debug("Registered external factors routes")
app.include_router(forecast_comparison_routes.router, prefix="/api/v1")
logger.debug("Registered forecast comparison routes")
app.include_router(sap_ibp_routes.router, prefix="/api/v1")
logger.debug("Registered SAP IBP routes")
app.include_router(master_data_routes.router, prefix="/api/v1")
logger.debug("Registered master data routes")
app.include_router(sales_data_routes.router, prefix="/api/v1")
logger.debug("Registered sales data routes")
app.include_router(dashboard_routes.router, prefix="/api/v1")
logger.debug("Registered dashboard routes")
logger.info("All API routes registered successfully")


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint with system metrics."""
    logger.debug("Health check requested")
    
    try:
        # Check database connection
        db_manager = get_db_manager()
        pool_status = db_manager.get_pool_status()
        db_healthy = pool_status["master_pool"]["initialized"]
        
        # Get performance metrics
        import psutil # type: ignore
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        health_data = {
            "status": "healthy" if db_healthy else "degraded",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "database": {
                "master_pool_initialized": db_healthy,
                "active_tenant_pools": pool_status["master_pool"]["active_pools"]
            },
            "performance": {
                "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
                "cpu_percent": round(cpu_percent, 2)
            }
        }
        
        logger.info(f"Health check: {health_data['status']}")
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "error": str(e) if settings.DEBUG else "Health check failed"
        }


# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    logger.debug("Root endpoint accessed")
    
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/api/docs",
        "health": "/health",
        "api_prefix": "/api/v1"
    }


# Debug endpoint (only in debug mode)
if settings.DEBUG:
    @app.get("/debug/performance", tags=["Debug"])
    async def debug_performance():
        """Debug endpoint for performance metrics."""
        logger.debug("Performance debug endpoint accessed")
        
        import psutil # type: ignore
        import tracemalloc
        
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        perf_data = {
            "memory": {
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": round(process.memory_percent(), 2)
            },
            "cpu": {
                "percent": round(cpu_percent, 2),
                "num_threads": process.num_threads()
            },
            "database": get_db_manager().get_pool_status()
        }
        
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            perf_data["tracemalloc"] = {
                "current_mb": round(current / 1024 / 1024, 2),
                "peak_mb": round(peak / 1024 / 1024, 2)
            }
        
        return perf_data


if __name__ == "__main__":
    import uvicorn # type: ignore
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_config=None,  # Use our custom logging configuration
        access_log=False  # Handled by our middleware
    )
