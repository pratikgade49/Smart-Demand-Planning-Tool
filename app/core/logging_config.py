"""
Enhanced logging configuration for the Smart Demand Planning Tool.
Provides structured logging with different levels and handlers.
"""

import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import psutil
import tracemalloc

from app.config import settings


class PerformanceLogger:
    """Logger for tracking performance metrics (memory and CPU)."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.process = psutil.Process()
        
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage."""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()
        
        self.logger.info(
            f"MEMORY - {context}: {memory_mb:.2f} MB ({memory_percent:.2f}%)",
            extra={
                "metric_type": "memory",
                "context": context,
                "memory_mb": memory_mb,
                "memory_percent": memory_percent
            }
        )
    
    def log_cpu_usage(self, context: str = "", interval: float = 0.1):
        """Log current CPU usage."""
        cpu_percent = self.process.cpu_percent(interval=interval)
        
        self.logger.info(
            f"CPU - {context}: {cpu_percent:.2f}%",
            extra={
                "metric_type": "cpu",
                "context": context,
                "cpu_percent": cpu_percent
            }
        )
    
    def log_performance_snapshot(self, context: str = ""):
        """Log both memory and CPU usage."""
        self.log_memory_usage(context)
        self.log_cpu_usage(context)


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def filter(self, record):
        # Add process info
        record.pid = os.getpid()
        
        # Add memory info if available
        if hasattr(record, 'memory_mb'):
            record.memory_info = f"[MEM: {record.memory_mb:.2f}MB]"
        else:
            record.memory_info = ""
        
        # Add CPU info if available
        if hasattr(record, 'cpu_percent'):
            record.cpu_info = f"[CPU: {record.cpu_percent:.2f}%]"
        else:
            record.cpu_info = ""
        
        return True


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        if sys.stdout.isatty():  # Only use colors if outputting to terminal
            levelname = record.levelname
            record.levelname = f"{self.COLORS.get(levelname, '')}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_performance_logging: bool = False
) -> None:
    """
    Setup comprehensive logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        enable_file_logging: Whether to enable file logging
        enable_performance_logging: Whether to track memory allocations
    """
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory
    if log_dir is None:
        log_dir = "logs"
    Path(log_dir).mkdir(exist_ok=True)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s %(memory_info)s%(cpu_info)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)
    
    if enable_file_logging:
        # General application log (rotating)
        app_log_file = os.path.join(log_dir, 'app.log')
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        app_handler.setLevel(numeric_level)
        app_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - [PID:%(pid)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        app_handler.setFormatter(app_formatter)
        app_handler.addFilter(ContextFilter())
        root_logger.addHandler(app_handler)
        
        # Error log (only errors and critical)
        error_log_file = os.path.join(log_dir, 'error.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - [PID:%(pid)s] - %(message)s\n'
            'Location: %(pathname)s:%(lineno)d\n'
            'Function: %(funcName)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        error_handler.addFilter(ContextFilter())
        root_logger.addHandler(error_handler)
        
        # Performance log (memory and CPU metrics)
        perf_log_file = os.path.join(log_dir, 'performance.log')
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3
        )
        perf_handler.setLevel(logging.INFO)
        perf_formatter = logging.Formatter(
            '%(asctime)s - [PERF] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        # Only log performance metrics
        perf_handler.addFilter(lambda record: hasattr(record, 'metric_type'))
        root_logger.addHandler(perf_handler)
    
    # Enable memory tracking if requested
    if enable_performance_logging:
        tracemalloc.start()
        logging.info("Memory tracking enabled")
    
    # Set specific log levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    
    logging.info(f"Logging initialized - Level: {log_level}, File Logging: {enable_file_logging}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with performance tracking capabilities.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # Attach performance logger
    if not hasattr(logger, 'perf'):
        logger.perf = PerformanceLogger(logger)
    
    return logger


# Convenience functions for structured logging
def log_operation_start(logger: logging.Logger, operation: str, **kwargs):
    """Log the start of an operation with context."""
    logger.info(
        f"Starting operation: {operation}",
        extra={"operation": operation, "phase": "start", **kwargs}
    )


def log_operation_end(logger: logging.Logger, operation: str, success: bool = True, **kwargs):
    """Log the end of an operation with context."""
    status = "completed" if success else "failed"
    level = logging.INFO if success else logging.ERROR
    logger.log(
        level,
        f"Operation {status}: {operation}",
        extra={"operation": operation, "phase": "end", "success": success, **kwargs}
    )


def log_database_query(logger: logging.Logger, query_type: str, table: str, duration_ms: float):
    """Log database query execution."""
    logger.debug(
        f"DB Query - {query_type} on {table} - {duration_ms:.2f}ms",
        extra={
            "query_type": query_type,
            "table": table,
            "duration_ms": duration_ms
        }
    )


def log_api_request(logger: logging.Logger, method: str, path: str, status_code: int, duration_ms: float):
    """Log API request."""
    logger.info(
        f"API {method} {path} - {status_code} - {duration_ms:.2f}ms",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms
        }
    )


# Initialize logging on module import if running as main application
if settings.DEBUG:
    setup_logging(
        log_level="DEBUG",
        enable_file_logging=True,
        enable_performance_logging=True
    )
else:
    setup_logging(
        log_level="INFO",
        enable_file_logging=True,
        enable_performance_logging=False
    )