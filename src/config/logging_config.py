"""
Logging configuration for the social infrastructure prediction system.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any
from loguru import logger


def setup_logging(
    level: str = "INFO",
    format_string: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation: str = "10 MB",
    retention: str = "30 days",
    log_dir: str = "monitoring/logs"
) -> None:
    """
    Set up logging configuration using loguru.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Log message format
        rotation: When to rotate log files
        retention: How long to keep log files
        log_dir: Directory to store log files
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=level,
        format=format_string,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Add file handler for general logs
    logger.add(
        log_path / "app.log",
        level=level,
        format=format_string,
        rotation=rotation,
        retention=retention,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler for errors only
    logger.add(
        log_path / "errors.log",
        level="ERROR",
        format=format_string,
        rotation=rotation,
        retention=retention,
        backtrace=True,
        diagnose=True
    )


def get_logger(name: str):
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class LoggingMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    @property
    def logger(self):
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def configure_external_loggers(level: str = "WARNING") -> None:
    """
    Configure logging for external libraries.
    
    Args:
        level: Logging level for external libraries
    """
    import logging
    
    # Set logging level for common libraries
    external_loggers = [
        "urllib3",
        "requests",
        "matplotlib",
        "PIL",
        "tensorflow",
        "torch",
        "sklearn",
        "pandas",
        "numpy"
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        func_logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            func_logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            func_logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_performance(func):
    """
    Decorator to log function performance metrics.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    import time
    
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            func_logger.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            func_logger.error(f"{func.__name__} failed after {execution_time:.4f} seconds: {e}")
            raise
    
    return wrapper