import logging
import logging.handlers
import os
import json
from datetime import datetime
from functools import wraps
from typing import Any, Optional, Dict
import asyncio

# Create logs directory if it doesn't exist
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_FILE = os.path.join(LOGS_DIR, "debug.log")
ERROR_FILE = os.path.join(LOGS_DIR, "error.log")

class CustomLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CustomLogger, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        """Initialize the logger with both file and console handlers."""
        self.logger = logging.getLogger("LLaVA-Video-Service")
        self.logger.setLevel(logging.DEBUG)

        # Prevent duplicate handlers
        if self.logger.handlers:
            return

        # Console Handler (INFO and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(LOG_FORMAT)
        console_handler.setFormatter(console_formatter)

        # Debug File Handler (DEBUG and above, with rotation)
        debug_handler = logging.handlers.RotatingFileHandler(
            DEBUG_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(LOG_FORMAT))

        # Error File Handler (ERROR and above, with rotation)
        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_FILE,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(LOG_FORMAT))

        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(debug_handler)
        self.logger.addHandler(error_handler)

    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger

def sanitize_log_data(data: Any) -> Any:
    """
    Sanitize data before logging to prevent logging of large objects or sensitive information.
    """
    if isinstance(data, (str, int, float, bool)):
        return data
    elif isinstance(data, dict):
        return {k: sanitize_log_data(v) for k, v in data.items() 
                if not any(skip in str(k).lower() for skip in ['image', 'file', 'binary', 'password', 'token'])}
    elif isinstance(data, (list, tuple)):
        return [sanitize_log_data(item) for item in data]
    else:
        return str(type(data))

def log_function_call(skip_args: bool = False):
    """
    Decorator to log function entry, exit, and execution time.
    Supports both sync and async functions.
    
    Args:
        skip_args (bool): If True, function arguments won't be logged
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                logger = CustomLogger().get_logger()
                func_name = func.__name__
                
                # Log function entry
                if not skip_args:
                    # Sanitize and format arguments
                    args_repr = [sanitize_log_data(arg) for arg in args]
                    kwargs_repr = {k: sanitize_log_data(v) for k, v in kwargs.items()}
                    logger.debug(f"Entering {func_name} - Args: {args_repr}, Kwargs: {kwargs_repr}")
                else:
                    logger.debug(f"Entering {func_name}")

                start_time = datetime.now()
                try:
                    result = await func(*args, **kwargs)
                    # Log successful execution
                    execution_time = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Exiting {func_name} - Execution time: {execution_time:.2f}s")
                    return result
                except Exception as e:
                    # Log exception
                    execution_time = (datetime.now() - start_time).total_seconds()
                    logger.exception(f"Exception in {func_name} after {execution_time:.2f}s: {str(e)}")
                    raise

            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                logger = CustomLogger().get_logger()
                func_name = func.__name__
                
                # Log function entry
                if not skip_args:
                    # Sanitize and format arguments
                    args_repr = [sanitize_log_data(arg) for arg in args]
                    kwargs_repr = {k: sanitize_log_data(v) for k, v in kwargs.items()}
                    logger.debug(f"Entering {func_name} - Args: {args_repr}, Kwargs: {kwargs_repr}")
                else:
                    logger.debug(f"Entering {func_name}")

                start_time = datetime.now()
                try:
                    result = func(*args, **kwargs)
                    # Log successful execution
                    execution_time = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Exiting {func_name} - Execution time: {execution_time:.2f}s")
                    return result
                except Exception as e:
                    # Log exception
                    execution_time = (datetime.now() - start_time).total_seconds()
                    logger.exception(f"Exception in {func_name} after {execution_time:.2f}s: {str(e)}")
                    raise

            return sync_wrapper
    return decorator

# Get logger instance
def get_logger():
    """Get the configured logger instance."""
    return CustomLogger().get_logger()

# Example usage:
# logger = get_logger()
# logger.debug("Debug message")
# logger.info("Info message")
# logger.warning("Warning message")
# logger.error("Error message")
# 
# @log_function_call()
# def example_function(arg1, arg2):
#     pass
