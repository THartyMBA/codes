"""
Robust error handling and logging utilities.
Provides standardized error management across applications.
"""

import logging
import traceback
from typing import Any, Optional, Callable
from functools import wraps

class ErrorHandler:
    """
    Centralized error handling with logging and custom responses.
    
    Examples:
        >>> handler = ErrorHandler(log_file="errors.log")
        >>> with handler.catch_errors("database-operation"):
        >>>     perform_db_query()
    """
    
    def __init__(self, log_file: str = "errors.log"):
        self.logger = self._setup_logger(log_file)
    
    def _setup_logger(self, log_file: str) -> logging.Logger:
        logger = logging.getLogger("error_handler")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def handle_error(self, error: Exception, context: str) -> None:
        """
        Process and log an error with context.
        
        Args:
            error: The caught exception
            context: Description of where/when error occurred
        """
        error_details = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context
        }
        self.logger.error(f"Error in {context}: {error_details}")