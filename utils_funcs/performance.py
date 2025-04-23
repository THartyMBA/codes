"""
Performance monitoring and optimization utilities.
"""

import time
import cProfile
import functools
from typing import Callable, Any

class PerformanceMonitor:
    """
    Monitor and profile code execution.
    
    Examples:
        >>> monitor = PerformanceMonitor()
        >>> @monitor.time_this
        >>> def slow_function():
        >>>     time.sleep(1)
    """
    
    def __init__(self):
        self.profiler = cProfile.Profile()
    
    def time_this(self, func: Callable) -> Callable:
        """
        Decorator to measure function execution time.
        
        Args:
            func: Function to be timed
            
        Returns:
            Wrapped function with timing
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
            return result
        return wrapper