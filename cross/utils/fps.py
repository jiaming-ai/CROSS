import time
import threading
from collections import defaultdict, deque
from functools import wraps
from typing import Dict, Callable, Optional
from loguru import logger


class FPSRegistry:
    """Central registry for tracking FPS of wrapped functions."""
    
    def __init__(self):
        self._functions: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        self._print_thread: Optional[threading.Thread] = None
        self._print_interval = 5.0  # Default 5 seconds
        self._running = False
        self._last_print_time = time.time()
    
    def register_call(self, func_name: str, call_time: float):
        """Register a function call with its timestamp."""
        try:
            with self._lock:
                self._functions[func_name].append(call_time)
        except Exception as e:
            # Don't let FPS monitoring break the main function
            logger.warning(f"FPS monitoring error for {func_name}: {e}")
    
    def calculate_fps(self, func_name: str, window_seconds: float = 5.0) -> float:
        """Calculate FPS for a function over the given time window."""
        with self._lock:
            if func_name not in self._functions:
                return 0.0
            
            calls = self._functions[func_name]
            if not calls:
                return 0.0
            
            # Copy the deque to avoid holding lock during calculation
            calls_copy = list(calls)
        
        # Calculate FPS without holding the lock
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Count calls within the time window
        recent_calls = [t for t in calls_copy if t >= cutoff_time]
        
        if len(recent_calls) < 2:
            return 0.0
        
        # Calculate FPS over the actual time span of recent calls
        time_span = recent_calls[-1] - recent_calls[0]
        if time_span <= 0:
            return 0.0
        
        return (len(recent_calls) - 1) / time_span
    
    def get_all_fps(self, window_seconds: float = 5.0) -> Dict[str, float]:
        """Get FPS for all registered functions."""
        # Get function names without holding lock during calculation
        with self._lock:
            func_names = list(self._functions.keys())
        
        # Calculate FPS for each function (this will acquire locks individually)
        return {
            func_name: self.calculate_fps(func_name, window_seconds)
            for func_name in func_names
        }
    
    def print_fps_stats(self, window_seconds: float = 5.0):
        """Print FPS statistics for all functions."""
        fps_stats = self.get_all_fps(window_seconds)
        
        if not fps_stats:
            return
        
        # Filter out functions with 0 FPS (no recent calls)
        active_stats = {name: fps for name, fps in fps_stats.items() if fps > 0}
        
        if not active_stats:
            return
        
        lines = [
            "",
            "=" * 60,
            f"FPS Statistics (last {window_seconds}s)",
            "=" * 60,
        ]

        # Sort by FPS descending
        sorted_stats = sorted(active_stats.items(), key=lambda x: x[1], reverse=True)

        for func_name, fps in sorted_stats:
            lines.append(f"{func_name:40} : {fps:8.2f} FPS")

        lines.append("=" * 60)
        logger.info("\n".join(lines))
    
    def start_periodic_printing(self, interval: float = 5.0):
        """Start periodic printing of FPS statistics."""
        if self._running:
            return
        
        self._print_interval = interval
        self._running = True
        self._print_thread = threading.Thread(target=self._print_loop, daemon=True)
        self._print_thread.start()
        logger.info(f"Started FPS monitoring with {interval}s interval")
    
    def stop_periodic_printing(self):
        """Stop periodic printing of FPS statistics."""
        self._running = False
        if self._print_thread and self._print_thread.is_alive():
            self._print_thread.join(timeout=1.0)
        logger.info("Stopped FPS monitoring")
    
    def _print_loop(self):
        """Main loop for periodic printing."""
        while self._running:
            time.sleep(self._print_interval)
            if self._running:  # Check again after sleep
                self.print_fps_stats(self._print_interval)
    
    def clear_stats(self):
        """Clear all stored statistics."""
        with self._lock:
            self._functions.clear()
        logger.info("Cleared FPS statistics")


# Global registry instance
_global_registry = FPSRegistry()


def fps_monitor(func_name: Optional[str] = None, 
                auto_start: bool = False,  # Changed default to False to avoid blocking
                print_interval: float = 5.0):
    """
    Decorator to monitor FPS of a function.
    
    Args:
        func_name: Custom name for the function (defaults to function.__name__)
        auto_start: Whether to automatically start periodic printing
        print_interval: Interval in seconds for periodic printing
    
    Usage:
        @fps_monitor()
        def my_function():
            pass
        
        @fps_monitor("custom_name")
        def another_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        nonlocal func_name
        if func_name is None:
            func_name = f"{func.__module__}.{func.__name__}"
        
        # Auto-start periodic printing if requested
        if auto_start and not _global_registry._running:
            _global_registry.start_periodic_printing(print_interval)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get timestamp as late as possible and register asynchronously
            try:
                result = func(*args, **kwargs)
                # Register call after function execution to avoid blocking
                call_time = time.time()
                _global_registry.register_call(func_name, call_time)
                return result
            except Exception:
                # Still register the call even if function fails
                call_time = time.time()
                _global_registry.register_call(func_name, call_time)
                raise
        
        return wrapper
    
    return decorator


def get_fps_stats(window_seconds: float = 5.0) -> Dict[str, float]:
    """Get current FPS statistics for all monitored functions."""
    return _global_registry.get_all_fps(window_seconds)


def print_fps_stats(window_seconds: float = 5.0):
    """Print current FPS statistics for all monitored functions."""
    _global_registry.print_fps_stats(window_seconds)


def start_fps_monitoring(interval: float = 5.0):
    """Start periodic FPS monitoring and printing."""
    _global_registry.start_periodic_printing(interval)


def stop_fps_monitoring():
    """Stop periodic FPS monitoring and printing."""
    _global_registry.stop_periodic_printing()


def clear_fps_stats():
    """Clear all FPS statistics."""
    _global_registry.clear_stats()


# Context manager for temporary FPS monitoring
class fps_monitoring:
    """Context manager for temporary FPS monitoring."""
    
    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.was_running = False
    
    def __enter__(self):
        self.was_running = _global_registry._running
        if not self.was_running:
            start_fps_monitoring(self.interval)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.was_running:
            stop_fps_monitoring()


if __name__ == "__main__":
    # Example usage
    import random
    
    @fps_monitor("fast_function")
    def fast_func():
        time.sleep(0.01)  # 10ms
    
    @fps_monitor("slow_function")
    def slow_func():
        time.sleep(0.1)   # 100ms
    
    @fps_monitor()  # Uses default name
    def variable_func():
        time.sleep(random.uniform(0.01, 0.05))
    
    print("Running FPS monitoring example...")
    print("Functions will be called for 15 seconds with FPS printed every 5s")
    
    # Run functions for demonstration
    start_time = time.time()
    while time.time() - start_time < 15:
        fast_func()
        if random.random() < 0.3:  # 30% chance
            slow_func()
        if random.random() < 0.5:  # 50% chance
            variable_func()
        time.sleep(0.005)  # Small delay between calls
    
    # Final stats
    print("\nFinal statistics:")
    print_fps_stats()
    
    stop_fps_monitoring()