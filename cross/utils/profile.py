import time
import os
import threading
from functools import wraps
from collections import defaultdict
from contextlib import contextmanager
from loguru import logger

# Profiling configuration
PROFILING_ENABLED = True
WARN_ON_RECURSION = True # Set to True for debugging circular calls

# Modified registry initialization
def create_timing_dict():
    return {
        'total_time': 0,
        'call_count': 0,
        'children': defaultdict(create_timing_dict)
    }

timing_registry = defaultdict(create_timing_dict)
_registry_lock = threading.Lock()
_thread_local = threading.local()

def _get_current_stack():
    """Get the thread-local call stack."""
    if not hasattr(_thread_local, 'stack'):
        _thread_local.stack = []
    return _thread_local.stack

@contextmanager
def timeblock(block_name):
    global timing_registry

    # Early exit if profiling is disabled
    if not PROFILING_ENABLED:
        yield
        return

    _current_stack = _get_current_stack()

    # Detect circular calls BEFORE appending
    is_circular = block_name in _current_stack
    if is_circular and WARN_ON_RECURSION:
        logger.warning(f"Circular call detected! {' -> '.join(_current_stack)} -> {block_name}")

    parent = _current_stack[-1] if _current_stack else None
    _current_stack.append(block_name)

    start_time = time.time()
    yield
    end_time = time.time()
    execution_time = end_time - start_time

    # Update timing registry with thread-safe atomic operations
    with _registry_lock:
        timing_registry[block_name]['total_time'] += execution_time
        timing_registry[block_name]['call_count'] += 1

        # Update parent's children ONLY if not circular
        # Skip if this creates a circular call (block_name already in stack above parent)
        if parent and not is_circular:
            timing_registry[parent]['children'][block_name]['total_time'] += execution_time
            timing_registry[parent]['children'][block_name]['call_count'] += 1

    _current_stack.pop()

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global timing_registry

        # Early exit if profiling is disabled
        if not PROFILING_ENABLED:
            return func(*args, **kwargs)

        _current_stack = _get_current_stack()
        func_name = func.__name__

        # Detect circular calls BEFORE appending
        is_circular = func_name in _current_stack
        if is_circular and WARN_ON_RECURSION:
            logger.warning(f"Circular call detected! {' -> '.join(_current_stack)} -> {func_name}")

        parent = _current_stack[-1] if _current_stack else None
        _current_stack.append(func_name)

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Update timing registry with thread-safe atomic operations
        with _registry_lock:
            timing_registry[func_name]['total_time'] += execution_time
            timing_registry[func_name]['call_count'] += 1

            # Update parent's children ONLY if not circular
            # Skip if this creates a circular call (func_name already in stack above parent)
            if parent and not is_circular:
                timing_registry[parent]['children'][func_name]['total_time'] += execution_time
                timing_registry[parent]['children'][func_name]['call_count'] += 1

        _current_stack.pop()
        return result
    return wrapper

def print_timing_registry(indent="   ", root=None, output_file=None, max_depth=50):
    # Create output lines
    output = ["\nTiming Statistics:",
              "Function Name\tTotal Time (s)\tCalls\tAvg Time (s)",
              "-" * 70]

    def _deep_copy_timing_dict(d):
        """Recursively deep copy timing dict structure."""
        result = {
            'total_time': d['total_time'],
            'call_count': d['call_count'],
            'children': {}
        }
        # Recursively copy all children
        for child_name, child_dict in d['children'].items():
            result['children'][child_name] = _deep_copy_timing_dict(child_dict)
        return result

    # Take a snapshot of the registry with lock to avoid race conditions during printing
    with _registry_lock:
        registry_snapshot = {}
        for k, v in timing_registry.items():
            registry_snapshot[k] = _deep_copy_timing_dict(v)

    # Debug: print what we captured
    if os.getenv('FATS_PROFILE_DEBUG', '0') == '1':
        logger.debug("=== Registry snapshot ===")
        for func_name, info in registry_snapshot.items():
            logger.debug(f"{func_name}: {info['call_count']} calls, {len(info['children'])} children")
            if info['children']:
                logger.debug(f"  Children: {list(info['children'].keys())}")
        logger.debug("=== END DEBUG ===")

    def print_entry(name, info, depth=0, visited=None):
        if visited is None:
            visited = set()

        # Safety: prevent infinite recursion with max depth
        if depth >= max_depth:
            return

        # Detect cycles
        if name in visited:
            return

        total_time = info['total_time']
        calls = info['call_count']
        avg_time = total_time / calls if calls > 0 else 0

        indented_name = indent * depth + name
        output.append(f"{indented_name:<40}\t{total_time:.4f}\t{calls}\t{avg_time:.4f}")

        # Only use children from the passed info, not from global registry
        # This prevents circular references
        children_info = info['children']

        sorted_children = sorted(
            children_info.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )

        # Add current name to visited set for this branch
        new_visited = visited | {name}

        for child_name, child_info in sorted_children:
            print_entry(child_name, child_info, depth + 1, new_visited)
    
    if root is None:
        top_level_funcs = []
        for func_name, info in registry_snapshot.items():
            is_child = False
            for parent_info in registry_snapshot.values():
                if func_name in parent_info['children']:
                    is_child = True
                    break
            if not is_child:
                top_level_funcs.append((func_name, info))

        sorted_funcs = sorted(
            top_level_funcs,
            key=lambda x: x[1]['total_time'],
            reverse=True
        )
        for func_name, info in sorted_funcs:
            print_entry(func_name, info)
    else:
        print_entry(root, registry_snapshot[root])
    
    # Join all lines
    full_output = '\n'.join(output)
    
    # Either print or write to file
    if output_file:
        with open(output_file, 'w') as f:
            f.write(full_output)
    else:
        logger.info(full_output)

def reset_timing_registry():
    global timing_registry
    with _registry_lock:
        timing_registry.clear()
    # Clear current thread's stack if it exists
    if hasattr(_thread_local, 'stack'):
        _thread_local.stack = []