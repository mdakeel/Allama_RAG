import time
import functools
from core.logging import logger

def retry(times: int = 3, delay: float = 1.0):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning(f"Retry {attempt+1}/{times} failed: {e}")
                    time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator
