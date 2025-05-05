from time import time


class Timer:
    """Context manager for timing code execution."""
    
    def __enter__(self):
        self.start = time()
        self.duration = None

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time() - self.start
