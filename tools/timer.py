"""
    tools / timer.py
    ----------------
    Timer instance to measure the time based performance of various methods, 
    functions or operations. This is used internally mainly to testing the 
    comparison between alternative code.
"""
from __future__ import annotations
from contextlib import ContextDecorator

import time



class Timer(ContextDecorator):
    """
    A lightweight high-resolution execution timer, under which block is being 
    timed from initialization of the block and clock halts at end of the block.
    """
    def __init__(self,
        name: str | None = None, show: bool = False, auto_format: bool = True
    ):
        """
            Initialize Timer Instance
        """
        self.clock = time.perf_counter
        self.name, self.show, self.auto_format = name, show, auto_format

        self.time_src: float | None = None
        self.time_dst: float | None = None
        self._elapsed: float | None = None
    

    def __enter__(self):
        self.time_src = self.clock()
        return self
    

    def __exit__(self, *args):
        self._elapsed = self.clock() - self.time_src
        if self.show:
            print(self.result())
    

    def __str__(self) -> str:
        return self.result()
    

    def __repr__(self) -> str:
        name = f" name={self.name!r}" if self.name else ""
        elapsed = f" elapsed={self.elapsed:.6f}s" if self.elapsed is not None else ""
        return f"Timer({name}{elapsed})"
    

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self._elapsed or 0.0


    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000


    @property
    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        return self.elapsed * 1_000_000


    def result(self) -> str:
        """Format the result with auto-formatting based on duration."""
        if self._elapsed is None:
            return "Timer not started or not stopped"

        if self.auto_format:
            if self._elapsed >= 1.0:
                time_str = f"{self._elapsed:.3f} s"

            elif self._elapsed >= 1e-3:
                time_str = f"{self.elapsed_ms:.3f} ms"
            
            else:
                time_str = f"{self.elapsed_us:.3f} µs"
        
        else:
            time_str = f"{self._elapsed:.6f} s"

        prefix = f"{self.name}: " if self.name else ""
        return f"{prefix}{time_str}"


    def reset(self):
        """Reset the timer for reuse."""
        self.time_src = None
        self._elapsed = None
        return self


    def restart(self):
        """Reset and restart the timer."""
        self.reset()
        return self.__enter__()



def timer(name: Optional[str] = None, show: bool = True) -> Timer:
    """Convenience function to create a timer."""
    return Timer(name=name, show=show)



def time_it(func):
    """Decorator to time a function with automatic result printing."""
    @wraps(func)
    def wrapper(*args, **kwargs):

        with Timer(name=func.__name__, show=True) as t:
            result = func(*args, **kwargs)

        return result
    return wrapper
