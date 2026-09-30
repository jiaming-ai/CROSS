"""Paced RGB acquisition/preprocessing on a bounded CPU worker.

Images become eligible at their simulated capture time, including decoding
and undistortion in capture-to-pose latency. Overflow fails explicitly; it
never silently drops benchmark inputs or builds an unbounded backlog.
"""

from collections import deque
from math import isfinite
from threading import Condition, Event, Thread
from time import perf_counter


class PacedRGBStream:
    def __init__(self, sequence, fps, start_time, capacity=2, capture_offsets=None):
        if not isfinite(fps) or fps <= 0 or capacity < 1:
            raise ValueError("Positive frame rate and buffer capacity required")
        self.sequence, self.fps, self.start_time, self.capacity = sequence, fps, start_time, capacity
        self.capture_offsets = None if capture_offsets is None else tuple(capture_offsets)
        if self.capture_offsets is not None:
            if (len(self.capture_offsets) != len(sequence) or
                    any(not isfinite(t) or t < 0 for t in self.capture_offsets) or
                    any(b <= a for a, b in zip(self.capture_offsets, self.capture_offsets[1:]))):
                raise ValueError("Capture offsets must match the sequence and increase from a nonnegative time")
        self.items = deque()
        self.condition = Condition()
        self.stop = Event()
        self.error = None
        self.ended = False
        self.maximum_queued = 0
        self.thread = Thread(target=self._run, name="cross-rgb-input", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            iterator = iter(self.sequence)
            for index in range(len(self.sequence)):
                offset = index/self.fps if self.capture_offsets is None else self.capture_offsets[index]
                arrival = self.start_time + offset
                if self.stop.wait(max(0., arrival-perf_counter())):
                    return
                tick = perf_counter()
                frame = next(iterator)
                preprocessing = perf_counter()-tick
                with self.condition:
                    if len(self.items) >= self.capacity:
                        raise RuntimeError(f"Paced RGB input overflow at frame {index}; no frames silently dropped")
                    self.items.append((frame, arrival, preprocessing))
                    self.maximum_queued = max(self.maximum_queued, len(self.items))
                    self.condition.notify()
        except BaseException as error:
            with self.condition:
                self.error = error
        finally:
            with self.condition:
                self.ended = True
                self.condition.notify()

    def __iter__(self):
        return self

    def __next__(self):
        with self.condition:
            self.condition.wait_for(lambda: self.items or self.ended)
            if self.error is not None:
                raise self.error
            if not self.items:
                raise StopIteration
            return self.items.popleft()

    def close(self):
        self.stop.set()
        self.thread.join()
