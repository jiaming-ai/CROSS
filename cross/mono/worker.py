"""Bounded background work, with explicit overload and error reporting."""

from collections import deque
from threading import Condition, Thread
from time import perf_counter


class LatestWorker:
    """One running item and one replaceable pending item; no growing backlog.

    Replacing a pending image is allowed for low-rate inference, never for
    motion accumulation. Callers attach cumulative motion to each snapshot.
    """

    def __init__(self, function, name):
        self.function, self.name = function, name
        self.condition = Condition()
        self.pending = None
        self.results = deque(maxlen=2)
        self.error = None
        self.closed = False
        self.submitted = self.replaced = self.completed = self.unread_replaced = 0
        self.thread = Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _check(self):
        if self.error is not None:
            raise RuntimeError(f"{self.name} failed") from self.error

    def submit(self, value):
        with self.condition:
            self._check()
            if self.closed:
                raise RuntimeError(f"{self.name} is closed")
            self.replaced += self.pending is not None
            self.pending = (value, perf_counter())
            self.submitted += 1
            self.condition.notify()

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.closed)
                    if self.pending is None:
                        return
                    value, submitted_at = self.pending
                    self.pending = None
                started_at = perf_counter()
                result = self.function(value)
                finished_at = perf_counter()
                with self.condition:
                    self.unread_replaced += len(self.results) == self.results.maxlen
                    self.results.append((result, dict(queue_seconds=started_at-submitted_at,
                                                      service_seconds=finished_at-started_at,
                                                      turnaround_seconds=finished_at-submitted_at)))
                    self.completed += 1
        except BaseException as error:
            with self.condition:
                self.error = error

    def poll(self):
        with self.condition:
            self._check()
            results = list(self.results)
            self.results.clear()
            return results

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()
        self.thread.join()
        self._check()

    def statistics(self):
        with self.condition:
            return dict(submitted=self.submitted, replaced_before_execution=self.replaced,
                        completed=self.completed, unread_results_replaced=self.unread_replaced)

