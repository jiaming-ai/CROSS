"""Bounded CPU acquisition in a spawned process, independent of CUDA workers.

Only timestamp metadata is available before capture. Decoding starts after
each original or simulated arrival, and every selected image must be read.
Overflow fails the run; the implementation never substitutes or drops RGB.
"""

from math import isfinite
import multiprocessing as mp
from queue import Empty, Full
from time import perf_counter


def _read_images(sequence, fps, offsets, frames, control, stop, peak):
    import cv2
    cv2.setNumThreads(1)
    normal_end = False
    try:
        control.send(("ready", None))
        command, started = control.recv()
        if command != "start":
            return
        iterator = iter(sequence)
        for index in range(len(sequence)):
            arrival = started + (index/fps if offsets is None else offsets[index])
            if stop.wait(max(0., arrival-perf_counter())):
                return
            before = perf_counter()
            frame = next(iterator)
            preprocessing = perf_counter()-before
            try:
                frames.put_nowait(("frame", (frame, arrival, preprocessing)))
            except Full as error:
                raise RuntimeError(f"Paced RGB input overflow at frame {index}; no frames silently dropped") from error
            # qsize is a sampled diagnostic; Queue's semaphore enforces the
            # actual bound even if the consumer runs before this sample.
            peak.value = max(peak.value, frames.qsize())
        while not stop.is_set():
            try:
                frames.put(("end", None), timeout=.05)
                normal_end = True
                return
            except Full:
                pass
    except BaseException as error:
        try:
            control.send(("error", f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, EOFError, OSError):
            pass  # the owner may already be shutting down
    finally:
        if not normal_end:
            # On overflow/early close, no consumer may remain to drain the
            # feeder. Do not hang shutdown on an undeliverable RGB payload.
            frames.cancel_join_thread()
        frames.close()
        if normal_end:
            frames.join_thread()  # preserve the final frame and end marker
        control.close()


class PacedRGBProcess:
    """Prepare the reader before starting the capture clock explicitly."""

    def __init__(self, sequence, fps, capacity=2, capture_offsets=None):
        if not isfinite(fps) or fps <= 0 or capacity < 1:
            raise ValueError("Positive frame rate and buffer capacity required")
        offsets = None if capture_offsets is None else tuple(capture_offsets)
        if offsets is not None and (len(offsets) != len(sequence) or
                                    any(not isfinite(t) or t < 0 for t in offsets) or
                                    any(b <= a for a, b in zip(offsets, offsets[1:]))):
            raise ValueError("Capture offsets must match the sequence and increase from a nonnegative time")
        context = mp.get_context("spawn")
        self.capacity, self.started, self.closed, self.ended = capacity, False, False, False
        self.frames = context.Queue(maxsize=capacity)
        self.control, child_control = context.Pipe()
        self.stop = context.Event()
        self.peak = context.Value("i", 0, lock=False)
        self.process = context.Process(target=_read_images,
                                       args=(sequence, fps, offsets, self.frames, child_control, self.stop, self.peak),
                                       name="cross-rgb-input", daemon=True)
        before = perf_counter()
        self.process.start()
        child_control.close()
        try:
            if not self.control.poll(30):
                raise TimeoutError("RGB process did not initialize")
            kind, message = self.control.recv()
            if kind != "ready":
                raise RuntimeError(f"RGB process initialization failed: {message}")
        except BaseException:
            self.close()
            raise
        self.startup_seconds = perf_counter()-before

    @property
    def maximum_queued(self):
        return self.peak.value

    def start(self, start_time):
        if self.started or self.closed:
            raise RuntimeError("RGB process can only be started once")
        self.control.send(("start", start_time))
        self.started = True

    def __iter__(self):
        return self

    def _check_error(self):
        if self.control.poll():
            try:
                kind, message = self.control.recv()
            except EOFError:
                return  # the queue carries normal completion
            if kind == "error":
                raise RuntimeError(f"RGB process failed: {message}")

    def __next__(self):
        if self.ended:
            raise StopIteration
        if not self.started:
            raise RuntimeError("Start the RGB capture clock before reading")
        while True:
            self._check_error()
            try:
                kind, value = self.frames.get(timeout=.05)
            except Empty:
                self._check_error()
                if self.process.is_alive():
                    continue
                # Normal exit flushes the queue before terminating. Allow
                # one final poll for an end marker crossing the prior poll.
                try:
                    kind, value = self.frames.get(timeout=.05)
                except Empty as error:
                    raise RuntimeError("RGB process exited without an end marker") from error
            self._check_error()
            if kind == "end":
                self.ended = True
                raise StopIteration
            return value

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        if not self.started:
            try:
                self.control.send(("stop", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        self.frames.cancel_join_thread()
        self.frames.close()
        self.control.close()
