from threading import Event

import pytest

from cross.mono.worker import LatestWorker


def test_slow_worker_replaces_only_pending_work_and_drains():
    entered, release = Event(), Event()
    processed = []

    def process(value):
        if value == 0:
            entered.set()
            assert release.wait(3)
        processed.append(value)
        return value*2

    worker = LatestWorker(process, "test-slow")
    worker.submit(0)
    assert entered.wait(3)
    for i in range(1, 100):
        worker.submit(i)
    assert worker.statistics()["replaced_before_execution"] == 98
    release.set()
    worker.close()
    assert processed == [0, 99]
    assert [item[0] for item in worker.poll()] == [0, 198]
    assert worker.statistics()["completed"] == 2
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(100)


def test_worker_propagates_failure_instead_of_silent_pose_freeze():
    def fail(_):
        raise ValueError("bad model output")
    worker = LatestWorker(fail, "test-error")
    worker.submit(1)
    with pytest.raises(RuntimeError, match="test-error failed") as raised:
        worker.close()
    assert isinstance(raised.value.__cause__, ValueError)


def test_paced_input_does_not_decode_future_frames_before_capture_time():
    from time import perf_counter
    from cross.mono.stream_input import PacedRGBStream
    start = perf_counter()
    class Sequence:
        def __len__(self):
            return 4
        def __iter__(self):
            for i in range(4):
                assert perf_counter() >= start+i/100
                yield i
    source = PacedRGBStream(Sequence(), 100, start)
    try:
        items = list(source)
        assert [item[0] for item in items] == [0, 1, 2, 3]
        assert all(item[2] >= 0 for item in items)
    finally:
        source.close()


def test_input_overload_is_reported_instead_of_accumulating_latency():
    from time import perf_counter
    from cross.mono.stream_input import PacedRGBStream
    source = PacedRGBStream(list(range(10)), 1000, perf_counter(), capacity=1)
    source.thread.join(timeout=3)
    try:
        with pytest.raises(RuntimeError, match="overflow"):
            next(source)
        assert len(source.items) <= 1
    finally:
        source.close()
