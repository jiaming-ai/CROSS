import os
from time import perf_counter

import cv2
import numpy as np
import pytest

from cross.mono.data import RGBSequence
from cross.mono.input_process import PacedRGBProcess


class TimestampSequence:
    def __len__(self):
        return 4

    def __iter__(self):
        for index in range(len(self)):
            yield dict(index=index, read_at=perf_counter(), process=os.getpid())


class BrokenSequence:
    def __len__(self):
        return 1

    def __iter__(self):
        raise ValueError("unreadable camera frame")
        yield  # make failure occur on the first read


class TerminatedSequence:
    def __len__(self):
        return 1

    def __iter__(self):
        os._exit(17)
        yield


def test_process_capture_is_causal_and_end_marker_retains_all_frames():
    offsets = [0., .009, .031, .067]
    reader = PacedRGBProcess(TimestampSequence(), 100, capacity=2, capture_offsets=offsets)
    try:
        with pytest.raises(RuntimeError, match="Start"):
            next(reader)
        started = perf_counter()
        reader.start(started)
        items = list(reader)
        assert [x[0]["index"] for x in items] == list(range(4))
        assert all(x[0]["process"] != os.getpid() for x in items)
        assert all(x[0]["read_at"] >= started+t for x, t in zip(items, offsets))
        assert [x[1] for x in items] == [started+t for t in offsets]
        with pytest.raises(StopIteration):
            next(reader)
    finally:
        reader.close()
    assert reader.process.exitcode == 0


def test_process_preserves_rgb_calibration_timestamps_and_stage_diagnostics(tmp_path):
    times = [1., 1.06, 1.15]
    for i in range(3):
        assert cv2.imwrite(str(tmp_path/f"{i}.png"), np.full((4, 6, 3), [i, 10, 20], np.uint8))
    (tmp_path/"rgb.txt").write_text("\n".join(f"{t} {i}.png" for i, t in enumerate(times)))
    sequence = RGBSequence(tmp_path, calibration=[5, 5, 3, 2])
    reader = PacedRGBProcess(sequence, 20, capture_offsets=[t-times[0] for t in times])
    try:
        reader.start(perf_counter())
        items = list(reader)
        for i, (frame, _, _) in enumerate(items):
            assert frame.timestamp == times[i] and frame.index == i
            np.testing.assert_array_equal(frame.rgb[0, 0], [20, 10, i])
            assert frame.input_timing["opencv_threads"] == 1
            assert frame.input_timing["read_decode_wall_seconds"] >= 0
    finally:
        reader.close()


@pytest.mark.parametrize("sequence, message", [(BrokenSequence(), "unreadable camera frame"),
                                                (TerminatedSequence(), "without an end marker")])
def test_process_reader_reports_decode_failure_or_sudden_exit(sequence, message):
    reader = PacedRGBProcess(sequence, 20)
    try:
        reader.start(perf_counter())
        with pytest.raises(RuntimeError, match=message):
            list(reader)
    finally:
        reader.close()
    assert not reader.process.is_alive()


def test_process_overflow_is_an_error_and_early_close_does_not_hang():
    reader = PacedRGBProcess(list(range(100)), 1000, capacity=1)
    try:
        reader.start(perf_counter())
        reader.process.join(timeout=3)
        assert not reader.process.is_alive()
        with pytest.raises(RuntimeError, match="overflow"):
            next(reader)
        assert reader.maximum_queued <= 1
    finally:
        reader.close()
    waiting = PacedRGBProcess(list(range(100)), 1)
    waiting.close()  # never started the capture clock
    waiting.close()
    assert not waiting.process.is_alive()
