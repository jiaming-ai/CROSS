"""T1 completeness rule of the benchmark (benchmark/eval/metrics.py, PROTOCOL.md section 4)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark" / "eval"))
from metrics import ate, completeness  # noqa: E402


def straight(n, speed, fps=10.0):
    """Ground truth of a straight drive along x at `speed` m/s."""
    gt = np.tile(np.eye(4), (n, 1, 1))
    gt[:, 0, 3] = np.arange(n) * speed / fps
    return gt


def loop(n_per_lap, laps, radius=5.0):
    """Ground truth of `laps` laps around a circle."""
    a = np.linspace(0, 2 * np.pi * laps, n_per_lap * laps, endpoint=False)
    gt = np.tile(np.eye(4), (len(a), 1, 1))
    gt[:, 0, 3], gt[:, 1, 3] = radius * np.cos(a), radius * np.sin(a)
    return gt


def test_slow_robot_sparse_keyframes_count():
    gt = straight(1000, speed=0.1)                   # 1 cm per frame: a keyframe every 5 s is 0.5 m apart
    assert completeness(range(0, 1000, 50), gt, path_m=1.0) == 1.0
    assert completeness(range(0, 1000, 50), gt, path_m=0.0) < 0.5      # the 1 s window alone fails it


def test_lost_tracking_in_new_ground_is_flagged():
    gt = straight(1000, speed=1.0)
    c = completeness(range(0, 400), gt, path_m=1.0)  # no pose after frame 399
    assert 0.39 < c < 0.42


def test_lost_second_lap_is_flagged_although_its_places_were_mapped():
    gt = loop(500, laps=2)                           # the second lap retraces the first
    c = completeness(range(0, 500), gt, path_m=1.0)  # tracked lap 1 only
    assert c < 0.6


def test_time_window_covers_fast_motion():
    gt = straight(200, speed=20.0)                   # 2 m per frame
    assert completeness(range(0, 200, 10), gt, path_m=1.0) == 1.0     # within 1 s of a keyframe


def test_ate_uses_cover_frames_for_completeness_only():
    gt = loop(500, laps=2)
    est = {i: gt[i] for i in range(0, 500, 5)}       # permanent keyframes of lap 1
    r = ate(est, gt, path_m=1.0)
    assert r["failed"]
    r2 = ate(est, gt, path_m=1.0, cover_frames=range(500, 1000, 5))   # revisit poses held by the system
    assert not r2["failed"] and r2["ate_rmse"] == r["ate_rmse"] and r2["n_poses"] == r["n_poses"]
