"""run.py: --load-map / --save-map / --save-poses around a session (a stand-in session: no models), the TUM pose file
from the frame on which the session is placed in the map, and the mode's shipped configuration."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from test_dataloaders import N, _posed_folder   # noqa: E402


def _run_module():
    spec = importlib.util.spec_from_file_location("cross_run_cli", ROOT / "run.py")
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run


class FakeSession:
    """Records the calls run.py makes; placed in the map from its third frame on."""

    def __init__(self, calls, config):
        self.calls, self.config, self.n = calls, config, 0
        self.hypothesis_manager = SimpleNamespace(nodes={})

    def load_map(self, path):
        self.calls.append(("load_map", path))

    def save_map(self, path):
        self.calls.append(("save_map", path))

    def process(self, frame):
        self.calls.append(("process", frame["frame_idx"], frame["delta_pose"] is None))
        self.n += 1

    def localized(self):
        return self.n >= 3

    def belief(self, to_mat):
        T = np.eye(4)
        T[0, 3] = float(self.n)
        return T, T, np.ones(1)

    def release(self):
        self.calls.append(("release",))


def test_load_relocalize_save_and_poses(tmp_path, monkeypatch):
    run = _run_module()
    calls, made = [], []

    def factory(args, camera, config, **kw):
        made.append((args, config))
        return lambda: FakeSession(calls, config)

    monkeypatch.setattr(run, "session_factory", factory)
    seq, poses, maps = _posed_folder(tmp_path / "seq"), tmp_path / "poses.txt", tmp_path / "maps"
    monkeypatch.setattr(sys, "argv", ["run.py", str(seq), "--mode", "rgbd_wheel", "--no-viz",
                                      "--load-map", str(maps / "a.pkl"), "--save-map", str(maps / "b" / "map.pkl"),
                                      "--save-poses", str(poses), "--set", "pose_est.obs_max_interval_steps=2"])
    run.main()

    assert calls[0] == ("load_map", str(maps / "a.pkl"))
    assert [c for c in calls if c[0] == "process"] == [("process", i, i == 0) for i in range(N)]
    assert calls[-2:] == [("save_map", str(maps / "b" / "map.pkl")), ("release",)]
    assert (maps / "b").is_dir()                                  # the map's folder is created
    rows = np.loadtxt(poses, ndmin=2)
    assert rows.shape == (N - 2, 8)                               # from the third frame (placed in the map) on
    np.testing.assert_allclose(rows[:, 0], np.arange(2, N) / 5.0)         # timestamps of the posed folder (5 fps)
    np.testing.assert_allclose(rows[:, 1], np.arange(3, N + 1))           # x of the session's pose
    np.testing.assert_allclose(rows[:, 4:], [[0, 0, 0, 1]] * (N - 2))     # identity rotation (qx qy qz qw)
    args, config = made[0]
    assert args.mode == "rgbd" and config.pose_est.obs_max_interval_steps == 2


def test_frames_and_start(tmp_path, monkeypatch):
    run = _run_module()
    calls = []
    monkeypatch.setattr(run, "session_factory", lambda *a, **k: (lambda: FakeSession(calls, None)))
    monkeypatch.setattr(sys, "argv", ["run.py", str(_posed_folder(tmp_path)), "--no-viz", "--start", "1",
                                      "--frames", "2"])
    run.main()
    assert [c for c in calls if c[0] == "process"] == [("process", 1, True), ("process", 2, False)]
    assert calls[-1] == ("release",) and not any(c[0] in ("load_map", "save_map") for c in calls)
