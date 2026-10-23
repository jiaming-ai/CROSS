"""Benchmark results storage (benchmark/results_io.py): metrics in results.json, per-frame data in per-scene files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from results_io import frames_ok, load_results, save_results  # noqa: E402


def cell(track, scene, seq, **kw):
    return {"track": track, "dataset": "kitti", "scene": scene, "system": "cross_mono", "setup": "mono", "seed": 0,
            "sequence": seq, "status": "ok", "ate_rmse": 1.0, "align": "se3", "scale": 1.0, **kw}


def test_split_keeps_the_metrics_and_restores_the_frames(tmp_path):
    traj = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 1.0]]
    rows = [cell("t1", "00", "00", traj_est=[[2 * x for x in p] for p in traj], traj_gt=traj),
            cell("t2", "05", None, query="05b", err_curve=[0.1, -1.0], trials=[{"start": 0}])]
    full = json.loads(json.dumps(rows))
    res, fr = tmp_path / "results.json", tmp_path / "frames"
    save_results(rows, res, frames_dir=fr, lock=tmp_path / "frames.json")
    slim = json.loads(res.read_text())["results"]
    assert all("traj_est" not in r and "err_curve" not in r for r in slim)
    assert slim[1]["trials"] == [{"start": 0}]                        # metric input: stays in results.json
    assert abs(slim[0]["scale_sim3"] - 0.5) < 1e-9                    # CROSS mono: fitted from the paired points
    back = load_results(res, frames=True, lock=tmp_path / "frames.json", frames_dir=fr)
    assert [{k: v for k, v in r.items() if k != "scale_sim3"} for r in back] == full


def test_a_rerun_without_frames_keeps_the_stored_ones_and_drops_removed_cells(tmp_path):
    res, fr, lock = tmp_path / "results.json", tmp_path / "frames", tmp_path / "frames.json"
    save_results([cell("t1", "00", "00", traj_gt=[[0, 0, 0]]), cell("t1", "00", "00b", traj_gt=[[1, 1, 1]])],
                 res, frames_dir=fr, lock=lock)
    save_results([cell("t1", "00", "00", ate_rmse=2.0)], res, frames_dir=fr, lock=lock)
    back = load_results(res, frames=True, lock=lock, frames_dir=fr)
    assert len(back) == 1 and back[0]["ate_rmse"] == 2.0 and back[0]["traj_gt"] == [[0, 0, 0]]
    assert frames_ok(lock, fr)                                        # no lock written outside the repo's folder
