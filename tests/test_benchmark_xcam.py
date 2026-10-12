"""Cross-camera benchmark harness (benchmark/run.py partial maps / map extension, dataset xcam_simchange):
map keys, coverage = union of the sessions' paths split by area, missing folders exit with code 3, per-folder stereo
baselines (scripts/map_and_reloc.py seq_baseline)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "scripts"))


def write_seq(folder: Path, xs, baselines=(0.3,), right_link=True):
    """A SimChange-like folder: poses along x at the given positions, calib.json with its rendered baselines."""
    folder.mkdir(parents=True)
    gt = np.tile(np.eye(4), (len(xs), 1, 1))
    gt[:, 0, 3] = xs
    np.savetxt(folder / "poses_left.txt", gt.reshape(len(xs), 16))
    for b in baselines:
        (folder / f"right_{b:.2f}").mkdir()
    if right_link:
        (folder / "right").symlink_to(f"right_{baselines[0]:.2f}")
    (folder / "calib.json").write_text(json.dumps({"K": np.eye(3).tolist(), "width": 64, "height": 48,
                                                   "baseline": baselines[0], "baselines": list(baselines),
                                                   "right_dirs": {f"{b:.2f}": f"right_{b:.2f}" for b in baselines},
                                                   "T_right_in_left": np.eye(4).tolist(), "fps": 10.0}))


def job(tmp_path, scene, task="query", query="classroom/reverse"):
    import run
    a = argparse.Namespace(dataset="xcam_simchange", scene=scene, system="cross_stereo", setup="stereo", seed=0,
                           task=task, seq=None, query=query, data=str(tmp_path / "data"), out=str(tmp_path / "out"),
                           odom_source="external", map_from="", variant="", args="", reeval=False, force=False,
                           wait_for_map=False, redo=[], only=[], keep_maps=False, keep_rows=False, timeout=60.0)
    return run.Job(a)


@pytest.fixture
def data(tmp_path):
    root = tmp_path / "data" / "simchange" / "classroom"
    write_seq(root / "map", np.linspace(0, 20, 159))                       # camA route: 0..20 m
    write_seq(root / "map__camB", np.linspace(0, 20, 150), baselines=(0.12,))
    write_seq(root / "reverse", np.linspace(25, -5, 140))                  # leaves the route at both ends
    return tmp_path


def test_dataset_entry():
    cfg = yaml.safe_load((ROOT / "benchmark/configs/datasets.yaml").read_text())["xcam_simchange"]
    assert "baseline" not in cfg and cfg["xcam"] and cfg["tier"] == "xcam" and cfg["dev"]
    ext = cfg["scenes"]["classroom_extB"]
    assert ext["map"] == "classroom/map" and ext["extend"] == "classroom/map__camB" and ext["map_end"] == 79
    assert "classroom/map__camB" in cfg["scenes"]["classroom_A"]["queries"]


def test_map_keys(data):
    j = job(data, "classroom_A")
    assert j.map_key == j.base_key == "classroom/map" and j.area_split
    j = job(data, "classroom_extB")
    assert j.base_key == "classroom/map@0-79" and j.map_key == "classroom/map@0-79+map__camB"


def test_areas_union_of_first_session_and_extension(data):
    j = job(data, "classroom_extB")
    x = np.linspace(25, -5, 140)
    area = j.areas("classroom/reverse")
    first_end = np.linspace(0, 20, 159)[78]                                 # last mapped frame of the first session
    assert np.all(area[(x >= 0) & (x <= first_end + 1.9)] == 1)
    assert np.all(area[(x > first_end + 2.1) & (x <= 21.9)] == 2)
    assert np.all(area[(x > 22.1) | (x < -2.1)] == 0)
    assert np.array_equal(j.covered("classroom/reverse"), area > 0)
    # the full camA map covers the extension area itself
    assert set(np.unique(job(data, "classroom_A").areas("classroom/reverse"))) == {0, 1}


def test_missing_folders_exit_3(data):
    with pytest.raises(SystemExit) as e:
        job(data, "classroom_extC").run()                                   # map__camC not rendered
    assert e.value.code == 3
    with pytest.raises(SystemExit) as e:
        job(data, "classroom_A", query="classroom/half__camB").run()
    assert e.value.code == 3


def test_seq_baseline(tmp_path):
    from map_and_reloc import seq_baseline
    write_seq(tmp_path / "a", np.zeros(3), baselines=(0.12, 0.3))
    write_seq(tmp_path / "b", np.zeros(3), baselines=(0.2, 0.3), right_link=False)
    assert seq_baseline(tmp_path / "a", None) is None          # right/ (the first rendered baseline) is loaded
    assert seq_baseline(tmp_path / "b", None) == 0.2            # no right/ link: calib.json's baseline
    assert seq_baseline(tmp_path / "b", 0.3) == 0.3
