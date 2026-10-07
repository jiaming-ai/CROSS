"""Benchmark rules of 2026-10-07: a pose counts only when it is a pose in the stored map (CROSS session_localized, the
stream systems' first link to the map), and only frames / trials in which the robot moves are evaluated."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "benchmark"))
sys.path.insert(0, str(ROOT / "benchmark" / "eval"))

from cross.core.system import System  # noqa: E402
from metrics import moving_fraction, moving_frames, moving_rule, still_intervals  # noqa: E402
from reloc_metrics import drop_unlocalized, first_map_link, is_localized, map_relative_errors  # noqa: E402


def _session(adjacency, loaded=frozenset({0, 1, 2}), start=10):
    h0 = SimpleNamespace(visual_adjacency=adjacency)
    return SimpleNamespace(loaded_node_ids=frozenset(loaded), _session_start_kf_id=start, _session_localized=False,
                           hypothesis_manager=SimpleNamespace(hypotheses={0: h0}))


def test_session_localized_needs_a_link_from_the_session_to_the_stored_map():
    s = _session({0: {1}, 1: {0, 2}, 10: {11}, 11: {10}})              # session keyframes 10, 11 among themselves
    assert not System.session_localized(s)
    s.hypothesis_manager.hypotheses[0].visual_adjacency[11].add(2)     # a merge brought the edge 11 - 2
    assert System.session_localized(s)
    del s.hypothesis_manager.hypotheses[0].visual_adjacency[11]        # the (temporary) keyframe holding it is removed
    assert System.session_localized(s)                                 # still in the map frame


def test_mapping_session_is_its_own_map():
    assert System.session_localized(_session({}, loaded=frozenset()))


def test_unlocalized_frames_have_no_pose():
    assert is_localized(SimpleNamespace())                             # systems without the notion
    assert not is_localized(SimpleNamespace(localized=lambda: False))
    row = {"c0_pose": list(np.eye(4).reshape(-1)), "c0_t_err": 3.0, "c0_r_err": 1.0, "gt_pose": list(np.eye(4).reshape(-1))}
    drop_unlocalized(row)
    assert row["c0_pose"] is None and row["c0_pose_unlocalized"] is not None and np.isinf(row["c0_t_err"])
    meta = {"kf_gt": {"0": list(np.eye(4).reshape(-1))}, "kf_est": {"0": list(np.eye(4).reshape(-1))}}
    assert np.isinf(map_relative_errors([row], meta, pose_keys=("c0",))[0]["c0_rel_t_err"])


def test_stream_systems_are_localized_from_their_first_link_to_the_map():
    assert first_map_link([], 100) is None
    assert first_map_link([(10, 20), (110, 105)], 100) is None         # map-map and query-query links
    assert first_map_link([(120, 30), (105, 50), (30, 115)], 100) == 105


def test_moving_rule():
    n, fps = 300, 10.0
    G = np.tile(np.eye(4), (n, 1, 1))
    G[100:, 0, 3] = np.arange(n - 100) * 0.05                          # still for 10 s, then 0.5 m/s
    m = moving_frames(G, fps, **moving_rule())
    assert not m[:95].any() and m[105:].all()
    still = still_intervals(m)
    assert still[0][0] == 0 and 94 <= still[0][1] <= 100
    assert moving_fraction(0, 100, still) < 0.1 and moving_fraction(200, 100, still) == 1.0
    assert abs(moving_fraction(50, 100, still) - moving_fraction(50, 100, moving=m)) < 1e-9
    R = np.tile(np.eye(4), (n, 1, 1))                                  # turning on the spot is moving
    for k in range(n):
        a = np.radians(10.0 * k / fps)
        R[k, :3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    assert moving_frames(R, fps, **moving_rule()).all()


def test_rescore_counts_only_moving_trials():
    import make_tables as mt
    ds = {"x": {"thresholds": [1.0, 2.0], "trial_len": 100, "setups": {"stereo": "stereo"}}}
    r = {"track": "t3", "status": "ok", "dataset": "x", "query": "q", "setup": "stereo",
         "trials": [{"start": 0, "final_err": 30.0, "moving": 0.0}, {"start": 50, "final_err": 0.5, "moving": 0.6},
                    {"start": 100, "final_err": 3.0, "moving": 1.0}]}
    out = mt.rescore_t3([r], ds)[0]
    assert out["n_trials"] == 2 and out["n_s2"] == 1
    assert [t["counted"] for t in out["trials"]] == [False, True, True]
