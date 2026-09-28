"""Dump / load the pose graph of a running CROSS system with ground truth (offline loop-closure studies).

A graph file is a JSON document with
    nodes:  [{id, pose (7: x y z qx qy qz qw, hypothesis-0 mean), std (6), temporary, step_created, session, gt (16, c2w
             ground truth, OpenCV convention) or null}]
    odom:   [{a, b, mean (7), std (6), n_frames}]            odometry edges a -> b (T_a_b); n_frames = integrated readings (or null)
    visual: [{a, b, mean (7), std (6), fc, tc, hyp, conf, rw, step, session, informative, noise_scale, rejected?}]
                                                             visual edges (T_a_b = pose of b in a), all hypotheses;
                                                             rejected: quarantined by the posterior test (not in the graph)
    meta:   free-form (scene, session, seed, ...)
Poses are in the map frame of the run; the ground truth is in the dataset world frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _pose7(lie) -> list:
    v = lie.tensor().detach().cpu().numpy().astype(np.float64).reshape(-1)
    v[3:7] /= max(np.linalg.norm(v[3:7]), 1e-12)
    return [round(float(x), 6) for x in v]


def _std6(lie) -> list:
    return [round(float(x), 6) for x in lie.tensor().detach().cpu().numpy().reshape(-1)]


def dump_graph(system, path, kf_gt: dict, session_id: int, meta: dict | None = None) -> dict:
    """Serialize the current graph of `system` (all keyframes, odometry edges, visual edges of every hypothesis).

    kf_gt: keyframe id -> ground-truth c2w (4x4 or 16 floats) for keyframes of this run (any session).
    Keyframes with id < system._session_start_kf_id belong to the loaded map (session 0 of the trace).
    """
    hm = system.hypothesis_manager
    sess_start = int(getattr(system, "_session_start_kf_id", 0))
    nodes = []
    with hm.graph_lock:
        for kid in sorted(hm.nodes.keys()):
            kf = hm.nodes[kid]
            g = kf_gt.get(int(kid))
            nodes.append({
                "id": int(kid), "pose": _pose7(kf.pose_mu[0]), "std": _std6(kf.pose_std[0]),
                "temporary": bool(kf.temporary), "step_created": getattr(kf, "step_created", None),
                "session": 0 if int(kid) < sess_start else int(session_id),
                "gt": None if g is None else [round(float(x), 6) for x in np.asarray(g, dtype=np.float64).reshape(-1)],
            })
        odom = [{"a": int(a), "b": int(b), "mean": _pose7(e.mean), "std": _std6(e.std),
                 "n_frames": getattr(e, "n_frames", None)} for (a, b), e in hm.odom_edges.items()]
        visual = []
        for comp, h in hm.hypotheses.items():
            for (a, b), factors in h.visual_edges.items():
                for f in factors:
                    visual.append({
                        "a": int(a), "b": int(b), "mean": _pose7(f.mean), "std": _std6(f.std),
                        "fc": int(f.from_comp_id), "tc": int(f.to_comp_id), "hyp": int(comp),
                        "conf": getattr(f, "conf", None), "rw": getattr(f, "rw", None),
                        "step": getattr(f, "step", None),
                        "informative": getattr(f, "informative", None), "noise_scale": getattr(f, "noise_scale", None),
                        "scale_corr": getattr(f, "scale_corr", None),
                        "session": getattr(f, "session", 0 if (int(a) < sess_start and int(b) < sess_start) else int(session_id)),
                    })
        v = getattr(system, "_lc_verifier", None)
        for (a, b, f) in (getattr(v, "quarantine", None) or []):
            visual.append({
                "a": int(a), "b": int(b), "mean": _pose7(f.mean), "std": _std6(f.std),
                "fc": int(f.from_comp_id), "tc": int(f.to_comp_id), "hyp": 0, "rejected": True,
                "conf": getattr(f, "conf", None), "rw": getattr(f, "rw", None), "step": getattr(f, "step", None),
                "informative": getattr(f, "informative", None), "noise_scale": getattr(f, "noise_scale", None),
                "session": getattr(f, "session", 0 if (int(a) < sess_start and int(b) < sess_start) else int(session_id)),
            })
    data = {"meta": dict(meta or {}, session=int(session_id), session_start_kf_id=sess_start),
            "nodes": nodes, "odom": odom, "visual": visual}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, separators=(",", ":")))
    return data


def load_graph(path) -> dict:
    return json.loads(Path(path).read_text())
