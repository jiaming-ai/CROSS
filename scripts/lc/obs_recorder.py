"""Record every observation of a running CROSS system (offline back-end studies).

Non-invasive hooks on a `System` instance write one JSON line per observation step:
    step        System._processed_frame_num of the observation (1 = first frame of the session)
    retrieved   ids of the retrieved keyframes that entered the forward pass (after max_refs), with their scores
    raw         the estimator's output for those references: valid mask, confidence, relative pose T_ref_cur
                (7: x y z qx qy qz qw) for the valid ones, the metric-scale correction it applied, and the metric
                camera-to-world poses of all views of the pass (feed-forward estimator: index 0 = current view)
    kept        references that survived the consistency tests, with the prior verdict, chi2, loop flag and std
and, at the end (`close`), the creation step of every keyframe and the final hypothesis-0 pose of every keyframe still
in the graph.  Steps map to frames as frame = map_start + (step - 1) * stride.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _pose7(lie) -> list:
    v = lie.tensor().detach().cpu().numpy().astype(np.float64).reshape(-1)
    v[3:7] /= max(np.linalg.norm(v[3:7]), 1e-12)
    return [round(float(x), 6) for x in v]


def _lst(x, nd=5):
    if x is None:
        return None
    if hasattr(x, "tensor"):
        x = x.tensor()
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return [round(float(v), nd) for v in np.asarray(x, dtype=np.float64).reshape(-1)]


class ObsRecorder:
    def __init__(self, system, path: Path):
        self.sys = system
        self.path = Path(path)
        self.f = open(self.path, "w")
        self.kf_step: dict = {}
        self._retrieved = None
        self._raw = None
        hm = system.hypothesis_manager

        orig_add_node = hm.add_node

        def add_node(keyframe):
            out = orig_add_node(keyframe)
            self.kf_step[int(keyframe.id)] = int(getattr(keyframe, "step_created", -1) or -1)
            return out
        hm.add_node = add_node

        orig_retrieve = system._retrieve_keyframes

        def retrieve(rgb_image):
            res = orig_retrieve(rgb_image)
            self._retrieved = ([int(k.id) for k in res.get("keyframes", [])], [round(float(s), 4) for s in res.get("scores", [])])
            return res
        system._retrieve_keyframes = retrieve

        pe = system.pose_est
        orig_est = pe.estimate_pose

        def estimate_pose(*a, **kw):
            out = orig_est(*a, **kw)
            try:
                poses, masks, conf = out
                info = getattr(pe, "last_info", {}) or {}
                self._raw = {
                    "mask": [bool(m) for m in np.asarray(masks).reshape(-1)],
                    "conf": _lst(conf, 4),
                    "T": [_pose7(p) for p in poses] if poses is not None and len(poses) else [],
                    "scale_corr": float(getattr(pe, "metric_scale_correction", 1.0) or 1.0),
                    "c2w": (np.round(np.asarray(info["c2w_metric"], dtype=np.float64), 5).tolist()
                            if info.get("valid") and "c2w_metric" in info else None),
                    "covis": _lst(info.get("covis"), 4) if info.get("covis") is not None else None,
                }
            except Exception as ex:      # the recorder must never break the run
                self._raw = {"error": str(ex)}
            return out
        pe.estimate_pose = estimate_pose

        orig_cod = system._construct_observation_dist

        def construct_observation_dist(*a, **kw):
            self._retrieved, self._raw = None, None
            ret = orig_cod(*a, **kw)
            try:
                kept = [int(k.id) for k in ret.get("valid_keyframes", [])]
                rec = {"step": int(system._processed_frame_num), "retrieved": self._retrieved, "raw": self._raw,
                       "kept": kept,
                       "h0_ok": ret.get("h0_ok"), "h0_chi2": [None if c is None else round(float(c), 3) for c in (ret.get("h0_chi2") or [])],
                       "h0_loop": [bool(x) for x in (ret.get("h0_loop") or [])],
                       "std": [_lst(s, 5) for s in ret["valid_stds"]] if len(kept) and ret.get("valid_stds") is not None else [],
                       "last_kf": None if system.last_added_kf_id is None else int(system.last_added_kf_id)}
                self.f.write(json.dumps(rec) + "\n")
            except Exception as ex:
                self.f.write(json.dumps({"step": int(system._processed_frame_num), "error": str(ex)}) + "\n")
            return ret
        system._construct_observation_dist = construct_observation_dist

    def close(self):
        hm = self.sys.hypothesis_manager
        final = {int(k): {"pose": _pose7(kf.pose_mu[0]), "temporary": bool(kf.temporary)} for k, kf in hm.nodes.items()}
        self.f.write(json.dumps({"final": final, "kf_step": self.kf_step}) + "\n")
        self.f.close()
