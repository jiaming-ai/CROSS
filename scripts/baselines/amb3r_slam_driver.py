#!/usr/bin/env python3
"""Run AMB3R-SLAM on one frame stream for scripts/baselines/run_amb3r_slam.py (runs in AMB3R-SLAM's own environment).

    <AMB3R_PY> amb3r_slam_driver.py <AMB3R_DIR> <manifest.json> <out.npz>

The manifest lists the stream's frames (colour or left image, and the right image or the depth image of each frame),
the sensor (mono, stereo, rgbd), the camera (K at the files' resolution, stereo baseline, depth scale) and the depth
range of the sensor.  The driver loads the frames at the model's input size by the upstream `demo` rule (518 px on the
longer side, the shorter by aspect on the patch grid), attaches the sensor's depth source (stereo: upstream
StereoDepth, SGBM on the full-resolution rectified pair; rgbd: the depth images, loaded on demand), gives the
intrinsics (upstream --calib) and runs AMB3R_SLAM.run with the upstream configuration (configs/base.yaml and the
sensor's layer) and models (DA3, bf16 encoder weights).  Upstream code is not modified: accepted loop closures are
logged by wrapping HierarchicalBackend._verify_loop, one line per loop
`AMB3R_LOOP <probe frame> <matched frame> <ka> <kb> <kb first frame> <kb end>` (ka: the matched frame's submap, kb:
the probe's).

Writes out.npz: poses (per frame, camera-to-world, final after the pose graph), online (the front end's online
trajectory), keyframes, loops (rows: probe frame, matched frame, ka, kb, kb start, kb end, ka start, ka end), and
info (JSON: frames, resolution, K, depth range, seconds, fps, loops, loop rejections, peak GPU memory).
Python 3.9 (the upstream environment).
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def model_resolution(h0, w0, width=518, patch=14):
    """Upstream datasets/demo.py: the model's width along the longer side, the other side by aspect on the patch grid."""
    long_ = width
    short = max(2 * patch, int(round(width * min(h0, w0) / max(h0, w0) / patch)) * patch)
    return (long_, short) if w0 >= h0 else (short, long_)


def main():
    amb3r_dir, manifest, out_path = sys.argv[1:4]
    sys.path.insert(0, amb3r_dir)
    os.chdir(amb3r_dir)
    import cv2
    import torch

    from amb3r_slam import backend as be
    from amb3r_slam.datasets.common import _LazyBatch, _LazyFrames
    from amb3r_slam.metric.depth import DepthSource, StereoDepth
    from amb3r_slam.model import load_model
    from amb3r_slam.pipeline import AMB3R_SLAM

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cv2.setNumThreads(2)
    m = json.loads(open(manifest).read())
    files, sensor = m["images"], m["sensor"]
    n = len(files)
    h0, w0 = cv2.imread(files[0]).shape[:2]
    W, H = model_resolution(h0, w0)
    imgs = np.empty((n, 3, H, W), dtype=np.uint8)

    def load(i):
        im = cv2.imread(files[i])
        if im is None:
            raise RuntimeError(f"cannot read {files[i]}")
        imgs[i] = cv2.resize(im[:, :, ::-1], (W, H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)

    t_load = time.time()
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(load, range(n)))
    t_load = time.time() - t_load
    K = np.asarray(m["K"], dtype=np.float64)
    Km = K.copy()
    Km[0] *= W / float(m["width"])
    Km[1] *= H / float(m["height"])

    class LazyDepth(DepthSource):
        """The sensor's depth images (uint16, depth_scale units per metre) at the model's input size, loaded on demand
        (as upstream ArrayDepth, which holds the whole stack)."""

        def __init__(self, paths, scale, min_depth, max_depth, cache=512):
            self.paths, self.scale = paths, float(scale)
            self.min_depth, self.max_depth = float(min_depth), float(max_depth)
            self._cache, self._cap = {}, int(cache)

        def _one(self, f):
            if f in self._cache:
                return self._cache[f]
            d = cv2.imread(self.paths[f], cv2.IMREAD_UNCHANGED)
            d = np.zeros((H, W), np.float32) if d is None else \
                cv2.resize(d.astype(np.float32) / self.scale, (W, H), interpolation=cv2.INTER_NEAREST)
            if self.min_depth > 0:
                d[d < self.min_depth] = 0.0
            if self.max_depth > 0:
                d[d > self.max_depth] = 0.0
            if len(self._cache) >= self._cap:
                self._cache.pop(next(iter(self._cache)))
            self._cache[f] = d
            return d

        def frame_depth(self, frames):
            out = np.zeros((len(frames), H, W), dtype=np.float32)
            for i, f in enumerate(frames):
                if 0 <= int(f) < len(self.paths):
                    out[i] = self._one(int(f))
            return torch.from_numpy(out)

    loops = []
    verify = be.HierarchicalBackend._verify_loop

    def logged_verify(self, imgs_, fj, fi):
        edge = verify(self, imgs_, fj, fi)
        if edge is not None:
            ka, kb = int(edge[0]), int(edge[1])
            sm = self._st.submaps
            row = [int(fj), int(fi), ka, kb, int(sm[kb].start), int(sm[kb].end), int(sm[ka].start), int(sm[ka].end)]
            loops.append(row)
            print("AMB3R_LOOP " + " ".join(str(v) for v in row[:6]), flush=True)
        return edge

    be.HierarchicalBackend._verify_loop = logged_verify

    model = load_model("da3", None, "cuda", fp32=False).eval()
    pipe = AMB3R_SLAM(model, modality=sensor, fps=float(m.get("fps", 10.0)), device="cuda")
    pipe.reset(calib_K=torch.from_numpy(Km).float())
    depth_range = None
    if sensor == "stereo":
        lo, hi = m["stereo_depth"]
        pipe.backend.depth_source = StereoDepth(files, m["right"], float(K[0, 0]), float(m["baseline"]), (W, H),
                                                max_depth=float(hi), min_depth=float(lo))
        depth_range = [float(lo), float(hi)]
    elif sensor == "rgbd":
        mdc = pipe.cfg.backend.metric_depth
        pipe.backend.depth_source = LazyDepth(m["depth"], m["depth_scale"], float(mdc.min_depth), float(mdc.max_depth))
        depth_range = [float(mdc.min_depth), float(mdc.max_depth)]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = pipe.run(_LazyBatch(_LazyFrames(imgs)), verbose=bool(m.get("verbose")))
    dt = time.time() - t0
    poses = res.poses[:n].double().numpy()
    online = res.frontend_poses[:n].double().numpy() if res.frontend_poses is not None else np.zeros((0, 4, 4))
    st = pipe.stats
    info = {"frames": n, "resolution": [W, H], "source_size": [w0, h0], "K_model": Km.tolist(), "sensor": sensor,
            "depth_range": depth_range, "seconds": dt, "load_seconds": t_load, "fps": n / max(dt, 1e-9),
            "loops": int(st.get("n_loops", 0)), "loop_reject_reasons": st.get("loop_reject_reasons", {}),
            "keyframes": len(res.keyframes), "submaps": len(res.submaps),
            "metric_submaps": int(sum(1 for sm in res.submaps if getattr(sm, "metric", False))),
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
            "config": {"size": int(pipe.cfg.backend.size), "step": int(pipe.cfg.backend.step),
                       "frame_stride": int(pipe.cfg.backend.frame_stride), "retrieval": str(pipe.cfg.retrieval.method)}}
    np.savez_compressed(out_path, poses=poses, online=online, keyframes=np.asarray(res.keyframes, dtype=np.int64),
                        loops=np.asarray(loops, dtype=np.int64).reshape(-1, 8), info=json.dumps(info))
    print("AMB3R_INFO " + json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
