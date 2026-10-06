"""Writing the scene cache (format: vggt_ft/dataio/scene.py) and checking it.

Converters (vggt_ft/prep/<dataset>.py) call

    sw = SceneWriter(root, "tartanair2", "OldTown_P000", world="OldTown", metric=True, synthetic=True)
    q = sw.sequence("P000_winter", session="winter")
    q.add(name, rgb_uint8, depth_float, K, E_cam_from_world)      # OpenCV axes, depth = z (not ray length)
    sw.close()

and finally `python -m vggt_ft.prep.common index <root>/<dataset>` and
`python -m vggt_ft.prep.common verify <root>/<dataset>` (overlap of nearby frame pairs: a convention error in the poses
or the depth shows up as near-zero overlap).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from vggt_ft.dataio.depthcodec import encode

cv2.setNumThreads(2)


def _atomic_write_bytes(path: Path, data: bytes):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class SeqWriter:
    def __init__(self, scene: "SceneWriter", name: str, session: str):
        self.scene, self.name, self.session = scene, name, session
        self.dir = scene.dir / name
        (self.dir / "rgb").mkdir(parents=True, exist_ok=True)
        (self.dir / "depth").mkdir(parents=True, exist_ok=True)
        self.K, self.E, self.names, self.t, self.dmed, self.valid = [], [], [], [], [], []

    def add(self, name: str, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, E: np.ndarray, t: float | None = None):
        """rgb HxWx3 uint8 RGB; depth HxW z-depth in scene units (<= 0 / non-finite = invalid), any resolution with the
        same aspect (resized to the image); K 3x3 for the rgb resolution; E 4x4 camera-from-world (OpenCV)."""
        units = self.scene.units
        rgb = np.ascontiguousarray(rgb[..., :3])
        H, W = rgb.shape[:2]
        K = np.asarray(K, np.float64).copy()
        s = min(1.0, self.scene.max_side / max(H, W))
        if s < 1.0:
            nw, nh = int(round(W * s)), int(round(H * s))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            sx, sy = nw / W, nh / H
            K[0, 0], K[1, 1] = K[0, 0] * sx, K[1, 1] * sy
            K[0, 2], K[1, 2] = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
            H, W = nh, nw
        d = np.asarray(depth, np.float32) * units
        d[~np.isfinite(d)] = 0
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        E = np.asarray(E, np.float64).copy()
        E[:3, 3] *= units
        R = E[:3, :3]
        if abs(np.linalg.det(R) - 1) > 1e-3 or np.abs(R @ R.T - np.eye(3)).max() > 1e-3:
            raise ValueError(f"{self.dir} {name}: not a rotation (det {np.linalg.det(R):.4f})")
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        _atomic_write_bytes(self.dir / "rgb" / f"{name}.jpg", jpg.tobytes())
        ok, png = cv2.imencode(".png", encode(d))
        _atomic_write_bytes(self.dir / "depth" / f"{name}.png", png.tobytes())
        v = d[d > 0]
        self.K.append(K)
        self.E.append(E)
        self.names.append(name)
        self.t.append(np.nan if t is None else float(t))
        self.dmed.append(float(np.median(v)) if v.size else 0.0)
        self.valid.append(v.size / d.size)

    def close(self) -> int:
        if not self.names:
            return 0
        tmp = self.dir / "frames.tmp.npz"
        np.savez(tmp, K=np.array(self.K, np.float32), E=np.array(self.E, np.float64), names=np.array(self.names),
                 t=np.array(self.t, np.float64), dmed=np.array(self.dmed, np.float32),
                 valid=np.array(self.valid, np.float32))
        os.replace(tmp, self.dir / "frames.npz")
        return len(self.names)


class SceneWriter:
    def __init__(self, root, dataset: str, scene: str, world: str | None = None, metric: bool = True,
                 synthetic: bool = False, dynamic: bool = False, kind: str = "", units: float = 1.0,
                 max_side: int = 768, extra: dict | None = None):
        """units: factor applied to depth and translations (non-metric data: bring the median depth to ~2)."""
        self.dir = Path(root) / dataset / scene
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(dataset=dataset, scene=scene, world=world or scene, metric=metric, synthetic=synthetic,
                         dynamic=dynamic, kind=kind, units=units, **(extra or {}))
        self.units, self.max_side = units, max_side
        self.seqs: list[SeqWriter] = []

    def sequence(self, name: str, session: str | None = None) -> SeqWriter:
        q = SeqWriter(self, name, session or name)
        self.seqs.append(q)
        return q

    def close(self) -> int:
        seqs = {}
        for q in self.seqs:
            n = q.close()
            if n >= 2:
                seqs[q.name] = {"session": q.session, "n": n}
        if not seqs:
            return 0
        self.meta["sequences"] = seqs
        _atomic_write_bytes(self.dir / "scene.json", json.dumps(self.meta, indent=1).encode())
        return sum(v["n"] for v in seqs.values())


def units_for(depths: list[np.ndarray], target: float = 2.0) -> float:
    """Factor that brings the median valid depth of a non-metric scene to `target`."""
    v = np.concatenate([d[(d > 0) & np.isfinite(d)].ravel()[::97] for d in depths])
    return float(target / np.median(v)) if v.size else 1.0


def write_index(dataset_dir: Path):
    dataset_dir = Path(dataset_dir)
    names = sorted(p.name for p in dataset_dir.iterdir() if (p / "scene.json").exists())
    _atomic_write_bytes(dataset_dir / "index.json", json.dumps(names).encode())
    return names


def verify(dataset_dir: Path, n_scenes: int = 20, n_pairs: int = 10, gaps=(1, 3), seed: int = 0):
    """Median GT overlap (geometry.gt_covisibility) of pairs of nearby frames, per scene, and of the closest
    cross-session pairs; plus depth / intrinsics statistics."""
    import torch
    from vggt_ft.dataio.scene import list_scenes, load_scene
    from vggt_ft.dataio.transforms import centred_crop_resize
    from vggt_ft.geometry import gt_covisibility

    rng = np.random.default_rng(seed)
    dirs = list_scenes(Path(dataset_dir).parent, Path(dataset_dir).name)
    pick = [dirs[i] for i in rng.choice(len(dirs), min(n_scenes, len(dirs)), replace=False)]
    rows = []
    for d in pick:
        sc = load_scene(d)
        seqs = [sc.seq(n) for n in sc.sequences]
        ov, cross_ov = [], []
        for _ in range(n_pairs):
            q = seqs[rng.integers(len(seqs))]
            if q.n < gaps[1] + 1:
                continue
            i = int(rng.integers(0, q.n - gaps[1]))
            j = i + int(rng.integers(gaps[0], gaps[1] + 1))
            ov.append(_pair_overlap(q, i, q, j, centred_crop_resize, gt_covisibility, torch))
        if len(seqs) > 1:
            for _ in range(n_pairs):
                a, b = rng.choice(len(seqs), 2, replace=False)
                qa, qb = seqs[a], seqs[b]
                i = int(rng.integers(qa.n))
                dd = np.linalg.norm(qb.centres - qa.centres[i], axis=1)
                ang = np.array([np.degrees(np.arccos(np.clip((np.trace(qa.E[i, :3, :3] @ qb.E[k, :3, :3].T) - 1) / 2,
                                                             -1, 1))) for k in range(qb.n)])
                cost = dd / max(1e-6, np.median(qa.centres.std(0)) + 1e-6) + ang / 30.0
                j = int(np.argmin(cost))
                cross_ov.append(_pair_overlap(qa, i, qb, j, centred_crop_resize, gt_covisibility, torch))
        z = np.load(seqs[0].dir / "frames.npz")
        rows.append(dict(scene=sc.name, seqs=len(seqs), frames=sc.n_frames, overlap=float(np.median(ov)) if ov else -1,
                         cross=float(np.median(cross_ov)) if cross_ov else -1, dmed=float(np.median(z["dmed"])),
                         valid=float(np.median(z["valid"])), fx=float(np.median(z["K"][:, 0, 0]))))
        print(json.dumps(rows[-1]), flush=True)
    o = [r["overlap"] for r in rows if r["overlap"] >= 0]
    c = [r["cross"] for r in rows if r["cross"] >= 0]
    print(f"== {dataset_dir}: scenes {len(rows)}, median overlap of nearby pairs {np.median(o):.3f}, "
          f"cross-session {np.median(c) if c else float('nan'):.3f}")


def _pair_overlap(qa, i, qb, j, crop, covis, torch):
    ims = []
    for q, k in ((qa, i), (qb, j)):
        img, dep, K, E = q.load(k)
        img, dep, K = crop(img, dep, K, (336, 448))
        ims.append((dep, K, E))
    dep = torch.from_numpy(np.stack([x[0] for x in ims]))[None]
    K = torch.from_numpy(np.stack([x[1] for x in ims])).float()[None]
    E = np.stack([x[2] for x in ims])
    c0 = -E[0, :3, :3].T @ E[0, :3, 3]
    E[:, :3, 3] = E[:, :3, 3] + np.einsum("nij,j->ni", E[:, :3, :3], c0)
    E = torch.from_numpy(E).float()[None]
    return float(covis(dep, dep > 0, E, K)[0, 0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "verify"])
    ap.add_argument("dataset_dir")
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--gaps", type=int, nargs=2, default=[1, 3])
    a = ap.parse_args()
    if a.cmd == "index":
        print(len(write_index(Path(a.dataset_dir))), "scenes")
    else:
        verify(Path(a.dataset_dir), a.scenes, a.pairs, tuple(a.gaps))


if __name__ == "__main__":
    main()
