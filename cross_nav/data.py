"""Training / evaluation windows with BEV labels.

The windows are those of the VGGT fine-tuning (vggt_ft/dataio/windows.py: sequential windows, cross-session windows,
grouped wrong-place references), restricted to places that have a world BEV (cross_nav/labels.py, or the simulator's
mesh labels).  Frame 0 is the current image of a CROSS pass; it must stand on known ground (camera height above the
ground known and within `cam_h_range`), else another frame of the window's own sequence takes its place.

Per window, besides the vggt_ft fields: `bev` (N, N) uint8 label in frame 0's BEV frame (bev.py), `vis` (N, N) uint8
(bit 0: a GT depth point of frame 0 falls in the cell, bit 1: of any frame of the window), `up_cam` (3,) gravity up in
frame 0's camera, `cam_h` frame 0's height above the ground.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from vggt_ft.dataio.transforms import target_shape
from vggt_ft.dataio.windows import DEFAULTS, DatasetPool, WindowSampler, collate, materialise

from .bev import BEVSpec, camera_bev_pose, crop_world, points_to_cells


class WorldBEVs:
    """Lazy, LRU-cached world BEVs of a dataset folder (<bev_root>/<dataset>/<world>.npz)."""

    def __init__(self, root: Path, capacity: int = 24):
        self.root, self.capacity = Path(root), capacity
        self._c: OrderedDict = OrderedDict()

    def path(self, dataset: str, world: str) -> Path:
        return self.root / dataset / f"{world}.npz"

    def get(self, dataset: str, world: str):
        k = (dataset, world)
        if k in self._c:
            self._c.move_to_end(k)
            return self._c[k]
        z = np.load(self.path(dataset, world), allow_pickle=False)
        d = {"bev": z["bev"], "origin": z["origin"].astype(np.float64), "res": float(z["res"]),
             "R_g": z["R_g"].astype(np.float64),
             "cam_h": {k[len("cam_h/"):]: z[k] for k in z.files if k.startswith("cam_h/")}}
        self._c[k] = d
        if len(self._c) > self.capacity:
            self._c.popitem(last=False)
        return d


def labelled_pool(pool: DatasetPool, bevs: WorldBEVs, label_dataset: str) -> bool:
    """Keep the scenes of a pool whose place has a world BEV.  False if none is left."""
    # exclude.txt (optional, one world per line): places whose labels failed a check (e.g. > 85 % occupied)
    ex = bevs.root / label_dataset / "exclude.txt"
    bad = set(ex.read_text().split()) if ex.exists() else set()
    keep = [s for s in pool.scenes if s.world not in bad and bevs.path(label_dataset, s.world).exists()]
    if not keep:
        return False
    pool.scenes = keep
    w = np.array([s.n_frames for s in keep], float) ** pool.cfg.get("scene_weight_pow", 0.5)
    pool.scene_p = w / w.sum()
    by_world = {}
    for s in keep:
        for q in s.sequences:
            by_world.setdefault(s.world_id, []).append((s, q))
    pool.multi = {k: v for k, v in by_world.items() if len(v) >= 2}
    pool.multi_keys = list(pool.multi)
    return True


def visible_cells(depths: torch.Tensor, Es: torch.Tensor, Ks: torch.Tensor, A_g0: np.ndarray, spec: BEVSpec,
                  step: int = 4, h_range=(-3.0, 2.5)) -> np.ndarray:
    """(N,N) uint8: bit 0 = cells hit by frame 0's GT depth points, bit 1 = by any frame's.  Es camera-from-world with
    the world origin at frame 0's centre (materialise), A_g0 (3,3) world -> BEV-frame rotation."""
    vis = np.zeros((spec.size, spec.size), np.uint8)
    S, H, W = depths.shape
    ys, xs = np.mgrid[step // 2:H:step, step // 2:W:step]
    for s in range(S):
        d = depths[s].numpy()[ys, xs]
        ok = d > 0
        if not ok.any():
            continue
        K, E = Ks[s].numpy().astype(np.float64), Es[s].numpy().astype(np.float64)
        P = np.stack([(xs[ok] - K[0, 2]) / K[0, 0] * d[ok], (ys[ok] - K[1, 2]) / K[1, 1] * d[ok], d[ok]], -1)
        Pw = (P - E[:3, 3]) @ E[:3, :3]
        Pb = Pw @ A_g0.T
        sel = (Pb[:, 2] > h_range[0]) & (Pb[:, 2] < h_range[1])
        r, c, inside = points_to_cells(Pb[sel], spec)
        bit = 3 if s == 0 else 2
        vis[r[inside], c[inside]] |= bit
    return vis


class NavStream(IterableDataset):
    """Endless stream of BEV training batches for one rank.  cfg: root (scene cache), bev_root, datasets {name: pool cfg
    (+ label_dataset: the folder name under bev_root, default = dataset name)}, frames [lo, hi], images_per_gpu,
    max_windows, aspect, area, aug, cam_h_range, bev {size, res}."""

    def __init__(self, cfg: dict, rank: int = 0, seed: int = 0, split: str = "train"):
        self.cfg, self.rank, self.seed, self.split = cfg, rank, seed, split
        self.sampler = None
        b = cfg.get("bev", {})
        self.spec = BEVSpec(b.get("size", 400), b.get("res", 0.05))

    def _build(self):
        self.bevs = WorldBEVs(self.cfg["bev_root"])
        pools, self.label_ds = [], {}
        for n, c in self.cfg["datasets"].items():
            if c.get("weight", 1.0) <= 0:
                continue
            try:
                base = {"val_mod": self.cfg.get("val_mod", 0), "zoom": self.cfg.get("zoom", DEFAULTS["zoom"]),
                        "split": self.split}
                pool = DatasetPool(self.cfg["root"], n, {**base, **c}, self.seed)
            except (RuntimeError, FileNotFoundError) as e:
                print(f"[data] skip dataset {n}: {e}", flush=True)
                continue
            lds = c.get("label_dataset", c.get("dataset", n))
            if not labelled_pool(pool, self.bevs, lds):
                print(f"[data] skip dataset {n}: no BEV labels under {self.cfg['bev_root']}/{lds}", flush=True)
                continue
            self.label_ds[pool.name] = lds
            print(f"[data] {n}: {len(pool.scenes)} labelled scenes", flush=True)
            pools.append(pool)
        if not pools:
            raise RuntimeError("no labelled dataset")
        self.sampler = WindowSampler(pools)

    def _cam_h(self, wb, sc, sq, i):
        a = wb["cam_h"].get(f"{sc.dir.name}/{sq.name}")
        return float(a[i]) if a is not None and i < len(a) else float("nan")

    def _frame0(self, rng, pool, frames):
        """Make sure frame 0 stands on known ground: swap in a valid frame of the window (same sequence first)."""
        lo, hi = self.cfg.get("cam_h_range", [0.1, 2.2])
        lds = self.label_ds[pool.name]
        order = [0] + [k for k in range(1, len(frames)) if frames[k][1] is frames[0][1]] + \
            [k for k in range(1, len(frames)) if frames[k][1] is not frames[0][1]]
        for k in order:
            sc, sq, i = frames[k]
            wb = self.bevs.get(lds, sc.world)
            h = self._cam_h(wb, sc, sq, i)
            if math.isfinite(h) and lo <= h <= hi:
                if k:
                    frames = list(frames)
                    frames[0], frames[k] = frames[k], frames[0]
                return frames, wb, h, k
        return None

    def make_item(self, rng, pool, frames, negs, kind, hw, aug=True):
        r = self._frame0(rng, pool, frames)
        if r is None:
            return None
        frames, wb, cam_h, k = r
        if k:
            negs = [0 if n == k else (k if n == 0 else n) for n in negs]
            if 0 in negs:              # a wrong-place group became frame 0's sequence: not a valid window
                return None
        sc, sq, i = frames[0]
        E0 = sq.E[i].astype(np.float64)
        p, right, fwd, up_cam = camera_bev_pose(E0, wb["R_g"])
        label = crop_world(wb["bev"], wb["origin"], wb["res"], p, right, fwd, self.spec)
        it = materialise(frames, hw, rng if aug else None, self.cfg.get("aug") if aug else None,
                         pool.cfg["zoom"] if aug else (1.0, 1.0))
        A = np.stack([right, fwd, np.array([0.0, 0.0, 1.0])]) @ wb["R_g"]          # world -> BEV frame (rotation)
        vis = visible_cells(it["depths"], it["extrinsics"], it["intrinsics"], A, self.spec)
        S = len(frames)
        it["bev"] = torch.from_numpy(label)
        it["vis"] = torch.from_numpy(vis)
        it["up_cam"] = torch.from_numpy(up_cam.astype(np.float32))
        it["cam_h"] = torch.tensor(cam_h, dtype=torch.float32)
        it["is_neg"] = torch.tensor([kk in negs for kk in range(S)], dtype=torch.bool)
        it["same_session"] = torch.tensor([f[1] is frames[0][1] for f in frames], dtype=torch.bool)
        it["dataset"] = pool.name
        it["kind"] = kind
        return it

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        rng = np.random.default_rng([self.seed, self.rank, wid])
        if self.sampler is None:
            self._build()
        cfg = self.cfg
        while True:
            S = int(rng.integers(cfg["frames"][0], cfg["frames"][1] + 1))
            B = int(np.clip(cfg["images_per_gpu"] // S, 1, cfg.get("max_windows", 12)))
            hw = target_shape(float(rng.uniform(*cfg["aspect"])), cfg.get("area", 512 * 512))
            items = []
            while len(items) < B:
                try:
                    pool, frames, negs, kind = self.sampler.sample(rng, S)
                    it = self.make_item(rng, pool, frames, negs, kind, hw)
                except (OSError, ValueError, KeyError) as e:
                    print(f"[data] skip window: {e!r}", flush=True)
                    continue
                if it is not None:
                    items.append(it)
            yield collate(items)
