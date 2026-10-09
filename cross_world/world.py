"""The reconstruction of a whole map: chunks of keyframe-anchored Gaussians.

Build: views of the map -> depth -> partition -> per chunk: initialise, train, keep the chunk's own Gaussians (its
cell: "near"; beyond the mapped region: "far") -> anchor every Gaussian to its nearest keyframes.

Anchoring: each Gaussian remembers its k nearest keyframes (camera centres; weights falling off with distance) and the
keyframe poses at build time.  When the map's poses change (a loop closure, a merged session, a re-optimised graph),
`repose` moves the Gaussians with a blend of their anchors' pose corrections (embedded deformation over the
topological graph, as surfel maps deform after loop closure): no retraining, and a chunk bends with its keyframes
instead of moving as one rigid block.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from cross_world.depth import StereoDepth, consistent_mask, view_depth
from cross_world.gaussians import (TrainConfig, TrainResult, ViewBatch, align_pose, apply_appearance, init_gaussians,
                                   lpips, matrix_to_quat, psnr, quat_mul, render, ssim, train_chunk)
from cross_world.map_views import MapViews, View
from cross_world.partition import Partition, assign_views, make_partition

LAYERS = ("near", "far")


@dataclass
class BuildConfig:
    max_views: int = 200            # keyframes per chunk cell (a map with fewer is one chunk)
    margin: float = 0.0             # cell margin (m); 0: 15 % of the median keyframe depth x 10, at least 2 m
    min_visible: float = 0.1        # share of a view's depth points in a cell that adds it to the chunk
    test_every: int = 8             # every n-th keyframe (by id) is held out for evaluation (0: none)
    depth: str = "auto"             # auto | sensor | sgbm | vggt | vggt_sgbm | fused | fstereo | none (cross_world/depth.py)
    depth_workers: int = 8
    vggt_checkpoint: str = "models/VGGT-Omega/vggt_omega_1b_512.pt"
    fstereo_checkpoint: str = ""    # FoundationStereo model_best_bp2.pth (cfg.yaml next to it); FOUNDATION_STEREO_REPO
    fstereo_iters: int = 32
    max_depth: float = 0.0          # 0: no clipping
    anchors: int = 4
    # after training, drop Gaussians below this opacity: MCMC keeps many near-transparent ones to relocate; on KITTI 07
    # 0.02 removed 58 % of them at -0.01 dB (16 held-out views, SSIM / LPIPS unchanged)
    prune_opacity: float = 0.02
    # extra (captured) views: second training stage after alignment against the keyframe-trained chunk (_second_stage)
    capture_align_iters: int = 60
    capture_keep: float = 1.5            # keep views whose aligned L1 residual is at most this x the median
    capture_max_move: float = 0.1        # ... whose alignment moved them at most this x the median depth (m) ...
    capture_max_rot_deg: float = 3.0     # ... and this many degrees
    capture_means_lr_scale: float = 0.5
    # sky model (cross_world/sky.py): sky masks of the training views from a segmentation network, a sky texture
    # behind the Gaussians, no depth and no Gaussians on sky pixels.  Outdoor scenes; needs `transformers`
    sky: bool = False
    sky_model: str = ""             # folder / hub id of the OneFormer weights ("": CROSS_SKY_MODEL, else the hub default)
    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class ChunkModel:
    index: int
    layers: Dict[str, Dict[str, torch.Tensor]]          # near / far -> splat tensors (CPU)
    anchor_ids: Dict[str, torch.Tensor]                 # layer -> (N, k) int64 keyframe ids
    anchor_w: Dict[str, torch.Tensor]                   # layer -> (N, k) float32
    stats: dict = field(default_factory=dict)
    sky: Optional[dict] = None                          # {"tex": (3, H, W) colours, "R": (3, 3)} (cross_world/sky.py)


class World:
    def __init__(self, partition: Partition, chunks: List[ChunkModel], kf_poses: Dict[int, np.ndarray],
                 pose_delta: Dict[int, np.ndarray], appearance: Dict[int, np.ndarray], meta: dict):
        self.partition = partition
        self.chunks = chunks
        self.kf_poses = kf_poses               # keyframe poses the Gaussians are expressed with (build / last repose)
        self.pose_delta = pose_delta           # photometric refinement of each training view (T @ delta)
        self.appearance = appearance
        self.meta = meta

    @property
    def sh_degree(self) -> int:
        return int(self.meta.get("sh_degree", 3))

    def count(self) -> Dict[str, int]:
        return {l: sum(len(c.layers[l]["means"]) for c in self.chunks if l in c.layers) for l in LAYERS}

    def chunk_of(self, center: np.ndarray) -> int:
        """Chunk of a camera position: the chunk that owns the nearest keyframe (at the keyframes' current poses, so
        it stays right after a repose bends the map; the build-time cells would not)."""
        if not hasattr(self, "_kf_chunk"):
            self._kf_chunk = {k: c.index for c in self.partition.chunks for k in c.core_ids}
        ids = [k for k in self.kf_poses if k in self._kf_chunk]
        if not ids:
            return int(self.partition.nearest_cell(center[None])[0])
        P = np.stack([self.kf_poses[k][:3, 3] for k in ids])
        return self._kf_chunk[ids[int(np.argmin(np.linalg.norm(P - center, axis=1)))]]

    def sky_for(self, center: np.ndarray, device="cuda"):
        """The SkyModel of the camera's chunk (None without a sky model)."""
        from cross_world.sky import sky_from_state
        own = self.chunk_of(center)
        c = next((c for c in self.chunks if c.index == own), None)
        return sky_from_state(c.sky if c is not None else None, device)

    def splats_for(self, center: Optional[np.ndarray] = None, device="cuda", chunks: Optional[List[int]] = None):
        """Every near layer, and the far layer of the camera's chunk (chunk_of; all far layers when None)."""
        own = self.chunk_of(center) if center is not None else None
        parts = []
        for c in self.chunks:
            if chunks is not None and c.index not in chunks:
                continue
            parts.append(c.layers["near"])
            if "far" in c.layers and (own is None or c.index == own):
                parts.append(c.layers["far"])
        return {k: torch.cat([p[k] for p in parts]).to(device) for k in parts[0]}

    # ------------------------------------------------------------------------------------------------ re-posing
    def repose(self, new_poses: Dict[int, np.ndarray]) -> dict:
        """Move the Gaussians with their anchor keyframes' pose changes (see the module doc)."""
        ids = sorted(self.kf_poses)
        lut = torch.full((max(ids) + 1,), -1, dtype=torch.int64)
        lut[torch.tensor(ids)] = torch.arange(len(ids))
        D = np.stack([new_poses.get(k, self.kf_poses[k]) @ np.linalg.inv(self.kf_poses[k]) for k in ids])
        Dt = torch.from_numpy(D).float()
        Dq = matrix_to_quat(Dt[:, :3, :3])
        moved = 0.0
        for c in self.chunks:
            for l, sp in c.layers.items():
                aid = c.anchor_ids[l]
                if len(aid) == 0:
                    continue
                ai = torch.where(aid <= lut.numel() - 1, lut[aid.clamp(max=lut.numel() - 1)], -1)
                w = c.anchor_w[l].clone()
                w[ai < 0] = 0
                ai = ai.clamp(min=0)
                w = w / w.sum(1, keepdim=True).clamp(min=1e-9)
                x = sp["means"]
                R, t = Dt[ai, :3, :3], Dt[ai, :3, 3]                      # (N, k, 3, 3), (N, k, 3)
                xs = torch.einsum("nkij,nj->nki", R, x) + t
                x_new = (w[..., None] * xs).sum(1)
                q = Dq[ai]                                                 # (N, k, 4); align signs to the first
                q = q * torch.where((q * q[:, :1]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
                qb = torch.nn.functional.normalize((w[..., None] * q).sum(1), dim=-1)
                moved = max(moved, float((x_new - x).norm(dim=1).max()) if len(x) else 0.0)
                sp["means"] = x_new
                sp["quats"] = quat_mul(qb, torch.nn.functional.normalize(sp["quats"], dim=-1))
                if sp["shN"].shape[1] > 0:
                    sp["shN"] = rotate_sh(sp["shN"], qb)
        for k in ids:
            if k in new_poses:
                self.kf_poses[k] = np.asarray(new_poses[k], np.float64)
        return {"max_move_m": moved}

    # ------------------------------------------------------------------------------------------------ io
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": 1, "partition": self.partition.to_dict(),
                    "chunks": [{"index": c.index, "layers": {l: {k: v.cpu() for k, v in sp.items()} for l, sp in c.layers.items()},
                                "anchor_ids": c.anchor_ids, "anchor_w": c.anchor_w, "stats": c.stats, "sky": c.sky}
                               for c in self.chunks],
                    "kf_poses": {int(k): v for k, v in self.kf_poses.items()},
                    "pose_delta": {int(k): v for k, v in self.pose_delta.items()},
                    "appearance": {int(k): v for k, v in self.appearance.items()}, "meta": self.meta}, path)

    @staticmethod
    def load(path) -> "World":
        d = torch.load(path, map_location="cpu", weights_only=False)
        chunks = [ChunkModel(c["index"], c["layers"], c["anchor_ids"], c["anchor_w"], c.get("stats", {}), c.get("sky"))
                  for c in d["chunks"]]
        return World(Partition.from_dict(d["partition"]), chunks, d["kf_poses"], d["pose_delta"], d["appearance"], d["meta"])


# ---------------------------------------------------------------------------------------------------- SH rotation
def sh_basis(dirs: torch.Tensor, degree: int) -> torch.Tensor:
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast
    return _eval_sh_bases_fast((degree + 1) ** 2, dirs)


_SH_DIRS = None


def rotate_sh(shN: torch.Tensor, q_wxyz: torch.Tensor) -> torch.Tensor:
    """Rotate the view-dependent SH coefficients (N, K-1, 3) of Gaussians rotated by q (N, 4): the colour seen
    along direction d after the rotation equals the colour along R^T d before.  The per-Gaussian rotation matrix of
    the SH bands is fitted on sample directions (exact for band-limited functions)."""
    global _SH_DIRS
    n, km1, _ = shN.shape
    deg = int(math.isqrt(km1 + 1)) - 1
    if deg <= 0 or n == 0:
        return shN
    if _SH_DIRS is None:
        g = torch.Generator().manual_seed(0)
        d = torch.nn.functional.normalize(torch.randn(64, 3, generator=g), dim=-1)
        _SH_DIRS = d
    dirs = _SH_DIRS
    Y = sh_basis(dirs, deg)[:, 1:]                                     # (M, K-1)
    pinv = torch.linalg.pinv(Y)                                        # (K-1, M)
    w, x, y, z = q_wxyz.unbind(-1)
    R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).view(-1, 3, 3)
    out = torch.empty_like(shN)
    for s in range(0, n, 200_000):
        Rs = R[s:s + 200_000]
        d_back = torch.einsum("nji,mj->nmi", Rs, dirs)                  # R^T d for each sample direction
        Yb = sh_basis(d_back.reshape(-1, 3), deg)[:, 1:].view(len(Rs), len(dirs), -1)
        M = torch.einsum("km,nml->nkl", pinv, Yb)                       # new coeff k from old coeff l
        out[s:s + 200_000] = torch.einsum("nkl,nlc->nkc", M, shN[s:s + 200_000])
    return out


# ---------------------------------------------------------------------------------------------------- build
def _load_batch(views: List[View], depths: Dict[int, np.ndarray], device,
                skies: Optional[Dict[int, np.ndarray]] = None) -> ViewBatch:
    imgs = torch.from_numpy(np.stack([v.image() for v in views])).to(device)
    sky = None
    if skies and all(v.id in skies for v in views):
        sky = torch.from_numpy(np.stack([skies[v.id] for v in views])).to(device)
    has_d = all(v.id in depths for v in views) and len(views) > 0
    dep = torch.from_numpy(np.stack([depths[v.id] for v in views])).to(device).half() if has_d else None
    T = torch.from_numpy(np.stack([v.T_wc for v in views])).float().to(device)
    K = torch.from_numpy(np.stack([v.K for v in views])).float().to(device)
    times = torch.tensor([v.timestamp if v.timestamp is not None else float(v.id) for v in views], device=device)
    return ViewBatch([v.id for v in views], imgs, dep, T, K, times, views[0].width, views[0].height, sky)


def _neighbours(T_wc: np.ndarray, k: int = 4) -> List[List[int]]:
    """Nearest views by camera centre, among those looking roughly the same way (optical axes within 60 deg)."""
    c = T_wc[:, :3, 3]
    z = T_wc[:, :3, 2]
    out = []
    for i in range(len(c)):
        d = np.linalg.norm(c - c[i], axis=1)
        d[(z @ z[i]) < 0.5] = np.inf
        d[i] = np.inf
        o = np.argsort(d)[:k]
        out.append([int(j) for j in o if np.isfinite(d[j])])
    return out


def compute_anchors(points: torch.Tensor, kf_ids: List[int], centers: np.ndarray, k: int):
    """k nearest keyframe camera centres of each point, weights (1 - d / d_{k+1})^2 (normalised)."""
    C = torch.from_numpy(centers).float()
    ids = torch.tensor(kf_ids, dtype=torch.int64)
    kk = min(k + 1, len(kf_ids))
    out_i, out_w = [], []
    for s in range(0, len(points), 500_000):
        p = points[s:s + 500_000].float().cpu()
        d = torch.cdist(p, C)
        dv, di = torch.topk(d, kk, largest=False)
        if kk > k:
            dmax = dv[:, -1:].clamp(min=1e-6)
            dv, di = dv[:, :k], di[:, :k]
        else:
            dmax = dv[:, -1:].clamp(min=1e-6) * 1.01 + 1e-6
        w = (1 - dv / dmax).clamp(min=0) ** 2 + 1e-6
        out_i.append(ids[di])
        out_w.append(w / w.sum(1, keepdim=True))
    if not out_i:
        return torch.zeros(0, k, dtype=torch.int64), torch.zeros(0, k)
    return torch.cat(out_i), torch.cat(out_w)


def split_views(mv: MapViews, test_every: int):
    if test_every <= 0:
        return list(mv.views), []
    vs = sorted(mv.views, key=lambda v: v.id)
    test = [v for i, v in enumerate(vs) if i % test_every == test_every // 2]
    tid = {v.id for v in test}
    return [v for v in vs if v.id not in tid], test


def build_world(mv: MapViews, cfg: BuildConfig, device="cuda", log=print, only_chunks: Optional[List[int]] = None,
                chunk_dir: Optional[Path] = None, extra_views: Optional[List[View]] = None) -> World:
    """`extra_views`: more training views that are not keyframes (frames kept by mapping.world_capture); they
    train the chunks that see them but anchor nothing (the Gaussians follow the keyframes)."""
    t_all = time.time()
    kf_train, test_views = split_views(mv, cfg.test_every)
    extra = list(extra_views or [])
    train_views = kf_train + extra
    log(f"{len(mv.views)} keyframes ({mv.mode}): {len(kf_train)} train, {len(test_views)} test; "
        f"{mv.views[0].width}x{mv.views[0].height} images; {len(extra)} extra (captured) training views")
    # ---- depth of every training view (needed for the partition's visibility and the initialisation)
    t0 = time.time()
    depths: Dict[int, np.ndarray] = {}
    learned = cfg.depth in ("vggt", "vggt_sgbm", "fused", "fstereo")
    if learned:
        # one network on the GPU, views in turn; freed before training
        from cross_world.depth import FoundationStereoDepth, VGGTStereoDepth
        kw = {}
        if cfg.depth == "fstereo":
            kw["fstereo"] = FoundationStereoDepth(cfg.fstereo_checkpoint, device, iters=cfg.fstereo_iters)
        else:
            kw["vggt"] = VGGTStereoDepth(cfg.vggt_checkpoint, device)
            kw["stereo"] = StereoDepth(train_views[0].width)
        for v in train_views:
            d = view_depth(v, mv, cfg.depth, cfg.max_depth or None, **kw)
            if d is not None:
                depths[v.id] = d
        del kw
        torch.cuda.empty_cache()
    elif cfg.depth != "none":
        # views in parallel threads (image decoding and SGBM release the GIL; one matcher per thread)
        import threading
        from concurrent.futures import ThreadPoolExecutor
        local = threading.local()

        def one(v):
            st = None
            if mv.mode == "stereo" or cfg.depth == "sgbm":
                st = getattr(local, "st", None)
                if st is None:
                    st = local.st = StereoDepth(train_views[0].width)
            return v.id, view_depth(v, mv, cfg.depth, cfg.max_depth or None, st)
        with ThreadPoolExecutor(max_workers=cfg.depth_workers) as ex:
            for vid, d in ex.map(one, train_views):
                if d is not None:
                    depths[vid] = d
    t_depth = time.time() - t0
    if cfg.train.consistency_tol > 0 and depths and len(train_views) == len(depths):
        T_all = np.stack([v.T_wc for v in train_views])
        nb = _neighbours(T_all)
        dl = [torch.from_numpy(depths[v.id]).to(device) for v in train_views]
        masks = consistent_mask(dl, torch.from_numpy(np.stack([v.K for v in train_views])).float().to(device),
                                torch.from_numpy(T_all).float().to(device), nb, rel_tol=cfg.train.consistency_tol)
        kept = 0
        tot = 0
        for v, m, d in zip(train_views, masks, dl):
            tot += int((d > 0).sum())
            kept += int(m.sum())
            depths[v.id] = depths[v.id] * m.cpu().numpy()
        log(f"depth consistency filter: kept {kept / max(tot, 1):.1%} of the depth pixels")
    skies: Dict[int, np.ndarray] = {}
    seg = None
    if cfg.sky:
        try:
            from cross_world.sky import SkySegmenter
            seg = SkySegmenter(cfg.sky_model, device)
        except Exception as ex:                    # transformers or the weights missing (docs/WORLD.md, Install)
            log(f"WARNING: no sky model (the sky segmenter did not load: {type(ex).__name__}: {ex})")
    if seg is not None:
        t1 = time.time()
        for v in train_views:
            skies[v.id] = seg(v.image())
            if v.id in depths:                     # stereo / sensor depth on the sky is noise
                depths[v.id] = np.where(skies[v.id], 0, depths[v.id]).astype(depths[v.id].dtype)
        del seg
        torch.cuda.empty_cache()
        log(f"sky masks of {len(skies)} views in {time.time() - t1:.0f}s: "
            f"{np.mean([m.mean() for m in skies.values()]):.1%} of the pixels sky")
    zs = np.concatenate([d[d > 0][::97] for d in depths.values()]) if depths else np.array([5.0])
    zmed = float(np.median(zs)) if len(zs) else 5.0
    log(f"depth ({cfg.depth}, {len(depths)} views) in {t_depth:.1f}s, median {zmed:.2f} m")
    # ---- partition (cells from the keyframes; captured frames join the chunks that see them)
    kf_ids = [v.id for v in kf_train]
    kf_centers = np.stack([v.center for v in kf_train])
    ids = [v.id for v in train_views]
    centers = np.stack([v.center for v in train_views])
    margin = cfg.margin or max(2.0, 1.5 * zmed)
    part = make_partition(kf_ids, kf_centers, mv.up(), cfg.max_views, margin)
    samples = {}
    for v in train_views:
        if v.id in depths:
            p, *_ = _backproject_np(depths[v.id], v.K, v.T_wc, stride=8)
            samples[v.id] = p[np.random.default_rng(0).permutation(len(p))[:2000]]
    assign_views(part, ids, centers, samples, cfg.min_visible)
    log(f"partition: {len(part.chunks)} chunks, margin {margin:.1f} m, train views per chunk "
        f"{[len(c.train_ids) for c in part.chunks]}")
    by_id = {v.id: v for v in train_views}
    kf_set = set(kf_ids)
    chunks, pose_delta, appearance = [], {}, {}
    for c in part.chunks:
        if only_chunks is not None and c.index not in only_chunks:
            continue
        tc = time.time()
        views = [by_id[i] for i in c.train_ids]
        kf_views = [v for v in views if v.id in kf_set]
        ex_views = [v for v in views if v.id not in kf_set]
        # extra (captured) views join in a second stage: their poses (odometry between keyframes) can be off by
        # degrees, and trained or initialised from as they are they spread floaters that ruin the chunk
        vb = _load_batch(kf_views if ex_views else views, depths, device, skies)
        g_lo, g_hi = c.lo - margin, c.hi + margin

        def keep_fn(p, lo=g_lo, hi=g_hi):
            g = (p - torch.from_numpy(part.origin).float().to(p.device)) @ torch.from_numpy(part.axes).float().to(p.device).T
            inside = ((g >= torch.from_numpy(lo).float().to(p.device)) & (g < torch.from_numpy(hi).float().to(p.device))).all(1)
            return inside
        init = init_gaussians(vb, cfg.train, keep_fn=keep_fn)
        # position learning rate in units of the scene's depth (the initial points are already metric and placed by
        # the depth; the camera extent of a long chunk would make the steps far too large)
        local_scale = zmed
        log(f"chunk {c.index}: {len(views)} views ({len(c.core_ids)} core), {len(init['means'])} initial Gaussians")
        sky = None
        if vb.sky is not None and float(vb.sky.float().mean()) < 0.002:
            vb.sky = None                          # (almost) no sky in this chunk's views: no sky model
        if vb.sky is not None:
            from cross_world.sky import SkyModel, sky_frame
            px = vb.images[vb.sky][::97].float() / 255.0
            init_rgb = tuple(px.median(0).values.tolist()) if len(px) else (0.7, 0.8, 0.9)
            sky = SkyModel(sky_frame(mv.up()), cfg.train.sky_res, init_rgb).to(device)
            del px
        res = train_chunk(vb, init, cfg.train, local_scale, log=log, sky=sky)
        if ex_views:
            res = _second_stage(res, kf_views, ex_views, depths, cfg, local_scale, device, log, skies)
        sp = {k: v.cpu() for k, v in res.splats.items()}
        if cfg.prune_opacity > 0:
            keep = torch.sigmoid(sp["opacities"]) >= cfg.prune_opacity
            sp = {k: v[keep] for k, v in sp.items()}
        # keep the chunk's own Gaussians: its cell (near) and beyond the mapped region (far)
        own = torch.from_numpy(part.cell_of(sp["means"].numpy()) == c.index)
        far = torch.from_numpy(~part.in_region(sp["means"].numpy()))
        layers = {"near": {k: v[own] for k, v in sp.items()}, "far": {k: v[far] for k, v in sp.items()}}
        aids, aws = {}, {}
        cids = [i for i in c.train_ids if i in kf_set]
        cc = np.stack([by_id[i].center for i in cids])
        for l, s in layers.items():
            aids[l], aws[l] = compute_anchors(s["means"], cids, cc, cfg.anchors)
        st = dict(res.stats, chunk_s=round(time.time() - tc, 1), kept_near=int(own.sum()), kept_far=int(far.sum()),
                  core=len(c.core_ids))
        log(f"chunk {c.index}: trained in {res.stats['train_s']:.0f}s, kept {st['kept_near']} near + {st['kept_far']} far "
            f"of {res.stats['n_final']}")
        sky_state = None
        if res.sky is not None:
            sky_state = {"tex": res.sky.colours().half().cpu(), "R": res.sky.R.cpu().clone()}
            st["sky_seen"] = float((res.sky.seen > 0).float().mean())
        cm = ChunkModel(c.index, layers, aids, aws, st, sky_state)
        for vid in c.core_ids:                      # a view's refinement / appearance from the chunk that owns it
            pose_delta[vid] = res.pose_delta[vid]
            appearance[vid] = res.appearance[vid]
        for vid in c.train_ids:                     # (captured views the second stage dropped have none)
            if vid in res.pose_delta:
                pose_delta.setdefault(vid, res.pose_delta[vid])
                appearance.setdefault(vid, res.appearance[vid])
        if chunk_dir is not None:
            Path(chunk_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"chunk": cm, "pose_delta": dict(res.pose_delta), "appearance": dict(res.appearance)},
                       Path(chunk_dir) / f"chunk_{c.index:03d}.pt")
        chunks.append(cm)
        del vb, res
        torch.cuda.empty_cache()
    meta = {"map": mv.map_path, "mode": mv.mode, "n_views": len(mv.views), "train_ids": kf_ids, "n_extra": len(extra),
            "test_ids": [v.id for v in test_views], "sh_degree": cfg.train.sh_degree, "zmed": zmed,
            "depth_s": round(t_depth, 1), "build_s": round(time.time() - t_all, 1),
            "config": {**{k: v for k, v in asdict(cfg).items() if k != "train"}, "train": asdict(cfg.train)},
            "image_size": [mv.views[0].width, mv.views[0].height]}
    return World(part, chunks, {v.id: v.T_wc.copy() for v in mv.views}, pose_delta, appearance, meta)


def _second_stage(res, kf_views, ex_views, depths, cfg: "BuildConfig", scene_scale, device, log, skies=None):
    """Stage 2 of a chunk with extra (captured) views: align each extra view's pose photometrically against the
    keyframe-trained Gaussians (as test views are aligned), drop the views that still do not fit (motion blur, people,
    a failed alignment), then train on keyframes and kept views together, from the stage-1 Gaussians."""
    import copy
    import torch.nn.functional as F
    t0 = time.time()
    sp = res.splats
    sh = cfg.train.sh_degree
    kf_t = sorted((v.timestamp if v.timestamp is not None else 0.0, v.id) for v in kf_views)
    aligned, resid, moves = [], [], []
    for v in ex_views:
        t = v.timestamp if v.timestamp is not None else 0.0
        near = min(kf_t, key=lambda r: abs(r[0] - t))[1]
        A = torch.from_numpy(res.appearance[near]).float().to(device)
        gt = torch.from_numpy(v.image()).to(device).float() / 255.0
        T0 = torch.from_numpy(v.T_wc).float().to(device)
        K = torch.from_numpy(v.K).float().to(device)
        T = align_pose(sp, T0, K, v.width, v.height, gt, sh, A, iters=cfg.capture_align_iters, sky=res.sky)
        with torch.no_grad():
            rgb, _, alpha, _ = render(sp, T[None], K[None], v.width, v.height, sh, render_mode="RGB")
            from cross_world.sky import composite
            rgb = composite(rgb, alpha, res.sky, T[None], K[None], v.width, v.height, A[None]).clamp(0, 1)
            resid.append(float((rgb[0] - gt).abs().mean()))
        D = (torch.linalg.inv(T0) @ T).cpu().numpy()
        moves.append((float(np.linalg.norm(D[:3, 3])),
                      float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))))))
        aligned.append(T.cpu().numpy().astype(np.float64))
    resid = np.array(resid)
    moves = np.array(moves)
    ok = (resid <= cfg.capture_keep * np.median(resid)) & (moves[:, 0] <= cfg.capture_max_move * scene_scale) \
        & (moves[:, 1] <= cfg.capture_max_rot_deg)
    kept = []
    for v, T, k in zip(ex_views, aligned, ok):
        if k:
            nv = copy.copy(v)
            nv.T_wc = T
            kept.append(nv)
    # keyframes at their stage-1 refined poses, kept extra views at their aligned poses
    kfs = []
    for v in kf_views:
        nv = copy.copy(v)
        nv.T_wc = v.T_wc @ res.pose_delta[v.id]
        kfs.append(nv)
    log(f"  stage 2: aligned {len(ex_views)} extra views in {time.time() - t0:.0f}s (pose change median "
        f"{np.median(moves[:, 0]):.3f} m / {np.median(moves[:, 1]):.2f} deg, p90 {np.percentile(moves[:, 0], 90):.3f} m / "
        f"{np.percentile(moves[:, 1], 90):.2f} deg), kept {len(kept)}")
    vb = _load_batch(kfs + kept, depths, device, skies)
    tcfg = copy.deepcopy(cfg.train)
    tcfg.means_lr *= cfg.capture_means_lr_scale
    res2 = train_chunk(vb, {k: v.detach() for k, v in sp.items()}, tcfg, scene_scale, log=log, sky=res.sky)
    # a keyframe's correction: stage 1, then stage 2
    pd = {v.id: res.pose_delta[v.id] @ res2.pose_delta[v.id] for v in kf_views}
    pd.update({v.id: res2.pose_delta[v.id] for v in kept})
    st = dict(res2.stats, stage1=res.stats, n_extra=len(ex_views), n_extra_kept=len(kept), align_s=round(time.time() - t0, 1),
              extra_move_median=[float(np.median(moves[:, 0])), float(np.median(moves[:, 1]))])
    st["train_s"] = round(res.stats["train_s"] + res2.stats["train_s"], 1)
    return TrainResult(res2.splats, pd, res2.appearance, st, sky=res2.sky if res2.sky is not None else res.sky)


@torch.no_grad()
def clean_world(world: World, views: List[View], min_views: int = 2, needle_ratio: float = 0.0,
                needle_size: float = 0.02, device="cuda", log=print, depths: Optional[Dict[int, np.ndarray]] = None,
                carve_tol: float = 0.15) -> dict:
    """Remove Gaussians no training view constrains: those that project into fewer than `min_views` of the chunk's
    training views (outside every frustum or seen once: the photometric loss never checked them from a second
    direction), and needles (largest scale > needle_size x the median depth and largest / middle scale > needle_ratio:
    long along one axis only), which are right edge-on from the training rays and streaks from anywhere else.
    Measured on home1-1 (512 px): the visibility test removes < 0.2 % (opacity pruning already took the junk) at no
    cost; the needle test (ratio 10) 15 % at -0.38 dB held-out PSNR, so it is off by default and the export puts
    needles in a layer of their own that the viewer hides only in its overview (export_world needle_ratio)."""
    by = {v.id: v for v in views}
    zmed = float(world.meta.get("zmed", 1.0))
    stats = {}
    for c in world.chunks:
        ids = [i for i in world.partition.chunks[c.index].train_ids if i in by] if c.index < len(world.partition.chunks) else []
        Ts, Ks, sizes = [], [], []
        for i in ids:
            v = by[i]
            T = v.T_wc @ world.pose_delta[i] if i in world.pose_delta else v.T_wc
            Ts.append(T)
            Ks.append(v.K)
            sizes.append((v.width, v.height))
        for l, sp in c.layers.items():
            n = len(sp["means"])
            if n == 0 or not ids:
                continue
            g = {k: v.to(device) for k, v in sp.items()}
            count = torch.zeros(n, device=device, dtype=torch.int32)
            for T, K, (w, h) in zip(Ts, Ks, sizes):
                _, _, _, info = render(g, torch.from_numpy(T).float().to(device)[None], torch.from_numpy(K).float().to(device)[None],
                                       w, h, 0, render_mode="RGB")
                r = info["radii"][0]
                vis = (r > 0).all(-1) if r.dim() == 2 else r > 0
                count += vis.int()
            keep = count >= min_views
            if depths:
                # free-space carving with every training view's depth (cross_world/floaters.py)
                from cross_world.floaters import free_space_votes
                dv = [i for i in ids if i in depths]
                viol, agree = free_space_votes(g["means"], torch.sigmoid(g["opacities"]),
                                               [Ts[ids.index(i)] for i in dv], [Ks[ids.index(i)] for i in dv],
                                               [torch.from_numpy(depths[i]).to(device) for i in dv], tol=carve_tol)
                keep &= ~((viol >= 2) & (agree == 0))
            if l == "near" and needle_ratio > 0:
                # a needle is long along one axis only (a flat disk, the usual surface Gaussian, has two long axes)
                sc = torch.exp(g["scales"]).sort(1, descending=True).values
                keep &= ~((sc[:, 0] / sc[:, 1].clamp(min=1e-9) > needle_ratio) & (sc[:, 0] > needle_size * zmed))
            keep = keep.cpu()
            c.layers[l] = {k: v[keep] for k, v in sp.items()}
            c.anchor_ids[l] = c.anchor_ids[l][keep]
            c.anchor_w[l] = c.anchor_w[l][keep]
            stats[f"c{c.index}_{l}"] = [n, int(keep.sum())]
    log("clean: " + ", ".join(f"{k} {a} -> {b}" for k, (a, b) in stats.items()))
    world.meta["cleaned"] = {"min_views": min_views, "needle_ratio": needle_ratio, "needle_size": needle_size}
    return stats


def _backproject_np(depth, K, T_wc, stride=4):
    d = depth[::stride, ::stride]
    ys, xs = np.nonzero(d > 0)
    z = d[ys, xs]
    x = (xs * stride + 0.5 - K[0, 2]) / K[0, 0] * z
    y = (ys * stride + 0.5 - K[1, 2]) / K[1, 1] * z
    pc = np.stack([x, y, z], 1)
    return pc @ T_wc[:3, :3].T + T_wc[:3, 3], ys, xs


# ---------------------------------------------------------------------------------------------------- evaluation
def _nearest_appearance(world: World, v: View, by_time: List) -> Optional[np.ndarray]:
    if not world.appearance or not by_time:
        return None
    t = v.timestamp if v.timestamp is not None else 0.0
    j = min(by_time, key=lambda r: abs(r[0] - t))
    return world.appearance.get(j[1])


def evaluate(world: World, views: List[View], device="cuda", align: bool = True, align_iters: int = 100,
             save_dir: Optional[Path] = None, save_max: int = 12, log=print, kf_times: Optional[Dict[int, float]] = None,
             common_width: int = 512, sky_seg=None) -> dict:
    """PSNR / SSIM / LPIPS (and depth error where the view has depth) of renders at the test views: as posed, and
    after test-time pose alignment.  Appearance: the affine colour of the training keyframe nearest in time.
    `sky_seg` (a SkySegmenter): also the mean opacity of the Gaussians on the view's sky pixels (sky_alpha; the sky
    painted by Gaussians, which float once the viewpoint moves)."""
    by_time = sorted((t, k) for k, t in (kf_times or {}).items() if k in world.appearance)
    rows = []
    cache = {}
    t0 = time.time()
    for n, v in enumerate(views):
        own = world.chunk_of(v.center)
        if own not in cache:
            cache.clear()
            cache[own] = (world.splats_for(v.center, device), world.sky_for(v.center, device))
        sp, sky = cache[own]
        sky_m = torch.from_numpy(sky_seg(v.image())).to(device) if sky_seg is not None else None
        gt = torch.from_numpy(v.image()).to(device).float() / 255.0
        T = torch.from_numpy(v.T_wc).float().to(device)
        K = torch.from_numpy(v.K).float().to(device)
        A = _nearest_appearance(world, v, by_time)
        At = torch.from_numpy(A).float().to(device) if A is not None else None
        row = {"id": v.id, "source_index": v.source_index}
        if v.id in world.pose_delta:              # a training view: its photometrically refined pose
            T = T @ torch.from_numpy(world.pose_delta[v.id]).float().to(device)
            if v.id in world.appearance:
                At = torch.from_numpy(world.appearance[v.id]).float().to(device)
        for tag in (("raw", "aligned") if align else ("raw",)):
            Tu = T if tag == "raw" else align_pose(sp, T, K, v.width, v.height, gt, world.sh_degree, At, iters=align_iters,
                                                   sky=sky)
            with torch.no_grad():
                rgb, ed, alpha, _ = render(sp, Tu[None], K[None], v.width, v.height, world.sh_degree)
                from cross_world.sky import composite
                rgb = composite(rgb, alpha, sky, Tu[None], K[None], v.width, v.height,
                                At[None] if At is not None else None).clamp(0, 1)
                if sky_m is not None and sky_m.any():
                    row[f"sky_alpha_{tag}"] = float(alpha[0, ..., 0][sky_m].mean())
                row[f"psnr_{tag}"] = psnr(rgb, gt[None])
                row[f"ssim_{tag}"] = float(ssim(rgb, gt[None]))
                row[f"lpips_{tag}"] = lpips(rgb, gt[None])
                if common_width and v.width > common_width:
                    # the same metrics after area-downsampling render and photo to a common width: models trained /
                    # evaluated at different resolutions compare here (blur costs more at a higher resolution)
                    hh = int(round(v.height * common_width / v.width))
                    ds = lambda x: torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(hh, common_width),
                                                                    mode="area").permute(0, 2, 3, 1)
                    r2, g2 = ds(rgb), ds(gt[None])
                    row[f"psnr_{tag}_w{common_width}"] = psnr(r2, g2)
                    row[f"ssim_{tag}_w{common_width}"] = float(ssim(r2, g2))
                    row[f"lpips_{tag}_w{common_width}"] = lpips(r2, g2)
                if v.has_depth:
                    dg = torch.from_numpy(v.depth()).to(device)
                    m = dg > 0
                    if m.any():
                        row[f"depth_absrel_{tag}"] = float(((ed[0, ..., 0] - dg).abs() / dg)[m].mean())
                        # floaters as the viewer sees them: share of the pixels with depth where the rendered surface
                        # is more than 15 % in front of the observed one
                        row[f"floater_px_{tag}"] = float(((ed[0, ..., 0] < 0.85 * dg) & (alpha[0, ..., 0] > 0.5))[m].float().mean())
            if save_dir is not None and n < save_max and tag == ("aligned" if align else "raw"):
                import cv2
                save_dir.mkdir(parents=True, exist_ok=True)
                im = np.concatenate([(gt.cpu().numpy() * 255).astype(np.uint8), (rgb[0].cpu().numpy() * 255).astype(np.uint8)], 0)
                cv2.imwrite(str(save_dir / f"view_{v.id}.jpg"), im[..., ::-1])
        rows.append(row)
    keys = sorted({k for r in rows for k in r if k.startswith(("psnr", "ssim", "lpips", "depth", "floater", "sky"))},
                  key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999) if rows else []
    summary = {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys
               if any(r.get(k) is not None for r in rows)}
    summary["n"] = len(rows)
    summary["eval_s"] = round(time.time() - t0, 1)
    log("eval: " + "  ".join(f"{k} {v:.4f}" for k, v in summary.items() if isinstance(v, float) and "_w" not in k))
    if any("_w" in k for k in summary):
        log(f"eval @{common_width}px: " + "  ".join(f"{k.replace(f'_w{common_width}', '')} {v:.4f}" for k, v in summary.items() if "_w" in k))
    return {"summary": summary, "rows": rows}
