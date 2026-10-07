"""Training windows: sequential, cross-session and with grouped (optionally appearance-mined) negatives.

A window is a list of frames (scene, sequence, index).  Three kinds, drawn per window:

* seq    : frames of one sequence within a random temporal span (`stride` = mean frame gap), any order.
* cross  : CROSS's relocalization / loop input: a query frame (+ a temporal neighbour) of one session and frames of the
           other sessions of the same place whose cameras lie within `cross_radius` (x median scene depth) of the query
           and face within `cross_max_rot` degrees of it.  With probability `p_wrong_place` the references all come
           from ONE other place of the other sessions (a wrong retrieval, all negatives).
* negatives (any window, probability `p_neg`): 1..neg_max frames (never frame 0) are replaced by GROUPS of
           `neg_group` frames of one other place: the same sequence far away, another session of the scene, or another
           scene (`neg_types`); with `p_hard_neg` the group's anchor is among the `hard_neg_topk` candidates whose
           global descriptor (vpr.npy) is closest to frame 0's.  Labels always come from the GT geometry
           (geometry.gt_covisibility), so the sampler only proposes likely non-overlapping frames.

Lessons carried over from the underwater fine-tune (cross-uw memo sec. 7): isolated negatives teach an "isolated frame"
shortcut, so negatives come in groups; the pose of a frame sharing no scene with frame 0 is unobservable and is not
supervised (the trainer masks it); multi-session windows should be frequent.

`TrainStream` is an IterableDataset that yields whole batches: frames per window S, the window aspect ratio and the
number of windows B = images_per_gpu // S are drawn per batch (VGGT-Omega: S uniform in [1, 24], area ~512^2).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import IterableDataset, get_worker_info

from .scene import Scene, load_scene, list_scenes
from .transforms import augment_frame, centred_crop_resize, target_shape

DEFAULTS = dict(
    weight=1.0, stride=[1, 4], p_cross=0.0, cross_radius=[0.0, 0.5], cross_max_rot=60.0, cross_query_neighbour=0.5,
    p_wrong_place=0.2, p_neg=0.3, neg_max=3, neg_group=[2, 3], neg_dist=[[2.0, 6.0], [6.0, 40.0]],
    neg_types={"same_seq": 0.4, "same_scene": 0.4, "other_scene": 0.2}, p_hard_neg=0.5, hard_neg_topk=5,
    zoom=[0.75, 1.0], shuffle_p=0.5, include=None, exclude=None, max_scenes=None,
)


class DatasetPool:
    """The scenes of one dataset in the mix, with its sampling parameters (DEFAULTS overridden by the config)."""

    def __init__(self, root: str, name: str, cfg: dict, seed: int = 0):
        self.name = name
        self.cfg = {**DEFAULTS, **cfg}
        dirs = list_scenes(root, cfg.get("dataset", name), self.cfg["include"], self.cfg["exclude"],
                           self.cfg.get("val_mod", 0), self.cfg.get("split", "train"))
        if self.cfg["max_scenes"] and len(dirs) > self.cfg["max_scenes"]:
            r = np.random.default_rng(seed)
            dirs = [dirs[i] for i in sorted(r.choice(len(dirs), self.cfg["max_scenes"], replace=False))]
        self.scenes: list[Scene] = [load_scene(d) for d in dirs]
        if not self.scenes:
            raise RuntimeError(f"dataset {name}: no scenes under {root}")
        if self.cfg.get("pseudo_metric"):
            self._apply_pseudo_metric(root, cfg.get("dataset", name))
        w = np.array([s.n_frames for s in self.scenes], float) ** self.cfg.get("scene_weight_pow", 0.5)
        self.scene_p = w / w.sum()
        # places with several sessions (cross-session windows): scenes grouped by world
        by_world: dict[int, list[tuple[Scene, str]]] = {}
        for s in self.scenes:
            for q in s.sequences:
                by_world.setdefault(s.world_id, []).append((s, q))
        self.multi = {k: v for k, v in by_world.items() if len(v) >= 2}
        self.multi_keys = list(self.multi)
        self._trees: dict = {}
        self._dmed: dict = {}

    def _apply_pseudo_metric(self, root, dataset):
        """Make the scenes of a non-metric dataset metric with per-scene factors from vggt_ft/prep/pseudo_metric.py.
        cfg: pseudo_metric (path of the json, or true = <root>/<dataset>/pseudo_metric.json), pseudo_labeler (key of
        the factor, default "ens"), pseudo_max_spread (drop a scene whose per-frame ratios have q75 / q25 above it),
        pseudo_max_disagree (drop a scene where some labeler's factor differs from the chosen one by more than this
        factor, e.g. 1.25)."""
        path = self.cfg["pseudo_metric"]
        path = Path(root) / dataset / "pseudo_metric.json" if path is True else Path(path)
        table = json.load(open(path))
        key, max_spread = self.cfg.get("pseudo_labeler", "ens"), self.cfg.get("pseudo_max_spread", 0)
        for sc in self.scenes:
            r = table.get(sc.dir.name)
            if sc.metric or r is None or key not in r["k"]:
                continue
            q25, q75 = r["iqr"][key]
            if max_spread and q75 / max(q25, 1e-9) > max_spread:
                continue
            max_dis = self.cfg.get("pseudo_max_disagree", 0)
            if max_dis and max(abs(math.log(v / r["k"][key])) for v in r["k"].values()) > math.log(max_dis):
                continue
            sc.scale, sc.metric, sc.pseudo_metric = float(r["k"][key]), True, True

    # ------------------------------------------------------------------ helpers
    def tree(self, seq):
        k = (seq.scene.dir, seq.name)
        if k not in self._trees:
            self._trees[k] = cKDTree(seq.centres)
        return self._trees[k]

    def depth_unit(self, seq) -> float:
        """Typical scene depth of a sequence (median of the per-frame median depths, `dmed` in frames.npz)."""
        k = (seq.scene.dir, seq.name)
        if k not in self._dmed:
            z = np.load(seq.dir / "frames.npz")
            self._dmed[k] = (float(np.median(z["dmed"])) if "dmed" in z else 2.0) * seq.scene.scale
        return self._dmed[k]


def _rot_angle(R1, R2):
    c = (np.trace(R1 @ R2.T) - 1) / 2
    return math.degrees(math.acos(float(np.clip(c, -1, 1))))


class WindowSampler:
    def __init__(self, pools: list[DatasetPool]):
        self.pools = pools
        w = np.array([p.cfg["weight"] for p in pools], float)
        self.pool_p = w / w.sum()

    # ------------------------------------------------------------------ window kinds
    def _seq_frames(self, rng, pool, S):
        st = pool.cfg
        scene = pool.scenes[rng.choice(len(pool.scenes), p=pool.scene_p)]
        seq = scene.seq(scene.sequences[rng.integers(len(scene.sequences))])
        lo, hi = st["stride"]
        gap = rng.uniform(lo, hi)
        span = max(S - 1, int(round(gap * (S - 1))))
        if seq.n <= span:
            span = seq.n - 1
        if seq.n < 2:
            return None
        start = int(rng.integers(0, seq.n - span)) if seq.n > span else 0
        pool_idx = np.arange(start, start + span + 1)
        k = min(S, len(pool_idx))
        idx = np.sort(rng.choice(pool_idx, k, replace=False))
        frames = [(scene, seq, int(i)) for i in idx]
        if len(frames) < S:        # short sequence: repeat-free fill from the rest of the sequence
            rest = np.setdiff1d(np.arange(seq.n), idx)
            if len(rest) < S - len(frames):
                return None
            frames += [(scene, seq, int(i)) for i in rng.choice(rest, S - len(frames), replace=False)]
        return frames

    def _cross_frames(self, rng, pool, S):
        """query (+ neighbour) of one session, references from the other sessions of the same place."""
        st = pool.cfg
        if not pool.multi_keys:
            return None
        members = pool.multi[pool.multi_keys[rng.integers(len(pool.multi_keys))]]
        qs, qn = members[rng.integers(len(members))]
        seq = qs.seq(qn)
        q = int(rng.integers(seq.n))
        unit = pool.depth_unit(seq)
        frames = [(qs, seq, q)]
        if rng.random() < st["cross_query_neighbour"]:
            nb = q + int(rng.choice([-3, -2, -1, 1, 2, 3])) * max(1, int(round(np.mean(st["stride"]))))
            if 0 <= nb < seq.n:
                frames.append((qs, seq, nb))
        others = [(s, n) for s, n in members if not (s is qs and n == qn)]
        n_map = S - len(frames)
        if n_map < 1:
            return frames[:S], []
        Rq = seq.E[q, :3, :3]
        wrong = rng.random() < st["p_wrong_place"]
        centre = seq.centres[q]
        if wrong:      # one wrong place of the other sessions: a coherent group far from the query
            lo, hi = st["neg_dist"][rng.integers(len(st["neg_dist"]))]
            os_, on = others[rng.integers(len(others))]
            oseq = os_.seq(on)
            d = np.linalg.norm(oseq.centres - centre, axis=1) / unit
            cand = np.flatnonzero((d > lo) & (d < hi))
            if not len(cand):
                return None
            a = int(rng.choice(cand))
            centre, Rq = oseq.centres[a], oseq.E[a, :3, :3]
        r = rng.uniform(*st["cross_radius"]) if st["cross_radius"][1] > st["cross_radius"][0] else st["cross_radius"][1]
        for radius in (max(r, 0.05), st["cross_radius"][1], 2 * st["cross_radius"][1]):
            cand = []
            for os_, on in others:
                oseq = os_.seq(on)
                for a in pool.tree(oseq).query_ball_point(centre, radius * unit):
                    if _rot_angle(Rq, oseq.E[a, :3, :3]) <= st["cross_max_rot"]:
                        cand.append((os_, oseq, int(a)))
            if len(cand) >= n_map:
                break
        if len(cand) < n_map:
            return None
        n_q = len(frames)
        frames += [cand[k] for k in rng.choice(len(cand), n_map, replace=False)]
        negs = list(range(n_q, S)) if wrong else []
        return frames, negs

    def _neg_anchor(self, rng, pool, keep, lo, hi):
        st = pool.cfg
        types, probs = zip(*st["neg_types"].items())
        kind = types[rng.choice(len(types), p=np.array(probs) / sum(probs))]
        s0, q0, i0 = keep[0]
        if kind == "same_scene" and len(s0.sequences) > 1:
            sc, sq = s0, s0.seq(s0.sequences[rng.integers(len(s0.sequences))])
        elif kind == "same_scene" and pool.multi.get(s0.world_id):
            m = pool.multi[s0.world_id]
            s_, n_ = m[rng.integers(len(m))]
            sc, sq = s_, s_.seq(n_)
        elif kind == "other_scene" and len(pool.scenes) > 1:
            sc = pool.scenes[rng.integers(len(pool.scenes))]
            if sc.world_id == s0.world_id:
                return None
            sq = sc.seq(sc.sequences[rng.integers(len(sc.sequences))])
        else:
            sc, sq = s0, q0
        cand = np.arange(sq.n)
        same = [(s, q, i) for s, q, i in keep if s.world_id == sc.world_id]
        if same:
            unit = pool.depth_unit(q0)
            ref = np.stack([q.centres[i] for _, q, i in same])
            dmin = np.linalg.norm(sq.centres[:, None] - ref[None], axis=-1).min(1) / unit
            cand = cand[(dmin > lo) & (dmin < hi)]
        if not len(cand):
            return None
        v0, vn = q0.vpr, sq.vpr
        if v0 is not None and vn is not None and rng.random() < st["p_hard_neg"]:
            sim = np.asarray(vn[cand], np.float32) @ np.asarray(v0[i0], np.float32)
            top = cand[np.argsort(-sim)[: int(st["hard_neg_topk"])]]
            return sc, sq, int(rng.choice(top))
        return sc, sq, int(rng.choice(cand))

    def _add_negatives(self, rng, pool, frames):
        st = pool.cfg
        n = int(rng.integers(1, min(st["neg_max"], len(frames) - 2) + 1))
        slots = [int(x) for x in rng.choice(np.arange(1, len(frames)), n, replace=False)]
        keep = [f for k, f in enumerate(frames) if k not in set(slots)]
        frames = list(frames)
        done = []
        glo, ghi = st["neg_group"]
        stride = max(1, int(round(np.mean(st["stride"]))))
        while slots:
            lo, hi = st["neg_dist"][rng.integers(len(st["neg_dist"]))]
            a = self._neg_anchor(rng, pool, keep, lo, hi)
            if a is None:
                break
            sc, sq, i = a
            k = min(len(slots), int(rng.integers(glo, ghi + 1)))
            members = [i] + [i + d * stride for d in rng.permutation([-2, -1, 1, 2]) if 0 <= i + d * stride < sq.n]
            for j in members[:k]:
                slot = slots.pop(0)
                frames[slot] = (sc, sq, int(j))
                done.append(slot)
        return frames, done

    def sample(self, rng, S: int):
        """(pool, frames, neg slots, kind)."""
        for _ in range(50):
            pool = self.pools[rng.choice(len(self.pools), p=self.pool_p)]
            st = pool.cfg
            kind, negs, frames = "seq", [], None
            if S >= 2 and pool.multi_keys and rng.random() < st["p_cross"]:
                r = self._cross_frames(rng, pool, S)
                if r is not None:
                    frames, negs = r
                    kind = "cross"
            if frames is None:
                frames = self._seq_frames(rng, pool, S)
                if frames is None:
                    continue
            if len(frames) != S:
                continue
            if S >= 3 and rng.random() < st["p_neg"] and len(negs) < S - 2:
                frames, extra = self._add_negatives(rng, pool, frames)
                negs = sorted(set(negs) | set(extra))
            if rng.random() < st["shuffle_p"]:
                perm = rng.permutation(S)
                frames = [frames[p] for p in perm]
                negs = [int(np.flatnonzero(perm == k)[0]) for k in negs]
            return pool, frames, negs, kind
        raise RuntimeError("could not sample a window")


def materialise(frames, out_hw, rng=None, aug=None, zoom=(1.0, 1.0)):
    """Load and transform the frames of a window.  Extrinsics are camera-from-world with the world re-centred on frame
    0's camera centre (frames of other worlds on their own first frame), in float64 until the end."""
    imgs, deps, Es, Ks, wids = [], [], [], [], []
    z = float(rng.uniform(*zoom)) if rng is not None else zoom[1]
    origin = {}
    for sc, sq, i in frames:
        img, dep, K, E = sq.load(i)
        img, dep, K = centred_crop_resize(img, dep, K, out_hw, z)
        c = origin.setdefault(sc.world_id, sq.centres[i].copy())
        E = E.copy()
        E[:3, 3] = E[:3, 3] + E[:3, :3] @ c            # x_world = x_world' + c
        imgs.append(augment_frame(img, rng, aug) if rng is not None else img.astype(np.float32) / 255.0)
        deps.append(dep)
        Es.append(E)
        Ks.append(K)
        wids.append(sc.world_id)
    depths = torch.from_numpy(np.stack(deps))
    return {
        "images": torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous(),
        "depths": depths,
        "masks": depths > 0,
        "extrinsics": torch.from_numpy(np.stack(Es)).float(),
        "intrinsics": torch.from_numpy(np.stack(Ks)).float(),
        "world_id": torch.tensor(wids, dtype=torch.long),
        "metric": torch.tensor(all(sc.metric for sc, _, _ in frames)),
        "pseudo_metric": torch.tensor(any(sc.pseudo_metric for sc, _, _ in frames)),
        "posed": torch.tensor(all(sc.posed for sc, _, _ in frames)),
    }


def collate(items):
    out = {}
    for k in items[0]:
        if isinstance(items[0][k], torch.Tensor):
            out[k] = torch.stack([it[k] for it in items])
        else:
            out[k] = [it[k] for it in items]
    return out


class TrainStream(IterableDataset):
    """Endless stream of batches for one rank.  cfg keys: root, datasets {name: pool cfg}, frames [lo, hi],
    images_per_gpu, aspect [lo, hi] (H / W), area, aug."""

    def __init__(self, cfg: dict, rank: int = 0, seed: int = 0):
        self.cfg, self.rank, self.seed = cfg, rank, seed
        self.sampler = None

    def _build(self):
        pools = []
        for n, c in self.cfg["datasets"].items():
            if c.get("weight", 1.0) <= 0:
                continue
            try:
                base = {"val_mod": self.cfg.get("val_mod", 0), "zoom": self.cfg.get("zoom", DEFAULTS["zoom"])}
                pools.append(DatasetPool(self.cfg["root"], n, {**base, **c}, self.seed))
            except (RuntimeError, FileNotFoundError) as e:      # not converted yet: train on the others
                print(f"[data] skip dataset {n}: {e}", flush=True)
        self.sampler = WindowSampler(pools)

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        rng = np.random.default_rng([self.seed, self.rank, wid])
        if self.sampler is None:
            self._build()
        cfg = self.cfg
        while True:
            S = int(rng.integers(cfg["frames"][0], cfg["frames"][1] + 1))
            B = max(1, cfg["images_per_gpu"] // S)
            hw = target_shape(float(rng.uniform(*cfg["aspect"])), cfg.get("area", 512 * 512))
            items = []
            while len(items) < B:
                try:
                    pool, frames, negs, kind = self.sampler.sample(rng, S)
                    it = materialise(frames, hw, rng, cfg.get("aug"), pool.cfg["zoom"])
                except (OSError, ValueError) as e:      # incl. transient read errors of the shared file system
                    print(f"[data] skip window: {e}", flush=True)
                    continue
                it["is_neg"] = torch.tensor([k in negs for k in range(S)], dtype=torch.bool)
                it["same_session"] = torch.tensor([f[1] is frames[0][1] for f in frames], dtype=torch.bool)
                it["dataset"] = pool.name
                it["kind"] = kind
                items.append(it)
            yield collate(items)
