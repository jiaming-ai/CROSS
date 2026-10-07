"""Descriptor and location indexes of the keyframe database, for maps of 10^3 to 10^6+ keyframes.

DescriptorIndex holds the keyframes' place-recognition descriptors (one row per permanent keyframe) and searches them.
Without a projection it stores the full BoQ descriptors in float32 and scores them exactly as the original database
did (bit-identical).  With a projection (PCA, optionally whitened, of the 16384-d BoQ descriptor) the rows are short
float16 codes: 1 KB per keyframe at 512 dimensions instead of 64 KB, so a million keyframes fit in 1 GB and an exact
search over them takes about a millisecond on a GPU.  An inverted-file (IVF) backend restricts the search to the
`nprobe` nearest of `nlist` k-means cells, for CPU-only machines or databases beyond ~10^7 rows.

SpatialIndex holds keyframe positions (map frame) in a KD-tree for radius queries: the candidates near the predicted
positions of the belief's hypotheses (or near a GPS fix) that locality-aware retrieval scores before the global list.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------------------------------------------------------
class PCAProjection:
    """x -> ((x - mean) @ components) * scale, optionally L2-normalized: a linear projection of the L2-normalized
    descriptors to `dim` dimensions.

    The default (`fit(center=False, normalize=False)`: the top right-singular vectors of the raw descriptors, codes
    not renormalized) keeps the score scale: the inner product of two codes approximates the cosine of the full
    descriptors (from below, by the residual-residual term: mean |error| 0.02 at 512 dims over the 50 best matches),
    which CROSS uses as weights, measurement noise and the new-keyframe test.  Centring and renormalizing (plain PCA)
    or whitening keep the ranking but move the scores (0.1-0.3 at 512 dims), so the score thresholds would need
    recalibration."""

    def __init__(self, mean: np.ndarray, components: np.ndarray, scale: Optional[np.ndarray] = None,
                 meta: Optional[dict] = None, normalize: bool = False):
        self.mean = np.asarray(mean, dtype=np.float32)                 # (D,) (zeros: uncentred)
        self.components = np.asarray(components, dtype=np.float32)     # (D, d)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)   # (d,)
        self.normalize = bool(normalize)
        self.meta = dict(meta or {})
        self._dev: Dict[str, tuple] = {}

    @property
    def dim_in(self) -> int:
        return int(self.components.shape[0])

    @property
    def dim(self) -> int:
        return int(self.components.shape[1])

    @classmethod
    def fit(cls, X: torch.Tensor, dim: int, center: bool = False, normalize: bool = False, whiten_power: float = 0.0,
            niter: int = 6, eps: float = 1e-6, meta: Optional[dict] = None) -> "PCAProjection":
        """Top-`dim` right-singular vectors of X (N, D) (centred first if `center`), by randomized SVD on X's device."""
        X = X.float()
        mean = X.mean(0) if center else torch.zeros(X.shape[1], device=X.device)
        Xc = X - mean
        q = min(dim + 32, min(Xc.shape))
        torch.manual_seed(0)
        _, S, V = torch.svd_lowrank(Xc, q=q, niter=niter)
        V = V[:, :dim]
        eig = (S[:dim] ** 2) / max(X.shape[0] - 1, 1)
        scale = None
        if whiten_power > 0:
            scale = (eig + eps).pow(-whiten_power)
            scale = scale / scale.max()
        m = dict(meta or {})
        total = float((Xc ** 2).sum() / max(X.shape[0] - 1, 1))
        m.update(n_fit=int(X.shape[0]), whiten_power=float(whiten_power), center=bool(center),
                 explained=float(eig.sum() / max(total, 1e-12)))
        return cls(mean.cpu().numpy(), V.cpu().numpy(), None if scale is None else scale.cpu().numpy(), m,
                   normalize=normalize)

    def _on(self, device) -> tuple:
        key = str(device)
        if key not in self._dev:
            mean = torch.from_numpy(self.mean).to(device)
            comp = torch.from_numpy(self.components).to(device)
            if self.scale is not None:
                comp = comp * torch.from_numpy(self.scale).to(device)[None]
            self._dev[key] = (mean, comp)
        return self._dev[key]

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """(D,) or (B, D) -> (d,) or (B, d), float32 (unit norm only if `normalize`)."""
        mean, comp = self._on(x.device)
        y = (x.float() - mean) @ comp
        return torch.nn.functional.normalize(y, dim=-1) if self.normalize else y

    def state(self) -> dict:
        return {"type": "pca", "mean": self.mean, "components": self.components.astype(np.float16),
                "scale": self.scale, "meta": self.meta, "normalize": self.normalize}

    @classmethod
    def from_state(cls, s: dict) -> "PCAProjection":
        return cls(s["mean"], np.asarray(s["components"], dtype=np.float32), s.get("scale"), s.get("meta"),
                   normalize=bool(s.get("normalize", False)))

    def save(self, path: str) -> None:
        s = self.state()
        np.savez(path, mean=s["mean"], components=s["components"],
                 scale=s["scale"] if s["scale"] is not None else np.zeros(0, np.float32),
                 normalize=np.array(self.normalize), meta=np.array(repr(s["meta"])))

    @classmethod
    def load(cls, path: str) -> "PCAProjection":
        z = np.load(path, allow_pickle=False)
        scale = z["scale"] if z["scale"].size else None
        try:
            import ast
            meta = ast.literal_eval(str(z["meta"]))
        except Exception:
            meta = {}
        meta["file"] = str(path)
        return cls(z["mean"], z["components"].astype(np.float32), scale, meta,
                   normalize=bool(z["normalize"]) if "normalize" in z.files else False)


def projection_from_config(spec, dim_in: int) -> Optional[PCAProjection]:
    """None (no projection) or a PCAProjection loaded from a .npz path."""
    if spec in (None, "", "none", False):
        return None
    p = PCAProjection.load(str(spec))
    if p.dim_in != dim_in:
        raise ValueError(f"descriptor projection {spec} expects {p.dim_in}-d descriptors, the VPR model gives {dim_in}")
    return p


# --------------------------------------------------------------------------------------------------------------------
# Descriptor index
# --------------------------------------------------------------------------------------------------------------------
class DescriptorIndex:
    """Rows of keyframe descriptors with their keyframe ids; exact or IVF search.

    Rows are dense (0..n-1); removal moves the last row into the hole (the database mirrors it).  Without a
    projection the buffer is the original float32 embedding buffer (same growth, same matmul), so scores are
    bit-identical to the database before this class existed.

    Map-fitted projection (`fit_at` > 0, no fixed projection): the full descriptors are kept until the database
    holds `fit_at` rows; then an uncentred projection to `fit_dim` dimensions is fitted on them (a projection fitted on
    the map's own descriptors keeps recall and scores: 512 dims, mean |score error| 0.003-0.005 on ROVER / OpenLORIS /
    SimChange cross-session queries; one fitted on other data does not), every row is re-encoded and the full
    descriptors are dropped.  The explained energy |V^T x|^2 of each new descriptor is tracked; when its running mean
    falls `extend_margin` below the level at the fit (the robot entered another kind of place, e.g. outdoors after
    indoors), the subspace is extended by the main directions of the residuals of the `recent` last descriptors
    (whose full descriptors are kept): older rows get zeros in the new dimensions, up to `max_dim`."""

    def __init__(self, dim_in: int, device: str = "cuda", initial_capacity: int = 1000,
                 projection: Optional[PCAProjection] = None, store_dtype: str = "auto", backend: str = "exact",
                 ivf_nlist: int = 0, ivf_nprobe: int = 16, ivf_min_rows: int = 200000, ann_shortlist: int = 256,
                 fit_at: int = 0, fit_dim: int = 512, extend: bool = True, extend_margin: float = 0.05,
                 extend_dims: int = 64, max_dim: int = 1024, recent: int = 1024):
        self.dim_in = int(dim_in)
        self.device = device
        self.projection = projection
        self.dim = projection.dim if projection is not None else self.dim_in
        self._store_dtype = store_dtype
        if store_dtype == "auto":
            store_dtype = "float16" if projection is not None else "float32"
        self.dtype = getattr(torch, store_dtype)
        self.backend = backend
        self.ivf_nlist, self.ivf_nprobe, self.ivf_min_rows = int(ivf_nlist), int(ivf_nprobe), int(ivf_min_rows)
        self.ann_shortlist = int(ann_shortlist)
        self.fit_at, self.fit_dim = int(fit_at), int(fit_dim)
        self.extend, self.extend_margin, self.extend_dims = bool(extend), float(extend_margin), int(extend_dims)
        self.max_dim, self.recent_size = int(max_dim), int(recent)
        self._recent: list = []          # (kf_id, full descriptor fp16) of the latest rows, after a map fit
        self._e_ema = None
        self._since_extend = 0
        self.buf = torch.zeros((int(initial_capacity), self.dim), device=device, dtype=self.dtype)
        self.ids = torch.full((int(initial_capacity),), -1, device=device, dtype=torch.long)
        self.n = 0
        self._ivf: Optional[_IVF] = None

    @property
    def map_fitted(self) -> bool:
        return self.projection is not None and bool(self.projection.meta.get("map_fit"))

    # -------------------------------------------------------------- storage
    def encode(self, desc: torch.Tensor) -> torch.Tensor:
        """VPR descriptor(s) -> stored code(s) (float32 compute; the query side of a search)."""
        if self.projection is None:
            return desc
        return self.projection.apply(desc.to(self.device))

    def reserve(self, min_size: int) -> None:
        """Grow the buffer to at least min_size rows (doubling), as the original database did."""
        if min_size <= self.buf.shape[0]:
            return
        new_size = max(min_size, self.buf.shape[0] * 2)
        nb = torch.zeros((new_size, self.dim), device=self.device, dtype=self.dtype)
        ni = torch.full((new_size,), -1, device=self.device, dtype=torch.long)
        n = min(self.n, self.buf.shape[0])
        nb[:n] = self.buf[:n]
        ni[:n] = self.ids[:n]
        self.buf, self.ids = nb, ni

    def add(self, desc: torch.Tensor, kf_id: int) -> int:
        """Append one descriptor (full VPR descriptor) for keyframe kf_id; returns its row."""
        self.reserve(self.n + 100)
        row = self.n
        code = self.encode(desc)
        self.buf[row] = code.to(self.dtype)
        self.ids[row] = int(kf_id)
        self.n += 1
        if self.map_fitted:
            self._track(desc, code, kf_id)
        elif self.projection is None and self.fit_at > 0 and self.n >= self.fit_at:
            self.fit_projection()
        if self._ivf is not None:
            self._ivf.add(self.buf[row:row + 1].float(), row)
        elif self.backend == "ivf" and self.n >= self.ivf_min_rows:
            self._train_ivf()
        return row

    def fit_projection(self) -> None:
        """Fit the map projection on the stored full descriptors, re-encode every row (see the class docstring)."""
        X = self.buf[:self.n].float()
        g = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(self.n, generator=g).to(X.device)
        n_hold = max(1, self.n // 10)
        dim = min(self.fit_dim, self.n - n_hold - 1)
        proj = PCAProjection.fit(X[perm[n_hold:]], dim, meta={"map_fit": True})
        held = proj.apply(X[perm[:n_hold]])
        proj.meta["e_ref"] = float((held.pow(2).sum(1) / X[perm[:n_hold]].pow(2).sum(1)).mean())
        proj.meta["fit_rows"] = int(self.n)
        proj.meta["extensions"] = 0
        recent = [(int(self.ids[r]), X[r].half()) for r in range(max(0, self.n - self.recent_size), self.n)]
        codes = proj.apply(X)
        self._set_projection(proj, codes)
        self._recent = recent
        self._e_ema = proj.meta["e_ref"]
        self._since_extend = 0
        logger.info(f"descriptor index: map projection {self.dim_in} -> {dim} fitted on {self.n} keyframes "
                    f"(held-out explained energy {proj.meta['e_ref']:.3f}); {self.n * self.dim_in * 4 / 1e6:.0f} MB -> "
                    f"{self.n * dim * 2 / 1e6:.1f} MB")

    def _set_projection(self, proj: PCAProjection, codes: torch.Tensor) -> None:
        self.projection = proj
        self.dim = proj.dim
        self.dtype = torch.float16 if self._store_dtype == "auto" else getattr(torch, self._store_dtype)
        nb = torch.zeros((self.buf.shape[0], self.dim), device=self.device, dtype=self.dtype)
        nb[:codes.shape[0]] = codes.to(self.dtype)
        self.buf = nb
        if self._ivf is not None:
            self._train_ivf()

    def _track(self, desc: torch.Tensor, code: torch.Tensor, kf_id: int) -> None:
        """Explained energy of a new descriptor; extend the subspace when the scene has changed."""
        x = desc.to(self.device).float()
        e = float(code.float().pow(2).sum() / x.pow(2).sum().clamp_min(1e-12))
        a = 1.0 / 200
        self._e_ema = e if self._e_ema is None else (1 - a) * self._e_ema + a * e
        self._recent.append((int(kf_id), x.half()))
        if len(self._recent) > self.recent_size:
            self._recent.pop(0)
        self._since_extend += 1
        e_ref = self.projection.meta.get("e_ref", 1.0)
        if (self.extend and self.dim < self.max_dim and self._e_ema < e_ref - self.extend_margin
                and self._since_extend >= max(2 * self.extend_dims, 200) and len(self._recent) >= 2 * self.extend_dims):
            self._extend_subspace()

    def _extend_subspace(self) -> None:
        V = torch.from_numpy(self.projection.components).to(self.device)
        Xr = torch.stack([x for _, x in self._recent]).float()
        R = Xr - (Xr @ V) @ V.T
        a = min(self.extend_dims, self.max_dim - self.dim)
        _, _, W = torch.svd_lowrank(R, q=min(a + 16, min(R.shape)), niter=6)
        W = W[:, :a]
        W = W - V @ (V.T @ W)                       # keep the basis orthonormal
        W = torch.linalg.qr(W).Q
        Vn = torch.cat([V, W], 1)
        meta = dict(self.projection.meta)
        proj = PCAProjection(np.zeros(self.dim_in, np.float32), Vn.cpu().numpy(), None, meta)
        codes = torch.zeros((self.n, Vn.shape[1]), device=self.device)
        codes[:, :self.dim] = self.buf[:self.n].float()
        rows = {int(k): i for i, k in enumerate(self.ids[:self.n].tolist())}
        for kf, x in self._recent:
            if kf in rows:
                codes[rows[kf]] = proj.apply(x.float())
        held = proj.apply(Xr)
        proj.meta["e_ref"] = float((held.pow(2).sum(1) / Xr.pow(2).sum(1)).mean())
        proj.meta["extensions"] = int(meta.get("extensions", 0)) + 1
        logger.info(f"descriptor index: scene change (explained energy {self._e_ema:.3f} < {meta.get('e_ref', 1):.3f} - "
                    f"{self.extend_margin}); subspace {self.dim} -> {Vn.shape[1]} dims")
        self._set_projection(proj, codes)
        self._e_ema = proj.meta["e_ref"]
        self._since_extend = 0

    def remove_row(self, row: int) -> Optional[int]:
        """Remove `row`; the last row moves into it.  Returns the moved row's old index (None if row was last)."""
        last = self.n - 1
        if row == last:
            self.n -= 1
            if self._ivf is not None:
                self._ivf.drop_last(last)
            return None
        self.buf[row] = self.buf[last]
        self.ids[row] = self.ids[last]
        self.n -= 1
        if self._ivf is not None:
            self._ivf.move(last, row)
        return last

    def set_rows(self, codes: torch.Tensor, kf_ids: Sequence[int]) -> None:
        """Replace the content by stored codes (load).  Full descriptors of a map at least `fit_at` rows long are
        projected now when the index fits map projections."""
        n = int(codes.shape[0])
        self.n = 0
        self.reserve(n)
        self.n = n
        self.buf[:n] = codes.to(self.device, self.dtype)
        self.ids[:n] = torch.as_tensor(list(kf_ids), dtype=torch.long, device=self.device)
        self._ivf = None
        if self.projection is None and self.fit_at > 0 and n >= self.fit_at:
            self.fit_projection()
        if self.backend == "ivf" and n >= self.ivf_min_rows:
            self._train_ivf()

    def nbytes(self) -> int:
        return int(self.n * self.dim * self.buf.element_size())

    # -------------------------------------------------------------- search
    def scores(self, q: torch.Tensor, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inner products of the code q with all rows [0, n) or with `rows` (in that order)."""
        db = self.buf[:self.n] if rows is None else self.buf.index_select(0, torch.as_tensor(rows, device=self.buf.device))
        if self.dtype == torch.float32:
            return (db @ q.unsqueeze(-1)).squeeze(-1)          # the original database's computation
        if db.is_cuda:
            return (db @ q.to(self.dtype).unsqueeze(-1)).squeeze(-1).float()
        return (db.float() @ q.float().unsqueeze(-1)).squeeze(-1)

    def id_mask(self, max_kf_id: Optional[int] = None, min_kf_id: Optional[int] = None) -> torch.Tensor:
        ids = self.ids[:self.n]
        m = torch.ones(self.n, dtype=torch.bool, device=self.device)
        if max_kf_id is not None:
            m &= ids < int(max_kf_id)
        if min_kf_id is not None:
            m &= ids >= int(min_kf_id)
        return m

    def ann_rows(self, q: torch.Tensor) -> Optional[torch.Tensor]:
        """Rows of the probed IVF cells (sorted), or None when the index searches exactly."""
        if self._ivf is None:
            return None
        return self._ivf.probe(q.float(), self.ivf_nprobe, self.n)

    def _train_ivf(self) -> None:
        nlist = self.ivf_nlist or int(min(65536, max(64, 4 * math.sqrt(self.n))))
        self._ivf = _IVF.train(self.buf[:self.n].float(), nlist)
        logger.info(f"descriptor index: IVF with {nlist} cells over {self.n} rows")

    # -------------------------------------------------------------- persistence
    def state(self) -> dict:
        return {"projection": None if self.projection is None else self.projection.state(),
                "dtype": str(self.dtype).replace("torch.", ""), "dim": self.dim, "backend": self.backend}


class _IVF:
    """Inverted file: k-means centroids, the cell of every row, and the rows sorted by cell (contiguous lists, so a
    probe reads nprobe slices instead of scanning every row).  Rows added since the last sort are scanned apart and
    the lists are re-sorted once they exceed 10 % (or after a removal)."""

    def __init__(self, centroids: torch.Tensor):
        self.centroids = torch.nn.functional.normalize(centroids.float(), dim=-1)
        self.cell = torch.empty(0, dtype=torch.long, device=centroids.device)
        self._order = None          # rows sorted by cell (first _n_sorted rows)
        self._offsets = None        # (nlist + 1,) start of each cell's list in _order
        self._n_sorted = 0

    @classmethod
    def train(cls, X: torch.Tensor, nlist: int, iters: int = 12, sample: int = 256) -> "_IVF":
        g = torch.Generator(device="cpu").manual_seed(0)
        n = X.shape[0]
        take = torch.randperm(n, generator=g)[: min(n, nlist * sample)].to(X.device)
        S = X[take].float()
        C = S[torch.randperm(S.shape[0], generator=g)[:nlist].to(X.device)].clone()
        for _ in range(iters):     # spherical k-means
            C = torch.nn.functional.normalize(C, dim=-1)
            a = _argmax_chunked(S, C)
            Cn = torch.zeros_like(C).index_add_(0, a, S)
            cnt = torch.bincount(a, minlength=nlist)
            empty = cnt == 0
            Cn[empty] = S[torch.randint(0, S.shape[0], (int(empty.sum()),), generator=g).to(X.device)]
            C = Cn
        ivf = cls(C)
        ivf.cell = _argmax_chunked(X, ivf.centroids)
        return ivf

    def add(self, x: torch.Tensor, row: int) -> None:
        c = _argmax_chunked(x, self.centroids)
        if row >= self.cell.shape[0]:
            grow = torch.full((max(row + 1, 2 * self.cell.shape[0]) - self.cell.shape[0],), -1, dtype=torch.long,
                              device=self.cell.device)
            self.cell = torch.cat([self.cell, grow])
        self.cell[row] = c[0]

    def move(self, src: int, dst: int) -> None:
        self.cell[dst] = self.cell[src]
        self.cell[src] = -1
        self._order = None

    def drop_last(self, row: int) -> None:
        self.cell[row] = -1
        self._order = None

    def _sort(self, n: int) -> None:
        cells = self.cell[:n]
        self._order = torch.argsort(cells, stable=True)
        counts = torch.bincount(cells, minlength=self.centroids.shape[0])
        self._offsets = torch.zeros(counts.numel() + 1, dtype=torch.long, device=cells.device)
        self._offsets[1:] = torch.cumsum(counts, 0)
        self._offsets = self._offsets.cpu()
        self._n_sorted = n

    def probe(self, q: torch.Tensor, nprobe: int, n: int) -> torch.Tensor:
        if self._order is None or n < self._n_sorted or n - self._n_sorted > max(1000, n // 10):
            self._sort(n)
        cs = self.centroids @ q.to(self.centroids.dtype)
        top = cs.topk(min(nprobe, cs.shape[0])).indices
        off = self._offsets
        parts = [self._order[off[c]:off[c + 1]] for c in top.cpu().tolist()]
        if n > self._n_sorted:
            tail = torch.arange(self._n_sorted, n, device=self.cell.device)
            parts.append(tail[torch.isin(self.cell[self._n_sorted:n], top)])
        rows = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=self.cell.device)
        return rows.sort().values


def _argmax_chunked(X: torch.Tensor, C: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    out = []
    for i in range(0, X.shape[0], chunk):
        out.append((X[i:i + chunk] @ C.T).argmax(1))
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long, device=X.device)


# --------------------------------------------------------------------------------------------------------------------
# Spatial index
# --------------------------------------------------------------------------------------------------------------------
class SpatialIndex:
    """Positions of keyframes (rows of the descriptor index) with radius queries.

    A KD-tree over the positions at the last rebuild plus a list of rows added since, scanned exactly; the tree is
    rebuilt when `epoch` (incremented by whoever moves keyframe poses: PGO, merges) changes or when the pending list
    grows past `max(256, rebuild_fraction * n)`."""

    def __init__(self, rebuild_fraction: float = 0.1):
        self.rebuild_fraction = float(rebuild_fraction)
        self._tree = None
        self._tree_rows = np.zeros(0, dtype=np.int64)
        self._pending_rows: list = []
        self._pending_pos: list = []
        self.epoch = None
        self.n_rebuilds = 0

    def rebuild(self, rows: np.ndarray, positions: np.ndarray, epoch) -> None:
        from scipy.spatial import cKDTree
        rows = np.asarray(rows, dtype=np.int64)
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self._tree = cKDTree(positions) if len(rows) else None
        self._tree_rows = rows
        self._pending_rows, self._pending_pos = [], []
        self.epoch = epoch
        self.n_rebuilds += 1

    def add(self, row: int, position) -> None:
        self._pending_rows.append(int(row))
        self._pending_pos.append(np.asarray(position, dtype=np.float64).reshape(3))

    def needs_rebuild(self, epoch, n: int) -> bool:
        return (self.epoch != epoch or self._tree is None and n > 0
                or len(self._pending_rows) > max(256, self.rebuild_fraction * n))

    def remap_row(self, old: int, new: Optional[int]) -> None:
        """Row `old` moved to `new` (None: removed) in the descriptor index."""
        self._tree_rows = np.where(self._tree_rows == old, -1 if new is None else new, self._tree_rows)
        self._pending_rows = [(-1 if new is None else new) if r == old else r for r in self._pending_rows]

    def query(self, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Unique rows within radii[i] of centers[i] for any i (sorted)."""
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(centers),))
        found = []
        if self._tree is not None and len(centers):
            for c, r in zip(centers, radii):
                idx = self._tree.query_ball_point(c, r)
                if idx:
                    found.append(self._tree_rows[np.asarray(idx, dtype=np.int64)])
        if self._pending_rows and len(centers):
            P = np.stack(self._pending_pos)
            d = np.linalg.norm(P[None] - centers[:, None], axis=-1)            # (m, p)
            hit = (d <= radii[:, None]).any(0)
            if hit.any():
                found.append(np.asarray(self._pending_rows, dtype=np.int64)[hit])
        if not found:
            return np.zeros(0, dtype=np.int64)
        rows = np.unique(np.concatenate(found))
        return rows[rows >= 0]
