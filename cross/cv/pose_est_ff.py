"""Feed-forward relative pose estimation with stereo scale anchors.

`PoseEstFeedForward` replaces the classical detect/match/PnP relative pose module of
CROSS with one multi-view forward pass of a learned geometry model (VGGT-Omega or
Depth Anything 3).  All retrieved reference keyframes and the current frame are
placed in one context so that they share a single similarity gauge; the current
stereo pair (and optionally stored right images of the references, and the previous
frame connected by odometry) fixes the metric scale of that gauge.

Outputs follow the CROSS `PoseEst` contract: for each reference keyframe, the pose of
the current camera expressed in the reference camera frame (T_ref_cam), a validity
mask and a scalar confidence.  The confidence is a geometric covisibility score
computed from the predicted depth maps (fraction of reference pixels that reproject
into the current view with a consistent depth), which plays the role of the PnP
inlier count.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pypose as pp
import torch
import torch.nn.functional as F
from loguru import logger

from cross.core.config import FeedForwardConfig, FFBackend
from cross.cv.stereo_scale import ScaleAnchor, ScaleEstimate, estimate_scale, invert_poses, scale_camera_centers
from cross.utils.fingerprint import content_keys
from cross.utils.profile import timeit


@dataclass
class FFPrediction:
    """Raw output of one forward pass (all views)."""
    c2w: np.ndarray                     # (S, 4, 4) camera-to-world, model gauge
    K: np.ndarray                       # (S, 3, 3) intrinsics at processed resolution
    depth: torch.Tensor                 # (S, H, W) model-gauge depth
    depth_conf: torch.Tensor            # (S, H, W) or None
    hw: tuple
    covis: Optional[np.ndarray] = None  # (S, S) covisibility head (vggt_ft fine-tune), probabilities, if present
    log_scale: Optional[float] = None   # scale head: metric = exp(log_scale) x model gauge, if present


class _Backend:
    def infer(self, images: torch.Tensor, n_depth: Optional[int] = None, focal: Optional[float] = None) -> FFPrediction:
        """n_depth: depth is needed for the first n_depth views only (None: all)."""
        raise NotImplementedError


class _PatchEmbedCache(torch.nn.Module):
    """The aggregator's patch embedding (a DINO ViT-L, a third of the transformer blocks) with a cache.

    The embedding of an image does not depend on the other views of the forward pass, so the tokens of an image seen
    before (a keyframe that was the current view when it was observed, or a map keyframe retrieved in an earlier pass)
    are reused.  Images are identified by their content (`fingerprints`), so a cache entry can never be stale."""

    def __init__(self, embed: torch.nn.Module, capacity: int):
        super().__init__()
        self.embed = embed
        self.capacity = capacity
        self.store: OrderedDict = OrderedDict()
        self.keys: Optional[list] = None          # fingerprints of the views of the next call
        self.override: Optional[torch.Tensor] = None   # tokens to return as they are (the CUDA-graph pass)
        self.embed_missing = None                 # callable (normalized images) -> tokens; None: self.embed
        self.hits = self.misses = 0

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        if self.embed_missing is not None:
            return self.embed_missing(images)
        new = self.embed(images)
        return new["x_norm_patchtokens"] if isinstance(new, dict) else new

    def gather(self, images: torch.Tensor, keys: list, out: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Tokens of the normalized `images` (S, 3, H, W): cached ones reused, the others embedded and stored."""
        missing = list(OrderedDict.fromkeys(k for k in keys if k not in self.store))
        if missing:
            first = [keys.index(k) for k in missing]
            new = self._embed(images[first])
            for k, t in zip(missing, new):
                self.store[k] = t.clone() if self.embed_missing is not None else t   # graph outputs are reused
        self.hits += len(keys) - len(missing)
        self.misses += len(missing)
        tokens = torch.stack([self.store[k] for k in keys], out=out)
        for k in keys:
            self.store.move_to_end(k)
        while len(self.store) > self.capacity:
            self.store.popitem(last=False)
        return tokens

    def forward(self, images: torch.Tensor):
        if self.override is not None:
            return self.override
        keys, self.keys = self.keys, None
        if keys is None or self.capacity <= 0:
            return self.embed(images)
        return self.gather(images, keys)


class _VGGTOmegaBackend(_Backend):
    def __init__(self, checkpoint: str, device: str, half_weights: bool = True, token_cache: int = 0,
                 compile_blocks: bool = False, dense_head_bf16: bool = False,
                 compile_mode: str = "default", cuda_graphs: bool = False):
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.pose_enc import encoding_to_camera
        self._decode = encoding_to_camera
        self.device = device
        ckpt = Path(checkpoint)
        if not ckpt.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {ckpt}")
        with _no_weight_init():      # every parameter is overwritten by the checkpoint (checked below)
            self.model = VGGTOmega().eval()
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=False)
        head_cfg = sd.get("scale_head_cfg") if isinstance(sd, dict) else None
        sd = sd["model"] if "model" in sd and isinstance(sd["model"], dict) else sd
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"{ckpt.name} lacks {len(missing)} weights of the model, e.g. {missing[:3]}")
        # heads of the vggt_ft fine-tune (covisibility, metric scale): used when the checkpoint has them
        self.covis_head, self.scale_head = _load_ft_heads(sd, device, head_cfg)
        # the multi-layer and dense scale heads also read the DINO tokens and the aggregator layers the dense head keeps
        self._scale_multilayer = type(self.scale_head).__name__ in ("MultiLayerScaleHead", "DenseScaleHead")
        # LoRA scale encoder of the dense head (vggt_ft/scale_encoder.py): one extra DINO pass on the first view
        self.scale_encoder = _load_scale_encoder(sd, self.model.aggregator.patch_embed, head_cfg, device)
        unexpected = [u for u in unexpected if not u.startswith(("covis_head.", "scale_head.", "scale_encoder."))]
        if unexpected:      # e.g. the text-alignment head of vggt_omega_1b_256_text.pt, unused here
            logger.info(f"{ckpt.name}: ignored {len(unexpected)} weights of heads not in use")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if half_weights:
            # the aggregator already runs under autocast; storing its weights in bf16 halves memory
            self.model.aggregator.to(self.dtype)
        self.model = self.model.to(device)
        self.token_cache = None
        if token_cache > 0:
            self.token_cache = _PatchEmbedCache(self.model.aggregator.patch_embed, token_cache)
            self.model.aggregator.patch_embed = self.token_cache
        self.dense_head_bf16 = dense_head_bf16
        self._head_stream = None
        self.cuda_graphs = bool(cuda_graphs) and self.token_cache is not None
        self._graphs: dict = {}
        self._dino_graphs: dict = {}
        self._graph_pool = None
        if self.cuda_graphs:
            self.token_cache.embed_missing = self._embed_graphed
        self._compiled = []
        if compile_blocks:
            # torch.compile of every transformer block (frame, global and DINO blocks share one graph); dynamic shapes
            # cover any number of views.  The kernels are tuned once and cached on disk (TORCHINDUCTOR_CACHE_DIR).
            agg = self.model.aggregator
            embed = self.token_cache.embed if self.token_cache is not None else agg.patch_embed
            self._compiled = list(agg.frame_blocks) + list(agg.inter_frame_blocks) + list(getattr(embed, "blocks", []))
            for block in self._compiled:
                block.compile(dynamic=True, mode=None if compile_mode == "default" else compile_mode)

    def warmup(self, resolution: int):
        """Load / tune the compiled kernels now (a few seconds) rather than in the first observation.  The kernels
        are compiled for dynamic shapes, so one pass of any size serves all later ones."""
        if not self._compiled and not self.cuda_graphs:
            return
        t0 = time.perf_counter()
        h = max(16, int(round(resolution * 0.75 / 16)) * 16)
        for n in (4, 3):        # two sizes: the second call settles the dynamic-shape graphs
            self.infer(torch.rand(n, 3, h, resolution, device=self.device), n_depth=2)
        if self.token_cache is not None:
            self.token_cache.store.clear()
            self.token_cache.hits = self.token_cache.misses = 0
        torch.cuda.synchronize()
        logger.info(f"compiled VGGT-Omega blocks ready in {time.perf_counter() - t0:.1f}s")

    def _eager(self, err: Exception):
        logger.warning(f"compiled VGGT-Omega blocks failed ({type(err).__name__}: {str(err)[:200]}); running eagerly")
        for block in self._compiled:
            block._compiled_call_impl = None
        self._compiled = []

    def fingerprints(self, images: torch.Tensor) -> list:
        """Content keys of the views (the same image always gets the same key, whatever the batch)."""
        return content_keys(images.float())

    def _heads(self, x: torch.Tensor, tokens: list, start: int, k: int, dino: Optional[torch.Tensor] = None):
        """Camera head (all views) and depth head (first k views); the camera head runs on a side stream, concurrently
        with the depth head (both only read the tokens), except inside a CUDA-graph capture: there it runs first, on
        the capture stream.  A side-stream branch in the captured graphs made replays die with "illegal memory access"
        (~2 % of benchmark jobs with the fine-tuned checkpoint, H20 2026-10-07; never without graphs)."""
        m = self.model
        main = torch.cuda.current_stream()
        side = not torch.cuda.is_current_stream_capturing()
        if side:
            if self._head_stream is None:
                self._head_stream = torch.cuda.Stream(device=x.device)
            self._head_stream.wait_stream(main)
        with torch.cuda.stream(self._head_stream if side else main), torch.autocast(device_type="cuda", enabled=False):
            pose_enc = m.camera_head(tokens, patch_token_start=start)
        if k == 0:
            # no view needs depth (covis_source=head without a frontend observation): the dense head is skipped
            H, W = x.shape[-2:]
            depth = conf = x.new_zeros((x.shape[0], 0, H, W))
        else:
            sub = [t if t is None or k == x.shape[1] else t[:, :k] for t in tokens]
            with torch.autocast(device_type="cuda", dtype=self.dtype, enabled=self.dense_head_bf16):
                depth, conf = m.dense_head(sub, images=x[:, :k], patch_token_start=start)
        if side:
            main.wait_stream(self._head_stream)
        # the fine-tune's covisibility / scale heads (small) run on the main stream after the join: on the side stream,
        # concurrently with the dense head, they made the side-stream race (illegal memory access with CUDA graphs)
        # frequent (H20 benchmark, 2026-10-06)
        with torch.autocast(device_type="cuda", enabled=False):
            final = tokens[-1]
            covis = self.covis_head(final[:, :, :start]) if self.covis_head is not None else pose_enc.new_zeros(0)
            if getattr(self, "_scale_multilayer", False):
                B, S = x.shape[:2]
                p = m.aggregator.patch_size
                dino = dino.reshape(B, S, -1, dino.shape[-1]) if dino is not None else None
                kw = {}
                if getattr(self.scale_head, "canonical_f", None) is not None:
                    # canonical-camera head: the calibrated focal / width when the system set one (buffer filled
                    # before each pass, so CUDA graphs see the current value), else the model's own focal
                    pred_f = 0.5 / torch.tan(pose_enc[..., 8].float() / 2)
                    fb = getattr(self, "_focal_buf", None)
                    kw["focal"] = torch.where(fb > 0, fb, pred_f) if fb is not None else pred_f
                if getattr(self, "scale_encoder", None) is not None:
                    n = min(self.scale_encoder.frames, S)
                    xe = (x[:, :n] - m.aggregator._resnet_mean) / m.aggregator._resnet_std
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        enc = self.scale_encoder(xe.reshape(B * n, *x.shape[2:]))
                    kw["enc"] = enc.reshape(B, n, -1, enc.shape[-1])
                log_scale = self.scale_head(tokens, start, (x.shape[-2] // p, x.shape[-1] // p), dino, **kw)
                if isinstance(log_scale, tuple):         # dense head: (log scale, per-patch votes, confidences)
                    log_scale = log_scale[0]
            else:
                log_scale = (self.scale_head(final, start, pose_enc[:, 0, 8] if getattr(self.scale_head, "canonical_f",
                                                                                        None) else None)
                             if self.scale_head is not None else pose_enc.new_zeros(0))
        return pose_enc, depth.float(), conf.float(), covis, log_scale

    def _forward(self, x: torch.Tensor, k: int, patch_tokens: Optional[torch.Tensor] = None):
        """VGGTOmega.forward (aggregator, camera head, depth head for the first k views).  patch_tokens: the views'
        DINO tokens, computed beforehand (the aggregator then skips its patch embedding)."""
        if patch_tokens is not None:
            self.token_cache.override = patch_tokens
        dino, hook = {}, None
        if getattr(self, "_scale_multilayer", False):
            hook = self.model.aggregator.patch_embed.register_forward_hook(
                lambda mod, inp, out: dino.__setitem__("x", out["x_norm_patchtokens"] if isinstance(out, dict) else out))
        try:
            with torch.autocast(device_type="cuda", dtype=self.dtype):
                tokens, start = self.model.aggregator(x)
        finally:
            if patch_tokens is not None:
                self.token_cache.override = None
            if hook is not None:
                hook.remove()
        return self._heads(x, tokens, start, k, dino.get("x"))

    def _capture(self, fn):
        """CUDA graph of fn() (after two warm-up calls on a side stream); all graphs share one memory pool."""
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool):
            out = fn()
        return graph, out

    def _embed_graphed(self, normalized: torch.Tensor) -> torch.Tensor:
        """DINO tokens of normalized images through a CUDA graph per batch shape (outputs are overwritten by the next
        replay; the token cache stores copies)."""
        key = tuple(normalized.shape)
        entry = self._dino_graphs.get(key)
        if entry is None:
            x_in = normalized.clone()
            embed = self.token_cache.embed

            def fn():
                with torch.autocast(device_type="cuda", dtype=self.dtype):
                    out = embed(x_in)
                return out["x_norm_patchtokens"] if isinstance(out, dict) else out
            graph, out = self._capture(fn)
            entry = self._dino_graphs[key] = (graph, x_in, out)
        graph, x_in, out = entry
        x_in.copy_(normalized)
        graph.replay()
        return out

    def _forward_graphed(self, x: torch.Tensor, keys: list, k: int):
        """The whole pass as CUDA graphs: one per input shape (views, image size, depth views) for the aggregator and
        heads, and one per batch size for the DINO embedding of the images not cached.  One launch replaces the
        thousands of kernel launches (and Python / guard overhead) of a pass; the kernels are the same."""
        agg = self.model.aggregator
        normalized = (x[0] - agg._resnet_mean[0]) / agg._resnet_std[0]     # as Aggregator.forward does
        tokens = self.token_cache.gather(normalized, keys)
        key = (tuple(x.shape), tuple(tokens.shape), tokens.dtype, k)
        entry = self._graphs.get(key)
        if entry is None:
            x_in, t_in = x.clone(), tokens.clone()
            graph, out = self._capture(lambda: self._forward(x_in, k, patch_tokens=t_in))
            entry = self._graphs[key] = (graph, x_in, t_in, out)
        graph, x_in, t_in, out = entry
        x_in.copy_(x)
        t_in.copy_(tokens)
        graph.replay()
        return tuple(o.clone() for o in out)

    @torch.inference_mode()
    def infer(self, images: torch.Tensor, n_depth: Optional[int] = None, focal: Optional[float] = None) -> FFPrediction:
        """focal: calibrated focal length / image width of the views, for a canonical-camera scale head (None: use
        the model's predicted FoV).  Without one, the calibrated focal the estimator registered (calib_focal) is used,
        so every caller of the shared backend (also the VGGT-IMU front end's own passes) gets it."""
        images = images.to(self.device)
        x = images[None]
        if focal is None:
            focal = getattr(self, "calib_focal", None)
        if getattr(self.scale_head, "canonical_f", None) is not None:
            if getattr(self, "_focal_buf", None) is None:
                self._focal_buf = torch.zeros((), device=self.device)
            self._focal_buf.fill_(float(focal) if focal else 0.0)
        k = x.shape[1] if n_depth is None else min(int(n_depth), x.shape[1])
        keys = self.fingerprints(images) if self.token_cache is not None else None
        out = None
        if self.cuda_graphs and self.token_cache is not None and self.token_cache.capacity > 0:
            try:
                out = self._forward_graphed(x, keys, k)
            except Exception as err:      # noqa: BLE001  capture not supported here: run without graphs
                logger.warning(f"CUDA graphs of VGGT-Omega failed ({type(err).__name__}: {str(err)[:200]}); running without")
                self.cuda_graphs = False
                self.token_cache.embed_missing = None
        if out is None:
            if self.token_cache is not None:
                self.token_cache.keys = keys
            try:
                out = self._forward(x, k)
            except Exception as err:      # noqa: BLE001  a compiler failure (missing toolchain, unsupported GPU)
                if not self._compiled:
                    raise
                self._eager(err)
                if self.token_cache is not None:
                    self.token_cache.keys = keys
                out = self._forward(x, k)
        pose_enc, depth, conf, covis, log_scale = out
        hw = tuple(images.shape[-2:])
        extr, intr = self._decode(pose_enc, hw)                  # (1,S,3,4) w2c, (1,S,3,3)
        w2c = extr[0].float().cpu().numpy().astype(np.float64)
        c2w = invert_poses(_to_h(w2c))
        depth = depth[0]                                         # (k,H,W,1) or (k,H,W)
        if depth.dim() == 4:
            depth = depth[..., 0]
        conf = conf[0]
        if conf.dim() == 4:
            conf = conf[..., 0]
        return FFPrediction(c2w=c2w, K=intr[0].float().cpu().numpy(), depth=depth, depth_conf=conf, hw=hw,
                            covis=torch.sigmoid(covis[0]).cpu().numpy() if covis.numel() else None,
                            log_scale=float(log_scale[0]) if log_scale.numel() else None)


def _load_ft_heads(sd: dict, device: str, scale_head_cfg: Optional[dict] = None):
    """Covisibility / scale heads of a vggt_ft checkpoint (vggt_ft/heads.py), or None where absent."""
    heads = []
    for prefix, name in (("covis_head.", "CovisHead"), ("scale_head.", "ScaleHead")):
        sub = {k[len(prefix):]: v.float() for k, v in sd.items() if k.startswith(prefix)}
        if not sub:
            heads.append(None)
            continue
        from vggt_ft import heads as ft_heads
        cfg = dict(scale_head_cfg or {})
        if cfg.pop("encoder", None) and name == "ScaleHead":
            cfg["enc_dim"] = sub["proj_enc.1.weight"].shape[1]
        h = ft_heads.build_scale_head(cfg) if name == "ScaleHead" else getattr(ft_heads, name)()
        h.load_state_dict(sub)
        heads.append(h.eval().to(device))
        logger.info(f"loaded the fine-tune's {type(h).__name__}")
    return tuple(heads)


def _load_scale_encoder(sd: dict, patch_embed, scale_head_cfg: Optional[dict], device: str):
    """LoRA scale encoder of a vggt_ft checkpoint: base weights from the model's patch embedding, adapters from the
    checkpoint; None if the scale head has none."""
    enc_cfg = (scale_head_cfg or {}).get("encoder")
    if not enc_cfg:
        return None
    from vggt_ft.scale_encoder import ScaleEncoder
    enc = ScaleEncoder(patch_embed, **enc_cfg)
    if enc.full:      # fully fine-tuned copy: all of its weights come from the checkpoint
        own = {k[len("scale_encoder.dino."):]: v.float() for k, v in sd.items() if k.startswith("scale_encoder.dino.")}
        enc.dino.load_state_dict(own, strict=True)
    else:             # LoRA: base = the patch embedding, adapters from the checkpoint
        enc.base_state_from(patch_embed)
        lora = {k[len("scale_encoder.dino."):]: v.float() for k, v in sd.items()
                if k.startswith("scale_encoder.") and "lora_" in k}
        missing, _ = enc.dino.load_state_dict(lora, strict=False)
        if any("lora_" in k for k in missing):
            raise RuntimeError(f"scale encoder adapters missing in the checkpoint, e.g. {[k for k in missing if 'lora_' in k][:3]}")
    logger.info(f"loaded the fine-tune's scale encoder ({'fully fine-tuned' if enc.full else f'{enc.n_lora} LoRA layers'})")
    return enc.eval().to(device)


def pairwise_covisibility(pred: FFPrediction, views: List[int], grid: int = 48, rel_depth_tol: float = 0.15,
                          min_conf_quantile: float = 0.3) -> np.ndarray:
    """covisibility_scores for every ordered pair of the given views in one batched pass (one device
    synchronisation): C[a, b] = score of source views[a] in destination views[b]; the diagonal is 1."""
    n = len(views)
    C = np.ones((n, n))
    if n < 2:
        return C
    device = pred.depth.device
    S, H, W = pred.depth.shape
    pa, pb = np.nonzero(~np.eye(n, dtype=bool))                         # ordered pairs (source a, destination b)
    vi = torch.as_tensor(views, device=device)
    ys = torch.linspace(0, H - 1, grid, device=device)
    xs = torch.linspace(0, W - 1, grid, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy, gx = gy.reshape(-1), gx.reshape(-1)
    norm = torch.stack([gx / (W - 1) * 2 - 1, gy / (H - 1) * 2 - 1], dim=-1).view(1, 1, -1, 2).expand(n, 1, -1, 2)
    depth = pred.depth[vi]
    d = F.grid_sample(depth[:, None], norm, align_corners=True).view(n, -1)       # each view as a source, once
    valid = torch.isfinite(d) & (d > 1e-6)
    if pred.depth_conf is not None:
        conf = pred.depth_conf[vi]
        c = F.grid_sample(conf[:, None], norm, align_corners=True).view(n, -1)
        thr = torch.quantile(conf.flatten(1)[:, :: max(1, (H * W) // 20000)], min_conf_quantile, dim=1)
        valid &= c >= thr[:, None]
    c2w = torch.from_numpy(pred.c2w).to(device=device, dtype=torch.float32)[vi]
    K = torch.from_numpy(pred.K).to(device=device, dtype=torch.float32)[vi]
    x = (gx - K[:, 0, 2:3]) / K[:, 0, 0:1] * d
    y = (gy - K[:, 1, 2:3]) / K[:, 1, 1:2] * d
    P = torch.stack([x, y, d, torch.ones_like(d)], dim=1)                          # (n,4,N) in the source cameras
    a, b = torch.as_tensor(pa, device=device), torch.as_tensor(pb, device=device)
    Pd = (torch.linalg.inv(c2w[b]) @ c2w[a] @ P[a])[:, :3]                         # into the destination cameras
    z = Pd[:, 2]
    Kd = K[b]
    u = Kd[:, 0, 0:1] * Pd[:, 0] / z.clamp(min=1e-6) + Kd[:, 0, 2:3]
    v = Kd[:, 1, 1:2] * Pd[:, 1] / z.clamp(min=1e-6) + Kd[:, 1, 2:3]
    inside = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    un = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], dim=-1)[:, None]
    zd = F.grid_sample(depth[b][:, None], un, align_corners=True).view(len(pa), -1)
    ok = inside & ((z - zd).abs() <= rel_depth_tol * zd.clamp(min=1e-6)) & valid[a]
    n_valid = valid[a].sum(dim=1)
    scores = torch.where(n_valid >= 16, ok.sum(dim=1).double() / n_valid.clamp(min=1).double(),
                         torch.zeros_like(n_valid, dtype=torch.float64))
    C[pa, pb] = scores.cpu().numpy()
    return C


def filter_map_anchors(pred: FFPrediction, anchors: List[ScaleAnchor], min_covis: float, max_pairs: int,
                       source: str = "geometric", grid: int = 48, rel_depth_tol: float = 0.15) -> List[ScaleAnchor]:
    """Map anchors only between references that overlap each other: covisibility >= min_covis both ways, from the
    predicted depth and poses ("geometric", needs both views' depth), the covisibility head ("head") or the smaller
    of the two ("min"); the max_pairs first that pass are kept (the system lists them longest first).  Other anchors
    are kept as they are."""
    maps = [a for a in anchors if a.kind == "map"]
    if not maps:
        return anchors
    views = sorted({a.idx_a for a in maps} | {a.idx_b for a in maps})
    has_depth = pred.depth is not None and pred.depth.shape[0] > views[-1]
    use_head = pred.covis is not None and (source in ("head", "min") or not has_depth)
    use_geo = has_depth and (source != "head" or pred.covis is None)
    if not (use_head or use_geo):
        return anchors
    C = np.ones((len(views), len(views)))
    if use_geo:
        C = pairwise_covisibility(pred, views, grid=grid, rel_depth_tol=rel_depth_tol)
    if use_head:
        C = np.minimum(C, np.asarray(pred.covis, dtype=np.float64)[np.ix_(views, views)])
    pos = {v: i for i, v in enumerate(views)}
    keep = [a for a in maps if min(C[pos[a.idx_a], pos[a.idx_b]], C[pos[a.idx_b], pos[a.idx_a]]) >= min_covis]
    keep = keep[:max_pairs]
    logger.debug(f"FF map anchors with pair covisibility >= {min_covis}: {len(keep)} of {len(maps)}")
    return [a for a in anchors if a.kind != "map"] + keep


class _DA3Backend(_Backend):
    def __init__(self, checkpoint: str, device: str, process_res: int = 504):
        from depth_anything_3.api import DepthAnything3
        ckpt = Path(checkpoint)
        if not (ckpt / "model.safetensors").is_file():
            raise FileNotFoundError(f"DA3 checkpoint not found under {ckpt}")
        self.device = device
        self.model = DepthAnything3.from_pretrained(str(ckpt)).to(device).eval()
        self.process_res = process_res

    @torch.inference_mode()
    def infer(self, images: torch.Tensor, n_depth: Optional[int] = None, focal: Optional[float] = None) -> FFPrediction:
        # DA3 preprocesses from uint8 arrays; keep the first view as the reference.
        arr = (images.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        pred = self.model.inference(
            list(arr), process_res=self.process_res, process_res_method="upper_bound_resize",
            ref_view_strategy="first",
        )
        w2c = np.asarray(pred.extrinsics, dtype=np.float64)
        c2w = invert_poses(_to_h(w2c))
        depth = torch.from_numpy(np.asarray(pred.depth, dtype=np.float32)).to(self.device)
        conf = None if pred.conf is None else torch.from_numpy(np.asarray(pred.conf, dtype=np.float32)).to(self.device)
        return FFPrediction(c2w=c2w, K=np.asarray(pred.intrinsics, dtype=np.float64), depth=depth,
                            depth_conf=conf, hw=tuple(depth.shape[-2:]))


@contextmanager
def _no_weight_init():
    """Skip the random initialisation of a model whose weights are then loaded (saves ~8 s for VGGT-Omega-1B)."""
    names = [n for n in dir(torch.nn.init) if n.endswith("_") and not n.startswith("_")]
    saved = {n: getattr(torch.nn.init, n) for n in names}
    try:
        for n in names:
            setattr(torch.nn.init, n, lambda tensor, *args, **kwargs: tensor)
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)


def _to_h(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape[-2:] == (4, 4):
        return T
    out = np.zeros((*T.shape[:-2], 4, 4), dtype=np.float64)
    out[..., :3, :4] = T
    out[..., 3, 3] = 1.0
    return out


def covisibility_scores(
    pred: FFPrediction,
    src_indices: List[int],
    dst_index: int,
    grid: int = 48,
    rel_depth_tol: float = 0.15,
    min_conf_quantile: float = 0.3,
) -> np.ndarray:
    """Geometric consistency between each source view and the destination view.

    For every source view, a regular grid of pixels is unprojected with the predicted
    depth, transformed into the destination camera with the predicted relative pose,
    and projected with the destination intrinsics.  A pixel is an "inlier" when it
    lands inside the destination image and its transformed depth agrees with the
    destination depth map within `rel_depth_tol`.  The score is the inlier fraction
    over confident source pixels, i.e. a learned-geometry analogue of the PnP inlier
    ratio; it is low for wrongly retrieved (non-overlapping) references and for badly
    registered views.
    """
    if len(src_indices) == 0:
        return np.zeros(0, dtype=np.float32)
    device = pred.depth.device
    S, H, W = pred.depth.shape
    B = len(src_indices)
    ys = torch.linspace(0, H - 1, grid, device=device)
    xs = torch.linspace(0, W - 1, grid, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gy, gx = gy.reshape(-1), gx.reshape(-1)
    norm = torch.stack([gx / (W - 1) * 2 - 1, gy / (H - 1) * 2 - 1], dim=-1).view(1, 1, -1, 2).expand(B, 1, -1, 2)

    # all source views at once (one device synchronisation for all of them)
    src = torch.as_tensor(src_indices, device=device)
    c2w = torch.from_numpy(pred.c2w).to(device=device, dtype=torch.float32)
    K = torch.from_numpy(pred.K).to(device=device, dtype=torch.float32)
    d = F.grid_sample(pred.depth[src][:, None], norm, align_corners=True).view(B, -1)
    valid = torch.isfinite(d) & (d > 1e-6)
    if pred.depth_conf is not None:
        conf = pred.depth_conf[src]
        c = F.grid_sample(conf[:, None], norm, align_corners=True).view(B, -1)
        thr = torch.quantile(conf.flatten(1)[:, :: max(1, (H * W) // 20000)], min_conf_quantile, dim=1)
        valid &= c >= thr[:, None]
    Ks = K[src]
    x = (gx - Ks[:, 0, 2:3]) / Ks[:, 0, 0:1] * d
    y = (gy - Ks[:, 1, 2:3]) / Ks[:, 1, 1:2] * d
    P = torch.stack([x, y, d, torch.ones_like(d)], dim=1)                 # (B,4,N) in the source cameras
    T = torch.linalg.inv(c2w[dst_index])[None] @ c2w[src]                 # source cam -> destination cam
    Pd = (T @ P)[:, :3]
    z = Pd[:, 2]
    Kd = K[dst_index]
    u = Kd[0, 0] * Pd[:, 0] / z.clamp(min=1e-6) + Kd[0, 2]
    v = Kd[1, 1] * Pd[:, 1] / z.clamp(min=1e-6) + Kd[1, 2]
    inside = (z > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    un = torch.stack([u / (W - 1) * 2 - 1, v / (H - 1) * 2 - 1], dim=-1).view(B, 1, -1, 2)
    zd = F.grid_sample(pred.depth[dst_index][None, None].expand(B, 1, H, W), un, align_corners=True).view(B, -1)
    consistent = inside & ((z - zd).abs() <= rel_depth_tol * zd.clamp(min=1e-6))
    n_valid = valid.sum(dim=1)
    n_ok = (consistent & valid).sum(dim=1)
    scores = torch.where(n_valid >= 16, n_ok.double() / n_valid.clamp(min=1).double(), torch.zeros_like(n_valid, dtype=torch.float64))
    return scores.cpu().numpy().astype(np.float32)


def principal_point_rotation(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rotation R (3x3) from the camera frame of a model that places the principal point at the image centre to the
    calibrated camera frame: R maps the model's optical axis onto the ray through the image centre.  Relative poses
    transform as T_calibrated = R T_model R^T (exact to first order in the offset; pixel centres at integers)."""
    K = np.asarray(K, dtype=np.float64)
    ray = np.array([((width - 1) / 2.0 - K[0, 2]) / K[0, 0], ((height - 1) / 2.0 - K[1, 2]) / K[1, 1], 1.0])
    ray /= np.linalg.norm(ray)
    v = np.cross([0.0, 0.0, 1.0], ray)
    s, c = float(np.linalg.norm(v)), float(ray[2])
    if s < 1e-12:
        return np.eye(3)
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Vx + Vx @ Vx * ((1 - c) / s ** 2)


class PoseEstFeedForward:
    """Relative pose estimation with a feed-forward geometry model and stereo scale anchors."""

    def __init__(self, device: str, config: FeedForwardConfig, T_right_in_left: Optional[np.ndarray] = None):
        self.device = device
        self.config = config
        self.T_right_in_left = None if T_right_in_left is None else np.asarray(T_right_in_left, dtype=np.float64)
        t0 = time.perf_counter()
        if config.backend == FFBackend.VGGT_OMEGA:
            self.backend = _VGGTOmegaBackend(config.checkpoint, device, half_weights=config.half_precision_weights,
                                             token_cache=config.token_cache, compile_blocks=config.compile,
                                             dense_head_bf16=config.dense_head_bf16, compile_mode=config.compile_mode,
                                             cuda_graphs=config.cuda_graphs)
            self.backend.warmup(config.image_resolution)
        elif config.backend == FFBackend.DA3:
            self.backend = _DA3Backend(config.checkpoint, device, process_res=config.da3_process_res)
        else:
            raise ValueError(f"Unknown feed-forward backend {config.backend}")
        logger.info(f"Loaded {config.backend.value} from {config.checkpoint} in {time.perf_counter() - t0:.1f}s")
        self.last_info: dict = {}
        self.last_scale: Optional[ScaleEstimate] = None
        self.last_frontend_obs: Optional[dict] = None   # raw output for a frontend's anchor (vgio)
        self._R_model_to_cam: Optional[np.ndarray] = None    # (4, 4) right-multiplied onto model camera poses

    def set_stereo_calibration(self, T_right_in_left: np.ndarray):
        self.T_right_in_left = np.asarray(T_right_in_left, dtype=np.float64)

    def set_camera(self, K: np.ndarray, width: int, height: int):
        """Calibrated intrinsics at the model's input resolution (the system calls this after its transforms):
        the focal of a canonical-camera scale head (config.scale_focal = "calibrated")."""
        self._focal_norm = float(np.asarray(K)[0, 0]) / max(1, int(width))
        if getattr(self.config, "scale_focal", "calibrated") == "calibrated":
            self.backend.calib_focal = self._focal_norm      # also for passes made outside estimate_pose

    def _as_model_input(self, views: List[torch.Tensor]) -> torch.Tensor:
        images = torch.stack([v.to(self.device) for v in views], dim=0).float()
        if images.max() > 1.5:
            images = images / 255.0
        if self.config.quantize_input:
            # the 8-bit grid keyframes are stored on (cross.db.db.to_uint8_image): a keyframe made from this image then
            # comes back as a reference with the same pixels, and its tokens are found in the token cache
            images = (images.clamp(0, 1) * 255.0).round() / 255.0
        return images

    @torch.inference_mode()
    def precompute(self, images: List[torch.Tensor], batch: int = 8):
        """Patch tokens of images that will be references (the keyframes of a loaded map), into the token cache."""
        backend = self.backend
        cache = getattr(backend, "token_cache", None)
        if cache is None or cache.capacity <= 0 or not images:
            return
        t0, n0 = time.perf_counter(), cache.misses
        images = images[: cache.capacity]
        for i in range(0, len(images), batch):
            x = self._as_model_input(images[i:i + batch])
            agg = backend.model.aggregator
            with torch.autocast(device_type="cuda", dtype=backend.dtype):     # as inside Aggregator.forward
                cache.gather((x - agg._resnet_mean[0]) / agg._resnet_std[0], backend.fingerprints(x))
        torch.cuda.synchronize()
        logger.info(f"token cache: {cache.misses - n0} map images embedded in {time.perf_counter() - t0:.1f}s")

    def set_camera(self, K: np.ndarray, width: int, height: int):
        """Intrinsics of the images given to the model (after its crop / resize): the principal-point correction."""
        if not getattr(self.config, "principal_point_correction", True):
            self._R_model_to_cam = None
            return
        R = principal_point_rotation(K, width, height)
        M = np.eye(4)
        M[:3, :3] = R.T            # c2w_cam = c2w_model @ [R^T 0; 0 1]
        self._R_model_to_cam = M
        logger.info(f"FF principal-point correction: {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.2f} deg")

    # ------------------------------------------------------------------ #
    @timeit
    @torch.inference_mode()
    def estimate_pose(
        self,
        ref_image: torch.Tensor,
        ref_depth: Optional[torch.Tensor],
        curr_image: torch.Tensor,
        curr_depth: Optional[torch.Tensor],
        curr_image_right: Optional[torch.Tensor] = None,
        ref_images_right: Optional[List[Optional[torch.Tensor]]] = None,
        odom_anchor: Optional[dict] = None,
        ref_rel_poses: Optional[List[Tuple[int, int, np.ndarray]]] = None,
        **kwargs,
    ):
        """Estimate T_ref_cam for every reference image in one forward pass.

        Args:
            ref_image: (B, 3, H, W) reference (left) images in [0, 1]
            ref_depth: unused (kept for interface compatibility)
            curr_image: (3, H, W) current left image
            curr_depth: unused
            curr_image_right: (3, H, W) current right image (stereo anchor) or None
            ref_images_right: list of B entries, each (3, H, W) right image or None;
                at most `config.n_ref_anchors` of them are used as extra anchors
            odom_anchor: {"image": (3,H,W) previous left image, "T_prev_curr": (4,4) metric}
                optional temporal anchor from odometry
        Returns:
            poses: pp.SE3 (B_valid, 7) T_ref_cam (metric), valid_masks (B,) np.bool, confidences (B_valid,)
        """
        cfg = self.config
        t_start = time.perf_counter()
        B = int(ref_image.shape[0])
        views = [curr_image] + [ref_image[i] for i in range(B)]
        view_tags = ["curr_L"] + [f"ref{i}_L" for i in range(B)]
        anchors: List[ScaleAnchor] = []

        # temporal (odometry) anchor.  A frontend's anchor (vgio: its last measured frame) is always in the pass, the
        # frontend reads its relative pose and depths (last_frontend_obs); it is a scale anchor once its motion is metric
        frontend_anchor = odom_anchor is not None and bool(odom_anchor.get("frontend"))
        prev_idx = kf_idx = None
        if odom_anchor is not None and (cfg.use_odom_anchor or frontend_anchor):
            T_pc = np.asarray(odom_anchor["T_prev_curr"], dtype=np.float64)
            long_enough = np.linalg.norm(T_pc[:3, 3]) >= cfg.odom_anchor_min_translation
            if long_enough or frontend_anchor:
                views.append(odom_anchor["image"])
                view_tags.append("prev_L")
                prev_idx = len(views) - 1
                kf_idx = None
                if frontend_anchor and odom_anchor.get("extra_images"):
                    views.append(odom_anchor["extra_images"][0])       # the frontend's keyframe
                    view_tags.append("kf_L")
                    kf_idx = len(views) - 1
                if long_enough and cfg.use_odom_anchor and odom_anchor.get("metric", True):
                    anchors.append(ScaleAnchor(idx_a=prev_idx, idx_b=0, T_ab=T_pc, kind="odom",
                                               weight=cfg.odom_anchor_weight))

        # stereo anchors: current pair first, then stored right images of the best references
        if self.T_right_in_left is not None:
            if curr_image_right is not None and cfg.use_curr_anchor:
                views.append(curr_image_right)
                view_tags.append("curr_R")
                anchors.append(ScaleAnchor(idx_a=0, idx_b=len(views) - 1, T_ab=self.T_right_in_left, kind="stereo"))
            if ref_images_right is not None and cfg.n_ref_anchors > 0:
                n_added = 0
                for i, img_r in enumerate(ref_images_right):
                    if img_r is None:
                        continue
                    views.append(img_r)
                    view_tags.append(f"ref{i}_R")
                    anchors.append(ScaleAnchor(idx_a=1 + i, idx_b=len(views) - 1, T_ab=self.T_right_in_left, kind="stereo"))
                    n_added += 1
                    if n_added >= cfg.n_ref_anchors:
                        break

        # map anchors: known metric relative poses between pairs of references (from the map)
        if ref_rel_poses and cfg.use_map_anchors:
            for i, j, T_ij in ref_rel_poses:
                if 0 <= i < B and 0 <= j < B and i != j:
                    anchors.append(ScaleAnchor(idx_a=1 + i, idx_b=1 + j, T_ab=np.asarray(T_ij, dtype=np.float64),
                                               kind="map", weight=cfg.map_anchor_weight))

        images = self._as_model_input(views)

        t_model = time.perf_counter()
        # depth is used for the current view and the references (geometric covisibility) and the frontend's anchor
        # view; with the learned covisibility head and no frontend observation the depth head is not run at all
        if frontend_anchor and prev_idx is not None:
            n_depth = None
        elif getattr(cfg, "covis_source", "geometric") == "head" and getattr(self.backend, "covis_head", None) is not None:
            n_depth = 0
        else:
            n_depth = 1 + B
        focal = getattr(self, "_focal_norm", None) if getattr(cfg, "scale_focal", "calibrated") == "calibrated" else None
        pred = self.backend.infer(images, n_depth=n_depth, focal=focal)
        torch.cuda.synchronize()
        t_model = time.perf_counter() - t_model
        if frontend_anchor and prev_idx is not None:
            conf = pred.depth_conf
            self.last_frontend_obs = {
                "token": odom_anchor.get("token"), "c2w_curr": pred.c2w[0], "c2w_prev": pred.c2w[prev_idx],
                "depth_curr": pred.depth[0], "depth_prev": pred.depth[prev_idx],
                "conf_curr": None if conf is None else conf[0], "conf_prev": None if conf is None else conf[prev_idx],
                "log_scale": pred.log_scale}
            if kf_idx is not None:
                self.last_frontend_obs.update(c2w_kf=pred.c2w[kf_idx], depth_kf=pred.depth[kf_idx],
                                              conf_kf=None if conf is None else conf[kf_idx])
            if "curr_R" in view_tags:            # the current stereo pair: the frontend's metric scale of the pass
                self.last_frontend_obs["c2w_right"] = pred.c2w[view_tags.index("curr_R")]

        # ---- metric scale ----
        if getattr(cfg, "map_anchor_min_pair_covis", 0.0) > 0:
            anchors = filter_map_anchors(pred, anchors, cfg.map_anchor_min_pair_covis, cfg.map_anchor_max_pairs,
                                         source=getattr(cfg, "covis_source", "geometric"), grid=cfg.covis_grid,
                                         rel_depth_tol=cfg.covis_depth_tol)
        scale_est = estimate_scale(
            pred.c2w, anchors, method=cfg.scale_method,
            max_rot_err_deg=cfg.anchor_max_rot_err_deg, min_dir_cos=cfg.anchor_min_dir_cos,
            weight_by_baseline=cfg.anchor_weight_by_baseline,
            view_logstd_floor=float(getattr(cfg, "anchor_view_logstd_floor", 0.0) or 0.0),
        )
        src = getattr(cfg, "scale_source", "anchors")
        if src != "anchors" and pred.log_scale is not None and (src == "head" or not scale_est.valid):
            # the scale head's metric scale of the model gauge (translations scale with it, frame 0 at the origin)
            scale_est = ScaleEstimate(scale=float(np.exp(pred.log_scale)), valid=True, n_anchors=scale_est.n_anchors,
                                      n_used=0, method="head")
        self.last_scale = scale_est
        if not scale_est.valid:
            logger.debug(f"FF pose est: no valid scale anchor ({scale_est.n_anchors} anchors)")
            self.last_info = {"valid": False, "scale": scale_est.to_dict(), "t_model": t_model,
                              "n_views": len(views), "view_tags": view_tags}
            return torch.empty((0, 7)), np.zeros(B, dtype=bool), torch.empty(0)

        # calibrated metric-scale correction (the estimator's translations divided by their measured/true ratio)
        metric = scale_est.scale / float(getattr(self, "metric_scale_correction", 1.0) or 1.0)
        c2w_metric = scale_camera_centers(pred.c2w, metric, origin_index=0)
        if self._R_model_to_cam is not None:
            c2w_metric = c2w_metric @ self._R_model_to_cam

        # ---- per-reference confidence (covisibility / geometric consistency) ----
        ref_idx = list(range(1, 1 + B))
        source = getattr(cfg, "covis_source", "geometric")
        if source != "geometric" and pred.covis is None:
            raise ValueError(f"covis_source={source} needs a checkpoint with a covisibility head (vggt_ft)")
        if source == "head":
            # learned overlap fraction of the current view with each reference (same definition as the geometric score)
            covis = np.asarray(pred.covis[0, ref_idx], dtype=np.float64)
        else:
            covis = covisibility_scores(pred, ref_idx, 0, grid=cfg.covis_grid, rel_depth_tol=cfg.covis_depth_tol)
            if cfg.covis_symmetric:
                covis_back = np.asarray([covisibility_scores(pred, [0], r, grid=cfg.covis_grid,
                                                             rel_depth_tol=cfg.covis_depth_tol)[0] for r in ref_idx])
                covis = np.minimum(covis, covis_back)
            if source == "min":
                covis = np.minimum(covis, np.asarray(pred.covis[0, ref_idx], dtype=np.float64))

        # ---- relative poses T_ref_cam = X_ref^-1 X_curr ----
        poses, confs, valid = [], [], []
        for i in range(B):
            T_ref_cam = invert_poses(c2w_metric[1 + i]) @ c2w_metric[0]
            dist = float(np.linalg.norm(T_ref_cam[:3, 3]))
            ok = (covis[i] >= cfg.min_covis) and (dist <= cfg.max_rel_distance)
            valid.append(bool(ok))
            if ok:
                poses.append(pp.from_matrix(torch.from_numpy(T_ref_cam).float(), ltype=pp.SE3_type))
                confs.append(float(covis[i]))
        valid = np.asarray(valid, dtype=bool)
        if valid.sum() > 0:
            poses = torch.stack([p.tensor() for p in poses], dim=0)
            poses = pp.SE3(poses)
            confs = torch.tensor(confs, dtype=torch.float32)
        else:
            poses, confs = torch.empty((0, 7)), torch.empty(0)

        self.last_info = {
            "valid": True,
            "scale": scale_est.to_dict(),
            "covis": covis.tolist(),
            "valid_masks": valid.tolist(),
            "t_model": t_model,
            "t_total": time.perf_counter() - t_start,
            "n_views": len(views),
            "view_tags": view_tags,
            "c2w_metric": c2w_metric,
        }
        logger.debug(
            f"FF pose est: {len(views)} views, scale={scale_est.scale:.3f} "
            f"({scale_est.n_used}/{scale_est.n_anchors} anchors, logstd={scale_est.log_std:.3f}), "
            f"covis={np.round(covis, 2).tolist()}, model {t_model * 1e3:.0f} ms"
        )
        return poses, valid, confs

    @property
    def scale_rel_std(self) -> float:
        """Relative (multiplicative) std of the last scale estimate; used to inflate translation std."""
        if self.last_scale is None or not self.last_scale.valid:
            return 1.0
        return float(self.last_scale.log_std)
