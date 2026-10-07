from .boq import BoQ
from .index import DescriptorIndex, PCAProjection, ScoreCalibration, SpatialIndex, projection_from_config
import numpy as np
import torch
from cross.core.types import Atlas, Keyframe
from cross.utils.profile import timeit
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict

from cross.core.config import RetrievalConfig, VPRModelType


def to_uint8_image(t):
    """float image in [0, 1] -> uint8 (compact keyframe storage); None and uint8 pass through."""
    if t is None or t.dtype == torch.uint8:
        return t
    return (t.clamp(0, 1) * 255.0).round().to(torch.uint8)


def as_float_image(t):
    """stored keyframe image -> float in [0, 1] (inverse of to_uint8_image); fp16 depth -> fp32."""
    if t is None:
        return None
    if t.dtype == torch.uint8:
        return t.float() / 255.0
    return t.float() if t.dtype == torch.float16 else t


class KeyframeDatabase:
    def __init__(
        self,
        system,
        device: str = "cuda",
        config: Union[RetrievalConfig, None] = None,
    ):
        """Database for posed RGBD with efficient embedding management.

        Args:
            system: System
            device: Device to store tensors
            config: RetrievalConfig with VPR model type, buffer size, and query parameters.
        """
        cfg = config or RetrievalConfig()
        self.config = cfg
        self.system = system
        self.device = device

        # Atlas management
        self._atlases: Dict[int, Atlas] = {}  # Maps atlas_id to Atlas object
        self._next_atlas_id: int = 0

        # Store PosedRGBD objects by atlas
        self._keyframe_by_atlas: Dict[Atlas, List[Keyframe]] = defaultdict(list)

        # Index management
        self._atlas_to_indices: Dict[Atlas, List[int]] = defaultdict(list)  # Maps atlas to list of buffer indices
        self._index_to_atlas_idx: Dict[int, Tuple[Atlas, int]] = {}  # Maps buffer index to (atlas, list_idx)

        # VPR model setup
        self.vpr_model_type = cfg.vpr_model_type
        if self.vpr_model_type == VPRModelType.BOQ:
            self.vpr_model = BoQ(backbone_name="resnet50", device=device)
        else:
            raise ValueError(f"VPR model {self.vpr_model_type} not supported")

        # Descriptor index (cross/db/index.py): without a projection the float32 buffer of the original database
        self._initial_buffer_size = cfg.initial_buffer_size
        icfg = getattr(cfg, "index", None)
        spec = getattr(icfg, "projection", None)
        self.index = self._new_index(None if spec == "map" else projection_from_config(spec, self.vpr_model.get_embed_dim()))
        self._row_kf: List[Keyframe] = []          # keyframe of each descriptor row
        self._id_to_row: Dict[int, int] = {}
        self.spatial = SpatialIndex()              # keyframe positions, for locality-aware retrieval
        self._track_positions = bool(getattr(getattr(cfg, "locality", None), "enabled", False))

        # Query parameters
        self.score_threshold_high = cfg.vpr_score_threshold_high
        self.score_threshold_low = cfg.vpr_score_threshold_low
        self.top_k = cfg.top_k

    def _new_index(self, projection) -> DescriptorIndex:
        icfg = getattr(self.config, "index", None)
        g = lambda k, d: getattr(icfg, k, d) if icfg is not None else d   # noqa: E731
        return DescriptorIndex(
            self.vpr_model.get_embed_dim(), device=self.device, initial_capacity=self._initial_buffer_size,
            projection=projection, store_dtype=g("store_dtype", "auto"), backend=g("backend", "exact"),
            ivf_nlist=g("ivf_nlist", 0), ivf_nprobe=g("ivf_nprobe", 16), ivf_min_rows=g("ivf_min_rows", 200000),
            fit_at=g("fit_at", 0) if g("projection", None) == "map" else 0, fit_dim=g("fit_dim", 0),
            fit_energy=g("fit_energy", 0.9), extend=g("extend", True), extend_margin=g("extend_margin", 0.05),
            extend_dims=g("extend_dims", 64), max_dim=g("max_dim", 2048), recent=g("recent", 1024))

    def _extend_buffer(self, min_size: int):
        """Extend the descriptor buffer if needed."""
        self.index.reserve(min_size)

    @property
    def _embedding_buffer(self) -> torch.Tensor:
        return self.index.buf

    @property
    def _current_size(self) -> int:
        return self.index.n

    def keyframe_at(self, row: int) -> Keyframe:
        return self._row_kf[row]

    def get_all_keyframes(self, atlas: Atlas= None) -> List[Keyframe]:
        """Get all keyframes from the database."""
        if atlas is None:
            # return all keyframes
            keyframes = []
            for atlas in self._keyframe_by_atlas:
                keyframes.extend(self._keyframe_by_atlas[atlas])
            return keyframes
        else:
            return self._keyframe_by_atlas[atlas]

    @timeit
    def insert(
        self, 
        id: int,
        raw_rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        mu: torch.Tensor = None,
        sigma: torch.Tensor = None,
        weights: torch.Tensor = None,
        timestamp: float = None,
        atlas: Atlas = None,
        temporary: bool = False,
        last_pgo_step: int = -1,
        pose_charts: torch.Tensor = None,
        metric_source: dict = None,
        raw_rgb_right: torch.Tensor = None,
    ) -> Keyframe:
        """Insert a Keyframe into the database.

        Args:
            id: The id of the keyframe
            img_tensor: The image of the place, tensor of shape (3, H, W)
            atlas: The atlas of the place
            pose_in_atlas: The pose of the place in the atlas frame
            temporary: Whether the keyframe is temporary
            t: The timestamp of the place
            raw_rgb_image: The raw RGB image of the place, shape (H, W, 3) nd uint8
        """
        # Get embedding
        embedding = self.vpr_model.get_embedding(raw_rgb_image)
        if getattr(self.config, "store_images_uint8", False):
            # compact storage (after the embedding): images as uint8, depth as fp16; consumers convert with as_float_image
            raw_rgb_image = to_uint8_image(raw_rgb_image)
            raw_rgb_right = to_uint8_image(raw_rgb_right)
            depth_image = depth_image.half() if depth_image is not None else None
        
        # Create PosedRGBD object
        keyframe = Keyframe(
            raw_rgb_image=raw_rgb_image,
            depth_image=depth_image,
            raw_rgb_right=raw_rgb_right,
            pose_mu=mu,
            pose_std=sigma,
            pose_weights=weights,
            timestamp=timestamp,
            atlas=atlas,
            temporary=temporary,
            last_pgo_step=last_pgo_step,
            pose_charts=pose_charts,
            metric_source=metric_source,
        )
        
        # Add to storage
        list_idx = len(self._keyframe_by_atlas[atlas])
        self._keyframe_by_atlas[atlas].append(keyframe)
        
        # Add the descriptor (the index grows its buffer by 100 rows at least, as before)
        row = self.index.add(embedding, keyframe.id)
        self._atlas_to_indices[atlas].append(row)
        self._index_to_atlas_idx[row] = (atlas, list_idx)
        self._row_kf.append(keyframe)
        self._id_to_row[int(keyframe.id)] = row
        if mu is not None and getattr(self, "_track_positions", True):
            self.spatial.add(row, _position(keyframe))

        return keyframe

    def remove(
        self, 
        idx: int,
        buffer_idx: int,
        atlas: Atlas = 0, 
    ):
        """Remove a Keyframe from the database.
        Args:
            atlas: The atlas containing the Keyframe
            idx: Index of the Keyframe in the atlas's list
        """
        if atlas not in self._keyframe_by_atlas or idx >= len(self._keyframe_by_atlas[atlas]):
            return
            
        # Find the buffer index for this place
        buffer_idx = None
        for i, (a, list_i) in self._index_to_atlas_idx.items():
            if a == atlas and list_i == idx:
                buffer_idx = i
                break
                
        if buffer_idx is None:
            return
            
        # Remove from storage
        self._keyframe_by_atlas[atlas].pop(idx)
        
        # Remove from indices
        self._atlas_to_indices[atlas].remove(buffer_idx)
        del self._index_to_atlas_idx[buffer_idx]
        
        kf_removed = self._row_kf[buffer_idx]
        self._id_to_row.pop(int(kf_removed.id), None)
        moved = self.index.remove_row(buffer_idx)
        self.spatial.remap_row(buffer_idx, None)
        if moved is None:
            self._row_kf.pop()
            return

        # the last row moved into buffer_idx
        last_idx = moved
        last_atlas, last_list_idx = self._index_to_atlas_idx[last_idx]
        self._atlas_to_indices[last_atlas].remove(last_idx)
        self._atlas_to_indices[last_atlas].append(buffer_idx)
        self._index_to_atlas_idx[buffer_idx] = (last_atlas, last_list_idx)
        del self._index_to_atlas_idx[last_idx]
        self._row_kf[buffer_idx] = self._row_kf.pop()
        self._id_to_row[int(self._row_kf[buffer_idx].id)] = buffer_idx
        self.spatial.remap_row(last_idx, buffer_idx)

    def get_size(self):
        return self._current_size
    
    def create_atlas(self) -> Atlas:
        """Create a new atlas and return it."""
        atlas = Atlas(id=self._next_atlas_id)
        self._atlases[self._next_atlas_id] = atlas
        self._next_atlas_id += 1
        return atlas
    
    def get_atlas(self, atlas_id: int) -> Optional[Atlas]:
        """Get an atlas by ID."""
        return self._atlases.get(atlas_id)
    
    def get_all_atlases(self) -> List[Atlas]:
        """Get all atlases."""
        return list(self._atlases.values())
    
    def save_state(self):
        """Save the database state for map persistence.
        
        Returns:
            dict: Database state including keyframes, embeddings, and atlases
        """
        from cross.core.conditional_pose import records
        self.index.finalize()            # map projection: refit on the whole map before its codes are stored
        db_keyframes = []
        for atlas in self._keyframe_by_atlas:
            for kf in self._keyframe_by_atlas[atlas]:
                db_keyframes.append({
                    "id": kf.id,
                    "raw_rgb_image": kf.raw_rgb_image.cpu() if kf.raw_rgb_image is not None else None,
                    "depth_image": kf.depth_image.cpu() if kf.depth_image is not None else None,
                    "raw_rgb_right": kf.raw_rgb_right.cpu() if kf.raw_rgb_right is not None else None,
                    "pose_mu": kf.pose_mu.cpu() if kf.pose_mu is not None else None,
                    "pose_std": kf.pose_std.cpu() if kf.pose_std is not None else None,
                    "pose_weights": kf.pose_weights.cpu() if kf.pose_weights is not None else None,
                    "pose_charts": kf.pose_charts.cpu() if kf.pose_charts is not None else None,
                    "metric_source": kf.metric_source,
                    "conditional_poses": records(kf.conditional_poses),
                    "atlas_id": kf.atlas.id if kf.atlas is not None else None,
                    "timestamp": kf.timestamp,
                    "temporary": kf.temporary,
                    "last_pgo_step": kf.last_pgo_step,
                })
        
        return {
            "keyframes": db_keyframes,
            "embeddings": self._embedding_buffer[:self._current_size].cpu(),
            "descriptor_index": self.index.state(),   # projection of the stored descriptors (None: full BoQ)
            "atlas_to_indices": {atlas.id: indices for atlas, indices in self._atlas_to_indices.items()},
            "index_to_atlas_idx": {buf_idx: (atlas.id, list_idx) for buf_idx, (atlas, list_idx) in self._index_to_atlas_idx.items()},
            "current_size": self._current_size,
            "atlases": {atlas_id: {"id": atlas.id} for atlas_id, atlas in self._atlases.items()},
            "next_atlas_id": self._next_atlas_id,
        }
    
    def load_state(self, db_data: dict, storage_device: str, pose_device=None):
        """Load the database state from saved data.
        
        Args:
            db_data: Dictionary containing saved database state
            storage_device: Device to store tensors
            
        Returns:
            dict: Mapping from keyframe ID to Keyframe object
        """
        from cross.core.conditional_pose import restore
        self._keyframe_by_atlas.clear()
        self._atlas_to_indices.clear()
        self._index_to_atlas_idx.clear()
        self._atlases.clear()
        
        # Restore atlases
        for atlas_id, atlas_data in db_data["atlases"].items():
            atlas = Atlas(id=atlas_data["id"])
            self._atlases[atlas_data["id"]] = atlas
        self._next_atlas_id = db_data["next_atlas_id"]
        
        # Create a map to track all keyframes by ID
        all_keyframes_map = {}
        
        # Restore permanent keyframes from database
        for kf_data in db_data["keyframes"]:
            atlas = self._atlases[kf_data["atlas_id"]] if kf_data["atlas_id"] is not None else None
            
            # Create Keyframe object
            kf = Keyframe(
                raw_rgb_image=kf_data["raw_rgb_image"].to(storage_device) if kf_data["raw_rgb_image"] is not None else None,
                depth_image=kf_data["depth_image"].to(storage_device) if kf_data["depth_image"] is not None else None,
                raw_rgb_right=kf_data.get("raw_rgb_right").to(storage_device) if kf_data.get("raw_rgb_right") is not None else None,
                pose_mu=kf_data["pose_mu"].to(pose_device or storage_device) if kf_data["pose_mu"] is not None else None,
                pose_std=kf_data["pose_std"].to(pose_device or storage_device) if kf_data["pose_std"] is not None else None,
                pose_weights=kf_data["pose_weights"].to(pose_device or storage_device) if kf_data["pose_weights"] is not None else None,
                atlas=atlas,
                timestamp=kf_data["timestamp"],
                temporary=kf_data["temporary"],
                last_pgo_step=kf_data["last_pgo_step"],
                pose_charts=kf_data["pose_charts"].to(pose_device or storage_device) if kf_data.get("pose_charts") is not None else None,
                metric_source=kf_data.get("metric_source"),
                conditional_poses=restore(kf_data.get("conditional_poses")),
            )

            # Manually set the ID to match the saved one
            kf.id = kf_data["id"]

            # Add to database storage
            list_idx = len(self._keyframe_by_atlas[atlas])
            self._keyframe_by_atlas[atlas].append(kf)
            
            all_keyframes_map[kf.id] = kf
        
        # Restore database index mappings
        for atlas_id, indices in db_data["atlas_to_indices"].items():
            atlas = self._atlases[int(atlas_id)]
            self._atlas_to_indices[atlas] = indices
        
        for buffer_idx, (atlas_id, list_idx) in db_data["index_to_atlas_idx"].items():
            atlas = self._atlases[atlas_id]
            self._index_to_atlas_idx[buffer_idx] = (atlas, list_idx)

        # Restore the descriptors (rows in buffer order) and the row -> keyframe maps
        n = int(db_data["current_size"])
        self._row_kf = [self._keyframe_by_atlas[a][li] for a, li in (self._index_to_atlas_idx[r] for r in range(n))]
        self._id_to_row = {int(kf.id): r for r, kf in enumerate(self._row_kf)}
        self._load_descriptors(db_data["embeddings"][:n], db_data.get("descriptor_index"))
        self.spatial = SpatialIndex()

        return all_keyframes_map

    def _load_descriptors(self, codes: torch.Tensor, index_state: Optional[dict]) -> None:
        """Stored rows -> index.  A map saved with a projection brings it (it defines the codes); full descriptors
        of a map without one are projected now when this database is configured with a projection (a fixed one, or a
        map projection once the map holds fit_at keyframes)."""
        proj_state = (index_state or {}).get("projection")
        if proj_state is not None:
            proj = PCAProjection.from_state(proj_state)
            if self.index.projection is not None and not np.array_equal(
                    self.index.projection.components.astype(np.float16), np.asarray(proj_state["components"])):
                logger.info(f"descriptor index: the map's projection ({proj.dim_in} -> {proj.dim}) replaces the configured one")
            self.index = self._new_index(proj)
            self.index.calibration = ScoreCalibration.from_state((index_state or {}).get("calibration"))
        elif self.index.projection is not None and codes.shape[0] and codes.shape[1] == self.index.dim_in:
            codes = torch.cat([self.index.encode(c.to(self.device).float()) for c in codes.split(4096)])
        else:
            self.index = self._new_index(self.index.projection)
        self.index.set_rows(codes, [int(kf.id) for kf in self._row_kf])

    @timeit
    def similarities(self, img: torch.Tensor, kf_ids) -> dict:
        """VPR cosine similarity of the query image to the given keyframes (by id), {kf_id: score}."""
        want = {int(k) for k in kf_ids}
        if not want or self._current_size == 0:
            return {}
        q_full = self.vpr_model.get_embedding(img)
        q = self.index.encode(q_full)
        out = {}
        rows = sorted(self._id_to_row[k] for k in want if k in self._id_to_row)
        if self.index.projection is not None and rows:
            sc = self.index.query_scores(q_full, torch.tensor(rows, device=self.device), shortlist=len(rows))
            return {int(self._row_kf[bi].id): float(v) for bi, v in zip(rows, sc.tolist())}
        for bi in rows:
            kid = int(self._row_kf[bi].id)
            out[kid] = float(self._embedding_buffer[bi] @ q)
        return out

    def _reserved_rows(self, reserved_keyframe_ids) -> torch.Tensor:
        """Boolean mask over the descriptor rows of the given keyframe ids (cached for the same id collection)."""
        c = getattr(self, "_reserved_cache", None)
        n = self._current_size
        if c is not None and c[0] is reserved_keyframe_ids and c[1] == n:
            return c[2]
        ids = torch.as_tensor(sorted(int(k) for k in reserved_keyframe_ids), dtype=torch.long, device=self.device)
        mask = torch.isin(self.index.ids[:n], ids)
        self._reserved_cache = (reserved_keyframe_ids, n, mask)
        return mask

    def query(
        self, 
        img: torch.Tensor,
        target_atlases: Optional[List[Atlas]] = None,
        reserved_keyframe_ids=(),
        reserved_count: int = 0,
        reserved_min_score: Optional[float] = None,
        max_kf_id: Optional[int] = None,
        min_kf_id: Optional[int] = None,
        score_threshold: Optional[float] = None,
        rows=None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[float, Keyframe]]:
        """Query the database for the most likely Keyframe.

        Args:
            img: The image to query the place database with, shape (3, H, W)
            target_atlases: Optional list of atlases to restrict the search to
            rows: Optional descriptor rows to restrict the search to (e.g. the keyframes near a position)
            top_k: number of results (default: the configured top_k)

        Returns:
            {"scores": [...], "keyframes": [...]} in descending order of score
        """
        top_k = self.top_k if top_k is None else int(top_k)
        if reserved_count < 0 or reserved_count > top_k:
            raise ValueError("Historical retrieval slots must be between zero and top_k")
        if reserved_min_score is not None:
            if not 0 <= reserved_min_score <= 1 or not reserved_count:
                raise ValueError("Historical minimum score needs reserved slots and must be within [0,1]")
        empty = {"scores": [], "keyframes": []}
        n = self._current_size
        if n == 0:
            return empty

        # Get query embedding (the stored code of the index: identity without a projection)
        query_full = self.vpr_model.get_embedding(img)
        query_embedding = self.index.encode(query_full)

        # Rows to score (vectorized; same rows in the same order as the original per-keyframe filters)
        valid = None
        if target_atlases is not None:
            valid = torch.as_tensor([bi for atlas in target_atlases for bi in self._atlas_to_indices[atlas]],
                                    dtype=torch.long, device=self.device)
        elif rows is not None:
            valid = torch.as_tensor(np.unique(np.asarray(rows, dtype=np.int64)), dtype=torch.long, device=self.device)
        if max_kf_id is not None or min_kf_id is not None:
            keep = self.index.id_mask(max_kf_id, min_kf_id)
            valid = keep.nonzero(as_tuple=True)[0] if valid is None else valid[keep[valid]]
        if rows is not None and target_atlases is not None:
            valid = valid[torch.isin(valid, torch.as_tensor(np.asarray(rows, dtype=np.int64), device=self.device))]
        ann = self.index.ann_rows(query_embedding) if rows is None else None
        if ann is not None:
            # IVF: only the rows of the probed cells (sorted rows; a restriction keeps its own order)
            valid = ann if valid is None else valid[torch.isin(valid, ann)]
        if valid is not None and valid.numel() == 0:
            return empty

        # Compute similarity scores (projected index: calibrated to the full cosine, the best re-scored exactly)
        if self.index.projection is None:
            scores = self.index.scores(query_embedding, valid)
        else:
            scores = self.index.query_scores(query_full, valid, shortlist=4 * top_k)

        # Filter and sort scores
        # `scores` are similarity scores against the rows `valid` (or all rows)
        
        # Find the indices in `scores` that pass the threshold
        thr_high = self.score_threshold_high if score_threshold is None else score_threshold
        thr_low = self.score_threshold_low if score_threshold is None else score_threshold
        original_indices_passing_threshold = (scores > thr_high).nonzero(as_tuple=True)[0]
        if original_indices_passing_threshold.numel() == 0:
            original_indices_passing_threshold = (scores > thr_low).nonzero(as_tuple=True)[0]

        reserved_rows = None
        if reserved_count:
            reserved_rows = self._reserved_rows(reserved_keyframe_ids)
            if valid is not None:
                reserved_rows = reserved_rows[valid]
        if reserved_min_score is not None:
            # Permit at most the reserved budget of weaker historical views.
            # Retain original scores for CROSS's measurement uncertainty; query
            # nodes retain the original high/low thresholds. Geometry and CROSS
            # temporal evidence decide whether any such candidate is usable.
            order = scores.argsort(descending=True)
            passing = torch.cumprod((scores[order].double() > reserved_min_score).to(torch.uint8), 0).bool()
            historical = order[passing & reserved_rows[order]][:reserved_count]
            if historical.numel():
                original_indices_passing_threshold = torch.unique(torch.cat([
                    original_indices_passing_threshold, historical.to(torch.long)]), sorted=True)
        
        if original_indices_passing_threshold.numel() == 0:
            return empty
            
        # Get the actual scores for these candidates
        scores_of_candidates = scores[original_indices_passing_threshold]
        
        # Sort these candidate scores and get their relative indices (i.e., indices into scores_of_candidates)
        # in descending order of score.
        sorted_relative_indices = scores_of_candidates.argsort(descending=True)
        
        # Select the top_k relative indices
        top_k_relative_indices = sorted_relative_indices[:top_k]
        if reserved_count:
            # Reserve candidates within the same verification budget. Original
            # thresholds apply unless the explicit historical floor is set.
            res_in_rank = reserved_rows[original_indices_passing_threshold[sorted_relative_indices]]
            chosen = set(res_in_rank.nonzero(as_tuple=True)[0][:reserved_count].tolist())   # ranking positions
            for pos in range(sorted_relative_indices.numel()):
                if len(chosen) >= top_k:
                    break
                chosen.add(pos)
            top_k_relative_indices = sorted_relative_indices[torch.tensor(sorted(chosen), dtype=torch.long,
                                                                          device=scores.device)]
        
        # Use these top_k relative indices to get the actual top_k scores
        final_top_k_scores = scores_of_candidates[top_k_relative_indices]
        
        # And use them to get the top_k original indices (indices into `scores`)
        final_top_k_original_db_indices = original_indices_passing_threshold[top_k_relative_indices]
        if valid is not None:
            final_top_k_original_db_indices = valid[final_top_k_original_db_indices]
        
        # one transfer for all results
        result_scores = final_top_k_scores.tolist()
        result_keyframes = [self._row_kf[r] for r in final_top_k_original_db_indices.tolist()]

        return {
            "scores": result_scores,
            "keyframes": result_keyframes,
        }

    def refresh_spatial(self, epoch) -> None:
        """Rebuild the position index from the keyframes' current hypothesis-0 poses."""
        n = self._current_size
        pos = np.stack([_position(kf) for kf in self._row_kf[:n]]) if n else np.zeros((0, 3))
        self.spatial.rebuild(np.arange(n), pos, epoch)

    def rows_near(self, centers, radii, epoch=None) -> np.ndarray:
        """Descriptor rows of keyframes within radii[i] of centers[i] (map frame)."""
        if self.spatial.needs_rebuild(epoch, self._current_size):
            self.refresh_spatial(epoch)
        return self.spatial.query(centers, radii)

def _position(kf: Keyframe) -> np.ndarray:
    """Translation of a keyframe's hypothesis-0 pose (map frame)."""
    t = kf.pose_mu.tensor() if hasattr(kf.pose_mu, "tensor") else kf.pose_mu
    return t.reshape(-1, 7)[0, :3].detach().cpu().double().numpy()
