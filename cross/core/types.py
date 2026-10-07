import pickle
import json
import numpy as np
from typing import List, Dict, Tuple
import torch
from typing import Optional, ClassVar
from dataclasses import dataclass, field
import pypose as pp
from enum import Enum, auto


class EdgeType(Enum):
    """
    Enumeration of edge types in the pose graph.
    Using enum for type safety and memory efficiency.
    """
    VISUAL = auto()
    ODOMETRY = auto()
    LOOP_CLOSURE = auto()
    CHAIN = auto()  # Long-range shortcut edges at phase-aligned indices
    BACKBONE = auto()  # Direct edges between consecutive permanent keyframes
    PROXIMITY = auto()


@dataclass
class Camera:
    K: np.array
    fx: float
    fy: float
    px: float
    py: float
    frame_width: int
    frame_height: int
    
    def __init__(self, K: np.array, frame_width: int, frame_height: int):
        self.K = K
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.px = K[0, 2]
        self.py = K[1, 2]
        self.frame_width = frame_width
        self.frame_height = frame_height


@dataclass
class Atlas:
    """
    A class for an atlas.
    """
    id: int
    
    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        if not isinstance(other, Atlas):
            return False
        return self.id == other.id

    def __repr__(self):
        return f"Atlas(id={self.id})"


        
def serialize_keyframes(keyframes, path: str, method: str = "pickle"):
    """Serialize a list of Keyframe objects to disk.
    
    Args:
        keyframes: list of Keyframe
        path: output file path
        method: "pickle" (default) or "json"
    """
    from cross.core.conditional_pose import records
    data = []
    for kf in keyframes:
        data.append({
            "id": kf.id,
            "raw_rgb_image": kf.raw_rgb_image.cpu().numpy() if kf.raw_rgb_image is not None else None,
            "depth_image": kf.depth_image.cpu().numpy() if kf.depth_image is not None else None,
            "pose_mu": kf.pose_mu.tensor().cpu().numpy() if kf.pose_mu is not None else None,
            "pose_std": kf.pose_std.tensor().cpu().numpy() if kf.pose_std is not None else None,
            "pose_weights": kf.pose_weights.cpu().numpy() if kf.pose_weights is not None else None,
            "pose_charts": kf.pose_charts.cpu().numpy() if kf.pose_charts is not None else None,
            "metric_source": kf.metric_source,
            "conditional_poses": records(kf.conditional_poses),
            "atlas": kf.atlas.id if kf.atlas is not None else None,
            "timestamp": kf.timestamp
        })

    if method == "pickle":
        with open(path, "wb") as f:
            pickle.dump(data, f)
    elif method == "json":
        # convert numpy arrays to lists for JSON
        json_ready = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k,v in item.items()} for item in data]
        with open(path, "w") as f:
            json.dump(json_ready, f)
    else:
        raise ValueError(f"Unknown method {method}")
    
class _ImageField:
    """A keyframe image field: holds a tensor, None, or a reference into a map's image pack
    (cross.db.store.ImageRef), which is decoded on access (cached).  `Keyframe.stored_image(name)` returns what is
    held without decoding."""

    def __set_name__(self, owner, name):
        self._attr = "_" + name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return None                     # the dataclass default
        v = obj.__dict__.get(self._attr)
        return v.load() if getattr(type(v), "_cross_image_ref", False) else v

    def __set__(self, obj, value):
        obj.__dict__[self._attr] = value


@dataclass
class Keyframe:
    pose_mu: pp.LieTensor # SE3 (K, 7) mean of K SE3 components, use pose_mu.tensor() to get the tensor
    pose_std: pp.LieTensor # se3 (K, 6) std of K se3 components, assuming isotropic (diagonal)
    pose_weights: torch.Tensor # (K,) weights of K se3 components
    raw_rgb_image: torch.Tensor = _ImageField()
    depth_image: torch.Tensor = _ImageField()
    raw_rgb_right: torch.Tensor = _ImageField() # right stereo image (3, H, W), used as a metric scale anchor
    atlas: Atlas = None # the atlas of the keyframe
    timestamp: float = None # the timestamp of the keyframe
    id: int = field(init=False)
    _next_id: ClassVar[int] = 0  # Class variable for ID counter only
    temporary: bool = False # whether the keyframe is temporary
    # Throttle info for PGO to avoid repeated LC when revisiting
    last_pgo_step: int = -1
    # Coordinate frame of each pose component; independent of acquisition atlas
    # and bounded hypothesis slot. None denotes a legacy map without provenance.
    pose_charts: Optional[torch.Tensor] = None
    # Identity of the image/model prediction that supplied this node's depth.
    # Independent of pose charts and hypothesis slots; None for legacy maps.
    metric_source: Optional[dict] = None
    # Per-component H(x|b); bias covariance is stored once by the manager.
    conditional_poses: Optional[list] = None

    def __post_init__(self):
        self.id = Keyframe._next_id
        Keyframe._next_id += 1

    def stored_image(self, name: str):
        """The tensor, store reference or None held by an image field, without decoding it."""
        return self.__dict__.get("_" + name)

    def has_image(self, name: str) -> bool:
        return self.__dict__.get("_" + name) is not None

    def __hash__(self):
        return self.id
    
    def __eq__(self, other):
        if not isinstance(other, Keyframe):
            return False
        return self.id == other.id

    def __repr__(self):
        return f"Keyframe(id={self.id}, timestamp={self.timestamp})"

@dataclass
class Particle:
    """
    A class for a particle.
    """
    pose: torch.Tensor # (7, ) translation and quaternion SE3
    weight: Optional[float] = 1.0
    id: int = field(init=False)
    _next_id: ClassVar[int] = 0  # Class variable for ID counter only

    def __post_init__(self):
        self.id = Particle._next_id
        Particle._next_id += 1

    def __repr__(self):
        return f"Particle(id={self.id}, weight={self.weight})"

        

class Edge:
    """
    Represents a relative pose constraint (factor) between two keyframes in the graph.
    """
    def __init__(
        self,
        mean: pp.LieTensor,
        std: pp.LieTensor,
        type: EdgeType = EdgeType.VISUAL,
        cost: Optional[float] = None,
    ):
        """
        Args:
            mean (pp.LieTensor): The relative pose measurement (SE3).
            std (pp.LieTensor): The diagonal std of the measurement (se3).
            type (EdgeType): The type of measurement (EdgeType enum).
            cost (float, optional): Pre-computed edge cost (translation norm).
                                   If None, will be computed from mean on first access.
        """
        self.mean: pp.LieTensor = mean
        self.std: pp.LieTensor = std
        # pypose's optimizer uses the information matrix (inverse of covariance) for weighting.
        # We ensure the diagonal is non-zero to prevent division by zero errors.
        self.information: torch.Tensor = torch.diag(1.0 / (std.tensor().flatten() + 1e-9))
        self.type = type
        self._cost = cost
        self.conditional_pose = None
        # measurement metadata used by the calibrated noise model (see cross/core/lc_verify.py)
        self.n_frames: Optional[int] = None   # odometry: number of integrated readings
        self.conf: Optional[float] = None     # visual: estimator confidence (covisibility)

    @property
    def cost(self) -> float:
        """Get the edge cost (translation norm). Computed lazily if not provided."""
        if self._cost is None:
            self._cost = float(self.mean_np[:3] @ self.mean_np[:3]) ** 0.5
        return self._cost

    @property
    def mean_np(self) -> np.ndarray:
        """Measurement as a (7,) float64 numpy array [x y z qx qy qz qw], cached (avoids repeated device syncs)."""
        m = getattr(self, "_mean_np", None)
        if m is None:
            m = self.mean.tensor().detach().cpu().numpy().astype(np.float64).reshape(-1)
            self._mean_np = m
        return m

    @property
    def std_np(self) -> np.ndarray:
        s = getattr(self, "_std_np", None)
        if s is None:
            s = self.std.tensor().detach().cpu().numpy().astype(np.float64).reshape(-1)
            self._std_np = s
        return s

class VisualEdge(Edge):
    def __init__(
        self,
        mean: pp.LieTensor,
        std: pp.LieTensor,
        type: EdgeType = EdgeType.VISUAL,
        from_comp_id: int = 0,
        to_comp_id: int = 0,
    ):
        super().__init__(mean, std, type)
        self.from_comp_id = from_comp_id
        self.to_comp_id = to_comp_id
