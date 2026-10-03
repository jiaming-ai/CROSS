"""One CROSS session: a local motion source feeding the CROSS back end (cross.core.system.System).

Two independent choices define a run:

  mode      rgbd | stereo | mono     what the back end observes.  rgbd: RGB + depth, PnP relative poses; stereo: stereo
                                     pairs, feed-forward multi-view relative poses with stereo scale; mono: RGB only,
                                     learned metric depth and monocular two-view / feed-forward relative poses
  odometry  external | visual        where the motion between frames comes from.  external: the dataset's odometry
                                     (wheel, VIO, simulated); visual: DPVO visual odometry whose metric scale comes
                                     from the mode's depth (sensor depth, stereo depth, or learned metric depth)

Every motion source feeds the same channel of the back end: a relative pose (and optionally its covariance) per frame,
accumulated into the odometry chain of the pose graph.  With external odometry in the rgbd and stereo modes the
frames go to System.step unchanged (the behaviour before the pipeline existed).
"""

from __future__ import annotations

import copy
import os

import numpy as np

MODES = ("rgbd", "stereo", "mono")
ODOMETRY = ("external", "visual")


def _inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def restrict_inputs(frame: dict, mode: str, odometry: str) -> dict:
    """The inputs a mode may use: mono sees no depth and no right image; visual odometry sees no odometry."""
    out = dict(frame)
    if mode == "mono":
        out["depth"] = None
        out["rgb_right"] = None
    elif mode == "rgbd":
        out["rgb_right"] = None
    if odometry == "visual":
        out["delta_pose"] = None
    return out


class OdometryFrontend:
    """External odometry as a motion source for a pipeline that maps every few frames (mono mode with odometry).

    The motion covariance is left to the back end's odometry model (tracking.odom_std_*), as in the rgbd and stereo
    modes; the frontend provides no depth (the pipeline's depth model predicts it for mapping frames)."""

    def __init__(self):
        self.index = 0
        self.pose = np.eye(4)

    def track(self, frame):
        from cross.mono.frontend import MonoEstimate
        delta = frame.get("delta_pose")
        delta = np.eye(4) if (delta is None or self.index == 0) else np.asarray(delta, dtype=np.float64)
        self.pose = self.pose @ delta
        self.index += 1
        return MonoEstimate(frame["timestamp"], self.pose.copy(), delta, None, None, dict(valid=True, frame=self.index - 1))

    def step(self, rgb, timestamp):
        return self.track({"rgb": rgb, "timestamp": timestamp})

    def shutdown(self):
        pass


class Pipeline:
    """A motion source and the CROSS back end.

    frontend None: every frame goes to System.step unchanged (external odometry in the rgbd / stereo modes).
    Otherwise the frontend turns each frame into a relative pose (+ covariance); the back end gets that motion every
    frame and an observation (image, depth, right image) every `mapping_interval` frames while tracking is valid.
    Between observations the reported pose is the frontend pose carried into the map frame by the alignment of the
    last observation.  `frontend_factory` builds a fresh frontend for a new session (load_map)."""

    def __init__(self, mapper, frontend=None, mapping_interval: int = 1, mode: str = "rgbd",
                 odometry: str = "external", depth_model=None, K=None, frontend_factory=None):
        if mode not in MODES or odometry not in ODOMETRY:
            raise ValueError(f"Unknown mode / odometry: {mode} / {odometry}")
        self.mapper = mapper
        self.frontend = frontend
        self.mapping_interval = int(mapping_interval)
        self.mode, self.odometry = mode, odometry
        self.depth_model, self.K = depth_model, K
        self.frontend_factory = frontend_factory
        self.map_alignment = np.eye(4)
        self.best_alignment = np.eye(4)
        self.initialized = False
        self.mapped_now = False
        self.last_estimate = None
        self.last_mapping_seconds = 0.0
        self._frames = 0

    # ------------------------------------------------------------------ back-end passthroughs
    @property
    def hypothesis_manager(self):
        return self.mapper.hypothesis_manager

    @property
    def last_added_kf_id(self):
        return self.mapper.last_added_kf_id

    def save_map(self, path):
        self.mapper.save_map(str(path))

    def load_map(self, path):
        """Start a new session in a stored map (fresh frontend, no alignment)."""
        if self.frontend is not None and getattr(self.frontend, "index", 0):
            if self.frontend_factory is None:
                raise RuntimeError("Load a map before processing images of a new session")
            self.frontend.shutdown() if hasattr(self.frontend, "shutdown") else None
            self.frontend = self.frontend_factory()
        self.map_alignment, self.best_alignment, self.initialized = np.eye(4), np.eye(4), False
        self.mapper.load_map(str(path))

    # ------------------------------------------------------------------ one frame
    def process(self, frame: dict):
        """Process one frame (loader dict: rgb, timestamp, and depth / rgb_right / delta_pose as the mode allows).
        (MonocularSystem.step(rgb, timestamp) is the monocular CLI's per-image entry point.)"""
        if self.frontend is None:
            self.mapper.step(obs=frame, data=frame)
            self.mapped_now = True
            return None
        frame = restrict_inputs(frame, self.mode, self.odometry)
        if frame.get("timestamp") is None:            # loaders without timestamps: frame count (DPVO needs them)
            frame["timestamp"] = float(self._frames)
        self._frames += 1
        estimate, _ = self.advance(frame)
        return estimate

    def advance(self, frame: dict):
        """Frontend step, back-end step, alignment.  Returns (estimate with the pose in the map frame, mapped now)."""
        from time import perf_counter
        import cv2
        frontend = self.frontend
        estimate = frontend.track(frame) if hasattr(frontend, "track") else frontend.step(frame["rgb"], frame["timestamp"])
        start = perf_counter()
        valid = bool(estimate.diagnostics.get("valid", True))
        index = getattr(frontend, "index", 1) - 1
        map_now = valid and (not self.initialized or index % self.mapping_interval == 0)
        rgb = frame["rgb"]
        depth = right = None
        if map_now:
            depth = frame.get("depth")
            if depth is None:
                depth = estimate.depth
            if depth is None and self.depth_model is not None:
                depth = self.depth_model.predict_metric(rgb, self.K, rgb.shape[:2])
            if depth is not None and depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
            right = frame.get("rgb_right")
        # Every input increment is accumulated; image retrieval / filtering runs every mapping_interval frames.
        self.mapper.step({
            "rgb": rgb if map_now else None,
            "depth": depth,
            "rgb_right": right,
            "conf": None,
            "delta_pose": estimate.delta_pose,
            "motion_covariance": estimate.motion_covariance,
            "timestamp": frame["timestamp"],
            "initial_chart_pose": estimate.pose.copy() if map_now and not self.initialized else None,
        })
        if map_now:
            mu, _, w = self.mapper.hypothesis_manager.dist
            mapped = self.mapper.get_current_pose().matrix().detach().cpu().numpy()
            best = mu[int(w.argmax())].matrix().detach().cpu().numpy()
            self.map_alignment = mapped @ _inverse(estimate.pose)
            self.best_alignment = best @ _inverse(estimate.pose)
            self.initialized = True
        self.mapped_now = map_now
        self.last_mapping_seconds = perf_counter() - start
        estimate.diagnostics["frontend_pose"] = estimate.pose.tolist()
        self.frontend_pose = estimate.pose.copy()
        estimate.pose = self.map_alignment @ estimate.pose
        self.last_estimate = estimate
        return estimate, map_now

    def belief(self, to_mat):
        """(pose of hypothesis 0, pose of the most likely hypothesis, weights) in the map frame, as 4x4 arrays.

        On observation frames (and always without a frontend) these are the back end's belief, converted by
        `to_mat` (pypose SE3 -> 4x4); in between, the frontend pose carried by the last alignment."""
        dist = self.mapper.hypothesis_manager.dist
        if dist is None:
            pose = self.last_estimate.pose if self.last_estimate is not None else np.eye(4)
            return pose, pose, np.ones(1)
        mu, _, w = dist
        w = w.detach().cpu().numpy()
        if self.frontend is None or self.mapped_now:
            return to_mat(mu[0]), to_mat(mu[int(np.argmax(w))]), w
        return self.map_alignment @ self.frontend_pose, self.best_alignment @ self.frontend_pose, w

    def release(self):
        """Shut the session down and drop its GPU models (a runner creates many sessions)."""
        import atexit
        import gc
        if self.frontend is not None and hasattr(self.frontend, "shutdown"):
            self.frontend.shutdown()
        system = self.mapper
        system.shutdown()
        try:
            atexit.unregister(system.shutdown)
        except Exception:
            pass
        for name in ("pose_est", "hypothesis_manager"):
            if hasattr(system, name):
                setattr(system, name, None)
        if getattr(system, "db", None) is not None:
            system.db.vpr_model = None
            system.db = None
        self.frontend = self.mapper = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------- construction
_MODEL_CACHE: dict = {}


def _cached(key, factory):
    """Heavy models shared by the sessions of one process (a query run starts a session per trial)."""
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = factory()
    return _MODEL_CACHE[key]


def mono_config_from_profile(profile, dpvo_checkpoint=None, extra_args=()):
    """MonoConfig from a profile JSON (configs/mono*.json: {"arguments": [cross.mono.run arguments]})."""
    import json
    from cross.mono.run import build_parser, config_from_args
    args = list(json.load(open(profile))["arguments"]) + list(extra_args)
    if dpvo_checkpoint:
        args += ["--dpvo-checkpoint", str(dpvo_checkpoint)]
    return config_from_args(build_parser().parse_args(["unused", "--output", "unused", *args]))


def visual_odometry_config(dpvo_checkpoint, seed=0, mask_people=False):
    """MonoConfig of the DPVO visual odometry of the rgbd / stereo modes: metric scale from the input depth."""
    from cross.mono.config import MonoConfig, ScaleConfig
    # input depth is a measurement, not a learned prior: a scale observation every frame, a tight floor (the learned
    # prior's floors, 0.12 / 0.08 in log scale, absorb its domain bias) and a faster-moving scale state (DPVO's unit
    # scale drifts by ~20 % over a minute; frontend-only on KITTI 07, 300 frames: ATE 0.63 m vs 1.02 m every 5 frames)
    scale = ScaleConfig(interval=1, observation_std_floor=0.03, posterior_std_floor=0.01, process_std_per_frame=0.02)
    return MonoConfig(frontend="dpvo", dpvo_checkpoint=str(dpvo_checkpoint), depth_input=True, mapping_interval=1,
                      seed=seed, mask_people=mask_people, mask_interval=3, scale=scale)


def build_session(mode: str, odometry: str, camera, system_config, *, T_right_in_left=None, mono_config=None,
                  vo_config=None, device="cuda", visualize=False) -> Pipeline:
    """A fresh session (empty map; call load_map to relocalize in a stored one).

    camera: cross.core.types.Camera of the input images; system_config: SystemConfig (copied); mono_config: MonoConfig
    of the mono mode (mono_config_from_profile); vo_config: MonoConfig of the visual odometry of the rgbd / stereo
    modes (visual_odometry_config)."""
    from cross.core.system import System
    from cross.core.types import Camera
    if mode not in MODES or odometry not in ODOMETRY:
        raise ValueError(f"Unknown mode / odometry: {mode} / {odometry}")
    K = np.array(camera.K, dtype=np.float64).copy()        # before System rescales camera.K to its storage size
    size = (int(camera.frame_width), int(camera.frame_height))
    cfg = copy.deepcopy(system_config)

    if mode in ("rgbd", "stereo"):
        system = System(visualize=visualize, debug=False, camera=Camera(K.copy(), *size), config=cfg,
                        T_right_in_left=T_right_in_left)
        if odometry == "external":
            return Pipeline(system, None, 1, mode, odometry)
        if vo_config is None:
            raise ValueError("Visual odometry needs vo_config (visual_odometry_config)")
        from cross.mono.dpvo_frontend import DPVOFrontend

        def factory():
            return DPVOFrontend(K, vo_config, device)
        return Pipeline(system, factory(), 1, mode, odometry, frontend_factory=factory)

    # mono
    if mono_config is None:
        raise ValueError("The mono mode needs a mono_config (mono_config_from_profile)")
    from cross.mono.models import DA3Geometry, DA3MetricDepth
    from cross.mono.system import MonocularSystem
    mc = mono_config
    geometry = _cached(("geometry", mc.pose_model, mc.resolution), lambda: DA3Geometry(mc.pose_model, device, mc.resolution))
    metric = _cached(("metric", mc.metric_model, mc.metric_resolution),
                     lambda: DA3MetricDepth(mc.metric_model, device, mc.metric_resolution))
    if odometry == "external":
        def factory():
            return OdometryFrontend()
    else:
        if mc.frontend != "dpvo":
            raise ValueError("Mono benchmark sessions use the synchronous DPVO frontend (--frontend dpvo)")
        from cross.mono.dpvo_frontend import DPVOFrontend

        def factory():
            return DPVOFrontend(K, mc, device, geometry_model=geometry, metric_model=metric)
    key = ("mono_pose_estimator", repr(mc))
    session = MonocularSystem(K, size, mc, system_config=cfg, device=device, frontend=factory(),
                              pose_estimator=_MODEL_CACHE.get(key), mono_defaults=True,
                              external_odometry=odometry == "external")
    _MODEL_CACHE[key] = session.mapper.pose_est
    session.mode, session.odometry = "mono", odometry
    session.frontend_factory = factory
    session.depth_model, session.K = metric, K
    return session


# ---------------------------------------------------------------------- command line (scripts/map_and_reloc*.py)
def add_session_args(ap):
    """Mode / odometry options shared by the map-and-relocalize runners."""
    import os
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="sensor mode (default: the runner's own: rgbd for posed RGB-D, stereo / rgbd for --estimator ff / pnp)")
    ap.add_argument("--odometry", choices=ODOMETRY, default="external",
                    help="external: the dataset's odometry; visual: DPVO visual odometry scaled by the mode's depth")
    ap.add_argument("--dpvo-checkpoint", default=os.environ.get("CROSS_DPVO_CHECKPOINT", "models/dpvo.pth"))
    ap.add_argument("--mono-profile", default="configs/mono_benchmark_10hz.json",
                    help="mono mode: cross.mono.run arguments (JSON with an 'arguments' list)")
    ap.add_argument("--mono-args", default="",
                    help="mono mode: extra cross.mono.run arguments after the profile's (one quoted string), e.g. "
                         "'--cross-config mapping.loop_closure.noise.odom_k_t=0.06'")
    ap.add_argument("--vo-mask-people", action="store_true", help="visual odometry: exclude detected people from DPVO patches")
    ap.add_argument("--fast", action="store_true",
                    help="stereo mode: faster preset (configs/stereo_fast.yaml: 6 images per pass instead of 10, ~35 %% "
                         "less time per observation; relocalization within noise on OpenLORIS, maps slightly less accurate)")


FAST_STEREO_PRESET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "stereo_fast.yaml")


def session_factory(args, camera, system_config, T_right_in_left=None, seed=0, visualize=False):
    """A callable building fresh sessions for the runner's --mode / --odometry (one per map run or query trial)."""
    mode, odometry = args.mode, args.odometry
    cfg = copy.deepcopy(system_config)
    mono_config = vo_config = None
    if mode == "mono":
        import shlex
        extra = shlex.split(args.mono_args or "") + ["--seed", str(seed)]
        mono_config = mono_config_from_profile(args.mono_profile, args.dpvo_checkpoint if odometry == "visual" else None,
                                               extra)
    elif odometry == "visual":
        # The back end keeps the odometry noise model of the mode (the depth-scaled DPVO drifts about as much as wheel
        # odometry, ~0.5 % on KITTI 07); the monocular DPVO noise constants (mapping.loop_closure.noise.odom_k_t = 1.0,
        # fitted for learned scale at 20 Hz indoors) made the verified loop closure ignore the chain at KITTI's
        # 0.6 m per frame (map ATE 9.3 m vs 1.0 m for the odometry alone on 300 frames).
        vo_config = visual_odometry_config(args.dpvo_checkpoint, seed=seed, mask_people=args.vo_mask_people)

    def make():
        return build_session(mode, odometry, camera, cfg, T_right_in_left=T_right_in_left, mono_config=mono_config,
                             vo_config=vo_config, visualize=visualize)
    return make

