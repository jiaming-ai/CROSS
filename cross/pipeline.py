"""One CROSS session: a local motion source feeding the CROSS back end (cross.core.system.System).

Two independent choices define a run:

  mode      rgbd | stereo | mono     what the back end observes.  rgbd: RGB + depth, PnP relative poses; stereo: stereo
                                     pairs, feed-forward multi-view relative poses with stereo scale; mono: RGB only,
                                     learned metric depth and monocular two-view / feed-forward relative poses
  odometry  external | visual | vio | vgio
                                     where the motion between frames comes from.  external: the dataset's odometry
                                     (wheel, VIO, simulated); visual: DPVO visual odometry whose metric scale comes
                                     from the mode's depth (sensor depth, stereo depth, or learned metric depth);
                                     vio (mono mode): DPVO with its metric scale from the IMU (cross/imu), learned depth
                                     as an optional weak prior; vgio (mono mode with the ff back end, or the stereo
                                     mode): no DPVO, the IMU carries the pose and VGGT-Omega's relative poses correct it
                                     in a local pose graph (cross/mono/vggt_imu_frontend), from the back end's forward
                                     passes where it observes; mono: learned depth as a prior on the metric scale,
                                     stereo: stereo depth observes it and the tracked corners' metric motion (PnP)
                                     is a factor too

The mono mode has two back ends (mono_estimator): da3, the monocular system of cross/mono (learned metric depth,
two-view / feed-forward DA3 relative poses), and ff, the stereo mode's feed-forward multi-view estimator (VGGT-Omega) on
the single image, whose metric scale comes from the motion (the previous observation and the metric odometry between
them) and from the map (pairs of retrieved keyframes with known relative pose) instead of a stereo pair.

Every motion source feeds the same channel of the back end: a relative pose (and optionally its covariance) per frame,
accumulated into the odometry chain of the pose graph.  With external odometry in the rgbd and stereo modes the
frames go to System.step unchanged (the behaviour before the pipeline existed).
"""

from __future__ import annotations

import copy
import os

import numpy as np

MODES = ("rgbd", "stereo", "mono")
ODOMETRY = ("external", "visual", "vio", "vgio")
INERTIAL = ("vio", "vgio")
MONO_ESTIMATORS = ("da3", "ff")


def _inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def restrict_inputs(frame: dict, mode: str, odometry: str) -> dict:
    """The inputs a mode may use: mono sees no depth and no right image; visual odometry sees no odometry; only
    visual-inertial odometry sees the IMU."""
    out = dict(frame)
    if mode == "mono":
        out["depth"] = None
        out["rgb_right"] = None
    elif mode == "rgbd":
        out["rgb_right"] = None
    if odometry in ("visual",) + INERTIAL:
        out["delta_pose"] = None
    if odometry not in INERTIAL:
        for key in ("imu", "imu_t0", "imu_t1", "imu_calib"):
            out.pop(key, None)
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
        self.continuous_start_in_map = False

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
        if self.frontend is not None and hasattr(self.frontend, "continuous_start"):
            self.frontend.continuous_start = self.continuous_start_in_map
        self.mapper.load_map(str(path))

    # ------------------------------------------------------------------ one frame
    def process(self, frame: dict):
        """Process one frame (loader dict: rgb, timestamp, and depth / rgb_right / delta_pose as the mode allows).
        (MonocularSystem.step(rgb, timestamp) is the monocular CLI's per-image entry point.)"""
        if self.frontend is None:
            if self.mode == "mono":           # (rgbd / stereo: frames unchanged, as before the pipeline existed)
                frame = restrict_inputs(frame, self.mode, self.odometry)
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
        # a frontend with a continuous start reports unknown motion (wide covariance) while it initializes: the back
        # end may observe then (relocalization in a loaded map needs no metric motion)
        observable = valid or bool(estimate.diagnostics.get("unknown_motion", False))
        index = getattr(frontend, "index", 1) - 1
        map_now = observable and (not self.initialized or index % self.mapping_interval == 0)
        # a frontend measuring with the back end's forward pass (vgio) gives its last measured frame as the temporal
        # anchor of the pass, and reads the pass's raw output afterwards
        anchor = frontend.backend_anchor(map_now) if map_now and hasattr(frontend, "backend_anchor") else None
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
            "frontend_anchor": anchor,
        })
        if hasattr(frontend, "after_backend"):
            pose_est = getattr(self.mapper, "pose_est", None)
            frontend.after_backend(getattr(pose_est, "last_frontend_obs", None) if anchor is not None else None)
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


def mono_ff_config(cfg):
    """The stereo mode's feed-forward estimator for the mono mode (in place): no stereo anchors; the metric scale of a
    forward pass comes from the previous observation with the odometry between them and from pairs of references with
    their relative pose in the map.  Observation cadence of the stereo mode (configs/stereo.yaml, the benchmark's
    cross_stereo arguments)."""
    from cross.core.config import PoseEstType
    cfg.pose_est.type = PoseEstType.FF
    ff = cfg.pose_est.ff
    ff.use_curr_anchor = False
    ff.n_ref_anchors = 0
    ff.use_odom_anchor = True
    ff.use_map_anchors = True
    cfg.pose_est.obs_min_translation = 0.3
    cfg.pose_est.obs_min_rotation = 0.15
    cfg.pose_est.obs_max_interval_steps = 3
    return cfg


def build_session(mode: str, odometry: str, camera, system_config, *, T_right_in_left=None, mono_config=None,
                  vo_config=None, device="cuda", visualize=False, mono_estimator: str = "da3",
                  continuous_start: bool = True) -> Pipeline:
    """A fresh session (empty map; call load_map to relocalize in a stored one).

    camera: cross.core.types.Camera of the input images; system_config: SystemConfig (copied); mono_config: MonoConfig
    of the mono mode (mono_config_from_profile; with odometry vio, its imu.enabled is set); vo_config: MonoConfig of
    the visual odometry of the rgbd / stereo modes (visual_odometry_config); mono_estimator: the mono mode's back end
    (da3 | ff; ff needs system_config prepared by mono_ff_config)."""
    from cross.core.system import System
    from cross.core.types import Camera
    if mode not in MODES or odometry not in ODOMETRY or mono_estimator not in MONO_ESTIMATORS:
        raise ValueError(f"Unknown mode / odometry / mono estimator: {mode} / {odometry} / {mono_estimator}")
    if odometry == "vio" and mode != "mono" or odometry == "vgio" and mode not in ("mono", "stereo"):
        raise ValueError("Visual-inertial odometry: vio in the mono mode, vgio in the mono and stereo modes")
    if odometry == "vgio" and mode == "mono" and mono_estimator != "ff":
        raise ValueError("vgio uses the feed-forward back end's model (--mono-estimator ff)")
    K = np.array(camera.K, dtype=np.float64).copy()        # before System rescales camera.K to its storage size
    size = (int(camera.frame_width), int(camera.frame_height))
    cfg = copy.deepcopy(system_config)
    mc = mono_config
    if odometry in INERTIAL:
        if mc is None:
            raise ValueError("Visual-inertial odometry needs a mono_config (mono_config_from_profile; its imu section)")
        mc = copy.deepcopy(mc)
        mc.imu.enabled = True

    if mode == "mono" and mono_estimator == "ff":
        system = System(visualize=visualize, debug=False, camera=Camera(K.copy(), *size), config=cfg)
        if odometry == "external":
            return Pipeline(system, None, 1, mode, odometry)
        if odometry == "vgio":
            return _vgio_pipeline(system, mc, K, device, mode, continuous_start)
        if mc is None or mc.frontend != "dpvo":
            raise ValueError("The mono mode's visual odometry needs a mono_config with the dpvo frontend")
        from cross.mono.dpvo_frontend import DPVOFrontend
        from cross.mono.models import DA3MetricDepth
        # learned metric depth only for DPVO's scale observations (the feed-forward back end needs no depth)
        learned_scale = not (mc.imu.enabled and not mc.imu.depth_prior)
        metric = _cached(("metric", mc.metric_model, mc.metric_resolution),
                         lambda: DA3MetricDepth(mc.metric_model, device, mc.metric_resolution)) if learned_scale else None

        def factory():
            frontend = DPVOFrontend(K, mc, device, metric_model=metric, load_metric=learned_scale)
            frontend.provide_mapping_depth = False
            return frontend
        pipeline = Pipeline(system, factory(), 1, mode, odometry, frontend_factory=factory)
        # in a loaded map the feed-forward back end takes its scale from the map: it can relocalize while DPVO
        # initializes (a new map waits for DPVO's metric trajectory, so that its first keyframes are placed right)
        pipeline.continuous_start_in_map = continuous_start
        return pipeline

    if mode in ("rgbd", "stereo"):
        if odometry == "vgio":
            # the odometry scale guard keeps its sample filter but does not rescale this odometry (LoopClosureConfig.
            # odom_guard_rescale): it is metric itself, and the back end's passes collapse with the frontend's (KITTI 01)
            cfg.mapping.loop_closure.odom_guard_rescale = False
        system = System(visualize=visualize, debug=False, camera=Camera(K.copy(), *size), config=cfg,
                        T_right_in_left=T_right_in_left)
        if odometry == "external":
            return Pipeline(system, None, 1, mode, odometry)
        if odometry == "vgio":
            return _vgio_pipeline(system, mc, K, device, mode, continuous_start, T_right_in_left)
        if vo_config is None:
            raise ValueError("Visual odometry needs vo_config (visual_odometry_config)")
        from cross.mono.dpvo_frontend import DPVOFrontend

        def factory():
            return DPVOFrontend(K, vo_config, device)
        return Pipeline(system, factory(), 1, mode, odometry, frontend_factory=factory)

    # mono
    if mc is None:
        raise ValueError("The mono mode needs a mono_config (mono_config_from_profile)")
    from cross.mono.models import DA3Geometry, DA3MetricDepth
    from cross.mono.system import MonocularSystem
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
                              external_odometry=odometry == "external", odometry=odometry)
    _MODEL_CACHE[key] = session.mapper.pose_est
    session.mode, session.odometry = "mono", odometry
    session.frontend_factory = factory
    session.depth_model, session.K = metric, K
    return session


def _vgio_config(mc, mode, T_right_in_left=None):
    """(In place) the stereo mode's VGGT-inertial odometry takes the metric scale from the stereo pair: no learned
    depth and no bias state for it."""
    if mode == "stereo":
        if T_right_in_left is None:
            raise ValueError("The stereo mode's VGGT-inertial odometry needs the stereo calibration (T_right_in_left)")
        mc.imu.depth_prior = False
        mc.imu.vgio_depth_bias = False
    return mc


def vgio_frontend(K, mc, mode, T_right_in_left=None, device="cuda", system=None, metric=None):
    """The VGGT-Omega + IMU frontend (cross/mono/vggt_imu_frontend): with the back end (system) its pass service runs
    in-process on the back end's model and image transforms; without, it has none (the edge of a remote session,
    cross/remote, whose server holds it)."""
    from cross.mono.vggt_imu_frontend import VggtImuFrontend
    service = dict(backend=system.pose_est.backend, rgb_transform=system.rgb_transform,
                   depth_transform=system.depth_transform, metric_model=metric) if system is not None \
        else dict(local_service=False)
    frontend = VggtImuFrontend(K, mc, device, interval=mc.imu.visual_interval, depth_every=mc.imu.depth_every,
                               context=mc.imu.vgio_context, keyframe_age=mc.imu.vgio_keyframe_age,
                               visual_rotation=mc.imu.vgio_visual_rotation, graph=mc.imu.vgio_graph,
                               T_right_in_left=T_right_in_left if mode == "stereo" else None, **service)
    frontend.standalone = False
    return frontend


def _vgio_pipeline(system, mc, K, device, mode, continuous_start, T_right_in_left=None) -> Pipeline:
    """The VGGT-Omega + IMU frontend (cross/mono/vggt_imu_frontend) on the back end's model and image transforms.  The
    mono mode takes the metric scale from learned depth (DA3, with its bias in the graph), the stereo mode from the
    current stereo pair (no learned depth)."""
    _vgio_config(mc, mode, T_right_in_left)
    metric = None
    if mc.imu.depth_prior and mc.imu.depth_prior_source != "head":
        from cross.mono.models import DA3MetricDepth
        metric = _cached(("metric", mc.metric_model, mc.metric_resolution),
                         lambda: DA3MetricDepth(mc.metric_model, device, mc.metric_resolution))

    def vgio_factory():
        return vgio_frontend(K, mc, mode, T_right_in_left, device, system=system, metric=metric)
    pipeline = Pipeline(system, vgio_factory(), 1, mode, "vgio", frontend_factory=vgio_factory)
    pipeline.continuous_start_in_map = continuous_start
    return pipeline


def edge_session(mode: str, odometry: str, camera, system_config, link_factory, *, T_right_in_left=None,
                 mono_config=None, mono_estimator: str = "da3", upload: str = "predicted", obs_cap: float = 0.0):
    """The edge of a remote session behind a real link (cross/remote/grpc_link.py): the odometry only, no GPU model.
    link_factory(open message) -> link; the open message carries what the server needs to build the same session
    (build_session): mode, odometry, camera, stereo calibration and the resolved configurations."""
    from cross.core.config import config_to_dict
    from cross.core.config import _to_dict
    from cross.remote.edge import ObservationCadence, RemotePipeline
    if odometry not in ("external", "vgio"):
        raise ValueError("Remote sessions: external odometry or VGGT-inertial odometry (--odometry vgio)")
    K = np.array(camera.K, dtype=np.float64).copy()
    cfg = copy.deepcopy(system_config)
    mc = copy.deepcopy(mono_config)
    frontend = factory = None
    if odometry == "vgio":
        mc.imu.enabled = True
        edge_mc = _vgio_config(copy.deepcopy(mc), mode, T_right_in_left)

        def factory():
            return vgio_frontend(K, edge_mc, mode, T_right_in_left, device="cpu")
        frontend = factory()
    open_msg = {"mode": mode, "odometry": odometry, "mono_estimator": mono_estimator,
                "camera": {"K": K, "width": int(camera.frame_width), "height": int(camera.frame_height)},
                "T_right_in_left": None if T_right_in_left is None else np.asarray(T_right_in_left, dtype=np.float64),
                "system_config": config_to_dict(cfg), "mono_config": None if mc is None else _to_dict(mc)}
    return RemotePipeline(frontend, link_factory(open_msg), mode, odometry, ObservationCadence(cfg.pose_est), 1, upload,
                          frontend_factory=factory, server=None, continuous_start_in_map=odometry == "vgio", K=K,
                          obs_cap=obs_cap)


def server_session(open_msg, device="cuda"):
    """The server of a remote session from the edge's open message (edge_session): (MapServer, factory of a new pass
    service for a new session of the edge's odometry, information for the edge)."""
    from cross.core.config import SystemConfig, _from_dict
    from cross.core.types import Camera
    from cross.mono.config import MonoConfig
    from cross.remote.server import MapServer
    cam = open_msg["camera"]
    camera = Camera(K=np.asarray(cam["K"], dtype=np.float64), frame_width=int(cam["width"]),
                    frame_height=int(cam["height"]))
    cfg = _from_dict(SystemConfig, open_msg["system_config"])
    mc = None if open_msg.get("mono_config") is None else _from_dict(MonoConfig, open_msg["mono_config"])
    p = build_session(open_msg["mode"], open_msg["odometry"], camera, cfg, T_right_in_left=open_msg.get("T_right_in_left"),
                      mono_config=mc, device=device, mono_estimator=open_msg.get("mono_estimator", "da3"))
    service = p.frontend.service if p.frontend is not None else None
    factory = (lambda: p.frontend_factory().service) if p.frontend_factory is not None else None
    import torch
    info = {"gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    return MapServer(p.mapper, service, keep_lie=False), factory, info


# ---------------------------------------------------------------------- command line (scripts/map_and_reloc*.py)
def add_session_args(ap):
    """Mode / odometry options shared by the map-and-relocalize runners."""
    import os
    ap.add_argument("--mode", choices=MODES, default=None,
                    help="sensor mode (default: the runner's own: rgbd for posed RGB-D, stereo / rgbd for --estimator ff / pnp)")
    ap.add_argument("--odometry", choices=ODOMETRY, default="external",
                    help="external: the dataset's odometry; visual: DPVO visual odometry scaled by the mode's depth; "
                         "vio (mono mode): DPVO scaled by the IMU (imu.txt / imu.json next to the images); "
                         "vgio (mono mode with --mono-estimator ff, or stereo mode): IMU odometry corrected by "
                         "VGGT-Omega relative poses in a local pose graph (stereo: metric scale from the stereo pair)")
    ap.add_argument("--mono-estimator", choices=MONO_ESTIMATORS, default="da3",
                    help="mono mode back end: da3 (learned metric depth, cross/mono) or ff (the stereo mode's "
                         "VGGT-Omega multi-view estimator, metric scale from the odometry and the map)")
    ap.add_argument("--dpvo-checkpoint", default=os.environ.get("CROSS_DPVO_CHECKPOINT", "models/dpvo.pth"))
    ap.add_argument("--mono-profile", default="configs/mono_benchmark_10hz.json",
                    help="mono mode (and the IMU options of --odometry vgio): cross.mono.run arguments (JSON with an "
                         "'arguments' list)")
    ap.add_argument("--mono-args", default="",
                    help="mono mode / --odometry vgio: extra cross.mono.run arguments after the profile's (one quoted "
                         "string), e.g. '--cross-config mapping.loop_closure.noise.odom_k_t=0.06' or "
                         "'--imu-config vgio_stereo_std=0.03'")
    ap.add_argument("--vo-mask-people", action="store_true", help="visual odometry: exclude detected people from DPVO patches")
    ap.add_argument("--fast", action="store_true",
                    help="stereo mode: faster preset (configs/stereo_fast.yaml: 6 images per pass instead of 10, ~35 %% "
                         "less time per observation; relocalization within noise on OpenLORIS, maps slightly less accurate)")
    g = ap.add_argument_group("remote session (cross/remote): the back end and the GPU work on a server, the odometry "
                              "on the edge, behind a simulated network on the dataset's clock")
    g.add_argument("--remote", action="store_true",
                   help="split the session (external or --odometry vgio): with the defaults below the replies come back "
                        "after the server's modelled compute time; --remote-compute zero with --remote-rtt 0 reproduces "
                        "the local session")
    g.add_argument("--remote-rtt", type=float, default=0.0, help="network round-trip time (s)")
    g.add_argument("--remote-jitter", type=float, default=0.0, help="mean of an exponential extra delay per direction (s)")
    g.add_argument("--remote-compute", choices=("model", "measured", "zero"), default="model",
                   help="server time per message: model (cross/remote/link.py COMPUTE_MODEL, --remote-costs), the measured "
                        "wall time of this machine, or zero")
    g.add_argument("--remote-costs", default="", help="compute model overrides, e.g. 'observe=0.08,own_pass=0.03'")
    g.add_argument("--remote-outage", default="",
                   help="link outages, seconds after the session's first frame: 'start:duration[,start:duration]' or "
                        "'every:period:duration' (from one period on)")
    g.add_argument("--remote-jpeg", type=int, default=0, help="JPEG quality of the uploaded images (0: lossless)")
    g.add_argument("--remote-uplink-mbps", type=float, default=0.0, help="uplink bandwidth (0: unlimited)")
    g.add_argument("--remote-upload", choices=("predicted", "all"), default="predicted",
                   help="images of the frames the back end will observe and the odometry measures, or of every mapped frame")
    g.add_argument("--remote-seed", type=int, default=0, help="seed of the jitter")
    g.add_argument("--remote-max-backlog", type=float, default=0.3,
                   help="overload policy (s, 0: off): a frame that waited longer than this on the server is stepped "
                        "without its observation (the back end observes the next fresh frame); measurements are kept.  "
                        "It acts only when the server is behind (outputs/2026-10-06_remote_mode: it never fired where "
                        "the server kept up, and it kept every overloaded case close to the local session)")
    g.add_argument("--remote-obs-cap", type=float, default=0.1,
                   help="rate cap (s, 0: off): the edge sends an observation only if the server can start it within "
                        "this time (a model of the server's queue from the server time of each kind of message, learned "
                        "from the replies); otherwise the back end observes at the next frame it can.  It acts only "
                        "when the server is behind; there it kept the results of the overload policy and sent 9-35 %% "
                        "fewer images (outputs/2026-10-07_remote_cadence)")
    g.add_argument("--remote-server", default="",
                   help="HOST:PORT of a remote-session server (scripts/remote/serve.py): a real gRPC link instead of the "
                        "simulated one; this process runs only the edge")
    g.add_argument("--remote-extra-delay", type=float, default=0.0,
                   help="real link: round-trip delay added on top of the network's (s), half in each direction")
    g.add_argument("--remote-realtime", action="store_true",
                   help="real link: feed the frames at their timestamps (wall clock), as a sensor would")


FAST_STEREO_PRESET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "stereo_fast.yaml")


def session_factory(args, camera, system_config, T_right_in_left=None, seed=0, visualize=False):
    """A callable building fresh sessions for the runner's --mode / --odometry (one per map run or query trial)."""
    mode, odometry = args.mode, args.odometry
    mono_estimator = getattr(args, "mono_estimator", "da3") if mode == "mono" else "da3"
    cfg = copy.deepcopy(system_config)
    if mono_estimator == "ff":
        mono_ff_config(cfg)
    mono_config = vo_config = None
    if mode == "mono" or odometry == "vgio":
        # the mono mode's configuration; for the stereo mode's VGGT-inertial odometry only its imu section is used
        import shlex
        extra = shlex.split(args.mono_args or "") + ["--seed", str(seed)] + (["--imu"] if odometry in INERTIAL else [])
        mono_config = mono_config_from_profile(args.mono_profile,
                                               args.dpvo_checkpoint if odometry in ("visual",) + INERTIAL else None, extra)
    elif odometry == "visual":
        # The back end keeps the odometry noise model of the mode (the depth-scaled DPVO drifts about as much as wheel
        # odometry, ~0.5 % on KITTI 07); the monocular DPVO noise constants (mapping.loop_closure.noise.odom_k_t = 1.0,
        # fitted for learned scale at 20 Hz indoors) made the verified loop closure ignore the chain at KITTI's
        # 0.6 m per frame (map ATE 9.3 m vs 1.0 m for the odometry alone on 300 frames).
        vo_config = visual_odometry_config(args.dpvo_checkpoint, seed=seed, mask_people=args.vo_mask_people)

    def make():
        if getattr(args, "remote_server", ""):
            from cross.remote.grpc_link import GrpcLink

            def link(open_msg):
                return GrpcLink(args.remote_server, dict(open_msg, max_backlog=args.remote_max_backlog),
                                jpeg=args.remote_jpeg, extra_delay=args.remote_extra_delay, realtime=args.remote_realtime)
            return edge_session(mode, odometry, camera, cfg, link, T_right_in_left=T_right_in_left,
                                mono_config=mono_config, mono_estimator=mono_estimator, upload=args.remote_upload,
                                obs_cap=args.remote_obs_cap)
        session = build_session(mode, odometry, camera, cfg, T_right_in_left=T_right_in_left, mono_config=mono_config,
                                vo_config=vo_config, visualize=visualize, mono_estimator=mono_estimator)
        if getattr(args, "remote", False):
            from cross.remote import remote_session
            session = remote_session(session, remote_link_factory(args), upload=args.remote_upload,
                                     obs_cap=args.remote_obs_cap)
        return session
    return make


def _parse_outages(spec: str, horizon: float = 7200.0):
    out = []
    for part in [p for p in spec.split(",") if p.strip()]:
        f = part.split(":")
        if f[0] == "every":
            period, duration = float(f[1]), float(f[2])
            out += [(k * period, duration) for k in range(1, int(horizon / period) + 1)]
        else:
            out.append((float(f[0]), float(f[1])))
    return out


def remote_link_factory(args):
    """server -> the simulated link of the runner's --remote-* options (cross/remote/link.py)."""
    from cross.remote.link import SimLink
    costs = {k: float(v) for k, v in (kv.split("=") for kv in args.remote_costs.split(",") if kv.strip())}
    outages = _parse_outages(args.remote_outage)

    def make(server):
        return SimLink(server, rtt=args.remote_rtt, jitter=args.remote_jitter, compute=args.remote_compute,
                       costs=costs, outages=outages, jpeg=args.remote_jpeg, uplink_mbps=args.remote_uplink_mbps,
                       seed=args.remote_seed, max_backlog=args.remote_max_backlog)
    return make

