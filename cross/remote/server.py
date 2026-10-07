"""The server half of a remote CROSS session (cross.remote): the back end and the GPU side of the VGGT-inertial
odometry, stepping the edge's messages in their order."""

from time import perf_counter

import numpy as np


def pose_matrix(p) -> np.ndarray:
    """pypose SE3 -> 4x4 (float64), quaternion normalised (as scripts/map_and_reloc.py pose_to_mat)."""
    from cross.utils.lie_tensor import normalize_SE3
    return normalize_SE3(p).matrix().detach().cpu().numpy().astype(np.float64)


class MapServer:
    """One session's back end (cross.core.system.System) and, with VGGT-inertial odometry, the odometry's pass service
    (cross.mono.vgio_service.VgioPassService).  handle(message) -> reply, in the edge's frame order:

    message  index, timestamp, delta_pose, motion_covariance, map_frame (the edge maps this frame: the back end steps
             it), initial_chart_pose, request (the odometry's measurement request), rgb / rgb_right / depth (only on
             the frames the back end will observe or the odometry measures)
    reply    index, map (the back end's belief at this frame: poses of hypothesis 0 and of the most likely one,
             weights), token + summary (the measurement), work (what the frame cost: observation, own pass, learned
             depth, stereo matching), server_seconds"""

    def __init__(self, system, service=None, keep_lie=True):
        self.system = system
        self.service = service
        self.keep_lie = keep_lie                 # in-process: the replies keep the pypose beliefs (exact conversion)
        self.stats = dict(messages=0, images=0, observed=0, deferred=0, own_passes=0, seconds=0.0)

    def handle(self, msg: dict, stale: bool = False) -> dict:
        """Step one message.  stale: it waited longer than the backlog budget (the server is behind the edge): the back
        end steps its motion but not its image (the observation waits for a fresh frame); the odometry's measurement
        is still made."""
        t0 = perf_counter()
        system, sv = self.system, self.service
        req = msg.get("request")
        map_frame = bool(msg.get("map_frame"))
        if msg.get("rgb") is not None:
            self.stats["images"] += 1
        if req is not None and sv is not None:
            sv.prune(req.get("keep_from"))
            sv.add_frame(req["index"], msg["rgb"], msg.get("rgb_right"))
        if (stale or msg.get("no_observation")) and msg.get("rgb") is not None and system.hypothesis_manager.dist is not None:
            msg = dict(msg, rgb=None, rgb_right=None, depth=None)     # (the back end defers the observation)
            key = "shed" if stale else "capped"
            self.stats[key] = self.stats.get(key, 0) + 1
        anchor = sv.anchor(req) if (sv is not None and req is not None and map_frame) else None
        own0 = sv.stats["own_calls"] if sv is not None else 0
        da30 = sv.stats.get("depth_priors", 0) if sv is not None else 0
        last_obs = getattr(system, "_last_obs_rgb", None)
        system.step({
            "rgb": msg.get("rgb") if map_frame else None,
            "depth": msg.get("depth") if map_frame else None,
            "rgb_right": msg.get("rgb_right") if map_frame else None,
            "conf": None,
            "delta_pose": msg.get("delta_pose"),
            "motion_covariance": msg.get("motion_covariance"),
            "timestamp": msg["timestamp"],
            "initial_chart_pose": msg.get("initial_chart_pose"),
            "frontend_anchor": anchor,
            "remote_frame": map_frame,
        })
        observed = map_frame and getattr(system, "_last_obs_rgb", None) is not last_obs
        deferred = bool(map_frame and system.last_step_diagnostics.get("observation_deferred")) if map_frame else False
        reply = {"index": msg["index"], "timestamp": msg["timestamp"], "last_added_kf_id": system.last_added_kf_id}
        if req is not None and sv is not None:
            backend_obs = getattr(getattr(system, "pose_est", None), "last_frontend_obs", None) if anchor is not None else None
            reply["token"] = req["token"]
            reply["summary"] = sv.measure(req, backend_obs)
        if map_frame and system.hypothesis_manager.dist is not None:
            mu, _, w = system.hypothesis_manager.dist
            reply["map"] = {"index": msg["index"],
                            "T0": system.get_current_pose().matrix().detach().cpu().numpy(),
                            "Tbest": mu[int(w.argmax())].matrix().detach().cpu().numpy(),
                            "w": w.detach().cpu().numpy(),
                            "B0": pose_matrix(mu[0]), "Bbest": pose_matrix(mu[int(w.argmax())]),
                            "localized": bool(system.session_localized()) if hasattr(system, "session_localized") else True}
            cfg = system.config.pose_est
            if cfg.obs_confident_max_interval_steps > 0:
                # the back end's adaptive cadence for its next frames (System._should_skip_observation), for the edge
                reply["map"]["confident"] = bool(getattr(system, "_last_obs_mapped", False)
                                                 and float(w.max()) >= cfg.obs_confident_weight)
            if self.keep_lie:
                wn = reply["map"]["w"]
                reply["map"].update(mu0=mu[0].clone(), mubest=mu[int(np.argmax(wn))].clone())
        own = (sv.stats["own_calls"] - own0) if sv is not None else 0
        reply["work"] = {"observed": bool(observed), "deferred": deferred, "own_pass": bool(own), "stale": bool(stale),
                         "depth": bool(sv is not None and sv.stats.get("depth_priors", 0) > da30),
                         "stereo": bool(req is not None and sv is not None and sv.T_rl is not None)}
        dt = perf_counter() - t0
        reply["server_seconds"] = dt
        self.stats["messages"] += 1
        self.stats["observed"] += int(observed)
        self.stats["deferred"] += int(deferred)
        self.stats["own_passes"] += own
        self.stats["seconds"] += dt
        return reply

    def cadence_state(self) -> dict:
        """The back end's observation-cadence state (read only), for the edge's copy (ObservationCadence.sync): a back
        end reused for a new session (load_map) keeps its frame and step counters and its motion since the last
        observation."""
        s = self.system
        acc = getattr(s, "odom_accumulator", None)
        means = getattr(acc, "_odoms_means", {}) if acc is not None else {}
        m_obs, m_step = means.get("since_last_obs"), means.get("since_last_step")
        T = None
        if m_obs is not None:
            from cross.utils.lie_tensor import normalize_se3
            T = normalize_se3(m_obs.Inv() @ acc._accumulated_odom).matrix().detach().cpu().numpy().astype(np.float64)
        return {"processed": int(getattr(s, "_processed_frame_num", 0)),
                "session_start": int(getattr(s, "_session_start_frame", 0)),
                "steps_since_obs": int(getattr(s, "_steps_since_obs", 0)), "T_since_obs": T,
                "kidnap": bool(acc is not None and "since_last_step" in means and m_step is None)}

    # ------------------------------------------------------------------ session
    def save_map(self, path):
        self.system.save_map(str(path))

    def load_map(self, path):
        self.system.load_map(str(path))

    def release(self):
        """Shut the back end down and drop its GPU models (as cross.pipeline.Pipeline.release)."""
        import atexit
        import gc
        system = self.system
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
        self.system = self.service = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
