"""The edge half of a remote CROSS session (cross.remote): the odometry, the messages to the server, and the map-frame
pose from the server's late replies."""

import numpy as np

from cross.pipeline import _inverse, restrict_inputs


class ObservationCadence:
    """The back end's observation cadence (cross.core.system.System._should_skip_observation: every frame for
    obs_warmup_steps frames after an initialization, then after obs_min_translation m, obs_min_rotation rad or
    obs_max_interval_steps mapped frames since the last observation) on the edge's own odometry: the frames whose images
    the server needs.  The back end decides on the same motion (the deltas the edge sends), so the two agree up to
    rounding; a frame the back end wants without its image is observed at the next one that has it.  Its adaptive
    relaxation while one hypothesis dominates (obs_confident_*) follows the server's last reply (confident): exact
    without latency; with latency the edge sends a few images the back end skips, or the back end observes a frame later."""

    def __init__(self, pose_est_cfg):
        c = self.cfg = pose_est_cfg
        self.min_t, self.min_r = float(c.obs_min_translation), float(c.obs_min_rotation)
        self.max_steps, self.warmup = int(c.obs_max_interval_steps), int(c.obs_warmup_steps)
        self.every_frame = self.min_t <= 0 and self.min_r <= 0 and self.max_steps <= 1
        relax = int(getattr(c, "obs_confident_max_interval_steps", 0))
        self.relaxed = (relax, float(c.obs_confident_min_translation), float(c.obs_confident_min_rotation)) if relax > 0 else None
        self.confident = False                   # the back end's state in its last reply (MapServer: map.confident)
        self.processed = self.start = self.steps = 0
        self.T = np.eye(4)
        self.missing = False

    def sync(self, state):
        """Continue from the back end's actual state (MapServer.cadence_state, after a map load)."""
        self.processed, self.start = int(state["processed"]), int(state["session_start"])
        self.steps = int(state["steps_since_obs"])
        T = state.get("T_since_obs")
        self.T = np.eye(4) if T is None else np.asarray(T, dtype=np.float64)
        self.unknown = T is None and self.processed > 0       # its motion since the observation is not known: observe
        self.missing = bool(state.get("kidnap"))

    def frame(self, delta, map_frame: bool) -> bool:
        """The motion of this frame (None: missing) and whether the back end steps it; True if it will observe it (or
        initialize on it)."""
        self._undo = None
        if delta is None:
            self.missing = True                  # the back end re-initializes at its next step (kidnapped)
        else:
            self.T = self.T @ np.asarray(delta, dtype=np.float64)
        if not map_frame:
            return False
        self.processed += 1
        if self.processed == 1 or self.missing:
            self.missing = False
            self.start, self.steps, self.T = self.processed, 0, np.eye(4)
            return True                          # (the image is needed: no veto)
        if self.every_frame:
            return True
        self.steps += 1
        unknown = getattr(self, "unknown", False)
        max_steps, min_t, min_r = self.max_steps, self.min_t, self.min_r
        if self.relaxed is not None and self.confident:
            max_steps, min_t, min_r = self.relaxed
        if self.processed - self.start <= self.warmup or self.steps >= max_steps or unknown:
            observe = True
        else:
            moved = min_t > 0 and float(np.linalg.norm(self.T[:3, 3])) >= min_t
            angle = float(np.arccos(np.clip((np.trace(self.T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))
            observe = moved or (min_r > 0 and angle >= min_r)
        if observe:
            self._undo = (self.steps, self.T, unknown)
            self.steps, self.T, self.unknown = 0, np.eye(4), False
        return observe

    def can_veto(self) -> bool:
        return self._undo is not None

    def veto(self):
        """This frame's image is not sent after all (the rate cap): the back end defers the observation to the next
        frame with an image, and the copy keeps counting as it does."""
        if self._undo is not None:
            self.steps, self.T, self.unknown = self._undo
            self._undo = None


class RemotePipeline:
    """A CROSS session whose back end runs behind a link (cross.remote.link): the edge's side, with the interface of
    cross.pipeline.Pipeline (process, belief, save_map, load_map, release).

    frontend  the VGGT-inertial odometry without its pass service (VggtImuFrontend(local_service=False)), or None:
              the frames' external odometry (delta_pose) is the motion
    link      send(message, t), poll(t) -> replies that have arrived by the dataset time t
    upload    "predicted": images only on the frames the back end will observe (ObservationCadence) or the odometry
              measures; "all": on every mapped frame (the local session's inputs exactly)
    server    the in-process server of a simulated link (keyframes, map files), or None

    The pose of frame t in the map frame is the back end's pose of the last replied mapped frame b carried to t by the
    odometry: T_map(b) T_odom(b)^-1 T_odom(t)."""

    def __init__(self, frontend, link, mode, odometry, cadence, mapping_interval=1, upload="predicted",
                 frontend_factory=None, server=None, continuous_start_in_map=False, depth_model=None, K=None,
                 obs_cap=0.0, send_right=True):
        if upload not in ("predicted", "all"):
            raise ValueError(f"Unknown upload policy {upload}")
        self.frontend, self.link, self.server = frontend, link, server
        self.mode, self.odometry = mode, odometry
        self.cadence, self.upload = cadence, upload
        self._pose_est_cfg = cadence.cfg
        self.mapping_interval = int(mapping_interval)
        self.frontend_factory = frontend_factory
        self.continuous_start_in_map = continuous_start_in_map
        self.depth_model, self.K = depth_model, K
        self.send_right = bool(send_right)       # False: the back end observes on the left image only (ff.right_image)
        self.initialized = self.mapped_now = False
        self.index = -1                          # frames processed in this session
        self._frames = 0
        self._fpose = {}                         # frame index -> odometry pose reported at that frame
        self._odom = np.eye(4)
        self._map = None                         # the last map reply
        self._sent = {}                          # frame index -> dataset time it was sent
        self.last_estimate = None
        self.frontend_pose = None
        self.stats = dict(frames=0, uploads=0, requests=0, replies=0, capped=0, map_lag_frames=[], measurement_lag_s=[],
                          map_lag_s=[])
        # rate cap (obs_cap s, 0: off): an observation is sent only if the server can start it within obs_cap s, by a
        # model of its queue in send time from the server time of each kind of message, learned from the replies
        self.obs_cap = float(obs_cap or 0.0)
        self._costs = {}                         # (observed, request) -> server seconds (running mean)
        self._q_free = -np.inf                   # when the server will have finished what was sent (send-time clock)
        self._kind = {}                          # frame index -> request sent with it
        self._inflight = {}                      # frame index -> (time sent, its expected server seconds), in order
        self._net = np.inf                       # the network's delay: the smallest (reply lag - server seconds)

    # ------------------------------------------------------------------ server passthroughs (simulated link)
    @property
    def mapper(self):
        return None if self.server is None else self.server.system

    @property
    def hypothesis_manager(self):
        if self.server is not None:
            return self.server.system.hypothesis_manager
        return _RemoteKeyframes(self.link)

    def keyframe_frames(self):
        """Behind a real link: the frame index each keyframe was made at (the replies naming it come late), else None."""
        if self.server is not None:
            return None
        self.link.flush()
        return {int(k): int(v) for k, v in self.link.call("keyframes")["frames"].items()}

    @property
    def last_added_kf_id(self):
        return self.server.system.last_added_kf_id if self.server is not None else self.link.last_added_kf_id

    def save_map(self, path):
        self.link.flush()
        if self.server is not None:
            self.server.save_map(str(path))
        else:
            self.link.call("save_map", path=str(path))

    def load_map(self, path):
        """Start a new session in a stored map (fresh frontend, no alignment)."""
        replaced = False
        if self.frontend is not None and getattr(self.frontend, "index", 0):
            if self.frontend_factory is None:
                raise RuntimeError("Load a map before processing images of a new session")
            self.frontend.shutdown()
            self.frontend = self.frontend_factory()
            replaced = True
        self.initialized, self._map, self._fpose, self._odom = False, None, {}, np.eye(4)
        self._sent, self.index = {}, -1
        self.cadence = ObservationCadence(self._pose_est_cfg)
        if hasattr(self.link, "flush") and self.server is None:
            self.link.flush()
        if hasattr(self.link, "reset"):
            self.link.reset()
        if self.frontend is not None:
            self.frontend.continuous_start = self.continuous_start_in_map
        if self.server is not None:
            self.server.load_map(str(path))
            self.cadence.sync(self.server.cadence_state())
        else:
            r = self.link.call("load_map", path=str(path), new_service=replaced)
            if r.get("cadence") is not None:
                self.cadence.sync(r["cadence"])

    def release(self):
        if self.frontend is not None:
            self.frontend.shutdown()
        if self.server is not None:
            self.server.release()
        elif hasattr(self.link, "close"):
            self.link.close()
        self.frontend = self.server = None

    # ------------------------------------------------------------------ one frame
    def process(self, frame: dict):
        if self.frontend is not None or self.mode == "mono":
            frame = restrict_inputs(frame, self.mode, self.odometry)
        if self.frontend is not None and frame.get("timestamp") is None:
            frame["timestamp"] = float(self._frames)
        self._frames += 1
        t = float(frame["timestamp"])
        if hasattr(self.link, "pace"):
            self.link.pace(t)                    # a real link in real time: the frame is due at its timestamp
        self._receive(t)                         # the replies that arrived since the last frame
        self.index += 1
        i = self.index
        rgb = frame["rgb"]
        estimate = None
        if self.frontend is not None:
            estimate = self.frontend.track(frame)
            valid = bool(estimate.diagnostics.get("valid", True))
            observable = valid or bool(estimate.diagnostics.get("unknown_motion", False))
            index = getattr(self.frontend, "index", 1) - 1
            map_now = observable and (not self.initialized or index % self.mapping_interval == 0)
            req, self.frontend.request = self.frontend.request, None
            delta, cov, fpose = estimate.delta_pose, estimate.motion_covariance, estimate.pose.copy()
            chart = fpose.copy() if map_now and not self.initialized else None
        else:
            map_now, req = True, None
            delta, cov = frame.get("delta_pose"), frame.get("motion_covariance")
            if delta is not None and i > 0:
                self._odom = self._odom @ np.asarray(delta, dtype=np.float64)
            fpose = self._odom.copy()
            chart = None
        observe = self.cadence.frame(delta, map_now)
        capped = False
        if observe and self.obs_cap > 0 and self.cadence.can_veto():
            if self._cost(True, req is not None) is not None and self._q_free - t > self.obs_cap:
                self.cadence.veto()              # the server could not start it in time: the next frame instead
                observe, capped = False, True
                self.stats["capped"] += 1
        if self.frontend is not None:
            if req is None and self.frontend.align is not None:
                req = self.frontend.aligned_request(map_now and observe)
                self.frontend.request = None
            if req is not None:
                req["anchor"] = bool(map_now and req["m"] is not None)
        send_images = (map_now and (observe or self.upload == "all")) or req is not None
        depth = right = None
        if send_images:
            depth, right = frame.get("depth"), frame.get("rgb_right") if self.send_right else None
            if self.frontend is not None:
                import cv2
                if depth is None:
                    depth = estimate.depth
                if depth is None and self.depth_model is not None and map_now:
                    depth = self.depth_model.predict_metric(rgb, self.K, rgb.shape[:2])
                if depth is not None and depth.shape[:2] != rgb.shape[:2]:
                    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
        msg = {"index": i, "timestamp": t, "delta_pose": delta, "motion_covariance": cov, "map_frame": map_now,
               "initial_chart_pose": chart, "request": req}
        if capped and map_now and send_images:
            msg["no_observation"] = True         # the images serve the odometry's request only
        if send_images:
            msg.update(rgb=rgb, rgb_right=right, depth=depth)
            self.stats["uploads"] += 1
        self.stats["requests"] += int(req is not None)
        self.stats["frames"] += 1
        self._fpose[i] = fpose
        self._sent[i] = t
        self._kind[i] = req is not None
        expected = self._cost(bool(observe and map_now), req is not None) or 0.0
        self._inflight[i] = (t, expected)
        self._q_free = max(self._q_free, t) + expected
        self.link.send(msg, t)
        self._receive(t)                         # a reply that arrived at once (no latency)
        if map_now:
            self.initialized = True
        self.mapped_now = map_now
        self.frontend_pose = fpose
        if estimate is not None:
            estimate.diagnostics["frontend_pose"] = fpose.tolist()
            A0, _ = self._alignment()
            estimate.pose = A0 @ fpose
        self.last_estimate = estimate
        self._prune()
        return estimate

    def _cost(self, observed, request):
        return self._costs.get((bool(observed), bool(request)))

    def _receive(self, t):
        for r in self.link.poll(t):
            self.stats["replies"] += 1
            sent = self._sent.pop(r["index"], None)
            had_req = self._kind.pop(r["index"], "token" in r)
            cost = r.get("link_cost", r.get("server_seconds"))
            flight = self._inflight.pop(r["index"], None)
            if cost is not None:                 # the server time of this kind of message (running mean)
                key = (bool((r.get("work") or {}).get("observed")), bool(had_req))
                old = self._costs.get(key)
                self._costs[key] = float(cost) if old is None else 0.8 * old + 0.2 * float(cost)
                if flight is not None:
                    # the queue model re-anchored on this message: it waited lag - server time - network delay, the
                    # server finished it then, and the messages sent after it follow
                    lag = t - flight[0]
                    self._net = min(self._net, lag - float(cost))
                    q = flight[0] + max(lag - float(cost) - self._net, 0.0) + float(cost)
                    for t_j, c_j in self._inflight.values():
                        q = max(q, t_j) + c_j
                    self._q_free = q
            if r.get("summary") is not None or "token" in r:
                if self.frontend is not None:
                    self.frontend.close(r["token"], r.get("summary"))
                if sent is not None:
                    self.stats["measurement_lag_s"].append(t - sent)
            if r.get("map") is not None:
                self._map = r["map"]
                self.cadence.confident = bool(r["map"].get("confident", False))
                self.stats["map_lag_frames"].append(self.index - r["index"])
                if sent is not None:
                    self.stats["map_lag_s"].append(t - sent)

    def _alignment(self):
        """T_map T_odom^-1 of the last replied mapped frame (hypothesis 0, most likely hypothesis); identity before."""
        m = self._map
        if m is None or m["index"] not in self._fpose:
            return np.eye(4), np.eye(4)
        F_inv = _inverse(self._fpose[m["index"]])
        return m["T0"] @ F_inv, m["Tbest"] @ F_inv

    def belief(self, to_mat):
        """(pose of hypothesis 0, pose of the most likely hypothesis, weights) in the map frame, as 4x4 arrays: the back
        end's belief when it replied for this frame already (a zero-latency link), else that of its last reply carried
        by the odometry since."""
        m = self._map
        if m is None:
            pose = self._fpose[self.index] if self.index in self._fpose else np.eye(4)
            return pose, pose, np.ones(1)
        if m["index"] == self.index and self.mapped_now:
            if m.get("mu0") is not None:
                return to_mat(m["mu0"]), to_mat(m["mubest"]), m["w"]
            return m["B0"], m["Bbest"], m["w"]
        A0, Ab = self._alignment()
        F = self._fpose[self.index]
        return A0 @ F, Ab @ F, m["w"]

    def _prune(self):
        keep = min([self._map["index"] if self._map is not None else 0] + list(self._sent))
        for k in [k for k in self._fpose if k < keep and k != self.index]:
            del self._fpose[k]

    def remote_stats(self) -> dict:
        """Uploads, lags (dataset seconds from sending a frame to the reply), the link's and the server's counts."""
        out = {k: v for k, v in self.stats.items() if not isinstance(v, list)}
        for k in ("map_lag_frames", "measurement_lag_s", "map_lag_s"):
            v = np.asarray(self.stats[k], dtype=np.float64)
            out[k] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
                      "p50": float(np.median(v)) if len(v) else None,
                      "p95": float(np.percentile(v, 95)) if len(v) else None, "max": float(v.max()) if len(v) else None}
        if hasattr(self.link, "summary"):
            out["link"] = self.link.summary()
        if self.server is not None:
            out["server"] = dict(self.server.stats)
        if self.frontend is not None:
            out["frontend"] = {k: v for k, v in self.frontend.stats.items() if isinstance(v, (int, float))}
        return out


class _RemoteKeyframes:
    """The server's keyframes as the runners read them (hypothesis_manager.nodes: temporary, pose_mu[0]), behind a
    real link."""

    def __init__(self, link):
        self.link = link

    @property
    def nodes(self):
        from types import SimpleNamespace
        import pypose as pp
        import torch
        self.link.flush()
        r = self.link.call("keyframes")
        return {int(k): SimpleNamespace(temporary=bool(v["temporary"]),
                                        pose_mu=pp.mat2SE3(torch.as_tensor(np.asarray(v["pose"])[None], dtype=torch.float32)))
                for k, v in r["nodes"].items()}


def remote_session(pipeline, link_factory, upload="predicted", obs_cap=0.0):
    """A local session (cross.pipeline.build_session: a Pipeline with external or VGGT-inertial odometry) split into a
    server (its back end and the odometry's pass service) and an edge behind the link link_factory(server)."""
    from .server import MapServer
    from cross.pipeline import Pipeline
    if not isinstance(pipeline, Pipeline):
        raise ValueError("Remote sessions: the rgbd / stereo modes and the mono mode with the ff back end")
    if pipeline.odometry not in ("external", "vgio"):
        raise ValueError("Remote sessions: external odometry or VGGT-inertial odometry (--odometry vgio); DPVO runs on "
                         "the GPU")
    frontend = pipeline.frontend
    service = None
    if frontend is not None:
        service, frontend.service = frontend.service, None
    server = MapServer(pipeline.mapper, service)
    factory = None
    if pipeline.frontend_factory is not None:
        def factory():
            f = pipeline.frontend_factory()
            server.service, f.service = f.service, None
            return f
    return RemotePipeline(frontend, link_factory(server), pipeline.mode, pipeline.odometry,
                          ObservationCadence(pipeline.mapper.config.pose_est), pipeline.mapping_interval, upload,
                          frontend_factory=factory, server=server,
                          continuous_start_in_map=pipeline.continuous_start_in_map,
                          depth_model=pipeline.depth_model, K=pipeline.K, obs_cap=obs_cap,
                          send_right=pipeline.mapper.config.pose_est.ff.right_image != "left" or pipeline.odometry == "vgio")
