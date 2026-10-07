"""The edge's side of a remote CROSS session: the messages to the server, the rate cap, the replies and the map-frame
pose (numpy only).

Every frame the edge sends its odometry motion (~300 B); on the frames the back end will observe (its cadence, copied
by cross_edge.cadence.ObservationCadence on the edge's own odometry) also the image(s).  Each reply carries the back
end's belief at its frame b; the pose of the current frame t in the map frame is that belief carried to t by the
odometry: T_map(b) T_odom(b)^-1 T_odom(t).

EdgeSession runs external odometry (a VIO, wheel odometry: the motion of each frame given with it).  The CROSS
repository's cross.remote.edge.RemotePipeline extends it with the VGGT-inertial frontend (hooks _track, _request,
_images, _on_reply, _finish) and with an in-process simulated server."""

import numpy as np

from .cadence import ObservationCadence
from .geometry import inverse


class EdgeSession:
    """link       send(message, t), poll(t) -> replies arrived by time t; call(op, ...) for control messages
    cadence    ObservationCadence of the back end's settings
    upload     "predicted": images only on the frames the back end will observe; "all": on every mapped frame
    obs_cap    rate cap (s, 0: off): an observation is sent only if the server can start it within obs_cap s, by a
               model of its queue from the server time of each kind of message, learned from the replies
    send_right False when the back end observes on the left image only (pose_est.ff.right_image: left)"""

    def __init__(self, link, cadence, mode="stereo", odometry="external", mapping_interval=1, upload="predicted",
                 obs_cap=0.0, send_right=True):
        if upload not in ("predicted", "all"):
            raise ValueError(f"Unknown upload policy {upload}")
        self.link = link
        self.mode, self.odometry = mode, odometry
        self.cadence, self.upload = cadence, upload
        self._pose_est_cfg = cadence.cfg
        self.mapping_interval = int(mapping_interval)
        self.send_right = bool(send_right)
        self.initialized = self.mapped_now = False
        self.index = -1                          # frames processed in this session
        self._frames = 0
        self._fpose = {}                         # frame index -> odometry pose reported at that frame
        self._odom = np.eye(4)
        self._map = None                         # the last map reply
        self._map_loaded = False                 # a stored map was loaded: the belief is a map pose once it joined
        self._sent = {}                          # frame index -> time it was sent
        self.stats = dict(frames=0, uploads=0, requests=0, replies=0, capped=0, map_lag_frames=[], measurement_lag_s=[],
                          map_lag_s=[])
        self.obs_cap = float(obs_cap or 0.0)
        self._costs = {}                         # (observed, request) -> server seconds (running mean)
        self._q_free = -np.inf                   # when the server will have finished what was sent (send-time clock)
        self._kind = {}                          # frame index -> request sent with it
        self._inflight = {}                      # frame index -> (time sent, its expected server seconds), in order
        self._net = np.inf                       # the network's delay: the smallest (reply lag - server seconds)

    # ------------------------------------------------------------------ hooks (odometry with its own requests)
    def _prepare(self, frame: dict) -> dict:
        """The frame as this session's odometry sees it."""
        return frame

    def _track(self, frame: dict, i: int):
        """This frame's odometry: (motion since the last frame or None, its covariance, odometry pose, whether the back
        end maps this frame, initial chart pose, measurement request, estimate).  External odometry: the frame's
        delta_pose / motion_covariance."""
        delta, cov = frame.get("delta_pose"), frame.get("motion_covariance")
        if delta is not None and i > 0:
            self._odom = self._odom @ np.asarray(delta, dtype=np.float64)
        return delta, cov, self._odom.copy(), True, None, None, None

    def _request(self, req, map_now: bool, observe: bool):
        """The measurement request sent with this frame (an odometry that measures on the server), or None."""
        return req

    def _images(self, frame: dict, estimate, map_now: bool):
        """(depth, right image) sent with the left image."""
        return frame.get("depth"), frame.get("rgb_right") if self.send_right else None

    def _on_reply(self, r: dict, sent):
        """A reply carrying a measurement summary (an odometry that measures on the server)."""

    def _finish(self, estimate, fpose):
        """What process returns: the pose of this frame in the map frame (hypothesis 0)."""
        A0, _ = self._alignment()
        return A0 @ fpose

    # ------------------------------------------------------------------ one frame
    def process(self, frame: dict):
        """Send one frame (dict: timestamp, rgb, rgb_right, depth, delta_pose, motion_covariance); returns _finish."""
        frame = self._prepare(frame)
        self._frames += 1
        t = float(frame["timestamp"])
        if hasattr(self.link, "pace"):
            self.link.pace(t)                    # a real link in real time: the frame is due at its timestamp
        self._receive(t)                         # the replies that arrived since the last frame
        self.index += 1
        i = self.index
        delta, cov, fpose, map_now, chart, req, estimate = self._track(frame, i)
        observe = self.cadence.frame(delta, map_now)
        capped = False
        if observe and self.obs_cap > 0 and self.cadence.can_veto():
            if self._cost(True, req is not None) is not None and self._q_free - t > self.obs_cap:
                self.cadence.veto()              # the server could not start it in time: the next frame instead
                observe, capped = False, True
                self.stats["capped"] += 1
        req = self._request(req, map_now, observe)
        send_images = (map_now and (observe or self.upload == "all")) or req is not None
        depth = right = None
        if send_images:
            depth, right = self._images(frame, estimate, map_now)
        msg = {"index": i, "timestamp": t, "delta_pose": delta, "motion_covariance": cov, "map_frame": map_now,
               "initial_chart_pose": chart, "request": req}
        if capped and map_now and send_images:
            msg["no_observation"] = True         # the images serve the odometry's request only
        if send_images:
            msg.update(rgb=frame["rgb"], rgb_right=right, depth=depth)   # (read only now: a lazy frame loads it here)
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
        out = self._finish(estimate, fpose)
        self._prune()
        return out

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
                self._on_reply(r, sent)
                if sent is not None:
                    self.stats["measurement_lag_s"].append(t - sent)
            if r.get("map") is not None:
                self._map = r["map"]
                self.cadence.confident = bool(r["map"].get("confident", False))
                self.stats["map_lag_frames"].append(self.index - r["index"])
                if sent is not None:
                    self.stats["map_lag_s"].append(t - sent)

    # ------------------------------------------------------------------ map-frame pose
    def _alignment(self):
        """T_map T_odom^-1 of the last replied mapped frame (hypothesis 0, most likely hypothesis); identity before."""
        m = self._map
        if m is None or m["index"] not in self._fpose:
            return np.eye(4), np.eye(4)
        F_inv = inverse(self._fpose[m["index"]])
        return m["T0"] @ F_inv, m["Tbest"] @ F_inv

    def belief(self, to_mat=None):
        """(pose of hypothesis 0, pose of the most likely hypothesis, weights) in the map frame, as 4x4 arrays: the back
        end's belief when it replied for this frame already (a zero-latency link), else that of its last reply carried
        by the odometry since.  to_mat converts the in-process server's Lie poses (cross.remote) when present."""
        m = self._map
        if m is None:
            pose = self._fpose[self.index] if self.index in self._fpose else np.eye(4)
            return pose, pose, np.ones(1)
        if m["index"] == self.index and self.mapped_now:
            if m.get("mu0") is not None and to_mat is not None:
                return to_mat(m["mu0"]), to_mat(m["mubest"]), m["w"]
            return m["B0"], m["Bbest"], m["w"]
        A0, Ab = self._alignment()
        F = self._fpose[self.index]
        return A0 @ F, Ab @ F, m["w"]

    def localized(self) -> bool:
        """Whether the belief is a pose in the stored map: the server's last reply; before any reply of a session that
        loaded a map, not.  A mapping session is its own map."""
        if not self._map_loaded:
            return True
        return bool(self._map is not None and self._map.get("localized", True))

    def _prune(self):
        keep = min([self._map["index"] if self._map is not None else 0] + list(self._sent))
        for k in [k for k in self._fpose if k < keep and k != self.index]:
            del self._fpose[k]

    # ------------------------------------------------------------------ session (over the link)
    @property
    def last_added_kf_id(self):
        return self.link.last_added_kf_id

    def keyframe_frames(self):
        """The frame index each keyframe was made at (the replies naming it come late)."""
        self.link.flush()
        return {int(k): int(v) for k, v in self.link.call("keyframes")["frames"].items()}

    def save_map(self, path):
        """Store the server's map at path (on the server)."""
        self.link.flush()
        self.link.call("save_map", path=str(path))

    def _reset(self):
        self.initialized, self._map, self._fpose, self._odom = False, None, {}, np.eye(4)
        self._map_loaded = True
        self._sent, self.index = {}, -1
        self.cadence = ObservationCadence(self._pose_est_cfg)
        if hasattr(self.link, "flush") and getattr(self, "server", None) is None:
            self.link.flush()
        if hasattr(self.link, "reset"):
            self.link.reset()

    def load_map(self, path, new_service=False):
        """Start a new session in a map stored on the server (no alignment until the back end relocalizes)."""
        self._reset()
        r = self.link.call("load_map", path=str(path), new_service=new_service)
        if r.get("cadence") is not None:
            self.cadence.sync(r["cadence"])
        return r

    def release(self):
        if hasattr(self.link, "close"):
            return self.link.close()

    def remote_stats(self) -> dict:
        """Uploads, lags (seconds from sending a frame to the reply) and the link's counts."""
        out = {k: v for k, v in self.stats.items() if not isinstance(v, list)}
        for k in ("map_lag_frames", "measurement_lag_s", "map_lag_s"):
            v = np.asarray(self.stats[k], dtype=np.float64)
            out[k] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
                      "p50": float(np.median(v)) if len(v) else None,
                      "p95": float(np.percentile(v, 95)) if len(v) else None, "max": float(v.max()) if len(v) else None}
        if hasattr(self.link, "summary"):
            out["link"] = self.link.summary()
        return out
