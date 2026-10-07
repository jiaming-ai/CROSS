"""The edge half of a remote CROSS session (cross.remote): the odometry, the messages to the server, and the map-frame
pose from the server's late replies."""

import numpy as np

from cross.pipeline import restrict_inputs
from cross_edge.cadence import ObservationCadence
from cross_edge.session import EdgeSession


class RemotePipeline(EdgeSession):
    """A CROSS session whose back end runs behind a link (cross.remote.link): the edge's side, with the interface of
    cross.pipeline.Pipeline (process, belief, save_map, load_map, release).  cross_edge.session.EdgeSession (the edge
    package's session: messages, rate cap, replies, map-frame pose) with the VGGT-inertial frontend and the in-process
    server of a simulated link.

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
        self.frontend, self.server = frontend, server
        super().__init__(link, cadence, mode=mode, odometry=odometry, mapping_interval=mapping_interval, upload=upload,
                         obs_cap=obs_cap, send_right=send_right)
        self.frontend_factory = frontend_factory
        self.continuous_start_in_map = continuous_start_in_map
        self.depth_model, self.K = depth_model, K
        self.last_estimate = None
        self.frontend_pose = None

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
        return super().keyframe_frames()

    @property
    def last_added_kf_id(self):
        return self.server.system.last_added_kf_id if self.server is not None else self.link.last_added_kf_id

    def save_map(self, path):
        if self.server is not None:
            self.link.flush()
            self.server.save_map(str(path))
        else:
            super().save_map(path)

    def load_map(self, path):
        """Start a new session in a stored map (fresh frontend, no alignment)."""
        replaced = False
        if self.frontend is not None and getattr(self.frontend, "index", 0):
            if self.frontend_factory is None:
                raise RuntimeError("Load a map before processing images of a new session")
            self.frontend.shutdown()
            self.frontend = self.frontend_factory()
            replaced = True
        self._reset()
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

    # ------------------------------------------------------------------ the frontend's hooks (cross_edge.session)
    def _prepare(self, frame):
        if self.frontend is not None or self.mode == "mono":
            frame = restrict_inputs(frame, self.mode, self.odometry)
        if self.frontend is not None and frame.get("timestamp") is None:
            frame["timestamp"] = float(self._frames)
        return frame

    def _track(self, frame, i):
        if self.frontend is None:
            return super()._track(frame, i)
        estimate = self.frontend.track(frame)
        valid = bool(estimate.diagnostics.get("valid", True))
        observable = valid or bool(estimate.diagnostics.get("unknown_motion", False))
        index = getattr(self.frontend, "index", 1) - 1
        map_now = observable and (not self.initialized or index % self.mapping_interval == 0)
        req, self.frontend.request = self.frontend.request, None
        delta, cov, fpose = estimate.delta_pose, estimate.motion_covariance, estimate.pose.copy()
        chart = fpose.copy() if map_now and not self.initialized else None
        return delta, cov, fpose, map_now, chart, req, estimate

    def _request(self, req, map_now, observe):
        if self.frontend is not None:
            if req is None and self.frontend.align is not None:
                req = self.frontend.aligned_request(map_now and observe)
                self.frontend.request = None
            if req is not None:
                req["anchor"] = bool(map_now and req["m"] is not None)
        return req

    def _images(self, frame, estimate, map_now):
        depth, right = super()._images(frame, estimate, map_now)
        if self.frontend is not None:
            import cv2
            rgb = frame["rgb"]
            if depth is None:
                depth = estimate.depth
            if depth is None and self.depth_model is not None and map_now:
                depth = self.depth_model.predict_metric(rgb, self.K, rgb.shape[:2])
            if depth is not None and depth.shape[:2] != rgb.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))
        return depth, right

    def _on_reply(self, r, sent):
        if self.frontend is not None:
            self.frontend.close(r["token"], r.get("summary"))

    def _finish(self, estimate, fpose):
        self.frontend_pose = fpose
        if estimate is not None:
            estimate.diagnostics["frontend_pose"] = fpose.tolist()
            A0, _ = self._alignment()
            estimate.pose = A0 @ fpose
        self.last_estimate = estimate
        return estimate

    def remote_stats(self) -> dict:
        """Uploads, lags (dataset seconds from sending a frame to the reply), the link's and the server's counts."""
        out = super().remote_stats()
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
