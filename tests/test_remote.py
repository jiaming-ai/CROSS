"""Remote sessions (cross/remote): the VGGT-inertial frontend whose measurements come back frames after their request
(a remote server's latency) tracks a simulated drive about as well as with in-frame measurements, also when a
measurement fails; the simulated link's timing (round trip, compute queue, outages, order); the edge's copy of the back
end's observation cadence."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from cross.dataloader.imu import ImuCalibration       # noqa: E402
from cross.imu.simulate import simulate_imu           # noqa: E402
from test_imu_scale import robot_path                  # noqa: E402

FPS = 10.0


def _sequence(seconds=40.0, seed=0):
    poses = robot_path(seconds, FPS, seed, 0.6)
    t, w, a = simulate_imu(poses, FPS, seed)
    samples = np.concatenate([t[:, None], w, a], 1)
    calib = ImuCalibration()
    frames = []
    for k in range(len(poses)):
        tk = k / FPS
        win = samples[(t > tk - 1 / FPS - 1e-9) & (t <= tk + 1e-9)]
        frames.append({"rgb": np.zeros((48, 64, 3), np.uint8), "timestamp": tk, "imu": win, "imu_t0": tk - 1 / FPS,
                       "imu_t1": tk, "imu_calib": calib})
    return poses, frames


class _FakeService:
    """Summaries of a pass from the true poses: each pass in a gauge of its own (metres per unit s), its links the
    ratios of the gauges, learned depth observing s with noise (cross.mono.vgio_service.VgioPassService.measure)."""

    def __init__(self, poses, seed=3, fail=()):
        self.poses, self.rng, self.gauge, self.fail = poses, np.random.default_rng(seed), {}, set(fail)

    def __call__(self, req):
        b, m, kf = req["index"], req["m"], req["kf"]
        out = {"token": req["token"], "index": b, "source": "own", "finite": b not in self.fail}
        if not out["finite"]:
            return out
        s = float(np.exp(self.rng.normal(0, 0.3)))
        self.gauge[b] = s

        def c2w(f):
            T = self.poses[f].copy()
            T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(self.rng.normal(0, 0.002, 3)).as_matrix()
            T[:3, 3] = T[:3, 3] / s
            return T
        out.update(c2w_curr=c2w(b), c2w_prev=None if m is None else c2w(m), c2w_kf=None if kf is None else c2w(kf),
                   c2w_right=None, da3=(float(np.log(s) + self.rng.normal(0, 0.05)), 0.15), stereo=None,
                   stereo_info=None, corner_z=None, log_median_depth=None if m is not None else float(np.log(s)),
                   link=(s / self.gauge[m], 0.01) if m in self.gauge else (float("nan"), float("inf")),
                   link_kf=(s / self.gauge[kf], 0.01) if kf in self.gauge else None)
        return out


def _run(delay, seconds=40.0, fail=()):
    """The frontend on a simulated drive with every measurement closed `delay` frames after its request (0: in the
    same frame, as a local session).  Returns (position RMSE after a rigid alignment over the valid frames, frontend)."""
    from cross.mono.config import MonoConfig
    from cross.mono.vggt_imu_frontend import VggtImuFrontend
    poses, frames = _sequence(seconds)
    mc = MonoConfig()
    mc.imu.vgio_klt = False                       # no images to track corners in
    fe = VggtImuFrontend(np.array([[300.0, 0, 32], [0, 300.0, 24], [0, 0, 1]]), mc, device="cpu", interval=3,
                         context=2, keyframe_age=2.0, graph=True, local_service=False)
    fe.standalone = False
    service = _FakeService(poses, fail=fail)
    pending, est, gt = [], [], []
    for k, frame in enumerate(frames):
        for item in [p for p in pending if p[0] <= k]:
            pending.remove(item)
            fe.close(item[1], item[2])
        e = fe.track(dict(frame))
        req, fe.request = fe.request, None
        if req is not None:
            pending.append((k + delay, req["token"], service(req)))
            if delay == 0:
                fe.close(req["token"], pending.pop()[2])
        if e.diagnostics["valid"]:
            est.append(e.pose[:3, 3])
            gt.append(poses[k][:3, 3])
    src, dst = np.asarray(est), np.asarray(gt)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - mu_s).T @ (dst - mu_d))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    err = np.linalg.norm((src - mu_s) @ R.T + mu_d - dst, axis=1)
    return float(np.sqrt(np.mean(err ** 2))), fe, len(est)


def test_late_measurements_track_the_drive():
    """Measurements closed 0, 3 (one interval) and 10 frames (1 s) late: the late ones are replayed (the IMU carries
    the state from the measured frame to the current one again) and the trajectory stays within a few centimetres of
    the in-frame one's error over a ~20 m drive; output becomes valid later by about the delay."""
    ate0, fe0, n0 = _run(0)
    assert ate0 < 0.25, ate0
    assert fe0.stats["replays"] == 0
    for delay in (3, 10):
        ate, fe, n = _run(delay)
        assert fe.stats["replays"] > 0 and fe.stats["chain_breaks"] == 0
        assert ate < ate0 + 0.15, (delay, ate, ate0)
        assert n >= n0 - delay - 5


def test_failed_late_measurement_restarts_the_chain():
    """A pass that comes back unusable (non-finite) while later requests were made on it: those are dropped, the next
    request measures from the last measured frame, and tracking continues."""
    ate, fe, _ = _run(5, fail={150})
    assert fe.stats["chain_breaks"] == 1, fe.stats
    assert ate < 0.4, ate


class _Server:
    def __init__(self):
        self.seen, self.stale = [], []

    def handle(self, msg, stale=False):
        self.seen.append(msg["index"])
        self.stale.append(stale)
        return {"index": msg["index"], "work": {"observed": msg.get("observe", False) and not stale, "stale": stale},
                "server_seconds": 0.0}


def test_sim_link_timing():
    """Round trip + modelled compute; a busy server queues; replies keep their order; an outage holds messages and
    replies until the link is back; the server sees every message in order."""
    from cross.remote.link import SimLink
    srv = _Server()
    link = SimLink(srv, rtt=0.2, compute="model", costs={"frame": 0.01, "observe": 0.25}, measure_bytes=False)
    link.send({"index": 0, "timestamp": 0.0}, 0.0)                     # 0.1 up, 0.01 server, 0.1 down
    assert link.poll(0.2) == [] and [r["index"] for r in link.poll(0.21 + 1e-6)] == [0]
    link.send({"index": 1, "timestamp": 0.1, "observe": True}, 0.1)    # at server 0.2, done 0.46, back 0.56
    link.send({"index": 2, "timestamp": 0.2}, 0.2)                     # waits for the server: done 0.47, back 0.57
    assert link.poll(0.55) == []
    assert [r["index"] for r in link.poll(0.57 + 1e-6)] == [1, 2]
    link = SimLink(_Server(), rtt=0.0, compute="zero", outages=[(0.5, 1.0)], measure_bytes=False)
    for k in range(20):
        link.send({"index": k, "timestamp": 0.1 * k}, 0.1 * k)
        got = [r["index"] for r in link.poll(0.1 * k)]
        assert got == ([] if 5 <= k < 15 else ([k] if k < 5 else list(range(5, 16)) if k == 15 else [k])), (k, got)
    assert srv.seen == [0, 1, 2]


def test_observation_cadence():
    """The edge's copy of the back end's cadence: every frame through the warm-up after the first, then after 0.3 m or
    3 mapped frames; a missing odometry reading re-initializes (the image is needed)."""
    from types import SimpleNamespace
    from cross.remote.edge import ObservationCadence
    cfg = SimpleNamespace(obs_min_translation=0.3, obs_min_rotation=0.15, obs_max_interval_steps=3, obs_warmup_steps=10)
    c = ObservationCadence(cfg)
    small, big = np.eye(4), np.eye(4)
    small[2, 3], big[2, 3] = 0.01, 0.2
    assert all(c.frame(None if k == 0 else small, True) for k in range(11))       # initialization + warm-up
    assert [c.frame(small, True) for _ in range(6)] == [False, False, True] * 2    # 3 mapped frames
    assert [c.frame(big, True) for _ in range(4)] == [False, True, False, True]    # 0.3 m
    assert c.frame(small, False) is False and c.frame(None, False) is False       # frames the back end does not map
    assert c.frame(small, True) is True                    # re-initialization after the missing reading
    # the adaptive relaxation (obs_confident_*) while the server's last reply says the back end is confident
    cfg.obs_confident_max_interval_steps, cfg.obs_confident_min_translation, cfg.obs_confident_min_rotation = 10, 0.6, 0.3
    c = ObservationCadence(cfg)
    assert all(c.frame(None if k == 0 else small, True) for k in range(11))
    c.confident = True
    assert [c.frame(small, True) for _ in range(10)] == [False] * 9 + [True]      # 10 mapped frames
    assert [c.frame(big, True) for _ in range(3)] == [False, False, True]         # 0.6 m
    c.confident = False
    assert [c.frame(small, True) for _ in range(3)] == [False, False, True]       # the strict rule again


def test_codec_roundtrip():
    """The wire format: nested dicts / lists, numbers (nan and inf too), None, arrays of any dtype, images (lossless
    PNG; JPEG close), no other types."""
    import pytest
    from cross.remote.codec import decode, encode
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)
    msg = {"index": 3, "t": 0.25, "x": float("nan"), "big": float("inf"), "none": None, "flag": True,
           "T": np.eye(4), "corners": rng.random((5, 2)).astype(np.float32), "link": (1.5, 0.01),
           "nested": {"w": np.arange(3), "s": "ok"}, "rgb": img}
    out = decode(encode(msg))
    assert out["index"] == 3 and out["t"] == 0.25 and np.isnan(out["x"]) and out["big"] == float("inf")
    assert out["none"] is None and out["flag"] is True and out["link"] == [1.5, 0.01] and out["nested"]["s"] == "ok"
    assert np.array_equal(out["T"], np.eye(4)) and out["corners"].dtype == np.float32
    assert np.array_equal(out["nested"]["w"], np.arange(3)) and np.array_equal(out["rgb"], img)
    lossy = decode(encode({"rgb": img}, jpeg=90))["rgb"]
    assert lossy.shape == img.shape and lossy.dtype == np.uint8
    with pytest.raises(TypeError):
        encode({"f": lambda: 0})


def test_grpc_link():
    """A real gRPC stream on localhost: the open message reaches the builder, frames are answered in order with their
    images intact, an extra delay holds the replies, control messages are answered, close releases the server."""
    import socket
    import time
    import pytest
    pytest.importorskip("grpc")
    from cross.remote.grpc_link import GrpcLink, serve

    class Fake:
        stats = {"messages": 0}
        released = False

        def handle(self, msg):
            self.stats["messages"] += 1
            return {"index": msg["index"], "sum": int(msg["rgb"].sum()) if msg.get("rgb") is not None else None,
                    "server_seconds": 0.0}

        def save_map(self, path):
            self.saved = path

        def release(self):
            Fake.released = True
    seen = {}

    def build(msg):
        seen.update(msg)
        return Fake(), None, {"gpu": "none"}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = serve(port, build)
    try:
        link = GrpcLink(f"127.0.0.1:{port}", {"mode": "stereo", "K": np.eye(3)}, extra_delay=0.2)
        assert link.opened["gpu"] == "none" and seen["mode"] == "stereo" and np.array_equal(seen["K"], np.eye(3))
        img = np.full((8, 8, 3), 7, np.uint8)
        t0 = time.monotonic()
        for k in range(5):
            link.send({"index": k, "timestamp": 0.1 * k, "rgb": img if k % 2 == 0 else None}, 0.1 * k)
        got = []
        while len(got) < 5 and time.monotonic() - t0 < 10:
            got += link.poll()
            time.sleep(0.01)
        assert time.monotonic() - t0 >= 0.2                          # the emulated round trip
        assert [r["index"] for r in got] == list(range(5))
        assert [r["sum"] for r in got] == [7 * 192, None, 7 * 192, None, 7 * 192]
        assert link.call("save_map", path="/x/map.pkl")["op"] == "saved"
        assert link.close()["stats"]["messages"] == 5 and Fake.released
        assert link.summary()["latency_s"]["p50"] >= 0.2
    finally:
        srv.stop(0)


def test_overload_policy():
    """A server slower than the frames (0.25 s per observation at 10 Hz): without a budget the queue and the replies'
    lag grow without bound; with a 0.3 s budget the frames that waited longer are stepped without their observation
    and the lag stays bounded."""
    from cross.remote.link import SimLink

    def run(budget):
        srv = _Server()
        link = SimLink(srv, rtt=0.0, compute="model", costs={"frame": 0.01, "observe": 0.25}, measure_bytes=False,
                       max_backlog=budget)
        for k in range(100):
            link.send({"index": k, "timestamp": 0.1 * k, "observe": True}, 0.1 * k)
        lag = [e["arrival"] - e["sent"] for e in link.log]
        return srv, lag
    srv, lag = run(None)
    assert not any(srv.stale) and lag[-1] > 10 * lag[5]
    srv, lag = run(0.3)
    assert 0 < sum(srv.stale) < 100 and max(lag) < 0.3 + 0.26 + 1e-9


def test_rate_cap():
    """External odometry, a server slower than the observations asked for (0.5 s each, one every ~3 frames at 10 Hz):
    without the cap the replies' lag grows for the whole run; with a 0.1 s cap the edge sends an observation only when
    the server could start it within 0.1 s (by its model of the server's queue, learned from the replies), the back
    end observes at the next frame instead, and the lag stays bounded."""
    from types import SimpleNamespace
    from cross.remote.edge import ObservationCadence, RemotePipeline
    from cross.remote.link import SimLink

    class Srv:
        stats = {}

        def handle(self, msg, stale=False):
            obs = msg.get("rgb") is not None and not msg.get("no_observation") and not stale
            return {"index": msg["index"], "work": {"observed": obs}, "server_seconds": 0.0}

    def run(cap):
        cfg = SimpleNamespace(obs_min_translation=0.3, obs_min_rotation=0.15, obs_max_interval_steps=3, obs_warmup_steps=10)
        link = SimLink(Srv(), rtt=0.05, compute="model", costs={"frame": 0.001, "observe": 0.5}, measure_bytes=False)
        p = RemotePipeline(None, link, "stereo", "external", ObservationCadence(cfg), obs_cap=cap)
        step = np.eye(4)
        step[2, 3] = 0.12
        for k in range(300):
            p.process({"rgb": np.zeros((4, 4, 3), np.uint8), "rgb_right": None, "depth": None,
                       "delta_pose": None if k == 0 else step, "timestamp": 0.1 * k})
        lag = [e["arrival"] - e["sent"] for e in link.log]
        observed = sum(e["work"]["observed"] for e in link.log)
        return lag, observed, p.stats["capped"]
    lag, obs, capped = run(0.0)
    assert capped == 0 and lag[-1] > 20.0                  # the queue grows for the whole run
    lag, obs_c, capped = run(0.1)
    assert capped > 0 and max(lag[100:]) < 0.1 + 0.5 + 0.06 + 1e-6, max(lag[100:])
    assert 0.8 * 300 * 0.1 / 0.5 < obs_c < obs                # about as many observations as the server can do
