"""cross-edge without CROSS: the wire format, the cadence copy, the session's messages and map-frame pose, the
odometry conventions, the sensor replay, the Basalt calibration, a whole run against a fake gRPC server, and that no
module imports torch."""

import json
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

EDGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDGE))


def translation(x, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return T


CADENCE = {"obs_min_translation": 0.3, "obs_min_rotation": 0.15, "obs_max_interval_steps": 3, "obs_warmup_steps": 10,
           "obs_confident_max_interval_steps": 0, "obs_confident_min_translation": 0.6, "obs_confident_min_rotation": 0.3}


def test_codec_roundtrip():
    """Nested dicts / lists, numbers (nan and inf too), None, arrays of any dtype, images (lossless PNG; JPEG close),
    no other types."""
    from cross_edge.codec import decode, encode
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


def test_observation_cadence():
    """Every frame through the warm-up after the first, then after 0.3 m or 3 mapped frames; a missing odometry reading
    re-initializes; the adaptive relaxation while the server's last reply says the back end is confident."""
    from cross_edge.cadence import ObservationCadence, cadence_config
    cfg = SimpleNamespace(obs_min_translation=0.3, obs_min_rotation=0.15, obs_max_interval_steps=3, obs_warmup_steps=10)
    c = ObservationCadence(cfg)
    small, big = translation(0, 0, 0.01), translation(0, 0, 0.2)
    assert all(c.frame(None if k == 0 else small, True) for k in range(11))       # initialization + warm-up
    assert [c.frame(small, True) for _ in range(6)] == [False, False, True] * 2    # 3 mapped frames
    assert [c.frame(big, True) for _ in range(4)] == [False, True, False, True]    # 0.3 m
    assert c.frame(small, False) is False and c.frame(None, False) is False       # frames the back end does not map
    assert c.frame(small, True) is True                    # re-initialization after the missing reading
    c = ObservationCadence(cadence_config(dict(CADENCE, obs_confident_max_interval_steps=10)))
    assert all(c.frame(None if k == 0 else small, True) for k in range(11))
    c.confident = True
    assert [c.frame(small, True) for _ in range(10)] == [False] * 9 + [True]      # 10 mapped frames
    assert [c.frame(big, True) for _ in range(3)] == [False, False, True]         # 0.6 m
    c.confident = False
    assert [c.frame(small, True) for _ in range(3)] == [False, False, True]       # the strict rule again


class LagLink:
    """A link whose server replies `lag` frames later with its belief: the odometry pose shifted by `offset` (a fixed
    map alignment); it observes the frames that carry an image."""

    def __init__(self, lag=0, offset=translation(10.0)):
        self.lag, self.offset = lag, offset
        self.pose, self.queue, self.sent, self.calls = np.eye(4), [], [], []
        self.last_added_kf_id = None

    def send(self, msg, t):
        if msg["delta_pose"] is not None:
            self.pose = self.pose @ msg["delta_pose"]
        self.sent.append(msg)
        B = self.offset @ self.pose
        self.queue.append({"index": msg["index"], "server_seconds": 0.0, "work": {"observed": msg.get("rgb") is not None},
                           "map": {"index": msg["index"], "T0": B, "Tbest": B, "B0": B, "Bbest": B, "w": np.ones(1),
                                   "localized": msg["index"] >= 5}})

    def poll(self, t):
        n = len(self.sent)
        out = [r for r in self.queue if r["index"] <= n - 1 - self.lag]
        self.queue = [r for r in self.queue if r["index"] > n - 1 - self.lag]
        return out

    def flush(self):
        pass

    def call(self, op, **kw):
        self.calls.append((op, kw))
        return {"op": op, "cadence": None}


@pytest.mark.parametrize("lag", [0, 3])
def test_session_map_pose_and_uploads(lag):
    """The published map pose is the server's belief carried by the odometry, whatever the reply lag; images only on
    the frames the cadence copy predicts; the localized flag of a map session follows the replies."""
    from cross_edge.cadence import ObservationCadence, cadence_config
    from cross_edge.session import EdgeSession
    link = LagLink(lag)
    s = EdgeSession(link, ObservationCadence(cadence_config(CADENCE)))
    s.load_map("/maps/x.pkl")
    assert link.calls[0] == ("load_map", {"path": "/maps/x.pkl", "new_service": False}) and not s.localized()
    odom = np.eye(4)
    for k in range(40):
        step = translation(0.05, 0.0, 0.01 * (k % 3))
        if k:
            odom = odom @ step
        T = s.process({"timestamp": 0.1 * k, "rgb": np.zeros((2, 2, 3), np.uint8), "rgb_right": None,
                       "delta_pose": None if k == 0 else step})
        if k >= lag:
            assert np.allclose(T, translation(10.0) @ odom), k
    images = [m["index"] for m in link.sent if m.get("rgb") is not None]
    assert images[:11] == list(range(11)) and 11 < len(images) < 30          # warm-up, then the cadence
    assert s.localized() and s.stats["uploads"] == len(images)


def test_motion_conventions():
    """Unknown motion at a session's first frame, zero motion while the odometry has no estimate, steps after."""
    from cross_edge.odometry import MotionTracker
    poses = {0: None, 1: None, 2: translation(1.0), 3: translation(1.5), 4: translation(2.5)}
    tr = MotionTracker(SimpleNamespace(pose=lambda f: poses[f["index"]]))
    deltas = [tr.motion({"index": i})[0] for i in range(5)]
    assert deltas[0] is None and np.allclose(deltas[1], np.eye(4)) and np.allclose(deltas[2], np.eye(4))
    assert np.allclose(deltas[3], translation(0.5)) and np.allclose(deltas[4], translation(1.0))
    tr.new_session()
    assert tr.motion({"index": 4})[0] is None


def test_basalt_calibration():
    """Basalt's calibration of a rectified pair: the IMU-camera transforms of both cameras (right at +baseline), the
    pinhole intrinsics, the noise densities."""
    from cross_edge.basalt import basalt_calibration
    from cross_edge.geometry import matrix_from_quat, quat_from_matrix
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.normal(size=4)
        R = matrix_from_quat(q)
        assert np.allclose(matrix_from_quat(quat_from_matrix(R)), R)
    T_cam_imu = np.eye(4)
    T_cam_imu[:3, :3] = matrix_from_quat([0.1, -0.2, 0.3, 0.9])
    T_cam_imu[:3, 3] = (0.1, -0.2, 0.05)
    K = np.array([[500.0, 0, 320], [0, 501.0, 240], [0, 0, 1]])
    noise = {"gyro_noise_density": 1e-3, "accel_noise_density": 1e-2, "gyro_random_walk": 1e-5, "accel_random_walk": 1e-3}
    c = basalt_calibration(K, (640, 480), 0.12, T_cam_imu, noise, 200.0)["value0"]

    def T(p):
        out = np.eye(4)
        out[:3, :3] = matrix_from_quat([p["qx"], p["qy"], p["qz"], p["qw"]])
        out[:3, 3] = (p["px"], p["py"], p["pz"])
        return out
    T0, T1 = T(c["T_imu_cam"][0]), T(c["T_imu_cam"][1])
    assert np.allclose(T0, np.linalg.inv(T_cam_imu)) and np.allclose(np.linalg.inv(T0) @ T1, translation(0.12))
    assert c["intrinsics"][0]["intrinsics"] == {"fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0}
    assert c["gyro_noise_std"] == [1e-3] * 3 and c["accel_bias_std"] == [1e-3] * 3 and c["resolution"][1] == [640, 480]


def _stereo_folder(root: Path, n=12, size=(32, 24)):
    """A prepared stereo folder: images, calibration, ground truth, an odometry file, an IMU stream."""
    rng = np.random.default_rng(0)
    for d in ("left", "right"):
        (root / d).mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(root / d / f"{i:06d}.png"), rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
    K = [[30.0, 0, 16], [0, 30.0, 12], [0, 0, 1]]
    T_rl = np.eye(4)
    T_rl[0, 3] = 0.1
    (root / "calib.json").write_text(json.dumps({"K": K, "width": size[0], "height": size[1], "fps": 10.0,
                                                 "baseline": 0.1, "T_right_in_left": T_rl.tolist()}))
    poses = np.stack([translation(0.2 * i) for i in range(n)])
    np.savetxt(root / "poses_left.txt", poses.reshape(n, 16))
    np.savetxt(root / "odom_vio.txt", poses.reshape(n, 16))
    t = np.arange(-1.0, n / 10.0 + 1.0, 0.01)
    np.savetxt(root / "imu.txt", np.column_stack([t, np.zeros((len(t), 3)), np.tile([0, 0, 9.81], (len(t), 1))]))
    (root / "imu.json").write_text(json.dumps({"T_cam_imu": np.eye(4).tolist(), "gyro_noise_density": 1e-3,
                                               "accel_noise_density": 1e-2, "gyro_random_walk": 1e-5,
                                               "accel_random_walk": 1e-3, "frame_times": "times.txt"}))
    np.savetxt(root / "times.txt", np.arange(n) / 10.0)
    return poses


def test_stereo_folder(tmp_path):
    from cross_edge.sensors import StereoFolder
    _stereo_folder(tmp_path)
    src = StereoFolder(tmp_path)
    assert len(src) == 12 and src.imu.shape[1] == 7 and np.allclose(src.T_right_in_left[0, 3], 0.1)
    f = src.frame(3)
    assert f["timestamp"] == pytest.approx(0.3) and f["t_imu"] == pytest.approx(0.3)
    assert not dict.__contains__(f, "rgb")                  # not read until used
    assert f["rgb"].shape == (24, 32, 3) and f.get("gray").shape == (24, 32) and dict.__contains__(f, "rgb")
    assert np.allclose(src.odometry_file("odom_vio.txt")[5], translation(1.0))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_against_a_fake_server(tmp_path):
    """cross-edge run with a recorded odometry over a real gRPC stream: the open message carries the camera and the
    protocol version, the server's cadence is used, the published poses are the server's belief carried by the
    odometry, the outputs are written."""
    grpc = pytest.importorskip("grpc")
    from concurrent import futures
    from cross_edge import PROTOCOL_VERSION
    from cross_edge.cli import main
    from cross_edge.codec import decode, encode
    data = tmp_path / "data"
    poses = _stereo_folder(data)
    seen = {}

    def session(request_iterator, context):
        pose = np.eye(4)
        for raw in request_iterator:
            msg = decode(raw)
            op = msg.pop("op")
            if op == "open":
                seen["open"] = msg
                yield encode({"op": "opened", "protocol": PROTOCOL_VERSION, "cadence": CADENCE, "right_image": "pair",
                              "gpu": "fake"})
            elif op == "frame":
                if msg["delta_pose"] is not None:
                    pose = pose @ msg["delta_pose"]
                seen.setdefault("images", []).append(msg.get("rgb") is not None)
                B = translation(0, 5.0) @ pose
                yield encode({"op": "reply", "index": msg["index"], "server_seconds": 0.001,
                              "work": {"observed": msg.get("rgb") is not None},
                              "map": {"index": msg["index"], "T0": B, "Tbest": B, "B0": B, "Bbest": B, "w": np.ones(1)}})
            elif op == "keyframes":
                yield encode({"op": "keyframes", "frames": {"0": 0}, "nodes": {"0": {"temporary": False, "pose": np.eye(4)}}})
            elif op == "save_map":
                seen["saved"] = msg["path"]
                yield encode({"op": "saved"})
            elif op == "close":
                yield encode({"op": "closed", "stats": {"messages": len(seen["images"])}})
                return
    handler = grpc.method_handlers_generic_handler("cross.remote.Remote", {
        "Session": grpc.stream_stream_rpc_method_handler(session, request_deserializer=None, response_serializer=None)})
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    srv.add_generic_rpc_handlers((handler,))
    port = _free_port()
    srv.add_insecure_port(f"127.0.0.1:{port}")
    srv.start()
    try:
        out = tmp_path / "run"
        main(["run", "--server", f"127.0.0.1:{port}", "--data", str(data), "--odometry", "file", "--lockstep",
              "--jpeg", "0", "--save-map", "/maps/m.pkl", "--out", str(out), "--quiet"])
    finally:
        srv.stop(0)
    o = seen["open"]
    assert o["protocol"] == PROTOCOL_VERSION and o["mode"] == "stereo" and o["odometry"] == "external"
    assert o["camera"]["width"] == 32 and np.allclose(o["T_right_in_left"][0, 3], 0.1) and seen["saved"] == "/maps/m.pkl"
    assert all(seen["images"][:11]) and len(seen["images"]) == 12        # warm-up: every frame
    P = np.loadtxt(out / "poses.txt")
    assert P.shape == (12, 19)
    for row, gt in zip(P, poses):
        assert np.allclose(row[3:].reshape(4, 4), translation(0, 5.0) @ np.linalg.inv(poses[0]) @ gt)
    stats = json.loads((out / "stats.json").read_text())
    assert stats["frames"] == 12 and stats["server_stats"] == {"messages": 12}


def test_no_torch():
    """The edge package imports no torch (nor the CROSS package): a run's modules load with torch blocked."""
    code = ("import sys; sys.modules['torch'] = None; sys.modules['cross'] = None\n"
            "import cross_edge, cross_edge.cli, cross_edge.session, cross_edge.basalt, cross_edge.odometry, "
            "cross_edge.sensors, cross_edge.codec, cross_edge.cadence, cross_edge.geometry\n"
            "import importlib.util; assert importlib.util.find_spec('cross_edge.grpc_client')\n"
            "print('ok')")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(EDGE), capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr
