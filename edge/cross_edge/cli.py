"""cross-edge run: one remote CROSS session from the edge.

  cross-edge run --server HOST:50051 --data <stereo folder> --odometry basalt --save-map /maps/office.pkl --out run1
  cross-edge run --server HOST:50051 --data <query folder> --odometry basalt --load-map /maps/office.pkl --out run2

The server is the CROSS repository's scripts/remote/serve.py on the GPU machine; map paths are on the server.  The
sensors are replayed from a prepared stereo folder (cross_edge.sensors.StereoFolder); --realtime feeds the frames at
their timestamps (a live robot), otherwise as fast as the edge and the link go.  --lockstep waits for each frame's
reply before the next (no latency: the session the simulated zero-latency link gives).

Output (--out): poses.txt (per frame: index, timestamp, localized flag, the published map-frame pose of hypothesis 0
as 16 values), odometry.txt (the odometry's camera pose per frame, 16 values; nan before its first estimate),
keyframes.json (the server's keyframes: frame index, pose, temporary), stats.json (uploads, lags, link, odometry)."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import PROTOCOL_VERSION, __version__


def build_parser():
    ap = argparse.ArgumentParser(prog="cross-edge", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one session")
    r.add_argument("--server", required=True, help="HOST:PORT of scripts/remote/serve.py")
    r.add_argument("--data", required=True, help="prepared stereo folder (left/, right/, calib.json, IMU)")
    r.add_argument("--baseline", type=float, default=None, help="SimChange: the rendered baseline to use (right_dirs)")
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--end", type=int, default=None)
    r.add_argument("--odometry", choices=["basalt", "file"], default="basalt")
    r.add_argument("--odom-file", default="odom_vio.txt", help="--odometry file: the folder's pose file")
    r.add_argument("--basalt", default=None, help="basalt_live binary (default $BASALT_LIVE)")
    r.add_argument("--basalt-config", default=None, help="Basalt configuration JSON (default: its euroc_config.json)")
    r.add_argument("--basalt-threads", type=int, default=4)
    r.add_argument("--basalt-wait", type=float, default=5.0, help="seconds to wait for a frame's VIO state")
    r.add_argument("--load-map", default=None, help="map file on the server to relocalize in")
    r.add_argument("--save-map", default=None, help="map file on the server to store at the end")
    r.add_argument("--config-file", nargs="*", default=[], help="configuration files of the server's configs/ folder "
                   "layered on the mode's (e.g. outdoor.yaml)")
    r.add_argument("--set", nargs="*", action="extend", default=[], help="server configuration overrides key=value")
    r.add_argument("--obs-cap", type=float, default=0.1, help="rate cap (s; 0: off), as the CROSS runners' default; "
                   "with --realtime only (it models the server's queue on the frames' timestamps)")
    r.add_argument("--max-backlog", type=float, default=0.3, help="server overload policy (s; 0: off)")
    r.add_argument("--jpeg", type=int, default=90, help="JPEG quality of the uploads (0: lossless PNG)")
    r.add_argument("--realtime", action="store_true", help="feed the frames at their timestamps")
    r.add_argument("--lockstep", action="store_true", help="wait for each frame's reply before the next")
    r.add_argument("--extra-delay", type=float, default=0.0, help="emulated extra round trip (s)")
    r.add_argument("--out", default=None, help="output folder")
    r.add_argument("--quiet", action="store_true")
    return ap


def open_session(args, src):
    """The gRPC link and the edge session of a stereo source with external odometry."""
    from .cadence import ObservationCadence, cadence_config
    from .grpc_client import GrpcLink
    from .session import EdgeSession
    open_msg = {"protocol": PROTOCOL_VERSION, "edge_version": __version__, "mode": "stereo", "odometry": "external",
                "camera": {"K": src.K, "width": src.width, "height": src.height},
                "T_right_in_left": src.T_right_in_left, "config_files": list(args.config_file), "set": list(args.set),
                "max_backlog": args.max_backlog}
    link = GrpcLink(args.server, open_msg, jpeg=args.jpeg, extra_delay=args.extra_delay, realtime=args.realtime)
    opened = link.opened
    if int(opened.get("protocol", -1)) != PROTOCOL_VERSION:
        raise RuntimeError(f"server speaks protocol {opened.get('protocol')}, this edge {PROTOCOL_VERSION}")
    # the rate cap models the server's queue on the frames' timestamps: only meaningful when they arrive in real time
    obs_cap = args.obs_cap if args.realtime else 0.0
    session = EdgeSession(link, ObservationCadence(cadence_config(opened["cadence"])), mode="stereo",
                          odometry="external", obs_cap=obs_cap, send_right=opened.get("right_image") != "left")
    return session, opened


def run(args):
    from .odometry import FileOdometry, MotionTracker
    from .sensors import StereoFolder
    src = StereoFolder(args.data, baseline=args.baseline)
    if args.odometry == "file":
        poses = src.odometry_file(args.odom_file)
        if poses is None:
            raise FileNotFoundError(f"{args.data}/{args.odom_file}")
        odom = FileOdometry(poses)
    else:
        from .basalt import BasaltOdometry
        out_dir = Path(args.out) if args.out else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        odom = BasaltOdometry(src, binary=args.basalt, config=args.basalt_config, threads=args.basalt_threads,
                              wait=args.basalt_wait, log=None if out_dir is None else out_dir / "basalt_live.log")
    session, opened = open_session(args, src)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    log(f"cross-edge {__version__}: session opened on {args.server} ({opened.get('gpu')}), cadence {opened['cadence']}, "
        f"right image {'sent' if session.send_right else 'not sent'}")
    if args.load_map:
        r = session.load_map(args.load_map)
        log(f"map {args.load_map}: {r.get('n_keyframes')} keyframes")
    tracker = MotionTracker(odom)
    rows, odo_rows, t_odom, t_sess = [], [], 0.0, 0.0
    t_start = time.monotonic()
    n = 0
    for frame in src.frames(args.start, args.end):
        if hasattr(session.link, "pace"):
            session.link.pace(frame["timestamp"])    # real time: the frame arrives now, then the VIO runs
        t0 = time.monotonic()
        delta, P = tracker.motion(frame)
        t1 = time.monotonic()
        frame["delta_pose"] = delta
        T = session.process(frame)
        if args.lockstep:
            session.link.flush()
            session._receive(frame["timestamp"])
            T = session.belief()[0]
        t_odom += t1 - t0
        t_sess += time.monotonic() - t1
        rows.append([frame["index"], frame["timestamp"], float(session.localized())] + list(np.asarray(T).reshape(-1)))
        odo_rows.append(list((np.full((4, 4), np.nan) if P is None else P).reshape(-1)))
        n += 1
        if n % 100 == 0:
            log(f"frame {frame['index']}: {n / (time.monotonic() - t_start):.1f} fps, uploads {session.stats['uploads']}, "
                f"replies {session.stats['replies']}")
    session.link.flush()
    session._receive(float("inf"))
    wall = time.monotonic() - t_start
    keyframes = session.link.call("keyframes")
    if args.save_map:
        session.save_map(args.save_map)
    stats = session.remote_stats()
    stats.update(frames=n, wall_s=wall, fps=n / max(wall, 1e-9), odometry_s_per_frame=t_odom / max(n, 1),
                 session_s_per_frame=t_sess / max(n, 1), odometry=getattr(odom, "stats", {}), server=opened)
    closed = session.release()
    odom.close()
    if closed is not None:
        stats["server_stats"] = closed.get("stats")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savetxt(out / "poses.txt", np.asarray(rows), fmt="%.9g",
                   header="index timestamp localized T_map_cam(16, row-major, hypothesis 0)")
        np.savetxt(out / "odometry.txt", np.asarray(odo_rows), fmt="%.9g")
        (out / "keyframes.json").write_text(json.dumps(
            {"frames": keyframes.get("frames"),
             "nodes": {k: {"temporary": v["temporary"], "pose": np.asarray(v["pose"]).tolist()}
                       for k, v in keyframes.get("nodes", {}).items()}}))
        stats.pop("server", None)
        stats["server_open"] = {k: v for k, v in opened.items() if k != "op"}
        (out / "stats.json").write_text(json.dumps(stats, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"done: {n} frames in {wall:.1f} s ({n / max(wall, 1e-9):.1f} fps), odometry {1e3 * t_odom / max(n, 1):.1f} "
        f"ms/frame, uploads {session.stats['uploads']}")
    return stats


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
