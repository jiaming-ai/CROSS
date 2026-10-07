# cross-edge

The robot's side of a remote [CROSS](../README.md) session.

- **Robot (edge computer, CPU only):** Basalt stereo-inertial VIO runs live. A copy of the CROSS back end's
  observation cadence decides which frames' images the server needs. The edge sends every frame's motion (~300 B)
  plus images on those frames, and publishes the map-frame pose of every frame from the server's replies.
- **Server (GPU machine):** the CROSS back end (retrieval, VGGT-Omega passes, hypotheses, loop closure, the map),
  run by `scripts/remote/serve.py` of the CROSS repository.

The map-frame pose of frame t is the back end's pose of the last replied frame b, carried to t by the odometry:
`T_map(b) T_odom(b)^-1 T_odom(t)`. Latency and outages delay only the map corrections; the odometry never waits
for the network.

This package needs only numpy, OpenCV (headless) and grpcio. It imports neither torch nor CROSS.

## Install (edge computer)

```bash
pip install ./edge                                  # from a clone of the CROSS repository
edge/native/install_basalt_live.sh ~/basalt_build   # Basalt + basalt_live (cmake >= 3.24, < 4; ninja; C++17)
export BASALT_LIVE=~/basalt_build/basalt/build/release/basalt_live
```

`install_basalt_live.sh` builds the Basalt commit that produced the CROSS benchmark's Basalt odometry
(`scripts/vio/install_basalt.sh`), plus `basalt_live` (`edge/native/basalt_live.cpp`). That program is Basalt's
estimator fed from a stream instead of a dataset folder. It has been tested on x86-64 only; aarch64 is untested.

## Run

On the GPU machine:

```bash
python scripts/remote/serve.py --port 50051
```

On the robot, with sensors replayed from a prepared stereo folder: rectified `left/`, `right/`, `calib.json`, and
the IMU stream `imu_vio.txt` / `imu.txt` with its `.json`.

```bash
# map a place (the map is stored on the server)
cross-edge run --server gpu-host:50051 --data /data/office1-1/stereo --save-map /maps/office.pkl --realtime --out run_map
# relocalize in it later
cross-edge run --server gpu-host:50051 --data /data/office1-2/stereo --load-map /maps/office.pkl --realtime --out run_q
```

Options:

| option | effect |
|---|---|
| `--odometry basalt` (default) | live VIO |
| `--odometry file --odom-file odom_vio.txt` | replay a recorded odometry |
| `--jpeg 90` (default) | lossy uploads; `0` for lossless PNG |
| `--obs-cap 0.1` | rate cap: no observation the server could not start within 0.1 s |
| `--max-backlog 0.3` | the server's overload policy |
| `--extra-delay` | emulates a longer round trip |
| `--lockstep` | waits for each reply: the zero-latency session, for verification |

Outputs (`--out`):

- `poses.txt`: per frame, the published map-frame pose and whether the session is localized in the map.
- `odometry.txt`: the VIO's camera poses.
- `keyframes.json`: the server's keyframes.
- `stats.json`: uploads, reply lags, link and odometry timing.

## Layout

| module | role |
|---|---|
| `cross_edge.codec` | wire format: JSON header + raw arrays + JPEG/PNG images; no pickle |
| `cross_edge.grpc_client` | one bidirectional gRPC stream per session |
| `cross_edge.cadence` | the back end's observation cadence, on the edge's odometry (parameters sent by the server) |
| `cross_edge.session` | `EdgeSession`: messages, rate cap, replies, map-frame pose (the CROSS repository's `cross.remote.edge.RemotePipeline` extends it) |
| `cross_edge.sensors` | `StereoFolder`: replay of a prepared stereo folder |
| `cross_edge.odometry`, `cross_edge.basalt` | recorded pose file, live Basalt |
| `cross_edge.cli` | `cross-edge run` |

Tests: `pytest edge/tests` (no GPU, no CROSS).
