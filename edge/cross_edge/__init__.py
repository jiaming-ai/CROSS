"""cross-edge: the robot's side of a remote CROSS session, without the GPU code of CROSS.

The robot runs its odometry (live Basalt stereo-inertial VIO, cross_edge.basalt, or a recorded odometry file), copies
the CROSS back end's observation cadence (cross_edge.cadence) to send images only on the frames the server will
observe, and publishes the map-frame pose of every frame from the server's late replies (cross_edge.session).  The
server (scripts/remote/serve.py of the CROSS repository) runs the back end: retrieval, the VGGT-Omega passes, the map.

  cross_edge.codec        wire format (JSON header + raw arrays + JPEG / PNG images; no pickle)
  cross_edge.grpc_client  one bidirectional gRPC stream per session
  cross_edge.cadence      the back end's observation cadence, run on the edge's odometry
  cross_edge.session      EdgeSession: messages, rate cap, replies, map-frame pose
  cross_edge.sensors      StereoFolder: replay of a prepared stereo folder (images, IMU, times)
  cross_edge.odometry     odometry sources: a recorded pose file, live Basalt (cross_edge.basalt)
  cross_edge.cli          cross-edge run ...

Only numpy, OpenCV (headless) and grpcio are needed; nothing here imports torch."""

__version__ = "0.1.0"
PROTOCOL_VERSION = 1          # bumped when a message's meaning changes; the server refuses another version
