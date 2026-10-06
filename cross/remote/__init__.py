"""Remote CROSS sessions: the GPU work on a server, the high-rate odometry on the robot's edge computer.

    edge (cross.remote.edge.RemotePipeline)            server (cross.remote.server.MapServer)
    ------------------------------------------         ----------------------------------------------------------
    IMU propagation, tracked corners, the local         the CROSS back end (cross.core.system.System): retrieval,
    pose graph of the VGGT-inertial odometry            the VGGT-Omega pass, hypotheses, loop closure, the map;
    (cross.mono.vggt_imu_frontend, no service)          the odometry's GPU side (cross.mono.vgio_service): its own
    or the external odometry; the map-frame pose        passes, depth ratios, learned / stereo depth scale
    = map->odom(b) * odom(now)

Every frame the edge sends its motion (and, on the frames the back end will observe or the odometry measures, the
images and the measurement request); every reply carries the map pose of its frame b and the summary of b's
measurement.  Replies come back late: the measurement joins the local pose graph at b's time (the IMU carries the state
to the current frame again), and the map pose is the back end's pose of b carried to now by the odometry since b.

A link connects the two: SimLink (cross.remote.link) simulates the network on the dataset's clock (latency, jitter,
outages, upload bandwidth, server compute time; deterministic), GrpcLink (cross.remote.grpc_link) is a real one.
With a zero-latency link a remote session gives the same results as the local session (cross.pipeline.Pipeline)."""

from .edge import ObservationCadence, RemotePipeline, remote_session
from .link import SimLink
from .server import MapServer

__all__ = ["MapServer", "ObservationCadence", "RemotePipeline", "SimLink", "remote_session"]
