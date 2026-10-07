"""Odometry sources of the edge: the camera pose of each frame in the odometry's own frame (c2w, 4x4), or None while
the odometry has no estimate yet.

  FileOdometry    a recorded pose file of the folder (odom_vio.txt of benchmark/datasets/prepare_vio.py: Basalt's
                  causal output; odom_left.txt: wheel / INS odometry)
  BasaltOdometry  Basalt's stereo-inertial VIO running live (cross_edge.basalt)

MotionTracker turns the poses into the motion the server expects (T_prev_curr of the left camera), as the CROSS
loaders do: unknown (None) at a session's first frame; zero while the odometry has no estimate (the recorded files
hold the first estimate before it: zero motion too); the step between two estimates otherwise."""

import numpy as np

from .geometry import inverse


class FileOdometry:
    def __init__(self, poses):
        self.poses = np.asarray(poses, dtype=np.float64)

    def pose(self, frame):
        return self.poses[frame["index"]]

    def close(self):
        pass


class MotionTracker:
    def __init__(self, odometry):
        self.odometry = odometry
        self.prev = None
        self.first = True

    def new_session(self):
        """The next frame starts a session (its motion is unknown: the back end initializes or relocalizes)."""
        self.first = True

    def motion(self, frame):
        """(motion of this frame or None, its odometry pose or None)."""
        P = self.odometry.pose(frame)
        if self.first:
            self.first = False
            delta = None
        elif P is None or self.prev is None:
            delta = np.eye(4)
        else:
            delta = inverse(self.prev) @ P
        if P is not None:
            self.prev = P
        return delta, P
