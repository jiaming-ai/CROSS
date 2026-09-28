import os
import numpy as np
import pandas as pd
import pickle
import json
import time
import cv2

def quaternion_to_rotation_matrix_numpy(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2,   2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x**2 - 2*y**2]
    ])
    return R

class RawSeedLoader:
    def __init__(self, scene_path, use_depth_pred=False):
        self.scene_path = scene_path
        self._rgb_folder = os.path.join(scene_path, "rgb")
        self._depth_folder = os.path.join(scene_path, "depth")
        self._gt_file = os.path.join(scene_path, "groundtruth_aligned.txt")
        self._rgb_files = sorted(os.listdir(self._rgb_folder))
        self._aligned_poses = []
        self._timestamps = []
        self.use_depth_pred = use_depth_pred

        # RAWSEED intrinsics
        self.rgb_K = np.array([
            [194.88, 0.0,   172.16],
            [0.0,   194.88, 125.09],
            [0.0,   0.0,     1.0  ]
        ])
        self.rgb_width = 320
        self.rgb_height = 240

        self.R_robot_to_cv = np.array([
            [0, -1,  0],
            [0,  0, -1],
            [1,  0,  0]
        ])
        self.T_robot_to_cv = np.eye(4)
        self.T_robot_to_cv[:3, :3] = self.R_robot_to_cv

        self._align_ground_truth()

        # Map rgb filenames (without extension) to corresponding depth npy files
        self._depth_files = []
        if not self.use_depth_pred and os.path.exists(self._depth_folder):
            available_depth = {os.path.splitext(f)[0]: f for f in os.listdir(self._depth_folder) if f.endswith('.npy')}
            for rgb_fname in self._rgb_files:
                name_no_ext = os.path.splitext(rgb_fname)[0]
                # expect depth to be named like rgbfile_depth.npy
                depth_fname = f"{name_no_ext}_depth.npy"
                if depth_fname in available_depth.values():
                    self._depth_files.append(depth_fname)
                else:
                    self._depth_files.append(None)
        else:
            self._depth_files = [None] * len(self._rgb_files)

    def _align_ground_truth(self):
        df = pd.read_csv(self._gt_file, sep=r"\s+", comment="#", header=None)
        for _, row in df.iterrows():
            ts, px, py, pz, qx, qy, qz, qw = row
            T = np.eye(4)
            T[:3, 3] = [px, py, pz]
            T[:3, :3] = quaternion_to_rotation_matrix_numpy([qx, qy, qz, qw])
            T_cam = self.T_robot_to_cv @ T @ np.linalg.inv(self.T_robot_to_cv) 
            ## No need to invert for mixed and outdoor scenes
            self._aligned_poses.append(T_cam)
            self._timestamps.append(float(ts))

    def load_sequence(self, seq_idx=None):
        return

    def get_sequence_frequency(self):
        timestamps = np.array(self._timestamps)
        diffs = np.diff(timestamps)
        return 1.0 / np.mean(diffs)

    def __len__(self):
        return len(self._rgb_files)

    def __getitem__(self, idx):
        rgb_file = os.path.join(self._rgb_folder, self._rgb_files[idx])
        timestamp = os.path.splitext(self._rgb_files[idx])[0]

        rgb = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
        depth = None
        if not self.use_depth_pred and self._depth_files[idx] is not None:
            depth_file = os.path.join(self._depth_folder, self._depth_files[idx])
            if depth_file.endswith('.npy'):
                depth = np.load(depth_file)  # Already in meters!
            else:
                # fallback for legacy support, if needed
                depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)/10.0

        if idx == 0:
            delta_T = np.eye(4)
        else:
            delta_T = np.linalg.inv(self._aligned_poses[idx - 1]) @ self._aligned_poses[idx]

        return {
            "rgb": rgb,
            "depth": depth,
            "delta_pose": delta_T,
            "world_pose": self._aligned_poses[idx],
            "timestamp": timestamp,
            "conf": None
        }

    def get_idx_from_timestamp(self, timestamp):
        timestamps = np.array(self._timestamps)
        idx = np.searchsorted(timestamps, float(timestamp), side='right')
        return idx if idx < len(timestamps) else None

    def replay_data(self, fps=None, start_timestamp=None, end_timestamp=None,
                    start_idx=0, end_idx=None):
        if fps is None:
            fps = self.get_sequence_frequency()
        interval = 1.0 / fps

        if start_timestamp is not None and start_idx == 0:
            start_idx = self.get_idx_from_timestamp(start_timestamp)
        if end_idx is not None:
            end_idx = min(end_idx, len(self))
        elif end_timestamp is not None:
            end_idx = self.get_idx_from_timestamp(end_timestamp)
        else:
            end_idx = len(self)

        for i in range(start_idx, end_idx):
            try:
                yield self[i]
            except Exception as e:
                print(f"Error yielding frame {i}: {e}")
            time.sleep(interval)
