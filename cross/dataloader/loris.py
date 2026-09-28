import pandas as pd
import numpy as np
import os
from cross.utils.rotation import quaternion_to_rotation_matrix_numpy
import pickle
import cv2
import time
import json
import yaml

def get_delta_t(
    closest_odometry_idx: np.ndarray,
    odometry_data: pd.DataFrame,
    ) -> list[np.ndarray]:
    """Get the delta T between two images
    Args:
        closest_odometry_idx: the index of the odometry timestamp
        that corresponds to each image timestamp
        odometry_data: the odometry data
    Returns:
        delta_T: the delta T between the two images
    """
    delta_T = []
    # add the first delta T, which is the identity
    delta_T.append(np.eye(4)) 

    # get T_0
    _, px,py,pz, qx,qy,qz,qw, _, _, _, _, _, _ = odometry_data.iloc[closest_odometry_idx[0]]
    T_0 = np.eye(4)
    T_0[:3, 3] = np.array([px, py, pz])
    T_0[:3, :3] = quaternion_to_rotation_matrix_numpy([qx, qy, qz, qw])

    prev_T = T_0
    for i in range(1,len(closest_odometry_idx)):
        _, px,py,pz, qx,qy,qz,qw, _, _, _, _, _, _ = odometry_data.iloc[closest_odometry_idx[i]]
        T = np.eye(4)
        T[:3, 3] = np.array([px, py, pz])
        T[:3, :3] = quaternion_to_rotation_matrix_numpy([qx, qy, qz, qw])
        delta_T.append(np.linalg.inv(prev_T) @ T)
        prev_T = T

    return delta_T

GT_TO_OPENCV_TRANSFORM = np.array([
        [ 0, -1,  0,  0],
        [ 0,  0, -1,  0],
        [ 1,  0,  0,  0],
        [ 0,  0,  0,  1]
    ], dtype=np.float64)
GT_TO_OPENCV_TRANSFORM_INV = np.linalg.inv(GT_TO_OPENCV_TRANSFORM)

class OpenLorisLoader():
    """Loader for the OpenLoris dataset
    The coordinates are x forward, y left, z up
    (https://lifelong-robotic-vision.github.io/dataset/scene.html)
    """
    
    def __init__(self, scene_path: str, seq_idx: int = 0):
        self.scene_path = scene_path

        # get all sequences in the scene
        sequences = sorted([os.path.join(scene_path, seq) for seq in os.listdir(scene_path)])
        self._sequence_paths = [seq for seq in sequences if os.path.isdir(seq)]
        self._num_sequences = len(self._sequence_paths)

        # data
        self._cur_seq_idx = 0
        self._cur_seq_path = None
        self._aligned_odometry = None
        self._rgb_files = None
        self._depth_files = None
        self._ground_truth_poses = None
        # preprocess the data
        for seq_idx, seq_path in enumerate(self._sequence_paths):
            # try to read aligned odometry timestamps
            aligned_odometry_path = os.path.join(seq_path, "aligned_odometry.pkl")
            if not os.path.exists(aligned_odometry_path):
                print(f"Processing {seq_path}")
                self._align_odometry(seq_path)

            if not os.path.exists(os.path.join(seq_path, "ground_truth_poses.pkl")):
                self._align_ground_truth(seq_path)
            
        self.load_sequence(seq_idx)


    def _load_camera_intrinsics(self):
        """Load the camera intrinsics from the yaml file"""
        assert self._cur_seq_path is not None, "A sequence must be loaded first"
        yaml_file = os.path.join(self._cur_seq_path, "sensors.yaml")
        
        # Use cv2.FileStorage to read OpenCV YAML format
        fs = cv2.FileStorage(yaml_file, cv2.FILE_STORAGE_READ)
        
        # Load color camera intrinsics
        color_data = fs.getNode('d400_color_optical_frame')
        # color_data = fs.getNode('d400_depth_optical_frame')
        intrinsics = color_data.getNode('intrinsics').mat()
        fx, fy, cx, cy = intrinsics[0]
        self.rgb_width = int(color_data.getNode('width').real())
        self.rgb_height = int(color_data.getNode('height').real())
        self.rgb_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # Load depth camera intrinsics
        depth_data = fs.getNode('d400_depth_optical_frame')
        intrinsics = depth_data.getNode('intrinsics').mat()
        fx, fy, cx, cy = intrinsics[0]
        self.depth_width = int(depth_data.getNode('width').real())
        self.depth_height = int(depth_data.getNode('height').real())
        self.depth_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        fs.release()

        # Load the transformation matrix
        trans_matrix_path = os.path.join(self._cur_seq_path, "trans_matrix.yaml")
        fs_trans = cv2.FileStorage(trans_matrix_path, cv2.FILE_STORAGE_READ)
        trans_matrix_node = fs_trans.getNode('trans_matrix')
        
        # Get number of transformations
        num_transforms = trans_matrix_node.size()
        
        # Iterate through transformations
        for i in range(num_transforms):
            transform = trans_matrix_node.at(i)
            child_frame = transform.getNode('child_frame').string()
            parent_frame = transform.getNode('parent_frame').string()
            
            if child_frame == 'd400_color_optical_frame' and \
               parent_frame == 'base_link':
                self.T_base_rgb = transform.getNode('matrix').mat()
            elif child_frame == 'd400_depth_optical_frame' and \
                 parent_frame == 'd400_color_optical_frame':
                self.T_rgb_depth = transform.getNode('matrix').mat()
        
        fs_trans.release()

    def _align_ground_truth(self, seq_path):
        """Align the rgb timestamps to the odometry timestamps"""
        df_gt = pd.read_csv(
            os.path.join(seq_path, "groundtruth.txt"),
            comment='#',
            header=None,
            sep='\s+'
        )
        ground_truth_timestamps = df_gt.iloc[:, 0].to_numpy()

        # align the rgb timestamps to the ground truth timestamps
        df_rgb = pd.read_csv(
            os.path.join(seq_path, "color.txt"),
            comment='#',
            header=None,
            sep='\s+'
        )
        rgb_timestamps = df_rgb.iloc[:, 0].to_numpy()

        # for each rgb timestamp, find the closest odometry timestamp
        distance = np.abs(ground_truth_timestamps[None, :] - rgb_timestamps[:, None])
        closest_rgb_idx = np.argmin(distance, axis=1)

        ground_truth_poses = []
        for i in range(len(closest_rgb_idx)):
            _, px,py,pz, qx,qy,qz,qw = df_gt.iloc[closest_rgb_idx[i]]
            T = np.eye(4)
            T[:3, 3] = np.array([px, py, pz])
            T[:3, :3] = quaternion_to_rotation_matrix_numpy([qx, qy, qz, qw])
            ground_truth_poses.append(T)

        
        with open(seq_path + "/ground_truth_poses.pkl", "wb") as f:
            pickle.dump(ground_truth_poses, f)
        
    def _align_odometry(self, seq_path):
        """Align the rgb timestamps to the odometry timestamps"""
        df = pd.read_csv(
            os.path.join(seq_path, "color.txt"),
            comment='#',
            header=None,
            sep='\s+'
        )
        rgb_timestamps = df.iloc[:, 0].to_numpy()
        
        # align the rgb timestamps to the odometry timestamps
        df = pd.read_csv(
            os.path.join(seq_path, "odom.txt"),
            comment='#',
            header=None,
            sep='\s+'
        )
        odometry_timestamps = df.iloc[:, 0].to_numpy()

        # for each rgb timestamp, find the closest odometry timestamp
        distance = np.abs(rgb_timestamps[:, None] - odometry_timestamps[None, :])
        closest_odometry_idx = np.argmin(distance, axis=1)

        delta_T = get_delta_t(closest_odometry_idx, df)

        # save the delta T
        with open(seq_path + "/aligned_odometry.pkl", "wb") as f:
            pickle.dump(delta_T, f)

    def load_sequence(self, seq_idx):
        assert seq_idx < self._num_sequences, "Sequence index out of bounds"
        seq_path = self._sequence_paths[seq_idx]
        aligned_odometry = os.path.join(seq_path, "aligned_odometry.pkl")
        ground_truth_poses = os.path.join(seq_path, "ground_truth_poses.pkl")
        with open(aligned_odometry, "rb") as f:
            self._aligned_odometry = pickle.load(f)
        with open(ground_truth_poses, "rb") as f:
            self._ground_truth_poses = pickle.load(f)

        rgb_file = os.path.join(seq_path, "color.txt")
        df = pd.read_csv(
            rgb_file,
            comment='#',
            header=None,
            sep='\s+'
        )
        self._rgb_files = df.iloc[:, 1].to_list()

        depth_file = os.path.join(seq_path, "aligned_depth.txt")
        df = pd.read_csv(
            depth_file,
            comment='#',
            header=None,
            sep='\s+'
        )
        self._depth_files = df.iloc[:, 1].to_list()

        self._cur_seq_idx = seq_idx
        self._cur_seq_path = seq_path

        self._load_camera_intrinsics()

    def get_sequence_frequency(self):
        """Calculate the approximate frequency of the image stream for a sequence.
        
        Args:
            seq_idx: Index of the sequence to analyze
            
        Returns:
            float: Approximate frequency in Hz
        """
        df = pd.read_csv(
            os.path.join(self._cur_seq_path, "color.txt"),
            comment='#',
            header=None,
            sep='\s+'
        )
        timestamps = df.iloc[:, 0].to_numpy()
        
        # Calculate average time difference between consecutive frames
        time_diffs = timestamps[1:] - timestamps[:-1]
        avg_period = np.mean(time_diffs)
        
        # Frequency is 1/period
        frequency = 1.0 / avg_period
        
        return frequency

    def __len__(self):
        assert self._rgb_files is not None, "A sequence must be loaded first"
        return len(self._rgb_files)

    def __getitem__(self, idx):
        assert self._rgb_files is not None, "A sequence must be loaded first"
        rgb_file = os.path.join(self._cur_seq_path, self._rgb_files[idx])
        rgb = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
        depth_file = os.path.join(self._cur_seq_path, self._depth_files[idx])
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 1000.0 # convert to meters
        delta_T = self._aligned_odometry[idx]
        delta_T = GT_TO_OPENCV_TRANSFORM @ delta_T @ GT_TO_OPENCV_TRANSFORM_INV
        ground_truth_pose = self._ground_truth_poses[idx]
        ground_truth_pose = GT_TO_OPENCV_TRANSFORM @ ground_truth_pose @ GT_TO_OPENCV_TRANSFORM_INV

        return {
            "rgb": rgb,
            "depth": depth,
            "delta_pose": delta_T,
            "world_pose": ground_truth_pose,
            "timestamp": self._rgb_files[idx],
            "conf": None
        }

    def load_test_json(self):
        assert self._cur_seq_path is not None, "A sequence must be loaded first"
        if os.path.exists(os.path.join(self._cur_seq_path, "test.json")):
            with open(os.path.join(self._cur_seq_path, "test.json"), "r") as f:
                self._test_json = json.load(f)
        else:
            # return a default test.json
            self._test_json = {
                "map_start_timestamp": None,
                "map_end_timestamp": None
            }

        return self._test_json

    def get_idx_from_timestamp(self, timestamp):
        """Get the index of the frame closest to the given timestamp.
        The index is the immediate next frame.
        
        Args:
            timestamp (float): The timestamp to search for
        Returns:
            idx (int): The index of the frame that comes immediately after
            the given timestamp, or None if the timestamp is after the last frame
        """
        assert self._rgb_files is not None, "A sequence must be loaded first"
        
        # Read timestamps from color.txt
        df = pd.read_csv(
            os.path.join(self._cur_seq_path, "color.txt"),
            comment='#',
            header=None,
            sep='\s+'
        )
        timestamps = df.iloc[:, 0].to_numpy()
        
        # Find index of first timestamp greater than input
        next_indices = np.where(timestamps > timestamp)[0]
        
        if len(next_indices) == 0:  # timestamp is after last frame
            return None
        
        return next_indices[0]  # return first index where timestamp is greater
        

    def replay_data(
        self, 
        fps=None, 
        start_timestamp=None, 
        end_timestamp=None,
        start_idx=0,
        end_idx=None,
    ):
        """Replay the sequence to the given frequency
        Args:
            fps (float): The frequency to replay to
        """
        assert self._rgb_files is not None, "A sequence must be loaded first"

        if fps is None:
            fps = self.get_sequence_frequency()

        # get the interval
        interval = 1.0 / fps

        # get the start index
        if start_timestamp is not None and start_idx == 0:
            start_idx = self.get_idx_from_timestamp(start_timestamp)

        if end_idx is not None:
            end_idx = min(end_idx, len(self._rgb_files))
        elif end_timestamp is not None:
            end_idx = self.get_idx_from_timestamp(end_timestamp)
        else:
            end_idx = len(self._rgb_files)

        for i in range(start_idx, end_idx):
            yield self[i]
            time.sleep(interval)

    def replay_data_msckf(
        self,
        start_timestamp=None,
        end_timestamp=None,
        start_idx=0,
        end_idx=None,
    ):
        # get the start index
        if start_timestamp is not None and start_idx == 0:
            start_idx = self.get_idx_from_timestamp(start_timestamp)

        if end_idx is not None:
            end_idx = min(end_idx, len(self._rgb_files))
        elif end_timestamp is not None:
            end_idx = self.get_idx_from_timestamp(end_timestamp)
        else:
            end_idx = len(self._rgb_files)