import pypose as pp
from typing import Tuple, Union
import torch
import numpy as np
MAX_STD = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

class OdomAccumulator():
    """Accumulate the odometry readings

    Odometry uncertainty grows proportionally to the distance traveled and rotation.
    For example, if std_per_meter=0.1 and std_per_radian=0.1, then:
    - 1m translation -> 0.1m std
    - 1 radian rotation -> 0.1 rad std
    """
    def __init__(
        self,
        std_per_meter: float = 0.1,
        std_per_radian: float = 0.1,
        min_std_translation: float = 0.01,
        min_std_rotation: float = 0.01,
    ):
        """
        Args:
            std_per_meter: Standard deviation per meter of translation (default: 0.1)
            std_per_radian: Standard deviation per radian of rotation (default: 0.1)
            min_std_translation: Minimum translation std in meters (default: 0.01)
            min_std_rotation: Minimum rotation std in radians (default: 0.01)
        """
        self._accumulated_odom = pp.identity_SE3()
        self._odoms_means = {}
        self._count_since_last_reading = {}
        self.configs = {}

        self.device = "cuda"

        # Configurable uncertainty parameters
        self.std_per_meter = std_per_meter
        self.std_per_radian = std_per_radian
        self.min_std_translation = min_std_translation
        self.min_std_rotation = min_std_rotation

    def register_item(
        self,
        name: str,
        max_std: torch.Tensor = MAX_STD,
    ):
        """Register an odometry tracking item

        Args:
            name: Name of the tracking item (e.g., 'since_last_add_kf')
            max_std: Maximum allowed std values [tx, ty, tz, rx, ry, rz]
        """
        self._odoms_means[name] = self._accumulated_odom.clone()
        self._count_since_last_reading[name] = 1
        self.configs[name] = {
            "max_std": max_std,
        }
    
    def get_since_last_reading(
        self,
        name: str,
        reset: bool = True,
        return_std: bool = True,
    ) -> Tuple[pp.SE3, pp.se3] | None:
        """Get the delta pose and uncertainty since the last reading

        Uncertainty is computed based on distance traveled and rotation:
        - Translation std = std_per_meter * translation_distance
        - Rotation std = std_per_radian * rotation_magnitude

        Returns:
            Tuple of (delta_pose, std) or None if no valid reading
        """
        if self._odoms_means[name] is None:
            # reset if there's missing reading in between
            self._odoms_means[name] = self._accumulated_odom.clone()
            self._count_since_last_reading[name] = 0
            return None, None

        # get delta pose
        delta = self._odoms_means[name].Inv() @ self._accumulated_odom
        if return_std:

            # Calculate std based on distance traveled
            delta_tensor = delta.tensor()  # [tx, ty, tz, qx, qy, qz, qw]

            # Translation distance (Euclidean norm)
            translation = delta_tensor[:3]
            translation_dist = torch.norm(translation).item()

            # Rotation magnitude (angle from quaternion)
            # For quaternion [qx, qy, qz, qw], angle = 2 * arccos(|qw|)
            quat = delta_tensor[3:]
            qw = quat[3]
            rotation_angle = 2.0 * torch.acos(torch.clamp(torch.abs(qw), -1.0, 1.0)).item()

            # Compute std proportional to motion
            trans_std = max(self.std_per_meter * translation_dist, self.min_std_translation)
            rot_std = max(self.std_per_radian * rotation_angle, self.min_std_rotation)

            # Create std tensor [tx, ty, tz, rx, ry, rz]
            std_values = torch.tensor(
                [trans_std, trans_std, trans_std, rot_std, rot_std, rot_std],
                dtype=torch.float32
            )

            # Clip to maximum allowed std
            std = pp.se3(torch.clip(std_values, max=self.configs[name]["max_std"])).to(self.device)
        else:
            std = None

        # update count and means
        self._count_since_last_reading[name] += 1

        if reset:
            self.reset_item(name)

        return delta.to(self.device), std
    
    def update_odom(
        self, 
        odom_reading: Union[pp.SE3, np.ndarray],
    ):
        """Update the accumulated odom"""
        if odom_reading is None:
            # all items will be None for next reading
            for key in self._odoms_means.keys():
                self._odoms_means[key] = None
                self._count_since_last_reading[key] = 0
            
        else:
            if isinstance(odom_reading, np.ndarray):
                odom_reading = pp.from_matrix(odom_reading, pp.SE3_type).float()
            self._accumulated_odom = self._accumulated_odom @ odom_reading

    def reset_item(self, name: str):
        """Reset the item"""
        self._odoms_means[name] = self._accumulated_odom.clone()
        self._count_since_last_reading[name] = 0

    def reset_odom(self):
        """Reset the accumulated odom"""
        self._accumulated_odom = pp.identity_SE3()
        for key in self._odoms_means.keys():
            self.reset_item(key)