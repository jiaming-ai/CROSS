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
        device: str = "cuda",
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

        self.device = device
        self._measurement_std_sums = {}

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
        self._measurement_std_sums[name] = torch.zeros(6)
    
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
            std_values = torch.clip(std_values, max=self.configs[name]["max_std"])
            # Supplied monocular uncertainty is never clipped by the historical
            # odometry ceiling: low metric certainty must remain low certainty.
            std = pp.se3(torch.maximum(std_values, self._measurement_std_sums[name])).to(self.device)
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
        covariance=None,
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
            odom_reading = odom_reading.cpu()
            if covariance is not None:
                covariance = torch.as_tensor(covariance, dtype=torch.float32, device="cpu")
                if covariance.shape != (6, 6) or not torch.isfinite(covariance).all():
                    raise ValueError("Motion covariance must be a finite 6x6 matrix")
                for name, previous in self._odoms_means.items():
                    if previous is None:
                        continue
                    delta = previous.Inv() @ self._accumulated_odom
                    matrix = delta.matrix()
                    rotation, translation = matrix[:3, :3], matrix[:3, 3]
                    tx, ty, tz = translation
                    skew = torch.zeros((3, 3), dtype=matrix.dtype)
                    skew[0, 1], skew[0, 2] = -tz, ty
                    skew[1, 0], skew[1, 2] = tz, -tx
                    skew[2, 0], skew[2, 1] = -ty, tx
                    adjoint = torch.zeros((6, 6), dtype=matrix.dtype)
                    adjoint[:3, :3] = rotation
                    adjoint[:3, 3:] = skew @ rotation
                    adjoint[3:, 3:] = rotation
                    transformed = adjoint @ covariance @ adjoint.T
                    # A common learned-scale bias persists across frames.
                    # Summing stds gives a conservative diagonal envelope even
                    # for fully correlated increments, unlike summing variances.
                    self._measurement_std_sums[name] += transformed.diagonal().clamp_min(0).sqrt()
            self._accumulated_odom = self._accumulated_odom @ odom_reading

    def reset_item(self, name: str):
        """Reset the item"""
        self._odoms_means[name] = self._accumulated_odom.clone()
        self._count_since_last_reading[name] = 0
        self._measurement_std_sums[name] = torch.zeros(6)

    def reset_odom(self):
        """Reset the accumulated odom"""
        self._accumulated_odom = pp.identity_SE3()
        for key in self._odoms_means.keys():
            self.reset_item(key)
