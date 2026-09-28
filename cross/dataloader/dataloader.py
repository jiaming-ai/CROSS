from abc import ABC, abstractmethod
import time
import numpy as np
import cv2
from PIL import Image

np.random.seed(42)


class Dataloader(ABC):
    """
    Abstract class for all dataloaders.
    """

    def __init__(
        self,
        snr=None,
        target_width: int | None = None,
        target_height: int | None = None,
        short_side: int = 480,
        **kwargs,
    ):
        self.snr = snr
        self.target_width = target_width
        self.target_height = target_height
        self.short_side = short_side
        self.crop_center_h = float(kwargs.get("crop_center_h", 0.5))
        self.crop_center_w = float(kwargs.get("crop_center_w", 0.5))
        # clamp to [0,1] to avoid surprises
        self.crop_center_h = min(1.0, max(0.0, self.crop_center_h))
        self.crop_center_w = min(1.0, max(0.0, self.crop_center_w))

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def get_sequence_frequency(self):
        pass

    @abstractmethod
    def get_idx_from_timestamp(self, timestamp):
        pass

    def get_item(self, idx, first_item=False):
        item = self[idx]
        if item.get("delta_pose") is not None and self.snr is not None:
            item["delta_pose"] = self._apply_se3_noise_snr(item["delta_pose"], snr=self.snr)
        if first_item:
            item["delta_pose"] = np.eye(4)
        return item

    # ---------- resizing helpers shared across dataloaders ----------

    def _compute_resize_config(self, orig_w: int, orig_h: int):
        """
        Compute a resize + crop/pad plan.
        - If target_width/height are provided, we scale uniformly by width (if set) or height,
          then crop/pad (optionally off-center) to reach the requested dims. This keeps horizontal
          FoV when forcing a square (e.g., 512x512).
        - Otherwise fall back to short_side scaling.
        """
        desired_w = self.target_width
        desired_h = self.target_height

        if desired_w is None and desired_h is None:
            scale = self.short_side / min(orig_h, orig_w)
        elif desired_w is not None:
            scale = desired_w / orig_w
        else:
            scale = desired_h / orig_h

        scaled_w = int(round(orig_w * scale))
        scaled_h = int(round(orig_h * scale))

        final_w = desired_w if desired_w is not None else scaled_w
        final_h = desired_h if desired_h is not None else scaled_h

        crop_left, crop_right, pad_left, pad_right = self._crop_pad_amounts(
            scaled_w, final_w, center_ratio=self.crop_center_w
        )
        crop_top, crop_bottom, pad_top, pad_bottom = self._crop_pad_amounts(
            scaled_h, final_h, center_ratio=self.crop_center_h
        )

        return {
            "scale_w": scaled_w / orig_w,
            "scale_h": scaled_h / orig_h,
            "scaled_w": scaled_w,
            "scaled_h": scaled_h,
            "final_w": final_w,
            "final_h": final_h,
            "crop_left": crop_left,
            "crop_right": crop_right,
            "crop_top": crop_top,
            "crop_bottom": crop_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
        }

    @staticmethod
    def _crop_pad_amounts(current: int, target: int, center_ratio: float = 0.5):
        """
        Compute crop/pad sizes with an optional center offset (0=bottom, 0.5=center, 1=top for height).
        """
        diff = current - target
        center_ratio = min(1.0, max(0.0, center_ratio))
        if diff >= 0:
            center_pos = current * center_ratio
            start = int(round(center_pos - target / 2.0))
            start = max(0, min(start, current - target))
            end = start + target
            crop_before = start
            crop_after = current - end
            pad_before = pad_after = 0
        else:
            pad_total = -diff
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            crop_before = crop_after = 0
        return crop_before, crop_after, pad_before, pad_after

    def _resize_image(self, image, cfg, mode: str):
        """Resize with uniform scaling then crop/pad to final shape."""
        if image is None:
            return None

        resample = Image.BILINEAR
        pad_value = 0
        if mode in ("depth", "conf"):
            resample = Image.NEAREST
        if mode == "conf":
            pad_value = False

        resized = np.array(Image.fromarray(image).resize((cfg["scaled_w"], cfg["scaled_h"]), resample=resample))

        # crop if needed
        if cfg["crop_top"] or cfg["crop_bottom"]:
            h_start = cfg["crop_top"]
            h_end = resized.shape[0] - cfg["crop_bottom"]
            resized = resized[h_start:h_end, :]
        if cfg["crop_left"] or cfg["crop_right"]:
            w_start = cfg["crop_left"]
            w_end = resized.shape[1] - cfg["crop_right"]
            resized = resized[:, w_start:w_end]

        # pad if needed
        if any([cfg["pad_top"], cfg["pad_bottom"], cfg["pad_left"], cfg["pad_right"]]):
            if resized.ndim == 3:
                pads = (
                    (cfg["pad_top"], cfg["pad_bottom"]),
                    (cfg["pad_left"], cfg["pad_right"]),
                    (0, 0),
                )
            else:
                pads = (
                    (cfg["pad_top"], cfg["pad_bottom"]),
                    (cfg["pad_left"], cfg["pad_right"]),
                )
            resized = np.pad(resized, pads, mode="constant", constant_values=pad_value)

        return resized

    def _resize_intrinsics(self, K: np.ndarray, cfg):
        """Scale intrinsics then adjust for crop/pad offsets."""
        K_scaled = K.copy()
        K_scaled[0, 0] *= cfg["scale_w"]
        K_scaled[0, 2] *= cfg["scale_w"]
        K_scaled[1, 1] *= cfg["scale_h"]
        K_scaled[1, 2] *= cfg["scale_h"]

        # crop subtracts pixels, padding adds pixels
        K_scaled[0, 2] -= cfg["crop_left"]
        K_scaled[1, 2] -= cfg["crop_top"]
        K_scaled[0, 2] += cfg["pad_left"]
        K_scaled[1, 2] += cfg["pad_top"]
        return K_scaled

    # ---------- noise helpers ----------

    def _apply_se3_noise_snr(
        self,
        T,
        snr=1.0,
        mode="separate",             # "separate" (recommended), or "combined"
        characteristic_length=1.0,   # meters per rad, used in "combined"
        use_expected_norm=True,      # include sqrt(3) so SNR = signal / E||noise||
        min_std_trans=0.0,           # meters
        min_std_rot=0.0,             # radians
        rot_clip_rad=None,           # optional cap on ||rot noise|| (radians)
        rng=None                     # np.random.Generator for reproducibility
    ):
        """
        Apply SE(3) noise with SNR semantics.
        - Translation noise ~ N(0, sigma_t^2 I_3) in meters (applied in body frame)
        - Rotation noise ~ N(0, sigma_r^2 I_3) in axis-angle radians
        SNR definitions:
            separate mode: sigma_t = ||t|| / (snr * sqrt(3)), sigma_r = ||logR|| / (snr * sqrt(3))
            combined mode: uses combined signal via characteristic_length
        """
        if T is None:
            return T

        dtype = T.dtype if hasattr(T, "dtype") else np.float64
        T = T.astype(dtype, copy=False)

        if rng is None:
            rng = np.random.default_rng()

        # Extract pose
        R = T[:3, :3]
        t = T[:3, 3]

        # SO(3) log map via cv2.Rodrigues
        # cv2.Rodrigues(R) -> (rot_vec(3x1), jacobian) where rot_vec is axis-angle vector
        rot_vec_R, _ = cv2.Rodrigues(R)
        rot_vec_R = rot_vec_R.flatten()

        # Signal magnitudes
        s_t = np.linalg.norm(t)
        s_r = np.linalg.norm(rot_vec_R)

        # Denominator for mapping signal -> per-axis std
        denom = np.sqrt(3.0) if use_expected_norm else 1.0

        # Decide sigmas
        if mode == "separate":
            sigma_t = s_t / (snr * denom) if snr > 0 else 0.0
            sigma_r = s_r / (snr * denom) if snr > 0 else 0.0

        elif mode == "combined":
            # Combined metric with characteristic length (meters per rad)
            s_comb = np.sqrt(s_t**2 + (characteristic_length * s_r)**2)
            sigma_t = s_comb / (snr * denom) if snr > 0 else 0.0
            # Keep rotation consistent with length scaling
            sigma_r = (s_comb / characteristic_length) / (snr * denom) if (snr > 0 and characteristic_length > 0) else 0.0
        else:
            raise ValueError("mode must be 'separate', or 'combined'.")

        # Floors for near-identity or tiny signals
        sigma_t = max(sigma_t, float(min_std_trans))
        sigma_r = max(sigma_r, float(min_std_rot))

        # Sample noise in tangent space
        rot_noise = rng.normal(0.0, sigma_r, size=3).astype(dtype)
        if rot_clip_rad is not None:
            # Clip the axis-angle magnitude, not per-axis components
            mag = np.linalg.norm(rot_noise)
            if mag > rot_clip_rad and mag > 1e-12:
                rot_noise = rot_noise * (rot_clip_rad / mag)

        t_noise = rng.normal(0.0, sigma_t, size=3).astype(dtype)

        # Build noise transform: Exp(se3)
        R_noise, _ = cv2.Rodrigues(rot_noise)
        T_noise = np.eye(4, dtype=dtype)
        T_noise[:3, :3] = R_noise
        T_noise[:3, 3] = t_noise

        # Right-multiply: T' = T @ T_noise  (body-frame perturbation)
        return T @ T_noise

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
        if fps is None:
            fps = self.get_sequence_frequency()

        # get the interval
        interval = 1.0 / fps

        # get the start index
        if start_timestamp is not None and start_idx == 0:
            start_idx = self.get_idx_from_timestamp(start_timestamp)

        if end_idx is not None:
            end_idx = min(end_idx, len(self))
        elif end_timestamp is not None:
            end_idx = self.get_idx_from_timestamp(end_timestamp)
        else:
            end_idx = len(self)

        for i in range(start_idx, end_idx):
            yield self.get_item(i, i==start_idx)
            time.sleep(interval)
