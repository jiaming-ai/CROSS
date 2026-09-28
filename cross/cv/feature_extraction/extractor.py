from abc import ABC, abstractmethod
import os
import logging
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cross.utils.profile import timeit

logger = logging.getLogger(__name__)

# SuperPoint pretrained weights (MagicLeap, TensorFlow -> PyTorch conversion)
SUPERPOINT_WEIGHTS_URL = (
    "https://github.com/magicleap/SuperPointPretrainedNetwork/raw/master/"
    "superpoint_v6_from_tf.pth"
)
SUPERPOINT_WEIGHTS_CACHE = os.path.join(
    os.path.expanduser("~"), ".cache", "cross", "models", "superpoint_v6_from_tf.pth"
)


def ensure_superpoint_weights(cache_path: str = SUPERPOINT_WEIGHTS_CACHE,
                              url: str = SUPERPOINT_WEIGHTS_URL) -> str:
    """Return the local path to SuperPoint weights, downloading if necessary."""
    if not os.path.isfile(cache_path):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        logger.info("Downloading SuperPoint weights to %s ...", cache_path)
        torch.hub.download_url_to_file(url, cache_path)
        logger.info("Download complete.")
    return cache_path

def bit_count(arr):
     # Make the values type-agnostic (as long as it's integers)
     t = arr.dtype.type
     mask = t(-1)
     s55 = t(0x5555555555555555 & mask)  # Add more digits for 128bit support
     s33 = t(0x3333333333333333 & mask)
     s0F = t(0x0F0F0F0F0F0F0F0F & mask)
     s01 = t(0x0101010101010101 & mask)

     arr = arr - ((arr >> 1) & s55)
     arr = (arr & s33) + ((arr >> 2) & s33)
     arr = (arr + (arr >> 4)) & s0F
     return (arr * s01) >> (8 * (arr.itemsize - 1))

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    print('inlier ratio: ', np.sum(mask)/len(mask))

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


class FeatureExtractor(ABC):
    def __init__(self, **kwargs):
        self.lowes_ratio_threshold = kwargs.get("lowes_ratio_threshold", 0.85)
        self.match_threshold = kwargs.get("match_threshold", 0.7)
        self.type = kwargs.get("type", "orb")
        self.norm = kwargs.get("norm", "euclidean")
    @abstractmethod
    def extract(self, image: torch.Tensor):
        pass
    
    @timeit
    @torch.inference_mode()
    def match_sparse(self, feats1: torch.Tensor, feats2: torch.Tensor):
        """Sparse matching
        Args:
            feats1: torch.Tensor, shape (N1, D)
            feats2: torch.Tensor, shape (N2, D)
        Returns:
            idxs0: torch.Tensor, shape (M,)
            idxs1: torch.Tensor, shape (M,)
        """
        if self.norm == "hamming":
            return self._match_hamming(feats1, feats2)
        elif self.norm == "cosine":
            return self._match_cosine(feats1, feats2)
        else:
            return self._match_euclidean(feats1, feats2)

    @staticmethod
    def draw_keypoints(img, keypoints, color=(0, 255, 0), radius=3):
        img = img.copy()
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(keypoints, np.ndarray):
            keypoints = FeatureExtractor.to_cv_keypoints(keypoints)

        img = cv2.drawKeypoints(img, keypoints, img, color)
        return img

    @staticmethod
    def plot_keypoints(img, keypoints, color=(0, 255, 0), radius=3):
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        img = FeatureExtractor.draw_keypoints(img, keypoints, color, radius)
        plt.figure(figsize=(12,12))
        plt.imshow(img[..., ::-1])
        plt.show(block=True)
        
    @staticmethod
    def to_cv_keypoints(keypoints):
        return [cv2.KeyPoint(x, y, 1) for x, y in keypoints]


    def _match_euclidean(self, feats1: torch.Tensor, feats2: torch.Tensor):
        """Sparse matching
        Args:
            feats1: torch.Tensor, shape (N1, D)
            feats2: torch.Tensor, shape (N2, D)
        Returns:
            idxs0: torch.Tensor, shape (M,)
            idxs1: torch.Tensor, shape (M,)
        """

        dist12 = (feats1[:, None, :] - feats2[None, :, :]).norm(dim=-1) # (N1, N2)
        dist21 = (feats2[:, None, :] - feats1[None, :, :]).norm(dim=-1) # (N2, N1)

        if dist12.shape[1] > 1:
            vals1, match12 = dist12.topk(k=2,dim=1, largest=False)
            vals2, match21 = dist21.topk(k=2,dim=1, largest=False)
            match12 = match12[:, 0]
            match21 = match21[:, 0]
            ratio = vals1[:, 0] / vals1[:, 1]
            good = ratio < self.lowes_ratio_threshold
        else:
            return None, None

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]

        return idx0, idx1

    def _match_cosine_lowes(self, feats1: torch.Tensor, feats2: torch.Tensor):
        """Sparse matching
        Args:
            feats1: torch.Tensor, shape (N1, D)
            feats2: torch.Tensor, shape (N2, D)
        Returns:
            idxs0: torch.Tensor, shape (M,)
            idxs1: torch.Tensor, shape (M,)
        """


        dist12 = feats1 @ feats2.t() # (N1, N2)
        dist21 = feats2 @ feats1.t() # (N2, N1)

        vals1, match12 = dist12.topk(k=2,dim=1, largest=True)
        vals2, match21 = dist21.topk(k=2,dim=1, largest=True)

        match12 = match12[:, 0]
        match21 = match21[:, 0]

        ratio = vals1[:, 1] / vals1[:, 0]
        good_ratio = ratio < self.lowes_ratio_threshold

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if self.match_threshold > 0:
            good_threshold = vals1[:, 0] > self.match_threshold
            good = mutual & good_threshold & good_ratio
            idx0 = idx0[good]
            idx1 = match12[good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1
    
    def _match_cosine(self, feats1: torch.Tensor, feats2: torch.Tensor):
        """Sparse matching
        Args:
            feats1: torch.Tensor, shape (N1, D)
            feats2: torch.Tensor, shape (N2, D)
        Returns:
            idxs0: torch.Tensor, shape (M,)
            idxs1: torch.Tensor, shape (M,)
        """


        dist12 = feats1 @ feats2.t() # (N1, N2)
        dist21 = feats2 @ feats1.t() # (N2, N1)

        dist12_max, match12 = dist12.max(dim=1)
        _, match21 = dist21.max(dim=1)


        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        if self.match_threshold > 0:
            good_threshold = dist12_max > self.match_threshold
            good = mutual & good_threshold
            idx0 = idx0[good]
            idx1 = match12[good]
        else:
            idx0 = idx0[mutual]
            idx1 = match12[mutual]

        return idx0, idx1


    def _match_hamming(self, feats1: torch.Tensor, feats2: torch.Tensor):
        """Sparse matching
        Args:
            feats1: torch.Tensor, shape (N1, D)
            feats2: torch.Tensor, shape (N2, D)
        Returns:
            idxs0: torch.Tensor, shape (M,)
            idxs1: torch.Tensor, shape (M,)
        """


        # TODO: optimize this
        # bitwise hamming distance
        xor1 = feats1.unsqueeze(1) ^ feats2.unsqueeze(0) # (N1, N2, D)
        xor2 = feats2.unsqueeze(1) ^ feats1.unsqueeze(0) # (N2, N1, D)
        bit_count1 = bit_count(xor1.cpu().numpy())
        bit_count2 = bit_count(xor2.cpu().numpy())
        dist12 = torch.from_numpy(bit_count1).to(feats1.device).sum(dim=-1) # (N1, N2)
        dist21 = torch.from_numpy(bit_count2).to(feats1.device).sum(dim=-1) # (N2, N1)

        assert self.lowes_ratio_threshold > 0
        if dist12.shape[1] > 1:
            vals1, match12 = dist12.topk(k=2,dim=1, largest=False)
            vals2, match21 = dist21.topk(k=2,dim=1, largest=False)
            match12 = match12[:, 0]
            match21 = match21[:, 0]
            ratio = vals1[:, 0] / vals1[:, 1]
            good = ratio < self.lowes_ratio_threshold
        else:
            return None, None

        idx0 = torch.arange(len(match12), device=match12.device)
        mutual = match21[match12] == idx0

        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]

        return idx0, idx1










# class BaseFeatureExtractor(ABC):
#     def __init__(self):
#         pass

#     @abstractmethod
#     def extract_features(self, x):
#         pass

#     @staticmethod
#     def draw_keypoints(img, keypoints, color=(0, 255, 0), radius=3):
#         img = img.copy()
#         if isinstance(keypoints, torch.Tensor):
#             keypoints = keypoints.cpu().numpy()
#         if isinstance(keypoints, np.ndarray):
#             keypoints = BaseFeatureExtractor.to_cv_keypoints(keypoints)

#         img = cv2.drawKeypoints(img, keypoints, img, color)
#         return img
    
    
#     @staticmethod
#     def to_cv_keypoints(keypoints):
#         return [cv2.KeyPoint(x, y, 1) for x, y in keypoints]
    
#     def extract_SIFT(self, img):
#         # input image is converted to gray scale image 
#         imagegray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
#         # using the SIRF algorithm to detect key 
#         # points in the image 
#         features = cv2.SIFT_create() 
#         keypoints = features.detect(imagegray, None) 
        
#         return keypoints, None

#     @staticmethod
#     def read_image(img_file, img_size):
#         img = cv2.imread(img_file, cv2.IMREAD_COLOR)
#         img = cv2.resize(img, img_size)
#         img_orig = img.copy()
#         img_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img = np.expand_dims(img, 2)
#         img = img.astype(np.float32)
#         img_preprocessed = img / 255.

#         return img_preprocessed, img_orig

# class SuperPointFeatureExtractor(BaseFeatureExtractor):
#     def __init__(self, device='cuda', max_keypoints=2000):
#         """
#         """
#         self.model = SuperPoint(
#                                 detection_threshold=0.2,
#                                 max_num_keypoints=max_keypoints)
        
#         weights_path = ensure_superpoint_weights()
#         state_dict = torch.load(weights_path, map_location=device)
#         self.model.load_state_dict(state_dict)
#         self.model.eval()
#         self.model.to(device)
#         self.device = device

#     def extract_features(self, x):
#         """
#         args:
#             x: torch.Tensor (B, C, H, W)
#         returns:
#             keypoints: torch.Tensor (B, N, 2)
#             descriptors: torch.Tensor (B, N, 256)
#         """
#         if isinstance(x, np.ndarray):
#             x = torch.from_numpy(x).float()
#         if x.dim() == 3:
#             x = x.unsqueeze(0)
#         if x.shape[-1] < 5: # channel first
#             x = x.permute(0, 3, 1, 2)

#         with torch.inference_mode():
#             out = self.model(x.to(self.device))
        
#         return out["keypoints"], out["descriptors"]
