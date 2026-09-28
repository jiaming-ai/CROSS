"""
2D visualization primitives based on OpenCV.
1) Plot images with `plot_image`.
2) Call `plot_keypoints` or `plot_matches` any number of times.
3) Use `update_plot` to display the image without blocking.
"""

import cv2
import numpy as np
import torch

USE_CV2 = True

def cm_RdGn(x):
    """Custom colormap: red (0) -> yellow (0.5) -> green (1)."""
    x = np.clip(x, 0, 1)[..., None] * 2
    c = x * np.array([[0, 255, 0]]) + (2 - x) * np.array([[255, 0, 0]])
    return np.clip(c, 0, 255).astype(np.uint8)

def cm_BlRdGn(x_):
    """Custom colormap: blue (-1) -> red (0.0) -> green (1)."""
    x = np.clip(x_, 0, 1)[..., None] * 2
    c = x * np.array([[0, 255, 0, 255]]) + (2 - x) * np.array([[255, 0, 0, 255]])

    xn = -np.clip(x_, -1, 0)[..., None] * 2
    cn = xn * np.array([[0, 25, 255, 255]]) + (2 - xn) * np.array([[255, 0, 0, 255]])
    out = np.clip(np.where(x_[..., None] < 0, cn, c), 0, 255).astype(np.uint8)
    return out

def cm_prune(x_):
    """Custom colormap to visualize pruning"""
    if isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
    max_i = max(x_)
    norm_x = np.where(x_ == max_i, -1, (x_ - 1) / 9)
    return cm_BlRdGn(norm_x)

def plot_image(img, title="Image", window_name="Visualization"):
    """Plot a single image using OpenCV.
    Args:
        img: NumPy RGB (H, W, 3) or PyTorch RGB (3, H, W) or mono (H, W).
        title: string, as title for the image.
        window_name: name of the OpenCV window.
    """
    if isinstance(img, torch.Tensor):
        if img.dim() == 3:
            img = img.permute(1, 2, 0).cpu().numpy()
        else:
            img = img.cpu().numpy()
    
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.setWindowTitle(window_name, title)

def plot_keypoints(img, kpts, color=(0, 255, 0), size=4):
    """Plot keypoints on the image.
    Args:
        img: NumPy array of shape (H, W, 3).
        kpts: ndarray of size (N, 2).
        color: BGR color tuple for keypoints.
        size: size of the keypoints.
    Returns:
        A new image with keypoints plotted.
    """
    if isinstance(kpts, torch.Tensor):
        kpts = kpts.cpu().numpy()
    
    # Create a copy of the image to draw on
    img_with_keypoints = img.copy()
    
    for kp in kpts:
        cv2.circle(img_with_keypoints, tuple(map(int, kp)), size, color, -1)
    
    return img_with_keypoints

def plot_masks(img, mask, color=(0, 0, 255), alpha=0.3):
    """Plot mask as an overlay on the image.
    Args:
        img: NumPy array of shape (H, W, 3).
        mask: ndarray of size (H, W) or (H, W, 1).
        color: BGR color tuple for the mask.
        alpha: float for the opacity of the mask.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    overlay = img.copy()
    overlay[mask > 0] = color
    
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def plot_matches(img1, img2, kpts1, kpts2, color=(0, 255, 0), thickness=2):
    """Plot matches between two images.
    Args:
        img1, img2: NumPy arrays of shape (H, W, 3).
        kpts1, kpts2: corresponding keypoints of size (N, 2).
        color: BGR color tuple for the lines.
        thickness: thickness of the lines.
    """
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
    if isinstance(kpts2, torch.Tensor):
        kpts2 = kpts2.cpu().numpy()
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1+w2] = img2
    
    for pt1, pt2 in zip(kpts1, kpts2):
        pt1 = tuple(map(int, pt1))
        pt2 = tuple(map(int, [pt2[0] + w1, pt2[1]]))
        cv2.line(combined_img, pt1, pt2, color, thickness)
    
    return combined_img

def add_text(img, text, pos=(10, 30), font_scale=1, color=(255, 255, 255), thickness=2):
    """Add text to the image.
    Args:
        img: NumPy array of shape (H, W, 3).
        text: string to be added.
        pos: tuple of (x, y) coordinates for the text.
        font_scale: scale of the font.
        color: BGR color tuple for the text.
        thickness: thickness of the text.
    """
    return cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def update_plot():
    """Update the plot window."""
    cv2.waitKey(1)

def show_plot(blocking=False):
    """Show the plot, either blocking or non-blocking."""
    if blocking:
        cv2.waitKey(0)
    else:
        update_plot()

def create_window(window_name="Visualization"):
    """Create a new window for plotting."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

def save_plot(path, img):
    """Save the current image."""
    cv2.imwrite(path, img)

