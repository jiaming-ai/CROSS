from cross.utils.profile import timeit
import pypose as pp
import torch
import numpy as np
from .lie_tensor import SE3_Adj
from sklearn.mixture import GaussianMixture


def convolve_gmm_batch_SE3(
    kf_gmm_means: pp.LieTensor,
    kf_gmm_stds_diag: pp.LieTensor,
    delta_means: pp.LieTensor,
    delta_stds_diag: pp.LieTensor,
    kf_gmm_component_weights: torch.Tensor,
) -> tuple[pp.LieTensor, pp.LieTensor]:
    """Performs batched convolution of kf GMMs with relative PnP Gaussians.

    This function takes a batch of B kfs, each with a K-component GMM,
    and convolves them with B corresponding single-Gaussian relative poses from PnP.
    It correctly handles pypose SE3 group elements and se3 Lie algebra vectors.

    Args:
        kf_gmm_means (pp.SE3): (B, K, 7) the means of K components for B kfs.
        kf_gmm_stds_diag (pp.se3): (B, K, 6) the diagonal of the stds matrices for the
                                              kf GMM components (in se3 Lie algebra).
        delta_means (pp.SE3): (B, 7) the mean of the relative pose, T_ref_current 
        delta_stds_diag (pp.se3): (B, 6) the diagonal of the stds 
        kf_gmm_component_weights (torch.Tensor): (B, K) the weights of the kf GMM components.
    Returns:
        tuple[pp.SE3, pp.se3]:
            - new_means (pp.SE3): The means of the convolved GMM, shape (B, K).
            - new_stds_diag (pp.se3): The diagonal of the stds of the convolved GMM,
                                            shape (B, K, 6).
    """
    # --- 1. Broadcasting and Reshaping for Batch Operations ---
    pnp_means_expanded = delta_means.unsqueeze(1) # (B, 1, 7)
    pnp_stds_diag_expanded = delta_stds_diag.tensor().unsqueeze(1) # (B, 1, 6)

    # --- 2. Batched SE(3) Composition for the New Means ---
    # Operation: (B, K) @ (B, 1) -> (B, K)
    new_means = kf_gmm_means @ pnp_means_expanded # (B, K)

    new_stds_diag = pp.se3((kf_gmm_stds_diag**2 + pnp_stds_diag_expanded**2)**0.5)

    # reset the mu and sigma from components with zero weight to identity
    zero_component_weights = kf_gmm_component_weights < 1e-3
    new_means[zero_component_weights] = pp.identity_SE3(device=new_means.device)
    new_stds_diag[zero_component_weights] = pp.identity_se3(device=new_stds_diag.device)

    return new_means, new_stds_diag


    


