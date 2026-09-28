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


def sample_gmm_numpy_vectorized(n_samples, weights, means, stds):
    """
    Vectorized GMM sampling with diagonal covariances using NumPy.

    Args:
        n_samples (int): Number of samples.
        weights (np.ndarray): (K,) mixture weights.
        means (np.ndarray): (K, D) component means.
        stds (np.ndarray): (K, D) component std deviations (sqrt of diagonal covariances).

    Returns:
        samples (np.ndarray): (n_samples, D)
    """
    K, D = means.shape

    # 1. Sample component indices
    component_indices = np.random.choice(K, size=n_samples, p=weights)

    # 2. Gather means and stds for each sampled component
    selected_means = means[component_indices]
    selected_stds = stds[component_indices]

    # 3. Sample standard normals and scale
    standard_normals = np.random.randn(n_samples, D)
    samples = selected_means + selected_stds * standard_normals
    return samples


def sample_gmm_torch_vectorized_SE3(
    n_samples: int,
    weights: torch.Tensor,
    means: pp.LieTensor,
    stds: pp.LieTensor,
):
    """Vectorized GMM sampling with diagonal covariances in SE(3)
    Important: the stds are in the local frame of the kf,
    so we need to use the right perturbation here

    Args:
        n_samples (int): Number of samples.
        weights (Tensor): (K,) mixture weights.
        means (pp.SE3): (K, 7) means of components.
        stds (pp.se3): (K, 6) std deviations of components.

    Returns:
        samples (Tensor): (n_samples, 7)
    """
    device = means.device

    # 1. Sample component indices
    cat = torch.distributions.Categorical(weights)
    component_indices = cat.sample((n_samples,))  # (n_samples,)

    # 2. Index means and stds
    selected_means = means[component_indices]     # (n_samples, 7)
    selected_stds = stds[component_indices]       # (n_samples, 6)


    z_samples = torch.randn(n_samples, 6, device=device)
    
    # 2. Scale by the standard deviation (sqrt of variance).
    # This works element-wise because the covariances are diagonal.
    tangent_space_samples = z_samples * selected_stds.tensor()
    
    # 3. Convert the tangent space perturbations to SE3 group elements.
    # This creates N small random transformations.
    perturbations = pp.se3(tangent_space_samples).Exp()

    samples = selected_means @ perturbations

    return samples


@timeit
def hierarchical_sampling_SE3(
    n_samples: int,
    retrieval_weights: torch.Tensor,
    gmm_mus: pp.LieTensor,
    gmm_sigmas: pp.LieTensor,
    gmm_weights: torch.Tensor,
) -> pp.LieTensor:
    """Performs batched hierarchical sampling for N particles.
    
    This function efficiently generates N samples by performing the three
    sampling stages in a vectorized manner.

    1. Sample N keyframe indices from the retrieval weights.
    2. For each chosen keyframe, sample a component index from its GMM weights.
    3. For each chosen (keyframe, component) pair, sample from the corresponding Gaussian.

    Args:
        retrieval_weights (torch.Tensor): Tensor of shape (B,) containing the probabilities
                                            for choosing each of the B retrieved keyframes.
        gmm_mus (pp.SE3): Tensor of shape (B, K, 7) containing the SE3 means
                                of K components for B landmark GMMs.
        gmm_sigmas (pp.se3): Tensor of shape (B, K, 6) containing the diagonal
                                    variances for the K components of B landmark GMMs.
        gmm_weights (torch.Tensor): Tensor of shape (B, K) containing the component
                                    weights for the K components of B landmark GMMs.
    
    Returns:
        pypose.SE3: A pypose.SE3 tensor of shape (N,) containing the final sampled poses,
                    where N is n_samples.
    """
    B, K = gmm_weights.shape
    device = retrieval_weights.device

    # --- Stage 1: Sample Keyframe Indices ---
    # Sample N keyframe indices based on the retrieval weights.
    # This gives us which of the B keyframes each particle should use.
    # Output shape: (N,)
    kf_indices = torch.multinomial(
        retrieval_weights,
        num_samples=n_samples,
        replacement=True
    )

    # --- Stage 2: Sample Component Indices ---
    # For each of the N chosen keyframes, we now need to sample a component.
    # We first gather the GMM weights corresponding to the chosen keyframes.
    # gmm_weights shape: (B, K)
    # kf_indices shape: (N,)
    # selected_gmm_weights shape: (N, K)
    selected_gmm_weights = gmm_weights[kf_indices]
    
    # Now sample a component index for each of the N particles from its
    # corresponding K-dimensional weight vector.
    # Output shape: (N, 1), which we squeeze to (N,)
    comp_indices = torch.multinomial(
        selected_gmm_weights,
        num_samples=1
    ).squeeze(-1)

    # --- Stage 3: Sample from the Chosen Gaussians ---
    # We now have the specific (keyframe, component) pair for each particle.
    # We use advanced indexing to select the corresponding means and sigmas.
    
    # Select the N means. Shape: (N,)
    chosen_means = gmm_mus[kf_indices, comp_indices]
    
    # Select the N diagonal variances. Shape: (N,)
    chosen_sigmas_diag = gmm_sigmas[kf_indices, comp_indices]


    # Now, generate N samples from the N chosen Gaussian distributions.
    # 1. Generate N standard normal random vectors in the Lie algebra.
    z_samples = torch.randn(n_samples, 6, device=device)
    
    # 2. Scale by the standard deviation (sqrt of variance).
    # This works element-wise because the covariances are diagonal.
    tangent_space_samples = z_samples * chosen_sigmas_diag.tensor()
    
    # 3. Convert the tangent space perturbations to SE3 group elements.
    # This creates N small random transformations.
    perturbations = pp.se3(tangent_space_samples).Exp()
    
    # 4. Apply the perturbations to the means.
    # We use left-multiplication: new_pose = perturbation * mean_pose
    # This is a common convention for applying noise on manifolds.
    # NOTE: the variance is defined in the local frame of the kf,
    # so we need to use the right perturbation here
    final_samples = chosen_means @ perturbations
    
    return final_samples

    

@timeit
def fit_gmm_bic(
    samples: pp.LieTensor,
    max_components: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a GMM to a batch of SE3 pose samples, adaptively selecting the number of components.

    This method uses the Bayesian Information Criterion (BIC) to find the optimal
    number of components up to a predefined maximum. The output is a fixed-size
    GMM, padded with zero-weight components if necessary.

    Args:
        samples (pp.SE3): A pypose.SE3 tensor of shape (N,) representing the poses of the particles.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - mus (torch.Tensor): The means of the GMM components, shape (K, 7).
        - sigmas_diag (torch.Tensor): The diagonal variances, shape (K, 6).
        - weights (torch.Tensor): The component weights, shape (K,).
            K is self.kf_gmm_n_components.
    """
    device = samples.device

    # --- 1. Pre-processing and Data Conversion ---

    # Convert SE3 samples to se3 tangent space vectors for GMM fitting
    # The mean of the distribution should be near the identity for the tangent space
    # representation to be most effective. We find the geometric mean and
    # express all samples relative to it.
    
    # Note: pp.mean is iterative and can be slow. For speed, we can approximate
    # by taking the mean in the Lie algebra.
    sample_vecs_cpu = samples.Log().tensor().cpu().numpy()
    

    # --- 2. Adaptive Component Selection using BIC ---

    n_components_range = range(1, max_components + 1)
    bics = []
    models = []

    for n_comps in n_components_range:
        gmm = GaussianMixture(
            n_components=n_comps,
            covariance_type='diag', # Use diagonal covariance for efficiency
            random_state=0,
            n_init=3 # Run a few initializations to find a better fit
        )
        gmm.fit(sample_vecs_cpu)
        models.append(gmm)
        bics.append(gmm.bic(sample_vecs_cpu))

    # Select the model with the lowest BIC score
    best_gmm_model = models[np.argmin(bics)]
    
    # --- 3. Convert Best Model to Our Format and Pad/Truncate ---

    # The number of "active" components found by BIC
    n_active_components = best_gmm_model.n_components
    
    # Get parameters from the best model
    # The means from sklearn are in the Lie algebra vector space
    means_vec = torch.from_numpy(best_gmm_model.means_).to(device, dtype=torch.float32)
    
    # Covariances are already diagonal. Variances are on the diagonal.
    sigmas_diag = torch.from_numpy(best_gmm_model.covariances_).to(device, dtype=torch.float32)
    
    weights = torch.from_numpy(best_gmm_model.weights_).to(device, dtype=torch.float32)

    # Convert means from se3 vectors back to SE3 group elements
    active_mus = pp.se3(means_vec).Exp()

    # Initialize fixed-size output tensors
    final_mus_tensor = pp.identity_SE3(max_components, device=device)
    final_sigmas_diag = pp.identity_se3(max_components, device=device)
    final_weights = torch.zeros(max_components, device=device)
    
    # Fill the tensors with the active components
    final_mus_tensor[:n_active_components] = active_mus
    final_sigmas_diag[:n_active_components] = sigmas_diag
    final_weights[:n_active_components] = weights
    
    # The remaining entries are already zero, effectively padding the GMM.
    # This fixed-size output is ideal for batched GPU operations.

    return final_mus_tensor, final_sigmas_diag, final_weights



def torch_multivariate_normal_pdf(x: torch.Tensor, std_diag: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    MVN(0, Σ) PDF with diagonal Σ. `std_diag` are the *standard deviations*.
    x: (N, D), std_diag: (D,)
    returns: (N,)
    """
    device, dtype = x.device, x.dtype
    N, D = x.shape
    if std_diag.shape != (D,):
        raise ValueError("std_diag must have shape (D,) matching x.shape[1].")

    std = std_diag.to(device=device, dtype=dtype).clamp_min(eps)          # σ_i
    var = std * std                                                       # σ_i^2
    log_det = torch.sum(torch.log(var))                                   # log det Σ
    log_norm = -0.5 * (D * torch.log(torch.tensor(2.0 * torch.pi, device=device, dtype=dtype)) + log_det)
    quad = torch.sum((x**2) / var, dim=1)                                 # x^T Σ^{-1} x
    return torch.exp(log_norm - 0.5 * quad)
