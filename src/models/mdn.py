import math
import warnings
import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, List, Optional, Tuple, Any

class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network (MDN) for probabilistic yield forecasting.
    Outputs the parameters of a Gaussian Mixture Model (GMM).

    Sigma parameterization uses log-scale: the network outputs raw log_sigma
    values, which are converted via sigma = softplus(log_sigma) + epsilon.
    This is mathematically principled — softplus alone allows sigma to approach 0
    asymptotically (causing NLL explosion), while torch.clamp creates non-differentiable
    boundaries that distort gradient flow.  The log-scale approach provides smooth,
    differentiable bounds that the optimizer can traverse cleanly.
    """
    def __init__(self, input_dim: int, num_mixtures: int = 5, output_dim: int = 1):
        super(MixtureDensityNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        
        # Minimum sigma floor: prevents variance collapse while staying
        # small enough not to affect well-behaved distributions
        self.sigma_min = 1e-4

        # MDN Head
        self.pi = nn.Sequential(
            nn.Linear(input_dim, num_mixtures),
            nn.Softmax(dim=1) # Mixing coefficients must sum to 1
        )
        # Log-sigma parameterization: network outputs unconstrained values,
        # converted to positive sigma via softplus + floor in forward()
        self.log_sigma = nn.Linear(input_dim, num_mixtures * output_dim)
        self.mu = nn.Linear(input_dim, num_mixtures * output_dim)
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, D) - Hidden representation from Transformer
        returns: (pi, sigma, mu)
        """
        # Access the underlying Linear layer in self.pi Sequential block to get raw logits
        # this preserves key compatibility with existing checkpoints
        pi_logits = self.pi[0](x)
        log_sigma_raw = self.log_sigma(x)
        mu = self.mu(x)
        
        # Reshape to (B, K, O)
        log_sigma_raw = log_sigma_raw.view(-1, self.num_mixtures, self.output_dim)
        mu = mu.view(-1, self.num_mixtures, self.output_dim)
        
        # Log-scale sigma: softplus provides smooth differentiable lower bound,
        # sigma_min provides an absolute floor without creating gradient discontinuities
        sigma = nn.functional.softplus(log_sigma_raw) + self.sigma_min
        
        # Stably compute pi and log_pi
        pi = nn.functional.softmax(pi_logits, dim=1)
        pi.log_pi = nn.functional.log_softmax(pi_logits, dim=1)
        
        return pi, sigma, mu


def mdn_expected_value(
    pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor
) -> torch.Tensor:
    """
    Return the predictive mean of the Gaussian mixture.

    Output shape: (B, O)
    """
    del sigma
    return torch.sum(pi.unsqueeze(-1) * mu, dim=1)


def mdn_predictive_std(
    pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor
) -> torch.Tensor:
    """
    Return the predictive standard deviation of the Gaussian mixture.

    Output shape: (B, O)
    """
    mean = mdn_expected_value(pi, sigma, mu)
    second_moment = torch.sum(pi.unsqueeze(-1) * (sigma.pow(2) + mu.pow(2)), dim=1)
    variance = torch.clamp(second_moment - mean.pow(2), min=1e-6)
    return torch.sqrt(variance)


class BimodalDistributionWarning(UserWarning):
    """Raised when the MDN output is bimodal and the weighted mean is unreliable."""


def _find_modes_gradient_ascent(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    lr: float = 0.05,
    steps: int = 100,
    convergence_tol: float = 1e-6,
) -> List[Tuple[float, "torch.Tensor"]]:
    """Find modes of a Gaussian Mixture via gradient ascent on log-density.

    Instead of evaluating the GMM PDF on a fixed 1-D grid (which is O(grid_points^D)
    and fails for multi-output D > 1), this initialises one candidate at each
    component mean and ascends the log-density surface.  Complexity is O(K * steps)
    regardless of output dimensionality D, making it the correct algorithm for
    multi-target forecasting (e.g. yield + water demand + protein content).

    Args:
        pi:    (K,)    mixing coefficients (summing to 1)
        sigma: (K, O)  per-component std devs
        mu:    (K, O)  per-component means
        lr:    Step size for gradient ascent
        steps: Maximum ascent iterations per candidate
        convergence_tol: Early-stop when the position delta < this threshold

    Returns:
        List of (log_density_value, position_tensor) tuples, one per discovered
        unique mode, sorted by density descending.  Duplicate modes (converged
        to within convergence_tol of each other) are merged.
    """
    pi = pi.detach()
    sigma = sigma.detach()
    mu = mu.detach()
    K, O = mu.shape
    candidates = []

    # ponytail: scale lr by min sigma to prevent gradient explosion when sigma ~ 1e-4
    min_sigma = float(sigma.min().clamp(min=1e-6))
    effective_lr = lr * min_sigma

    for k in range(K):
        if float(pi[k]) < 1e-6:
            continue
        # Initialise at component mean
        y = mu[k].clone().detach().requires_grad_(True)

        for _ in range(steps):
            # Compute log-density of GMM at y
            diff = y.unsqueeze(0) - mu  # (K, O)
            mahal = -0.5 * ((diff / sigma) ** 2).sum(dim=1)  # (K,)
            log_norm = -O * 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793)) - sigma.log().sum(dim=1)
            log_components = torch.log(pi + 1e-10) + log_norm + mahal  # (K,)
            log_density = torch.logsumexp(log_components, dim=0)

            log_density.backward()
            with torch.no_grad():
                grad = y.grad
                if grad is None:
                    break
                step = effective_lr * grad
                # Clamp step to prevent overshooting beyond component basins
                step = step.clamp(-min_sigma * 2, min_sigma * 2)
                y = (y + step).detach().requires_grad_(True)
                if float(step.abs().max()) < convergence_tol:
                    break

        # Evaluate final density
        with torch.no_grad():
            diff = y - mu
            mahal = -0.5 * ((diff / sigma) ** 2).sum(dim=1)
            log_norm = -O * 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793)) - sigma.log().sum(dim=1)
            log_components = torch.log(pi + 1e-10) + log_norm + mahal
            final_log_density = float(torch.logsumexp(log_components, dim=0).item())
        candidates.append((final_log_density, y.detach()))

    # Merge duplicates: modes within convergence_tol of each other
    unique_modes: List[Tuple[float, torch.Tensor]] = []
    for log_d, pos in sorted(candidates, key=lambda x: x[0], reverse=True):
        is_duplicate = False
        for _, existing_pos in unique_modes:
            if float((pos - existing_pos).abs().max()) < convergence_tol * 10:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_modes.append((log_d, pos))

    return unique_modes


def mdn_detect_bimodality_single(
    pi_b: torch.Tensor,
    sigma_b: torch.Tensor,
    mu_b: torch.Tensor,
    separation_threshold: float = 1.5,
    weight_threshold: float = 0.20,
) -> Dict[str, object]:
    """Detect whether a single mixture is bimodal."""
    import numpy as np

    weights = pi_b.detach().cpu().numpy()
    sigmas = sigma_b[:, 0].detach().cpu().numpy()
    means = mu_b[:, 0].detach().cpu().numpy()

    # Determine dynamic range: min(mu - 3*sigma) to max(mu + 3*sigma)
    y_min = float(np.min(means - 3.0 * sigmas))
    y_max = float(np.max(means + 3.0 * sigmas))
    
    # Crop yield cannot be negative
    y_min = max(0.0, y_min)
    y_max = max(0.1, y_max)

    grid = np.linspace(y_min, y_max, 200)
    pdf = np.zeros_like(grid)
    for w, s, m in zip(weights, sigmas, means):
        pdf += w * (1.0 / (s * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((grid - m) / s) ** 2)

    # Find local maxima (peaks) of the continuous PDF
    peaks: List[Tuple[float, float]] = []
    for i in range(1, len(grid) - 1):
        if pdf[i] > pdf[i - 1] and pdf[i] > pdf[i + 1]:
            peaks.append((grid[i], pdf[i]))

    # Sort peaks by density value descending
    peaks.sort(key=lambda x: x[1], reverse=True)

    is_bimodal = False
    valley_depth = 0.0
    expected_val = float((pi_b.unsqueeze(-1) * mu_b).sum().item())
    dominant_mode = expected_val

    # Calculate standard deviation of the mixture
    second_moment = float((pi_b.unsqueeze(-1) * (sigma_b.pow(2) + mu_b.pow(2))).sum().item())
    mix_var = second_moment - expected_val**2
    mix_std = math.sqrt(max(mix_var, 1e-6))

    # Find unique significant modes by mapping peaks back to closest components
    significant: List[Tuple[float, float]] = []
    seen_indices = set()
    for peak_y, peak_pdf in peaks:
        closest_idx = int(np.argmin(np.abs(means - peak_y)))
        if closest_idx not in seen_indices:
            seen_indices.add(closest_idx)
            w = float(weights[closest_idx])
            if w >= weight_threshold:
                significant.append((w, peak_y))

    # Sort the mode list by weight descending to satisfy tests
    significant.sort(key=lambda x: x[0], reverse=True)

    if len(peaks) >= 2:
        y_top1, p_top1 = peaks[0]
        y_top2, p_top2 = peaks[1]

        # Consider the second peak only if it has a significant relative density
        if p_top2 >= 0.1 * p_top1:
            # Find the valley (minimum density) between the top two peaks
            idx_1 = np.argmin(np.abs(grid - y_top1))
            idx_2 = np.argmin(np.abs(grid - y_top2))
            start_idx, end_idx = min(idx_1, idx_2), max(idx_1, idx_2)
            
            if end_idx > start_idx + 1:
                valley_idx = start_idx + np.argmin(pdf[start_idx:end_idx + 1])
                pdf_valley = pdf[valley_idx]
                
                # Check if there is a real drop of density (valley) between peaks
                # density drop threshold: at least 20%
                if pdf_valley < 0.8 * min(p_top1, p_top2):
                    # Map the peaks to their closest component weights to compute valley_depth split
                    closest_idx1 = int(np.argmin(np.abs(means - y_top1)))
                    closest_idx2 = int(np.argmin(np.abs(means - y_top2)))
                    top_w = float(weights[closest_idx1])
                    sec_w = float(weights[closest_idx2])
                    
                    # Calculate pooled component standard deviation
                    pooled_sigma = math.sqrt((top_w * sigmas[closest_idx1]**2 + sec_w * sigmas[closest_idx2]**2) / (top_w + sec_w + 1e-8))
                    
                    # Cap the standard deviation used for thresholding to prevent quadratic variance inflation
                    # when modes are extremely well-separated.
                    effective_std = min(mix_std, 4.0 * pooled_sigma)
                    
                    # Validate mode separation in units of effective standard deviations
                    distance = abs(y_top1 - y_top2)
                    if distance >= separation_threshold * effective_std:
                        is_bimodal = True
                        valley_depth = float(1.0 - pdf_valley / (min(p_top1, p_top2) + 1e-8))
                        dominant_mode = y_top1

    return {
        "is_bimodal": is_bimodal,
        "modes": significant,
        "dominant_mode": dominant_mode,
        "valley_depth": valley_depth,
    }


def mdn_detect_bimodality(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    separation_threshold: float = 1.5,
    weight_threshold: float = 0.20,
) -> Any:
    """Detect whether the mixture is bimodal and the mean falls in a probability valley.

    A distribution is flagged bimodal when two or more modes are:
      - each carrying >= weight_threshold of total probability mass, AND
      - separated by >= separation_threshold standard deviations of the mixture.

    Args:
        pi:    (B, K)    mixing coefficients
        sigma: (B, K, O) component std-devs
        mu:    (B, K, O) component means
        separation_threshold: minimum inter-mode distance in pooled-sigma units
        weight_threshold:     minimum weight for a component to count as a mode

    Returns:
        If batch size is 1 (or 1D tensors): a dict report.
        If batch size > 1: a list of dict reports.
    """
    import numpy as np

    # Standardize input dimensions to 2D (Batch, Component)
    is_single = False
    if pi.dim() == 1:
        pi = pi.unsqueeze(0)
        sigma = sigma.unsqueeze(0)
        mu = mu.unsqueeze(0)
        is_single = True
    elif pi.size(0) == 1:
        is_single = True

    B, K = pi.shape
    device = pi.device

    # Convert sigma/mu to (B, K) since output_dim O = 1
    sigmas = sigma[..., 0]  # (B, K)
    means = mu[..., 0]      # (B, K)

    # Determine dynamic range for each batch item: min(mu - 3*sigma) to max(mu + 3*sigma)
    y_min = torch.min(means - 3.0 * sigmas, dim=1).values
    y_max = torch.max(means + 3.0 * sigmas, dim=1).values
    y_min = torch.clamp(y_min, min=0.0)
    y_max = torch.clamp(y_max, min=0.1)

    # Construct linear grid of shape (B, 200) on device
    steps = torch.linspace(0.0, 1.0, steps=200, device=device)
    grid = y_min.unsqueeze(1) + steps.unsqueeze(0) * (y_max - y_min).unsqueeze(1)  # (B, 200)

    # Evaluate GMM PDF in a vectorized tensor operation across the entire batch
    grid_uns = grid.unsqueeze(1)       # (B, 1, 200)
    means_uns = means.unsqueeze(2)     # (B, K, 1)
    sigmas_uns = sigmas.unsqueeze(2)   # (B, K, 1)
    pi_uns = pi.unsqueeze(2)           # (B, K, 1)

    exponent = -0.5 * ((grid_uns - means_uns) / sigmas_uns).pow(2)
    coeff = pi_uns / (sigmas_uns * math.sqrt(2.0 * math.pi))
    pdf = torch.sum(coeff * torch.exp(exponent), dim=1)  # (B, 200)

    reports = []
    
    # Detach and move to CPU/numpy just for final scalar metrics extraction
    pi_np = pi.detach().cpu().numpy()
    sigmas_np = sigmas.detach().cpu().numpy()
    means_np = means.detach().cpu().numpy()
    pdf_np = pdf.detach().cpu().numpy()
    grid_np = grid.detach().cpu().numpy()

    for i in range(B):
        pdf_i = pdf_np[i]
        grid_i = grid_np[i]
        
        # Find local maxima (peaks) via gradient ascent
        modes_asc = _find_modes_gradient_ascent(
            pi[i],
            sigma[i],
            mu[i]
        )
        peaks = []
        for log_d, pos in modes_asc:
            peaks.append((float(pos[0].item()), float(math.exp(log_d))))

        # Sort peaks by density value descending
        peaks.sort(key=lambda x: x[1], reverse=True)

        is_bimodal = False
        valley_depth = 0.0
        
        w_t = pi_np[i]
        s_t = sigmas_np[i]
        m_t = means_np[i]

        expected_val = float((pi[i].unsqueeze(-1) * mu[i]).sum().item())
        dominant_mode = expected_val

        # Calculate standard deviation of the mixture
        second_moment = float((pi[i].unsqueeze(-1) * (sigma[i].pow(2) + mu[i].pow(2))).sum().item())
        mix_var = second_moment - expected_val**2
        mix_std = math.sqrt(max(mix_var, 1e-6))

        # Find unique significant modes by mapping peaks back to closest components
        significant = []
        seen_indices = set()
        for peak_y, peak_pdf in peaks:
            closest_idx = int(np.argmin(np.abs(m_t - peak_y)))
            if closest_idx not in seen_indices:
                seen_indices.add(closest_idx)
                w = float(w_t[closest_idx])
                if w >= weight_threshold:
                    significant.append((w, peak_y))

        significant.sort(key=lambda x: x[0], reverse=True)

        if len(peaks) >= 2:
            y_top1, p_top1 = peaks[0]
            y_top2, p_top2 = peaks[1]

            if p_top2 >= 0.1 * p_top1:
                idx_1 = np.argmin(np.abs(grid_i - y_top1))
                idx_2 = np.argmin(np.abs(grid_i - y_top2))
                start_idx, end_idx = min(idx_1, idx_2), max(idx_1, idx_2)

                if end_idx > start_idx + 1:
                    valley_idx = start_idx + np.argmin(pdf_i[start_idx:end_idx + 1])
                    pdf_valley = pdf_i[valley_idx]

                    if pdf_valley < 0.8 * min(p_top1, p_top2):
                        closest_idx1 = int(np.argmin(np.abs(m_t - y_top1)))
                        closest_idx2 = int(np.argmin(np.abs(m_t - y_top2)))
                        top_w = float(w_t[closest_idx1])
                        sec_w = float(w_t[closest_idx2])

                        pooled_sigma = math.sqrt((top_w * s_t[closest_idx1]**2 + sec_w * s_t[closest_idx2]**2) / (top_w + sec_w + 1e-8))
                        effective_std = min(mix_std, 4.0 * pooled_sigma)
                        distance = abs(y_top1 - y_top2)
                        if distance >= separation_threshold * effective_std:
                            is_bimodal = True
                            valley_depth = float(1.0 - pdf_valley / (min(p_top1, p_top2) + 1e-8))
                            dominant_mode = y_top1

        reports.append({
            "is_bimodal": is_bimodal,
            "modes": significant,
            "dominant_mode": dominant_mode,
            "valley_depth": valley_depth,
        })

    return reports[0] if is_single else reports


def mdn_safe_point_estimate(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    separation_threshold: float = 1.5,
    weight_threshold: float = 0.20,
) -> Tuple[Any, Any]:
    """Return a reliable point estimate, refusing to use the valley-mean when bimodal.

    For unimodal distributions: returns the standard weighted mean.
    For bimodal distributions: returns the dominant (highest-weight) mode mean
    and emits a BimodalDistributionWarning with full diagnostic information.

    Returns:
        If batch size is 1: (point_estimate, bimodality_report)
        If batch size > 1: (list_of_point_estimates, list_of_bimodality_reports)
    """
    if pi.dim() == 1 or pi.size(0) == 1:
        report = mdn_detect_bimodality(pi, sigma, mu, separation_threshold, weight_threshold)
        p_tensor = mdn_expected_value(pi if pi.dim() == 2 else pi.unsqueeze(0), 
                                      sigma if sigma.dim() == 3 else sigma.unsqueeze(0), 
                                      mu if mu.dim() == 3 else mu.unsqueeze(0))
        valley_mean = float(p_tensor[0, 0].item())

        if report["is_bimodal"]:
            dominant = report["dominant_mode"]
            mode_list = ", ".join(
                f"{m:.2f} t/ha (weight={w:.0%})" for w, m in report["modes"]
            )
            msg = (
                f"Bimodal yield distribution detected (valley depth={report['valley_depth']:.2f}). "
                f"Weighted mean ({valley_mean:.2f} t/ha) falls between two distinct scenarios. "
                f"Dominant mode: {dominant:.2f} t/ha. All significant modes: [{mode_list}]. "
                "Investigate satellite and weather signals independently for each scenario "
                "before acting on this forecast."
            )
            warnings.warn(msg, BimodalDistributionWarning, stacklevel=2)
            logger.warning(msg)
            return dominant, report

        return valley_mean, report
    else:
        reports = mdn_detect_bimodality(pi, sigma, mu, separation_threshold, weight_threshold)
        points = []
        for i, report in enumerate(reports):
            valley_mean = float(mdn_expected_value(pi[i:i+1], sigma[i:i+1], mu[i:i+1])[0, 0].item())
            if report["is_bimodal"]:
                dominant = report["dominant_mode"]
                mode_list = ", ".join(
                    f"{m:.2f} t/ha (weight={w:.0%})" for w, m in report["modes"]
                )
                msg = (
                    f"Batch index {i}: Bimodal yield distribution detected (valley depth={report['valley_depth']:.2f}). "
                    f"Weighted mean ({valley_mean:.2f} t/ha) falls between two distinct scenarios. "
                    f"Dominant mode: {dominant:.2f} t/ha. All significant modes: [{mode_list}]."
                )
                warnings.warn(msg, BimodalDistributionWarning, stacklevel=2)
                logger.warning(msg)
                points.append(dominant)
            else:
                points.append(valley_mean)
        return points, reports

def mdn_loss(pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, target: torch.Tensor, entropy_weight: float = 0.0):
    """
    Negative Log Likelihood (NLL) Loss for MDN with optional Entropy Regularization.
    target: (B, O)
    """
    # Calculate target standard deviation before expansion to adjust entropy_weight dynamically
    with torch.no_grad():
        if target.dim() <= 1 or target.size(0) <= 1:
            dynamic_scale = torch.tensor(1.0, device=target.device)
        else:
            # target standard deviation along the batch dimension
            target_std = torch.std(target.float(), dim=0)
            # Normalize target standard deviation relative to baseline (e.g. 2.0 t/ha std)
            dynamic_scale = torch.clamp(target_std / 2.0, max=1.0)
            dynamic_scale = torch.mean(dynamic_scale) # average over output dimension O

    # target reshaped to (B, 1, O) to broadcast with (B, K, O)
    if target.dim() == 1:
        target = target.unsqueeze(-1) # (B, 1)
    target_expanded = target.unsqueeze(1).expand_as(mu) # (B, K, O)
    
    # Calculate GMM probability
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(target_expanded) # (B, K, O)
    
    # Sum over output dimension
    log_prob = torch.sum(log_prob, dim=2) # (B, K)
    
    # Weight by mixing coefficients (pi)
    # Use LogSumExp for stability and leverage pre-computed log_pi if available
    log_pi = getattr(pi, "log_pi", None)
    if log_pi is None:
        log_pi = torch.log(pi + 1e-10)
        
    nll = -torch.logsumexp(log_pi + log_prob, dim=1) # (B,)
    
    loss = nll
    if entropy_weight > 0.0:
        # Entropy regularization: H = -sum(pi * log(pi)) is positive.
        # ponytail: old code ADDED sum(pi*log_pi) which is negative, rewarding
        # uniform distributions. Fixed: SUBTRACT it to penalize high entropy,
        # encouraging confident unimodal predictions when warranted.
        entropy = -torch.sum(pi * log_pi, dim=1)  # H >= 0
        loss = loss + (entropy_weight * dynamic_scale) * entropy
    
    return torch.mean(loss)

def select_optimal_mixtures(
    training_targets: torch.Tensor,
    input_dim: int,
    candidate_range: Tuple[int, ...] = (3, 5, 7, 10),
    max_samples: int = 2000,
) -> int:
    """Estimate the optimal number of GMM components using the Bayesian Information Criterion (BIC).

    Fits sklearn GaussianMixture models on a sample of the training yield
    targets and selects the K with the lowest BIC.  This prevents both
    under-fitting (too few modes for multi-cropping / extreme-climate regions)
    and over-fitting (too many modes on small datasets).

    Args:
        training_targets: 1-D tensor of observed yield values from the training set.
        input_dim: MDN input dimension (used only for logging context).
        candidate_range: Tuple of K values to evaluate.
        max_samples: Maximum number of samples to use (for speed).

    Returns:
        The optimal num_mixtures (int).  Falls back to 5 if sklearn is unavailable.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        logger.warning(
            "scikit-learn not installed — cannot auto-select num_mixtures. "
            "Falling back to default K=5."
        )
        return 5

    import numpy as np

    data = training_targets.detach().cpu().numpy().reshape(-1, 1)
    if len(data) > max_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(data), size=max_samples, replace=False)
        data = data[indices]

    best_k = candidate_range[0]
    best_bic = float("inf")
    for k in candidate_range:
        if k > len(data):
            continue
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, max_iter=100)
            gmm.fit(data)
            bic = gmm.bic(data)
            logger.debug(f"BIC for K={k}: {bic:.2f}")
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception as exc:
            logger.warning(f"GaussianMixture fit failed for K={k}: {exc}")
            continue

    logger.info(
        f"Adaptive mixture selection: optimal K={best_k} "
        f"(BIC={best_bic:.2f}, input_dim={input_dim})"
    )
    return best_k


def initialize_mdn_head(input_dim: int, num_mixtures: int = 5):
    """
    Initialize MDN head for the model.
    """
    logger.info(f"Initializing MDN Head with {num_mixtures} mixtures...")
    return MixtureDensityNetwork(input_dim, num_mixtures)
