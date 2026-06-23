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
    """
    def __init__(self, input_dim: int, num_mixtures: int = 5, output_dim: int = 1):
        super(MixtureDensityNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim
        
        # MDN Head
        self.pi = nn.Sequential(
            nn.Linear(input_dim, num_mixtures),
            nn.Softmax(dim=1) # Mixing coefficients must sum to 1
        )
        self.sigma = nn.Sequential(
            nn.Linear(input_dim, num_mixtures * output_dim),
            nn.Softplus() # Softplus guarantees sigma > 0
        )
        self.mu = nn.Linear(input_dim, num_mixtures * output_dim)
        self.epsilon = 1e-6 # Minimum variance for stability
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, D) - Hidden representation from Transformer
        returns: (pi, sigma, mu)
        """
        pi = self.pi(x)
        sigma = self.sigma(x)
        mu = self.mu(x)
        
        # Reshape sigma and mu to (B, K, O)
        sigma = sigma.view(-1, self.num_mixtures, self.output_dim)
        mu = mu.view(-1, self.num_mixtures, self.output_dim)
        
        # Enforce strict lower bound to prevent variance collapse and NaN NLL losses.
        # Softplus alone allows sigma to approach 0 asymptotically, which causes
        # the -log(sigma) term in the NLL loss to explode.
        sigma = torch.clamp(sigma, min=1e-3)
        
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
    if pi.dim() == 1:
        return mdn_detect_bimodality_single(pi, sigma, mu, separation_threshold, weight_threshold)
    
    if pi.size(0) == 1:
        return mdn_detect_bimodality_single(pi[0], sigma[0], mu[0], separation_threshold, weight_threshold)
        
    return [
        mdn_detect_bimodality_single(pi[i], sigma[i], mu[i], separation_threshold, weight_threshold)
        for i in range(pi.size(0))
    ]


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

def mdn_loss(pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, target: torch.Tensor, entropy_weight: float = 0.01):
    """
    Negative Log Likelihood (NLL) Loss for MDN with Entropy Regularization.
    target: (B, O)
    """
    # target reshaped to (B, 1, O) to broadcast with (B, K, O)
    if target.dim() == 1:
        target = target.unsqueeze(-1) # (B, 1)
    target = target.unsqueeze(1).expand_as(mu) # (B, K, O)
    
    # Calculate GMM probability
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(target) # (B, K, O)
    
    # Sum over output dimension
    log_prob = torch.sum(log_prob, dim=2) # (B, K)
    
    # Weight by mixing coefficients (pi)
    # Use LogSumExp for stability
    nll = -torch.logsumexp(torch.log(pi + 1e-10) + log_prob, dim=1) # (B,)
    
    # Entropy Regularization to prevent mode collapse
    # entropy = -sum(pi * log(pi))
    entropy_penalty = torch.sum(pi * torch.log(pi + 1e-10), dim=1) 
    
    loss = nll + entropy_weight * entropy_penalty
    
    return torch.mean(loss)

def initialize_mdn_head(input_dim: int, num_mixtures: int = 5):
    """
    Initialize MDN head for the model.
    """
    logger.info(f"Initializing MDN Head with {num_mixtures} mixtures...")
    return MixtureDensityNetwork(input_dim, num_mixtures)
