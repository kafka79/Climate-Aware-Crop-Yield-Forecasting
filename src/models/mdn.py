import warnings
import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, List, Optional, Tuple

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
        
        # Add epsilon for numerical stability
        sigma = sigma + self.epsilon
        
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


def mdn_detect_bimodality(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    separation_threshold: float = 1.5,
    weight_threshold: float = 0.20,
) -> Dict[str, object]:
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

    Returns a dict with:
        is_bimodal:    bool
        modes:         list of (weight, mean_t/ha) for all significant modes
        dominant_mode: float — the mean of the highest-weight component
        valley_depth:  float — how deep the probability valley is between the two
                       top modes (0.0 if unimodal, 1.0 = complete separation)
    """
    # Work with first batch item (inference is always batch=1 in production)
    pi_b = pi[0]          # (K,)
    sigma_b = sigma[0]    # (K, O)
    mu_b = mu[0]          # (K, O)

    # Collect significant modes (components with enough weight)
    significant: List[Tuple[float, float]] = []
    for k in range(pi_b.shape[0]):
        w = float(pi_b[k].item())
        m = float(mu_b[k, 0].item())
        if w >= weight_threshold:
            significant.append((w, m))

    significant.sort(key=lambda x: x[0], reverse=True)  # heaviest first

    is_bimodal = False
    valley_depth = 0.0
    dominant_mode = float(mdn_expected_value(pi, sigma, mu)[0, 0].item())

    if len(significant) >= 2:
        top_w, top_m = significant[0]
        sec_w, sec_m = significant[1]

        # Pooled sigma of the two dominant components (first output dim)
        pooled_sigma = float(
            (pi_b * sigma_b[:, 0]).sum().item() + 1e-8
        )
        separation = abs(top_m - sec_m) / pooled_sigma

        if separation >= separation_threshold:
            is_bimodal = True
            # Valley depth: how evenly the mass is split between the two modes
            # 0 = one mode dominates (shallow valley), 1 = perfectly split
            valley_depth = float(1.0 - abs(top_w - sec_w) / (top_w + sec_w + 1e-8))
            dominant_mode = top_m  # heaviest mode, not the valley-mean

    return {
        "is_bimodal": is_bimodal,
        "modes": significant,
        "dominant_mode": dominant_mode,
        "valley_depth": valley_depth,
    }


def mdn_safe_point_estimate(
    pi: torch.Tensor,
    sigma: torch.Tensor,
    mu: torch.Tensor,
    separation_threshold: float = 1.5,
    weight_threshold: float = 0.20,
) -> Tuple[float, Dict[str, object]]:
    """Return a reliable point estimate, refusing to use the valley-mean when bimodal.

    For unimodal distributions: returns the standard weighted mean.
    For bimodal distributions: returns the dominant (highest-weight) mode mean
    and emits a BimodalDistributionWarning with full diagnostic information.

    Returns:
        (point_estimate, bimodality_report)
    """
    report = mdn_detect_bimodality(pi, sigma, mu, separation_threshold, weight_threshold)

    if report["is_bimodal"]:
        valley_mean = float(mdn_expected_value(pi, sigma, mu)[0, 0].item())
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

    point = float(mdn_expected_value(pi, sigma, mu)[0, 0].item())
    return point, report

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
