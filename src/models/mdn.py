import torch
import torch.nn as nn
from loguru import logger
from typing import Tuple

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
