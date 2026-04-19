import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Exponential() # Variance must be positive
        )
        self.mu = nn.Linear(input_dim, num_mixtures * output_dim)

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
        
        return pi, sigma, mu

def mdn_loss(pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, target: torch.Tensor):
    """
    Negative Log Likelihood (NLL) Loss for MDN.
    target: (B, O)
    """
    # target reshaped to (B, 1, O) to broadcast with (B, K, O)
    target = target.unsqueeze(1).expand_as(mu)
    
    # Calculate GMM probability
    # m = (1 / sqrt(2*pi*sigma^2)) * exp(-(target-mu)^2 / (2*sigma^2))
    # We use Log Probability for numerical stability
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    log_prob = m.log_prob(target) # (B, K, O)
    
    # Sum over output dimension (usually 1 for crop yield)
    log_prob = torch.sum(log_prob, dim=2) # (B, K)
    
    # Weight by mixing coefficients (pi)
    # loss = -log(sum(pi * exp(log_prob)))
    # Use LogSumExp for stability
    loss = torch.logsumexp(torch.log(pi + 1e-10) + log_prob, dim=1) # (B,)
    
    return -torch.mean(loss)

def initialize_mdn_head(input_dim: int, num_mixtures: int = 5):
    """
    Initialize MDN head for the model.
    """
    logger.info(f"Initializing MDN Head with {num_mixtures} mixtures...")
    return MixtureDensityNetwork(input_dim, num_mixtures)
