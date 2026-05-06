import pytest
import torch
import numpy as np
from src.models.mdn import (
    MixtureDensityNetwork,
    mdn_expected_value,
    mdn_loss,
    mdn_predictive_std,
)

@pytest.fixture
def mdn_model():
    return MixtureDensityNetwork(input_dim=64, num_mixtures=5, output_dim=1)

def test_mdn_pi_sums_to_one(mdn_model):
    """Mixing coefficients must always sum to 1 (softmax guarantee)."""
    x = torch.randn(16, 64)
    pi, sigma, mu = mdn_model(x)
    pi_sums = pi.sum(dim=1)
    assert torch.allclose(pi_sums, torch.ones(16), atol=1e-5), \
        f"pi should sum to 1, got {pi_sums}"

def test_mdn_sigma_positive(mdn_model):
    """Sigma (std dev) must always be strictly positive (Exponential guarantee)."""
    x = torch.randn(16, 64)
    pi, sigma, mu = mdn_model(x)
    assert (sigma > 0).all(), "All sigma values must be positive"

def test_mdn_output_shapes(mdn_model):
    """Verify output tensor shapes match expected (B, K, O)."""
    B, K, O = 8, 5, 1
    x = torch.randn(B, 64)
    pi, sigma, mu = mdn_model(x)
    assert pi.shape == (B, K), f"pi shape mismatch: {pi.shape}"
    assert sigma.shape == (B, K, O), f"sigma shape mismatch: {sigma.shape}"
    assert mu.shape == (B, K, O), f"mu shape mismatch: {mu.shape}"

def test_mdn_loss_is_scalar(mdn_model):
    """MDN NLL loss must return a scalar."""
    x = torch.randn(8, 64)
    pi, sigma, mu = mdn_model(x)
    target = torch.randn(8, 1)
    loss = mdn_loss(pi, sigma, mu, target)
    assert loss.shape == (), f"Loss must be a scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss must not be NaN"

def test_mdn_expected_value_matches_manual_weighting():
    pi = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    sigma = torch.tensor([[[0.2], [0.4]]], dtype=torch.float32)
    mu = torch.tensor([[[2.0], [4.0]]], dtype=torch.float32)

    expected = mdn_expected_value(pi, sigma, mu)
    assert torch.allclose(expected, torch.tensor([[3.5]]))

def test_mdn_predictive_std_is_positive():
    pi = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    sigma = torch.tensor([[[0.2], [0.3]]], dtype=torch.float32)
    mu = torch.tensor([[[1.0], [1.4]]], dtype=torch.float32)

    std = mdn_predictive_std(pi, sigma, mu)
    assert std.shape == (1, 1)
    assert torch.all(std > 0)
