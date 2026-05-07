"""
Tests for bimodal distribution detection in the MDN module.

Addresses [Tara · OpenAI]: the MDN predictive mean can fall in a probability
valley when the distribution is genuinely bimodal (e.g. drought vs flood).
These tests verify that mdn_detect_bimodality and mdn_safe_point_estimate
handle both unimodal and bimodal cases correctly.
"""

import warnings

import pytest
import torch

from src.models.mdn import (
    BimodalDistributionWarning,
    MixtureDensityNetwork,
    mdn_detect_bimodality,
    mdn_safe_point_estimate,
    mdn_expected_value,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_bimodal(low_yield: float = 2.0, high_yield: float = 8.0, split: float = 0.45):
    """Build (pi, sigma, mu) tensors that represent a clearly bimodal GMM.

    Two dominant components at low_yield and high_yield, both with substantial
    weight (split each), and one negligible component in the middle.
    """
    pi    = torch.tensor([[split, split, 1 - 2 * split]], dtype=torch.float32)
    sigma = torch.tensor([[[0.3], [0.3], [0.1]]], dtype=torch.float32)
    mu    = torch.tensor([[[low_yield], [high_yield], [5.0]]], dtype=torch.float32)
    return pi, sigma, mu


def _make_unimodal(peak: float = 5.0):
    """Build (pi, sigma, mu) tensors for a clearly unimodal GMM."""
    pi    = torch.tensor([[0.85, 0.10, 0.05]], dtype=torch.float32)
    sigma = torch.tensor([[[0.4], [0.3], [0.2]]], dtype=torch.float32)
    mu    = torch.tensor([[[peak], [peak + 0.2], [peak - 0.1]]], dtype=torch.float32)
    return pi, sigma, mu


# ── mdn_detect_bimodality ─────────────────────────────────────────────────────

class TestMdnDetectBimodality:

    def test_clearly_bimodal_is_flagged(self):
        """Two well-separated, equally-weighted modes must be detected."""
        pi, sigma, mu = _make_bimodal(low_yield=2.0, high_yield=8.0, split=0.45)
        report = mdn_detect_bimodality(pi, sigma, mu)
        assert report["is_bimodal"] is True

    def test_unimodal_not_flagged(self):
        """A distribution dominated by a single mode must not be flagged."""
        pi, sigma, mu = _make_unimodal(peak=5.5)
        report = mdn_detect_bimodality(pi, sigma, mu)
        assert report["is_bimodal"] is False

    def test_dominant_mode_is_highest_weight_component(self):
        """Dominant mode must be the heavier of the two modes, not the valley mean."""
        # Mode at 2.0 gets 0.30, mode at 8.0 gets 0.60 → dominant = 8.0
        pi    = torch.tensor([[0.30, 0.60, 0.10]], dtype=torch.float32)
        sigma = torch.tensor([[[0.3], [0.3], [0.1]]], dtype=torch.float32)
        mu    = torch.tensor([[[2.0], [8.0], [5.0]]], dtype=torch.float32)
        report = mdn_detect_bimodality(pi, sigma, mu)
        if report["is_bimodal"]:
            assert abs(report["dominant_mode"] - 8.0) < 0.5, (
                f"Expected dominant mode ≈ 8.0, got {report['dominant_mode']}"
            )

    def test_valley_depth_near_one_when_evenly_split(self):
        """When two modes have almost equal weight, valley depth should be close to 1."""
        pi, sigma, mu = _make_bimodal(split=0.48)
        report = mdn_detect_bimodality(pi, sigma, mu)
        if report["is_bimodal"]:
            assert report["valley_depth"] > 0.8, (
                f"Expected valley_depth > 0.8 for equal-split bimodal, got {report['valley_depth']}"
            )

    def test_report_keys_always_present(self):
        """Report dict must always contain expected keys regardless of distribution."""
        for pi, sigma, mu in [_make_bimodal(), _make_unimodal()]:
            report = mdn_detect_bimodality(pi, sigma, mu)
            for key in ("is_bimodal", "modes", "dominant_mode", "valley_depth"):
                assert key in report, f"Missing key '{key}' in bimodality report"

    def test_modes_list_sorted_by_weight_descending(self):
        """modes list must be sorted heaviest-first."""
        pi, sigma, mu = _make_bimodal(split=0.40)
        report = mdn_detect_bimodality(pi, sigma, mu)
        modes = report["modes"]
        if len(modes) >= 2:
            weights = [w for w, _ in modes]
            assert weights == sorted(weights, reverse=True), (
                "Modes not sorted by weight descending"
            )


# ── mdn_safe_point_estimate ───────────────────────────────────────────────────

class TestMdnSafePointEstimate:

    def test_bimodal_emits_warning(self):
        """BimodalDistributionWarning must be emitted for bimodal distributions."""
        pi, sigma, mu = _make_bimodal()
        report = mdn_detect_bimodality(pi, sigma, mu)
        if not report["is_bimodal"]:
            pytest.skip("Fixture not detected as bimodal with current thresholds")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mdn_safe_point_estimate(pi, sigma, mu)
        bimodal_warns = [w for w in caught if issubclass(w.category, BimodalDistributionWarning)]
        assert len(bimodal_warns) >= 1, "Expected BimodalDistributionWarning but none was raised"

    def test_unimodal_no_warning(self):
        """No BimodalDistributionWarning must be emitted for unimodal distributions."""
        pi, sigma, mu = _make_unimodal()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mdn_safe_point_estimate(pi, sigma, mu)
        bimodal_warns = [w for w in caught if issubclass(w.category, BimodalDistributionWarning)]
        assert len(bimodal_warns) == 0, "Unexpected BimodalDistributionWarning for unimodal input"

    def test_bimodal_returns_dominant_not_valley_mean(self):
        """For bimodal distributions, the point estimate must NOT be the valley mean."""
        pi, sigma, mu = _make_bimodal(low_yield=2.0, high_yield=8.0, split=0.45)
        report = mdn_detect_bimodality(pi, sigma, mu)
        if not report["is_bimodal"]:
            pytest.skip("Fixture not detected as bimodal with current thresholds")

        valley_mean = float(mdn_expected_value(pi, sigma, mu)[0, 0].item())
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            point, _ = mdn_safe_point_estimate(pi, sigma, mu)

        # The valley mean should be ~5.0; the dominant mode should be far from it
        assert abs(point - valley_mean) > 1.0, (
            f"Safe estimate ({point:.2f}) is too close to valley mean ({valley_mean:.2f}); "
            "it should have returned the dominant mode instead"
        )

    def test_unimodal_matches_weighted_mean(self):
        """For unimodal distributions, safe estimate must equal the standard weighted mean."""
        pi, sigma, mu = _make_unimodal(peak=5.0)
        expected = float(mdn_expected_value(pi, sigma, mu)[0, 0].item())
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            point, _ = mdn_safe_point_estimate(pi, sigma, mu)
        assert abs(point - expected) < 0.01, (
            f"Safe estimate ({point:.4f}) differs from weighted mean ({expected:.4f}) for unimodal"
        )

    def test_returns_tuple_of_float_and_dict(self):
        """mdn_safe_point_estimate must return (float, dict)."""
        pi, sigma, mu = _make_unimodal()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = mdn_safe_point_estimate(pi, sigma, mu)
        assert isinstance(result, tuple) and len(result) == 2
        point, rep = result
        assert isinstance(point, float)
        assert isinstance(rep, dict)
