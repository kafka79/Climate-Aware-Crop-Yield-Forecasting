import numpy as np
import scipy.stats as stats
from loguru import logger
from typing import Dict, Any

class ProbabilisticMetrics:
    """
    Advanced metrics for quantifying uncertainty in crop yield forecasts.
    Mainly for Mixture Density Networks (MDN) or Bayesian models.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config

    def calculate_crps_gmm(self, y_true: np.ndarray, pi: np.ndarray, sigma: np.ndarray, mu: np.ndarray):
        """
        Calculate Continuous Ranked Probability Score (CRPS) for a Gaussian Mixture.
        CRPS measures both accuracy and calibration.
        """
        logger.info("Calculating CRPS for Gaussian Mixture...")
        # Approximation of CRPS for GMM using sampling or simplified metrics
        # For simplicity in this demo, we'll return a weighted absolute error
        # A true analytic CRPS for GMM involves pairwise integrals
        weighted_mu = np.sum(pi * mu, axis=1)
        crps_approx = np.mean(np.abs(y_true - weighted_mu))
        return {"CRPS_approx": float(crps_approx)}

    def calculate_pit(self, y_true: np.ndarray, pi: np.ndarray, sigma: np.ndarray, mu: np.ndarray):
        """
        Probability Integral Transform (PIT).
        If the model is perfectly calibrated, the PIT values should be Uniform[0, 1].
        """
        logger.info("Calculating PIT values for calibration checking...")
        pit_vals = np.zeros_like(y_true, dtype=float)
        
        for i in range(len(y_true)):
            # Sum of CDFs weighted by pi for the mixture component
            cdf_val = np.sum(pi[i] * stats.norm.cdf(y_true[i], loc=mu[i], scale=sigma[i]))
            pit_vals[i] = cdf_val
            
        return pit_vals

    def evaluate_calibration(self, pit_values: np.ndarray):
        """
        Check if PIT values follow a uniform distribution using Kolmogorov-Smirnov test.
        """
        ks_stat, p_val = stats.kstest(pit_values, 'uniform')
        logger.info(f"Calibration KS-Test: Stat={ks_stat:.4f}, P-Val={p_val:.4f}")
        return ks_stat, p_val

def get_prediction_intervals(pi: np.ndarray, sigma: np.ndarray, mu: np.ndarray, confidence: float = 0.95):
    """
    Generate 95% confidence intervals from GMM parameters using normal distribution percentiles.
    """
    logger.info(f"Generating {confidence*100}% prediction intervals...")
    # Simplified approach: take the lowest and highest component standard errors
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1 - alpha/2)
    
    # Weighted bounds across mixtures for a rough estimate
    lower = np.sum(pi * (mu - z * sigma), axis=1)
    upper = np.sum(pi * (mu + z * sigma), axis=1)
    return lower, upper
