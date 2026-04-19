import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger
from typing import Dict, Any, List

class YieldMetrics:
    """
    Comprehensive suite for evaluating crop yield prediction performance.
    Supports both point and probabilistic (MDN) evaluation.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def calculate_all(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate a broad range of standard and yield-specific metrics.
        """
        logger.info("Calculating evaluation metrics...")
        
        # Ensure flat arrays
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Standard Regression Metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Yield-Specific Metrics
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        bias = np.mean(y_pred - y_true)
        
        results = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4),
            "MAPE": round(float(mape), 2),
            "MeanBias": round(float(bias), 4)
        }
        
        logger.info(f"Evaluation complete: {results}")
        return results

    def calculate_probabilistic(self, y_true: np.ndarray, pi: np.ndarray, sigma: np.ndarray, mu: np.ndarray):
        """
        Calculate metrics specific to MDN (Probabilistic) outputs.
        Negative Log-Likelihood (NLL) is the primary reliability metric.
        """
        logger.info("Calculating probabilistic reliability metrics (NLL)...")
        # Simplified Gaussian Log-Likelihood for diagnostic purposes
        # In a real run, this would match the MDN loss calculation
        diff = (y_true - mu) ** 2
        var = sigma ** 2 + 1e-6
        log_prob = -0.5 * (diff / var + np.log(2 * np.pi * var))
        # Assuming weighted average over components (pi)
        nll = -np.mean(np.sum(pi * log_prob, axis=1))
        
        return {"NLL": float(nll)}

    def save_results(self, results: dict, base_path: str):
        """
        Save metrics to both JSON (for code) and CSV (for humans).
        """
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        # Save as CSV
        pd.DataFrame([results]).to_csv(f"{base_path}.csv", index=False)
        
        # Save as JSON
        with open(f"{base_path}.json", 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.success(f"Metrics saved to {base_path}.csv|.json")
