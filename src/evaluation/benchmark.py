import torch
import numpy as np
import json
import os
from loguru import logger
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.mdn import mdn_expected_value, mdn_predictive_std

class YieldBenchmarker:
    """
    Computes performance metrics for the yield forecasting model.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def evaluate(self, model, data_loader, device: str = "cpu"):
        """
        Runs evaluation on a test dataloader and returns metrics.
        """
        model.eval()
        all_preds = []
        all_labels = []
        all_stds = []

        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                pi, sigma, mu = model(X)
                
                pred = mdn_expected_value(pi, sigma, mu).squeeze(-1)
                std = mdn_predictive_std(pi, sigma, mu).squeeze(-1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_stds.extend(std.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_stds = np.array(all_stds)

        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        mae = mean_absolute_error(all_labels, all_preds)
        
        # Calibration: Fraction of labels within 1.96 * std (95% CI)
        within_ci = np.abs(all_labels - all_preds) <= (1.96 * all_stds)
        calibration_95 = np.mean(within_ci)

        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "calibration_95": float(calibration_95),
            "num_samples": len(all_labels)
        }

        logger.success(f"Benchmark complete: RMSE={rmse:.4f}, Calibration(95%)={calibration_95:.2f}")
        return metrics

    def save_report(self, metrics: Dict[str, Any], path: str = "experiments/benchmark_report.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Report saved to {path}")
