import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, List, Any

class YieldRiskClassifier:
    """
    Stratafies predicted crop yields into risk tiers (Low/Medium/High).
    Uses uncertainty measures from the MDN model for calibration.
    """
    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {
            "low": 0.2, # 20% below average
            "high": 0.5 # 50% below average
        }
        
    def classify_risk(self, predicted_yield: float, average_yield: float):
        """
        Classifies risk based on deviation from historical average.
        """
        deviation = (average_yield - predicted_yield) / average_yield
        
        if deviation > self.thresholds["high"]:
            return "High Risk"
        elif deviation > self.thresholds["low"]:
            return "Medium Risk"
        else:
            return "Low Risk"

    def calibrate_with_uncertainty(self, mean: float, std: float, average_yield: float):
        """
        Adjusts risk tier based on prediction confidence (std).
        If uncertainty is high, promote to a higher risk class for safety.
        """
        logger.info(f"Calibrating risk with uncertainty (Mean={mean}, Std={std})...")
        
        # Base classification on mean
        base_risk = self.classify_risk(mean, average_yield)
        
        # If the standard deviation is larger than 15% of the mean, elevate risk
        if std / (mean + 1e-10) > 0.15:
            if base_risk == "Low Risk":
                return "Medium Risk (Elevated due to uncertainty)"
            elif base_risk == "Medium Risk":
                return "High Risk (Elevated due to uncertainty)"
            
        return base_risk

def generate_risk_report(predictions: np.ndarray, uncertainties: np.ndarray, historical_avg: float, configs: dict):
    """
    Generate a full risk stratification report for a given region.
    """
    classifier = YieldRiskClassifier(configs.get("thresholds"))
    results = []
    
    for mean, std in zip(predictions, uncertainties):
        risk = classifier.calibrate_with_uncertainty(mean, std, historical_avg)
        results.append({
            "predicted_mean": float(mean),
            "uncertainty_std": float(std),
            "calibrated_risk": risk
        })
        
    logger.success("Risk stratification report generated.")
    return pd.DataFrame(results)
