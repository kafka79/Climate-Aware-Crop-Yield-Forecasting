import torch
from loguru import logger
from typing import Dict, Any, List
from captum.attr import IntegratedGradients

class YieldExplainer:
    """
    Explainability module using Integrated Gradients (Captum) for multi-modal transformers.
    Decomposes the prediction into contributions from Satellite, Weather, and Soil inputs.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.ig = IntegratedGradients(self.model)
        
    def calculate_attributions(self, sat: torch.Tensor, weather: torch.Tensor, 
                               soil: torch.Tensor, target_idx: int = 0, steps: int = 50):
        """
        Calculate Integrated Gradients attribution for each modality.
        
        Args:
            sat: (B, T, C) - Spectral Features
            weather: (B, T, F_w) - Temporal Features
            soil: (B, F_s) - Static Features
            target_idx: Index of the output to explain (0 for regression)
            steps: Approximation steps for the integral
            
        Returns:
            Dict[str, torch.Tensor]: Attributions per modality.
        """
        logger.info(f"Calculating multi-modal attributions (Steps={steps})...")
        
        # Baselines (usually zeros)
        sat_base = torch.zeros_like(sat)
        weather_base = torch.zeros_like(weather)
        soil_base = torch.zeros_like(soil)
        
        # Calculate attributions
        attributions = self.ig.attribute(
            inputs=(sat, weather, soil),
            baselines=(sat_base, weather_base, soil_base),
            target=target_idx,
            n_steps=steps
        )
        
        attr_dict = {
            "sat": attributions[0],
            "weather": attributions[1],
            "soil": attributions[2]
        }
        
        return attr_dict

    def summarize_importance(self, attr_dict: Dict[str, torch.Tensor]):
        """
        Aggregates attributions across time and channels to get global importance scores.
        """
        logger.info("Summarizing feature importance scores...")
        
        importance = {
            "satellite_overall": float(attr_dict["sat"].abs().mean()),
            "weather_overall": float(attr_dict["weather"].abs().mean()),
            "soil_overall": float(attr_dict["soil"].abs().mean()),
            # Temporal importance (how much each time step contributed)
            "temporal_importance": attr_dict["sat"].abs().mean(dim=(0, 2)).tolist()
        }
        
        return importance

def explain_prediction(model: torch.nn.Module, sample: dict):
    """
    Standard entry point for explaining a single prediction.
    """
    explainer = YieldExplainer(model)
    
    # Ensure tensors have batch dimension
    sat = sample["sat"].unsqueeze(0) if sample["sat"].dim() == 2 else sample["sat"]
    weather = sample["weather"].unsqueeze(0) if sample["weather"].dim() == 2 else sample["weather"]
    soil = sample["soil"].unsqueeze(0) if sample["soil"].dim() == 1 else sample["soil"]
    
    attr_dict = explainer.calculate_attributions(sat, weather, soil)
    summary = explainer.summarize_importance(attr_dict)
    
    logger.success("XAI Attribution Report generated successfully.")
    return summary, attr_dict
