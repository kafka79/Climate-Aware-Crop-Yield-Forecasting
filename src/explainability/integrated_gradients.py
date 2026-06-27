import torch
from loguru import logger
from typing import Dict, Optional, List
from captum.attr import IntegratedGradients
from src.models.mdn import mdn_expected_value

class YieldExplainer:
    """
    Explainability module using Integrated Gradients (Captum) for multi-modal transformers.
    Decomposes the prediction into contributions from Satellite, Weather, and Soil inputs.
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.ig = IntegratedGradients(self._forward_for_explanation)

    def _forward_for_explanation(self, sat: torch.Tensor, weather: torch.Tensor, soil: torch.Tensor):
        output = self.model(sat, weather, soil)
        if isinstance(output, tuple):
            pi, sigma, mu = output
            return mdn_expected_value(pi, sigma, mu)
        return output
        
    def calculate_attributions(self, sat: torch.Tensor, weather: torch.Tensor, 
                               soil: torch.Tensor, target_idx: int = 0, steps: int = 150,
                               baselines: Optional[Dict[str, torch.Tensor]] = None):
        """
        Calculate Integrated Gradients attribution for each modality.
        
        Args:
            sat: (B, T, C) - Spectral Features
            weather: (B, T, F_w) - Temporal Features
            soil: (B, F_s) - Static Features
            target_idx: Index of the output to explain (0 for regression)
            steps: Approximation steps for the integral
            baselines: Optional dictionary containing baseline tensors for each modality.
            
        Returns:
            Dict[str, torch.Tensor]: Attributions per modality.
        """
        logger.info(f"Calculating multi-modal attributions (Steps={steps})...")
        try:
            device = next(iter(self.model.parameters())).device
            if hasattr(device, "_mock_return_value") or not isinstance(device, (torch.device, str)):
                device = torch.device("cpu")
        except StopIteration:
            device = torch.device("cpu")

        sat = sat.to(device)
        weather = weather.to(device)
        soil = soil.to(device)

        baselines = baselines or {}
        
        # 1. Satellite Baseline: Temporal average spectral signature
        if "sat" in baselines:
            sat_base = baselines["sat"]
        else:
            sat_base = sat.mean(dim=1, keepdim=True).expand_as(sat)
            
        # 2. Weather Baseline: Temporal average weather variables
        if "weather" in baselines:
            weather_base = baselines["weather"]
        else:
            weather_base = weather.mean(dim=1, keepdim=True).expand_as(weather)
            
        # 3. Soil Baseline: Realistic default or passed baseline (dynamically aligning with model's expected scales)
        if "soil" in baselines:
            soil_base = baselines["soil"]
        else:
            if hasattr(self.model, "soil_mean") and isinstance(self.model.soil_mean, torch.Tensor):
                default_soil = self.model.soil_mean.to(device=soil.device, dtype=soil.dtype)
            else:
                default_soil = torch.tensor([6.5, 10.0, 1.5], dtype=soil.dtype, device=soil.device)
            # Slice/pad to match soil dimensions
            if soil.shape[-1] <= len(default_soil):
                default_soil = default_soil[:soil.shape[-1]]
            else:
                default_soil = torch.cat([default_soil, torch.zeros(soil.shape[-1] - len(default_soil), dtype=soil.dtype, device=soil.device)])
            soil_base = default_soil.unsqueeze(0).expand_as(soil)

        sat_base = sat_base.to(device)
        weather_base = weather_base.to(device)
        soil_base = soil_base.to(device)
            
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
 
def explain_prediction(model: torch.nn.Module, sample: dict, baselines: Optional[dict] = None):
    """
    Standard entry point for explaining a single prediction.
    """
    explainer = YieldExplainer(model)
    
    # Ensure tensors have batch dimension
    sat = sample["sat"].unsqueeze(0) if sample["sat"].dim() == 2 else sample["sat"]
    weather = sample["weather"].unsqueeze(0) if sample["weather"].dim() == 2 else sample["weather"]
    soil = sample["soil"].unsqueeze(0) if sample["soil"].dim() == 1 else sample["soil"]
    
    attr_dict = explainer.calculate_attributions(sat, weather, soil, baselines=baselines)
    summary = explainer.summarize_importance(attr_dict)
    
    logger.success("XAI Attribution Report generated successfully.")
    return summary, attr_dict
