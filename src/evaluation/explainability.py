import torch
from captum.attr import IntegratedGradients
from loguru import logger
import numpy as np

class ExplainabilityManager:
    """
    Handles feature attribution using Captum (Integrated Gradients).
    Identifies which time-steps or bands drove the model's yield prediction.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.ig = IntegratedGradients(self.model)

    def attribute(self, sat_input, weather_input, soil_input, target_class=0):
        """
        Computes attributions for the three input modalities.
        sat_input: (1, T, C, H, W) or (1, T, C)
        weather_input: (1, T, F_w)
        soil_input: (1, F_s)
        """
        logger.info("Computing Integrated Gradients attributions...")
        
        # Ensure tensors require gradients
        sat_input.requires_grad_()
        weather_input.requires_grad_()
        soil_input.requires_grad_()

        # Compute attributions
        # target=target_class if model has multiple outputs, else None
        attributions = self.ig.attribute(
            inputs=(sat_input, weather_input, soil_input),
            target=target_class,
            internal_batch_size=1
        )
        
        sat_attr, weather_attr, soil_attr = attributions
        
        return {
            "sat": sat_attr.detach().cpu().numpy(),
            "weather": weather_attr.detach().cpu().numpy(),
            "soil": soil_attr.detach().cpu().numpy()
        }

    def summarize_attributions(self, attr_dict, feature_names: dict):
        """
        Summarizes attribution scores into human-readable importance.
        """
        summary = {}
        
        # Summarize weather importance (sum across time)
        weather_sum = np.abs(attr_dict["weather"]).sum(axis=(0, 1))
        for i, val in enumerate(weather_sum):
            name = feature_names.get("weather", {}).get(i, f"Weather_{i}")
            summary[name] = float(val)
            
        # Summarize soil importance
        soil_sum = np.abs(attr_dict["soil"]).sum(axis=0)
        for i, val in enumerate(soil_sum):
            name = feature_names.get("soil", {}).get(i, f"Soil_{i}")
            summary[name] = float(val)
            
        return summary
