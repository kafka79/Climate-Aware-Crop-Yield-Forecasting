from loguru import logger
from typing import Dict, List, Any

class AgronomyAdvisor:
    """
    Generates explainable recommendations based on model feature importance.
    Converts black-box ML predictions into human-readable advice.
    """
    def __init__(self, crop_config: Dict[str, Any]):
        self.crop_config = crop_config
        self.feature_mappings = {
            "accumulated_precip": "Rainfall",
            "gdd": "Temperature/Accumulated Heat",
            "ph": "Soil acidity (pH)",
            "soc": "Soil Organic Carbon (Fertility)"
        }
        self.use_llm = crop_config.get("use_llm", False)
        self.llm_provider = crop_config.get("llm_provider", "mock")

    def _generate_llm_prompt(self, risk: str, features: Dict[str, float]):
        """
        Creates a structured prompt for an LLM to synthesize XAI data into agronomic advice.
        """
        prompt = f"System: You are an expert Agronomist.\n"
        prompt += f"Context: A crop yield model predicts a '{risk}' level for the upcoming harvest.\n"
        prompt += f"Feature Attributions (Integrated Gradients):\n"
        for k, v in features.items():
            prompt += f"- {self.feature_mappings.get(k, k)}: {v:.4f}\n"
        prompt += f"Task: Synthesize this data into 2-3 specific, actionable recommendations for the farmer. "
        prompt += "Focus on the features with positive attributions as they drove the risk."
        return prompt

    def _call_mock_llm(self, prompt: str):
        """
        Simulates an LLM call for environments without API keys.
        """
        logger.debug(f"Simulating LLM call with prompt: {prompt[:50]}...")
        # Simple simulated synthesis
        if "Rainfall" in prompt:
            return ["Irrigation scheduling should be prioritized given the negative water balance detected in the satellite indices."]
        return ["Monitor soil nutrients closely during the heading stage to offset atmospheric stress."]
        
    def generate_advice(self, low_yield_risk: str, feature_importance: Dict[str, float]):
        """
        Produce a list of recommendations based on the features that drove the low-yield risk.
        Example: If 'accumulated_precip' was low, recommend irrigation.
        """
        logger.info(f"Generating advice for risk level: {low_yield_risk}...")
        
        # Sort features by importance
        sorted_feats = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_feature = sorted_feats[0][0]
        
        if self.use_llm:
            prompt = self._generate_llm_prompt(low_yield_risk, feature_importance)
            return self._call_mock_llm(prompt)
            
        # Fallback to Rule-based heuristics if no LLM
        human_feat = self.feature_mappings.get(top_feature, top_feature)
        
        advice = []
        if low_yield_risk in ["Medium Risk", "High Risk"]:
            if "precip" in top_feature:
                advice.append(f"Low yield predicted due to {human_feat}. Consider supplemental irrigation.")
            elif "temp" in top_feature or "gdd" in top_feature:
                advice.append(f"Heat stress from {human_feat} detected. Use mulch or shade if possible.")
            elif "ph" in top_feature:
                advice.append(f"Soil acidity ({human_feat}) is a limiting factor. Consider liming.")
        else:
            advice.append("Condition looks optimal for high yield. Maintain current management practices.")
            
        return advice

def get_recommendation_from_xai(xai_explanation: dict, predicted_risk: str = "High Risk"):
    """
    Integrates results from Integrated Gradients / SHAP to generate advice.
    """
    advisor = AgronomyAdvisor({})
    
    if not xai_explanation:
        return ["No explanation data available to generate recommendations."]
        
    recommendations = advisor.generate_advice(predicted_risk, xai_explanation)
    logger.success("Agronomy recommendations generated from XAI data.")
    return recommendations
