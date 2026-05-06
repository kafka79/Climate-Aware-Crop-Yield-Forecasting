import os
import json
from typing import Dict, Any, List
from loguru import logger

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

class RecommendationEngine:
    """
    Translates yield forecasts, risk levels, and model attributions into actionable advice.
    Features a dynamic 'Expert Mode' using LLMs when an API key is provided.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if self.api_key and HAS_GENAI:
            logger.info("Initializing Generative Recommendation Engine (LLM-Powered).")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            logger.warning("No LLM key found. Falling back to Heuristic Dynamic Engine.")
            self.model = None

    def generate_advice(self, inference_result: Dict[str, Any]) -> List[str]:
        """
        Generates advice. Prefers LLM for 'Exceptional' depth, falls back to Heuristics.
        """
        if self.model:
            return self._generate_llm_advice(inference_result)
        return self._generate_heuristic_advice(inference_result)

    def _generate_llm_advice(self, result: Dict[str, Any]) -> List[str]:
        """
        Uses Gemini to generate a professional agronomic report based on model data.
        """
        prompt = f"""
        You are a senior agronomic consultant. Analyze the following crop yield forecast data and provide 
        3-4 highly specific, professional recommendations for a farmer or regional planner.
        
        DATA:
        - Region: {result['region']}
        - Forecasted Yield: {result['predicted_yield']:.2f} t/ha
        - Confidence Interval: [{result['lower_bound']:.2f} - {result['upper_bound']:.2f}]
        - Risk Level: {result['risk']}
        - Model Attribution (What drove this prediction): {json.dumps(result['attribution'])}
        
        Format the output as a list of bullet points. Be concise but technical.
        Mention specific interventions related to the highest attribution factors.
        """
        try:
            response = self.model.generate_content(prompt)
            # Split lines and clean up
            advice = [line.strip("* ").strip("- ") for line in response.text.strip().split("\n") if line.strip()]
            return advice[:5]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}. Falling back to heuristics.")
            return self._generate_heuristic_advice(result)

    def _generate_heuristic_advice(self, result: Dict[str, Any]) -> List[str]:
        """
        A sophisticated heuristic engine that maps attribution and risk to specific advice.
        """
        advice = []
        attr = result["attribution"]
        risk = result["risk"]
        
        # 1. Attribution-Specific Logic (Dynamic 'Why')
        top_factor = max(attr, key=attr.get)
        
        if top_factor == "Weather":
            advice.append(f"🌦️ **Weather Dominance ({attr['Weather']:.0%}):** The model is highly sensitive to recent climate shifts. Prioritize monitoring short-term weather alerts.")
        elif top_factor == "Satellite":
            advice.append(f"🛰️ **Biomass Signal ({attr['Satellite']:.0%}):** The forecast is driven primarily by current crop vigor (NDVI). The crop looks healthy, but maintain vigilance for late-season pests.")
        elif top_factor == "Soil":
            advice.append(f"🌱 **Soil Constraints ({attr['Soil']:.0%}):** Regional soil properties are limiting the yield ceiling. Consider a mid-season NPK top-dress if moisture allows.")

        # 2. Risk-Based Logic
        if risk == "HIGH":
            advice.append("🚨 **Emergency Action:** Yield is significantly below the 5-year trend. Conduct a field-level audit for water stress or nutrient deficiency immediately.")
        elif risk == "LOW":
            advice.append("📈 **Surplus Opportunity:** Yield is above average. Secure storage and transport logistics early to avoid post-harvest losses.")

        # 3. Uncertainty Logic
        range_pct = (result["upper_bound"] - result["lower_bound"]) / result["predicted_yield"]
        if range_pct > 0.4:
            advice.append("⚠️ **Data Volatility:** High variance in the forecast. This usually indicates conflicting signals between satellite imagery and weather data. Re-verify the forecast in 7 days.")

        return advice
