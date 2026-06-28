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
        region = result.get("region", "Unknown Region")
        prompt = f"""
        You are a senior agronomic consultant. Analyze the following crop yield forecast data and provide 
        3-4 highly specific, professional recommendations for a farmer or regional planner.
        
        DATA:
        - Region: {region}
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
        A sophisticated heuristic engine that maps attribution and risk to specific agronomic advice.
        """
        advice = []
        attr = result["attribution"]
        risk = str(result.get("risk", "")).upper()
        
        # 1. Attribution-Specific Logic (Dynamic 'Why' & Specific Interventions)
        top_factor = max(attr, key=attr.get)
        
        if top_factor == "Weather":
            advice.append(f"🌦️ **Weather Dominance ({attr['Weather']:.0%}):** High sensitivity to weather variations. If experiencing excessive seasonal precipitation, clear peripheral drainage trenches to prevent crop root rot. Under dry conditions, initiate deficit irrigation cycles and apply organic mulching to retard evaporation.")
        elif top_factor == "Satellite":
            advice.append(f"🛰️ **Biomass Signal ({attr['Satellite']:.0%}):** Yield is driven by crop vigor (NDVI). To maintain this progress, complete a split-nitrogen top-dressing before panicle initiation, and monitor canopy density closely to apply pest management protocols at the first sign of infestation.")
        elif top_factor == "Soil":
            advice.append(f"🌱 **Soil Constraints ({attr['Soil']:.0%}):** Soil composition limits regional yield ceiling. To bypass root absorption constraints, apply a customized foliar spray of micro-nutrients (specifically Zinc and Boron) alongside a targeted mid-season NPK top-dress.")

        # 2. Risk-Based Logic
        if "HIGH" in risk:
            advice.append("🚨 **Emergency Action:** Yield is significantly below trend. Conduct a soil-moisture profile audit and check leaf tissue for nitrogen deficiency. Consider micro-irrigation or nitrogen foliar application if stress is confirmed.")
        elif "LOW" in risk:
            advice.append("📈 **Surplus Preparation:** Expected yield is above average. Coordinate storage facility capacity, source drying equipment early to prevent post-harvest mold, and engage local distribution networks to lock in optimal pricing.")

        # 3. Uncertainty Logic (dimension-safe denominator to prevent ZeroDivisionError)
        predicted = result.get("predicted_yield", 0.0)
        denominator = max(abs(predicted), 0.01)  # never divide by zero regardless of model output
        range_pct = (result["upper_bound"] - result["lower_bound"]) / denominator
        if range_pct > 0.4:
            advice.append("⚠️ **Risk Hedging (Data Volatility):** High forecast variance indicates conflicting satellite and weather indicators. Postpone intensive fertilizer applications to avoid wasting inputs. Symmetrically prepare channels—clear drainage path (for potential wet spikes) and check pump readiness (for dry drops).")

        return advice
