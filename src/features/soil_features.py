import pandas as pd
import numpy as np
from loguru import logger

class SoilFeatureExtractor:
    """
    Extracts soil-specific features and interactions for crop yield.
    """
    def __init__(self, config: dict):
        self.config = config
        
    def calculate_interaction_features(self, soil_df: pd.DataFrame, weather_df: pd.DataFrame):
        """
        Merge soil and weather datasets and calculate interaction indices.
        Example: (Soil_PH * Mean_Precipitation)
        """
        logger.info("Calculating soil-weather interaction features...")
        
        # Calculate mean precip from weather if it has multiple days
        mean_precip = weather_df['precip'].mean() if 'precip' in weather_df.columns else 0.0
        
        interaction_df = soil_df.copy()
        if 'soil_pH' in interaction_df.columns:
            interaction_df["ph_precip_interaction"] = interaction_df["soil_pH"] * mean_precip
        return interaction_df

    def categorize_soil_texture(self, clay: float, silt: float, sand: float):
        """
        Categorize soil into texture classes based on USDA soil triangle.
        (Simplified version)
        """
        logger.debug(f"Categorizing soil with clay={clay}, silt={silt}, sand={sand}...")
        if clay > 40:
            return "Clay"
        elif sand > 85:
            return "Sand"
        else:
            return "Loam"


# ponytail: process_soil_metrics deleted — dead code, never imported anywhere.
# It also had a bug: std() on a 1-row dataframe returns NaN, crashing inference.
