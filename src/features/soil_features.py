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

def process_soil_metrics(soil_df: pd.DataFrame):
    """
    Normalize and scale soil metrics (pH, SOC, N, P, K).
    """
    logger.info("Normalizing soil metrics...")
    # Get numeric columns only
    metrics = soil_df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(metrics) == 0:
        return soil_df

    # Avoid div by zero
    std_devs = soil_df[metrics].std()
    std_devs[std_devs == 0] = 1e-10

    normalized = (soil_df[metrics] - soil_df[metrics].mean()) / std_devs
    
    # Merge normalized with non-numeric
    non_numeric = soil_df.select_dtypes(exclude=[np.number])
    return pd.concat([normalized, non_numeric], axis=1)
