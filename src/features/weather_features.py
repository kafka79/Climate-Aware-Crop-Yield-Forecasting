import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger
def calculate_gdd(t_max: np.ndarray, t_min: np.ndarray, base_temp: float = 10.0, cap_temp: float = 35.0):
    """
    Calculate Growing Degree Days (GDD).
    (Min(Cap, Max(Base, Tmax)) + Min(Cap, Max(Base, Tmin))) / 2 - Base
    """
    logger.debug(f"Calculating GDD with base_temp={base_temp}...")
    t_max_adj = np.clip(t_max, base_temp, cap_temp)
    t_min_adj = np.clip(t_min, base_temp, cap_temp)
    gdd = (t_max_adj + t_min_adj) / 2 - base_temp
    return np.maximum(gdd, 0)

class WeatherFeatureExtractor:
    """
    Extracts agronomic features from raw meteorological data.
    """
    def __init__(self, crop_config: dict):
        self.base_temp = crop_config.get("base_temp", 10.0)
        self.cap_temp = crop_config.get("cap_temp", 35.0)

    def extract_seasonal_features(self, weather_df: pd.DataFrame):
        """
        Calculate seasonal accumulations (Precipitation, GDD).
        """
        logger.info("Extracting seasonal weather features...")
        weather_df["gdd"] = calculate_gdd(weather_df["t_max"], weather_df["t_min"], self.base_temp, self.cap_temp)
        weather_df["accumulated_precip"] = weather_df["precip"].cumsum()
        weather_df["accumulated_gdd"] = weather_df["gdd"].cumsum()
        
        # Calculate 3-month SPI
        weather_df["spi_3"] = calculate_spi(weather_df["precip"], scale=3)
        return weather_df

def calculate_spi(precip: pd.Series, scale: int = 3):
    """
    Calculate Standardized Precipitation Index (SPI) for drought monitoring.
    Fits a Gamma distribution to the cumulative precipitation.
    """
    logger.info(f"Calculating SPI with scale={scale} months...")
    
    # 1. Calculate rolling sum for the scale
    rolling_precip = precip.rolling(window=scale).sum().dropna()
    
    if len(rolling_precip) < 10:
        logger.warning("Not enough data to fit Gamma distribution for SPI. Returning z-score.")
        return (precip - precip.mean()) / (precip.std() + 1e-10)

    # 2. Fit Gamma distribution
    # scipy.stats.gamma.fit returns (a, loc, scale)
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(rolling_precip, floc=0)
    
    # 3. Calculate Cumulative Probability (CDF)
    cdf = stats.gamma.cdf(rolling_precip, fit_alpha, loc=fit_loc, scale=fit_beta)
    
    # 4. Transform to Standard Normal (Inverse Normal CDF)
    spi_values = stats.norm.ppf(cdf)
    
    # Realign with the original series index
    spi_series = pd.Series(index=precip.index, dtype=float)
    spi_series.loc[rolling_precip.index] = spi_values
    
    return spi_series.ffill().bfill()
