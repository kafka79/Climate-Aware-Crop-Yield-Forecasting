import numpy as np
from loguru import logger
from typing import List, Tuple

def rescale_bands(data: np.ndarray, scale: float = 10000.0):
    """
    Rescale Sentinel DN (Digital Numbers) to TOA (Top of Atmosphere) reflectance.
    """
    logger.debug(f"Rescaling satellite bands with scale={scale}...")
    return data / scale

def get_bbox_from_point(lat: float, lon: float, buffer: float = 0.01):
    """
    Create a bounding box around a specific coordinate for satellite data fetching.
    """
    logger.debug(f"Generating bbox for point ({lat}, {lon}) with buffer {buffer}...")
    min_lat, max_lat = lat - buffer, lat + buffer
    min_lon, max_lon = lon - buffer, lon + buffer
    return [min_lon, min_lat, max_lon, max_lat]

def calculate_anomaly(current_val: float, historical_avg: float):
    """
    Calculate the climate anomaly as a percentage deviation from the mean.
    (Current - Mean) / Mean
    """
    return (current_val - historical_avg) / (historical_avg + 1e-10)

class GeospatialMapper:
    """
    Utility for mapping between administrative regions and coordinates.
    """
    @staticmethod
    def get_district_coordinates(district: str, state: str):
        # Placeholder for a lookup table or a GeoIP / Nominatim request
        # Example coordinate for "Burdwan, West Bengal"
        if "Burdwan" in district:
            return 23.2324, 87.8615
        return None, None
