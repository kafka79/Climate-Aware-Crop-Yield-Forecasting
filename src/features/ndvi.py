import numpy as np
import xarray as xr
from loguru import logger

def calculate_ndvi(red: np.ndarray, nir: np.ndarray):
    """
    Calculate Normalized Difference Vegetation Index (NDVI).
    (NIR - RED) / (NIR + RED)
    """
    logger.debug("Calculating NDVI...")
    return (nir - red) / (nir + red + 1e-10)

def calculate_evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray):
    """
    Calculate Enhanced Vegetation Index (EVI).
    2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))
    """
    logger.debug("Calculating EVI...")
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-10))

class VegetationFeatureExtractor:
    """
    Extracts vegetation indices from multi-spectral satellite imagery.
    """
    def __init__(self, bands_config: dict):
        self.red_idx = bands_config.get("red", "B04")
        self.nir_idx = bands_config.get("nir", "B08")
        self.blue_idx = bands_config.get("blue", "B02")

    def extract_from_xarray(self, ds: xr.Dataset):
        """
        Apply index calculations to an xarray dataset.
        """
        logger.info("Extracting vegetation features from xarray dataset...")
        ds["ndvi"] = calculate_ndvi(ds[self.red_idx], ds[self.nir_idx])
        ds["evi"] = calculate_evi(ds[self.blue_idx], ds[self.red_idx], ds[self.nir_idx])
        return ds
