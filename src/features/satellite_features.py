import xarray as xr
import numpy as np
from loguru import logger
from typing import Dict, Any

class SatelliteFeatureExtractor:
    """
    Extracts vegetation indices and other biophysical parameters from multispectral satellite data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def calculate_ndvi(self, ds: xr.Dataset):
        """
        Normalized Difference Vegetation Index (NDVI).
        Range: [-1, 1]. High values indicate dense green vegetation.
        """
        logger.info("Calculating NDVI...")
        # (NIR - Red) / (NIR + Red)
        # Sentinel-2: B8 is NIR, B4 is Red
        nir = ds.B08
        red = ds.B04
        ndvi = (nir - red) / (nir + red + 1e-10)
        return ndvi.rename("NDVI")

    def calculate_evi(self, ds: xr.Dataset):
        """
        Enhanced Vegetation Index (EVI).
        More sensitive to high biomass regions and reduces atmospheric/soil noise.
        """
        logger.info("Calculating EVI...")
        # 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
        # Sentinel-2: B8=NIR, B4=Red, B2=Blue
        nir = ds.B08
        red = ds.B04
        blue = ds.B02
        
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1.0 + 1e-10))
        return evi.rename("EVI")

    def calculate_lswI(self, ds: xr.Dataset):
        """
        Land Surface Water Index (LSWI).
        Sensitive to leaf water content and soil moisture.
        """
        logger.info("Calculating LSWI...")
        # (NIR - SWIR) / (NIR + SWIR)
        # Sentinel-2: B8=NIR, B11=SWIR1
        if 'B11' in ds:
            nir = ds.B08
            swir = ds.B11
            lswi = (nir - swir) / (nir + swir + 1e-10)
            return lswi.rename("LSWI")
        else:
            logger.warning("SWIR band (B11) not found. Skipping LSWI.")
            return None

    def extract_all_indices(self, ds: xr.Dataset):
        """
        Compute a stack of all relevant indices.
        """
        indices = []
        indices.append(self.calculate_ndvi(ds))
        indices.append(self.calculate_evi(ds))
        
        lswi = self.calculate_lswI(ds)
        if lswi is not None:
            indices.append(lswi)
            
        return xr.merge(indices)

def get_satellite_features(ds: xr.Dataset, config: Dict[str, Any]):
    extractor = SatelliteFeatureExtractor(config)
    return extractor.extract_all_indices(ds)
