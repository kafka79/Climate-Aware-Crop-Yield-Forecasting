import os
import xarray as xr
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, List
from src.utils.helpers import rescale_bands

class DataPreprocessor:
    """
    Cleans and prepares raw satellite and weather data for feature engineering.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_path = config["paths"]["raw"]
        self.proc_path = config["paths"]["processed"]
        
    def preprocess_sentinel(self, file_path: str):
        """
        Loads Sentinel data and applies cloud masking using the SCL (Scene Classification) band.
        SCL = 4 (Vegetation), 5 (Not-vegetated), 6 (Water) are considered "clear".
        """
        logger.info(f"Preprocessing Sentinel data: {file_path}")
        try:
            # Use chunks for Dask-backed lazy loading
            ds = xr.open_dataset(file_path, engine="netcdf4", chunks={"time": 12, "lat": 100, "lon": 100}) 
            if 'SCL' in ds:
                # Keep only vegetation, non-vegetated, and water (removing clouds/shadows)
                mask = ds.SCL.isin([4, 5, 6]) 
                ds_masked = ds.where(mask)
                logger.debug(f"Cloud mask applied. Dropped {100 - (mask.mean().values * 100):.2f}% pixels.")
                return ds_masked
            else:
                logger.warning(f"SCL band not found in {file_path}. Skipping cloud mask.")
                return ds
        except Exception as e:
            logger.error(f"Error preprocessing Sentinel data: {e}")
            return None

    def fill_temporal_gaps(self, ds: xr.Dataset):
        """
        Fills NaNs in the temporal dimension using linear interpolation.
        Essential for recovering from cloud-masked holes.
        """
        logger.info("Filling temporal gaps via linear interpolation...")
        # limit=3 ensures we don't interpolate over massive missing seasonal blocks
        return ds.interpolate_na(dim="time", method="linear", limit=3)
    
    def preprocess_weather(self, file_path: str):
        """
        Aggregates ERA5 hourly weather data into daily/monthly summaries.
        """
        logger.info(f"Preprocessing weather data: {file_path}")
        try:
            # Assuming ERA5 NetCDF format from CDS
            ds = xr.open_dataset(file_path)
            # Resample to Daily Mean
            ds_daily = ds.resample(time='1D').mean()
            
            # Resample to Monthly if needed for seasonal analysis
            # ds_monthly = ds.resample(time='1M').mean()
            
            return ds_daily
        except Exception as e:
            logger.error(f"Error preprocessing weather data: {e}")
            return None

    def align_modalities(self, sat_ds: xr.Dataset, weather_ds: xr.Dataset, yield_df: pd.DataFrame):
        """
        Aligns satellite and weather datasets spatially and temporally.
        Ensures both have the same time steps and spatial grid for the target region.
        """
        logger.info("Aligning multi-modal datasets...")
        
        # 1. Temporal Alignment (Intersect time ranges)
        common_times = np.intersect1d(sat_ds.time.values, weather_ds.time.values)
        sat_ds = sat_ds.sel(time=common_times)
        weather_ds = weather_ds.sel(time=common_times)
        
        # 2. Spatial Alignment (Regrid weather to match satellite resolution)
        # Using bilinear interpolation for smoother, physically consistent weather fields.
        logger.debug(f"Interpolating weather ({len(weather_ds.lat)}x{len(weather_ds.lon)}) to satellite resolution ({len(sat_ds.lat)}x{len(sat_ds.lon)})...")
        weather_ds_respaced = weather_ds.interp(
            lat=sat_ds.lat, 
            lon=sat_ds.lon, 
            method="bilinear"
        )
        
        logger.success("Fusion-ready alignment complete.")
        return sat_ds, weather_ds_respaced

def preprocess_all(config: Dict[str, Any]):
    """
    Main function to execute the preprocessing pipeline across all study areas.
    """
    preprocessor = DataPreprocessor(config)
    
    for area in config.get("study_areas", []):
        name = area["name"]
        sat_file = os.path.join(config["paths"]["raw"]["sentinel2"], f"{name}.nc")
        weather_file = os.path.join(config["paths"]["raw"]["era5"], f"{name}_2023.nc")
        
        if os.path.exists(sat_file) and os.path.exists(weather_file):
            ds_sat = preprocessor.preprocess_sentinel(sat_file)
            ds_weather = preprocessor.preprocess_weather(weather_file)
            
            if ds_sat is not None and ds_weather is not None:
                # Fill temporal gaps in satellite data before fusion
                ds_sat = preprocessor.fill_temporal_gaps(ds_sat)
                
                final_sat, final_weather = preprocessor.align_modalities(ds_sat, ds_weather, pd.DataFrame())
                
                # Save processed data
                final_sat.to_netcdf(os.path.join(config["paths"]["processed"]["features"], f"{name}_sat_proc.nc"))
                final_weather.to_netcdf(os.path.join(config["paths"]["processed"]["features"], f"{name}_weather_proc.nc"))
                
    logger.success("Preprocessing phase complete.")
