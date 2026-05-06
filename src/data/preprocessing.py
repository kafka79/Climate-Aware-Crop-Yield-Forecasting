import os
import xarray as xr
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, List

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
        Loads Sentinel data and applies cloud masking using the SCL band.
        Improved to check for data density.
        """
        logger.info(f"Preprocessing Sentinel data: {file_path}")
        try:
            ds = xr.open_dataset(file_path, engine="netcdf4", chunks={"time": 12, "lat": 100, "lon": 100}) 
            if 'SCL' in ds:
                mask = ds.SCL.isin([4, 5, 6]) 
                valid_ratio = mask.mean().values
                
                if valid_ratio < 0.1:
                    logger.warning(f"Extremely high cloud cover ({100 - valid_ratio*100:.1f}%) in {file_path}. Results may be unreliable.")
                
                ds_masked = ds.where(mask)
                return ds_masked
            else:
                logger.warning(f"SCL band not found in {file_path}. Skipping cloud mask.")
                return ds
        except Exception as e:
            logger.error(f"Error preprocessing Sentinel data: {e}")
            return None

    def fill_temporal_gaps(self, ds: xr.Dataset):
        """
        Fills temporal gaps in satellite data using PCHIP interpolation.
        Includes a fallback to linear interpolation if pchip fails or has insufficient points.
        """
        logger.info("Filling temporal gaps...")
        ds = ds.chunk(dict(time=-1))
        
        try:
            # PCHIP is better for growth curves but requires at least 2 points
            ds = ds.interpolate_na(dim="time", method="pchip", limit=5)
        except Exception as e:
            logger.warning(f"PCHIP interpolation failed ({e}), falling back to linear.")
            ds = ds.interpolate_na(dim="time", method="linear", limit=5)
            
        return ds.fillna(0)
    
    def preprocess_weather(self, file_path: str):
        """
        Aggregates ERA5 hourly weather data into daily/monthly summaries.
        """
        logger.info(f"Preprocessing weather data: {file_path}")
        try:
            # Assuming ERA5 NetCDF format from CDS
            ds = xr.open_dataset(file_path)
            
            # ERA5 often uses 'latitude' and 'longitude', rename if present
            if 'latitude' in ds.dims and 'longitude' in ds.dims:
                ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
                
            # Resample to Daily Mean
            ds_daily = ds.resample(time='1D').mean()
            
            # Resample to Monthly if needed for seasonal analysis
            # ds_monthly = ds.resample(time='1M').mean()
            
            return ds_daily
        except Exception as e:
            logger.error(f"Error preprocessing weather data: {e}")
            return None

    def align_modalities(self, sat_ds: xr.Dataset, weather_ds: xr.Dataset):
        """
        Aligns satellite and weather datasets spatially and temporally.
        Ensures both have the same time steps and spatial grid for the target region.
        """
        logger.info("Aligning multi-modal datasets...")
        
        # 1. Temporal Alignment (Intersect time ranges)
        if 'time' in sat_ds.coords:
            sat_ds['time'] = sat_ds.time.dt.floor('D')
            sat_ds = sat_ds.sel(time=~sat_ds.indexes['time'].duplicated())
            
        if 'time' in weather_ds.coords:
            weather_ds['time'] = weather_ds.time.dt.floor('D')
            weather_ds = weather_ds.sel(time=~weather_ds.indexes['time'].duplicated())
            
        common_times = np.intersect1d(sat_ds.time.values, weather_ds.time.values)
        sat_ds = sat_ds.sel(time=common_times)
        weather_ds = weather_ds.sel(time=common_times)
        
        # 2. Spatial Alignment (Regrid weather to match satellite resolution)
        # Using bilinear interpolation for smoother, physically consistent weather fields.
        logger.debug(f"Interpolating weather ({len(weather_ds.lat)}x{len(weather_ds.lon)}) to satellite resolution ({len(sat_ds.lat)}x{len(sat_ds.lon)})...")
        weather_ds_respaced = weather_ds.interp(
            lat=sat_ds.lat, 
            lon=sat_ds.lon, 
            method="linear"
        ).fillna(0)
        
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
                
                final_sat, final_weather = preprocessor.align_modalities(ds_sat, ds_weather)
                
                # Force uniform chunking for Zarr compatibility
                final_sat = final_sat.chunk({"time": -1, "lat": 10, "lon": 10})
                final_weather = final_weather.chunk({"time": -1, "lat": 10, "lon": 10})

                # Save processed data as Zarr for better lazy-loading performance
                out_dir = config["paths"]["processed"]["features"]
                os.makedirs(out_dir, exist_ok=True)
                final_sat.to_zarr(os.path.join(out_dir, f"{name}_sat_proc.zarr"), mode="w")
                final_weather.to_zarr(os.path.join(out_dir, f"{name}_weather_proc.zarr"), mode="w")
                
    logger.success("Preprocessing phase complete.")
