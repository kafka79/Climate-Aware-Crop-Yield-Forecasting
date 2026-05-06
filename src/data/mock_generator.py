import os
import pandas as pd
import numpy as np
import xarray as xr
from loguru import logger
from typing import Dict, Any

class MockDataGenerator:
    """
    Generates realistic dummy data for testing the pipeline without API keys.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_path = config["paths"]["raw"]
        
    def generate_yield_dataset(self, filename: str = "historical_yield.csv"):
        logger.info("Generating mock historical yield dataset...")
        np.random.seed(42)
        records = []
        for area in self.config.get("study_areas", []):
            lat, lon = area.get("lat"), area.get("lon")
            if lat and lon:
                for year in range(self.config["yield"]["historical_years"][0], self.config["yield"]["historical_years"][1] + 1):
                    records.append({
                        "lat": lat,
                        "lon": lon,
                        "time": f"{year}-12-31",
                        "yield": np.random.normal(loc=3.5, scale=0.5), # ~3.5 t/ha
                        "site_id": area["name"]
                    })
                    
        df = pd.DataFrame(records)
        os.makedirs(self.raw_path["yield"], exist_ok=True)
        df.to_csv(os.path.join(self.raw_path["yield"], filename), index=False)
        logger.success(f"Mock yield dataset generated at {filename}.")

    def generate_sentinel_netcdf(self, name: str, bbox: list, time_range: tuple):
        logger.info(f"Generating mock Sentinel NetCDF for {name}...")
        times = pd.date_range(start=time_range[0], end=time_range[1], freq='15D')
        
        # Approximate 50km box as roughly 0.5 degrees. Let's make an 10x10 grid.
        lats = np.linspace(bbox[1], bbox[3], 10)
        lons = np.linspace(bbox[0], bbox[2], 10)
        
        shape = (len(times), len(lats), len(lons))
        
        ds = xr.Dataset(
            {
                "B02": (["time", "lat", "lon"], np.random.uniform(0.01, 0.1, shape)),
                "B03": (["time", "lat", "lon"], np.random.uniform(0.01, 0.1, shape)),
                "B04": (["time", "lat", "lon"], np.random.uniform(0.01, 0.1, shape)),
                "B08": (["time", "lat", "lon"], np.random.uniform(0.2, 0.6, shape)), # Higher NIR
                "SCL": (["time", "lat", "lon"], np.random.choice([4, 5, 6, 8, 9], shape)), # Include some clouds (8,9)
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons
            }
        )
        os.makedirs(self.raw_path["sentinel2"], exist_ok=True)
        ds.to_netcdf(os.path.join(self.raw_path["sentinel2"], f"{name}.nc"))
        logger.success(f"Mock Sentinel data generated for {name}.")

    def generate_era5_netcdf(self, name: str, bbox: list, year: int):
        logger.info(f"Generating mock ERA5 NetCDF for {name} ({year})...")
        times = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='1D')
        lats = np.linspace(bbox[1], bbox[3], 5)
        lons = np.linspace(bbox[0], bbox[2], 5)
        
        shape = (len(times), len(lats), len(lons))
        
        ds = xr.Dataset(
            {
                "t_max": (["time", "lat", "lon"], np.random.normal(30, 5, shape)),
                "t_min": (["time", "lat", "lon"], np.random.normal(20, 5, shape)),
                "precip": (["time", "lat", "lon"], np.random.exponential(5, shape)),
            },
            coords={
                "time": times,
                "lat": lats,
                "lon": lons
            }
        )
        os.makedirs(self.raw_path["era5"], exist_ok=True)
        ds.to_netcdf(os.path.join(self.raw_path["era5"], f"{name}_{year}.nc"))
        logger.success(f"Mock ERA5 data generated for {name}.")

    def generate_soil_csv(self, name: str):
        logger.info(f"Generating mock soil CSV for {name}...")
        soil_data = {
            "ph": 6.5 + np.random.normal(0, 0.1),
            "soc": 12.5 + np.random.normal(0, 0.5),
            "nitrogen": 4.2 + np.random.normal(0, 0.2)
        }
        os.makedirs(self.raw_path["soil"], exist_ok=True)
        target_path = os.path.join(self.raw_path["soil"], f"{name}_soil.csv")
        pd.DataFrame([soil_data]).to_csv(target_path, index=False)
        logger.success(f"Mock soil data generated at {target_path}.")
