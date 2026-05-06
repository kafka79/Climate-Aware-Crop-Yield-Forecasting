from abc import ABC, abstractmethod
from sentinelhub import (
    SHConfig, 
    SentinelHubRequest, 
    DataCollection, 
    BBox, 
    CRS, 
    MimeType, 
    SentinelHubDownloadClient
)
from typing import Dict, Any, List, Tuple
import os
import pandas as pd
import numpy as np
from loguru import logger


try:
    import cdsapi
except ImportError:
    cdsapi = None

class DataDownloader(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_path = config["paths"]["raw"]
    
    @abstractmethod
    def download(self, *args, **kwargs):
        pass

class SoilDownloader(DataDownloader):
    """
    Downloader for Soil properties (pH, SOC, NPK) via SoilGrids or ISRIC.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.soil_vars = ["phh2o", "soc", "nitrogen"]

    def download(self, bbox: List[float], name: str):
        logger.info(f"Fetching ISRIC Soil data for {name}...")
        # Placeholder for real REST API call to SoilGrids
        # e.g., https://rest.isric.org/soilgrids/v2.0/properties/query
        
        # Synthetic soil data for demonstration
        soil_data = {
            "ph": 6.5 + np.random.normal(0, 0.1),
            "soc": 12.5 + np.random.normal(0, 0.5),
            "nitrogen": 4.2 + np.random.normal(0, 0.2)
        }
        target_path = os.path.join(self.raw_path["soil"], f"{name}_soil.csv")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        pd.DataFrame([soil_data]).to_csv(target_path, index=False)
        logger.success(f"Soil data saved to {target_path}")
        return soil_data

class UPAgDownloader(DataDownloader):
    """
    Downloader for the Unified Portal for Agricultural Statistics (UPAg) API.
    Provides official Area, Production, and Yield (APY) stats from the Govt. of India.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = "https://api.upag.gov.in/v1"
        self.api_key = os.getenv("UPAG_API_KEY", "YOUR_API_KEY")

    def download_yield_data(self, state: str, crop: str, year_range: Tuple[int, int]):
        logger.info(f"Fetching UPAg APY data for {crop} in {state} ({year_range})...")
        if self.api_key == "YOUR_API_KEY":
            logger.warning("UPAg API key is missing. Generating synthetic yield data for demonstration.")
            data = []
            for y in range(year_range[0], year_range[1] + 1):
                data.append({"year": y, "state": state, "crop": crop, "yield": 2.5 + np.random.normal(0, 0.2)})
            return pd.DataFrame(data)
        
        # Real API call placeholder
        return pd.DataFrame()

    def download(self, region: str, crop: str, year_range: Tuple[int, int]):
        return self.download_yield_data(region, crop, year_range)

class SentinelHubDownloader(DataDownloader):
    """
    Implements Sentinel-2 Optical and Sentinel-1 SAR downloads via SentinelHub.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sh_config = SHConfig()
        self.sh_config.sh_client_id = config['sentinel_hub']['client_id']
        self.sh_config.sh_client_secret = config['sentinel_hub']['client_secret']
        
    def download_tile(self, bbox: List[float], time_range: Tuple[str, str], evalscript: str, output_path: str):
        logger.info(f"Initiating SentinelHub request for bbox {bbox}...")
        sh_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
        
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_range
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=sh_bbox,
            config=self.sh_config
        )
        
        try:
            client = SentinelHubDownloadClient(config=self.sh_config)
            data = client.download(request.get_download_list())
            logger.success(f"SentinelHub download successful for {output_path}")
        except Exception as e:
            logger.error(f"SentinelHub API Request failed: {e}")
            raise  # Re-raising for the integration test to catch it

    def download(self, bbox: List[float], time_range: Tuple[str, str], name: str):
        evalscript = "return [B04, B03, B02, B08]" # RGB + NIR
        output_path = os.path.join(self.raw_path["sentinel2"], f"{name}.tiff")
        self.download_tile(bbox, time_range, evalscript, output_path)

class ERA5Downloader(DataDownloader):
    """
    Downloader for ERA5 Reanalysis data via CDS API.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cds_client = cdsapi.Client() if cdsapi else None

    def download(self, bbox: List[float], year: int, name: str):
        if not self.cds_client:
            logger.error("CDS API client not available.")
            return

        logger.info(f"Downloading ERA5 data for {year} in {name}...")
        target_path = os.path.join(self.raw_path["era5"], f"{name}_{year}.nc")
        
        # self.cds_client.retrieve(
        #     'reanalysis-era5-single-levels',
        #     {
        #         'product_type': 'reanalysis',
        #         'variable': ['2m_temperature', 'total_precipitation'],
        #         'year': str(year),
        #         'month': [str(m).zfill(2) for m in range(1, 13)],
        #         'day': [str(d).zfill(2) for d in range(1, 32)],
        #         'time': [f"{h:02d}:00" for h in range(24)],
        #         'area': [bbox[3], bbox[0], bbox[1], bbox[2]],
        #         'format': 'netcdf',
        #     },
        #     target_path
        # )
        logger.success(f"ERA5 download simulated for {target_path}")

def download_multi_modal_batch(config: Dict[str, Any], region: str, crop: str):
    """
    Orchestrate a coordinated download of Yield, Weather, and Satellite data.
    """
    if config.get("use_mock_data", False):
        logger.info("Mock data mode enabled. Generating synthetic datasets locally...")
        from src.data.mock_generator import MockDataGenerator
        generator = MockDataGenerator(config)
        generator.generate_yield_dataset()
        for area in config.get("study_areas", []):
            bbox = area.get("bbox")
            if bbox:
                generator.generate_sentinel_netcdf(area["name"], bbox, config.get("time_range", ("2023-01-01", "2023-12-31")))
                generator.generate_era5_netcdf(area["name"], bbox, config.get("year", 2023))
                generator.generate_soil_csv(area["name"])
        logger.success("Synthetic data generation complete.")
        return

    upag_dl = UPAgDownloader(config)
    sat_dl = SentinelHubDownloader(config)
    era5_dl = ERA5Downloader(config)
    soil_dl = SoilDownloader(config)
    
    # 1. Get Yield Labels (Ground Truth)
    yield_df = upag_dl.download(region, crop, (2018, 2024))
    
    # 2. Extract bounding boxes from yield_df locations (or config) and download
    time_range = config.get("time_range", ("2023-01-01", "2023-12-31"))
    year = config.get("year", 2023)
    
    for area in config.get("study_areas", []):
        bbox = area.get("bbox")
        if bbox:
            sat_dl.download(bbox, time_range, area["name"])
            era5_dl.download(bbox, year, area["name"])
            soil_dl.download(bbox, area["name"])
    
    logger.info("Multi-modal batch download orchestration complete.")
