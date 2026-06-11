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
from tenacity import retry, stop_after_attempt, wait_exponential


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

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def download(self, bbox: List[float], name: str):
        logger.info(f"Fetching ISRIC Soil data for {name}...")
        import requests
        lon_center = (bbox[0] + bbox[2]) / 2.0
        lat_center = (bbox[1] + bbox[3]) / 2.0
        
        try:
            url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon_center}&lat={lat_center}&property=phh2o&property=soc&property=nitrogen&depth=0-5cm&value=mean"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            props = data.get("properties", {}).get("layers", [])
            soil_data = {"ph": 0.0, "soc": 0.0, "nitrogen": 0.0}
            for layer in props:
                prop_name = layer.get("name")
                mean_val = layer.get("depths", [{}])[0].get("values", {}).get("mean", 0.0)
                if prop_name == "phh2o":
                    soil_data["ph"] = mean_val / 10.0 # ISRIC pH is scaled by 10
                elif prop_name == "soc":
                    soil_data["soc"] = mean_val / 10.0 # dg/kg to g/kg
                elif prop_name == "nitrogen":
                    soil_data["nitrogen"] = mean_val / 100.0 # cg/kg to g/kg
        except Exception as e:
            logger.error(f"ISRIC API failed: {e}. Falling back to default zeros.")
            soil_data = {"ph": 0.0, "soc": 0.0, "nitrogen": 0.0}
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

    def download_yield_data(self, state: str, crop: str, year_range: Tuple[int, int]) -> pd.DataFrame:
        logger.info(f"Fetching UPAg APY data for {crop} in {state} ({year_range})...")
        if self.api_key == "YOUR_API_KEY" or not self.api_key:
            raise ValueError("UPAg API key is missing. Set UPAG_API_KEY environment variable. Mock generation is disabled for production.")
        
        import requests
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        params = {
            "state": state,
            "crop": crop,
            "start_year": year_range[0],
            "end_year": year_range[1]
        }
        try:
            response = requests.get(f"{self.base_url}/apy-statistics", headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            records = []
            for item in data.get("data", []):
                records.append({
                    "site_id": state,
                    "time": f"{item.get('year')}-12-31",
                    "yield": float(item.get("yield_t_ha", 0.0)),
                    "lat": float(item.get("latitude", 0.0)),
                    "lon": float(item.get("longitude", 0.0))
                })
            
            if not records:
                raise RuntimeError("UPAg API returned no records for the requested parameters.")
                
            df = pd.DataFrame(records)
            target_path = os.path.join(self.raw_path["yield"], "historical_yield.csv")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            df.to_csv(target_path, index=False)
            logger.success(f"Yield data saved to {target_path}")
            return df
        except Exception as e:
            logger.error(f"UPAg API request failed: {e}")
            raise RuntimeError(f"Failed to fetch APY statistics from UPAg: {e}")

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
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=120))
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
        evalscript = """//VERSION=3
        function setup() {
            return {
                input: ["B04", "B03", "B02", "B08"],
                output: { bands: 4 }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02, sample.B08];
        }
        """
        output_path = os.path.join(self.raw_path["sentinel2"], f"{name}.tiff")
        self.download_tile(bbox, time_range, evalscript, output_path)

class ERA5Downloader(DataDownloader):
    """
    Downloader for ERA5 Reanalysis data via CDS API.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cds_client = cdsapi.Client() if cdsapi else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=10, max=120))
    def download(self, bbox: List[float], year: int, name: str):
        if not self.cds_client:
            raise ImportError(
                "CDS API client (cdsapi) is not installed or configured. "
                "Install with: pip install cdsapi"
            )

        logger.info(f"Downloading ERA5 data for {year} in {name}...")
        target_path = os.path.join(self.raw_path["era5"], f"{name}_{year}.nc")
        
        self.cds_client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['2m_temperature', 'total_precipitation'],
                'year': str(year),
                'month': [str(m).zfill(2) for m in range(1, 13)],
                'day': [str(d).zfill(2) for d in range(1, 32)],
                'time': [f"{h:02d}:00" for h in range(24)],
                'area': [bbox[3], bbox[0], bbox[1], bbox[2]],
                'format': 'netcdf',
            },
            target_path
        )
        logger.success(f"ERA5 download completed for {target_path}")

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
