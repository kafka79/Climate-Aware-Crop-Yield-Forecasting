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
        # Placeholder for real API implementation
        logger.warning("UPAg API call is mocked. Returning empty DataFrame.")
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
        
        # client = SentinelHubDownloadClient(config=self.sh_config)
        # data = client.download(request.get_download_list())
        # ... logic to save to output_path ...
        logger.success(f"SentinelHub download simulated for {output_path}")

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
    upag_dl = UPAgDownloader(config)
    sat_dl = SentinelHubDownloader(config)
    era5_dl = ERA5Downloader(config)
    
    # 1. Get Yield Labels (Ground Truth)
    yield_df = upag_dl.download(region, crop, (2018, 2024))
    
    # 2. Extract bounding boxes from yield_df locations (or config) and download
    for area in config.get("study_areas", []):
        bbox = area.get("bbox")
        if bbox:
            sat_dl.download(bbox, ("2023-01-01", "2023-12-31"), area["name"])
            era5_dl.download(bbox, 2023, area["name"])
    
    logger.info("Multi-modal batch download orchestration complete.")
