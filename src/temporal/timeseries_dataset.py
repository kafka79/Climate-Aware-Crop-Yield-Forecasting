import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import pandas as pd
import xarray as xr
import math
from loguru import logger
from typing import Iterator

class MultiModalCropIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for multi-modal crop yield prediction.
    Streams sequences lazily from Zarr stores in chunks to prevent OOM errors.
    """
    def __init__(self, yield_df: pd.DataFrame, sat_ds: xr.Dataset, 
                 weather_ds: xr.Dataset, soil_vectors: np.ndarray, config: dict, chunk_size: int = 1000):
        self.yield_df = yield_df
        self.sat_ds = sat_ds
        self.weather_ds = weather_ds
        self.soil_vectors = soil_vectors
        self.config = config
        self.chunk_size = chunk_size
        self.window_size = config.get("training", {}).get("window_size", 12)
        self.C = config["transformer"]["input_dim"]
        
        logger.info(f"Initialized IterableDataset with {len(self.yield_df)} samples.")

    def __iter__(self) -> Iterator[dict]:
        import torch.distributed as dist
        
        # Step 1: Partition data across DDP process ranks
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            indices = np.arange(len(self.yield_df))
            rank_indices = indices[rank::world_size]
            rank_df = self.yield_df.iloc[rank_indices]
            rank_soil = self.soil_vectors[rank_indices]
        else:
            rank_df = self.yield_df
            rank_soil = self.soil_vectors

        # Step 2: Partition rank slice across DataLoader worker subprocesses
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_df = rank_df
            worker_soil = rank_soil
        else:
            per_worker = int(math.ceil(len(rank_df) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(rank_df))
            worker_df = rank_df.iloc[start:end]
            worker_soil = rank_soil[start:end]

        if len(worker_df) == 0:
            return iter([])

        # Process in chunks to balance I/O and Memory
        for chunk_start in range(0, len(worker_df), self.chunk_size):
            chunk_df = worker_df.iloc[chunk_start:chunk_start + self.chunk_size]
            chunk_soil = worker_soil[chunk_start:chunk_start + self.chunk_size]
            
            lats = xr.DataArray(chunk_df["lat"].values, dims="location")
            lons = xr.DataArray(chunk_df["lon"].values, dims="location")
            
            # Load a chunk into memory
            sat_pixels = self.sat_ds.sel(lat=lats, lon=lons, method="nearest").load()
            weather_pixels = self.weather_ds.sel(lat=lats, lon=lons, method="nearest").load()

            for i, (_, row) in enumerate(chunk_df.iterrows()):
                yield_time = pd.to_datetime(row["time"])
                
                try:
                    sat_pixel = sat_pixels.isel(location=i)
                    weather_pixel = weather_pixels.isel(location=i)

                    sat_hist = sat_pixel.sel(time=slice(None, yield_time)).tail(time=self.window_size)
                    w_hist = weather_pixel.sel(time=slice(None, yield_time)).tail(time=self.window_size)

                    if len(sat_hist.time) < self.window_size:
                        sat_hist = sat_pixel.isel(time=slice(0, self.window_size))
                        w_hist = weather_pixel.isel(time=slice(0, self.window_size))

                    if len(sat_hist.time) < self.window_size:
                        continue

                    if self.config.get("use_spatial_patches", False):
                        # Extract a 3x3 patch around target lat/lon in self.sat_ds
                        lat, lon = row["lat"], row["lon"]
                        lat_idx = int(np.argmin(np.abs(self.sat_ds.lat.values - lat)))
                        lon_idx = int(np.argmin(np.abs(self.sat_ds.lon.values - lon)))
                        lat_slice = slice(max(0, lat_idx - 1), min(len(self.sat_ds.lat), lat_idx + 2))
                        lon_slice = slice(max(0, lon_idx - 1), min(len(self.sat_ds.lon), lon_idx + 2))
                        
                        sat_patch = self.sat_ds.isel(lat=lat_slice, lon=lon_slice).sel(time=slice(None, yield_time)).tail(time=self.window_size)
                        if len(sat_patch.time) < self.window_size:
                            sat_patch = self.sat_ds.isel(lat=lat_slice, lon=lon_slice).isel(time=slice(0, self.window_size))
                        
                        # Shape: (variable, time, lat, lon) -> transpose to (time, variable, lat, lon)
                        sat_data = sat_patch.to_array().values.transpose(1, 0, 2, 3)
                        # Pad patch if shape is smaller than 3x3 (near edges)
                        if sat_data.shape[2] != 3 or sat_data.shape[3] != 3:
                            padded = np.zeros((self.window_size, self.C, 3, 3), dtype=np.float32)
                            h, w = sat_data.shape[2], sat_data.shape[3]
                            padded[:, :, :h, :w] = sat_data
                            sat_data = padded
                        sat_tensor = torch.tensor(sat_data, dtype=torch.float32)
                        w_data = w_hist.to_array().values.T
                        weather_tensor = torch.tensor(w_data, dtype=torch.float32)
                    else:
                        sat_data = sat_hist.to_array().values.T
                        w_data = w_hist.to_array().values.T
                        X = np.hstack([sat_data, w_data])
                        sat_tensor = torch.tensor(X[:, :self.C], dtype=torch.float32)
                        weather_tensor = torch.tensor(X[:, self.C:], dtype=torch.float32)
                    
                    yield {
                        "sat": sat_tensor,
                        "weather": weather_tensor,
                        "soil": torch.tensor(chunk_soil[i], dtype=torch.float32),
                        "label": torch.tensor([float(row["yield"])], dtype=torch.float32)
                    }
                except Exception as e:
                    logger.debug(f"Skipping sequence due to error: {e}")

    def __len__(self):
        # Approximates length. IterableDatasets don't strictly require this, 
        # but useful for progress bars if no filtering occurs.
        return len(self.yield_df)


