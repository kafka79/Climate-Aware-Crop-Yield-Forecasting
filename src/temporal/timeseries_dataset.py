import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import pandas as pd
import xarray as xr
import math
import threading
import queue as _queue
from loguru import logger
from typing import Iterator

def _apply_physical_lags_numpy(w_data: np.ndarray) -> np.ndarray:
    """
    Applies first-principles physical lag models on weather features (tmax, tmin, precip).
    Input w_data: shape (T, C) where C >= 3: [tmax, tmin, precip, ...]
    - Precip is transformed into a soil moisture retention index using an exponential decay model:
      S_t = S_{t-1} * alpha + P_t, where alpha is the soil water retention coefficient (e.g., 0.8)
    - Temperature (tmax/tmin) is smoothed using an accumulated thermal heat proxy.
    """
    T, C = w_data.shape
    if C < 3:
        return w_data
        
    w_transformed = w_data.copy()
    
    # 1. Soil moisture retention index (exponential decay on precipitation)
    precip = w_data[:, 2]
    soil_moisture = np.zeros_like(precip)
    current_sm = 0.0
    alpha = 0.8  # daily soil water retention factor
    for t in range(T):
        current_sm = current_sm * alpha + precip[t]
        soil_moisture[t] = current_sm
    w_transformed[:, 2] = soil_moisture
    
    # 2. Accumulated growing degree days / smoothed thermal stress (mean of tmax and tmin)
    tmean = 0.5 * (w_data[:, 0] + w_data[:, 1])
    thermal_accumulation = np.zeros_like(tmean)
    current_thermal = 0.0
    beta = 0.9  # heat accumulation factor
    for t in range(T):
        current_thermal = current_thermal * beta + max(0.0, tmean[t] - 10.0) # base 10C for crop development
        thermal_accumulation[t] = current_thermal
    
    w_transformed[:, 0] = thermal_accumulation
    w_transformed[:, 1] = tmean
    
    return w_transformed


class MultiModalCropIterableDataset(IterableDataset):
    """
    PyTorch IterableDataset for multi-modal crop yield prediction.
    Streams sequences lazily from Zarr stores in chunks to prevent OOM errors.
    Uses background-thread prefetching to overlap disk I/O with GPU compute.
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

    def _prefetch_chunk(self, chunk_df, out_queue):
        """Load a chunk from Zarr in a background thread so the main thread can yield samples concurrently."""
        try:
            lats = xr.DataArray(chunk_df["lat"].values, dims="location")
            lons = xr.DataArray(chunk_df["lon"].values, dims="location")
            # ponytail: this .load() now runs in a background thread, overlapping I/O with GPU compute
            sat_pixels = self.sat_ds.sel(lat=lats, lon=lons, method="nearest").load()
            weather_pixels = self.weather_ds.sel(lat=lats, lon=lons, method="nearest").load()
            out_queue.put((sat_pixels, weather_pixels))
        except Exception as e:
            out_queue.put(e)

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

        chunk_boundaries = list(range(0, len(worker_df), self.chunk_size))

        # Determine prefetch queue depth from config
        prefetch_depth = int(self.config.get("training", {}).get("prefetch_queue_depth", 4))
        prefetch_q = _queue.Queue(maxsize=prefetch_depth)

        # Persistent background thread loading chunks sequentially
        def producer_worker():
            for c_start in chunk_boundaries:
                c_df = worker_df.iloc[c_start:c_start + self.chunk_size]
                self._prefetch_chunk(c_df, prefetch_q)
        
        t = threading.Thread(target=producer_worker, daemon=True)
        t.start()

        for ci, chunk_start in enumerate(chunk_boundaries):
            chunk_df = worker_df.iloc[chunk_start:chunk_start + self.chunk_size]
            chunk_soil = worker_soil[chunk_start:chunk_start + self.chunk_size]

            # Wait for prefetched data
            result = prefetch_q.get()
            if isinstance(result, Exception):
                raise result
            sat_pixels, weather_pixels = result

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
                        if self.config.get("use_physical_lags", True):
                            w_data = _apply_physical_lags_numpy(w_data)
                            
                        weather_tensor = torch.tensor(w_data, dtype=torch.float32)
                    else:
                        sat_data = sat_hist.to_array().values.T
                        w_data = w_hist.to_array().values.T
                        if self.config.get("use_physical_lags", True):
                            w_data = _apply_physical_lags_numpy(w_data)
                            
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


