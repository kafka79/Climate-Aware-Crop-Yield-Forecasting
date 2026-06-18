import xarray as xr
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, Tuple, Generator


class MultiModalFuser:
    """
    Orchestrates spatial-temporal alignment of Satellite, Weather, and Yield data.
    Uses Zarr/Dask for lazy loading of massive datasets.

    FIX: yield dates (e.g. Dec 31) that fall OUTSIDE the satellite time range are
    no longer dropped. Instead we use the closest available window ending at or
    before the yield date, falling back to the earliest available window when the
    yield date precedes all satellite observations.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get("training", {}).get("window_size", 12)

    def generate_lazy_sequences(
        self,
        yield_df: pd.DataFrame,
        sat_ds: xr.Dataset,
        weather_ds: xr.Dataset,
        chunk_size: int = 1000,
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Yields (X, y) one sequence at a time, loading Zarr data in memory-safe chunks.

        Temporal alignment strategy (robust to date mismatches):
        1. Slice everything up-to-and-including the yield date.
        2. If that slice has fewer than window_size steps, take the FIRST
           window_size observations instead (handles yield dates before sat range).
        3. If the dataset has fewer than window_size observations total, skip.
        """
        logger.info("Initialising lazy multi-modal fusion generator...")

        sat_times = pd.DatetimeIndex(sat_ds.time.values)
        total_sat = len(sat_times)

        if total_sat < self.window_size:
            logger.error(
                f"Satellite dataset has only {total_sat} time steps — "
                f"need at least {self.window_size}. Aborting."
            )
            return

        # Process in chunks to balance memory overhead and disk I/O
        for chunk_start in range(0, len(yield_df), chunk_size):
            chunk_df = yield_df.iloc[chunk_start : chunk_start + chunk_size]
            
            lats = xr.DataArray(chunk_df["lat"].values, dims="location")
            lons = xr.DataArray(chunk_df["lon"].values, dims="location")
            
            logger.info(f"Loading spatial data chunk ({chunk_start} to {chunk_start + len(chunk_df)})...")
            sat_pixels = sat_ds.sel(lat=lats, lon=lons, method="nearest").load()
            weather_pixels = weather_ds.sel(lat=lats, lon=lons, method="nearest").load()

            for i, (_, row) in enumerate(chunk_df.iterrows()):
                lat, lon = row["lat"], row["lon"]
                yield_time = pd.to_datetime(row["time"])

                try:
                    sat_pixel = sat_pixels.isel(location=i)
                    weather_pixel = weather_pixels.isel(location=i)

                    # --- Robust temporal window selection ---
                    # Strategy A: latest window_size steps up to yield_time
                    sat_hist = sat_pixel.sel(time=slice(None, yield_time)).tail(
                        time=self.window_size
                    )
                    w_hist = weather_pixel.sel(time=slice(None, yield_time)).tail(
                        time=self.window_size
                    )

                    # Strategy B fallback: if yield date is before satellite coverage,
                    # use the very FIRST window_size steps (phenologically closest season)
                    if len(sat_hist.time) < self.window_size:
                        logger.debug(
                            f"Yield date {yield_time.date()} is before or near start of "
                            f"satellite coverage. Using first {self.window_size} steps as fallback."
                        )
                        sat_hist = sat_pixel.isel(time=slice(0, self.window_size))
                        w_hist = weather_pixel.isel(time=slice(0, self.window_size))

                    # Final check — dataset truly too short
                    if len(sat_hist.time) < self.window_size:
                        logger.warning(
                            f"Skipping {lat},{lon} @ {yield_time.date()}: "
                            f"only {len(sat_hist.time)} steps available."
                        )
                        continue

                    # Trigger compute for this small pixel chunk only
                    if self.config.get("use_spatial_patches", False):
                        # Locate indices in original sat_ds
                        lat_idx = int(np.argmin(np.abs(sat_ds.lat.values - lat)))
                        lon_idx = int(np.argmin(np.abs(sat_ds.lon.values - lon)))
                        lat_slice = slice(max(0, lat_idx - 1), min(len(sat_ds.lat), lat_idx + 2))
                        lon_slice = slice(max(0, lon_idx - 1), min(len(sat_ds.lon), lon_idx + 2))
                        
                        sat_patch = sat_ds.isel(lat=lat_slice, lon=lon_slice).sel(time=slice(None, yield_time)).tail(time=self.window_size)
                        if len(sat_patch.time) < self.window_size:
                            sat_patch = sat_ds.isel(lat=lat_slice, lon=lon_slice).isel(time=slice(0, self.window_size))
                        
                        C_dim = self.config.get("transformer", {}).get("input_dim", 5)
                        sat_data = sat_patch.to_array().values.transpose(1, 0, 2, 3)
                        if sat_data.shape[2] != 3 or sat_data.shape[3] != 3:
                            padded = np.zeros((self.window_size, C_dim, 3, 3), dtype=np.float32)
                            h, w = sat_data.shape[2], sat_data.shape[3]
                            padded[:, :, :h, :w] = sat_data
                            sat_data = padded
                        w_data = w_hist.to_array().values.T
                        yield (sat_data, w_data), float(row["yield"])
                    else:
                        sat_data = sat_hist.to_array().values.T      # (T, F_sat)
                        w_data = w_hist.to_array().values.T           # (T, F_weather)
                        X = np.hstack([sat_data, w_data])             # (T, F_total)
                        yield X, float(row["yield"])

                except Exception as e:
                    logger.error(f"Failed to fuse {lat},{lon} @ {yield_time}: {e}")


def prepare_training_sequences(
    yield_df: pd.DataFrame,
    sat_ds: xr.Dataset,
    weather_ds: xr.Dataset,
    config: dict,
):
    """Lazy wrapper — collects generated sequences into arrays."""
    fuser = MultiModalFuser(config)
    X_list, y_list = [], []

    for X, y in fuser.generate_lazy_sequences(yield_df, sat_ds, weather_ds):
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        logger.warning("No valid sequences generated.")
        return None, None

    logger.success(f"Prepared {len(X_list)} sequences for training.")
    return np.array(X_list), np.array(y_list)
