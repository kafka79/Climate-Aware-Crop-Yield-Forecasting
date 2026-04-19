import xarray as xr
import pandas as pd
import numpy as np
from loguru import logger
from typing import Dict, Any, List, Tuple
from src.temporal.sequence_builder import SequenceBuilder

class MultiModalFuser:
    """
    Orchestrates the spatial-temporal alignment of Satellite, Weather, and Yield data.
    Ensures that for every Yield observation (Space-Time), we have the corresponding
    history of Sentinel and ERA5 features.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def fuse_by_location(self, yield_df: pd.DataFrame, sat_ds: xr.Dataset, weather_df: pd.DataFrame):
        """
        Calculates the aligned feature matrix for yield prediction.
        """
        logger.info("Fusing multi-modal datasets by spatial-temporal keys...")
        
        fused_records = []
        
        for _, row in yield_df.iterrows():
            lat, lon = row['lat'], row['lon']
            time = pd.to_datetime(row['time'])
            
            # 1. Spatial Sampling for Satellite (Nearest Neighbor)
            try:
                sat_pixel = sat_ds.sel(lat=lat, lon=lon, method="nearest")
                # Sample the history before the yield (e.g., 6 months prior)
                sat_history = sat_pixel.sel(time=slice(time - pd.DateOffset(months=6), time))
                
                # 2. Spatial Sampling for Weather
                weather_pixel = weather_df.sel(lat=lat, lon=lon, method="nearest")
                weather_history = weather_pixel.sel(time=slice(time - pd.DateOffset(months=6), time))
                
                # 3. Convert to Tabular for Sequence Builder
                sat_df = sat_history.to_dataframe().drop(columns=['lat', 'lon'], errors='ignore')
                weather_df_pix = weather_history.to_dataframe().drop(columns=['lat', 'lon'], errors='ignore')
                
                # Align on time
                combined = pd.merge(sat_df, weather_df_pix, left_index=True, right_index=True)
                combined['yield'] = row['yield']
                combined['site_id'] = row.get('site_id', 0)
                
                fused_records.append(combined)
            except Exception as e:
                logger.error(f"Failed to fuse data for record {lat}, {lon} at {time}: {e}")
                
        if not fused_records:
            return pd.DataFrame()
            
        return pd.concat(fused_records)

def prepare_training_sequences(yield_df: pd.DataFrame, sat_ds: xr.Dataset, weather_df: pd.DataFrame, config: dict):
    """
    Orchestrates fusion and sequence building for Transformer input.
    """
    fuser = MultiModalFuser(config)
    fused_df = fuser.fuse_by_location(yield_df, sat_ds, weather_df)
    
    if fused_df.empty:
        logger.warning("Fused dataframe is empty. Cannot build sequences.")
        return None, None
        
    window_size = config.get('training', {}).get('window_size', 12)
    builder = SequenceBuilder(window_size=window_size)
    
    X, y = builder.create_sequences(fused_df)
    
    logger.success(f"Prepared {len(X)} sequences for training.")
    return X, y
