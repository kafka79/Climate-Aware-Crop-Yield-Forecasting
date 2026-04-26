import pandas as pd
import numpy as np
from loguru import logger
from typing import List, Tuple

class SequenceBuilder:
    """
    Transforms tabular multi-modal data into temporal sequences for Transformers.
    """
    def __init__(self, window_size: int = 12, step_size: int = 1):
        self.window_size = window_size
        self.step_size = step_size

    def create_sequences(self, data: pd.DataFrame, target: str = "yield"):
        """
        Creates sequences of length `window_size` from a DataFrame.
        """
        logger.info(f"Creating sequences with window_size={self.window_size}...")
        sequences = []
        labels = []
        for i in range(0, len(data) - self.window_size, self.step_size):
            # Causal Windowing Check
            window_df = data.iloc[i : i + self.window_size]
            
            # Ensure target is NOT in features
            if target in window_df.columns:
                window_df = window_df.drop(columns=[target])
            
            window = window_df.values
            label = data.iloc[i + self.window_size][target]
            
            # Final integrity check (no NaN in labels, labels are strictly in the future of the window)
            if pd.isna(label):
                continue
                
            sequences.append(window)
            labels.append(label)
        
        return np.array(sequences), np.array(labels)

    def create_temporal_features(self, df: pd.DataFrame):
        """
        Adds cyclical temporal features (Sin/Cos encoding of Month).
        """
        logger.info("Encoding cyclical temporal features (Month)...")
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        return df

def build_lag_features(df: pd.DataFrame, lags: List[int], column: str):
    """
    Create historical lag features for a specific column.
    """
    logger.info(f"Building lag features for {column} with lags={lags}...")
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df
def validate_temporal_integrity(df: pd.DataFrame, time_col: str = "time"):
    """
    Validates that the dataframe is sorted by time and contains no duplicate timestamps
    per region—preventing non-causal data shuffling.
    """
    logger.info("Validating temporal integrity (Causality check)...")
    if not df[time_col].is_monotonic_increasing:
        raise ValueError("Data is not temporally sorted. Sequence building will lead to data leakage.")
    
    if df.duplicated(subset=[time_col]).any():
        logger.warning("Duplicate timestamps detected. Ensure data is grouped by location first.")
    
    return True
