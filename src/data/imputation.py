import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from loguru import logger
from typing import Dict, Any, List

class RuralDataImputer:
    """
    Handles missing data specifically for rural and sparse datasets.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
        
    def impute_tabular(self, df: pd.DataFrame, columns: List[str]):
        """
        Impute missing values in tabular data using KNN techniques.
        Optimized for sparse regional data.
        """
        logger.info(f"Imputing missing values in columns: {columns}")
        if df[columns].isnull().sum().sum() == 0:
            logger.debug("No missing values found in tabular data. Skipping.")
            return df
            
        df_imputed = self.knn_imputer.fit_transform(df[columns])
        df[columns] = df_imputed
        return df
    
    def temporal_impute(self, series: pd.Series, method: str = 'linear'):
        """
        Fill gaps in time-series data (e.g., missing weather days) using interpolation.
        """
        if series.isnull().sum() == 0:
            return series
            
        logger.info(f"Interpolating sparse time-series: {series.name} using {method}")
        return series.interpolate(method=method, limit_direction='both')

def impute_data(df: pd.DataFrame, config: Dict[str, Any], target_cols: List[str]):
    """
    Standard wrapper to apply imputation logic to a dataset.
    """
    imputer = RuralDataImputer(config)
    
    # 1. First apply temporal interpolation if it's a time-series
    if isinstance(df.index, pd.DatetimeIndex):
        for col in target_cols:
            df[col] = imputer.temporal_impute(df[col])
            
    # 2. Then apply KNN for any remaining spatial/tabular gaps
    df = imputer.impute_tabular(df, target_cols)
    
    logger.success("Imputation pipeline complete.")
    return df
