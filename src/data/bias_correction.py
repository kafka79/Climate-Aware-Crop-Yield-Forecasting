import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger
from typing import Dict, Any, List

class DataBiasCorrector:
    """
    Handles distribution skewness and class imbalance in historical records.
    Crucial for correcting bias in sparse agricultural datasets.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str, factor: float = 1.5):
        """
        Detect and optionally cap outliers using Interquartile Range (IQR).
        """
        logger.info(f"Detecting outliers in column: {column} (factor={factor})")
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        logger.debug(f"Found {len(outliers)} outliers in {column}.")
        return lower_bound, upper_bound

    def apply_skew_correction(self, series: pd.Series):
        """
        Apply Box-Cox transformation to correct skewness in yield/weather data.
        Returns the transformed series and the estimated lambda.
        """
        # Ensure positive values for Box-Cox
        min_val = series.min()
        shift = 0
        if min_val <= 0:
            shift = abs(min_val) + 1e-6
            
        logger.info(f"Applying Box-Cox transformation to series: {series.name} (shift={shift})")
        transformed_series, lmbda = stats.boxcox(series + shift)
        return pd.Series(transformed_series, index=series.index), lmbda
    
    def balance_risk_classes(self, df: pd.DataFrame, target_col: str):
        """
        Simple oversampling for minority risk classes to address yield imbalance.
        """
        counts = df[target_col].value_counts()
        max_count = counts.max()
        
        logger.info(f"Balancing classes in {target_col}. Max count: {max_count}")
        
        balanced_df = df.copy()
        for cls, count in counts.items():
            if count < max_count:
                # Simple random oversampling
                minority_samples = df[df[target_col] == cls]
                replicated = minority_samples.sample(n=max_count - count, replace=True, random_state=42)
                balanced_df = pd.concat([balanced_df, replicated], axis=0)
        
        logger.success(f"Class balancing complete. New total samples: {len(balanced_df)}")
        return balanced_df

def correct_data_bias(df: pd.DataFrame, config: Dict[str, Any], columns_to_fix: List[str]):
    """
    Runner for the bias correction pipeline.
    """
    corrector = DataBiasCorrector(config)
    
    for col in columns_to_fix:
        # 1. Outlier handling
        lb, ub = corrector.detect_outliers_iqr(df, col)
        # Optional: Clip values instead of dropping
        df[col] = df[col].clip(lb, ub)
        
        # 2. Skew correction if specified
        if config.get("preprocessing", {}).get("fix_skew", True):
            df[col], _ = corrector.apply_skew_correction(df[col])
            
    logger.success("Bias correction phase complete.")
    return df
