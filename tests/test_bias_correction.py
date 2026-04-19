import pytest
import pandas as pd
import numpy as np
from src.data.bias_correction import DataBiasCorrector

@pytest.fixture
def corrector():
    return DataBiasCorrector({})

@pytest.fixture
def sample_df():
    np.random.seed(42)
    normal_data = np.random.normal(loc=4.0, scale=0.5, size=95).tolist()
    outliers = [0.01, 99.9, 100.0, -5.0, 50.0]
    return pd.DataFrame({"yield": normal_data + outliers})

def test_iqr_detects_outliers(corrector, sample_df):
    """IQR should detect extreme outlier values."""
    lb, ub = corrector.detect_outliers_iqr(sample_df, "yield")
    assert lb < 4.0, "Lower bound should be below the mean"
    assert ub > 4.0, "Upper bound should be above the mean"
    assert ub < 50.0, "Upper bound should catch the extreme outlier 99.9"

def test_boxcox_returns_series(corrector):
    """Box-Cox should return a transformed Series of same length."""
    series = pd.Series(np.random.exponential(scale=2.0, size=50))
    transformed, lmbda = corrector.apply_skew_correction(series)
    assert len(transformed) == len(series)
    assert isinstance(lmbda, float)

def test_boxcox_handles_negatives(corrector):
    """Box-Cox should not crash on series containing zeros or negatives."""
    series = pd.Series([-1.0, 0.0, 1.0, 2.0, 3.0])
    transformed, lmbda = corrector.apply_skew_correction(series)
    assert not transformed.isnull().any()
