import pytest
import pandas as pd
import numpy as np
from src.data.imputation import RuralDataImputer

@pytest.fixture
def imputer():
    return RuralDataImputer({})

def test_temporal_impute_fills_nans(imputer):
    """Linear interpolation should fill NaN gaps in a time series."""
    series = pd.Series([1.0, np.nan, np.nan, 4.0], name="precip")
    result = imputer.temporal_impute(series, method="linear")
    assert result.isnull().sum() == 0, "Should have no NaN after imputation"
    assert np.isclose(result.iloc[1], 2.0, atol=0.1), "Interpolated value should be ~2.0"
    assert np.isclose(result.iloc[2], 3.0, atol=0.1), "Interpolated value should be ~3.0"

def test_temporal_impute_no_op_if_clean(imputer):
    """Should return identical series if no NaN values present."""
    series = pd.Series([1.0, 2.0, 3.0, 4.0], name="gdd")
    result = imputer.temporal_impute(series)
    pd.testing.assert_series_equal(result, series)

def test_tabular_imputation(imputer):
    """KNN imputation should fill NaN in tabular data without crashing."""
    df = pd.DataFrame({
        "ph": [6.5, np.nan, 7.0, 6.8],
        "soc": [1.2, 1.5, np.nan, 1.3]
    })
    result = imputer.impute_tabular(df, ["ph", "soc"])
    assert result.isnull().sum().sum() == 0, "All NaN values should be filled"
