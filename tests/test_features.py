import pytest
import numpy as np
import pandas as pd
from src.features.ndvi import calculate_ndvi, calculate_evi
from src.features.weather_features import calculate_gdd

def test_calculate_ndvi():
    red = np.array([0.1, 0.4])
    nir = np.array([0.5, 0.2])
    
    ndvi = calculate_ndvi(red, nir)
    
    # (0.5 - 0.1) / (0.5 + 0.1) = 0.4 / 0.6 = 0.6667
    assert np.isclose(ndvi[0], 0.666666, atol=1e-5)
    
    # (0.2 - 0.4) / (0.2 + 0.4) = -0.2 / 0.6 = -0.3333
    assert np.isclose(ndvi[1], -0.333333, atol=1e-5)

def test_calculate_gdd():
    t_max = np.array([30.0, 40.0, 15.0])
    t_min = np.array([20.0, 25.0, 5.0])
    
    # Base=10, Cap=35
    # 1: (30 + 20)/2 - 10 = 25/2 - 10 = 15? No, (30+20)/2 = 25; 25 - 10 = 15.
    # 2: tmax clipped to 35. (35 + 25)/2 = 30. 30 - 10 = 20.
    # 3: tmin clipped to 10. (15 + 10)/2 = 12.5. 12.5 - 10 = 2.5
    
    gdds = calculate_gdd(t_max, t_min, base_temp=10.0, cap_temp=35.0)
    
    assert np.isclose(gdds[0], 15.0)
    assert np.isclose(gdds[1], 20.0)
    assert np.isclose(gdds[2], 2.5)
