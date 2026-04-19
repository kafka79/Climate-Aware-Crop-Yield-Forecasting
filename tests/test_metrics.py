import pytest
import numpy as np
from src.evaluation.metrics import YieldMetrics

def test_yield_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([9.0, 21.0, 30.0])
    
    metrics_engine = YieldMetrics({})
    results = metrics_engine.calculate_all(y_true, y_pred)
    
    assert "MAE" in results
    assert "RMSE" in results
    assert "R2" in results
    
    # Total absolute error = 1 + 1 + 0 = 2. MAE = 2/3 = 0.6667
    assert np.isclose(results["MAE"], 0.6667, atol=1e-3)
    
    # Mean bias = ((9-10) + (21-20) + (30-30)) / 3 = 0
    assert np.isclose(results["MeanBias"], 0.0)
