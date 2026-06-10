import os
import pytest
import numpy as np
import pandas as pd
import torch
import xarray as xr
from unittest.mock import MagicMock, patch

from src.recommendation.engine import RecommendationEngine
from src.schema.validators import align_and_validate_soil
from src.explainability.integrated_gradients import explain_prediction
from src.data.preprocessing import DataPreprocessor


def test_flaw_a_heuristic_risk_matching():
    # Setup mock result with "High Risk" (title-cased)
    result = {
        "region": "Burdwan, West Bengal",
        "predicted_yield": 3.5,
        "lower_bound": 3.0,
        "upper_bound": 4.0,
        "risk": "High Risk",
        "attribution": {"Weather": 0.6, "Satellite": 0.2, "Soil": 0.2}
    }
    
    engine = RecommendationEngine({"paths": {"raw": {"soil": "data/raw/soil"}}})
    advice = engine.generate_advice(result)
    
    # Assert that emergency action heuristic actually got triggered (High Risk match worked)
    assert any("🚨 **Emergency Action:**" in item for item in advice)


def test_flaw_b_llm_advice_missing_region():
    # Result missing "region" key
    result = {
        "predicted_yield": 3.5,
        "lower_bound": 3.0,
        "upper_bound": 4.0,
        "risk": "Low Risk",
        "attribution": {"Weather": 0.6, "Satellite": 0.2, "Soil": 0.2}
    }
    
    engine = RecommendationEngine({"paths": {"raw": {"soil": "data/raw/soil"}}})
    # Mock self.model to simulate LLM execution without network
    engine.model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "* Bullet 1\n* Bullet 2"
    engine.model.generate_content.return_value = mock_response
    
    advice = engine.generate_advice(result)
    # The generation prompt should run without raising KeyError: 'region'
    assert advice == ["Bullet 1", "Bullet 2"]
    engine.model.generate_content.assert_called_once()


def test_flaw_c_temporal_gaps_no_unphysical_zeros():
    # Setup satellite dataset with NaNs (masked cloud cover)
    times = pd.date_range("2023-01-01", periods=5)
    ds = xr.Dataset(
        data_vars={
            "B04": (("time", "lat", "lon"), [[[np.nan]], [[0.4]], [[np.nan]], [[0.5]], [[np.nan]]])
        },
        coords={"time": times, "lat": [23.0], "lon": [87.0]}
    )
    
    preprocessor = DataPreprocessor({"paths": {"raw": "", "processed": ""}})
    filled_ds = preprocessor.fill_temporal_gaps(ds)
    
    # Assert there are no NaN values and no unphysical 0.0 values where ffill/bfill could fill
    b4_values = filled_ds["B04"].values.flatten()
    assert not np.isnan(b4_values).any()
    # Zeros should not be generated since ffill/bfill will carry 0.4 and 0.5 to fill the gaps
    assert 0.0 not in b4_values
    assert abs(b4_values[0] - 0.4) < 1e-4  # extrapolated nearest backward
    assert abs(b4_values[2] - 0.45) < 1e-4  # interpolated linear in between
    assert abs(b4_values[4] - 0.5) < 1e-4  # extrapolated nearest forward


def test_flaw_f_soil_dimension_validation_fail_loud():
    # Validate dataframe with 3 columns against soil_dim = 4 should raise ValueError
    soil_df = pd.DataFrame([{"ph": 6.5, "soc": 12.0, "nitrogen": 1.2}])
    
    with pytest.raises(ValueError) as exc:
        align_and_validate_soil(soil_df, soil_dim=4)
        
    assert "Dimension mismatch" in str(exc.value)


def test_flaw_e_integrated_gradients_non_zero_baselines():
    # Setup model mock that returns pi, sigma, mu
    model = MagicMock()
    # Let's say model output is (pi, sigma, mu)
    # pi shape (1, 5), sigma (1, 5, 1), mu (1, 5, 1)
    # But since we explain on actual tensors, we just mock IntegratedGradients class call or check our baseline construction
    from src.explainability.integrated_gradients import YieldExplainer
    
    explainer = YieldExplainer(model)
    
    sat = torch.randn(1, 12, 5)
    weather = torch.randn(1, 12, 3)
    soil = torch.randn(1, 3)
    
    with patch("src.explainability.integrated_gradients.IntegratedGradients.attribute") as mock_attribute:
        mock_attribute.return_value = (torch.zeros_like(sat), torch.zeros_like(weather), torch.zeros_like(soil))
        
        explainer.calculate_attributions(sat, weather, soil)
        
        # Verify that baselines are passed as non-zero tensors
        called_args, called_kwargs = mock_attribute.call_args
        baselines = called_kwargs.get("baselines")
        
        assert baselines is not None
        sat_base, weather_base, soil_base = baselines
        
        # Baselines must not be all zeros
        assert not torch.all(sat_base == 0.0)
        assert not torch.all(weather_base == 0.0)
        assert not torch.all(soil_base == 0.0)
        
        # Soil base should match the default pH ~6.5, SOC ~10.0, N ~1.5
        assert abs(soil_base[0, 0].item() - 6.5) < 1e-4
        assert abs(soil_base[0, 1].item() - 10.0) < 1e-4
        assert abs(soil_base[0, 2].item() - 1.5) < 1e-4


def test_telemetry_tracker_logs_json():
    from src.utils.telemetry import TelemetryTracker
    import json
    
    with patch("src.utils.telemetry.logger.info") as mock_log_info:
        with TelemetryTracker("test_span") as tracker:
            tracker.set_attribute("key", "value")
        
        # Verify logger.info was called with a TELEMETRY_JSON prefix
        called = False
        for call in mock_log_info.call_args_list:
            arg = call[0][0]
            if "TELEMETRY_JSON:" in arg:
                called = True
                json_str = arg.split("TELEMETRY_JSON:")[1].strip()
                payload = json.loads(json_str)
                assert payload["span_name"] == "test_span"
                assert payload["attributes"]["key"] == "value"
                assert "latency_ms" in payload
                assert "trace_id" in payload
        assert called


def test_telemetry_metric_logging():
    from src.utils.telemetry import log_business_metric
    import json
    
    with patch("src.utils.telemetry.logger.info") as mock_log_info:
        log_business_metric("crop_yield", 4.2, "t/ha", {"region": "test"})
        
        called = False
        for call in mock_log_info.call_args_list:
            arg = call[0][0]
            if "METRIC_JSON:" in arg:
                called = True
                json_str = arg.split("METRIC_JSON:")[1].strip()
                payload = json.loads(json_str)
                assert payload["metric_name"] == "crop_yield"
                assert payload["value"] == 4.2
                assert payload["unit"] == "t/ha"
                assert payload["tags"]["region"] == "test"
        assert called


def test_run_inference_gmm_params():
    from src.inference.runtime import run_inference
    
    # Mock initialize_model to return a MagicMock model
    mock_model = MagicMock()
    # Mock model output: tuple (pi, sigma, mu)
    pi_tensor = torch.tensor([[0.6, 0.4, 0.0, 0.0, 0.0]])
    sigma_tensor = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5]]])
    mu_tensor = torch.tensor([[[3.0], [5.0], [0.0], [0.0], [0.0]]])
    mock_model.return_value = (pi_tensor, sigma_tensor, mu_tensor)
    
    mock_prepared = {
        "sat_tensor": torch.randn(1, 12, 5),
        "weather_tensor": torch.randn(1, 12, 3),
        "soil_tensor": torch.randn(1, 3),
        "soil_source": "mock",
        "ndvi_series": [0.5] * 12,
        "modality_warnings": []
    }
    
    with patch("src.inference.runtime.initialize_model", return_value=mock_model), \
         patch("src.inference.runtime.load_model_weights"), \
         patch("src.inference.runtime._prepare_model_inputs", return_value=mock_prepared), \
         patch("src.inference.runtime.Path.exists", return_value=True), \
         patch("src.inference.runtime.explain_prediction", return_value=({"satellite_overall": 0.5, "weather_overall": 0.3, "soil_overall": 0.2}, None)), \
         patch("src.inference.runtime._get_region_history", return_value=pd.DataFrame(columns=["year", "yield"])):
             
        res = run_inference("Burdwan, West Bengal", 2023)
        assert "gmm_params" in res
        assert res["gmm_params"] is not None
        assert abs(res["gmm_params"]["pi"][0] - 0.6) < 1e-5
        assert abs(res["gmm_params"]["sigma"][1] - 0.2) < 1e-5
        assert abs(res["gmm_params"]["mu"][0] - 3.0) < 1e-5

