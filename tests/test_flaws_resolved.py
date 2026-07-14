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
    # LLM returns valid JSON array of recommendations (matching new structured output format)
    mock_response.text = '["Apply deficit irrigation to conserve water during dry spells", "Monitor NDVI trends weekly for early stress detection"]'
    engine.model.generate_content.return_value = mock_response
    
    advice = engine.generate_advice(result)
    # The generation prompt should run without raising KeyError: 'region'
    # First item is the safety disclaimer, followed by the validated recommendations
    assert len(advice) >= 2
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


def test_weather_spi_clipping():
    from src.features.weather_features import calculate_spi
    # Zero precip yields CDF=0, which would normally map to -inf
    precip = pd.Series([0.0] * 12)
    spi = calculate_spi(precip, scale=3)
    
    # Verify no infinite or NaN values are present
    assert not np.isinf(spi).any()
    assert not spi.isnull().any()
    # Check that it clipped at standard normal values
    assert spi.min() > -5.0


def test_fuser_chunked_lazy_loading():
    from src.data.fusion import MultiModalFuser
    # Mock datasets
    times = pd.date_range("2023-01-01", periods=15)
    sat_ds = xr.Dataset(
        data_vars={
            "B04": (("time", "lat", "lon"), np.random.rand(15, 2, 2)),
            "B08": (("time", "lat", "lon"), np.random.rand(15, 2, 2)),
        },
        coords={"time": times, "lat": [23.0, 24.0], "lon": [87.0, 88.0]}
    )
    weather_ds = xr.Dataset(
        data_vars={
            "t2m": (("time", "lat", "lon"), np.random.rand(15, 2, 2))
        },
        coords={"time": times, "lat": [23.0, 24.0], "lon": [87.0, 88.0]}
    )
    yield_df = pd.DataFrame([
        {"lat": 23.0, "lon": 87.0, "time": "2023-12-31", "yield": 4.5},
        {"lat": 24.0, "lon": 88.0, "time": "2023-12-31", "yield": 5.2}
    ])
    
    config = {
        "training": {"window_size": 12},
        "transformer": {"input_dim": 2, "temporal_dim": 1}
    }
    
    fuser = MultiModalFuser(config)
    sequences = list(fuser.generate_lazy_sequences(yield_df, sat_ds, weather_ds, chunk_size=1))
    
    # We chunked with size=1. We should successfully yield 2 sequences without OOM or ValueError
    assert len(sequences) == 2
    X, y = sequences[0]
    assert X.shape == (12, 3)  # window_size=12, features = 2 (sat) + 1 (weather)
    assert abs(y - 4.5) < 1e-4


def test_sentinel_downloader_writes_to_disk(tmp_path):
    from src.data.downloader import SentinelHubDownloader
    
    config = {
        "sentinel_hub": {
            "client_id": "test_id",
            "client_secret": "test_secret"
        },
        "paths": {
            "raw": {
                "sentinel2": str(tmp_path)
            }
        }
    }
    
    downloader = SentinelHubDownloader(config)
    
    # Mock client and download return value containing dummy bytes
    mock_client = MagicMock()
    mock_client.download.return_value = [b"TIFF_HEADER_AND_BYTES"]
    
    with patch("src.data.downloader.SentinelHubDownloadClient", return_value=mock_client):
        downloader.download([77.0, 28.0, 78.0, 29.0], ("2023-01-01", "2023-01-31"), "test_area")
        
    expected_file = tmp_path / "test_area.tiff"
    assert expected_file.exists()
    assert expected_file.read_bytes() == b"TIFF_HEADER_AND_BYTES"


def test_sagemaker_no_wait():
    import sys
    mock_boto3 = MagicMock()
    sys.modules["boto3"] = mock_boto3
    mock_sm = MagicMock()
    mock_boto3.client.return_value = mock_sm
    
    from src.training.sagemaker_launcher import launch_sagemaker_training
    with patch("src.training.sagemaker_launcher._package_and_upload_code", return_value="s3://test/code.tar.gz"):
        res = launch_sagemaker_training(
            s3_bucket="test-bucket",
            s3_features_prefix="features",
            s3_output_prefix="output",
            role_arn="arn:aws:iam::123456789012:role/service-role/SageMakerRole",
            no_wait=True
        )
        assert res["TrainingJobStatus"] == "InProgress"
        mock_sm.create_training_job.assert_called_once()
        mock_sm.describe_training_job.assert_not_called()


def test_soil_downloader_fallback_physical_defaults(tmp_path):
    from src.data.downloader import SoilDownloader
    config = {
        "paths": {
            "raw": {
                "soil": str(tmp_path)
            }
        }
    }
    downloader = SoilDownloader(config)
    with patch("requests.get") as mock_get:
        mock_get.side_effect = Exception("API Timeout")
        res = downloader.download([77.0, 28.0, 78.0, 29.0], "test_region")
        assert res == {"ph": 6.5, "soc": 10.0, "nitrogen": 1.5}
        
    expected_file = tmp_path / "test_region_soil.csv"
    assert expected_file.exists()
    df = pd.read_csv(expected_file)
    assert df.loc[0, "ph"] == 6.5


def test_safe_cache_serializer_validation():
    from src.inference.runtime import SafeCacheSerializer
    import pickle
    import pytest
    import os
    
    class MaliciousPayload:
        def __reduce__(self):
            return (os.system, ('echo hacked',))
    
    # 1. Test that MaliciousPayload fails deserialization
    malicious_bytes = pickle.dumps(MaliciousPayload())
    
    with pytest.raises(pickle.UnpicklingError, match="forbidden"):
        SafeCacheSerializer.deserialize(malicious_bytes)
        
    # 2. Test that a normal payload succeeds
    import torch
    safe_data = {"data": torch.tensor([1, 2, 3])}
    safe_bytes = SafeCacheSerializer.serialize(safe_data)
    deserialized = SafeCacheSerializer.deserialize(safe_bytes)
    assert "data" in deserialized
    assert torch.equal(deserialized["data"], safe_data["data"])


def test_trainer_sync_termination_flag_no_op():
    from src.training.trainer import TrainManager
    
    model = MagicMock()
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    config = {
        "training": {
            "learning_rate": 1e-3,
            "mode": "deterministic",
            "device": "cpu"
        }
    }
    
    trainer = TrainManager(model, config)
    
    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.all_reduce") as mock_reduce:
            # _sync_termination_flag should be a no-op and not perform DDP all_reduce
            trainer._sync_termination_flag()
            mock_reduce.assert_not_called()


def test_transformer_spatial_patch_input():
    from src.models.transformer import initialize_model
    config = {
        "transformer": {
            "input_dim": 5,
            "temporal_dim": 3,
            "soil_dim": 3,
            "hidden_dim": 64,
            "num_heads": 2,
            "num_layers": 1,
            "dropout": 0.1,
            "use_jitter": False
        },
        "mdn": {
            "num_mixtures": 3,
            "output_dim": 1
        }
    }
    model = initialize_model(config)
    
    # 5D input tensor: (B, T, C, H, W)
    # B=2, T=12, C=5, H=3, W=3
    sat = torch.randn(2, 12, 5, 3, 3)
    weather = torch.randn(2, 12, 3)
    soil = torch.randn(2, 3)
    
    pi, sigma, mu = model(sat, weather, soil)
    assert pi.shape == (2, 3)
    assert sigma.shape == (2, 3, 1)
    assert mu.shape == (2, 3, 1)


def test_model_cache_eviction():
    from src.inference.runtime import _get_cached_model, _MODEL_CACHE
    from pathlib import Path
    
    _MODEL_CACHE.clear()
    
    mock_model = MagicMock()
    with patch("src.inference.runtime.initialize_model", return_value=mock_model), \
         patch("src.inference.runtime.load_model_weights"):
             
        path1 = Path("models/model1.pth")
        path2 = Path("models/model2.pth")
        path3 = Path("models/model3.pth")
        path4 = Path("models/model4.pth")
        
        m1 = _get_cached_model(path1, {})
        m2 = _get_cached_model(path2, {})
        m3 = _get_cached_model(path3, {})
        
        assert len(_MODEL_CACHE) == 3
        # Cache keys now include mtime (path::mtime format)
        cache_keys = list(_MODEL_CACHE.keys())
        assert any(str(path1.resolve()) in k for k in cache_keys)
        
        m4 = _get_cached_model(path4, {})
        assert len(_MODEL_CACHE) == 3
        cache_keys = list(_MODEL_CACHE.keys())
        assert not any(str(path1.resolve()) in k for k in cache_keys)
        assert any(str(path4.resolve()) in k for k in cache_keys)

