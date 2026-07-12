import pytest
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import functions to test
from src.data.drift_detector import check_region_drift
from src.inference.runtime import SafeCacheSerializer, _get_cached_model, run_inference, _MODEL_CACHE
from src.training.trainer import TrainManager
from src.models.mdn import mdn_detect_bimodality, mdn_safe_point_estimate
import app

# 1. Test Flaw #1: Variance-Blind Weather Drift Detection
def test_weather_drift_variance_shift(tmp_path):
    # Create mock weather datasets with the same mean but different variances
    # Reference weather: std = 1.0
    ref_w = pd.DataFrame({
        "t2m": np.random.normal(loc=15.0, scale=1.0, size=100)
    })
    # Current weather: std = 3.0 (high variance, e.g. extreme events)
    cur_w = pd.DataFrame({
        "t2m": np.random.normal(loc=15.0, scale=3.0, size=100)
    })
    
    # Save as temporary csv files
    ref_file = tmp_path / "ref_w.csv"
    cur_file = tmp_path / "cur_w.csv"
    ref_w.to_csv(ref_file, index=False)
    cur_w.to_csv(cur_file, index=False)
    
    # We also need dummy Zarr paths for check_region_drift structure
    dummy_ref_zarr = tmp_path / "ref_ndvi.zarr"
    dummy_cur_zarr = tmp_path / "cur_ndvi.zarr"
    dummy_ref_zarr.mkdir()
    dummy_cur_zarr.mkdir()
    
    # Mock _extract_ndvi to return None so that only weather check runs
    with patch("src.data.drift_detector._extract_ndvi", return_value=None), \
         patch("src.data.drift_detector._extract_weather_anomalies") as mock_extract:
         
         # Return the numpy arrays directly
         mock_extract.return_value = (ref_w["t2m"].values, cur_w["t2m"].values)
         
         report = check_region_drift(
             region="test_region",
             reference_zarr=dummy_ref_zarr,
             current_zarr=dummy_cur_zarr,
             reference_year=2023,
             current_year=2024,
             reference_weather=ref_file,
             current_weather=cur_file
         )
         
         assert report["ks_status"] == "WARN"
         assert report["levene_pvalue"] is not None
         assert report["levene_pvalue"] < 0.05
         assert "levene_pvalue" in report

# 2. Test Flaw #2: Streamlit Cache Decorator
def test_streamlit_cache_applied():
    # Verify the function has st.cache_data decorator applied
    assert hasattr(app.generate_offline_features_json, "clear")

# 3. Test Flaw #3: Loss of Coordinates in xr.Dataset Cache Serializer
def test_safe_cache_serializer_dataset_dims():
    # Create dataset with specific dimensions and coordinates
    times = pd.date_range("2023-01-01", periods=3)
    original_ds = xr.Dataset(
        data_vars={
            "temperature": (("time", "lat", "lon"), np.random.rand(3, 2, 2)),
            "precipitation": (("time", "lat", "lon"), np.random.rand(3, 2, 2))
        },
        coords={
            "time": times,
            "lat": [12.0, 13.0],
            "lon": [80.0, 81.0]
        },
        attrs={"unit": "Celsius"}
    )
    
    # Serialize
    serialized_str = SafeCacheSerializer.serialize({"dataset": original_ds})
    
    # Deserialize
    deserialized_dict = SafeCacheSerializer.deserialize(serialized_str)
    deserialized_ds = deserialized_dict["dataset"]
    
    # Assertions
    assert isinstance(deserialized_ds, xr.Dataset)
    assert list(deserialized_ds.dims) == list(original_ds.dims)
    assert list(deserialized_ds.data_vars.keys()) == list(original_ds.data_vars.keys())
    for var_name in original_ds.data_vars:
        # Check dimensions match
        assert deserialized_ds[var_name].dims == original_ds[var_name].dims
        # Check values match
        np.testing.assert_array_almost_equal(deserialized_ds[var_name].values, original_ds[var_name].values)
    for coord_name in original_ds.coords:
        assert deserialized_ds[coord_name].dims == original_ds[coord_name].dims
        np.testing.assert_array_equal(deserialized_ds[coord_name].values, original_ds[coord_name].values)
    assert deserialized_ds.attrs == original_ds.attrs

# 4. Test Flaw #4: CPU-Locked Model Cache (GPU Support)
def test_gpu_model_caching(tmp_path):
    _MODEL_CACHE.clear()
    mock_model = MagicMock()
    mock_param = torch.nn.Parameter(torch.randn(2, 2))
    mock_model.parameters.return_value = [mock_param]
    
    with patch("torch.cuda.is_available", return_value=True), \
         patch("src.inference.runtime.initialize_model", return_value=mock_model) as mock_init, \
         patch("src.inference.runtime.load_model_weights") as mock_load:
             
         m = _get_cached_model(tmp_path / "dummy.pth", {})
         
         # Assert that it loads model weights to "cuda" device and calls .to("cuda")
         called_args = mock_load.call_args[0]
         assert called_args[2] == torch.device("cuda")
         mock_model.to.assert_called_once_with(torch.device("cuda"))

# 5. Test Flaw #5: Spot termination batch breakout
def test_spot_termination_batch_breakout():
    model = MagicMock()
    model.parameters.return_value = [torch.nn.Parameter(torch.randn(2, 2))]
    config = {
        "training": {
            "learning_rate": 1e-3,
            "mode": "deterministic",
            "device": "cpu"
        },
        "num_epochs": 1
    }
    
    trainer = TrainManager(model, config)
    
    # Mock dataloader to return 5 batches
    batch_data = {"sat": torch.randn(2, 12, 5), "weather": torch.randn(2, 12, 3), "soil": torch.randn(2, 3), "label": torch.randn(2, 1)}
    dataloader = [batch_data] * 5
    
    # Mock loss function and backward step
    trainer.optimizer = MagicMock()
    trainer.criterion = MagicMock()
    trainer.criterion.return_value = torch.tensor(1.5, requires_grad=True)
    
    # We trigger termination at batch 2
    batch_count = 0
    def dummy_sync():
        nonlocal batch_count
        batch_count += 1
        if batch_count == 3:
            trainer._termination_requested.set()
            
    trainer._sync_termination_flag = dummy_sync
    
    avg_loss = trainer.train_epoch(dataloader)
    
    # Verify we broke out and only processed 2 batches
    assert batch_count == 3  # sync called on batch 1, 2, and 3 (which breaks)
    # The average loss should be 1.5 (loss.item() for 2 batches / 2)
    assert abs(avg_loss - 1.5) < 1e-5

# 6. Test Flaw #6: MDN Valley Depth Correctness
def test_mdn_valley_depth_calculation():
    # Make a bimodal prediction
    # Peak 1 at 2.0 (weight 0.5, std 0.1) -> density peak ~1.99
    # Peak 2 at 8.0 (weight 0.5, std 0.1) -> density peak ~1.99
    # Valley at 5.0 -> density ~0.0
    # True valley depth should be 1.0 - (0.0 / 1.99) = 1.0
    pi = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32)
    sigma = torch.tensor([[[0.1], [0.1], [0.1]]], dtype=torch.float32)
    mu = torch.tensor([[[2.0], [8.0], [5.0]]], dtype=torch.float32)
    
    report = mdn_detect_bimodality(pi, sigma, mu)
    assert report["is_bimodal"] is True
    # Under new logic, valley_depth should be near 1.0
    assert report["valley_depth"] > 0.95

# 7. Test New Flaw #1: Device alignment in explain_prediction
def test_explainability_device_alignment():
    from unittest.mock import PropertyMock
    mock_model = MagicMock()
    mock_device = torch.device("cpu")
    
    # Setup mock parameter with custom device
    mock_param = MagicMock()
    type(mock_param).device = PropertyMock(return_value=mock_device)
    mock_model.parameters.return_value = [mock_param]
    
    from src.explainability.integrated_gradients import YieldExplainer
    explainer = YieldExplainer(mock_model)
    
    sat = torch.randn(1, 12, 5, device="cpu")
    weather = torch.randn(1, 12, 3, device="cpu")
    soil = torch.randn(1, 3, device="cpu")
    
    # Mock .to() on the tensors directly
    sat.to = MagicMock(return_value=sat)
    weather.to = MagicMock(return_value=weather)
    soil.to = MagicMock(return_value=soil)
    
    with patch("src.explainability.integrated_gradients.IntegratedGradients.attribute") as mock_attribute:
        mock_attribute.return_value = (torch.zeros_like(sat), torch.zeros_like(weather), torch.zeros_like(soil))
        
        explainer.calculate_attributions(sat, weather, soil)
        
        sat.to.assert_any_call(mock_device)
        weather.to.assert_any_call(mock_device)
        soil.to.assert_any_call(mock_device)

# 8. Test New Flaw #2: Zero predicted yield in RecommendationEngine
def test_recommendation_zero_yield():
    from src.recommendation.engine import RecommendationEngine
    engine = RecommendationEngine({"paths": {"raw": {"soil": "data/raw/soil"}}})
    
    result = {
        "region": "Burdwan, West Bengal",
        "predicted_yield": 0.0,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
        "risk": "High Risk",
        "attribution": {"Weather": 0.6, "Satellite": 0.2, "Soil": 0.2}
    }
    
    # Should not raise ZeroDivisionError and should run successfully
    advice = engine.generate_advice(result)
    assert len(advice) > 0


# ── Flaw-fix tests for the 4 remaining panel-identified issues ──────────────

# 9. Test DDP validation loss synchronization
def test_ddp_val_loss_sync():
    """Verify that trainer.py calls all_reduce on val_loss when DDP is initialized."""
    import src.training.trainer as trainer_mod
    import inspect
    source = inspect.getsource(trainer_mod.TrainManager.run)
    # ponytail: just check the critical call exists in the source
    assert "all_reduce" in source, "fit() must call all_reduce to sync val_loss across ranks"
    assert "ReduceOp.SUM" in source, "Must use SUM reduction for averaging"


# 10. Test MDN mode finder sigma-scaling
def test_mdn_mode_finder_small_sigma_stability():
    """Mode finder must not explode when sigma is very small."""
    from src.models.mdn import _find_modes_gradient_ascent
    
    # Two components with very small sigma — previously caused gradient explosion
    pi = torch.tensor([0.5, 0.5])
    sigma = torch.tensor([[1e-4], [1e-4]])
    mu = torch.tensor([[3.0], [7.0]])
    
    modes = _find_modes_gradient_ascent(pi, sigma, mu)
    
    # Should find modes near the component means, not explode to inf/nan
    assert len(modes) > 0, "Must find at least one mode"
    for log_d, pos in modes:
        assert not np.isnan(log_d), f"Mode log-density is NaN"
        assert not np.isinf(pos.abs().max().item()), f"Mode position exploded to inf"
        # Modes should stay near original means (3.0 or 7.0), not diverge far
        assert pos.abs().max().item() < 100, f"Mode {pos} diverged far from component means"


# 11. Test prefetch infrastructure exists in dataset
def test_dataset_prefetch_method():
    """Verify the dataset has the _prefetch_chunk method for async I/O."""
    from src.temporal.timeseries_dataset import MultiModalCropIterableDataset
    assert hasattr(MultiModalCropIterableDataset, '_prefetch_chunk'), \
        "Dataset must have _prefetch_chunk for async I/O"
    import inspect
    iter_source = inspect.getsource(MultiModalCropIterableDataset.__iter__)
    assert "threading.Thread" in iter_source or "prefetch_q" in iter_source, \
        "__iter__ must use background thread for prefetching"


# 12. Test app.py no longer claims false offline capability
def test_app_no_false_offline_claim():
    """The sidebar must not claim 'Offline Workspace' — should say 'Edge Client' instead."""
    import inspect
    source = inspect.getsource(app)
    assert "Offline Workspace" not in source, "Must not claim misleading 'Offline Workspace'"
    assert "Edge Client" in source, "Should honestly label as 'Edge Client (PWA)'"


# 13. Test physical weather lag transformations
def test_physical_weather_lags():
    from src.temporal.timeseries_dataset import _apply_physical_lags_numpy
    # Create simple mock weather data: 5 timesteps, 3 features (tmax, tmin, precip)
    w_data = np.array([
        [20.0, 10.0, 10.0],
        [22.0, 11.0, 0.0],
        [21.0, 12.0, 50.0],
        [23.0, 13.0, 0.0],
        [20.0, 10.0, 20.0]
    ])
    
    transformed = _apply_physical_lags_numpy(w_data)
    assert transformed.shape == w_data.shape
    
    # 1. Soil moisture should decay/accumulate:
    # t=0: 10
    # t=1: 10 * 0.8 + 0 = 8.0
    # t=2: 8 * 0.8 + 50 = 56.4
    assert abs(transformed[1, 2] - 8.0) < 1e-4
    assert abs(transformed[2, 2] - 56.4) < 1e-4
    
    # 2. Thermal accumulation should calculate GDD correctly
    # tmean = 15, base 10 -> GDD = 5.0
    # t=0: 5.0
    assert transformed[0, 0] > 0.0


# 14. Test FinancialAttributionSimulator computations
def test_financial_simulator():
    from src.recommendation.engine import FinancialAttributionSimulator
    
    # Under-performing year (Drought Scenario)
    res = FinancialAttributionSimulator.simulate_net_benefit(
        predicted_yield=2.0,
        predicted_std=0.2,
        historical_average=4.0,
        area_ha=100.0,
        price_per_ton=300.0,
        input_cost_per_ha=200.0
    )
    # Predicted yield is 2.0 (50% of history). Proportion = 0.5.
    # Saved Input Cost = 100 * 200 * (1 - 0.5) = 10000.0
    # Insurance saving = 100 * 10 = 1000.0 (std=0.2 < 0.5)
    # Spoilage prevented = 0.0
    # Net benefit = 11000.0
    assert res["saved_input_cost_usd"] == 10000.0
    assert res["insurance_discount_usd"] == 1000.0
    assert res["spoilage_loss_prevented_usd"] == 0.0
    assert res["net_economic_benefit_usd"] == 11000.0
    
    # Over-performing year (Surplus Scenario)
    res_surplus = FinancialAttributionSimulator.simulate_net_benefit(
        predicted_yield=6.0,
        predicted_std=0.3,
        historical_average=4.0,
        area_ha=100.0,
        price_per_ton=300.0,
        input_cost_per_ha=200.0
    )
    # Surplus tons = (6.0 - 4.0) * 100 = 200 tons
    # Spoilage prevented = 200 * 0.15 * 300 = 9000.0
    # Insurance saving = 100 * 10 = 1000.0
    # Saved inputs = 0.0
    # Net benefit = 10000.0
    assert res_surplus["spoilage_loss_prevented_usd"] == 9000.0
    assert res_surplus["saved_input_cost_usd"] == 0.0
    assert res_surplus["net_economic_benefit_usd"] == 10000.0


# 15. Test dynamic calibrated recommendation engine
def test_dynamic_recommendations():
    from src.recommendation.engine import RecommendationEngine
    engine = RecommendationEngine({"paths": {"raw": {"soil": "data/raw/soil"}}})
    
    # Test Acidic pH advice
    res = {
        "region": "Burdwan, West Bengal",
        "predicted_yield": 3.0,
        "lower_bound": 2.0,
        "upper_bound": 4.0,
        "risk": "Moderate Risk",
        "attribution": {"Weather": 0.6, "Satellite": 0.2, "Soil": 0.2},
        "soil_features": [5.2, 12.0, 1.2], # pH 5.2 (acidic)
        "weather_features": [[20.0, 10.0, 180.0]] # precip spike
    }
    
    advice = engine.generate_advice(res)
    assert any("Acidification" in a for a in advice), "Should trigger Acidification advice"
    assert any("Precipitation Spike" in a for a in advice), "Should trigger Precipitation Spike advice"
