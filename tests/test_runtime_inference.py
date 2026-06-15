from src.inference.runtime import build_region_context, load_runtime_config


def test_region_context_reports_real_feature_store():
    config = load_runtime_config()
    context = build_region_context("Burdwan, West Bengal", 2023, config)

    assert context["feature_store_ready"] is True
    assert 2023 in context["feature_years"]
    assert context["ndvi_series"] is not None


def test_region_context_flags_missing_processed_region():
    config = load_runtime_config()
    context = build_region_context("Purnia, Bihar", 2023, config)

    assert context["feature_store_ready"] is False
    assert context["live_ready"] is False
    assert context["ndvi_series"] is None


def test_model_caching():
    from src.inference.runtime import _get_cached_model, _MODEL_CACHE
    import unittest.mock as mock
    from pathlib import Path
    
    _MODEL_CACHE.clear()
    config = load_runtime_config()
    model_path = Path("models/checkpoints/best_model.pth")
    
    # Ensure best_model.pth exists or create dummy
    existed = model_path.exists()
    if not existed:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()
        
    try:
        with mock.patch("src.inference.runtime.initialize_model") as mock_init, \
             mock.patch("src.inference.runtime.load_model_weights") as mock_load:
             
            mock_model = mock.MagicMock()
            mock_init.return_value = mock_model
            
            m1 = _get_cached_model(model_path, config)
            m2 = _get_cached_model(model_path, config)
            
            assert m1 is m2
            assert mock_init.call_count == 1
            assert mock_load.call_count == 1
    finally:
        if not existed and model_path.exists():
            model_path.unlink()
        _MODEL_CACHE.clear()
