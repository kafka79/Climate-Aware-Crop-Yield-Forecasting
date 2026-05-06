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
