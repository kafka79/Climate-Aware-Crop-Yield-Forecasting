import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.explainability.integrated_gradients import explain_prediction
from src.models.mdn import (
    mdn_expected_value,
    mdn_predictive_std,
    mdn_safe_point_estimate,
    BimodalDistributionWarning,
)
from src.models.transformer import initialize_model
from src.risk.risk_classifier import YieldRiskClassifier
from src.recommendation.engine import RecommendationEngine
from src.utils.config import load_config

DEFAULT_CONFIG_PATHS = (
    "configs/data_config.yaml",
    "configs/model_config.yaml",
    "configs/training_config.yaml",
)


class InferenceUnavailableError(RuntimeError):
    """Raised when real model inference cannot be served honestly."""


def load_runtime_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    seen_paths = set()

    for path in DEFAULT_CONFIG_PATHS:
        config.update(load_config(path))
        seen_paths.add(os.path.normpath(path))

    if config_path:
        normalized = os.path.normpath(config_path)
        if normalized not in seen_paths:
            config.update(load_config(config_path))

    return config


def list_configured_regions(config: Dict[str, Any]) -> List[str]:
    return [area["name"] for area in config.get("study_areas", [])]


def load_yield_history(config: Dict[str, Any]) -> pd.DataFrame:
    yield_path = Path(config["paths"]["raw"]["yield"]) / "historical_yield.csv"
    if not yield_path.exists():
        return pd.DataFrame(columns=["site_id", "time", "yield", "year"])

    history = pd.read_csv(yield_path)
    if "time" in history.columns:
        history["time"] = pd.to_datetime(history["time"])
        history["year"] = history["time"].dt.year
    else:
        history["year"] = pd.Series(dtype=int)

    return history


def list_available_years(config: Dict[str, Any]) -> List[int]:
    history = load_yield_history(config)
    if history.empty:
        return []
    return sorted(int(year) for year in history["year"].dropna().unique())


def _get_region_record(config: Dict[str, Any], region: str) -> Dict[str, Any]:
    for area in config.get("study_areas", []):
        if area["name"] == region:
            return area
    raise InferenceUnavailableError(
        f"Region '{region}' is not configured in configs/data_config.yaml."
    )


def _get_region_history(config: Dict[str, Any], region: str) -> pd.DataFrame:
    history = load_yield_history(config)
    if history.empty:
        return history
    if "site_id" not in history.columns:
        return history.iloc[0:0].copy()
    return history[history["site_id"] == region].copy()


def _get_feature_paths(config: Dict[str, Any], region: str) -> Tuple[Path, Path]:
    processed_dir = Path(config["paths"]["processed"]["features"])
    sat_path = processed_dir / f"{region}_sat_proc.zarr"
    weather_path = processed_dir / f"{region}_weather_proc.zarr"
    return sat_path, weather_path


def _dataset_years(dataset: xr.Dataset) -> List[int]:
    years = pd.to_datetime(dataset.time.values).year
    return sorted(int(year) for year in np.unique(years))


def _align_vector_length(values: np.ndarray, target_dim: int) -> np.ndarray:
    if len(values) >= target_dim:
        return values[:target_dim]

    padded = np.zeros(target_dim, dtype=np.float32)
    padded[: len(values)] = values
    return padded


class MissingSoilDataWarning(UserWarning):
    """Raised when soil data is unavailable and a zero-vector fallback is used."""


def _load_soil_vector(
    config: Dict[str, Any], region: str, soil_dim: int
) -> Tuple[np.ndarray, str, List[str]]:
    """Load soil vector with explicit failure tracking.

    Returns (vector, source_label, warnings_list).  The warnings list is
    always populated when a zero-vector fallback is used so that callers
    can surface the degradation to the user instead of hiding it.
    """
    modality_warnings: List[str] = []
    soil_path = Path(config["paths"]["raw"]["soil"]) / f"{region}_soil.csv"

    if not soil_path.exists():
        msg = (
            f"Soil data file missing for '{region}' at {soil_path}. "
            "Using a zero-vector fallback — the Transformer's soil attention "
            "head will receive no signal, which may bias the forecast."
        )
        warnings.warn(msg, MissingSoilDataWarning, stacklevel=2)
        from loguru import logger
        logger.warning(msg)
        modality_warnings.append(msg)
        return np.zeros(soil_dim, dtype=np.float32), "MISSING → zero-vector", modality_warnings

    soil_df = pd.read_csv(soil_path).select_dtypes(include=[np.number])
    if soil_df.empty:
        msg = (
            f"Soil file for '{region}' contains no numeric columns. "
            "Using a zero-vector fallback — forecast may be unreliable."
        )
        warnings.warn(msg, MissingSoilDataWarning, stacklevel=2)
        from loguru import logger
        logger.warning(msg)
        modality_warnings.append(msg)
        return np.zeros(soil_dim, dtype=np.float32), "NON-NUMERIC → zero-vector", modality_warnings

    soil_values = soil_df.iloc[0].to_numpy(dtype=np.float32)
    if np.all(soil_values == 0):
        msg = (
            f"Soil file for '{region}' exists but all values are zero. "
            "This is equivalent to missing data and may skew the model."
        )
        warnings.warn(msg, MissingSoilDataWarning, stacklevel=2)
        from loguru import logger
        logger.warning(msg)
        modality_warnings.append(msg)

    return _align_vector_length(soil_values, soil_dim), f"file:{soil_path.name}", modality_warnings


def _select_time_window(
    dataset: xr.DataArray, year: int, window_size: int, region: str, label: str
) -> xr.DataArray:
    timestamps = pd.to_datetime(dataset.time.values)
    indices = np.flatnonzero(timestamps.year == year)
    if len(indices) == 0:
        available_years = sorted(int(item) for item in np.unique(timestamps.year))
        raise InferenceUnavailableError(
            f"No {label} features for {region} in {year}. "
            f"Available years: {available_years or 'none'}."
        )

    year_slice = dataset.isel(time=indices)
    if year_slice.sizes.get("time", 0) < window_size:
        raise InferenceUnavailableError(
            f"{label.capitalize()} data for {region} in {year} has only "
            f"{year_slice.sizes.get('time', 0)} timesteps; need at least {window_size}."
        )

    return year_slice.isel(time=slice(-window_size, None))


def _compute_ndvi_series(sat_hist: xr.DataArray) -> Optional[List[float]]:
    if "B08" not in sat_hist or "B04" not in sat_hist:
        return None

    nir = sat_hist["B08"].values.reshape(-1)
    red = sat_hist["B04"].values.reshape(-1)
    denom = nir + red
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    return ndvi.astype(float).tolist()


def _prepare_model_inputs(
    config: Dict[str, Any], region: str, year: int
) -> Dict[str, Any]:
    area = _get_region_record(config, region)
    sat_path, weather_path = _get_feature_paths(config, region)
    if not sat_path.exists() or not weather_path.exists():
        raise InferenceUnavailableError(
            f"Processed feature stores are missing for {region}. "
            "Run download and preprocess before requesting live inference."
        )

    sat_ds = xr.open_zarr(sat_path)
    weather_ds = xr.open_zarr(weather_path)

    lat = area.get("lat")
    lon = area.get("lon")
    if lat is None or lon is None:
        bbox = area.get("bbox")
        if not bbox or len(bbox) < 2:
            raise InferenceUnavailableError(
                f"Region '{region}' is missing coordinates in configs/data_config.yaml."
            )
        lon = bbox[0]
        lat = bbox[1]

    window_size = int(config.get("training", {}).get("window_size", 12))
    sat_pixel = sat_ds.sel(lat=lat, lon=lon, method="nearest")
    weather_pixel = weather_ds.sel(lat=lat, lon=lon, method="nearest")

    sat_hist = _select_time_window(sat_pixel, year, window_size, region, "satellite")
    weather_hist = _select_time_window(weather_pixel, year, window_size, region, "weather")

    sat_data = sat_hist.to_array().values.T
    weather_data = weather_hist.to_array().values.T

    sat_dim = int(config["transformer"]["input_dim"])
    weather_dim = int(config["transformer"]["temporal_dim"])
    soil_dim = int(config["transformer"]["soil_dim"])

    if sat_data.shape[1] < sat_dim:
        raise InferenceUnavailableError(
            f"Satellite feature store for {region} exposes {sat_data.shape[1]} channels, "
            f"but the checkpoint expects {sat_dim}."
        )
    if weather_data.shape[1] < weather_dim:
        raise InferenceUnavailableError(
            f"Weather feature store for {region} exposes {weather_data.shape[1]} channels, "
            f"but the checkpoint expects {weather_dim}."
        )

    soil_vector, soil_source, soil_warnings = _load_soil_vector(config, region, soil_dim)

    return {
        "sat_hist": sat_hist,
        "sat_tensor": torch.tensor(sat_data[:, :sat_dim], dtype=torch.float32).unsqueeze(0),
        "weather_tensor": torch.tensor(
            weather_data[:, :weather_dim], dtype=torch.float32
        ).unsqueeze(0),
        "soil_tensor": torch.tensor(soil_vector, dtype=torch.float32).unsqueeze(0),
        "soil_source": soil_source,
        "ndvi_series": _compute_ndvi_series(sat_hist),
        "modality_warnings": soil_warnings,
    }


def build_region_context(
    region: str, year: int, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    config = config or load_runtime_config()
    region_history = _get_region_history(config, region)
    observed_row = region_history[region_history["year"] == year]
    observed_yield = None
    if not observed_row.empty:
        observed_yield = float(observed_row.iloc[-1]["yield"])

    sat_path, weather_path = _get_feature_paths(config, region)
    feature_store_ready = sat_path.exists() and weather_path.exists()
    feature_years: List[int] = []
    ndvi_series: Optional[List[float]] = None
    live_error: Optional[str] = None

    if feature_store_ready:
        sat_ds = xr.open_zarr(sat_path)
        feature_years = _dataset_years(sat_ds)
        if year in feature_years:
            try:
                prepared = _prepare_model_inputs(config, region, year)
                ndvi_series = prepared["ndvi_series"]
            except InferenceUnavailableError as exc:
                ndvi_series = None
                live_error = str(exc)

    model_ready = Path("models/checkpoints/best_model.pth").exists()
    live_ready = (
        feature_store_ready
        and model_ready
        and year in feature_years
        and live_error is None
    )

    if live_ready:
        status = "Ready for live inference"
    elif live_error:
        status = live_error
    elif not feature_store_ready:
        status = "Missing processed feature store for this region"
    elif not model_ready:
        status = "Missing trained checkpoint"
    else:
        status = f"Processed features do not cover {year}"

    if region_history.empty:
        history_view = pd.DataFrame(columns=["year", "yield"])
        historical_average = None
    else:
        history_view = region_history[["year", "yield"]].copy()
        historical_average = float(region_history["yield"].mean())

    return {
        "region": region,
        "year": year,
        "historical_average": historical_average,
        "observed_yield": observed_yield,
        "yield_history": history_view,
        "feature_store_ready": feature_store_ready,
        "feature_years": feature_years,
        "model_ready": model_ready,
        "live_ready": live_ready,
        "status": status,
        "ndvi_series": ndvi_series,
    }


def run_inference(
    region: str,
    year: int,
    crop: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    del crop
    config = load_runtime_config(config_path)
    prepared = _prepare_model_inputs(config, region, year)
    model_path = Path("models/checkpoints/best_model.pth")
    if not model_path.exists():
        raise InferenceUnavailableError(
            "Trained checkpoint not found at models/checkpoints/best_model.pth."
        )

    model = initialize_model(config)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        output = model(
            prepared["sat_tensor"], prepared["weather_tensor"], prepared["soil_tensor"]
        )

    if isinstance(output, tuple):
        pi, sigma, mu = output
        # Use the safe estimator: detects bimodal distributions and refuses
        # to return a valley-mean that sits between two real scenarios.
        predicted_yield, bimodality_report = mdn_safe_point_estimate(pi, sigma, mu)
        std = mdn_predictive_std(pi, sigma, mu)
        predicted_std = float(std.squeeze().cpu().item())
    else:
        predicted_yield = float(output.squeeze().cpu().item())
        predicted_std = 0.0
        bimodality_report = {"is_bimodal": False, "modes": [], "dominant_mode": predicted_yield, "valley_depth": 0.0}

    if predicted_yield <= 0 or predicted_yield >= 15:
        raise InferenceUnavailableError(
            f"Checkpoint produced an implausible forecast ({predicted_yield:.3f} t/ha) "
            "for the selected inputs. Retrain or recalibrate the model before exposing "
            "live forecasts."
        )

    region_history = _get_region_history(config, region)
    historical_average = (
        float(region_history["yield"].mean()) if not region_history.empty else None
    )
    observed_row = region_history[region_history["year"] == year]
    observed_yield = (
        float(observed_row.iloc[-1]["yield"]) if not observed_row.empty else None
    )

    classifier = YieldRiskClassifier()
    if historical_average:
        risk = classifier.calibrate_with_uncertainty(
            predicted_yield, predicted_std, historical_average
        )
    else:
        risk = "Unknown"

    explanation_summary, _ = explain_prediction(
        model,
        {
            "sat": prepared["sat_tensor"].squeeze(0),
            "weather": prepared["weather_tensor"].squeeze(0),
            "soil": prepared["soil_tensor"].squeeze(0),
        },
    )
    attribution = {
        "Satellite": round(explanation_summary.get("satellite_overall", 0.0), 4),
        "Weather": round(explanation_summary.get("weather_overall", 0.0), 4),
        "Soil": round(explanation_summary.get("soil_overall", 0.0), 4),
    }

    lower_bound = max(0.0, predicted_yield - 1.96 * predicted_std)
    upper_bound = predicted_yield + 1.96 * predicted_std

    # Surface modality warnings so the dashboard/CLI can show them
    modality_warnings = prepared.get("modality_warnings", [])

    engine = RecommendationEngine(config)
    advice = engine.generate_advice({
        "predicted_yield": predicted_yield,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "risk": risk,
        "attribution": attribution
    })

    return {
        "region": region,
        "year": year,
        "predicted_yield": predicted_yield,
        "prediction_std": predicted_std,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "risk": risk,
        "historical_average": historical_average,
        "observed_yield": observed_yield,
        "soil_source": prepared["soil_source"],
        "ndvi_series": prepared["ndvi_series"],
        "attribution": attribution,
        "recommendations": advice,
        "modality_warnings": modality_warnings,
        "bimodality_report": bimodality_report,
        "source": "checkpoint+zarr",
    }
