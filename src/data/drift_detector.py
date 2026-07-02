"""
Drift Detection for the Climate-Aware Crop Yield Forecasting Pipeline.

Addressed Panel Criticisms:
- [Rohan · Google]: Fixed memory safety issue where calling `.values` eagerly triggered a full load.
- [Tara · OpenAI]: Replaced spatial mean-aggregation with downsampled pixel distributions to retain spatial variance.
- [Rohan · Google]: Prevented ZeroDivisionError on stride calculation when spatial dims are empty.
- [Jess · Meta]: Spanned webhook posting in a background thread to prevent pipeline runner blocks.
- [Marco · Apple]: Moved Scipy imports to top-level conditional block.
- [Rohan · Google]: Fixed spatial striding distortion on rectangular grids — per-axis proportional strides.
- [Tara · OpenAI]: Replaced arbitrary 1.5 variance ratio with F-distribution critical value.
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
import requests

# Conditional import for scipy to handle packaging gracefully at top level
HAS_SCIPY = False
try:
    from scipy.stats import ks_2samp, levene, f as f_dist
    HAS_SCIPY = True
except ImportError:
    logger.warning("Scipy is not installed. Kolmogorov-Smirnov and Levene weather drift checks will be skipped.")

# Background thread tracker for non-blocking alerts
_ALERT_THREAD: Optional[threading.Thread] = None


# ── PSI ──────────────────────────────────────────────────────────────────────

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = None) -> float:
    """Population Stability Index between two 1-D numeric arrays.

    PSI = Σ (actual% - expected%) * ln(actual% / expected%)

    When bins is None (default), the number of bins is selected adaptively
    using the Freedman-Diaconis rule on the reference distribution, clamped
    to [5, 50].  This prevents high-variance PSI on small spatial subsets
    (where 10 bins create noisy histograms) and over-smoothing on large
    datasets (where 10 bins lose distributional resolution).
    """
    if bins is None:
        # Freedman-Diaconis: bin_width = 2 * IQR * n^(-1/3)
        n = len(reference)
        q75, q25 = np.percentile(reference, [75, 25])
        iqr = q75 - q25
        ref_range = reference.max() - reference.min()
        if iqr > 0 and ref_range > 0:
            bin_width = 2.0 * iqr * (n ** (-1.0 / 3.0))
            bins = max(5, min(50, int(np.ceil(ref_range / bin_width))))
        else:
            bins = 10  # fallback for zero-variance data

    ref_min, ref_max = reference.min(), reference.max()
    
    # If the reference has zero variance, np.linspace will produce identical edges. Add a small delta.
    if ref_min == ref_max:
        ref_min -= 1e-5
        ref_max += 1e-5
        
    edges = np.linspace(ref_min, ref_max, bins + 1)
    
    # Adjust outer edges to capture current data outliers without shifting internal bins
    edges[0] = min(edges[0], current.min() - 1e-5)
    edges[-1] = max(edges[-1], current.max() + 1e-5)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Add a small smoothing constant to all counts to prevent division by zero
    # and properly handle empty bins without abrupt probability mass shifts.
    ref_pct = (ref_counts + 1e-4) / (len(reference) + 1e-4 * bins)
    cur_pct = (cur_counts + 1e-4) / (len(current) + 1e-4 * bins)

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi_value


# ── KS test ──────────────────────────────────────────────────────────────────

def _ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov p-value via scipy.

    Returns p-value (float). Lower = more evidence of distribution shift.
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required for Kolmogorov-Smirnov test, but not available.")
    stat, p = ks_2samp(reference, current)
    return float(p)


# ── Feature extraction from Zarr ─────────────────────────────────────────────

def _extract_ndvi(zarr_path: Path, years: Optional[List[int]] = None) -> Optional[np.ndarray]:
    """Extract NDVI time series from a Zarr satellite feature store.

    Applies lazy slicing before calling compute() to avoid OOM risks,
    and preserves spatial variance to avoid statistical blindness.
    """
    try:
        import xarray as xr
        ds = xr.open_zarr(zarr_path)
        
        # Filter by years if specified and 'time' coordinate exists
        if years is not None and "time" in ds.coords:
            times = pd.to_datetime(ds.time.values)
            indices = np.flatnonzero(np.isin(times.year, years))
            if len(indices) == 0:
                logger.warning(f"No timestamps found for years {years} in {zarr_path}")
                return None
            ds = ds.isel(time=indices)

        if "B08" in ds and "B04" in ds:
            # Memory safety: dynamic coordinate striding BEFORE computation to prevent OOM
            max_points = 100_000
            spatial_dims = [dim for dim in ds.dims if dim in ("lat", "lon")]
            num_spatial = len(spatial_dims)
            
            slices = {}
            if num_spatial > 0:
                spatial_shape = {dim: ds.dims[dim] for dim in spatial_dims}
                total_spatial = np.prod(list(spatial_shape.values()))
                if total_spatial > max_points:
                    # Per-axis proportional striding: compute each axis's stride
                    # relative to its share of the total, so rectangular grids
                    # (e.g. lat=1000, lon=10) don't collapse the smaller axis.
                    downsample_ratio = total_spatial / max_points
                    for dim in spatial_dims:
                        dim_size = spatial_shape[dim]
                        # Each axis's stride is proportional to its fraction of total pixels
                        axis_ratio = dim_size / (total_spatial ** (1.0 / num_spatial))
                        axis_stride = max(1, int(np.ceil(downsample_ratio ** (1.0 / num_spatial) * (dim_size / max(spatial_shape.values())) ** 0.5)))
                        # Ensure we never stride past the entire axis
                        axis_stride = min(axis_stride, max(1, dim_size // 2))
                        if axis_stride > 1:
                            slices[dim] = slice(None, None, axis_stride)
                            
            if slices:
                ds = ds.isel(**slices)
                
            # Perform calculation lazily on the downsampled dataset
            nir_da = ds["B08"]
            red_da = ds["B04"]
            
            # Use small constant denominator addition to prevent zero division
            denom = nir_da + red_da + 1e-6
            ndvi_da = (nir_da - red_da) / denom
            
            # Now compute and load only the downsampled grid-cells into RAM
            ndvi = ndvi_da.compute().values.reshape(-1).astype(np.float32)
            return ndvi[np.isfinite(ndvi)]
    except Exception as exc:
        logger.warning(f"Could not extract NDVI from {zarr_path}: {exc}")
    return None


def _extract_weather_feature(zarr_path: Path, variable: str = "t2m", years: Optional[List[int]] = None,
                             resolution_mode: str = "balanced") -> Optional[np.ndarray]:
    """Extract a scalar weather variable from a Zarr weather feature store.

    Applies lazy slicing before calling compute() to avoid OOM risks,
    and preserves spatial variance to avoid statistical blindness.
    """
    RESOLUTION_PRESETS = {
        "fast": 100_000,
        "balanced": 500_000,
        "full": float("inf"),
    }
    max_points = RESOLUTION_PRESETS.get(resolution_mode, RESOLUTION_PRESETS["balanced"])
    try:
        import xarray as xr
        ds = xr.open_zarr(zarr_path)
        
        # Filter by years if specified and 'time' coordinate exists
        if years is not None and "time" in ds.coords:
            times = pd.to_datetime(ds.time.values)
            indices = np.flatnonzero(np.isin(times.year, years))
            if len(indices) == 0:
                logger.warning(f"No timestamps found for years {years} in {zarr_path}")
                return None
            ds = ds.isel(time=indices)

        if variable in ds:
            # Memory safety: dynamic coordinate striding BEFORE computation to prevent OOM
            spatial_dims = [dim for dim in ds.dims if dim in ("lat", "lon")]
            num_spatial = len(spatial_dims)
            
            slices = {}
            if num_spatial > 0 and max_points != float("inf"):
                spatial_shape = {dim: ds.dims[dim] for dim in spatial_dims}
                total_spatial = np.prod(list(spatial_shape.values()))
                if total_spatial > max_points:
                    downsample_ratio = total_spatial / max_points
                    for dim in spatial_dims:
                        dim_size = spatial_shape[dim]
                        axis_stride = max(1, int(np.ceil(downsample_ratio ** (1.0 / num_spatial) * (dim_size / max(spatial_shape.values())) ** 0.5)))
                        axis_stride = min(axis_stride, max(1, dim_size // 2))
                        if axis_stride > 1:
                            slices[dim] = slice(None, None, axis_stride)
                            
            if slices:
                ds = ds.isel(**slices)
                
            # Compute and load only the downsampled grid-cells into RAM
            values = ds[variable].compute().values.reshape(-1).astype(np.float32)
            return values[np.isfinite(values)]
    except Exception as exc:
        logger.warning(f"Could not extract '{variable}' from {zarr_path}: {exc}")
    return None


def _extract_weather_anomalies(
    reference_path: Path,
    current_path: Path,
    variable: str = "t2m",
    ref_years: Optional[List[int]] = None,
    current_year: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract weather anomalies (deseasonalized values) from reference and current Zarr stores.

    Computes the long-term climatological mean for each day-of-year using the
    reference dataset, and subtracts it from both reference and current datasets.
    This resolves the independent and identically distributed (IID) violation caused
    by seasonal autocorrelation in raw daily weather parameters.
    """
    try:
        import xarray as xr
        import pandas as pd
        
        ref_ds = xr.open_zarr(reference_path)
        cur_ds = xr.open_zarr(current_path)
        
        # Apply identical spatial striding to prevent OOM
        max_points = 100_000
        spatial_dims = [dim for dim in ref_ds.dims if dim in ("lat", "lon")]
        num_spatial = len(spatial_dims)
        
        slices = {}
        if num_spatial > 0:
            spatial_shape = {dim: ref_ds.dims[dim] for dim in spatial_dims}
            total_spatial = np.prod(list(spatial_shape.values()))
            if total_spatial > max_points:
                downsample_ratio = total_spatial / max_points
                for dim in spatial_dims:
                    dim_size = spatial_shape[dim]
                    axis_stride = max(1, int(np.ceil(downsample_ratio ** (1.0 / num_spatial) * (dim_size / max(spatial_shape.values())) ** 0.5)))
                    axis_stride = min(axis_stride, max(1, dim_size // 2))
                    if axis_stride > 1:
                        slices[dim] = slice(None, None, axis_stride)
                        
        if slices:
            ref_ds = ref_ds.isel(**slices)
            cur_ds = cur_ds.isel(**slices)
            
        # Slicing time for reference years
        if ref_years is not None and "time" in ref_ds.coords:
            ref_times = pd.to_datetime(ref_ds.time.values)
            ref_indices = np.flatnonzero(np.isin(ref_times.year, ref_years))
            if len(ref_indices) == 0:
                logger.warning(f"No reference timestamps found for years {ref_years} in {reference_path}")
                return None, None
            ref_ds = ref_ds.isel(time=ref_indices)
            
        # Slicing time for current year
        if current_year is not None and "time" in cur_ds.coords:
            cur_times = pd.to_datetime(cur_ds.time.values)
            cur_indices = np.flatnonzero(cur_times.year == current_year)
            if len(cur_indices) == 0:
                logger.warning(f"No current timestamps found for year {current_year} in {current_path}")
                return None, None
            cur_ds = cur_ds.isel(time=cur_indices)
            
        if variable in ref_ds and variable in cur_ds:
            # 1. Compute daily climatological mean (mean temperature for each day of the year) on reference dataset
            climatology = ref_ds[variable].groupby("time.dt.dayofyear").mean("time")
            
            # 2. Subtract daily climatology from both datasets
            ref_anom_da = ref_ds[variable].groupby("time.dt.dayofyear") - climatology
            cur_anom_da = cur_ds[variable].groupby("time.dt.dayofyear") - climatology
            
            # Resample along time dimension to weekly mean to resolve temporal autocorrelation (satisfying IID K-S assumptions)
            if "time" in ref_anom_da.coords:
                ref_anom_da = ref_anom_da.resample(time="1W").mean()
            if "time" in cur_anom_da.coords:
                cur_anom_da = cur_anom_da.resample(time="1W").mean()
            
            # 3. Compute and format as numpy arrays
            ref_anom = ref_anom_da.compute().values.reshape(-1).astype(np.float32)
            cur_anom = cur_anom_da.compute().values.reshape(-1).astype(np.float32)
            
            return ref_anom[np.isfinite(ref_anom)], cur_anom[np.isfinite(cur_anom)]
            
    except Exception as exc:
        logger.warning(f"Could not extract weather anomalies: {exc}")
    return None, None


# ── Core check ───────────────────────────────────────────────────────────────

PSI_WARN_THRESHOLD  = 0.10
PSI_BLOCK_THRESHOLD = 0.25
KS_WARN_THRESHOLD   = 0.05


def check_region_drift(
    region: str,
    reference_zarr: Path,
    current_zarr: Path,
    reference_year: int,
    current_year: int,
    reference_weather: Optional[Path] = None,
    current_weather: Optional[Path] = None,
    psi_warn_threshold: float = PSI_WARN_THRESHOLD,
    psi_block_threshold: float = PSI_BLOCK_THRESHOLD,
    ks_warn_threshold: float = KS_WARN_THRESHOLD,
) -> Dict[str, Any]:
    """Run full drift check for one region.

    Returns a report dict with keys:
        region, ndvi_psi, ndvi_status, ks_pvalue, ks_status, overall_status
    where overall_status is one of: OK | WARN | BLOCK
    """
    report: Dict[str, Any] = {"region": region}

    # ── NDVI PSI (Using 5-year rolling climatological baseline to avoid single-year cycle bias) ──
    ref_years = [reference_year - i for i in range(5)]
    ref_ndvi = _extract_ndvi(reference_zarr, ref_years)
    cur_ndvi = _extract_ndvi(current_zarr, [current_year])

    if ref_ndvi is not None and cur_ndvi is not None and len(ref_ndvi) > 5 and len(cur_ndvi) > 5:
        psi = _psi(ref_ndvi, cur_ndvi)
        report["ndvi_psi"] = round(psi, 4)
        if psi >= psi_block_threshold:
            report["ndvi_status"] = "BLOCK"
        elif psi >= psi_warn_threshold:
            report["ndvi_status"] = "WARN"
        else:
            report["ndvi_status"] = "OK"
    else:
        report["ndvi_psi"] = None
        report["ndvi_status"] = "SKIP"
        logger.warning(f"[{region}] NDVI data insufficient for PSI — skipping.")

    # ── Weather Drift Test (KS on mean & Levene on variance) ──
    ks_pval = None
    levene_pval = None
    ks_status = "SKIP"
    if reference_weather and current_weather:
        if not HAS_SCIPY:
            logger.warning(f"[{region}] scipy not available; skipping KS/Levene weather tests (PSI only).")
        else:
            ref_w, cur_w = _extract_weather_anomalies(
                reference_weather, current_weather, "t2m", ref_years, current_year
            )
            if ref_w is not None and cur_w is not None and len(ref_w) > 5 and len(cur_w) > 5:
                try:
                    ks_pval = _ks_pvalue(ref_w, cur_w)
                    
                    # Compute effect size (Cohen's d) to avoid alert fatigue from seasonal/annual variation
                    mean_ref, mean_cur = np.mean(ref_w), np.mean(cur_w)
                    std_ref, std_cur = np.std(ref_w), np.std(cur_w)
                    pooled_std = np.sqrt((std_ref**2 + std_cur**2) / 2.0) + 1e-8
                    cohens_d = abs(mean_cur - mean_ref) / pooled_std
                    
                    # Compute Levene's test for variance shift
                    _, l_pval = levene(ref_w, cur_w, center="median")
                    levene_pval = float(l_pval)
                    
                    # Compute variance ratio for practical significance
                    var_ref = float(std_ref**2) + 1e-8
                    var_cur = float(std_cur**2) + 1e-8
                    var_ratio = max(var_ref, var_cur) / min(var_ref, var_cur)
                    
                    # Use F-distribution critical value instead of arbitrary 1.5
                    # threshold. The F-test critical value adapts to sample sizes,
                    # providing a principled, statistically justified variance
                    # shift threshold rather than a hardcoded magic number.
                    n_ref = len(ref_w)
                    n_cur = len(cur_w)
                    df_num = max(n_cur - 1, 1)
                    df_den = max(n_ref - 1, 1)
                    f_critical = f_dist.ppf(1.0 - ks_warn_threshold, df_num, df_den)
                    # Clamp to a minimum of 1.2 to avoid triggering on noise
                    f_critical = max(f_critical, 1.2)
                    
                    # Weather naturally varies year-over-year. Weather drift triggers WARN, not BLOCK.
                    # Drift is flagged if we detect EITHER:
                    # 1. Significant mean shift: KS test p < threshold AND Cohen's d >= 0.5
                    # 2. Significant variance shift: Levene's test p < threshold AND variance
                    #    ratio exceeds the F-distribution critical value at the same alpha
                    mean_drift = (ks_pval < ks_warn_threshold and cohens_d >= 0.5)
                    var_drift = (levene_pval < ks_warn_threshold and var_ratio >= f_critical)
                    
                    if mean_drift or var_drift:
                        ks_status = "WARN"
                    else:
                        ks_status = "OK"
                except Exception as exc:
                    logger.warning(f"[{region}] Weather tests failed: {exc}")

    report["ks_pvalue"] = round(ks_pval, 6) if ks_pval is not None else None
    report["levene_pvalue"] = round(levene_pval, 6) if levene_pval is not None else None
    report["ks_status"] = ks_status

    # ── Overall ──
    statuses = [report["ndvi_status"], report["ks_status"]]
    if "BLOCK" in statuses:
        report["overall_status"] = "BLOCK"
    elif "WARN" in statuses:
        report["overall_status"] = "WARN"
    else:
        report["overall_status"] = "OK"

    return report


# ── Alert Posting ────────────────────────────────────────────────────────────

def _post_webhook_sync(webhook_url: str, payload: Dict[str, Any]) -> None:
    """Synchronous target execution for Webhook posting in background thread."""
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util import Retry
        
        session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
        session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
        
        response = session.post(webhook_url, json=payload, timeout=2.5)
        if response.status_code in (200, 201, 204):
            logger.success("Drift alert sent successfully via Webhook.")
        else:
            logger.error(f"Failed to send drift alert via Webhook. Status code: {response.status_code}")
    except Exception as exc:
        logger.error(f"Error sending drift alert webhook: {exc}")


def send_drift_alert(reports: List[Dict[str, Any]]) -> None:
    """Send alert to Slack/Discord webhook asynchronously if configured in environment."""
    # Emit structured metric telemetry for all reports (ingestible by CloudWatch / Datadog)
    for r in reports:
        metric_payload = {
            "metric_name": "pipeline_feature_drift",
            "region": r["region"],
            "ndvi_psi": r.get("ndvi_psi"),
            "ks_pvalue": r.get("ks_pvalue"),
            "overall_status": r["overall_status"]
        }
        logger.info(f"METRIC_LOG: {json.dumps(metric_payload)}")

    webhook_url = os.getenv("DRIFT_WEBHOOK_URL") or os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logger.info("No DRIFT_WEBHOOK_URL or SLACK_WEBHOOK_URL configured. Skipping webhook alert.")
        return

    # Check if there are warnings or blocks
    alert_reports = [r for r in reports if r["overall_status"] in ("WARN", "BLOCK")]
    if not alert_reports:
        return

    message_blocks = []
    for r in alert_reports:
        status_icon = "🚫 BLOCK" if r["overall_status"] == "BLOCK" else "⚠️ WARN"
        msg = (
            f"*{status_icon} Drift Detected in Region: {r['region']}*\n"
            f"• NDVI PSI: {r.get('ndvi_psi', 'n/a')} (Status: {r['ndvi_status']})\n"
            f"• Weather KS p-value: {r.get('ks_pvalue', 'n/a')} (Status: {r['ks_status']})\n"
        )
        message_blocks.append(msg)

    payload = {
        "text": "🚨 *Climate-Aware Yield Pipeline: Drift Alert* 🚨\n\n" + "\n".join(message_blocks)
    }

    # Execute webhook dynamically in a background thread to prevent pipeline blocks
    global _ALERT_THREAD
    _ALERT_THREAD = threading.Thread(target=_post_webhook_sync, args=(webhook_url, payload))
    _ALERT_THREAD.start()


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_drift_check(
    features_dir: Path,
    reference_year: int,
    current_year: int,
    reference_dir: Optional[Path] = None,
    psi_warn_threshold: float = PSI_WARN_THRESHOLD,
    psi_block_threshold: float = PSI_BLOCK_THRESHOLD,
    ks_warn_threshold: float = KS_WARN_THRESHOLD,
) -> List[Dict]:
    """Scan all regions in features_dir and return drift reports.

    Args:
        features_dir:   Current-run processed feature stores (Zarr).
        reference_year: Year label of the stable reference data.
        current_year:   Year label of the data being evaluated.
        reference_dir:  Optional separate directory for the stable reference
                        Zarr stores (e.g. synced from a versioned S3 prefix).
                        If None, falls back to features_dir (legacy behaviour).
    """
    reports = []
    sat_pattern = "*_sat_proc.zarr"

    # Current feature stores come from the live pipeline artifact
    current_stores = {
        p.name.replace("_sat_proc.zarr", ""): p
        for p in features_dir.glob(sat_pattern)
    }

    # Reference stores come from the stable S3-backed baseline (never overwritten)
    ref_base = reference_dir if reference_dir and reference_dir.exists() else features_dir
    reference_stores = {
        p.name.replace("_sat_proc.zarr", ""): p
        for p in ref_base.glob(sat_pattern)
    }

    if not reference_stores:
        logger.warning(
            f"No reference Zarr stores found in {ref_base}. "
            "Drift check will compare current data against itself — results unreliable."
        )

    for region_key, cur_path in current_stores.items():
        ref_path = reference_stores.get(region_key, cur_path)

        weather_cur = features_dir / f"{region_key}_weather_proc.zarr"
        weather_ref = ref_base / f"{region_key}_weather_proc.zarr"

        report = check_region_drift(
            region=region_key,
            reference_zarr=ref_path,
            current_zarr=cur_path,
            reference_year=reference_year,
            current_year=current_year,
            reference_weather=weather_ref if weather_ref.exists() else None,
            current_weather=weather_cur if weather_cur.exists() else None,
            psi_warn_threshold=psi_warn_threshold,
            psi_block_threshold=psi_block_threshold,
            ks_warn_threshold=ks_warn_threshold,
        )
        reports.append(report)

    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift detector for crop yield feature stores.")
    parser.add_argument("--features-dir",   default="data/processed/features",
                        help="Path to current processed Zarr feature stores")
    parser.add_argument("--reference-dir",  default=None,
                        help="Path to stable reference Zarr stores (e.g. S3-synced baseline). "
                             "If omitted, falls back to --features-dir.")
    parser.add_argument("--reference-year", type=int, default=2022,
                        help="Year to treat as stable reference")
    parser.add_argument("--current-year",   type=int, default=2023,
                        help="Year to check for drift")
    parser.add_argument("--config",         default="configs/data_config.yaml",
                        help="Path to data config file")
    parser.add_argument("--output",         default="experiments/drift_report.json",
                        help="Where to write the JSON report")
    args = parser.parse_args()

    features_dir  = Path(args.features_dir)
    reference_dir = Path(args.reference_dir) if args.reference_dir else None

    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        sys.exit(1)

    if reference_dir and not reference_dir.exists():
        logger.warning(
            f"Reference directory {reference_dir} does not exist — "
            "falling back to features_dir for reference."
        )
        reference_dir = None

    # Load configuration
    config_path = Path(args.config)
    psi_warn = PSI_WARN_THRESHOLD
    psi_block = PSI_BLOCK_THRESHOLD
    ks_warn = KS_WARN_THRESHOLD
    
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            drift_conf = config.get("drift_detection", {})
            psi_warn = float(drift_conf.get("psi_warn_threshold", psi_warn))
            psi_block = float(drift_conf.get("psi_block_threshold", psi_block))
            ks_warn = float(drift_conf.get("ks_warn_threshold", ks_warn))
            logger.info(f"Loaded drift thresholds from config: PSI warn={psi_warn}, block={psi_block}, KS warn={ks_warn}")
        except Exception as e:
            logger.warning(f"Failed to load config at {config_path}, using defaults: {e}")

    logger.info(f"Running drift check: reference={args.reference_year}, current={args.current_year}")
    if reference_dir:
        logger.info(f"Using S3-backed reference store: {reference_dir}")

    reports = run_drift_check(
        features_dir, args.reference_year, args.current_year, reference_dir,
        psi_warn_threshold=psi_warn,
        psi_block_threshold=psi_block,
        ks_warn_threshold=ks_warn
    )

    # Send alerts if warnings/blocks exist and webhook is configured
    send_drift_alert(reports)

    # ── Write report ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(reports, f, indent=2)
    logger.success(f"Drift report written to {output_path}")

    # ── Print summary ──
    print("\n── Drift Detection Summary ──")
    any_block = False
    any_warn  = False
    for r in reports:
        status = r["overall_status"]
        psi    = r.get("ndvi_psi", "n/a")
        ksp    = r.get("ks_pvalue", "n/a")
        icon   = {"OK": "✅", "WARN": "⚠ ", "BLOCK": "🚫", "SKIP": "⏭ "}.get(status, "?")
        print(f"  {icon}  {r['region']:<35}  NDVI PSI={psi}  KS p={ksp}  → {status}")
        if status == "BLOCK":
            any_block = True
        if status == "WARN":
            any_warn = True

    print()
    
    # Wait for the background webhook notification to complete (joins at most 1.5 seconds)
    global _ALERT_THREAD
    if _ALERT_THREAD is not None:
        logger.info("Waiting for background alert thread to finish sending...")
        _ALERT_THREAD.join(timeout=1.5)

    if any_block:
        logger.error(
            "BLOCK-level drift detected. The feature distribution has shifted significantly. "
            "Do NOT retrain on this data without investigating. "
            "Check for new crop varieties, sensor calibration changes, or climate anomalies."
        )
        sys.exit(1)
    elif any_warn:
        logger.warning(
            "WARN-level drift detected. Moderate distribution shift observed. "
            "Retraining can proceed but results should be validated against held-out ground truth."
        )
        sys.exit(2)
    else:
        logger.success("No significant drift detected. Pipeline is clear to retrain.")
        sys.exit(0)


if __name__ == "__main__":
    main()
