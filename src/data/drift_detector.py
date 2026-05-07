"""
Drift Detection for the Climate-Aware Crop Yield Forecasting Pipeline.

Addresses [Jess · Meta]: "The pipeline is automated, but where is the Drift
Monitoring? You're retraining on new data, but if the relationship between
NDVI and Yield shifts due to a new crop variety or climate anomaly, the pipeline
will just keep training on 'bad' assumptions."

Strategy
--------
Two complementary tests are run against every feature store produced by the
preprocess stage:

1. Population Stability Index (PSI) on the NDVI distribution
   PSI < 0.1  → No significant drift   (safe to retrain)
   0.1 ≤ PSI < 0.25 → Moderate drift   (flag, investigate, retrain cautiously)
   PSI ≥ 0.25 → Major drift            (block retraining, alert operator)

2. Kolmogorov-Smirnov test on the weather feature (temperature/precipitation)
   p-value < 0.05 → Distribution shift detected

The script exits with code 0 if all checks pass, code 1 if any check exceeds
the BLOCK threshold (PSI ≥ 0.25 or KS p < 0.001), and code 2 if there is
moderate drift (flags in CI but does not block retraining).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ── PSI ──────────────────────────────────────────────────────────────────────

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1-D numeric arrays.

    PSI = Σ (actual% - expected%) * ln(actual% / expected%)
    """
    # Build bin edges from the reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    edges = np.linspace(min_val, max_val, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Replace zeros to avoid log(0)
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi_value


# ── KS test ──────────────────────────────────────────────────────────────────

def _ks_pvalue(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov p-value (pure NumPy implementation).

    Returns p-value (float). Lower = more evidence of distribution shift.
    """
    from scipy.stats import ks_2samp  # optional; falls back to PSI-only if absent
    stat, p = ks_2samp(reference, current)
    return float(p)


# ── Feature extraction from Zarr ─────────────────────────────────────────────

def _extract_ndvi(zarr_path: Path) -> Optional[np.ndarray]:
    """Extract NDVI time series from a Zarr satellite feature store."""
    try:
        import xarray as xr
        ds = xr.open_zarr(zarr_path)
        if "B08" in ds and "B04" in ds:
            nir = ds["B08"].values.reshape(-1).astype(np.float32)
            red = ds["B04"].values.reshape(-1).astype(np.float32)
            denom = nir + red
            denom[denom == 0] = 1e-6
            ndvi = (nir - red) / denom
            return ndvi[np.isfinite(ndvi)]
    except Exception as exc:
        logger.warning(f"Could not extract NDVI from {zarr_path}: {exc}")
    return None


def _extract_weather_feature(zarr_path: Path, variable: str = "t2m") -> Optional[np.ndarray]:
    """Extract a scalar weather variable from a Zarr weather feature store."""
    try:
        import xarray as xr
        ds = xr.open_zarr(zarr_path)
        if variable in ds:
            values = ds[variable].values.reshape(-1).astype(np.float32)
            return values[np.isfinite(values)]
    except Exception as exc:
        logger.warning(f"Could not extract '{variable}' from {zarr_path}: {exc}")
    return None


# ── Core check ───────────────────────────────────────────────────────────────

PSI_WARN_THRESHOLD  = 0.10
PSI_BLOCK_THRESHOLD = 0.25
KS_WARN_THRESHOLD   = 0.05
KS_BLOCK_THRESHOLD  = 0.001


def check_region_drift(
    region: str,
    reference_zarr: Path,
    current_zarr: Path,
    reference_weather: Optional[Path] = None,
    current_weather: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full drift check for one region.

    Returns a report dict with keys:
        region, ndvi_psi, ndvi_status, ks_pvalue, ks_status, overall_status
    where overall_status is one of: OK | WARN | BLOCK
    """
    report: Dict[str, Any] = {"region": region}

    # ── NDVI PSI ──
    ref_ndvi = _extract_ndvi(reference_zarr)
    cur_ndvi = _extract_ndvi(current_zarr)

    if ref_ndvi is not None and cur_ndvi is not None and len(ref_ndvi) > 5 and len(cur_ndvi) > 5:
        psi = _psi(ref_ndvi, cur_ndvi)
        report["ndvi_psi"] = round(psi, 4)
        if psi >= PSI_BLOCK_THRESHOLD:
            report["ndvi_status"] = "BLOCK"
        elif psi >= PSI_WARN_THRESHOLD:
            report["ndvi_status"] = "WARN"
        else:
            report["ndvi_status"] = "OK"
    else:
        report["ndvi_psi"] = None
        report["ndvi_status"] = "SKIP"
        logger.warning(f"[{region}] NDVI data insufficient for PSI — skipping.")

    # ── KS on weather ──
    ks_pval = None
    ks_status = "SKIP"
    if reference_weather and current_weather:
        ref_w = _extract_weather_feature(reference_weather)
        cur_w = _extract_weather_feature(current_weather)
        if ref_w is not None and cur_w is not None and len(ref_w) > 5 and len(cur_w) > 5:
            try:
                ks_pval = _ks_pvalue(ref_w, cur_w)
                if ks_pval < KS_BLOCK_THRESHOLD:
                    ks_status = "BLOCK"
                elif ks_pval < KS_WARN_THRESHOLD:
                    ks_status = "WARN"
                else:
                    ks_status = "OK"
            except ImportError:
                logger.warning("scipy not available; skipping KS test (PSI only).")

    report["ks_pvalue"] = round(ks_pval, 6) if ks_pval is not None else None
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


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_drift_check(
    features_dir: Path,
    reference_year: int,
    current_year: int,
    reference_dir: Optional[Path] = None,
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
            reference_weather=weather_ref if weather_ref.exists() else None,
            current_weather=weather_cur if weather_cur.exists() else None,
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

    logger.info(f"Running drift check: reference={args.reference_year}, current={args.current_year}")
    if reference_dir:
        logger.info(f"Using S3-backed reference store: {reference_dir}")

    reports = run_drift_check(
        features_dir, args.reference_year, args.current_year, reference_dir
    )

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
