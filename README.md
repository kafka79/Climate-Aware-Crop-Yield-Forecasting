# Climate-Aware Probabilistic Crop Yield Forecasting

This repository is a production-grade ML platform for crop-yield forecasting with uncertainty, built around a multi-modal PyTorch model that combines:

- Sentinel-2 satellite features
- Weather time series
- Soil features
- A Transformer backbone with a Mixture Density Network (MDN) head

The repo includes distributed training code, a Streamlit dashboard, a FastAPI service, Docker support, and a rigorous CI/CD pipeline integrated with AWS via OIDC.

## Current Status

This project now draws a hard line between real artifacts and placeholder behavior, and implements enterprise-level scale and security:

- **Offline-Capable PWA:** The Streamlit dashboard is configured as a Progressive Web App (PWA) with Service Worker caching. If a farmer is in the field with 2G connectivity, the app loads offline and serves the last successfully fetched forecast.
- **Bimodal Uncertainty Safety:** The MDN explicitly tracks distribution shapes. If the model outputs a bimodal distribution (e.g., conflicting drought vs. flood scenarios), the inference runtime refuses to serve a potentially misleading "valley-mean" point estimate, instead exposing the distinct dominant scenarios via the UI and CLI.
- **Modality Validation:** If required data like soil inputs are missing, the system actively raises a `MissingSoilDataWarning` and propagates the warning state to the dashboard rather than failing silently or hallucinating values.

## Architecture & Enterprise Hardening

### 1. Zero-Static-Secret Cloud Access (OIDC)
GitHub Actions communicates with AWS using OpenID Connect (OIDC). There are **zero long-lived AWS API keys** stored in GitHub Secrets. Workflows assume short-lived STS tokens for absolute supply-chain security.

### 2. S3-Backed Immutable Feature Store & Drift Detection
The CI pipeline automatically evaluates new data for covariate shift before training.
- Stable, baseline reference Zarr feature stores are synced from an immutable S3 bucket prefix (`s3://<bucket>/drift-reference/`).
- The `drift_detector.py` uses Population Stability Index (PSI) and Kolmogorov-Smirnov (K-S) tests to compare live data against this pinned reference.
- Any statistically significant drift halts the pipeline *before* compute is wasted.

### 3. Fault-Tolerant Resumable Training
Training on ephemeral runners is inherently risky. `trainer.py` writes complete state checkpoints (Model weights, Optimizer state, Scheduler state, current epoch) to `models/checkpoints/resume_checkpoint.pth` at the end of every epoch.
- The pipeline synchronizes these checkpoints with S3.
- If a runner dies or times out (e.g., at minute 89), the next triggered run automatically downloads the checkpoint from S3 and resumes at the exact epoch it left off, preventing catastrophic compute loss.

### 4. Dynamic SageMaker Dispatching
For massive datasets, the pipeline automatically outgrows GitHub Actions:
- `sagemaker_launcher.py` measures the local feature store Zarr size.
- If the dataset is under the configurable threshold (default 5.0 GiB), the pipeline trains locally on the GitHub runner.
- If it exceeds the threshold, the launcher dynamically dispatches the training job to an AWS SageMaker Managed Spot Instance (`ml.p3.2xlarge` V100 GPU), tracks the job to completion, and pulls the final `best_model.pth` back from S3.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard and API

```bash
docker-compose up
```

Available services:
- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### 3. Run a CLI prediction

```bash
python main.py --mode predict --region "Burdwan, West Bengal" --year 2023
```

If the selected region/year is missing processed artifacts, or the checkpoint output fails a plausibility guard, the command exits with a clear error instead of printing fake numbers.

## Automated Data Pipeline

The full download → preprocess → drift detection → train → validate cycle runs automatically via GitHub Actions (`.github/workflows/data_pipeline.yml`).

| Trigger | When |
|---------|------|
| **Scheduled** | Every Monday 02:00 UTC (aligned with Sentinel-2 revisit cycle) |
| **Push** | Any commit to `data/raw/`, `configs/data_config.yaml`, or model configs on `main` |
| **Manual** | Workflow dispatch with optional region/year overrides |

## Project Structure

```text
major-project/
|-- app.py
|-- main.py
|-- configs/
|-- data/
|-- deployment/
|-- experiments/
|-- models/
|-- src/
|   |-- data/            # Preprocessing & Drift Detection
|   |-- inference/       # Bimodal safety & Modality tracking
|   |-- models/          # Transformer + MDN
|   `-- training/        # Checkpointing & SageMaker Launcher
|-- static/              # PWA Service Worker, Manifest, Icons
`-- tests/               # Pytest suite
```

## Tests

Run the test suite with:

```bash
python -m pytest tests/ -v
```

The suite covers MDN behavior, bimodal detection, feature engineering, and the new runtime inference/context helpers.
