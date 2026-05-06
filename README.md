# Climate-Aware Probabilistic Crop Yield Forecasting

This repository is a working ML prototype for crop-yield forecasting with uncertainty, built around a multi-modal PyTorch model that combines:

- Sentinel-2 satellite features
- Weather time series
- Soil features
- A Transformer backbone with an MDN head

The repo includes training code, a Streamlit dashboard, a FastAPI service, Docker support, and a pytest suite.

## Current Status

This project now draws a hard line between real artifacts and placeholder behavior:

- `main.py --mode predict` uses the stored checkpoint and processed feature store, or exits clearly if those artifacts are not usable.
- The Streamlit dashboard only renders a live forecast when the selected region/year has matching processed Zarr data and a checkpoint.
- The dashboard defaults to historical context instead of simulated forecasts.
- The FastAPI service is available in `docker-compose.yml` at `http://localhost:8000`.

## What Is Actually Backed By Data

Committed artifacts in this repo currently include:

- Historical yield records in `data/raw/yield/historical_yield.csv`
- A processed Sentinel-2 and weather feature store for `Burdwan, West Bengal` in `2023`
- A saved checkpoint at `models/checkpoints/best_model.pth`

Important limitation:

- The committed dashboard artifacts are not crop-specific, so crop selection has been removed from the UI.
- Not every configured study area has processed Zarr features in the repo yet.
- The old benchmark table has been removed until a reproducible experiment report is committed alongside it.

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

## Training And Reproducibility

To regenerate artifacts locally:

```bash
python main.py --mode download
python main.py --mode preprocess
python main.py --mode train
```

Training writes:

- `models/checkpoints/best_model.pth`
- `experiments/latest_training_summary.json`

The training summary captures the trained regions, number of sequences, epoch count, and best validation loss for that run.

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
|-- notebooks/
|-- src/
|   |-- cli/
|   |-- data/
|   |-- evaluation/
|   |-- explainability/
|   |-- features/
|   |-- inference/
|   |-- models/
|   |-- recommendation/
|   |-- risk/
|   |-- temporal/
|   `-- training/
`-- tests/
```

## Tests

Run the test suite with:

```bash
python -m pytest tests/ -v
```

The suite covers MDN behavior, feature engineering, risk classification, preprocessing utilities, temporal sequence building, and the new runtime inference/context helpers.
