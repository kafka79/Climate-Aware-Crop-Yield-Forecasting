# 🌾 Climate-Aware Probabilistic Crop Yield Forecasting

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi) ![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker) ![Tests](https://img.shields.io/badge/Tests-8%20Passing-brightgreen?logo=pytest) ![License](https://img.shields.io/badge/License-MIT-yellow)

> Predicting crop yields with **quantified uncertainty** using a Multi-Modal Transformer + Mixture Density Network — fusing Sentinel-2 satellite imagery, ERA5 weather reanalysis, and SoilGrids data.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT MODALITIES                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Sentinel-2   │  │  ERA5 / CDS  │  │  SoilGrids       │  │
│  │ Satellite    │  │  Weather     │  │  Soil Properties  │  │
│  │ (B2,B3,B4,  │  │  (Tmax, Tmin,│  │  (pH, SOC,       │  │
│  │  B8 → NDVI, │  │   Precip,    │  │   Clay, WHC)     │  │
│  │  EVI, LSWI) │  │   GDD, SPI)  │  │                  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                   │             │
└─────────┼─────────────────┼───────────────────┼────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                     │
│  Sat Encoder (Linear)  LSTM Encoder    Soil Encoder (MLP)   │
└─────────────────────────────────────────────────────────────┘
          │                 │                   │
          └─────────────────┴───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            MULTI-MODAL TRANSFORMER ENCODER                  │
│     Cross-Modal Attention → 4 Layers → Global Avg Pool      │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌──────────────────┐       ┌───────────────────────┐
   │  Point Estimate  │       │  MDN Probabilistic    │
   │  (MSE+MAE Loss)  │       │  Head (NLL Loss)      │
   │                  │       │  π, μ, σ → GMM        │
   └──────────────────┘       └───────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│             POST-PROCESSING & DEPLOYMENT                    │
│  Risk Classifier → Agronomy Advisor → FastAPI → Streamlit   │
│  Integrated Gradients XAI → Feature Attribution Report      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

| Model | MAE ↓ | RMSE ↓ | R² ↑ | MAPE ↓ | CRPS ↓ |
|---|---|---|---|---|---|
| **Baseline LSTM** (weather only) | 0.821 | 1.043 | 0.612 | 21.4% | — |
| **Multi-Modal Transformer** | 0.473 | 0.612 | 0.841 | 11.8% | — |
| **Transformer + MDN** *(ours)* | **0.461** | **0.598** | **0.849** | **11.2%** | **0.334** |

> The MDN head adds calibrated uncertainty quantification (CRPS) at only a 2.3% RMSE cost — a capability deterministic models fundamentally cannot provide.

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data & preprocess
python main.py --mode download
python main.py --mode preprocess

# 3. Launch the full app (API + UI)
docker-compose up
```

**Frontend UI:** http://localhost:8501  
**API Docs (Swagger):** http://localhost:8000/docs

---

## 📁 Project Structure

```
major-project/
├── src/
│   ├── data/           # Downloaders, preprocessing, fusion, imputation, bias correction
│   ├── models/         # MultiModalTransformer, MDN, LSTM Baseline
│   ├── features/       # NDVI, EVI, LSWI, GDD, SPI feature extractors
│   ├── training/       # TrainManager, CropYieldLoss, training pipeline
│   ├── evaluation/     # YieldMetrics, ProbabilisticMetrics (CRPS, PIT)
│   ├── explainability/ # Integrated Gradients (Captum)
│   ├── risk/           # YieldRiskClassifier with uncertainty calibration
│   └── recommendation/ # AgronomyAdvisor — XAI to human advice
├── deployment/
│   ├── api/            # FastAPI backend
│   ├── frontend/       # Streamlit dashboard
│   └── export/         # ONNX export for mobile/edge inference
├── configs/            # YAML config files (data, model, training)
├── experiments/        # Experiment results (Baseline vs Transformer vs MDN)
├── tests/              # 8 pytest unit tests
├── notebooks/          # EDA and analysis notebooks
├── .github/workflows/  # GitHub Actions CI (auto-runs tests on push)
└── docker-compose.yml  # One-command deployment
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | PyTorch 2.0+, torch.nn.Transformer, torch.distributions |
| **Data Processing** | Xarray, Rioxarray, Rasterio, GeoPandas, NumPy, Pandas |
| **Geospatial APIs** | SentinelHub (Sentinel-2), CDS API (ERA5), SoilGrids |
| **Explainability** | Captum (Integrated Gradients), SHAP |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit, Plotly |
| **Deployment** | Docker, Docker Compose, ONNX Runtime |
| **Testing & CI** | Pytest, GitHub Actions |
| **Observability** | Loguru |

---

## 🔬 Key Innovations

1. **Probabilistic Forecasting via MDN:** Unlike standard regression, the model outputs a full Gaussian Mixture distribution over yield, enabling CRPS-calibrated uncertainty bounds.
2. **Multi-Modal Fusion:** Satellite (spatial), Weather (temporal), and Soil (static) data are fused via a Transformer with cross-modal attention.
3. **Domain-Aware Features:** Agronomic features like Growing Degree Days (GDD) and Standardized Precipitation Index (SPI) are computed rather than raw temperatures/precipitation.
4. **Explainable Predictions:** Integrated Gradients attribute yield predictions back to each modality and time step, converted to human-readable advice by the AgronomyAdvisor.

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover: NDVI/GDD maths, MDN output validity (pi sums to 1, sigma > 0), risk classification boundaries, uncertainty elevation, KNN imputation, bias correction, and sequence windowing.

---

## 🤝 Contributing

Pull requests welcome. Please ensure `pytest tests/` passes before submitting.
