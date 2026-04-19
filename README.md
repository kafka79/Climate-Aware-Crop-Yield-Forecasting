# Project: Climate-Aware Probabilistic Crop Yield Modeling

## 🌾 Overview
This project aims to predict crop yields with quantified uncertainty using multi-modal data fusion (Satellite, Weather, and Soil). It is designed to assist farmers and policymakers in Eastern India by providing explainable, risk-aware forecasts.

## 📂 Structure
- `data/`: Raw and processed datasets (Sentinel-1/2, ERA5, Yield).
- `src/`: Core Python modules for data processing, modeling, and evaluation.
- `configs/`: YAML files for model and training hyperparameters.
- `deployment/`: FastAPI backend and ONNX export for mobile deployment.
- `notebooks/`: Experimental and exploratory analysis.

## 🚀 Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your API keys in `.env`.
3. Run the pipeline:
   ```bash
   python main.py --config configs/data_config.yaml
   ```

## 🛠️ Tech Stack
- **Deep Learning**: PyTorch
- **Data Processing**: Xarray, Geopandas, Rasterio
- **Deployment**: FastAPI, ONNX Runtime
- **Explainability**: SHAP, Integrated Gradients
