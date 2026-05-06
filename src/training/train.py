import os
import json
import pandas as pd
import xarray as xr
import numpy as np
from loguru import logger
from src.utils.config import load_config
from src.models.transformer import initialize_model
from src.temporal.timeseries_dataset import MultiModalCropDataset, create_dataloaders
from src.training.trainer import TrainManager
from src.data.fusion import prepare_training_sequences
from src.inference.runtime import run_inference as runtime_inference
from src.evaluation.benchmark import YieldBenchmarker
import torch


def _align_soil_features(soil_df: pd.DataFrame, soil_dim: int) -> np.ndarray:
    soil_values = soil_df.select_dtypes(include=[np.number]).iloc[0].to_numpy(dtype=np.float32)
    if len(soil_values) >= soil_dim:
        return soil_values[:soil_dim]

    padded = np.zeros(soil_dim, dtype=np.float32)
    padded[: len(soil_values)] = soil_values
    return padded

def run_training_pipeline(config_path: str):
    """
    Complete training pipeline with Zarr lazy loading.
    """
    # 1. Load configs and merge
    config = load_config("configs/data_config.yaml")
    config.update(load_config("configs/model_config.yaml"))
    config.update(load_config("configs/training_config.yaml"))
    if os.path.normpath(config_path) != os.path.normpath("configs/data_config.yaml"):
        config.update(load_config(config_path))
    logger.info(f"Starting Training Pipeline: {config['project_name']}")
    
    # 2. Initialize Model
    model = initialize_model(config)
    
    # 3. Load Real Data
    yield_csv = config["paths"]["raw"]["yield"] + "/historical_yield.csv"
    if not os.path.exists(yield_csv):
        logger.error(f"Yield data not found at {yield_csv}. Run download/preprocess first.")
        return

    logger.info(f"Loading yield labels from {yield_csv}")
    yield_df = pd.read_csv(yield_csv)
    trained_regions = []
    
    processed_dir = config["paths"]["processed"]["features"]
    all_X, all_y, all_soil = [], [], []
    
    for area in config.get("study_areas", []):
        name = area["name"]
        sat_path = os.path.join(processed_dir, f"{name}_sat_proc.zarr")
        weather_path = os.path.join(processed_dir, f"{name}_weather_proc.zarr")
        
        if os.path.exists(sat_path) and os.path.exists(weather_path):
            logger.info(f"Fusing data for area: {name} (Lazy Zarr Load)")
            # Open Zarr datasets with chunks for Dask integration
            sat_ds = xr.open_zarr(sat_path)
            weather_ds = xr.open_zarr(weather_path)

            if "site_id" in yield_df.columns:
                area_yield_df = yield_df[yield_df["site_id"] == name].copy()
            else:
                area_yield_df = yield_df.copy()

            if area_yield_df.empty:
                logger.warning(f"No yield rows found for area: {name}. Skipping.")
                continue

            X_area, y_area = prepare_training_sequences(area_yield_df, sat_ds, weather_ds, config)
            
            if X_area is not None:
                all_X.append(X_area)
                all_y.append(y_area)
                trained_regions.append(name)
                
                # Handle Soil Data for this area
                soil_raw_dir = config["paths"]["raw"]["soil"]
                soil_path = os.path.join(soil_raw_dir, f"{name}_soil.csv")
                if os.path.exists(soil_path):
                    soil_df = pd.read_csv(soil_path)
                    soil_vec = _align_soil_features(
                        soil_df, config["transformer"].get("soil_dim", 3)
                    )
                    soil_vec = np.repeat(soil_vec.reshape(1, -1), len(y_area), axis=0)
                    all_soil.append(soil_vec)
    
    if not all_X:
        logger.error("No processed data found. Please run preprocess phase first.")
        return

    # Combine all areas
    X_combined = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    
    if all_soil:
        soil_final = np.concatenate(all_soil, axis=0)
    else:
        Fs = config["transformer"].get("soil_dim", 3)
        soil_final = np.zeros((len(X_combined), Fs))
    
    # Split into Sat and Weather
    C = config["transformer"]["input_dim"]
    sat_final = X_combined[:, :, :C]
    weather_final = X_combined[:, :, C:]
    
    dataset = MultiModalCropDataset(sat_final, weather_final, soil_final, y_final)
    train_loader, val_loader = create_dataloaders(dataset, 
                                                 batch_size=config["training"]["batch_size"],
                                                 split_ratio=config["training"]["val_split"])
    
    # 4. Initialize Trainer
    trainer = TrainManager(model, config)
    
    # 5. Run Training
    training_summary = trainer.run(train_loader, val_loader)

    os.makedirs(config["experiment"]["save_dir"], exist_ok=True)
    summary_path = os.path.join(
        config["experiment"]["save_dir"], "latest_training_summary.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "project_name": config["project_name"],
                "trained_regions": trained_regions,
                "num_sequences": int(len(dataset)),
                "best_val_loss": training_summary["best_val_loss"],
                "epochs": training_summary["epochs"],
                "checkpoint_path": os.path.join(
                    config["training"].get("save_path", "models/checkpoints"),
                    "best_model.pth",
                ),
            },
            f,
            indent=2,
        )
    logger.info(f"Training summary written to {summary_path}")


def run_benchmark_pipeline(config_path: str):
    """
    Evaluates the current best model on the available processed data.
    """
    config = load_config("configs/data_config.yaml")
    config.update(load_config("configs/model_config.yaml"))
    config.update(load_config("configs/training_config.yaml"))
    if os.path.normpath(config_path) != os.path.normpath("configs/data_config.yaml"):
        config.update(load_config(config_path))

    logger.info("Starting Benchmark Pipeline...")
    
    # 1. Load Model
    model = initialize_model(config)
    checkpoint_path = "models/checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found at {checkpoint_path}. Train the model first.")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    
    # 2. Prepare Data (Reuse logic from run_training_pipeline but maybe just use a subset or the val set)
    # For simplicity in this benchmark mode, we'll evaluate on the entire processed dataset
    yield_csv = config["paths"]["raw"]["yield"] + "/historical_yield.csv"
    yield_df = pd.read_csv(yield_csv)
    processed_dir = config["paths"]["processed"]["features"]
    
    all_X, all_y, all_soil = [], [], []
    for area in config.get("study_areas", []):
        name = area["name"]
        sat_path = os.path.join(processed_dir, f"{name}_sat_proc.zarr")
        weather_path = os.path.join(processed_dir, f"{name}_weather_proc.zarr")
        if os.path.exists(sat_path) and os.path.exists(weather_path):
            sat_ds = xr.open_zarr(sat_path)
            weather_ds = xr.open_zarr(weather_path)
            area_yield_df = yield_df[yield_df["site_id"] == name] if "site_id" in yield_df.columns else yield_df
            X_area, y_area = prepare_training_sequences(area_yield_df, sat_ds, weather_ds, config)
            if X_area is not None:
                all_X.append(X_area)
                all_y.append(y_area)
                soil_raw_dir = config["paths"]["raw"]["soil"]
                soil_path = os.path.join(soil_raw_dir, f"{name}_soil.csv")
                if os.path.exists(soil_path):
                    soil_df = pd.read_csv(soil_path)
                    soil_vec = _align_soil_features(soil_df, config["transformer"].get("soil_dim", 3))
                    soil_vec = np.repeat(soil_vec.reshape(1, -1), len(y_area), axis=0)
                    all_soil.append(soil_vec)

    if not all_X:
        logger.error("No data found for benchmarking.")
        return

    X_combined = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    C = config["transformer"]["input_dim"]
    sat_final = X_combined[:, :, :C]
    weather_final = X_combined[:, :, C:]
    
    if all_soil:
        soil_final = np.concatenate(all_soil, axis=0)
    else:
        soil_final = np.zeros((len(X_combined), config["transformer"].get("soil_dim", 3)))

    dataset = MultiModalCropDataset(sat_final, weather_final, soil_final, y_final)
    # Using a larger batch size for inference
    _, val_loader = create_dataloaders(dataset, batch_size=32, split_ratio=0.99) 

    # 3. Run Benchmark
    benchmarker = YieldBenchmarker(config)
    metrics = benchmarker.evaluate(model, val_loader)
    benchmarker.save_report(metrics)
