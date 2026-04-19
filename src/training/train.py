import os
import torch
import pandas as pd
import xarray as xr
from loguru import logger
from src.utils.config import load_config
from src.models.transformer import initialize_model
from src.temporal.timeseries_dataset import MultiModalCropDataset, create_dataloaders
from src.training.trainer import TrainManager
from src.data.fusion import prepare_training_sequences
from typing import Dict, Any
import numpy as np

def run_training_pipeline(config_path: str):
    """
    Complete training pipeline from config to best model saving.
    """
    # 1. Load configs and merge
    config = load_config("configs/data_config.yaml")
    config.update(load_config("configs/model_config.yaml"))
    config.update(load_config("configs/training_config.yaml"))
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
    
    # In a real run, we load the processed multi-modal stacks
    # For now, we assume at least one area is processed
    processed_dir = config["paths"]["processed"]["features"]
    
    # Placeholder for loading all available processed areas
    # In a full impl, we would loop through config['study_areas']
    all_X, all_y = [], []
    
    for area in config.get("study_areas", []):
        name = area["name"]
        sat_path = os.path.join(processed_dir, f"{name}_sat_proc.nc")
        weather_path = os.path.join(processed_dir, f"{name}_weather_proc.nc")
        
        if os.path.exists(sat_path) and os.path.exists(weather_path):
            logger.info(f"Fusing data for area: {name}")
            sat_ds = xr.open_dataset(sat_path)
            weather_ds = xr.open_dataset(weather_path)
            
            # Filter yield_df for this area if needed
            X_area, y_area = prepare_training_sequences(yield_df, sat_ds, weather_ds, config)
            if X_area is not None:
                all_X.append(X_area)
                all_y.append(y_area)
    
    if not all_X:
        logger.error("No processed data found for training. Please run download and preprocess phases first.")
        return
    else:
        # Assuming all_X contains combined sat and weather for now, splitting them
        X_combined = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        
        Fs = config["transformer"].get("soil_dim", 4)
        soil_final = np.zeros((len(X_combined), Fs))
        
        # Split X_combined into sat and weather based on config dims
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
    trainer.run(train_loader, val_loader)
