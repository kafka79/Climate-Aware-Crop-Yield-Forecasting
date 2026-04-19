import torch
from torch.utils.data import Dataset
import numpy as np
from loguru import logger
from typing import Dict, List, Any

class MultiModalCropDataset(Dataset):
    """
    PyTorch Dataset for multi-modal crop yield prediction.
    Fuses spatial (Satellite), temporal (Weather), and static (Soil) data.
    """
    def __init__(self, sat_tensors: np.ndarray, weather_tensors: np.ndarray, 
                 soil_tensors: np.ndarray, yield_labels: np.ndarray):
        """
        sat_tensors: (N, T, C, H, W) or (N, T, C)
        weather_tensors: (N, T, F_w)
        soil_tensors: (N, F_s)
        yield_labels: (N, 1)
        """
        self.sat = torch.tensor(sat_tensors, dtype=torch.float32)
        self.weather = torch.tensor(weather_tensors, dtype=torch.float32)
        self.soil = torch.tensor(soil_tensors, dtype=torch.float32)
        self.labels = torch.tensor(yield_labels, dtype=torch.float32)
        
        logger.info(f"Initialized dataset with {len(self.labels)} samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sat": self.sat[idx],
            "weather": self.weather[idx],
            "soil": self.soil[idx],
            "label": self.labels[idx]
        }

def create_dataloaders(dataset: Dataset, batch_size: int, split_ratio: float = 0.8):
    """
    Split the dataset and create PyTorch DataLoaders.
    """
    logger.info(f"Creating dataloaders with batch_size={batch_size}...")
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
