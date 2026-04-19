import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, Any

class LSTM_Baseline(nn.Module):
    """
    Standard LSTM architecture for weather time-series.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int = 1):
        super(LSTM_Baseline, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take last hidden state
        return self.fc(out)

class XGBoost_Baseline:
    """
    XGBoost baseline for tabular soil and yield data.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None

    def train(self, X_train, y_train):
        logger.info("Training XGBoost baseline...")
        # from xgboost import XGBRegressor
        # self.model = XGBRegressor(...)
        # self.model.fit(X_train, y_train)
        pass

    def predict(self, X):
        return self.model.predict(X)

def initialize_baselines(config: Dict[str, Any]):
    """
    Initialize baseline models for comparison.
    """
    logger.info("Initializing LSTM and XGBoost baselines...")
    return LSTM_Baseline(config["input_dim"], config["hidden_dim"], config["num_layers"])
