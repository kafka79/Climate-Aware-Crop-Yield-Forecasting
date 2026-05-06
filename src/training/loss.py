
import torch.nn as nn
from loguru import logger
from src.models.mdn import mdn_loss

class CropYieldLoss(nn.Module):
    """
    Custom Loss class for Crop Yield Prediction.
    Supports both MDN (Probabilistic) and Standard (Point) predictions.
    """
    def __init__(self, mode: str = "probabilistic"):
        super(CropYieldLoss, self).__init__()
        self.mode = mode
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
    def forward(self, pred, target, pi=None, sigma=None, mu=None):
        if self.mode == "probabilistic":
            # pi, sigma, mu are required for MDN Loss
            if pi is None or sigma is None or mu is None:
                raise ValueError("MDN parameters (pi, sigma, mu) are required for probabilistic loss.")
            return mdn_loss(pi, sigma, mu, target)
        else:
            # Standard point prediction loss (MSE + MAE)
            return self.mse_loss(pred, target) + 0.1 * self.mae_loss(pred, target)

def get_loss_function(config: dict):
    """
    Factory function to get the loss function based on configuration.
    """
    logger.info(f"Initializing Loss Function in {config.get('mode', 'probabilistic')} mode...")
    return CropYieldLoss(mode=config.get("mode", "probabilistic"))
