import torch
import torch.nn as nn
from loguru import logger
from typing import Dict, Any

class IncrementalTrainer:
    """
    Handles training on new year data (e.g. 2024 climate patterns) 
    using Elastic Weight Consolidation (EWC) to prevent forgetting.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.ewc_lambda = config.get("ewc_lambda", 0.4)
        self.importance = {} # Fisher information matrix

    def compute_fisher_information(self, dataloader):
        """
        Estimate the importance of each parameter based on historical data.
        """
        logger.info("Estimating parameter importance (Fisher Information)...")
        self.model.eval()
        for batch in dataloader:
            self.model.zero_grad()
            sat, weather, soil = batch["sat"], batch["weather"], batch["soil"]
            output = self.model(sat, weather, soil)
            
            # Simple squared gradient approximation
            loss = torch.mean(output**2)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if name not in self.importance:
                        self.importance[name] = param.grad.data.clone().pow(2)
                    else:
                        self.importance[name] += param.grad.data.clone().pow(2)
        logger.success("Fisher information computed for incremental update.")

    def ewc_loss(self):
        """
        Penalty for moving parameters away from historical values.
        """
        loss = 0
        for name, param in self.model.named_parameters():
            if name in self.importance:
                # distance from current param to its historical state
                loss += (self.importance[name] * (param - param.detach()).pow(2)).sum()
        return self.ewc_lambda * loss

    def update_model_online(self, new_batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
        """
        Train on a single new batch while applying EWC regularization.
        """
        self.model.train()
        optimizer.zero_grad()
        
        sat, weather, soil = new_batch["sat"], new_batch["weather"], new_batch["soil"]
        labels = new_batch["label"]
        
        preds = self.model(sat, weather, soil)
        base_loss = nn.MSELoss()(preds, labels)
        
        total_loss = base_loss + self.ewc_loss()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
