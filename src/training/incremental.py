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
        self.optimal_weights = {name: param.detach().clone() for name, param in self.model.named_parameters() if param.requires_grad}

    def compute_fisher_information(self, dataloader):
        """
        Estimate the importance of each parameter based on historical data.
        """
        logger.info("Estimating parameter importance (Fisher Information)...")
        self.model.eval()
        device = next(self.model.parameters()).device
        for batch in dataloader:
            self.model.zero_grad()
            sat = batch["sat"].to(device)
            weather = batch["weather"].to(device)
            soil = batch["soil"].to(device)
            output = self.model(sat, weather, soil)
            
            if isinstance(output, tuple):
                pi, sigma, mu = output
                from src.models.mdn import mdn_expected_value
                output = mdn_expected_value(pi, sigma, mu)
            
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
                loss += (self.importance[name] * (param - self.optimal_weights[name]).pow(2)).sum()
        return self.ewc_lambda * loss

    def update_model_online(self, new_batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
        """
        Train on a single new batch while applying EWC regularization.
        """
        self.model.train()
        optimizer.zero_grad()
        
        device = next(self.model.parameters()).device
        sat = new_batch["sat"].to(device)
        weather = new_batch["weather"].to(device)
        soil = new_batch["soil"].to(device)
        labels = new_batch["label"].to(device)
        
        preds = self.model(sat, weather, soil)
        if isinstance(preds, tuple):
            pi, sigma, mu = preds
            from src.models.mdn import mdn_loss
            base_loss = mdn_loss(pi, sigma, mu, labels)
        else:
            base_loss = nn.MSELoss()(preds, labels)
        
        total_loss = base_loss + self.ewc_loss()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()
