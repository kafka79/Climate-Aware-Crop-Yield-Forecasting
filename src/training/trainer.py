import torch
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
from typing import Dict, Any
import os

class TrainManager:
    """
    Manages the training and validation loops for multi-modal crop yield.
    """
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.full_config = config
        self.config = config["training"]
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        
        # Optimizer & Scheduler
        self.optimizer = optim.Adam(model.parameters(), 
                                    lr=self.config["learning_rate"], 
                                    weight_decay=self.config.get("weight_decay", 1e-5))
        
        # Optional Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        # Initializing Loss
        from src.training.loss import CropYieldLoss
        self.criterion = CropYieldLoss(mode=self.config.get("mode", "deterministic"))

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            # Move to device
            sat = batch["sat"].to(self.device)
            weather = batch["weather"].to(self.device)
            soil = batch["soil"].to(self.device)
            labels = batch["label"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            # Note: Model must handle the separate modalities or combined tensor
            preds = self.model(sat, weather, soil)
            
            if isinstance(preds, tuple):
                pi, sigma, mu = preds
                loss = self.criterion(None, labels, pi, sigma, mu)
            else:
                loss = self.criterion(preds, labels)
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                sat = batch["sat"].to(self.device)
                weather = batch["weather"].to(self.device)
                soil = batch["soil"].to(self.device)
                labels = batch["label"].to(self.device)
                
                preds = self.model(sat, weather, soil)
                if isinstance(preds, tuple):
                    pi, sigma, mu = preds
                    loss = self.criterion(None, labels, pi, sigma, mu)
                else:
                    loss = self.criterion(preds, labels)
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

    def run(self, train_loader, val_loader):
        logger.info(f"Starting training on {self.device}...")
        best_val_loss = float("inf")
        save_path = self.config.get("save_path", "models/checkpoints")
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(self.config["num_epochs"]):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # Step scheduler
            self.scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_file = os.path.join(save_path, "best_model.pth")
                torch.save(self.model.state_dict(), ckpt_file)
                logger.info(f"New best model saved to {ckpt_file} with Val Loss: {val_loss:.4f}")
                
        logger.success("Training run complete.")
        return {
            "best_val_loss": best_val_loss,
            "epochs": self.config["num_epochs"],
        }
