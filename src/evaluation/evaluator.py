import os
import torch
import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, Any, List
from src.evaluation.metrics import YieldMetrics
from src.evaluation.probabilistic_metrics import ProbabilisticMetrics
from src.models.transformer import initialize_model
from src.temporal.timeseries_dataset import MultiModalCropDataset, create_dataloaders

class EvaluationManager:
    """
    Manages the evaluation of trained crop yield models.
    Supports standard and probabilistic metrics.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_engine = YieldMetrics(config)
        self.probabilistic_engine = ProbabilisticMetrics(config)
        
    def load_best_model(self, model_path: str):
        """
        Load the best model from a checkpoint.
        """
        logger.info(f"Loading best model from {model_path}...")
        model = initialize_model(self.config)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def run_evaluation(self, model, val_loader):
        """
        Run the full evaluation suite on a given dataloader.
        """
        logger.info("Starting model evaluation...")
        all_preds = []
        all_labels = []
        all_pi, all_sigma, all_mu = [], [], []
        
        is_probabilistic = self.config.get("mode") == "probabilistic"
        
        with torch.no_grad():
            for batch in val_loader:
                sat = batch["sat"].to(self.device)
                weather = batch["weather"].to(self.device)
                soil = batch["soil"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = model(sat, weather, soil)
                
                if is_probabilistic:
                    pi, sigma, mu = outputs
                    all_pi.append(pi.cpu().numpy())
                    all_sigma.append(sigma.cpu().numpy())
                    all_mu.append(mu.cpu().numpy())
                    # For standard metrics, we use the mean of the mixture
                    # (Simplified: assuming components are sorted or picking the most likely)
                    all_preds.append(torch.sum(pi * mu, dim=1).cpu().numpy())
                else:
                    all_preds.append(outputs.cpu().numpy())
                
                all_labels.append(labels.cpu().numpy())
                
        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)
        
        # 1. Standard Metrics
        results = self.metrics_engine.calculate_all(y_true, y_pred)
        
        # 2. Probabilistic Metrics (if applicable)
        if is_probabilistic:
            pi = np.concatenate(all_pi, axis=0)
            sigma = np.concatenate(all_sigma, axis=0)
            mu = np.concatenate(all_mu, axis=0)
            prob_results = self.probabilistic_engine.calculate_crps_gmm(y_true, pi, sigma, mu)
            results.update(prob_results)
            
        return results, y_pred, y_true

    def run(self, val_loader=None):
        """
        Orchestrates the full evaluation process.
        """
        model_path = os.path.join(self.config["training"]["save_path"], "best_model.pth")
        if not os.path.exists(model_path):
            logger.error(f"Best model not found at {model_path}. Cannot evaluate.")
            return
            
        model = self.load_best_model(model_path)
        
        # If val_loader is not provided, we should ideally load the test set
        if val_loader is None:
            # Placeholder for loading the test/val set from processed data
            logger.warning("No dataloader provided. Evaluation skipped.")
            return
            
        results, y_pred, y_true = self.run_evaluation(model, val_loader)
        
        # Save results
        report_path = os.path.join(self.config["paths"]["processed"]["features"], "evaluation_report")
        self.metrics_engine.save_results(results, report_path)
        
        logger.success(f"Evaluation complete. Results saved to {report_path}")
        return results
