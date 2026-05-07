import torch
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
from typing import Any, Dict, Optional
import os


class TrainManager:
    """
    Manages the training and validation loops for multi-modal crop yield.

    FLAW FIX [Rohan · Google]: "If a GitHub runner dies at minute 58, you have no
    checkpointing logic to resume. You'd lose the entire compute cost of that hour."

    Resolution: every epoch writes a full resumable checkpoint containing model
    weights, optimizer state, scheduler state, current epoch index, and best
    validation loss. On __init__, the trainer checks for a resume checkpoint and
    fast-forwards training to where it left off — a runner failure loses at most
    one epoch, not the entire run.
    """

    RESUME_CKPT_NAME = "resume_checkpoint.pth"

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.full_config = config
        self.config = config["training"]
        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 1e-5),
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        from src.training.loss import CropYieldLoss
        self.criterion = CropYieldLoss(mode=self.config.get("mode", "deterministic"))

        # Resume state (populated in run() if a resume checkpoint is found)
        self._start_epoch: int = 0
        self._best_val_loss: float = float("inf")

    # ── Training / Validation loops ───────────────────────────────────────────

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            sat     = batch["sat"].to(self.device)
            weather = batch["weather"].to(self.device)
            soil    = batch["soil"].to(self.device)
            labels  = batch["label"].to(self.device)

            self.optimizer.zero_grad()
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
                sat     = batch["sat"].to(self.device)
                weather = batch["weather"].to(self.device)
                soil    = batch["soil"].to(self.device)
                labels  = batch["label"].to(self.device)

                preds = self.model(sat, weather, soil)
                if isinstance(preds, tuple):
                    pi, sigma, mu = preds
                    loss = self.criterion(None, labels, pi, sigma, mu)
                else:
                    loss = self.criterion(preds, labels)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _resume_checkpoint_path(self, save_path: str) -> str:
        return os.path.join(save_path, self.RESUME_CKPT_NAME)

    def _save_resume_checkpoint(
        self, save_path: str, epoch: int, best_val_loss: float
    ) -> None:
        """Write a full resumable checkpoint after every epoch."""
        ckpt = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        path = self._resume_checkpoint_path(save_path)
        torch.save(ckpt, path)
        logger.debug(f"Resume checkpoint saved → {path} (epoch {epoch + 1})")

    def _load_resume_checkpoint(self, save_path: str) -> Optional[int]:
        """If a resume checkpoint exists, restore all state and return start epoch."""
        path = self._resume_checkpoint_path(save_path)
        if not os.path.exists(path):
            return None

        logger.warning(
            f"Resume checkpoint found at {path}. "
            "Restoring model / optimizer / scheduler state — training will "
            "continue from where the previous run stopped."
        )
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._best_val_loss = ckpt["best_val_loss"]
        return int(ckpt["epoch"]) + 1  # resume from NEXT epoch

    # ── Main training loop ────────────────────────────────────────────────────

    def run(self, train_loader, val_loader):
        save_path = self.config.get("save_path", "models/checkpoints")
        os.makedirs(save_path, exist_ok=True)

        # Auto-resume if a previous run was interrupted
        resumed_epoch = self._load_resume_checkpoint(save_path)
        start_epoch = resumed_epoch if resumed_epoch is not None else 0
        best_val_loss = self._best_val_loss
        num_epochs = self.config["num_epochs"]

        if start_epoch > 0:
            logger.info(
                f"Resuming from epoch {start_epoch + 1}/{num_epochs} "
                f"(best val loss so far: {best_val_loss:.4f})"
            )
        else:
            logger.info(f"Starting training on {self.device}...")

        for epoch in range(start_epoch, num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.validate(val_loader)
            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Save best model weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(save_path, "best_model.pth")
                torch.save(self.model.state_dict(), best_path)
                logger.info(
                    f"✅ New best model → {best_path} (Val Loss: {val_loss:.4f})"
                )

            # Always write resumable checkpoint so a runner death wastes ≤ 1 epoch
            self._save_resume_checkpoint(save_path, epoch, best_val_loss)

        # Clean up resume checkpoint on successful completion so next run starts fresh
        resume_path = self._resume_checkpoint_path(save_path)
        if os.path.exists(resume_path):
            os.remove(resume_path)
            logger.info("Training complete — resume checkpoint removed.")

        logger.success("Training run complete.")
        return {
            "best_val_loss": best_val_loss,
            "epochs": num_epochs,
        }

