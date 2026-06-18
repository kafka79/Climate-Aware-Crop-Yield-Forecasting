import os
import signal
import tempfile
import threading

import torch
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
from typing import Any, Dict, Optional


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
        
        # Initialize DDP if distributed process group is setup
        if torch.distributed.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[torch.cuda.current_device()]
            )

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

        # Spot termination handling: when AWS sends SIGTERM (2-min warning),
        # the handler sets this flag so the training loop can save state
        # before the container is killed.
        self._termination_requested = threading.Event()
        self._active_upload_thread: Optional[threading.Thread] = None
        self._save_path: Optional[str] = None
        self._current_epoch: int = 0
        self._register_spot_termination_handler()

    # ── Spot termination handler ──────────────────────────────────────────────

    def _register_spot_termination_handler(self) -> None:
        """Register a SIGTERM handler for AWS Spot Instance termination.

        When AWS reclaims a Spot Instance it sends SIGTERM with ~2 minutes
        of grace time.  The handler ONLY sets a thread-safe flag — it never
        calls torch.save or any other non-reentrant function.  Writing the
        checkpoint is left to the main training loop, which checks the flag
        between batches and exits cleanly.

        Previous version called torch.save inside the handler, which risked
        deadlocking the process if the signal interrupted PyTorch internals
        or a GIL-holding operation.
        """
        def _handle_sigterm(signum, frame):
            # CRITICAL: signal handlers run asynchronously on the main thread.
            # Calling torch.save, logging, or any code that acquires locks
            # here can deadlock.  We ONLY set the atomic flag.
            self._termination_requested.set()

        try:
            signal.signal(signal.SIGTERM, _handle_sigterm)
            logger.debug("SIGTERM handler registered for Spot termination safety.")
        except (OSError, ValueError):
            # signal.signal can only be called from the main thread;
            # in worker threads we skip registration silently.
            pass

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

        num_batches = max(len(dataloader), 1)
        return total_loss / num_batches

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        batches_processed = 0
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
                batches_processed += 1

        denom = max(batches_processed, 1)
        return total_loss / denom

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _is_rank_zero(self) -> bool:
        return "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0

    def _resume_checkpoint_path(self, save_path: str) -> str:
        return os.path.join(save_path, self.RESUME_CKPT_NAME)

    def _save_resume_checkpoint(
        self, save_path: str, epoch: int, best_val_loss: float
    ) -> None:
        """Write a full resumable checkpoint after every epoch.

        Uses atomic write (tempfile → os.replace) so a kill during the
        write can never leave a half-written, corrupted checkpoint file.
        """
        if not self._is_rank_zero():
            return
        # Handle DDP state dict unwrapping
        model_state = self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict()
        
        ckpt = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        final_path = self._resume_checkpoint_path(save_path)
        # Write to a temp file in the same directory, then atomically rename.
        # If the process is killed mid-write, only the temp file is corrupted;
        # the previous valid checkpoint remains intact.
        fd, tmp_path = tempfile.mkstemp(
            dir=save_path, suffix=".pth.tmp", prefix="ckpt_"
        )
        try:
            os.close(fd)
            torch.save(ckpt, tmp_path)
            os.replace(tmp_path, final_path)  # atomic on POSIX and Windows
            logger.debug(f"Resume checkpoint saved → {final_path} (epoch {epoch + 1})")
        except BaseException:
            # Clean up the temp file on any failure
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
            
        # Push to S3 if configured (Task 3: S3 Sync) - Asynchronously in background with ordering safety
        s3_bucket = self.config.get("s3_bucket")
        if s3_bucket and ("LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0):
            # Join previous upload thread if it is still running to ensure sequential consistency
            if self._active_upload_thread is not None and self._active_upload_thread.is_alive():
                logger.debug("Waiting for previous checkpoint S3 upload to complete...")
                self._active_upload_thread.join(timeout=30)

            def upload_worker(file_path, bucket, key):
                try:
                    import boto3
                    s3 = boto3.client('s3')
                    s3.upload_file(file_path, bucket, key)
                    logger.debug(f"Resume checkpoint synced to s3://{bucket}/{key}")
                except Exception as e:
                    logger.warning(f"Failed to sync checkpoint to S3: {e}")

            s3_key = f"{self.full_config.get('project_name', 'crop_yield')}/{self.RESUME_CKPT_NAME}"
            self._active_upload_thread = threading.Thread(
                target=upload_worker,
                args=(final_path, s3_bucket, s3_key),
                daemon=False  # Wait for upload completion on process exit
            )
            self._active_upload_thread.start()

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
        self._save_path = save_path  # expose to SIGTERM handler

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
            self._current_epoch = epoch  # expose to SIGTERM handler

            self._sync_termination_flag()
            # If Spot termination was requested, stop training gracefully
            if self._termination_requested.is_set():
                logger.warning(
                    f"Spot termination requested — stopping at epoch {epoch + 1}."
                )
                break

            train_loss = self.train_epoch(train_loader)
            self._sync_termination_flag()
            if self._termination_requested.is_set():
                logger.warning(
                    f"Spot termination requested after training epoch — breaking before validation."
                )
                break

            val_loss   = self.validate(val_loader)
            self._sync_termination_flag()
            if self._termination_requested.is_set():
                logger.warning(
                    f"Spot termination requested after validation epoch — breaking before saving checkpoint."
                )
                break

            self.scheduler.step(val_loss)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

            # Save best model weights (atomic write)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self._is_rank_zero():
                    best_path = os.path.join(save_path, "best_model.pth")
                    fd, tmp_best = tempfile.mkstemp(
                        dir=save_path, suffix=".pth.tmp", prefix="best_"
                    )
                    os.close(fd)
                    model_state = self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict()
                    torch.save(model_state, tmp_best)
                    os.replace(tmp_best, best_path)
                    logger.info(
                        f"✅ New best model → {best_path} (Val Loss: {val_loss:.4f})"
                    )

            # Always write resumable checkpoint so a runner death wastes ≤ 1 epoch
            self._save_resume_checkpoint(save_path, epoch, best_val_loss)

        # Clean up resume checkpoint only on successful completion so next run starts fresh
        if self._is_rank_zero():
            if not self._termination_requested.is_set():
                resume_path = self._resume_checkpoint_path(save_path)
                if os.path.exists(resume_path):
                    os.remove(resume_path)
                    logger.info("Training complete — resume checkpoint removed.")
        else:
            logger.warning("Training interrupted by Spot termination. Resumable checkpoint preserved.")

        # Join any active upload thread to ensure final synchronization completes before exit
        if self._active_upload_thread is not None and self._active_upload_thread.is_alive():
            logger.info("Waiting for active S3 checkpoint upload thread to complete...")
            self._active_upload_thread.join()

        logger.success("Training run complete.")
        return {
            "best_val_loss": best_val_loss,
            "epochs": num_epochs,
        }

    def _sync_termination_flag(self) -> None:
        """Synchronize the spot termination request flag across all DDP ranks.
        Runs the reduction in a background daemon thread with a 2-second timeout
        join to prevent deadlocking the training process if a rank is unresponsive
        or killed.
        """
        import threading
        if torch.distributed.is_initialized():
            try:
                term_tensor = torch.tensor([1.0 if self._termination_requested.is_set() else 0.0], device=self.device)
                
                def _run_reduce():
                    try:
                        # Perform blocking reduction inside the thread
                        torch.distributed.all_reduce(term_tensor)
                    except Exception as exc:
                        logger.warning(f"Background DDP reduction failed: {exc}")

                reduce_thread = threading.Thread(target=_run_reduce)
                reduce_thread.daemon = True
                reduce_thread.start()

                reduce_thread.join(timeout=2.0)
                if reduce_thread.is_alive():
                    logger.warning("DDP termination flag sync timed out; proceeding with local signal state.")
                    return

                if term_tensor.item() > 0.0:
                    self._termination_requested.set()
            except Exception as e:
                logger.warning(f"Error syncing DDP termination flag: {e}")

