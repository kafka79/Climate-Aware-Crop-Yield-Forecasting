import os
import signal
import tempfile
import threading
import queue

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
        self._save_path: Optional[str] = None
        self._current_epoch: int = 0
        self._register_spot_termination_handler()
        self._clear_termination_pill()  # Remove stale poison pills from previous interrupted runs

        # Background uploader queue for non-blocking S3 sync to prevent DDP node blocking
        self._upload_queue = queue.Queue(maxsize=10)
        self._uploader_thread = None
        self._start_uploader_worker()

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

    def _start_uploader_worker(self) -> None:
        """Start a background daemon thread that processes S3 upload requests sequentially.

        This completely decouples checkpoint uploads from the training loop,
        preventing DDP stalls.  Each upload is retried with exponential backoff
        up to MAX_UPLOAD_RETRIES times.  If all retries fail, the thread logs
        the error and continues processing the next upload—it never crashes
        silently, which would cause the training loop to hang on queue.put().
        """
        s3_bucket = self.config.get("s3_bucket")
        if not s3_bucket or not self._is_rank_zero():
            return

        MAX_UPLOAD_RETRIES = 3

        def uploader_loop():
            try:
                import boto3
                s3 = boto3.client('s3')
            except Exception as e:
                logger.error(f"Failed to initialize boto3 client in uploader loop: {e}")
                return

            while True:
                task = self._upload_queue.get()
                if task is None:  # Sentinel value to terminate thread
                    self._upload_queue.task_done()
                    break
                
                local_path, bucket, key = task
                uploaded = False
                for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
                    try:
                        logger.debug(f"Uploading checkpoint in background: {key} (attempt {attempt}/{MAX_UPLOAD_RETRIES})...")
                        s3.upload_file(local_path, bucket, key)
                        logger.debug(f"Resume checkpoint synced to s3://{bucket}/{key}")
                        uploaded = True
                        break
                    except Exception as e:
                        wait_seconds = min(2 ** attempt, 30)
                        logger.warning(
                            f"S3 upload attempt {attempt}/{MAX_UPLOAD_RETRIES} failed for {key}: {e}. "
                            f"Retrying in {wait_seconds}s..."
                        )
                        import time as _time
                        _time.sleep(wait_seconds)
                        # Refresh boto3 client on credential/connection errors
                        try:
                            import boto3 as _boto3
                            s3 = _boto3.client('s3')
                        except Exception:
                            pass

                if not uploaded:
                    logger.error(
                        f"FAILED to upload checkpoint {key} after {MAX_UPLOAD_RETRIES} retries. "
                        "Continuing to next upload task — training loop will NOT hang."
                    )

                self._upload_queue.task_done()

        self._uploader_thread = threading.Thread(target=uploader_loop, daemon=True)
        self._uploader_thread.start()

    # ── Training / Validation loops ───────────────────────────────────────────

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        batches_processed = 0
        for batch in tqdm(dataloader, desc="Training"):
            self._sync_termination_flag()
            if self._termination_requested.is_set():
                logger.warning("Spot termination requested during training batch — breaking batch loop.")
                break

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
            batches_processed += 1

        num_batches = max(batches_processed, 1)
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
            
        # Push to S3 if configured (Task 3: S3 Sync) - Non-blocking background uploader queue
        s3_bucket = self.config.get("s3_bucket")
        if s3_bucket and self._is_rank_zero():
            s3_key = f"{self.full_config.get('project_name', 'crop_yield')}/{self.RESUME_CKPT_NAME}"
            self._upload_queue.put((final_path, s3_bucket, s3_key))
            logger.debug(f"Queued checkpoint for background S3 upload: {s3_key}")

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

            # ponytail: DDP sync — all ranks must agree on val_loss before scheduler/checkpoint decisions
            if torch.distributed.is_initialized():
                val_loss_tensor = torch.tensor(val_loss, device=self.device)
                torch.distributed.all_reduce(val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                val_loss = (val_loss_tensor / torch.distributed.get_world_size()).item()

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

        # Wait for uploader queue to complete on successful training
        if self._is_rank_zero() and self._uploader_thread is not None:
            logger.info("Waiting for background S3 checkpoint uploads to finish...")
            self._upload_queue.put(None)  # Termination sentinel
            self._uploader_thread.join(timeout=30)

        logger.success("Training run complete.")
        return {
            "best_val_loss": best_val_loss,
            "epochs": num_epochs,
        }

    def _sync_termination_flag(self) -> None:
        """Broadcast Spot termination across all DDP ranks via filesystem poison pill.

        Why NOT use torch.distributed.all_reduce or barrier?
        ──────────────────────────────────────────────────────
        If one rank is already dead (SIGTERM'd and exited the loop), the
        surviving ranks will block forever on a collective operation waiting
        for the dead rank. This is the exact deadlock the panel flagged.

        Instead, the first rank to receive SIGTERM writes a tiny poison file
        to a shared directory (tmpdir or checkpoint dir). Every rank polls
        for this file on each sync call. Because filesystem operations are
        independent and non-blocking, no rank can deadlock another.

        Cleanup: The poison file is removed on the next fresh training start
        by _clear_termination_pill() called during __init__.
        """
        # If already flagged locally, nothing more to do
        if self._termination_requested.is_set():
            # Ensure the poison pill exists so other ranks see it too
            self._write_poison_pill()
            return

        # Check if another rank has written the poison pill
        pill_path = self._poison_pill_path()
        if pill_path.exists():
            logger.warning(
                f"Rank detected poison pill at {pill_path} — "
                "another rank has received SIGTERM. Initiating graceful shutdown."
            )
            self._termination_requested.set()
            return

    def _poison_pill_path(self) -> "Path":
        """Return the path to the shared termination poison file.

        Uses the checkpoint directory (shared across ranks in multi-node
        training via EFS/FSx) so all ranks can see it.
        """
        from pathlib import Path
        save_dir = Path(self.config.get("training", {}).get(
            "save_path", "models/checkpoints"
        ))
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / ".ddp_termination_pill"

    def _write_poison_pill(self) -> None:
        """Write the poison pill file so other DDP ranks detect termination."""
        try:
            pill = self._poison_pill_path()
            if not pill.exists():
                pill.write_text(f"SIGTERM received by rank at epoch {self._current_epoch}")
                logger.info(f"Poison pill written to {pill}")
        except Exception as exc:
            logger.warning(f"Could not write poison pill: {exc}")

    def _clear_termination_pill(self) -> None:
        """Remove stale poison pill from a previous interrupted run."""
        try:
            pill = self._poison_pill_path()
            if pill.exists():
                pill.unlink()
                logger.info(f"Stale poison pill removed: {pill}")
        except Exception as exc:
            logger.debug(f"Could not remove stale poison pill: {exc}")

