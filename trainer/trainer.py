"""
ASRTrainer: the main training loop.

Supports mixed precision (fp16/bf16), gradient accumulation, gradient
checkpointing (delegated to the underlying model when available),
distributed data-parallel training, resume-from-checkpoint, periodic
+ best checkpoint saving, TensorBoard logging, and early stopping.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data_pipeline.collator import ASRBatch
from evaluation.metrics import compute_wer, compute_cer
from models.base_asr_model import BaseASRModel
from utils.checkpoint_utils import CheckpointManager, CheckpointState
from utils.logging_utils import get_tensorboard_writer

from .callbacks import EarlyStopping
from .scheduler import build_scheduler

logger = logging.getLogger(__name__)


class ASRTrainer:
    def __init__(
        self,
        model: BaseASRModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        training_cfg: DictConfig,
        hardware_cfg: DictConfig,
        checkpoint_dir: str,
        log_dir: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = training_cfg
        self.hardware_cfg = hardware_cfg

        self.device = hardware_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        self.is_distributed = hardware_cfg.get("distributed", False) and dist.is_available()
        self.rank = dist.get_rank() if self.is_distributed and dist.is_initialized() else 0

        self.model.to(self.device)
        if self.cfg.get("gradient_checkpointing", False):
            self._enable_gradient_checkpointing()

        if self.is_distributed:
            self._underlying_module = self.model.model  # wrap the nn.Module directly
            self._underlying_module = DDP(
                self._underlying_module, device_ids=[torch.cuda.current_device()]
            )
            self.model.model = self._underlying_module

        self.optimizer = self._build_optimizer()
        total_steps = self._estimate_total_steps()
        self.scheduler = build_scheduler(self.optimizer, self.cfg.scheduler, total_steps)

        precision = hardware_cfg.get("precision", "fp32")
        self.use_amp = precision in ("fp16", "bf16") and self.device == "cuda"
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=(precision == "fp16" and self.use_amp))

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            keep_last_n=self.cfg.checkpointing.keep_last_n,
            best_metric=self.cfg.checkpointing.best_metric,
            best_mode=self.cfg.checkpointing.best_mode,
        )
        self.early_stopping = (
            EarlyStopping(
                monitor=self.cfg.early_stopping.monitor,
                mode=self.cfg.early_stopping.mode,
                patience=self.cfg.early_stopping.patience,
                min_delta=self.cfg.early_stopping.min_delta,
            )
            if self.cfg.early_stopping.enabled
            else None
        )

        self.tb_writer = get_tensorboard_writer(log_dir) if self.rank == 0 else None

        self.global_step = 0
        self.epoch = 0
        self._maybe_resume()

    # -- setup helpers -----------------------------------------------------

    def _enable_gradient_checkpointing(self) -> None:
        underlying = getattr(self.model, "model", None)
        if underlying is not None and hasattr(underlying, "encoder"):
            encoder = underlying.encoder
            if hasattr(encoder, "enable_gradient_checkpointing"):
                encoder.enable_gradient_checkpointing()
                logger.info("Gradient checkpointing enabled on encoder.")
            elif hasattr(encoder, "gradient_checkpointing_enable"):
                encoder.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled on encoder.")
            else:
                logger.warning(
                    "Model encoder does not expose a gradient-checkpointing toggle; skipping."
                )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.cfg.optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        if opt_cfg.name == "adamw":
            return torch.optim.AdamW(
                params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
                eps=opt_cfg.eps,
            )
        raise ValueError(f"Unsupported optimizer '{opt_cfg.name}'")

    def _estimate_total_steps(self) -> int:
        if self.cfg.get("max_steps"):
            return int(self.cfg.max_steps)
        steps_per_epoch = max(1, len(self.train_loader) // self.cfg.gradient_accumulation_steps)
        return steps_per_epoch * self.cfg.epochs

    def _maybe_resume(self) -> None:
        resume_from = self.cfg.checkpointing.get("resume_from")
        resolved = self.checkpoint_manager.resolve_resume_path(resume_from)
        if not resolved:
            if resume_from:
                logger.warning("resume_from='%s' not found; starting from scratch.", resume_from)
            return

        logger.info("Resuming training from checkpoint: %s", resolved)
        state = CheckpointManager.load(resolved, map_location=self.device)
        self.model.load_state_dict(state.model_state_dict)
        if state.optimizer_state_dict:
            self.optimizer.load_state_dict(state.optimizer_state_dict)
        if state.scheduler_state_dict:
            self.scheduler.load_state_dict(state.scheduler_state_dict)
        if state.scaler_state_dict:
            self.scaler.load_state_dict(state.scaler_state_dict)
        self.epoch = state.epoch
        self.global_step = state.global_step
        logger.info("Resumed at epoch=%d global_step=%d", self.epoch, self.global_step)

    # -- training loop -------------------------------------------------------

    def train(self) -> None:
        max_steps = self.cfg.get("max_steps")
        logger.info(
            "Starting training: epochs=%d grad_accum=%d amp=%s device=%s",
            self.cfg.epochs, self.cfg.gradient_accumulation_steps, self.use_amp, self.device,
        )

        for epoch in range(self.epoch, self.cfg.epochs):
            self.epoch = epoch
            self._train_one_epoch()

            if self.val_loader is not None:
                metrics = self.evaluate()
                self._log_metrics(metrics, prefix="val", step=self.global_step)

                is_best = self.checkpoint_manager.is_new_best(
                    metrics.get(self.cfg.checkpointing.best_metric, float("inf"))
                )
                self._save_checkpoint(tag=f"epoch{epoch}", is_best=is_best)

                if self.early_stopping and self.early_stopping.step(metrics):
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

            if max_steps and self.global_step >= max_steps:
                logger.info("Reached max_steps=%d; stopping.", max_steps)
                break

        if self.tb_writer:
            self.tb_writer.close()

    def _train_one_epoch(self) -> None:
        self.model.train_mode()
        accum_steps = self.cfg.gradient_accumulation_steps
        running_loss = 0.0
        epoch_start = time.time()

        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(self.train_loader):
            batch = self._batch_to_device(batch)

            with torch.autocast(
                device_type="cuda" if self.device == "cuda" else "cpu",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                loss = self.model.compute_loss(batch) / accum_steps

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * accum_steps

            if (batch_idx + 1) % accum_steps == 0:
                self._optimizer_step()
                self.global_step += 1

                if self.global_step % self.cfg.logging.log_every_n_steps == 0:
                    avg_loss = running_loss / self.cfg.logging.log_every_n_steps
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        "epoch=%d step=%d loss=%.4f lr=%.2e",
                        self.epoch, self.global_step, avg_loss, lr,
                    )
                    self._log_metrics({"loss": avg_loss, "lr": lr}, prefix="train", step=self.global_step)
                    running_loss = 0.0

                if self.global_step % self.cfg.checkpointing.save_every_n_steps == 0:
                    self._save_checkpoint(tag=f"step{self.global_step}", is_best=False)

        elapsed = time.time() - epoch_start
        logger.info("Epoch %d finished in %.1fs", self.epoch, elapsed)

    def _optimizer_step(self) -> None:
        if self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.gradient_clip_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.gradient_clip_norm,
            )
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    # -- evaluation ------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        assert self.val_loader is not None
        self.model.eval_mode()

        all_hyps: list[str] = []
        all_refs: list[str] = []
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            batch = self._batch_to_device(batch)
            loss = self.model.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1

            results = self.model.transcribe_batch(
                batch.audio_signal, batch.audio_signal_lengths, decoding_strategy="greedy"
            )
            all_hyps.extend(r.text for r in results)
            all_refs.extend(batch.raw_texts or [])

        wer = compute_wer(references=all_refs, hypotheses=all_hyps)
        cer = compute_cer(references=all_refs, hypotheses=all_hyps)
        avg_loss = total_loss / max(num_batches, 1)

        return {"val_loss": avg_loss, "val_wer": wer, "val_cer": cer}

    # -- helpers -----------------------------------------------------------

    def _batch_to_device(self, batch: ASRBatch) -> ASRBatch:
        batch.audio_signal = batch.audio_signal.to(self.device)
        batch.audio_signal_lengths = batch.audio_signal_lengths.to(self.device)
        if batch.text_tokens is not None:
            batch.text_tokens = batch.text_tokens.to(self.device)
            batch.text_lengths = batch.text_lengths.to(self.device)
        return batch

    def _save_checkpoint(self, tag: str, is_best: bool) -> None:
        if self.rank != 0:
            return
        state = CheckpointState(
            epoch=self.epoch,
            global_step=self.global_step,
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            scaler_state_dict=self.scaler.state_dict() if self.scaler.is_enabled() else None,
        )
        self.checkpoint_manager.save(state, tag=tag, is_best=is_best)

    def _log_metrics(self, metrics: dict[str, float], prefix: str, step: int) -> None:
        if self.tb_writer is None:
            return
        for key, value in metrics.items():
            self.tb_writer.add_scalar(f"{prefix}/{key}", value, step)
