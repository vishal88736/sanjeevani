"""
Checkpoint management: saving, loading, resuming, and pruning old
checkpoints. Designed to be optimizer/scheduler-agnostic by storing
their state_dicts alongside the model.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    epoch: int
    global_step: int
    model_state_dict: dict
    optimizer_state_dict: Optional[dict] = None
    scheduler_state_dict: Optional[dict] = None
    scaler_state_dict: Optional[dict] = None
    best_metric_value: Optional[float] = None
    extra: dict = field(default_factory=dict)


class CheckpointManager:
    """Handles periodic + best-checkpoint saving and pruning of old ones."""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        best_metric: str = "val_wer",
        best_mode: str = "min",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_metric = best_metric
        self.best_mode = best_mode
        self._best_value: Optional[float] = None
        self._saved_paths: list[Path] = []

    def save(
        self,
        state: CheckpointState,
        tag: str = "step",
        is_best: bool = False,
    ) -> Path:
        filename = f"checkpoint_{tag}_{state.global_step}.pt"
        path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": state.epoch,
                "global_step": state.global_step,
                "model_state_dict": state.model_state_dict,
                "optimizer_state_dict": state.optimizer_state_dict,
                "scheduler_state_dict": state.scheduler_state_dict,
                "scaler_state_dict": state.scaler_state_dict,
                "best_metric_value": state.best_metric_value,
                "extra": state.extra,
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
        self._saved_paths.append(path)
        self._prune()

        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            shutil.copyfile(path, best_path)
            logger.info("Updated best checkpoint -> %s", best_path)

        latest_path = self.checkpoint_dir / "latest.pt"
        shutil.copyfile(path, latest_path)
        return path

    def _prune(self) -> None:
        if self.keep_last_n <= 0 or len(self._saved_paths) <= self.keep_last_n:
            return
        while len(self._saved_paths) > self.keep_last_n:
            old_path = self._saved_paths.pop(0)
            if old_path.exists():
                old_path.unlink()
                logger.debug("Pruned old checkpoint: %s", old_path)

    def is_new_best(self, metric_value: float) -> bool:
        if self._best_value is None:
            self._best_value = metric_value
            return True
        improved = (
            metric_value < self._best_value
            if self.best_mode == "min"
            else metric_value > self._best_value
        )
        if improved:
            self._best_value = metric_value
        return improved

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> CheckpointState:
        if not Path(path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        raw = torch.load(path, map_location=map_location)
        return CheckpointState(
            epoch=raw.get("epoch", 0),
            global_step=raw.get("global_step", 0),
            model_state_dict=raw["model_state_dict"],
            optimizer_state_dict=raw.get("optimizer_state_dict"),
            scheduler_state_dict=raw.get("scheduler_state_dict"),
            scaler_state_dict=raw.get("scaler_state_dict"),
            best_metric_value=raw.get("best_metric_value"),
            extra=raw.get("extra", {}),
        )

    def resolve_resume_path(self, resume_from: Optional[str]) -> Optional[str]:
        """Resolve a resume path, supporting 'latest' and 'best' aliases."""
        if resume_from is None:
            return None
        if resume_from in ("latest", "best"):
            candidate = self.checkpoint_dir / f"{resume_from}.pt"
            return str(candidate) if candidate.exists() else None
        return resume_from if Path(resume_from).exists() else None
