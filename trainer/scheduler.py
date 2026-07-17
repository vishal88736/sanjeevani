"""
Learning-rate scheduler factory. Implements linear-warmup-then-cosine
(default), linear-warmup-then-constant, and Noam (Transformer-style)
schedules as `torch.optim.lr_scheduler.LambdaLR` instances so they
compose cleanly with gradient accumulation and AMP.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from omegaconf import DictConfig


def _linear_warmup_cosine(warmup_steps: int, total_steps: int, min_lr_ratio: float) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return lr_lambda


def _linear_warmup_constant(warmup_steps: int) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    return lr_lambda


def _noam(warmup_steps: int, model_dim: int = 512) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        step = max(step, 1)
        scale = model_dim ** -0.5
        return scale * min(step ** -0.5, step * warmup_steps ** -1.5)

    return lr_lambda


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: DictConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    name = scheduler_cfg.name
    warmup_steps = scheduler_cfg.warmup_steps

    if name == "linear_warmup_cosine":
        lr_lambda = _linear_warmup_cosine(
            warmup_steps, total_steps, scheduler_cfg.get("min_lr_ratio", 0.05)
        )
    elif name == "linear_warmup_constant":
        lr_lambda = _linear_warmup_constant(warmup_steps)
    elif name == "noam":
        lr_lambda = _noam(warmup_steps, scheduler_cfg.get("model_dim", 512))
    else:
        raise ValueError(f"Unknown scheduler '{name}'")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
