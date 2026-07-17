"""
Abstract interface that every pluggable ASR backend must implement.

Keeping training, inference, and evaluation code against this interface
(rather than against NeMo or Whisper APIs directly) is what makes the
model swappable via `configs/model/*.yaml` + `models/registry.py`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: Optional[float] = None
    tokens: Optional[list[str]] = None
    processing_time_sec: Optional[float] = None


class BaseASRModel(abc.ABC):
    """Common interface for all ASR backends (NeMo Conformer, Whisper, ...)."""

    sample_rate: int = 16000
    supported_languages: list[str] = []

    @abc.abstractmethod
    def to(self, device: str) -> "BaseASRModel":
        """Move the underlying model to the given device."""

    @abc.abstractmethod
    def train_mode(self) -> None:
        """Put the underlying model in training mode."""

    @abc.abstractmethod
    def eval_mode(self) -> None:
        """Put the underlying model in evaluation mode."""

    @abc.abstractmethod
    def parameters(self):
        """Return trainable parameters (for optimizer construction)."""

    @abc.abstractmethod
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute the training loss for a batch (see data_pipeline.collator.ASRBatch)."""

    @abc.abstractmethod
    def transcribe_batch(
        self,
        audio_signal: torch.Tensor,
        audio_signal_lengths: torch.Tensor,
        language: Optional[str] = None,
        decoding_strategy: str = "beam",
    ) -> list[TranscriptionResult]:
        """Transcribe a padded batch of audio to text."""

    @abc.abstractmethod
    def save_pretrained(self, path: str) -> None:
        """Persist model weights (and any tokenizer/config) to `path`."""

    @abc.abstractmethod
    def state_dict(self) -> dict:
        """Return a torch-compatible state dict for checkpointing."""

    @abc.abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """Load a torch-compatible state dict produced by `state_dict()`."""

    def freeze_encoder(self) -> None:
        """Optional: freeze encoder parameters. Default no-op; override if supported."""
        return None

    def unfreeze_encoder(self) -> None:
        """Optional: unfreeze encoder parameters. Default no-op; override if supported."""
        return None
