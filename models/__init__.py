"""Pluggable ASR model wrappers. See `registry.py` for the factory function."""

from .registry import build_model
from .base_asr_model import BaseASRModel, TranscriptionResult

__all__ = ["build_model", "BaseASRModel", "TranscriptionResult"]
