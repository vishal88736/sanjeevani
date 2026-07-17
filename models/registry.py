"""
Model registry: maps the `type` field in a model YAML config to the
concrete BaseASRModel subclass that implements it. Adding a new
backend later means writing a class + registering it here — nothing
else in the pipeline needs to change.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from omegaconf import DictConfig

from .base_asr_model import BaseASRModel

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[..., BaseASRModel]] = {}


def register_model(type_key: str):
    def decorator(cls):
        _REGISTRY[type_key] = cls
        return cls
    return decorator


def _build_nemo_conformer(cfg: DictConfig) -> BaseASRModel:
    from .indic_conformer import IndicConformerModel

    return IndicConformerModel(
        hf_repo_id=cfg.hf_repo_id,
        nemo_filename=cfg.nemo_filename,
        local_path=cfg.local_path,
        sample_rate=cfg.sample_rate,
        supported_languages=list(cfg.supported_languages),
        decoding_strategy=cfg.decoding.strategy,
        beam_size=cfg.decoding.get("beam_size", 8),
        freeze_encoder=cfg.get("freeze_encoder", False),
        dropout=cfg.get("dropout", 0.1),
    )


def _build_hf_whisper(cfg: DictConfig) -> BaseASRModel:
    from .whisper_fallback import WhisperFallbackModel

    return WhisperFallbackModel(
        hf_repo_id=cfg.hf_repo_id,
        local_path=cfg.local_path,
        sample_rate=cfg.sample_rate,
        supported_languages=list(cfg.supported_languages),
        decoding_strategy=cfg.decoding.strategy,
        beam_size=cfg.decoding.get("beam_size", 5),
    )


_REGISTRY["nemo_conformer"] = _build_nemo_conformer
_REGISTRY["hf_whisper"] = _build_hf_whisper


def build_model(model_cfg: DictConfig) -> BaseASRModel:
    """Instantiate the ASR backend named by `model_cfg.type`.

    Raises:
        ValueError: if `model_cfg.type` has no registered builder.
    """
    model_type = model_cfg.get("type")
    if model_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. Registered types: "
            f"{sorted(_REGISTRY.keys())}. Add a new backend in models/registry.py."
        )
    logger.info("Building ASR model of type '%s' (name=%s)", model_type, model_cfg.get("name"))
    return _REGISTRY[model_type](model_cfg)
