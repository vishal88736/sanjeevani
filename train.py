"""
Training entrypoint for the Sanjeevani ASR module.

Usage:
    python train.py
    python train.py model=whisper training.epochs=5
    python train.py languages='[hi,ta]' hardware.precision=fp16

All configuration is composed via Hydra from configs/config.yaml and
its model/dataset/training sub-configs; any field can be overridden on
the command line using dotted-path syntax.
"""

from __future__ import annotations

import logging
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data_pipeline.collator import ASRDataCollator
from data_pipeline.indicvoices_dataset import IndicVoicesConfig, build_dataset
from models.registry import build_model
from trainer.trainer import ASRTrainer
from utils.audio_utils import AudioProcessingConfig
from utils.config_utils import save_config
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset_config(cfg: DictConfig) -> IndicVoicesConfig:
    audio_cfg = AudioProcessingConfig(
        target_sample_rate=cfg.dataset.audio.target_sample_rate,
        mono=cfg.dataset.audio.mono,
        normalize=cfg.dataset.audio.normalize,
        trim_silence=cfg.dataset.audio.trim_silence,
        min_duration_sec=cfg.dataset.audio.min_duration_sec,
        max_duration_sec=cfg.dataset.audio.max_duration_sec,
    )
    return IndicVoicesConfig(
        hf_repo_id=cfg.dataset.hf_repo_id,
        cache_dir=cfg.dataset.cache_dir,
        streaming=cfg.dataset.streaming,
        text_column=cfg.dataset.text_column,
        audio_column=cfg.dataset.audio_column,
        language_column=cfg.dataset.language_column,
        train_split=cfg.dataset.train_split,
        validation_split=cfg.dataset.validation_split,
        test_split=cfg.dataset.test_split,
        max_samples_per_language=cfg.dataset.max_samples_per_language,
        languages=list(cfg.languages),
        audio_config=audio_cfg,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.project.log_dir, level=cfg.logging.level)
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.project.seed)

    save_config(cfg, f"{cfg.project.output_dir}/resolved_config.yaml")

    dataset_cfg = build_dataset_config(cfg)
    train_dataset = build_dataset(dataset_cfg, split="train").materialize() \
        if not cfg.dataset.streaming else build_dataset(dataset_cfg, split="train")
    val_dataset = build_dataset(dataset_cfg, split="validation").materialize() \
        if not cfg.dataset.streaming else build_dataset(dataset_cfg, split="validation")

    collator = ASRDataCollator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers,
        pin_memory=cfg.hardware.pin_memory,
        collate_fn=collator,
        shuffle=not cfg.dataset.streaming,  # IterableDataset can't be shuffled by the loader
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        num_workers=cfg.hardware.num_workers,
        pin_memory=cfg.hardware.pin_memory,
        collate_fn=collator,
        shuffle=False,
    )

    model = build_model(cfg.model)

    trainer = ASRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        training_cfg=cfg.training,
        hardware_cfg=cfg.hardware,
        checkpoint_dir=cfg.project.checkpoint_dir,
        log_dir=cfg.project.log_dir,
    )
    trainer.train()


if __name__ == "__main__":
    main()
