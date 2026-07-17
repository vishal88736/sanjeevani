"""
Evaluation entrypoint: runs WER/CER evaluation over a dataset split and
writes a JSON report (overall + per-language breakdown).

Usage:
    python evaluate.py
    python evaluate.py model.local_path=./checkpoints/indic_conformer/best.nemo
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from data_pipeline.collator import ASRDataCollator
from evaluation.evaluator import Evaluator
from models.registry import build_model
from train import build_dataset_config  # reuse the same dataset-config builder
from data_pipeline.indicvoices_dataset import build_dataset
from utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.project.log_dir, level=cfg.logging.level)

    dataset_cfg = build_dataset_config(cfg)
    test_dataset = build_dataset(dataset_cfg, split="test")
    if not cfg.dataset.streaming:
        test_dataset = test_dataset.materialize()

    loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.eval_batch_size,
        num_workers=cfg.hardware.num_workers,
        collate_fn=ASRDataCollator(),
        shuffle=False,
    )

    model = build_model(cfg.model)
    evaluator = Evaluator(model=model, device=cfg.hardware.device)

    report = evaluator.evaluate_dataloader(
        loader,
        decoding_strategy=cfg.model.decoding.strategy,
        output_path=f"{cfg.project.output_dir}/evaluation_report.json",
    )
    logger.info("Overall WER: %.4f | CER: %.4f", report["overall"]["wer"], report["overall"]["cer"])


if __name__ == "__main__":
    main()
