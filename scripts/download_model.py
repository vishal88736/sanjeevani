"""
Pre-download the IndicConformer (or Whisper fallback) checkpoint so
inference/training don't stall on first-run downloads.

Usage:
    python scripts/download_model.py --model indic_conformer
    python scripts/download_model.py --model whisper
"""

from __future__ import annotations

import argparse
import logging

from omegaconf import OmegaConf

from models.registry import build_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download an ASR model checkpoint")
    parser.add_argument("--model", default="indic_conformer", choices=["indic_conformer", "whisper"])
    parser.add_argument(
        "--config-path",
        default=None,
        help="Override path to the model YAML config (defaults to configs/model/<model>.yaml)",
    )
    args = parser.parse_args()

    config_path = args.config_path or f"configs/model/{args.model}.yaml"
    logger.info("Loading model config from %s", config_path)
    model_cfg = OmegaConf.load(config_path)

    logger.info("Downloading + instantiating model '%s'...", args.model)
    build_model(model_cfg)
    logger.info("Model '%s' downloaded and cached successfully.", args.model)


if __name__ == "__main__":
    main()
