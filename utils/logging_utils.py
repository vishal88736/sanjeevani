"""
Centralized logging setup: console + rotating file handler, plus an
optional TensorBoard SummaryWriter factory used by the trainer.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - tensorboard is an optional extra
    SummaryWriter = None  # type: ignore


def setup_logging(
    log_dir: str,
    level: str = "INFO",
    log_filename: str = "sanjeevani_asr.log",
) -> logging.Logger:
    """Configure the root logger with console + rotating file handlers.

    Idempotent: calling this multiple times will not duplicate handlers.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if root_logger.handlers:
        return root_logger  # already configured

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_path / log_filename, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    return root_logger


def get_tensorboard_writer(log_dir: str) -> Optional["SummaryWriter"]:
    """Return a SummaryWriter, or None if tensorboard is not installed."""
    if SummaryWriter is None:
        logging.getLogger(__name__).warning(
            "tensorboard is not installed; TensorBoard logging disabled. "
            "Install with `pip install tensorboard` to enable it."
        )
        return None
    tb_dir = Path(log_dir) / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))
