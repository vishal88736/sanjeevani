"""
Central logging setup for Sanjeevani.

Configures the root logger once with two handlers:
  - console (stdout), for live viewing during development
  - a rotating file at logs/sanjeevani.log, so every action and error
    across the whole app (ASR, translation, both Gemma stages, WHO
    fetches, function calls, TTS, and every API request) ends up in
    one place you can grep or tail — not just whatever scrolled past
    in the terminal.

Call setup_logging() once, as early as possible (website/backend/main.py
does this at import time). Every module in this codebase just does
`logger = logging.getLogger(__name__)` and logs normally; because
Python's logging module is hierarchical, all of those records flow
through the handlers configured here without each module needing to
know about the log file.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_LOG_FILE = "sanjeevani.log"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: Path | str = DEFAULT_LOG_DIR,
    level: str = "INFO",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configures the root logger with console + rotating file handlers.

    Idempotent: calling this more than once (e.g. if both the API and a
    standalone script import it) will not duplicate handlers or log lines.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        return root_logger  # already configured

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_path / DEFAULT_LOG_FILE, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initialized. Writing to %s", log_path / DEFAULT_LOG_FILE)
    return root_logger
