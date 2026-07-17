"""
Pre-download / cache ai4bharat/IndicVoices for a set of languages so
training doesn't stall on first-run downloads.

Usage:
    python scripts/download_dataset.py --languages hi bn ta --split train
    python scripts/download_dataset.py --languages hi --split train --max-samples 500
"""

from __future__ import annotations

import argparse
import logging

import datasets as hf_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "ai4bharat/IndicVoices"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IndicVoices language subsets")
    parser.add_argument("--languages", nargs="+", required=True, help="e.g. hi bn ta te mr")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--cache-dir", default="./data/hf_cache")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    args = parser.parse_args()

    for language in args.languages:
        logger.info("Downloading %s / %s (split=%s)", args.repo_id, language, args.split)
        ds = hf_datasets.load_dataset(
            args.repo_id,
            language,
            split=args.split,
            cache_dir=args.cache_dir,
        )
        if args.max_samples:
            ds = ds.select(range(min(len(ds), args.max_samples)))
        logger.info("Cached %d examples for language=%s at %s", len(ds), language, args.cache_dir)


if __name__ == "__main__":
    main()
