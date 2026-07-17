"""
Batch inference helper: transcribes many files and writes results to a
JSON or CSV manifest. Thin wrapper around Transcriber.transcribe_folder
for CLI use.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Optional

from .transcriber import Transcriber

logger = logging.getLogger(__name__)


def run_batch_inference(
    transcriber: Transcriber,
    input_folder: str,
    output_path: str,
    language: Optional[str] = None,
    decoding_strategy: str = "beam",
    batch_size: int = 8,
) -> None:
    results = transcriber.transcribe_folder(
        folder=input_folder,
        language=language,
        decoding_strategy=decoding_strategy,
        batch_size=batch_size,
    )

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    if output_path_obj.suffix.lower() == ".csv":
        with open(output_path_obj, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "text", "language", "confidence", "processing_time_sec"])
            for path, result in results.items():
                writer.writerow(
                    [path, result.text, result.language, result.confidence, result.processing_time_sec]
                )
    else:
        payload = {
            path: {
                "text": result.text,
                "language": result.language,
                "confidence": result.confidence,
                "processing_time_sec": result.processing_time_sec,
            }
            for path, result in results.items()
        }
        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d transcriptions to %s", len(results), output_path)
