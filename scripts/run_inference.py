"""
CLI inference tool for single files or entire folders.

Usage:
    python scripts/run_inference.py --input sample.wav --language hi
    python scripts/run_inference.py --input ./audio_folder --output results.json --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from inference.batch_inference import run_batch_inference
from inference.transcriber import Transcriber
from utils.config_utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanjeevani ASR command-line inference")
    parser.add_argument("--input", required=True, help="Path to an audio file or folder")
    parser.add_argument("--output", default=None, help="Output JSON/CSV path (folder mode)")
    parser.add_argument("--language", default=None, help="ISO language code hint, e.g. 'hi'")
    parser.add_argument("--decoding", default="beam", choices=["beam", "greedy"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    transcriber = Transcriber.from_config(
        model_cfg=cfg.model,
        audio_cfg=cfg.dataset.audio,
        device=cfg.hardware.device,
    )

    input_path = Path(args.input)
    if input_path.is_dir():
        output_path = args.output or "inference_results.json"
        run_batch_inference(
            transcriber,
            input_folder=str(input_path),
            output_path=output_path,
            language=args.language,
            decoding_strategy=args.decoding,
            batch_size=args.batch_size,
        )
        print(f"Wrote results to {output_path}")
    else:
        result = transcriber.transcribe_file(
            input_path, language=args.language, decoding_strategy=args.decoding
        )
        print(json.dumps({
            "text": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "processing_time_sec": result.processing_time_sec,
        }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
