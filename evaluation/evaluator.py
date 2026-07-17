"""
Full-dataset evaluator: runs inference over an entire split, reports
overall + per-language WER/CER, and can plot training/validation loss
curves from a TensorBoard event file (or a plain list of logged values)
using matplotlib.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from models.base_asr_model import BaseASRModel
from .metrics import compute_wer, compute_cer

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model: BaseASRModel, device: str = "cuda"):
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate_dataloader(
        self,
        loader: DataLoader,
        decoding_strategy: str = "beam",
        output_path: Optional[str] = None,
    ) -> dict:
        self.model.eval_mode()

        refs_by_lang: dict[str, list[str]] = defaultdict(list)
        hyps_by_lang: dict[str, list[str]] = defaultdict(list)
        all_refs, all_hyps = [], []

        for batch in loader:
            audio = batch.audio_signal.to(self.device)
            lengths = batch.audio_signal_lengths.to(self.device)
            results = self.model.transcribe_batch(audio, lengths, decoding_strategy=decoding_strategy)

            for result, ref, lang in zip(results, batch.raw_texts or [], batch.languages or []):
                refs_by_lang[lang].append(ref)
                hyps_by_lang[lang].append(result.text)
                all_refs.append(ref)
                all_hyps.append(result.text)

        report = {
            "overall": {
                "wer": compute_wer(all_refs, all_hyps),
                "cer": compute_cer(all_refs, all_hyps),
                "num_examples": len(all_refs),
            },
            "per_language": {
                lang: {
                    "wer": compute_wer(refs_by_lang[lang], hyps_by_lang[lang]),
                    "cer": compute_cer(refs_by_lang[lang], hyps_by_lang[lang]),
                    "num_examples": len(refs_by_lang[lang]),
                }
                for lang in refs_by_lang
            },
        }

        logger.info("Evaluation report: %s", json.dumps(report, indent=2))

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info("Wrote evaluation report to %s", output_path)

        return report

    @staticmethod
    def plot_curves(
        train_losses: list[float],
        val_losses: list[float],
        val_wers: list[float],
        output_path: str,
    ) -> None:
        """Plot training/validation loss and validation WER curves to a PNG."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(train_losses, label="train_loss")
        axes[0].plot(val_losses, label="val_loss")
        axes[0].set_xlabel("Evaluation step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training / Validation Loss")
        axes[0].legend()

        axes[1].plot(val_wers, label="val_wer", color="tab:red")
        axes[1].set_xlabel("Evaluation step")
        axes[1].set_ylabel("WER")
        axes[1].set_title("Validation WER")
        axes[1].legend()

        fig.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info("Saved training curves to %s", output_path)
