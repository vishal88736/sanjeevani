"""
Wrapper around AI4Bharat's IndicConformer 600M multilingual model,
loaded via NVIDIA NeMo's ASRModel.restore_from(), implementing the
BaseASRModel interface so it can be trained/evaluated/served through
the same generic pipeline as any other backend.

NeMo's EncDecCTCModel / EncDecRNNTModel classes already provide
tokenization, CTC/RNNT loss, and greedy/beam decoding internally;
this wrapper mostly adapts NeMo's calling convention to our
ASRBatch / TranscriptionResult data classes.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from .base_asr_model import BaseASRModel, TranscriptionResult

logger = logging.getLogger(__name__)

try:
    import nemo.collections.asr as nemo_asr
except ImportError as exc:  # pragma: no cover
    nemo_asr = None
    logger.warning(
        "NVIDIA NeMo is not installed (%s). Install with "
        "`pip install nemo_toolkit['asr']` to use IndicConformerModel.",
        exc,
    )


class IndicConformerModel(BaseASRModel):
    """BaseASRModel implementation backed by a NeMo `.nemo` checkpoint."""

    def __init__(
        self,
        hf_repo_id: str,
        nemo_filename: str,
        local_path: str,
        sample_rate: int = 16000,
        supported_languages: Optional[list[str]] = None,
        decoding_strategy: str = "beam",
        beam_size: int = 8,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ):
        if nemo_asr is None:
            raise ImportError(
                "NVIDIA NeMo is required for IndicConformerModel. "
                "Install with: pip install 'nemo_toolkit[asr]'"
            )

        self.sample_rate = sample_rate
        self.supported_languages = supported_languages or []
        self.decoding_strategy = decoding_strategy
        self.beam_size = beam_size
        self.dropout = dropout

        checkpoint_path = self._ensure_checkpoint(
            hf_repo_id, nemo_filename, local_path
        )
        logger.info("Restoring IndicConformer from %s", checkpoint_path)
        self.model = nemo_asr.models.ASRModel.restore_from(
            restore_path=checkpoint_path, map_location="cpu"
        )
        self._configure_decoding()

        if freeze_encoder:
            self.freeze_encoder()

    @staticmethod
    def _ensure_checkpoint(hf_repo_id: str, nemo_filename: str, local_path: str) -> str:
        """Download the .nemo archive from the HF Hub if not already cached locally."""
        path = Path(local_path)
        if path.exists():
            return str(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Downloading %s from HF Hub repo %s (first run only; cached afterwards)",
            nemo_filename, hf_repo_id,
        )
        downloaded_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=nemo_filename,
            local_dir=str(path.parent),
        )
        if downloaded_path != str(path):
            os.replace(downloaded_path, path)
        return str(path)

    def _configure_decoding(self) -> None:
        if not hasattr(self.model, "change_decoding_strategy"):
            return
        decoding_cfg = self.model.cfg.decoding
        decoding_cfg.strategy = "beam" if self.decoding_strategy == "beam" else "greedy"
        if self.decoding_strategy == "beam":
            decoding_cfg.beam.beam_size = self.beam_size
        self.model.change_decoding_strategy(decoding_cfg)

    # -- BaseASRModel interface -------------------------------------------------

    def to(self, device: str) -> "IndicConformerModel":
        self.model = self.model.to(device)
        return self

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def compute_loss(self, batch) -> torch.Tensor:
        """Delegates to NeMo's internal training_step-style loss computation.

        NeMo ASR models compute loss (CTC or RNNT depending on the
        checkpoint's decoder head) from `(audio_signal, audio_signal_lengths,
        transcript, transcript_lengths)`. Since this wrapper's collator
        passes raw text (NeMo tokenizes internally via the model's own
        `self.model.transcribe`/training data layer conventions), we
        forward through the model's forward + loss module directly.
        """
        transcript, transcript_lengths = self.model.tokenizer_and_pad(
            batch.raw_texts
        ) if hasattr(self.model, "tokenizer_and_pad") else (
            batch.text_tokens, batch.text_lengths
        )

        log_probs, encoded_len, greedy_predictions = self.model.forward(
            input_signal=batch.audio_signal,
            input_signal_length=batch.audio_signal_lengths,
        )
        loss = self.model.loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_lengths,
        )
        return loss

    def transcribe_batch(
        self,
        audio_signal: torch.Tensor,
        audio_signal_lengths: torch.Tensor,
        language: Optional[str] = None,
        decoding_strategy: str = "beam",
    ) -> list[TranscriptionResult]:
        if decoding_strategy != self.decoding_strategy:
            self.decoding_strategy = decoding_strategy
            self._configure_decoding()

        start = time.time()
        self.eval_mode()
        with torch.no_grad():
            hypotheses = self.model.transcribe(
                audio=[
                    audio_signal[i, : audio_signal_lengths[i]].cpu().numpy()
                    for i in range(audio_signal.shape[0])
                ],
                batch_size=audio_signal.shape[0],
            )
        elapsed = time.time() - start
        per_item_time = elapsed / max(len(hypotheses), 1)

        results = []
        for hyp in hypotheses:
            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            confidence = getattr(hyp, "score", None)
            results.append(
                TranscriptionResult(
                    text=text,
                    language=language or "unknown",
                    confidence=confidence,
                    processing_time_sec=per_item_time,
                )
            )
        return results

    def save_pretrained(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_to(path)
        logger.info("Saved IndicConformer checkpoint to %s", path)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)

    def freeze_encoder(self) -> None:
        if hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen.")

    def unfreeze_encoder(self) -> None:
        if hasattr(self.model, "encoder"):
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            logger.info("Encoder unfrozen.")
