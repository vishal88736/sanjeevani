"""
Optional Whisper fallback backend (Hugging Face `transformers`), used
only if IndicConformer is unavailable or explicitly requested via
`model=whisper`. Kept behind the same BaseASRModel interface.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .base_asr_model import BaseASRModel, TranscriptionResult

logger = logging.getLogger(__name__)


class WhisperFallbackModel(BaseASRModel):
    """BaseASRModel implementation backed by a Hugging Face Whisper checkpoint."""

    def __init__(
        self,
        hf_repo_id: str,
        local_path: str,
        sample_rate: int = 16000,
        supported_languages: Optional[list[str]] = None,
        decoding_strategy: str = "beam",
        beam_size: int = 5,
    ):
        self.sample_rate = sample_rate
        self.supported_languages = supported_languages or []
        self.decoding_strategy = decoding_strategy
        self.beam_size = beam_size

        cache_dir = str(Path(local_path).parent)
        logger.info("Loading Whisper fallback model %s (cache_dir=%s)", hf_repo_id, cache_dir)
        self.processor = WhisperProcessor.from_pretrained(hf_repo_id, cache_dir=cache_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            hf_repo_id, cache_dir=cache_dir
        )

    def to(self, device: str) -> "WhisperFallbackModel":
        self.model = self.model.to(device)
        return self

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def compute_loss(self, batch) -> torch.Tensor:
        inputs = self.processor(
            [w.numpy() for w in batch.audio_signal],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        labels = self.processor.tokenizer(
            batch.raw_texts, return_tensors="pt", padding=True
        ).input_ids
        outputs = self.model(input_features=inputs.input_features, labels=labels)
        return outputs.loss

    def transcribe_batch(
        self,
        audio_signal: torch.Tensor,
        audio_signal_lengths: torch.Tensor,
        language: Optional[str] = None,
        decoding_strategy: str = "beam",
    ) -> list[TranscriptionResult]:
        start = time.time()
        self.eval_mode()

        arrays = [
            audio_signal[i, : audio_signal_lengths[i]].cpu().numpy()
            for i in range(audio_signal.shape[0])
        ]
        inputs = self.processor(
            arrays, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        generate_kwargs = {"language": language} if language else {}
        num_beams = self.beam_size if decoding_strategy == "beam" else 1

        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs.input_features,
                num_beams=num_beams,
                **generate_kwargs,
            )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        elapsed = time.time() - start
        per_item_time = elapsed / max(len(texts), 1)

        return [
            TranscriptionResult(
                text=text.strip(),
                language=language or "unknown",
                confidence=None,
                processing_time_sec=per_item_time,
            )
            for text in texts
        ]

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        logger.info("Saved Whisper fallback checkpoint to %s", path)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)
