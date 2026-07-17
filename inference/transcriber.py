"""
High-level inference wrapper. Loads a trained (or pretrained) ASR model
plus an optional Whisper fallback, and exposes single-file, folder, and
batch transcription convenience methods used by both the CLI script
(`scripts/run_inference.py`) and the FastAPI service (`api/main.py`).
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Optional, Union

import torch
from omegaconf import DictConfig

from data_pipeline.collator import ASRDataCollator
from data_pipeline.indicvoices_dataset import ASRExample
from models.base_asr_model import BaseASRModel, TranscriptionResult
from models.registry import build_model
from utils.audio_utils import AudioProcessingConfig, preprocess_audio

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        primary_model: BaseASRModel,
        audio_config: Optional[AudioProcessingConfig] = None,
        fallback_model: Optional[BaseASRModel] = None,
        device: str = "cuda",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.primary_model = primary_model.to(self.device)
        self.fallback_model = fallback_model.to(self.device) if fallback_model else None
        self.audio_config = audio_config or AudioProcessingConfig()
        self.collator = ASRDataCollator()

    @classmethod
    def from_config(
        cls,
        model_cfg: DictConfig,
        audio_cfg: Optional[DictConfig] = None,
        fallback_cfg: Optional[DictConfig] = None,
        device: str = "cuda",
    ) -> "Transcriber":
        primary = build_model(model_cfg)
        fallback = build_model(fallback_cfg) if fallback_cfg is not None else None
        audio_config = (
            AudioProcessingConfig(**audio_cfg) if audio_cfg is not None else AudioProcessingConfig()
        )
        return cls(primary, audio_config=audio_config, fallback_model=fallback, device=device)

    def _transcribe_examples(
        self,
        examples: list[ASRExample],
        language: Optional[str],
        decoding_strategy: str,
    ) -> list[TranscriptionResult]:
        batch = self.collator(examples)
        audio = batch.audio_signal.to(self.device)
        lengths = batch.audio_signal_lengths.to(self.device)

        try:
            return self.primary_model.transcribe_batch(
                audio, lengths, language=language, decoding_strategy=decoding_strategy
            )
        except Exception as exc:  # noqa: BLE001
            if self.fallback_model is None:
                raise
            logger.warning(
                "Primary model failed (%s); falling back to secondary model.", exc
            )
            return self.fallback_model.transcribe_batch(
                audio, lengths, language=language, decoding_strategy=decoding_strategy
            )

    def transcribe_file(
        self,
        path: Union[str, Path],
        language: Optional[str] = None,
        decoding_strategy: str = "beam",
    ) -> TranscriptionResult:
        """Transcribe a single audio file from disk."""
        start = time.time()
        waveform, sr = preprocess_audio(path, config=self.audio_config)
        example = ASRExample(waveform=waveform, sample_rate=sr, text="", language=language or "unknown",
                              duration_sec=len(waveform) / sr)
        result = self._transcribe_examples([example], language, decoding_strategy)[0]
        result.processing_time_sec = time.time() - start
        return result

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename_hint: str,
        language: Optional[str] = None,
        decoding_strategy: str = "beam",
    ) -> TranscriptionResult:
        """Transcribe raw audio bytes (e.g. an uploaded file in the FastAPI endpoint)."""
        start = time.time()
        with io.BytesIO(audio_bytes) as buf:
            waveform, sr = preprocess_audio(buf, config=self.audio_config, filename_hint=filename_hint)
        example = ASRExample(waveform=waveform, sample_rate=sr, text="", language=language or "unknown",
                              duration_sec=len(waveform) / sr)
        result = self._transcribe_examples([example], language, decoding_strategy)[0]
        result.processing_time_sec = time.time() - start
        return result

    def transcribe_folder(
        self,
        folder: Union[str, Path],
        language: Optional[str] = None,
        decoding_strategy: str = "beam",
        batch_size: int = 8,
        extensions: tuple[str, ...] = ("wav", "mp3", "flac", "ogg"),
    ) -> dict[str, TranscriptionResult]:
        """Transcribe every supported audio file in a folder (non-recursive)."""
        folder = Path(folder)
        files = sorted(
            p for p in folder.iterdir() if p.suffix.lower().lstrip(".") in extensions
        )
        if not files:
            logger.warning("No supported audio files found in %s", folder)
            return {}

        results: dict[str, TranscriptionResult] = {}
        for i in range(0, len(files), batch_size):
            chunk = files[i : i + batch_size]
            batch_results = self.transcribe_batch_files(chunk, language, decoding_strategy)
            for path, result in zip(chunk, batch_results):
                results[str(path)] = result
        return results

    def transcribe_batch_files(
        self,
        paths: list[Union[str, Path]],
        language: Optional[str] = None,
        decoding_strategy: str = "beam",
    ) -> list[TranscriptionResult]:
        """Transcribe a list of file paths as a single padded batch."""
        examples = []
        for path in paths:
            waveform, sr = preprocess_audio(path, config=self.audio_config)
            examples.append(
                ASRExample(waveform=waveform, sample_rate=sr, text="", language=language or "unknown",
                           duration_sec=len(waveform) / sr)
            )
        return self._transcribe_examples(examples, language, decoding_strategy)
