"""
Audio preprocessing utilities.

Handles format loading (wav/mp3/flac/ogg), resampling to a target sample
rate, mono conversion, float32 normalization, and silence trimming. All
functions operate on numpy arrays or torch tensors and avoid any
framework-specific state so they can be reused by the dataset loader,
the training collator, and the inference pipeline alike.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
import librosa

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"wav", "mp3", "flac", "ogg"}


class UnsupportedAudioFormatError(ValueError):
    """Raised when an audio file's extension is not in SUPPORTED_FORMATS."""


@dataclass
class AudioProcessingConfig:
    target_sample_rate: int = 16000
    mono: bool = True
    normalize: bool = True
    trim_silence: bool = True
    min_duration_sec: float = 0.5
    max_duration_sec: float = 30.0
    top_db: float = 30.0  # silence-trim threshold for librosa.effects.trim


def _assert_supported_format(path: Union[str, Path]) -> None:
    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in SUPPORTED_FORMATS:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio format '{ext}'. Supported: {sorted(SUPPORTED_FORMATS)}"
        )


def load_audio(
    source: Union[str, Path, bytes, io.BytesIO],
    filename_hint: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    """Load audio from a file path or in-memory bytes.

    Args:
        source: Path to an audio file, raw bytes, or a BytesIO buffer.
        filename_hint: Original filename (used to validate format when
            `source` is bytes without an inferable extension).

    Returns:
        (waveform, sample_rate) where waveform is float32, shape (T,) or (C, T).
    """
    if hasattr(source, "array") and hasattr(source, "sample_rate"):
        # Pre-decoded in-memory source (e.g. HF `Audio` feature output).
        # See data_pipeline/indicvoices_dataset.py::_InMemoryAudioSource.
        waveform = np.asarray(source.array, dtype=np.float32)  # type: ignore[attr-defined]
        sr = source.sample_rate  # type: ignore[attr-defined]
    elif isinstance(source, (str, Path)):
        _assert_supported_format(source)
        waveform, sr = sf.read(str(source), dtype="float32", always_2d=False)
    else:
        if filename_hint is not None:
            _assert_supported_format(filename_hint)
        raw_bytes = source.read() if isinstance(source, io.BytesIO) else source
        with io.BytesIO(raw_bytes) as buf:
            waveform, sr = sf.read(buf, dtype="float32", always_2d=False)

    if waveform.ndim > 1:
        waveform = waveform.T  # soundfile returns (T, C); torchaudio expects (C, T)
    else:
        waveform = waveform[np.newaxis, :]

    return waveform.astype(np.float32), sr


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Collapse a (C, T) waveform to (T,) by averaging channels."""
    if waveform.ndim == 1:
        return waveform
    return waveform.mean(axis=0)


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1D or (C, T) waveform to `target_sr` using torchaudio."""
    if orig_sr == target_sr:
        return waveform
    tensor = torch.from_numpy(np.atleast_2d(waveform))
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    result = resampled.numpy()
    return result[0] if waveform.ndim == 1 else result


def normalize_volume(waveform: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """Peak-safe RMS normalization to a target dBFS level."""
    rms = np.sqrt(np.mean(np.square(waveform)) + 1e-12)
    if rms < 1e-9:
        return waveform  # silent clip, nothing to normalize
    target_rms = 10 ** (target_dbfs / 20)
    gain = target_rms / rms
    normalized = waveform * gain
    # Prevent clipping after gain application.
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak
    return normalized.astype(np.float32)


def trim_silence(waveform: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence using librosa's energy-based trimmer."""
    trimmed, _ = librosa.effects.trim(waveform, top_db=top_db)
    if trimmed.size == 0:
        # Guard against over-aggressive trimming wiping out short clips.
        return waveform
    return trimmed


def preprocess_audio(
    source: Union[str, Path, bytes, io.BytesIO],
    config: Optional[AudioProcessingConfig] = None,
    filename_hint: Optional[str] = None,
) -> tuple[np.ndarray, int]:
    """Full preprocessing pipeline: load -> mono -> resample -> normalize -> trim.

    Returns:
        (waveform, sample_rate) with waveform as a 1D float32 numpy array.
    """
    config = config or AudioProcessingConfig()

    waveform, sr = load_audio(source, filename_hint=filename_hint)

    if config.mono:
        waveform = to_mono(waveform)
    elif waveform.ndim > 1:
        waveform = waveform[0]  # fall back to first channel if mono not requested

    waveform = resample(waveform, sr, config.target_sample_rate)
    sr = config.target_sample_rate

    if config.normalize:
        waveform = normalize_volume(waveform)

    if config.trim_silence:
        waveform = trim_silence(waveform, top_db=config.top_db)

    duration = len(waveform) / sr
    if duration < config.min_duration_sec:
        logger.warning(
            "Audio duration %.2fs is below min_duration_sec=%.2fs after trimming",
            duration, config.min_duration_sec,
        )
    if duration > config.max_duration_sec:
        logger.warning(
            "Audio duration %.2fs exceeds max_duration_sec=%.2fs; "
            "consider chunking upstream",
            duration, config.max_duration_sec,
        )

    return waveform.astype(np.float32), sr


def waveform_to_tensor(waveform: np.ndarray) -> torch.Tensor:
    """Convert a 1D numpy waveform to a (1, T) float32 torch tensor."""
    return torch.from_numpy(waveform).float().unsqueeze(0)
