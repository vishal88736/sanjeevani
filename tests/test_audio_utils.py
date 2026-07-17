"""Unit tests for utils/audio_utils.py using synthetic sine-wave audio."""

from __future__ import annotations

import numpy as np
import pytest

from utils.audio_utils import (
    AudioProcessingConfig,
    normalize_volume,
    preprocess_audio,
    resample,
    to_mono,
    trim_silence,
)


def _make_sine(freq: float = 440.0, sr: int = 8000, duration: float = 1.0) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class _ArraySource:
    def __init__(self, array: np.ndarray, sample_rate: int):
        self.array = array
        self.sample_rate = sample_rate


def test_to_mono_averages_channels():
    stereo = np.stack([np.ones(100, dtype=np.float32), np.zeros(100, dtype=np.float32)])
    mono = to_mono(stereo)
    assert mono.shape == (100,)
    assert np.allclose(mono, 0.5)


def test_to_mono_passthrough_for_1d():
    mono_in = np.random.randn(50).astype(np.float32)
    assert np.array_equal(to_mono(mono_in), mono_in)


def test_resample_changes_length_proportionally():
    wave = _make_sine(sr=8000, duration=1.0)
    resampled = resample(wave, orig_sr=8000, target_sr=16000)
    assert abs(len(resampled) - 16000) < 10


def test_resample_noop_when_same_rate():
    wave = _make_sine(sr=16000, duration=0.5)
    result = resample(wave, orig_sr=16000, target_sr=16000)
    assert np.array_equal(result, wave)


def test_normalize_volume_scales_rms():
    wave = _make_sine(sr=8000, duration=1.0) * 0.01  # very quiet
    normalized = normalize_volume(wave, target_dbfs=-20.0)
    rms_before = np.sqrt(np.mean(wave ** 2))
    rms_after = np.sqrt(np.mean(normalized ** 2))
    assert rms_after > rms_before
    assert np.max(np.abs(normalized)) <= 1.0 + 1e-6


def test_normalize_volume_handles_silence():
    silence = np.zeros(1000, dtype=np.float32)
    result = normalize_volume(silence)
    assert np.array_equal(result, silence)


def test_trim_silence_removes_leading_trailing_zeros():
    tone = _make_sine(sr=8000, duration=0.5)
    padded = np.concatenate([np.zeros(4000, dtype=np.float32), tone, np.zeros(4000, dtype=np.float32)])
    trimmed = trim_silence(padded, top_db=30.0)
    assert len(trimmed) < len(padded)
    assert len(trimmed) > 0


def test_preprocess_audio_full_pipeline():
    wave = _make_sine(sr=22050, duration=1.0)
    source = _ArraySource(wave, sample_rate=22050)
    config = AudioProcessingConfig(target_sample_rate=16000, mono=True, normalize=True, trim_silence=False)
    processed, sr = preprocess_audio(source, config=config)
    assert sr == 16000
    assert processed.dtype == np.float32
    assert abs(len(processed) - 16000) < 10
