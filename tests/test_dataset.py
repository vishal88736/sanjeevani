"""Unit tests for the ASR data collator (dynamic padding logic)."""

from __future__ import annotations

import numpy as np

from data_pipeline.collator import ASRDataCollator
from data_pipeline.indicvoices_dataset import ASRExample


def _example(length: int, text: str, language: str = "hi") -> ASRExample:
    return ASRExample(
        waveform=np.random.randn(length).astype(np.float32),
        sample_rate=16000,
        text=text,
        language=language,
        duration_sec=length / 16000,
    )


def test_collator_pads_to_max_length():
    examples = [_example(1000, "a"), _example(500, "b"), _example(1500, "c")]
    collator = ASRDataCollator()
    batch = collator(examples)

    assert batch.audio_signal.shape == (3, 1500)
    assert batch.audio_signal_lengths.tolist() == [1000, 500, 1500]


def test_collator_preserves_raw_text_and_language():
    examples = [_example(100, "hello", "hi"), _example(200, "world", "ta")]
    collator = ASRDataCollator()
    batch = collator(examples)

    assert batch.raw_texts == ["hello", "world"]
    assert batch.languages == ["hi", "ta"]


def test_collator_zero_pads_beyond_actual_length():
    examples = [_example(10, "short"), _example(20, "longer")]
    collator = ASRDataCollator()
    batch = collator(examples)

    # The first example's audio should be zero-padded beyond its 10 real samples.
    assert (batch.audio_signal[0, 10:] == 0).all()
