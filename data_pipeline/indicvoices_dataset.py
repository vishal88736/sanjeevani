"""
Loader for ai4bharat/IndicVoices from the Hugging Face Hub.

Supports:
  - selecting one or many languages
  - streaming mode (no full-corpus download) or full materialization
  - on-disk caching of preprocessed audio via HF datasets' own cache
  - per-language sample caps for quick iteration

The dataset yields raw (waveform, sample_rate, text, language) tuples;
audio-array preprocessing (resample/mono/normalize/trim) happens lazily
per-example via `utils.audio_utils.preprocess_audio` so it works
identically in streaming and non-streaming modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import IterableDataset, Dataset

import datasets as hf_datasets  # HuggingFace `datasets` library (see note in README
                                 # re: local `data_pipeline` package naming to avoid
                                 # shadowing this import)

from utils.audio_utils import AudioProcessingConfig, preprocess_audio

logger = logging.getLogger(__name__)


@dataclass
class IndicVoicesConfig:
    hf_repo_id: str = "ai4bharat/IndicVoices"
    cache_dir: str = "./data/hf_cache"
    streaming: bool = True
    text_column: str = "text"
    audio_column: str = "audio"
    language_column: str = "language"
    train_split: str = "train"
    validation_split: str = "valid"
    test_split: str = "test"
    max_samples_per_language: Optional[int] = None
    languages: list[str] = field(default_factory=lambda: ["hi"])
    audio_config: AudioProcessingConfig = field(default_factory=AudioProcessingConfig)


@dataclass
class ASRExample:
    waveform: np.ndarray
    sample_rate: int
    text: str
    language: str
    duration_sec: float


class IndicVoicesDataset(IterableDataset):
    """Streaming-friendly multilingual dataset wrapper.

    Falls back to a plain map-style `Dataset` (via `.materialize()`)
    when `config.streaming` is False, which enables shuffling and
    random access at the cost of upfront download time.
    """

    def __init__(self, config: IndicVoicesConfig, split: str):
        self.config = config
        self.split = split
        self._hf_split_name = {
            "train": config.train_split,
            "validation": config.validation_split,
            "test": config.test_split,
        }[split]

    def _load_language_stream(self, language: str) -> hf_datasets.IterableDataset | hf_datasets.Dataset:
        logger.info("Loading IndicVoices split=%s language=%s streaming=%s",
                    self._hf_split_name, language, self.config.streaming)
        ds = hf_datasets.load_dataset(
            self.config.hf_repo_id,
            language,
            split=self._hf_split_name,
            streaming=self.config.streaming,
            cache_dir=None if self.config.streaming else self.config.cache_dir,
        )
        if self.config.max_samples_per_language is not None:
            ds = ds.take(self.config.max_samples_per_language) if self.config.streaming \
                else ds.select(range(min(len(ds), self.config.max_samples_per_language)))
        return ds

    def _example_from_row(self, row: dict, language: str) -> Optional[ASRExample]:
        audio_field = row[self.config.audio_column]
        # HF `Audio` feature decodes to {"array": np.ndarray, "sampling_rate": int}
        raw_array = np.asarray(audio_field["array"], dtype=np.float32)
        sr = audio_field["sampling_rate"]

        try:
            waveform, out_sr = preprocess_audio(
                _InMemoryAudioSource(raw_array, sr),
                config=self.config.audio_config,
            )
        except Exception as exc:  # noqa: BLE001 - log and skip malformed rows
            logger.warning("Skipping malformed audio row (language=%s): %s", language, exc)
            return None

        duration = len(waveform) / out_sr
        if duration < self.config.audio_config.min_duration_sec:
            return None

        return ASRExample(
            waveform=waveform,
            sample_rate=out_sr,
            text=row[self.config.text_column].strip(),
            language=language,
            duration_sec=duration,
        )

    def __iter__(self) -> Iterator[ASRExample]:
        for language in self.config.languages:
            stream = self._load_language_stream(language)
            for row in stream:
                example = self._example_from_row(row, language)
                if example is not None:
                    yield example

    def materialize(self) -> "MaterializedASRDataset":
        """Eagerly load and preprocess all configured languages into memory.

        Intended for small language subsets / dev runs where map-style
        random access (needed for standard PyTorch DataLoader shuffling)
        is worth the upfront cost.
        """
        examples: list[ASRExample] = list(self.__iter__())
        logger.info("Materialized %d examples for split=%s", len(examples), self.split)
        return MaterializedASRDataset(examples)


class MaterializedASRDataset(Dataset):
    """Map-style dataset over a pre-loaded list of ASRExample objects."""

    def __init__(self, examples: list[ASRExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ASRExample:
        return self.examples[idx]


class _InMemoryAudioSource:
    """Adapter so `preprocess_audio` can accept an already-decoded numpy
    array (as returned by HF's `Audio` feature) without a redundant
    file round-trip through soundfile."""

    def __init__(self, array: np.ndarray, sample_rate: int):
        self.array = array
        self.sample_rate = sample_rate


def build_dataset(config: IndicVoicesConfig, split: str) -> IndicVoicesDataset:
    """Factory used by train.py / evaluate.py to construct a dataset from config."""
    return IndicVoicesDataset(config=config, split=split)
