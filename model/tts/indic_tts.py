"""
Wrapper around AI4Bharat's Indic Parler-TTS.

Follows the model card's usage pattern: a `ParlerTTSForConditionalGeneration`
generation model, a prompt tokenizer (for the text to speak) and a
separate description tokenizer (for an English voice-style prompt).
The model infers which language to speak from the *language the
prompt text is written in* — you don't pass a language code.

The model card notes generations are trained for clips up to ~30s, so
longer answers are split into sentence-level chunks and the resulting
audio is concatenated with a short silence gap between chunks.

Requires: pip install git+https://github.com/huggingface/parler-tts.git
"""

from __future__ import annotations

import io
import logging
import re

import numpy as np
import soundfile as sf
import torch

from model.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

DEFAULT_VOICE_DESCRIPTION = (
    "A calm, clear voice speaks at a moderate pace with good, natural audio quality."
)

# Roughly targets sub-30s chunks: split on sentence-ending punctuation
# (Western + Devanagari/Indic '।') and regroup up to a character budget.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")
_CHUNK_CHAR_BUDGET = 220


def _chunk_text(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > _CHUNK_CHAR_BUDGET:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


class IndicTTS:
    """Lazy-loading wrapper around Indic Parler-TTS."""

    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg
        self._model = None
        self._prompt_tokenizer = None
        self._description_tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self._device = "cuda" if (self.cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
        logger.info("Loading Indic Parler-TTS '%s' onto %s (first use only)...",
                    self.cfg.tts_model_id, self._device)

        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.cfg.tts_model_id
        ).to(self._device)
        self._prompt_tokenizer = AutoTokenizer.from_pretrained(self.cfg.tts_model_id)
        self._description_tokenizer = AutoTokenizer.from_pretrained(
            self._model.config.text_encoder._name_or_path
        )
        logger.info("Indic Parler-TTS loaded.")

    def _synthesize_chunk(self, text: str, description: str) -> np.ndarray:
        description_ids = self._description_tokenizer(description, return_tensors="pt").input_ids.to(self._device)
        prompt_ids = self._prompt_tokenizer(text, return_tensors="pt").input_ids.to(self._device)

        with torch.no_grad():
            generation = self._model.generate(input_ids=description_ids, prompt_input_ids=prompt_ids)

        return generation.cpu().numpy().squeeze()

    def synthesize(self, text: str, description: str = DEFAULT_VOICE_DESCRIPTION) -> bytes:
        """Synthesize `text` to speech (auto-detects language from the
        script/content of `text` itself) and returns WAV bytes."""
        self._ensure_loaded()
        text = text.strip()
        if not text:
            raise ValueError("Cannot synthesize empty text")

        chunks = _chunk_text(text)
        sample_rate = self._model.config.sampling_rate
        silence_gap = np.zeros(int(sample_rate * 0.25), dtype=np.float32)

        audio_segments = []
        for i, chunk in enumerate(chunks):
            audio = self._synthesize_chunk(chunk, description)
            audio_segments.append(audio)
            if i < len(chunks) - 1:
                audio_segments.append(silence_gap)

        full_audio = np.concatenate(audio_segments) if audio_segments else np.zeros(0, dtype=np.float32)

        buffer = io.BytesIO()
        sf.write(buffer, full_audio, sample_rate, format="WAV")
        return buffer.getvalue()
