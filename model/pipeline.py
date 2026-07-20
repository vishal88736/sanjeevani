"""
SanjeevaniPipeline: the single entrypoint the website backend calls.

Handles both input modes:
  - process_audio(...)  speech -> (transcript) -> English -> answer
  - process_text(...)   typed text -> English -> answer

All three underlying models (ASR, translation, Groq) are lazy-loaded
on first use inside their own wrapper classes, so importing this
module / starting the API server is cheap; the first real request
pays the model-loading cost once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from model.asr.indic_asr import IndicASR
from model.config import DEFAULT_LANGUAGE, LANGUAGE_BY_CODE
from model.llm.groq_client import GroqAnswerer
from model.llm.knowledge_base import retrieve_context
from model.translation.indic_translator import IndicTranslator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    transcript: str            # what was said/typed, in its original language
    detected_language: str     # short UI code, e.g. "hi"
    detected_language_name: str
    english_text: str          # translated (or original, if already English)
    answer: str
    is_grounded: bool          # whether a real medical source backed the answer
    sources: list[str] = field(default_factory=list)


# Unicode script ranges used for a real, deterministic language guess on
# *typed* text (unlike audio, script blocks are a reliable signal here).
_SCRIPT_RANGES: list[tuple[str, tuple[int, int]]] = [
    ("hi", (0x0900, 0x097F)),   # Devanagari (Hindi/Marathi share this block; default to Hindi)
    ("bn", (0x0980, 0x09FF)),   # Bengali/Assamese share this block; default to Bengali
    ("gu", (0x0A80, 0x0AFF)),   # Gujarati
    ("pa", (0x0A00, 0x0A7F)),   # Gurmukhi (Punjabi)
    ("or", (0x0B00, 0x0B7F)),   # Odia
    ("ta", (0x0B80, 0x0BFF)),   # Tamil
    ("te", (0x0C00, 0x0C7F)),   # Telugu
    ("kn", (0x0C80, 0x0CFF)),   # Kannada
    ("ml", (0x0D00, 0x0D7F)),   # Malayalam
    ("ur", (0x0600, 0x06FF)),   # Arabic block (Urdu)
]


def _detect_text_language(text: str) -> str:
    """Deterministic Unicode-script-based detection for typed text."""
    counts: dict[str, int] = {}
    for ch in text:
        codepoint = ord(ch)
        for code, (lo, hi) in _SCRIPT_RANGES:
            if lo <= codepoint <= hi:
                counts[code] = counts.get(code, 0) + 1
                break
    if not counts:
        return "en"  # no Indic script characters found -> assume Latin/English
    return max(counts, key=counts.get)


class SanjeevaniPipeline:
    def __init__(self):
        self.asr = IndicASR()
        self.translator = IndicTranslator()
        self.answerer = GroqAnswerer()

    def process_audio(self, audio_bytes: bytes, language: Optional[str] = None) -> PipelineResult:
        if language and language != "auto":
            asr_result = self.asr.transcribe(audio_bytes, language)
        else:
            asr_result = self.asr.detect_and_transcribe(audio_bytes)

        return self._finish(asr_result.text, asr_result.language)

    def process_text(self, text: str, language: Optional[str] = None) -> PipelineResult:
        text = text.strip()
        if not text:
            raise ValueError("Empty text input")
        lang_code = language if (language and language != "auto") else _detect_text_language(text)
        return self._finish(text, lang_code)

    def _finish(self, transcript: str, lang_code: str) -> PipelineResult:
        lang_info = LANGUAGE_BY_CODE.get(lang_code, LANGUAGE_BY_CODE[DEFAULT_LANGUAGE])

        english_text = self.translator.translate_to_english(transcript, lang_info.code)
        if not english_text:
            raise ValueError("Could not extract any text to answer from.")

        kb_result = retrieve_context(english_text)
        answer = self.answerer.answer(english_text, kb_result)

        return PipelineResult(
            transcript=transcript,
            detected_language=lang_info.code,
            detected_language_name=lang_info.name,
            english_text=english_text,
            answer=answer,
            is_grounded=kb_result.is_grounded,
            sources=kb_result.sources,
        )
