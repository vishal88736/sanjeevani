"""
SanjeevaniPipeline: the single entrypoint the website backend calls.

Full flow for one turn:

  1. ASR (speech) or straight through (typed text) -> native-language transcript
  2. IndicTrans2 -> English text
  3. GemmaReasoner.extract_and_plan()  -> structured symptoms + candidate WHO topics
  4. knowledge_base.match_topics()     -> live WHO fact-sheet excerpts for those topics
  5. GemmaReasoner.clinical_reasoning() -> triage judgment + plain-language answer + next_action
  6. functions.dispatch(next_action)    -> "function calling": emergency escalation
                                            or a (placeholder) hospital-finder note

Session memory (model/llm/memory.py) carries prior turns into steps 3
and 5 so Gemma can resolve follow-ups. `speak()` turns an answer back
into audio in the person's own language.

If the multi-stage reasoning pipeline fails end-to-end (e.g. Gemma's
JSON couldn't be parsed even after a retry, or the WHO fetch and the
model both hiccup at once), `_finish` falls back to OllamaAnswerer's
single-call path rather than surfacing a hard error — see
`_reason_with_fallback`.

All underlying models (ASR, translation, Gemma 4, TTS) are lazy-loaded
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
from model.llm.functions import FunctionCallResult, dispatch as dispatch_function
from model.llm.knowledge_base import KnowledgeBaseResult, match_topics, retrieve_context
from model.llm.memory import ConversationStore
from model.llm.ollama_client import OllamaAnswerer
from model.llm.reasoning import ExtractionResult, GemmaReasoner, ReasoningParseError, TriageResult
from model.translation.indic_translator import IndicTranslator
from model.tts.indic_tts import IndicTTS

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    session_id: str
    transcript: str            # what was said/typed, in its original language
    detected_language: str     # short UI code, e.g. "hi"
    detected_language_name: str
    english_text: str          # translated (or original, if already English)
    answer: str
    is_grounded: bool          # whether at least one real WHO source backed the answer
    sources: list[str] = field(default_factory=list)

    # -- reasoning/triage detail (empty/defaults if the fallback path was used) --
    triage: str = "unknown"                 # self_care | routine | urgent | emergency | unknown
    confidence: float = 0.0
    possible_conditions: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    is_emergency: bool = False
    function_note: Optional[str] = None     # message from a dispatched function call, if any
    used_fallback: bool = False             # True if the simple single-call path was used


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
        self.reasoner = GemmaReasoner()
        self.fallback_answerer = OllamaAnswerer()
        self.tts = IndicTTS()
        self.memory = ConversationStore()

    def process_audio(
        self, audio_bytes: bytes, language: Optional[str] = None, session_id: Optional[str] = None
    ) -> PipelineResult:
        logger.info(
            "ASR stage starting  session=%s language=%s audio_bytes=%d",
            session_id or "new", language or "auto", len(audio_bytes),
        )
        try:
            if language and language != "auto":
                asr_result = self.asr.transcribe(audio_bytes, language)
            else:
                asr_result = self.asr.detect_and_transcribe(audio_bytes)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ASR stage failed  session=%s", session_id or "new")
            raise RuntimeError(f"Speech recognition failed: {exc}") from exc

        logger.info(
            "ASR stage complete  session=%s language=%s chars=%d",
            session_id or "new", asr_result.language, len(asr_result.text),
        )
        return self._finish(asr_result.text, asr_result.language, session_id)

    def process_text(
        self, text: str, language: Optional[str] = None, session_id: Optional[str] = None
    ) -> PipelineResult:
        text = text.strip()
        if not text:
            logger.warning("process_text called with empty input  session=%s", session_id or "new")
            raise ValueError("Empty text input")
        lang_code = language if (language and language != "auto") else _detect_text_language(text)
        logger.info(
            "Text input received  session=%s language=%s chars=%d",
            session_id or "new", lang_code, len(text),
        )
        return self._finish(text, lang_code, session_id)

    def _finish(self, transcript: str, lang_code: str, session_id: Optional[str]) -> PipelineResult:
        session_id = session_id or self.memory.new_session_id()
        lang_info = LANGUAGE_BY_CODE.get(lang_code, LANGUAGE_BY_CODE[DEFAULT_LANGUAGE])

        try:
            english_text = self.translator.translate_to_english(transcript, lang_info.code)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Translation (to English) failed  session=%s", session_id)
            raise RuntimeError(f"Translation failed: {exc}") from exc

        if not english_text:
            logger.warning("Translation produced empty output  session=%s", session_id)
            raise ValueError("Could not extract any text to answer from.")
        logger.info("Translated to English  session=%s chars=%d", session_id, len(english_text))

        history = self.memory.get_history(session_id)
        triage, contexts, used_fallback = self._reason_with_fallback(session_id, english_text, history)

        self.memory.append_turn(session_id, english_text, triage.answer)
        logger.info("Session memory updated  session=%s turns_stored=%d", session_id, len(self.memory.get_history(session_id)) // 2)

        function_result: Optional[FunctionCallResult] = dispatch_function(triage.next_action)
        if function_result:
            logger.info(
                "Function dispatched  session=%s action=%s", session_id, function_result.action,
            )
        sources = sorted({url for c in contexts for url in c.sources})

        return PipelineResult(
            session_id=session_id,
            transcript=transcript,
            detected_language=lang_info.code,
            detected_language_name=lang_info.name,
            english_text=english_text,
            answer=triage.answer,
            is_grounded=any(c.is_grounded for c in contexts),
            sources=sources,
            triage=triage.triage,
            confidence=triage.confidence,
            possible_conditions=triage.possible_conditions,
            red_flags=triage.red_flags,
            is_emergency=(triage.triage == "emergency" or triage.next_action == "emergency_escalation"),
            function_note=function_result.note if function_result else None,
            used_fallback=used_fallback,
        )

    def _reason_with_fallback(
        self, session_id: str, english_text: str, history: list[dict]
    ) -> tuple[TriageResult, list[KnowledgeBaseResult], bool]:
        """Runs the full extract -> retrieve -> reason pipeline. On any
        failure (Gemma JSON unparseable, retrieval planning stage down,
        etc.) falls back to OllamaAnswerer's single-call path so the
        person still gets an answer instead of an error."""
        try:
            logger.info("Gemma stage 1 (extract_and_plan) starting  session=%s", session_id)
            extraction: ExtractionResult = self.reasoner.extract_and_plan(english_text, history)
            logger.info(
                "Gemma stage 1 complete  session=%s urgency=%s topics=%s",
                session_id, extraction.urgency, extraction.possible_topics,
            )

            contexts = match_topics(extraction.possible_topics) if extraction.possible_topics else []
            logger.info(
                "WHO retrieval  session=%s matched=%d grounded=%d",
                session_id, len(contexts), sum(1 for c in contexts if c.is_grounded),
            )

            logger.info("Gemma stage 2 (clinical_reasoning) starting  session=%s", session_id)
            triage = self.reasoner.clinical_reasoning(english_text, extraction, contexts, history)
            if not triage.answer:
                raise ReasoningParseError("Gemma returned an empty answer field")
            logger.info(
                "Gemma stage 2 complete  session=%s triage=%s next_action=%s confidence=%.2f",
                session_id, triage.triage, triage.next_action, triage.confidence,
            )
            return triage, contexts, False
        except (ReasoningParseError, RuntimeError) as exc:
            logger.warning(
                "Multi-stage reasoning failed  session=%s error=%s — falling back to single-call answer.",
                session_id, exc,
            )

        # -- fallback: old single-topic keyword match + one plain Gemma call --
        try:
            kb_result = retrieve_context(english_text)
            answer_text = self.fallback_answerer.answer(english_text, kb_result, history=history)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Fallback answer path also failed  session=%s", session_id)
            raise RuntimeError(f"Answer generation failed: {exc}") from exc

        logger.info("Fallback answer produced  session=%s grounded=%s", session_id, kb_result.is_grounded)
        fallback_triage = TriageResult(triage="unknown", confidence=0.0, answer=answer_text)
        return fallback_triage, [kb_result], True

    def speak(self, english_text: str, target_language: str) -> bytes:
        """Translate `english_text` (typically a PipelineResult.answer)
        back into `target_language` if needed, and synthesize it to
        WAV audio bytes."""
        try:
            native_text = self.translator.translate_from_english(english_text, target_language)
            audio_bytes = self.tts.synthesize(native_text or english_text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Text-to-speech failed  language=%s", target_language)
            raise RuntimeError(f"Speech synthesis failed: {exc}") from exc
        logger.info("TTS complete  language=%s audio_bytes=%d", target_language, len(audio_bytes))
        return audio_bytes

    def reset_session(self, session_id: str) -> None:
        self.memory.reset(session_id)
        logger.info("Session memory cleared  session=%s", session_id)
