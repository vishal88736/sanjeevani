"""
GemmaReasoner: the multi-stage reasoning pipeline that makes Gemma 4 do
the actual clinical reasoning, not just phrase a keyword-search result.

Two focused Gemma calls per turn:

  1. extract_and_plan()   English text -> structured symptoms/duration/
                          severity + a short list of WHO topics worth
                          retrieving (this is the "retrieval planning"
                          stage — it replaces a fixed keyword table with
                          Gemma's own judgment about what's relevant).

  2. clinical_reasoning() structured symptoms + retrieved WHO excerpts
                          + conversation history -> a triage judgment
                          (urgency, possible conditions, red flags,
                          recommendation) and a plain-language answer,
                          plus a `next_action` the backend should take
                          (this is the "function calling" stage —
                          Gemma decides whether to escalate to an
                          emergency notice or suggest finding a nearby
                          hospital; model/pipeline.py executes it).

Why two calls and not five-plus: the "multilingual understanding" and
"dialect normalization" stages from the original proposal are already
done for free by the existing ASR (IndicConformer) + translation
(IndicTrans2) steps — asking Gemma to redo that would be duplicate
work for no extra reasoning benefit. And "clinical reasoning", "triage
JSON", and "decide the next action" are naturally one generation task,
not three independent ones — splitting them into separate calls would
only add latency (this is a voice interface; every extra call is a
few more seconds of silence) without Gemma reasoning any harder.

Important caveat: Ollama Cloud does not support enforced JSON-schema
structured outputs (that's a local-Ollama-only feature as of this
writing) — so both stages get Gemma to emit JSON through careful
prompting + low temperature, not a hard schema constraint. Responses
are parsed defensively (stripping markdown fences, extracting the
outermost {...}) with one retry on a parse failure before giving up.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from model.config import Settings, settings as default_settings
from model.llm.knowledge_base import KnowledgeBaseResult
from model.llm.ollama_client import get_client

logger = logging.getLogger(__name__)


class ReasoningParseError(RuntimeError):
    """Raised when Gemma's output couldn't be parsed as the expected JSON,
    even after a retry. Callers (model/pipeline.py) catch this and fall
    back to OllamaAnswerer's simple single-call path."""


@dataclass
class ExtractionResult:
    symptoms: list[str] = field(default_factory=list)
    duration: str = "unknown"
    severity: str = "unknown"
    age_group: str = "unknown"
    possible_topics: list[str] = field(default_factory=list)
    urgency: str = "unknown"  # low | medium | high


@dataclass
class TriageResult:
    triage: str = "routine"              # self_care | routine | urgent | emergency
    confidence: float = 0.5
    possible_conditions: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    recommendation: str = ""
    reasoning: str = ""
    answer: str = ""
    next_action: str = "none"            # none | emergency_escalation | find_nearest_hospital


def _get_extraction_prompt(mode: str = "patient") -> str:
    target = "an ASHA worker or community health worker" if mode == "asha_worker" else "a person"
    return f"""You are the language-understanding stage of Sanjeevani, a \
health-information assistant for rural India. You will be given an English description of \
{target}'s query (already translated from their own language) and, if this is a \
follow-up question, the earlier conversation.

Extract structured information and suggest which health topics are worth consulting.

Respond with ONLY a single valid JSON object — no markdown fences, no explanation, no text \
before or after it — matching exactly this shape:
{{
  "symptoms": ["<short symptom phrases>"],
  "duration": "<how long, or 'unknown'>",
  "severity": "<mild|moderate|severe|unknown>",
  "age_group": "<infant|child|adult|elderly|unknown>",
  "possible_topics": ["<up to 4 short disease/condition names worth checking, e.g. 'malaria', 'dengue'>"],
  "urgency": "<low|medium|high>"
}}
If the message isn't really a health question, return empty/"unknown" fields and an empty \
possible_topics list."""


def _get_reasoning_prompt(mode: str = "patient") -> str:
    if mode == "asha_worker":
        role_description = "You are assisting an ASHA worker (community health worker) in rural India."
        answer_rules = (
            "- \"answer\" should use medical terminology suitable for a health worker. "
            "Provide differential diagnosis hints, drug class suggestions (but not specific dosages), "
            "and clear referral criteria (e.g., 'refer if fever >3 days'). Use a checklist-style format if helpful."
        )
    else:
        role_description = "You are assisting a layperson in rural and semi-urban India."
        answer_rules = (
            "- \"answer\" must be short, plain, calm, written for someone reading or listening in their "
            "second or third language, and must tell them to confirm with a real doctor, nurse, or "
            "health worker (e.g. an ASHA worker) — never state a diagnosis as fact in \"answer\"."
        )

    return f"""You are the clinical-reasoning stage of Sanjeevani, a \
preliminary health-information assistant. {role_description} You are NOT a \
doctor and must not diagnose with certainty. You'll receive structured symptom \
information, optionally some reference material excerpts, and the \
conversation so far.

Weigh the symptoms against the reference material (if any) and produce a cautious triage \
judgment and an answer.

Rules:
- Never invent statistics, drug dosages, or a confident diagnosis.
- If anything suggests a medical emergency (e.g. chest pain, difficulty breathing, severe \
bleeding, loss of consciousness, suspected stroke, a child with a very high fever or \
seizures), set "triage" to "emergency" and "next_action" to "emergency_escalation".
- If in-person care would help but it isn't urgent, set "next_action" to \
"find_nearest_hospital"; otherwise "none".
{answer_rules}
- If a reference excerpt was provided and used, say so plainly in "answer" (e.g. "According to WHO/NIH/First Aid...").
- If no reference excerpt was provided, be extra cautious in "answer" and lean toward recommending in-person care.

Respond with ONLY a single valid JSON object — no markdown fences, no explanation, no text \
before or after it — matching exactly this shape:
{{
  "triage": "<self_care|routine|urgent|emergency>",
  "confidence": <0.0-1.0>,
  "possible_conditions": ["<short condition names, may be empty>"],
  "red_flags": ["<specific concerning signs found, may be empty>"],
  "recommendation": "<one short sentence: what to do next>",
  "reasoning": "<1-3 sentences explaining the triage judgment, for internal/debug display>",
  "answer": "<the actual response to show and speak to the user>",
  "next_action": "<none|emergency_escalation|find_nearest_hospital>"
}}"""


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_outer_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ReasoningParseError(f"No JSON object found in Gemma's response: {text[:200]!r}")
    return text[start : end + 1]


def _parse_json_response(raw: str) -> dict:
    candidate = _extract_outer_json(_strip_json_fences(raw))
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ReasoningParseError(f"Could not parse Gemma's JSON output: {exc}") from exc


class GemmaReasoner:
    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            self._client = get_client(self.cfg)
        return self._client

    def _chat_json(self, system_prompt: str, user_content: str, history: Optional[list[dict]] = None) -> dict:
        client = self._ensure_client()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_content})

        last_error: Optional[Exception] = None
        for attempt in range(2):  # one retry on a parse failure
            try:
                response = client.chat(
                    model=self.cfg.ollama_model,
                    messages=messages,
                    options={"temperature": 0.1},
                )
                raw = response["message"]["content"]
                return _parse_json_response(raw)
            except ReasoningParseError as exc:
                last_error = exc
                logger.warning("Gemma JSON parse failed (attempt %d): %s", attempt + 1, exc)
                messages.append({
                    "role": "user",
                    "content": "Your previous reply was not valid JSON. Respond again with "
                    "ONLY the JSON object, nothing else.",
                })
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ollama request failed during reasoning stage")
                raise RuntimeError(f"Reasoning request failed: {exc}") from exc

        raise last_error or ReasoningParseError("Unknown JSON parse failure")

    def extract_and_plan(self, english_text: str, history: Optional[list[dict]] = None, mode: str = "patient") -> ExtractionResult:
        data = self._chat_json(_get_extraction_prompt(mode), english_text, history)
        return ExtractionResult(
            symptoms=list(data.get("symptoms") or []),
            duration=str(data.get("duration", "unknown")),
            severity=str(data.get("severity", "unknown")),
            age_group=str(data.get("age_group", "unknown")),
            possible_topics=list(data.get("possible_topics") or []),
            urgency=str(data.get("urgency", "unknown")),
        )

    def clinical_reasoning(
        self,
        english_text: str,
        extraction: ExtractionResult,
        contexts: list[KnowledgeBaseResult],
        history: Optional[list[dict]] = None,
        mode: str = "patient",
    ) -> TriageResult:
        context_block = "\n\n".join(c.context for c in contexts if c.context) or "(no reference excerpt matched)"
        user_content = (
            f"Query: {english_text}\n\n"
            f"Extracted info: {json.dumps({'symptoms': extraction.symptoms, 'duration': extraction.duration, 'severity': extraction.severity, 'age_group': extraction.age_group, 'urgency': extraction.urgency})}\n\n"
            f"Reference material:\n{context_block}"
        )
        data = self._chat_json(_get_reasoning_prompt(mode), user_content, history)
        return TriageResult(
            triage=str(data.get("triage", "routine")),
            confidence=float(data.get("confidence", 0.5) or 0.5),
            possible_conditions=list(data.get("possible_conditions") or []),
            red_flags=list(data.get("red_flags") or []),
            recommendation=str(data.get("recommendation", "")),
            reasoning=str(data.get("reasoning", "")),
            answer=str(data.get("answer", "")).strip(),
            next_action=str(data.get("next_action", "none")),
        )
