"""
Wrapper around AI4Bharat's IndicConformer 600M multilingual model.

Loading and calling convention follows the model's own HF model card
(`ai4bharat/indic-conformer-600m-multilingual`, `trust_remote_code=True`,
called as `model(wav, lang_code, decoding)`), not a NeMo restore_from —
this repo uses the transformers-native AutoModel interface AI4Bharat
ships for this checkpoint.

IMPORTANT — language detection caveat:
IndicConformer requires you to *tell it* which language to decode in;
it has no built-in "auto-detect" mode. `detect_and_transcribe()` below
is a best-effort workaround: it decodes the same audio with a short
list of candidate languages and picks the most speech-plausible
result by word-rate heuristics. This is meaningfully weaker than a
dedicated language-ID model and costs one extra decode per candidate.
For production accuracy, prefer letting the user pick their language
explicitly (the website's language dropdown defaults to this), or
swap in a dedicated LID model such as AI4Bharat's IndicLID.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio

from model.config import AUTO_DETECT_SHORTLIST, Settings, settings as default_settings

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    text: str
    language: str
    decoding: str


class IndicASR:
    """Lazy-loading wrapper: the (large) model is only pulled into memory
    on first use, not at import time."""

    def __init__(self, cfg: Settings = default_settings):
        self.cfg = cfg
        self._model = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel

        self._device = _resolve_device(self.cfg.device)
        logger.info("Loading IndicConformer '%s' onto %s (first use only)...",
                    self.cfg.asr_model_id, self._device)
        self._model = AutoModel.from_pretrained(
            self.cfg.asr_model_id, trust_remote_code=True
        )
        if self._device == "cuda":
            self._model = self._model.to("cuda")
        logger.info("IndicConformer loaded.")

    def _load_waveform(self, audio_bytes: bytes) -> torch.Tensor:
        with io.BytesIO(audio_bytes) as buf:
            wav, sr = torchaudio.load(buf)
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono, shape (1, T)
        if sr != self.cfg.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.cfg.sample_rate)
            wav = resampler(wav)
        return wav

    def transcribe(self, audio_bytes: bytes, language: str, decoding: Optional[str] = None) -> ASRResult:
        """Transcribe audio in a known, explicitly-specified language."""
        self._ensure_loaded()
        decoding = decoding or self.cfg.asr_decoding
        wav = self._load_waveform(audio_bytes)
        if self._device == "cuda":
            wav = wav.to("cuda")

        with torch.no_grad():
            text = self._model(wav, language, decoding)

        return ASRResult(text=str(text).strip(), language=language, decoding=decoding)

    def detect_and_transcribe(
        self,
        audio_bytes: bytes,
        candidates: Optional[list[str]] = None,
        decoding: Optional[str] = None,
    ) -> ASRResult:
        """Best-effort language auto-detection (see module docstring)."""
        self._ensure_loaded()
        decoding = decoding or self.cfg.asr_decoding
        candidates = candidates or AUTO_DETECT_SHORTLIST
        wav = self._load_waveform(audio_bytes)
        duration_sec = wav.shape[-1] / self.cfg.sample_rate
        if self._device == "cuda":
            wav = wav.to("cuda")

        best: Optional[ASRResult] = None
        best_score = float("-inf")

        for lang in candidates:
            with torch.no_grad():
                text = str(self._model(wav, lang, decoding)).strip()
            score = _plausibility_score(text, duration_sec)
            logger.debug("Auto-detect candidate lang=%s score=%.3f text=%r", lang, score, text)
            if score > best_score:
                best_score = score
                best = ASRResult(text=text, language=lang, decoding=decoding)

        assert best is not None
        return best


def _plausibility_score(text: str, duration_sec: float) -> float:
    """Heuristic: penalize empty output and implausible words-per-second
    rates; reward transcripts near a typical 1.5-3.5 wps speaking rate."""
    words = text.split()
    if not words or duration_sec <= 0:
        return -1.0
    wps = len(words) / duration_sec
    if 1.0 <= wps <= 4.5:
        rate_score = 1.0 - abs(wps - 2.5) / 2.5
    else:
        rate_score = -0.5
    length_bonus = min(len(words) / 5.0, 1.0)
    return rate_score + 0.3 * length_bonus


def _resolve_device(device_setting: str) -> str:
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting
