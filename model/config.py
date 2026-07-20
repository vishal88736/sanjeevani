"""
Central configuration for the Sanjeevani model layer.

All secrets (Groq API key) and model choices are read from environment
variables so nothing is hardcoded. Copy `.env.example` to `.env` and
fill in your own GROQ_API_KEY before running the backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()  # reads a .env file in the working directory, if present


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class LanguageInfo:
    code: str          # short UI code (matches AI4Bharat IndicConformer's lang arg)
    name: str          # English display name
    native_name: str   # name written in the language's own script
    flores_code: str   # FLORES-200 code used by IndicTrans2


# A practical subset of the 22 scheduled languages. IndicConformer and
# IndicTrans2 both support the full set; extend this table with more
# entries (see each model's card for the remaining language/FLORES
# codes) if your deployment needs them.
SUPPORTED_LANGUAGES: list[LanguageInfo] = [
    LanguageInfo("hi", "Hindi", "हिन्दी", "hin_Deva"),
    LanguageInfo("bn", "Bengali", "বাংলা", "ben_Beng"),
    LanguageInfo("ta", "Tamil", "தமிழ்", "tam_Taml"),
    LanguageInfo("te", "Telugu", "తెలుగు", "tel_Telu"),
    LanguageInfo("mr", "Marathi", "मराठी", "mar_Deva"),
    LanguageInfo("gu", "Gujarati", "ગુજરાતી", "guj_Gujr"),
    LanguageInfo("kn", "Kannada", "ಕನ್ನಡ", "kan_Knda"),
    LanguageInfo("ml", "Malayalam", "മലയാളം", "mal_Mlym"),
    LanguageInfo("pa", "Punjabi", "ਪੰਜਾਬੀ", "pan_Guru"),
    LanguageInfo("or", "Odia", "ଓଡ଼ିଆ", "ory_Orya"),
    LanguageInfo("as", "Assamese", "অসমীয়া", "asm_Beng"),
    LanguageInfo("ur", "Urdu", "اردو", "urd_Arab"),
    LanguageInfo("en", "English", "English", "eng_Latn"),
]

LANGUAGE_BY_CODE: dict[str, LanguageInfo] = {lang.code: lang for lang in SUPPORTED_LANGUAGES}

# Shortlist tried during audio auto-detection. Kept small on purpose:
# each candidate costs one extra decode pass. See
# model/asr/indic_asr.py::detect_and_transcribe for the heuristic and
# its documented limitations.
AUTO_DETECT_SHORTLIST = ["hi", "bn", "ta", "te", "mr", "en"]

DEFAULT_LANGUAGE = "hi"


@dataclass(frozen=True)
class Settings:
    # -- ASR (speech -> native-language text) --------------------------------
    asr_model_id: str = field(default_factory=lambda: _env(
        "ASR_MODEL_ID", "ai4bharat/indic-conformer-600m-multilingual"
    ))
    asr_decoding: str = field(default_factory=lambda: _env("ASR_DECODING", "ctc"))  # ctc | rnnt

    # -- Translation (native-language text -> English) ------------------------
    translation_model_id: str = field(default_factory=lambda: _env(
        "TRANSLATION_MODEL_ID", "ai4bharat/indictrans2-indic-en-dist-200M"
    ))

    # -- Groq LLM (English question -> English answer) ------------------------
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "llama-3.3-70b-versatile"))

    # -- Runtime ---------------------------------------------------------------
    device: str = field(default_factory=lambda: _env("MODEL_DEVICE", "auto"))
    sample_rate: int = 16000


settings = Settings()
