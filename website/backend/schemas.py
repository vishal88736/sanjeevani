"""Request/response schemas for the Sanjeevani website API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TextAskRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: Optional[str] = Field(
        default="auto", description="Language code, e.g. 'hi', or 'auto' to detect"
    )
    session_id: Optional[str] = Field(
        default=None, description="Omit on the first message; reuse the returned id to continue the conversation"
    )


class AskResponse(BaseModel):
    session_id: str
    transcript: str
    detected_language: str
    detected_language_name: str
    english_text: str
    answer: str
    is_grounded: bool
    sources: list[str]

    # -- Gemma reasoning/triage detail --
    triage: str
    confidence: float
    possible_conditions: list[str]
    red_flags: list[str]
    is_emergency: bool
    function_note: Optional[str] = None
    used_fallback: bool


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: str = Field(..., description="Language to speak the text in, e.g. 'hi' or 'en'")


class LanguageOption(BaseModel):
    code: str
    name: str
    native_name: str


class ErrorResponse(BaseModel):
    detail: str
