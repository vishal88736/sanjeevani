"""Request/response schemas for the Sanjeevani website API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TextAskRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    language: Optional[str] = Field(
        default="auto", description="Language code, e.g. 'hi', or 'auto' to detect"
    )


class AskResponse(BaseModel):
    transcript: str
    detected_language: str
    detected_language_name: str
    english_text: str
    answer: str
    is_grounded: bool
    sources: list[str]


class LanguageOption(BaseModel):
    code: str
    name: str
    native_name: str


class ErrorResponse(BaseModel):
    detail: str
