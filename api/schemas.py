"""Pydantic schemas for the /transcribe endpoint and health check."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Detected or specified language code")
    processing_time: float = Field(..., description="End-to-end processing time in seconds")
    confidence: Optional[float] = Field(None, description="Model confidence score, if available")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ErrorResponse(BaseModel):
    detail: str
