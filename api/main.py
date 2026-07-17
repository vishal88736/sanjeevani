"""
FastAPI service for the Sanjeevani ASR module.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health      -> service + model status
    POST /transcribe   -> multipart file upload -> transcription JSON

This module is intentionally independent of any downstream medical
reasoning component (e.g. Gemma 4); it only ever returns transcribed
text plus metadata, per the "Speech -> Text only" scope of this module.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from inference.transcriber import Transcriber
from utils.config_utils import load_config
from utils.logging_utils import setup_logging

from .schemas import ErrorResponse, HealthResponse, TranscriptionResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sanjeevani ASR API",
    description="Multilingual Indian-language Speech-to-Text service.",
    version="1.0.0",
)

_state: dict = {"transcriber": None, "cfg": None}


@app.on_event("startup")
def startup_event() -> None:
    cfg = load_config("configs/config.yaml")
    setup_logging(cfg.project.log_dir, level=cfg.logging.level)
    logger.info("Loading ASR model for API service...")

    _state["cfg"] = cfg
    _state["transcriber"] = Transcriber.from_config(
        model_cfg=cfg.model,
        audio_cfg=cfg.dataset.audio,
        device=cfg.hardware.device,
    )
    logger.info("Model loaded. API ready to serve requests.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    transcriber = _state.get("transcriber")
    cfg = _state.get("cfg")
    return HealthResponse(
        status="ok" if transcriber is not None else "loading",
        model_loaded=transcriber is not None,
        device=cfg.hardware.device if cfg is not None else "unknown",
    )


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file: wav, mp3, flac, or ogg"),
    language: str | None = Query(
        default=None, description="ISO language code hint (e.g. 'hi', 'ta'); optional."
    ),
    decoding_strategy: str = Query(default="beam", pattern="^(beam|greedy)$"),
) -> TranscriptionResponse:
    transcriber: Transcriber | None = _state.get("transcriber")
    cfg = _state.get("cfg")
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Model is still loading; try again shortly.")

    supported_ext = {"wav", "mp3", "flac", "ogg"}
    ext = (audio.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in supported_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Supported: {sorted(supported_ext)}",
        )

    audio_bytes = await audio.read()
    max_bytes = cfg.api.max_audio_duration_sec * cfg.dataset.audio.target_sample_rate * 4 if cfg else None
    if max_bytes and len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Audio exceeds max duration of {cfg.api.max_audio_duration_sec}s.",
        )

    start = time.time()
    try:
        result = transcriber.transcribe_bytes(
            audio_bytes,
            filename_hint=audio.filename or "upload.wav",
            language=language or (cfg.api.default_language if cfg else None),
            decoding_strategy=decoding_strategy,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

    elapsed = time.time() - start
    return TranscriptionResponse(
        text=result.text,
        language=result.language,
        processing_time=elapsed,
        confidence=result.confidence,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):  # noqa: ANN001
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
