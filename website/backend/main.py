"""
Sanjeevani website backend.

Serves the static frontend (website/frontend/) and the API that
drives it: speech-or-text in any supported Indian language in,
English-grounded triage-style answer out.

Run from the repository root:
    uvicorn website.backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make the repo root importable so `import model...` (the sibling
# top-level package) works regardless of the working directory this
# is launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from model.config import SUPPORTED_LANGUAGES  # noqa: E402
from model.pipeline import PipelineResult, SanjeevaniPipeline  # noqa: E402

from .schemas import AskResponse, LanguageOption, TextAskRequest  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FRONTEND_DIR = REPO_ROOT / "website" / "frontend"

app = FastAPI(title="Sanjeevani", description="Speak or type. Understand your health.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real deployment origin(s) in production
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {"pipeline": None}

MAX_AUDIO_BYTES = 15 * 1024 * 1024  # ~15MB, generous for a short voice note
ALLOWED_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "webm", "m4a"}


@app.on_event("startup")
def startup_event() -> None:
    # Models are lazy-loaded on first request inside SanjeevaniPipeline's
    # own components, so this is cheap — it does not pull any weights.
    _state["pipeline"] = SanjeevaniPipeline()
    logger.info("Sanjeevani pipeline initialized (models load lazily on first request).")


def _get_pipeline() -> SanjeevaniPipeline:
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service is starting up; try again shortly.")
    return pipeline


def _to_response(result: PipelineResult) -> AskResponse:
    return AskResponse(
        transcript=result.transcript,
        detected_language=result.detected_language,
        detected_language_name=result.detected_language_name,
        english_text=result.english_text,
        answer=result.answer,
        is_grounded=result.is_grounded,
        sources=result.sources,
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/languages", response_model=list[LanguageOption])
def languages() -> list[LanguageOption]:
    return [
        LanguageOption(code=lang.code, name=lang.name, native_name=lang.native_name)
        for lang in SUPPORTED_LANGUAGES
    ]


@app.post("/api/ask/text", response_model=AskResponse, responses={400: {"model": None}, 500: {"model": None}})
def ask_text(payload: TextAskRequest) -> AskResponse:
    pipeline = _get_pipeline()
    try:
        result = pipeline.process_text(payload.text, language=payload.language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _to_response(result)


@app.post("/api/ask/audio", response_model=AskResponse)
async def ask_audio(
    audio: UploadFile = File(...),
    language: str = Form(default="auto"),
) -> AskResponse:
    pipeline = _get_pipeline()

    ext = (audio.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Supported: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    audio_bytes = await audio.read()
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail="Audio file too large.")

    try:
        result = pipeline.process_audio(audio_bytes, language=language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _to_response(result)


# -- Static frontend --------------------------------------------------------
# Mounted last so it doesn't shadow the /api/* routes above.
app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
