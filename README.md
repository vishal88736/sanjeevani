# Sanjeevani

Speak or type a health question in any major Indian language. Sanjeevani transcribes it,
translates it to English, has **Gemma 4 reason through it in two stages** — extracting
symptoms and planning what to retrieve, then weighing that against real WHO reference
material to produce a triage judgment — remembers the conversation, and can read the
answer back to you in your own language.

```
sanjeevani/
├── website/                 All website content
│   ├── frontend/              static HTML/CSS/JS UI
│   └── backend/                FastAPI server (serves the UI + the API)
├── model/                    All model content — see the full breakdown below
├── requirements.txt
├── .env.example
└── README.md
```

---

## Architecture: how Gemma 4 actually reasons here

Early versions of this project used Gemma for one thing — phrasing an answer after a
plain keyword search. That undersells what a reasoning model can do, and undersells it
to hackathon judges specifically. The current design gives Gemma 4 **two distinct jobs**,
each with a focused prompt, sitting around a real WHO retrieval step:

```
Speech or text
      │
      ▼
IndicConformer (ASR)              ◄── AI4Bharat, real pretrained model
      │  native-language transcript
      ▼
IndicTrans2 (indic → en)          ◄── AI4Bharat, real pretrained model
      │  English text
      ▼
┌─────────────────────────────────┐
│ GEMMA 4 — Stage 1                │
│ extract_and_plan()                │
│ symptoms, duration, severity,     │
│ age group, urgency, and a short   │
│ list of WHO topics worth checking │
└─────────────────────────────────┘
      │  possible_topics: ["malaria", "dengue", ...]
      ▼
knowledge_base.match_topics()      ◄── fuzzy-matches topics against a curated
      │  WHO fact-sheet excerpts       WHO_FACT_SHEETS table, live-fetches who.int
      ▼
┌─────────────────────────────────┐
│ GEMMA 4 — Stage 2                │
│ clinical_reasoning()              │
│ weighs symptoms + WHO excerpts,   │
│ produces triage judgment,         │
│ possible conditions, red flags,   │
│ a plain-language answer, and a    │
│ next_action decision              │
└─────────────────────────────────┘
      │  next_action: "none" | "emergency_escalation" | "find_nearest_hospital"
      ▼
functions.dispatch()                ◄── "function calling": backend executes
      │                                  whatever Gemma decided
      ▼
Answer shown + (on request) spoken back via IndicTrans2 (en → indic) + Indic Parler-TTS
```

**Why two Gemma calls instead of five-plus:** an earlier design sketch for this project
proposed five to seven separate Gemma calls (multilingual understanding, dialect
normalization, entity extraction, retrieval planning, clinical reasoning, triage
formatting, function-calling). Two of those are already done for free by the ASR +
IndicTrans2 steps above — asking Gemma to re-normalize text that's already been properly
translated would be duplicate work, not extra reasoning. And clinical reasoning, triage
formatting, and deciding the next action are naturally one generation task, not three —
splitting them apart would only add latency (this is a voice interface; every extra call
is a few more seconds of silence) without Gemma reasoning any harder. What's left —
**extract & plan**, then **reason & decide** — are genuinely separate jobs with different
inputs, so those stayed as two calls.

**A real constraint that shaped this:** Ollama Cloud (where Gemma 4 runs here) does not
support enforced JSON-schema structured outputs — that's a local-Ollama-only feature.
Both Gemma stages get their structured output through careful prompting and low
temperature instead of a hard schema constraint, and `model/llm/reasoning.py` parses the
response defensively (stripping markdown fences, extracting the outermost `{...}`, one
retry on a parse failure). If the whole reasoning pipeline still fails, `model/pipeline.py`
falls back to a single plain Gemma call (`OllamaAnswerer`) rather than showing an error —
worse reasoning quality, but the person still gets an answer.

---

## Logging: every action and error, in one file

Everything of consequence gets logged to **`logs/sanjeevani.log`** (rotating at 5MB,
keeping 5 backups), in addition to the console. This isn't just error logging — it's a
full action trail:

- every HTTP request (method, path, status code, response time), via a middleware in
  `website/backend/main.py`
- every pipeline stage: ASR start/complete, translation, Gemma stage 1 (extraction —
  urgency + topics found), WHO retrieval (how many topics matched, how many were
  actually grounded), Gemma stage 2 (triage result + next_action + confidence), function
  dispatch, session memory updates, and TTS
- every failure, at the right severity: input-validation problems as `WARNING` (e.g.
  empty text, unsupported audio format, Gemma's JSON not parsing), and unexpected
  failures as `ERROR` with a full traceback (e.g. the Ollama API being unreachable)
- when the multi-stage reasoning pipeline falls back to the single-call path, that's
  logged as a `WARNING` with the reason, so you can tell how often the fallback is
  actually triggering

```bash
tail -f logs/sanjeevani.log        # watch it live
grep ERROR logs/sanjeevani.log     # just the failures
grep 'session=abc123' logs/sanjeevani.log   # everything for one conversation
```

`model/logging_config.py` is the single place this is configured — it's idempotent, so
it's safe for both the API and any standalone script (e.g. a future admin/debug script)
to call `setup_logging()` without duplicating log lines.

---

## The `model/` folder, file by file

```
model/
├── __init__.py
├── config.py                  Every setting (model ids, API keys' env-var names,
│                               language table, memory/device settings) in one place.
│                               Nothing is hardcoded elsewhere.
├── logging_config.py           setup_logging() — configures console + rotating file
│                               logging (logs/sanjeevani.log) for the whole app. See
│                               the "Logging" section above for what ends up in it.
├── pipeline.py                 SanjeevaniPipeline — the one class the website backend
│                               calls. Owns every model wrapper below and orchestrates
│                               the full flow described in the diagram above, including
│                               the fallback path if Gemma's reasoning stages fail.
│
├── asr/
│   └── indic_asr.py            IndicASR — wraps AI4Bharat's IndicConformer 600M
│                                multilingual model. transcribe() decodes in a given
│                                language; detect_and_transcribe() is a best-effort
│                                language guess (see "known limitations" below) used
│                                when the person leaves the language on "Auto-detect."
│
├── translation/
│   └── indic_translator.py     IndicTranslator — wraps AI4Bharat's IndicTrans2, both
│                                directions. translate_to_english() handles incoming
│                                speech/text; translate_from_english() prepares Gemma's
│                                answer for text-to-speech in the person's language.
│                                Each direction is a separate checkpoint, lazy-loaded
│                                independently.
│
├── tts/
│   └── indic_tts.py            IndicTTS — wraps AI4Bharat's Indic Parler-TTS.
│                                synthesize() turns text into WAV audio bytes; the
│                                model infers the language from the script of the text
│                                itself. Longer answers are split into sentence-level
│                                chunks (the model's card recommends staying under ~30s
│                                per generation) and concatenated with a short pause.
│
└── llm/
    ├── config lives in ../config.py; this package is all Gemma/WHO/memory logic
    ├── reasoning.py             GemmaReasoner — the two-stage reasoning pipeline
    │                            described above (extract_and_plan, clinical_reasoning),
    │                            plus the defensive JSON parser both stages rely on.
    │                            ExtractionResult and TriageResult are the two
    │                            structured outputs; ReasoningParseError is raised
    │                            (and caught in pipeline.py) when Gemma's JSON can't
    │                            be salvaged even after a retry.
    ├── knowledge_base.py        WHO grounding. WHO_FACT_SHEETS is the curated table
    │                            of topic -> (keywords, title, who.int URL) — this is
    │                            the one place to edit to add more WHO coverage.
    │                            match_topics() fuzzy-matches Gemma's suggested topic
    │                            names against that table (used by the main pipeline);
    │                            retrieve_context() is the older single-keyword-query
    │                            version, kept as part of the fallback path.
    ├── functions.py              The "function calling" stage. dispatch() maps
    │                            TriageResult.next_action to a real action:
    │                            emergency_escalation() (just a strong structured
    │                            signal — no external call) or find_nearest_hospital()
    │                            (an explicit PLACEHOLDER — no facility directory or
    │                            maps API is wired up yet; it returns an honest note
    │                            saying so rather than pretending to find something).
    ├── memory.py                 ConversationStore — process-local, in-memory,
    │                            per-session conversation history (capped at
    │                            MAX_HISTORY_TURNS turns), fed into both Gemma stages
    │                            so follow-up questions work. Resets on server
    │                            restart; not shared across multiple backend
    │                            processes — swap for Redis/a DB before any real
    │                            multi-user deployment (noted again in "extending" below).
    └── ollama_client.py          get_client() — the shared, authenticated Ollama
                                 Cloud client both reasoning.py and OllamaAnswerer use.
                                 OllamaAnswerer is the simple single-call fallback
                                 (no structured extraction/triage) used when the
                                 multi-stage pipeline can't complete.
```

## What's real, what's a heuristic, and what's a placeholder

- **Speech recognition, both translation directions, and text-to-speech are real,
  pretrained AI4Bharat models.** Nothing there is mocked.
- **Gemma 4's reasoning is real** — both stages are genuine model calls that shape what
  gets retrieved and how the answer is triaged, not just a phrasing pass over a fixed
  template.
- **WHO grounding is real but narrow**: a curated table of ~18 topics, fuzzy-matched
  against whatever Gemma's extraction stage suggests, with a live fetch of the actual
  who.int page. There's no general WHO search API to fall back on — growing coverage
  means adding rows to `WHO_FACT_SHEETS`.
- **The hospital-finder function is an explicit placeholder** — see `functions.py`. It's
  wired into the reasoning loop (Gemma can trigger it) but returns an honest "not
  connected yet" note rather than a fake result.
- **Conversation memory is real but process-local** (see `memory.py` above).
- **Audio language auto-detection is still a documented weak point** — see below.

## 1. Install

```bash
cd sanjeevani
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**System dependency:** browser-recorded audio typically arrives as WebM/Opus.
`torchaudio` needs an `ffmpeg` install on your system PATH to decode it
(`apt install ffmpeg` / `brew install ffmpeg`). Uploaded `.wav`/`.mp3`/`.flac`/`.ogg`
files work either way.

## 2. Add your Ollama API key

```bash
cp .env.example .env
# then edit .env and set:
# OLLAMA_API_KEY=your-key-here
```

Create a key at [ollama.com/settings/keys](https://ollama.com/settings/keys). This is
**Ollama Cloud**, not a locally-run model — Gemma 4 runs on Ollama's servers, using their
`gemma4:cloud` model id.

## 3. Hugging Face access (for the AI4Bharat models)

```bash
huggingface-cli login
```

IndicConformer, both IndicTrans2 checkpoints, and Indic Parler-TTS all download
automatically on first use and are cached locally afterwards.

## 4. Run it

```bash
uvicorn website.backend.main:app --reload --port 8000
```

Open **http://localhost:8000**. The first request that actually uses a model (ASR,
translation, Gemma 4, or TTS) will be slow while that model loads and caches —
everything after that is fast.

## Conversation memory in the UI

Every response returns a `session_id`. The frontend keeps reusing it as you ask follow-up
questions ("Continue this conversation"), so Gemma 4 can resolve things like "is that
serious?" referring to what you asked before. "Start a new conversation" clears the
session both client-side and on the server (`POST /api/conversation/{session_id}/reset`).

## A known limitation: audio language auto-detection

IndicConformer doesn't have a built-in "detect the language" mode — you have to tell it
which language to decode. `detect_and_transcribe()` in `model/asr/indic_asr.py` works
around this by decoding the same clip against a handful of candidate languages and
picking the one whose output looks most like plausible speech (word-rate heuristics).
This is noticeably weaker than a dedicated language-ID model. For reliable results,
prefer having the person pick their language from the dropdown. For a stronger fix, look
at AI4Bharat's dedicated **IndicLID** model as a drop-in replacement for the heuristic.

## Extending this

- **Add more WHO topics:** add rows to `WHO_FACT_SHEETS` in `model/llm/knowledge_base.py`.
- **Swap the grounding source entirely:** `match_topics()` / `retrieve_context()` in the
  same file are the functions to replace.
- **Connect a real hospital directory:** replace the body of `find_nearest_hospital()` in
  `model/llm/functions.py`.
- **Add another function Gemma can call:** write the function in `functions.py`, add it
  to `dispatch()`, and mention the new `next_action` value in `reasoning.py`'s
  `REASONING_SYSTEM_PROMPT`.
- **Add more languages:** extend `SUPPORTED_LANGUAGES` in `model/config.py` (needs an
  IndicConformer language code and a FLORES code for both IndicTrans2 directions).
- **Make memory durable across restarts / multi-instance:** replace the dict in
  `model/llm/memory.py::ConversationStore` with Redis or a database, keyed the same way.
- **Swap the LLM provider:** `model/llm/ollama_client.py::get_client()` is the one place
  that authenticates with Ollama; point it at Google's Gemini API directly, or another
  OpenAI-compatible endpoint, if needed.
