# Sanjeevani ASR

Production-grade multilingual Speech-to-Text (ASR) module for **Sanjeevani**, an
AI-powered healthcare triage assistant for rural India.

**Scope:** this module handles Speech → Text only. Downstream medical
reasoning (symptom triage, recommendations, etc.) is handled separately by
Gemma 4 and is intentionally out of scope here — the ASR module is built to
be reused standalone in other applications.

- **Base model:** [AI4Bharat IndicConformer 600M Multilingual](https://huggingface.co/ai4bharat) (NVIDIA NeMo)
- **Optional fallback:** OpenAI Whisper (Hugging Face `transformers`)
- **Training data:** [`ai4bharat/IndicVoices`](https://huggingface.co/datasets/ai4bharat/IndicVoices)
- **Serving:** FastAPI (`POST /transcribe`)

---

## Table of contents

- [Folder structure](#folder-structure)
- [A note on the `data_pipeline/` naming](#a-note-on-the-data_pipeline-naming)
- [Installation](#installation)
- [Dataset download](#dataset-download)
- [Model download](#model-download)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [FastAPI service](#fastapi-service)
- [Testing](#testing)
- [Expected hardware / GPU recommendations](#expected-hardware--gpu-recommendations)
- [Extending: adding a new ASR backend](#extending-adding-a-new-asr-backend)
- [Known limitations](#known-limitations)

---

## Folder structure

```
asr/
├── configs/                  # Hydra/YAML configuration
│   ├── config.yaml           # master config (composes the three below)
│   ├── model/
│   │   ├── indic_conformer.yaml
│   │   └── whisper.yaml
│   ├── dataset/
│   │   └── indicvoices.yaml
│   └── training/
│       └── default.yaml
│
├── data_pipeline/             # dataset loading, preprocessing, collation
│   ├── indicvoices_dataset.py
│   └── collator.py
│
├── models/                   # pluggable ASR backends
│   ├── base_asr_model.py     # abstract interface every backend implements
│   ├── indic_conformer.py    # NeMo Conformer wrapper
│   ├── whisper_fallback.py   # optional HF Whisper wrapper
│   └── registry.py           # config -> model factory
│
├── trainer/                  # training loop
│   ├── trainer.py            # ASRTrainer (AMP, grad accum, DDP, resume, ckpt)
│   ├── scheduler.py          # LR schedules
│   └── callbacks.py          # early stopping
│
├── inference/                 # single-file / folder / batch / streaming
│   ├── transcriber.py
│   ├── batch_inference.py
│   └── streaming.py
│
├── evaluation/                # WER / CER + curve plotting
│   ├── metrics.py
│   └── evaluator.py
│
├── utils/                     # audio, logging, checkpoint, config helpers
│   ├── audio_utils.py
│   ├── logging_utils.py
│   ├── checkpoint_utils.py
│   └── config_utils.py
│
├── api/                        # FastAPI service
│   ├── main.py
│   └── schemas.py
│
├── checkpoints/                # downloaded / trained model weights (gitignored)
├── scripts/                    # CLI utilities
│   ├── download_dataset.py
│   ├── download_model.py
│   └── run_inference.py
├── tests/                      # pytest unit tests
│
├── train.py                    # training entrypoint
├── evaluate.py                 # evaluation entrypoint
├── requirements.txt
└── README.md
```

### A note on the `data_pipeline/` naming

The originally requested structure named this folder `datasets/`. That name
was changed to `data_pipeline/` because it would otherwise **shadow the
Hugging Face `datasets` library** on `sys.path` whenever a script is run
from the repository root (`import datasets` would resolve to the local
package instead of the installed library). This is a real, easy-to-hit bug
in Python projects with a top-level `datasets/` folder, so the folder was
renamed for correctness. Everything else follows the requested layout.

---

## Installation

```bash
git clone <your-repo-url> sanjeevani-asr
cd sanjeevani-asr

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**NeMo note:** `nemo_toolkit[asr]` has a heavy dependency tree (PyTorch,
NVIDIA Apex-adjacent kernels, etc.). If you only want to experiment with the
Whisper fallback path first, comment out the `nemo_toolkit` line in
`requirements.txt` and install it later.

---

## Dataset download

`ai4bharat/IndicVoices` is a large multilingual corpus. By default the
pipeline uses **streaming mode** (`configs/dataset/indicvoices.yaml:
streaming: true`) so nothing is downloaded upfront — the `IterableDataset`
pulls and preprocesses audio on the fly during training.

To pre-cache a specific language subset locally instead (recommended for
small experiments where you want map-style shuffling):

```bash
python scripts/download_dataset.py --languages hi ta te --split train --max-samples 2000
```

Then set `streaming: false` in `configs/dataset/indicvoices.yaml` (or via
override, see below) so `train.py` materializes the dataset into memory.

---

## Model download

The IndicConformer `.nemo` checkpoint is pulled automatically from the
Hugging Face Hub on first use and cached under
`checkpoints/indic_conformer/`. To pre-fetch it explicitly:

```bash
python scripts/download_model.py --model indic_conformer
# or, for the optional fallback:
python scripts/download_model.py --model whisper
```

---

## Configuration

Everything is controlled through YAML via Hydra. The master config
(`configs/config.yaml`) composes `model`, `dataset`, and `training`
sub-configs, and every field is overridable on the command line:

```bash
# Train only on Hindi and Tamil, with fp16 instead of bf16
python train.py languages='[hi,ta]' hardware.precision=fp16

# Swap in the Whisper fallback backend
python train.py model=whisper

# Change batch size, learning rate, and epochs
python train.py training.batch_size=16 training.optimizer.lr=1e-5 training.epochs=10

# Resume from the last checkpoint
python train.py training.checkpointing.resume_from=latest
```

Key config files:

| File | Purpose |
|---|---|
| `configs/config.yaml` | languages, hardware, logging, API defaults |
| `configs/model/indic_conformer.yaml` | HF repo id, decoding strategy, freeze/dropout |
| `configs/model/whisper.yaml` | fallback model settings |
| `configs/dataset/indicvoices.yaml` | streaming, splits, audio preprocessing params |
| `configs/training/default.yaml` | batch size, optimizer, scheduler, early stopping, checkpointing |

---

## Training

```bash
python train.py
```

Implemented training features:

- Mixed precision (`fp16` / `bf16` via `torch.autocast` + `GradScaler`)
- Gradient accumulation (`training.gradient_accumulation_steps`)
- Gradient checkpointing (enabled on the encoder when the backend supports it)
- Distributed Data Parallel (`hardware.distributed: true`, launch with `torchrun`)
- Resume-from-checkpoint (`training.checkpointing.resume_from: latest|best|<path>`)
- Automatic + best-checkpoint saving with configurable retention (`keep_last_n`)
- Early stopping on a monitored validation metric (default: `val_wer`)
- Linear-warmup + cosine / constant / Noam LR schedules
- TensorBoard logging (`logs/tensorboard/`) and rotating file logs (`logs/`)

For multi-GPU training:

```bash
torchrun --nproc_per_node=4 train.py hardware.distributed=true hardware.num_gpus=4
```

---

## Evaluation

```bash
python evaluate.py
```

Computes corpus-level **WER** and **CER**, with a per-language breakdown,
and writes `outputs/evaluation_report.json`. Training/validation loss and
WER curves can be plotted after a run via:

```python
from evaluation.evaluator import Evaluator
Evaluator.plot_curves(train_losses, val_losses, val_wers, "outputs/curves.png")
```

(Loss/WER histories are also visible live in TensorBoard: `tensorboard --logdir logs/tensorboard`.)

---

## Inference

### Single file

```bash
python scripts/run_inference.py --input sample.wav --language hi
```

### Folder (batch)

```bash
python scripts/run_inference.py --input ./audio_folder --output results.json --batch-size 16
```

### Programmatic

```python
from utils.config_utils import load_config
from inference.transcriber import Transcriber

cfg = load_config("configs/config.yaml")
transcriber = Transcriber.from_config(model_cfg=cfg.model, audio_cfg=cfg.dataset.audio)

result = transcriber.transcribe_file("sample.wav", language="hi")
print(result.text, result.confidence)
```

### Streaming (placeholder)

True low-latency streaming ASR (incremental encoder state, endpointing,
partial hypotheses) is **out of scope** for this module — `inference/streaming.py`
documents the interface and ships a reference *chunked* implementation
(`StreamingTranscriber`) suitable for demos, not production latency.
NeMo's cache-aware streaming Conformer variants are the natural next step
if/when real-time transcription is required.

---

## FastAPI service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**`GET /health`**

```json
{"status": "ok", "model_loaded": true, "device": "cuda"}
```

**`POST /transcribe`** — multipart upload, optional `language` and
`decoding_strategy` query params:

```bash
curl -X POST "http://localhost:8000/transcribe?language=hi&decoding_strategy=beam" \
     -F "audio=@sample.wav"
```

Response:

```json
{
  "text": "mujhe do din se bukhar hai",
  "language": "hi",
  "processing_time": 0.84,
  "confidence": 0.93
}
```

---

## Testing

```bash
pytest tests/ -v
```

Covers audio preprocessing (resampling, mono conversion, normalization,
silence trimming), WER/CER metric correctness, and dynamic-padding collator
behavior.

---

## Expected hardware / GPU recommendations

| Use case | Recommended GPU | Notes |
|---|---|---|
| Inference only (IndicConformer 600M) | 1x 8–12 GB VRAM (e.g. T4, RTX 3060) | fp16/bf16 inference; CPU inference works but is slow for beam search |
| Fine-tuning, small batch | 1x 16–24 GB VRAM (e.g. A10, RTX 4090) | `batch_size=4-8`, `gradient_accumulation_steps=4-8`, gradient checkpointing on |
| Fine-tuning, production-scale | 4–8x 40–80 GB VRAM (A100/H100) | enable `hardware.distributed=true`, launch via `torchrun` |
| Whisper-large-v3 fallback fine-tuning | 1x 24 GB+ VRAM | comparable footprint to IndicConformer at this size |

CPU-only operation is supported for correctness testing and low-throughput
inference, but is not recommended for training.

---

## Extending: adding a new ASR backend

1. Implement `models/base_asr_model.BaseASRModel` for the new backend (see
   `models/indic_conformer.py` or `models/whisper_fallback.py` as templates).
2. Register a builder function in `models/registry.py`:
   ```python
   _REGISTRY["my_new_backend"] = _build_my_new_backend
   ```
3. Add a `configs/model/my_new_backend.yaml` with `type: my_new_backend`.
4. Everything else — training, evaluation, inference, the API — works
   unchanged, since they're all written against `BaseASRModel`.

---

## Known limitations

- `IndicConformerModel.compute_loss` adapts NeMo's forward/loss API to this
  repo's generic `ASRBatch` interface; NeMo's own training recipes
  typically use its native PyTorch Lightning `Trainer` with its own data
  layers. This wrapper trades some of NeMo's built-in training conveniences
  for a unified interface across backends — for maximum training throughput
  with NeMo specifically, consider using NeMo's native `.fit()` path instead
  and only using this repo's `BaseASRModel` interface for inference/serving.
- Streaming inference is a reference chunked implementation, not true
  low-latency streaming (see [Inference](#inference) above).
- `IndicVoicesDataset` streaming mode processes languages sequentially, not
  interleaved; for a randomized multilingual training mix, pre-download and
  materialize the languages you need (see [Dataset download](#dataset-download)).
