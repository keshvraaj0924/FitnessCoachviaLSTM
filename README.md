# Exercise Analysis API

A complete ML pipeline for analyzing exercise videos — **video in, structured results out** — supporting **multiple exercises** and **real-time streaming**.

## Overview

This system implements a vertical slice of a fitness app backend that:
1. **Extracts features** from uploaded exercise videos using OpenCV (Component A)
2. **Classifies motion phases** (idle, concentric, eccentric) using a **causal LSTM** (Component B)
3. **Counts repetitions** with timestamps using a state machine (Component B)
4. **Generates coaching feedback** via LLM (Component C)
5. **Serves everything** over HTTP + WebSocket with proper model management (Component D)

**Exercises**: `pushup`, `squat`, `bicep_curl`, `jumping_jack` (each has its own trained LSTM checkpoint)
**Data**: Domain-mixed calibration (real human **MM-Fit** + synthetic render) for push-up & squat + calibrated synthetic for all four
**Real-time**: `/v1/stream` WebSocket counts reps frame-by-frame from a live webcam

---

## Features

- **Batch analysis** — upload a video, get reps + timestamps + coaching (`POST /v1/analyze`)
- **Real-time streaming** — a live camera (or replayed file) streams JPEG frames to `/v1/stream`; the server predicts the current phase and counts reps incrementally
- **Multi-exercise** — one endpoint, per-exercise models; select with `?exercise=` / `exercise` field
- **Honest data story** — push-up and squat are calibrated on **real human pose data** (MM-Fit, 21 subjects); bicep_curl and jumping_jack are calibrated on synthetic renders (documented in [NOTES.md](NOTES.md))
- **Causal features + causal model** — identical feature path and LSTM weights for batch and streaming, so live predictions share the training distribution
- **Deterministic LLM fallback** — coaching never fails; a template fallback covers any LLM outage

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key (for coaching feedback)

### Installation

```bash
# Clone and enter directory
cd pushup-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Calibrate, Generate Data & Train Models

```bash
# 1. (Optional, requires the local mm-fit/ dataset, which is not part of this
#    repo) Render the MM-Fit real-human calibration for push-up + squat. This
#    measures the feature statistics of REAL humans from the MM-Fit dataset, so
#    those two models transfer far better to a webcam. If you skip this, the
#    synthetic-render calibration is used instead.
python scripts/mmfit_pose_to_video.py --all

# 2. Regenerate the per-exercise synthetic datasets.
#    For pushup/squat this now samples from the MM-Fit real-human stats;
#    for bicep_curl/jumping_jack it uses the synthetic-render calibration.
python generate_synthetic_data.py --all

# 3. Train every exercise model (causal/unidirectional LSTM, all four on CPU
#    in ~10 minutes total thanks to the lean, calibrated dataset).
python train.py --all
```

This creates `checkpoints/lstm_<exercise>.pt` for each of the four exercises. You can train a single exercise with `python train.py --exercise squat`, or run a quick dev pass with `--fast`.

> **Data note**: generate_synthetic_data.py samples per-phase feature statistics that were **measured** from a calibration source — the real feature extractor running over either the MM-Fit real-human pose renders (pushup, squat, *domain-mixed with synthetic*) or the synthetic exercise clips from scripts/make_demo_video.py (all four). This keeps the training distribution aligned with what the API feeds the model at inference. See NOTES.md for the honest caveats.

Ready-made synthetic demo videos are included for smoke-testing:
`demo_pushup.mp4`, `demo_squat.mp4`, `demo_bicep_curl.mp4`, `demo_jumping_jack.mp4` (3 reps each). Regenerate them with:

```bash
python scripts/make_demo_video.py --exercise pushup --reps 3 --measure --stats-out data/phase_stats_pushup.json
```

### Run the API

```bash
# Start the server
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Test with curl

```bash
# Analyze a push-up video (demo video included in the repo)
curl -X POST "http://localhost:8000/v1/analyze" \
  -H "accept: application/json" \
  -F "video=@demo_pushup.mp4"

# Pick a different exercise with the `exercise` form field
curl -X POST "http://localhost:8000/v1/analyze" \
  -F "video=@demo_squat.mp4" -F "exercise=squat"

# Check health
curl "http://localhost:8000/healthz"

# Get model info (per-exercise)
curl "http://localhost:8000/v1/model?exercise=squat"
```

### Test real-time streaming

```bash
# Stream a video file through the WebSocket (replays at native rate)
python scripts/live_webcam.py --exercise pushup --file demo_pushup.mp4

# Or use a live webcam (webcam 0)
python scripts/live_webcam.py --exercise squat
```

The client prints per-frame phase + confidence + running rep count, then a final summary with coaching feedback. Requires the API to be running.

### Run Tests

```bash
# All tests (no network calls - mocks used)
pytest tests/ -v

# Specific component tests
pytest tests/test_features.py -v
pytest tests/test_lstm.py -v
pytest tests/test_reps.py -v
pytest tests/test_coach.py -v
pytest tests/test_serving.py -v
```

---

## Docker

### Build

```bash
docker build -t pushup-analysis .
```

### Run

```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-api-key" \
  pushup-analysis
```

The container includes a health check that verifies `/healthz` returns ready=true.

---

## API Reference

### POST /v1/analyze

Analyze an exercise video (batch path).

**Request**: Multipart form with `video` file (max 50MB, mp4/mov/avi/mkv) and optional `exercise` field (`pushup` default, or `squat`, `bicep_curl`, `jumping_jack`).

**Response**:
```json
{
  "rep_count": 10,
  "reps": [
    {"start_s": 1.2, "end_s": 3.5},
    {"start_s": 4.1, "end_s": 6.3}
  ],
  "per_class_confidence": {
    "idle": 0.92,
    "concentric": 0.87,
    "eccentric": 0.89
  },
  "coaching_feedback": {
    "summary": "Great work completing 10 push-ups with consistent tempo!",
    "strengths": ["Excellent rep consistency", "Well-controlled eccentric phase"],
    "improvements": ["Try slowing the concentric phase slightly"],
    "safety_notes": []
  },
  "model_version": "1.0.0",
  "latency_ms": 1250,
  "stage_timings": {
    "decode_ms": 120,
    "features_ms": 450,
    "lstm_ms": 80,
    "llm_ms": 600
  }
}
```

### WS /v1/stream (real-time)

Stream live frames and receive incremental rep counting.

1. Open `ws://localhost:8000/v1/stream`.
2. Send a config frame: `{"type": "config", "exercise": "pushup", "source_fps": 30.0}`.
3. Server replies `{"type": "ready", ...}`.
4. Send raw **JPEG** frames as binary messages (one per camera frame).
5. Server replies per sampled frame with:
   ```json
   {"type": "frame", "t_s": 2.13, "phase": "concentric", "phase_id": 1,
    "confidence": 0.93, "rep_count": 4}
   ```
6. Send `{"type": "summary"}` any time for a cumulative result, or `{"type": "reset"}` to restart.
7. On disconnect the server sends the final summary.

See [scripts/live_webcam.py](scripts/live_webcam.py) for a ready-made client.

### GET /healthz

Liveness and readiness probe.

```json
{
  "status": "ok",
  "ready": true,
  "model_loaded": true,
  "timestamp": "2026-08-16T12:00:00Z"
}
```

- `ready=true` means model weights loaded and warmup complete
- `ready=false` means service is starting up or model failed to load

### GET /v1/model

Model metadata for debugging (optionally per-exercise via `?exercise=`).

```json
{
  "exercise": "pushup",
  "model_version": "1.0.0",
  "feature_config": {
    "target_fps": 15,
    "max_seconds": 30.0,
    "feature_dim": 19,
    "working_resolution": [160, 160],
    "grid_size": [4, 4]
  },
  "checkpoint_hash": "a1b2c3d4e5f6...",
  "architecture": {
    "type": "LSTM (causal)",
    "input_dim": 19,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "num_classes": 3,
    "classes": ["idle", "concentric", "eccentric"]
  },
  "rep_counter_config": {
    "min_concentric_frames": 3,
    "min_eccentric_frames": 3,
    "min_idle_frames": 5
  },
  "available_exercises": ["pushup", "squat", "bicep_curl", "jumping_jack"]
}
```

---

## Configuration

All settings via environment variables (see `src/serving/settings.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_PATH` | `checkpoints/lstm_pushup.pt` | Default checkpoint path |
| `LSTM_BIDIRECTIONAL` | `false` | Use bidirectional LSTM (offline only; streaming requires causal) |
| `TARGET_FPS` | `15` | Video resampling rate |
| `MAX_SECONDS` | `30.0` | Max video duration to process |
| `MIN_CONCENTRIC_FRAMES` | `3` | Rep-counter debounce: concentric |
| `MIN_ECCENTRIC_FRAMES` | `3` | Rep-counter debounce: eccentric |
| `MIN_IDLE_FRAMES` | `5` | Rep-counter debounce: idle |
| `MAX_UPLOAD_MB` | `50` | Max upload size |
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | LLM model (fallback alias `OPENAI_MODEL` also accepted) |
| `OPENAI_TIMEOUT_S` | `10.0` | LLM request timeout |
| `OPENAI_MAX_TOKENS` | `300` | LLM response token cap |
| `OPENAI_TEMPERATURE` | `0.3` | LLM sampling temperature |
| `LOG_LEVEL` | `INFO` | Logging level |

Create a `.env` file for local development (never commit it — `.env` is gitignored):
```bash
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4.1-mini
LOG_LEVEL=INFO
```

---

## Architecture

### Component A: Video Preprocessing (`src/video/features.py`)

- Resamples video to fixed 15 FPS (doesn't trust CAP_PROP_FPS)
- Letterbox resize to 160×160 preserving aspect ratio
- Handles rotation metadata
- **Features (19D) — all causal** (never look ahead, so batch and streaming are identical):
  - 16: Frame-difference energy on 4×4 spatial grid
  - 1: Vertical centroid of motion (normalized 0-1)
  - 1: Optical flow mean magnitude (normalized)
  - 1: Optical flow mean vertical component (normalized)
- **Two consumption modes sharing one computation**:
  - `extract_features(path)` — batch: whole file → (T, 19)
  - `StreamingFeatureExtractor` — live: frame-by-frame → per-frame vectors
- Normalized for lighting/zoom invariance
- Graceful handling of corrupt/short/rotated videos

### Component B: LSTM Model (`src/model/lstm.py`, `src/model/reps.py`)

- **Unidirectional LSTM (causal)**: 2 layers, 64 hidden units, dropout=0.2 — causal so the same weights serve batch *and* the real-time `/v1/stream` path (`step()` feeds one frame at a time)
- **Per-timestep 3-class head**: idle (0), concentric (1), eccentric (2)
- **Packed sequences** for variable-length input
- **Rep counting**: State machine with minimum-duration debounce
  - IDLE → CONCENTRIC: ≥3 frames (0.2s)
  - CONCENTRIC → ECCENTRIC: ≥3 frames
  - ECCENTRIC → IDLE: ≥5 frames (0.33s)
- **Per-exercise weights**: one checkpoint per exercise, selected at request time
- **Domain mixing**: pushup/squat training sequences are randomly drawn from both MM-Fit real-human calibration and synthetic-render calibration, so the model learns both distributions and transfers better to a real webcam
- **Evaluation**: Per-class F1 + Rep-count MAE (split by video)

### Component C: Coaching (`src/llm/coach.py`, `prompts/coach_v1.md`)

- **Structured output** via OpenAI JSON schema + Pydantic validation
- **RepStats schema**: All metrics computed in Python, never invented by LLM
- **Fallback**: Template-based deterministic fallback on any failure
- **Retry logic**: One repair retry on validation failure
- **Rate limiting**: Exponential backoff with tenacity
- **Tests**: Fully mocked, no network calls

### Component D: Serving (`src/serving/`)

- **FastAPI** with thread-safe singleton model registry
- **Per-exercise models loaded once** at startup with a warmup pass
- **Readiness** reflects actual model state (`/healthz` → 503 until ready)
- **Blocking inference** off event loop via `run_in_threadpool`
- **`/v1/stream` WebSocket** for real-time analysis (`src/serving/stream.py` — `StreamSession` owns one extractor + one counter per connection)
- **Structured logging** with request ID and per-stage timings
- **Temp file cleanup** on success and failure paths
- **Dockerfile** with health check

---

## Project Structure

```
AerioneBharat/
├── src/
│   ├── video/features.py      # Component A: Video preprocessing (batch + streaming)
│   ├── model/lstm.py          # Component B: causal LSTM model
│   ├── model/reps.py          # Component B: rep-counting state machine
│   ├── llm/coach.py           # Component C: LLM coaching + deterministic fallback
│   └── serving/
│       ├── app.py             # FastAPI app: /v1/analyze, /v1/stream, /healthz, /v1/model
│       ├── registry.py        # Per-exercise model registry & inference
│       ├── stream.py          # StreamSession (real-time extraction + counting)
│       ├── schemas.py         # Pydantic request/response models
│       └── settings.py        # Configuration (incl. exercise list)
├── prompts/coach_v1.md        # Versioned, exercise-aware prompt template
├── scripts/
│   ├── make_demo_video.py     # Renders synthetic exercise clips + measures stats
│   ├── mmfit_pose_to_video.py # MM-Fit real-human calibration (pushup, squat)
│   └── live_webcam.py         # WebSocket streaming client (webcam or file replay)
├── tests/                     # Unit tests (all mocked, no network)
├── data/                      # Datasets + phase_stats*.json (gitignored)
├── checkpoints/               # Per-exercise trained models (gitignored)
├── generate_synthetic_data.py # Statistics-calibrated synthetic data generation
├── train.py                   # Per-exercise training script (--all)
├── demo_pushup.mp4            # Ready-made demo videos (3 reps each, 4 exercises)
├── demo_squat.mp4
├── demo_bicep_curl.mp4
├── demo_jumping_jack.mp4
├── mm-fit/                    # Real-human dataset (local; NOT part of the repo)
├── exercises-dataset/         # Exercise reference dataset (local; NOT part of the repo)
├── Dockerfile
├── requirements.txt
├── README.md
└── NOTES.md
```

---

## Development

### Code Style

```bash
# Format
black src/ tests/

# Lint
ruff src/ tests/

# Type check
mypy src/
```

### Adding a New Prompt Version

1. Create `prompts/coach_v2.md` following the same format
2. Update `summarize()` call to use `prompt_version="v2"`
3. Add a mock-client test in `tests/test_coach.py` (no network)

---

## License

Assessment submission for AerioneBharat AI Engineer position.