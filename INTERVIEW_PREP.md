# Interview Prep: Exercise Analysis Pipeline

This document explains every part of the codebase so you can discuss it confidently in an interview. It covers each component in detail, the engineering decisions made, the difficulties encountered, and how each problem was solved.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Component A: Video Preprocessing (OpenCV)](#2-component-a-video-preprocessing-opencv)
3. [Component B: LSTM Sequence Model + Rep Counting](#3-component-b-lstm-sequence-model--rep-counting)
4. [Component C: LLM Coaching Summary](#4-component-c-llm-coaching-summary)
5. [Component D: Serving Layer + Model Utilities](#5-component-d-serving-layer--model-utilities)
6. [Data Strategy: How We Trained Without Real Labels](#6-data-strategy-how-we-trained-without-real-labels)
7. [Difficulties Faced and How We Solved Them](#7-difficulties-faced-and-how-we-solved-them)
8. [What We Would Improve for Production](#8-what-we-would-improve-for-production)
9. [Quick Reference: Key Files and Their Roles](#9-quick-reference-key-files-and-their-roles)

---

## 1. Project Overview

### What We Built

A complete ML pipeline that takes a video of someone exercising and returns structured results:

- **Repetition count** with start/end timestamps for each rep
- **Coaching feedback** in plain language (from an LLM)
- **Debug metadata**: model version, per-stage timings, confidence scores

The system supports **four exercises**: pushup, squat, bicep_curl, and jumping_jack.

### The Four-Layer Architecture

```
Video File / Live Camera
        |
        v
  [Component A] Feature Extraction (OpenCV)
  19-dimensional feature vector per frame, causal (no look-ahead)
        |
        v
  [Component B] LSTM Phase Classifier + Rep Counter
  Per-frame prediction: idle / concentric / eccentric
  State machine counts reps with timestamps
        |
        v
  [Component C] LLM Coaching (OpenAI + deterministic fallback)
  Computed statistics -> structured coaching feedback
        |
        v
  [Component D] FastAPI Serving
  POST /v1/analyze (batch) + WebSocket /v1/stream (real-time)
```

### Why Causal Features Matter

Every feature at time `t` is computed from the **current frame and the previous frame only** — no averaging over the whole video, no look-ahead. This is critical because:

1. **Batch and streaming share the exact same computation**. When we process a video file, we use `extract_features()`. When we process a live webcam frame-by-frame, we use `StreamingFeatureExtractor.process_frame()`. Both call the same underlying `compute_frame_features()` function. The feature distribution at serving time is identical to the training distribution.

2. **The LSTM can be unidirectional**. A bidirectional LSTM looks at future frames, which makes real-time streaming impossible (you'd need the whole video before you can classify anything). A unidirectional/causal LSTM only sees past frames, so it works identically for batch and streaming.

---

## 2. Component A: Video Preprocessing (OpenCV)

### File: `src/video/features.py`

### The 19-Dimensional Feature Vector

Each frame pair produces one 19-dimensional feature vector. Here is exactly what each dimension means:

| Indices | Feature | What It Captures |
|---------|---------|------------------|
| 0-15 | Frame-difference energy on a 4×4 spatial grid | *Where* in the frame motion is happening. Divides the frame into 16 cells and measures how much pixel change occurred in each cell. |
| 16 | Vertical centroid of motion (normalized 0-1) | The average Y-position of all motion. When someone does a pushup, the centroid moves down (toward 1.0) as they lower and up (toward 0.0) as they push up. |
| 17 | Optical flow mean magnitude (normalized) | How much overall motion there is, regardless of direction. Uses Farneback optical flow. |
| 18 | Optical flow mean vertical component (normalized) | Whether motion is upward or downward. Positive values mean downward motion, negative means upward. This is the key signal that distinguishes concentric (pushing up) from eccentric (lowering down). |

### Why These Features

We chose features that are:
- **Invariant to lighting/zoom**: Frame-difference energy is normalized by cell area and clipped to [0,1]. Optical flow is normalized by a max-displacement constant. A darker, brighter, or more zoomed-in recording of the same movement produces similar feature values.
- **Interpretable**: The vertical centroid and vertical flow component directly encode the up/down motion that defines exercises like pushups and squats.
- **Causal**: Every feature is computed from the current frame and the previous frame only. No future information leaks in.

### The Preprocessing Pipeline

```
Raw Video File
     |
     v
OpenCV VideoCapture (reads frames)
     |
     v
Rotation metadata check (CAP_PROP_ORIENTATION_META)
     |
     v
Letterbox resize to 160×160 (preserves aspect ratio, adds black padding)
     |
     v
Grayscale conversion
     |
     v
Uniform resampling to target_fps (e.g., 15 FPS)
     |
     v
For each consecutive frame pair -> compute_frame_features() -> (19,) vector
```

### Key Design Decisions

**1. Letterbox resize, not stretch:**
We resize to 160×160 while preserving aspect ratio. If the input is 640×480 (4:3), it becomes 160×120 centered on a 160×160 canvas with black bars top and bottom. Stretching would distort the motion features.

**2. Don't trust CAP_PROP_FPS:**
Phone videos often have incorrect or zero FPS in the container metadata. We estimate FPS from `frame_count / duration` instead, falling back to 30.0 if that's also invalid.

**3. Handle rotation metadata:**
Phone videos recorded in portrait mode often have a rotation tag (90°, 180°, 270°). We read `CAP_PROP_ORIENTATION_META` and apply the corresponding rotation before processing. This is essential for correct feature extraction from phone recordings.

**4. Two consumption modes, one computation:**
- `extract_features(path)` — batch: reads a whole video file, returns `(T, 19)` array
- `StreamingFeatureExtractor` — live: processes one frame at a time, returns `(19,)` or None (if the frame is not a sampling point)

The streaming extractor uses **index-based sampling**: it samples frame `round(k * source_fps / target_fps)` for k = 0, 1, 2, ... This exactly reproduces the `np.linspace` sampling the batch path uses. A unit test verifies both modes produce numerically identical vectors on the same input frames.

### Graceful Error Handling

The function handles:
- **Corrupt/unreadable files**: raises `VideoProcessingError` with a descriptive message
- **Zero-length videos**: detected by frame count or dimension checks
- **Videos shorter than one frame at target_fps**: still processes at least 1 frame (duplicates it)
- **Rotation metadata**: applied transparently

---

## 3. Component B: LSTM Sequence Model + Rep Counting

### Files: `src/model/lstm.py` and `src/model/reps.py`

### The LSTM Model

**Architecture:**
- **Unidirectional (causal) LSTM**: 2 layers, 64 hidden units, dropout 0.2
- **3-class classification head**: idle (0), concentric (1), eccentric (2)
- **Input**: `(batch, seq_len, 19)` feature sequences
- **Output**: `(batch, seq_len, 3)` logits

**Why unidirectional?**
A bidirectional LSTM reads the entire sequence, looking at both past and future frames. This is fine for batch analysis of a completed video, but impossible for real-time streaming — you can't classify the current frame until you've seen the entire video. By using a unidirectional LSTM, the same weights serve both the batch path (`forward()`) and the streaming path (`step()`).

**Variable-length sequences:**
The model uses `pack_padded_sequence` and `pad_packed_sequence` to handle variable-length sequences efficiently. Padded timesteps get the ignore index (-1) in the target tensor, and the loss function (`CrossEntropyLoss(ignore_index=-1)`) excludes them from both the loss calculation and accuracy metrics.

**The `step()` method:**
This is the streaming interface. It takes exactly one timestep `(1, 1, 19)` and the previous hidden state `(h, c)`, runs one LSTM step, and returns the new logits and hidden state. Because the LSTM is unidirectional, `step()` is numerically identical to running `forward()` on a single frame — the same checkpoint serves both paths.

### Rep Counting State Machine

**File: `src/model/reps.py`**

The rep counter is a **state machine with minimum-duration debounce**. It lives outside the model because it operates on predicted phase sequences, not raw features.

**The state diagram:**

```
IDLE ──(≥3 concentric frames)──▶ CONCENTRIC ──(≥3 eccentric frames)──▶ ECCENTRIC ──(≥5 idle frames)──▶ IDLE
   ▲                              │                                       │
   │                              │                                       │
   └──(≥5 idle frames)◄───────────┘◄──(any concentric, abort)─────────────┘
```

**How it works:**

1. The state machine receives a sequence of predicted phases (one per frame) with optional confidence scores.
2. It only transitions to a new state when that phase has been predicted for a **minimum number of consecutive frames** (the debounce). This absorbs single-frame noise.
3. A rep is counted as a complete **CONCENTRIC → ECCENTRIC** cycle, closed by an idle run.
4. Consecutive reps with no real pause are absorbed into one continuous set (the rep is only closed by a validated idle run).

**Why a state machine instead of simple counting?**
Raw model predictions are noisy — a single frame might be misclassified as eccentric when it's actually concentric. The debounce thresholds (3/3/5 frames at 15 FPS = 0.2s/0.2s/0.33s) ensure we only count a phase transition when it's been consistently predicted long enough to be real.

**The confidence gate:**
When a confidence score is provided, low-confidence non-idle predictions are absorbed into the current state. This prevents the model from triggering phantom phase transitions on uncertain predictions (e.g., during real webcam idle). Idle predictions are always trusted because ending a rep early is less harmful than never completing one.

### Training

**File: `train.py`**

- **Optimizer**: Adam, learning rate 0.001
- **Loss**: CrossEntropyLoss with label smoothing 0.1 and ignore_index=-1 for padded timesteps
- **Epochs**: 20 (configurable)
- **Batch size**: 16
- **Device**: CPU only
- **Split**: by video/sequence, never by frame

The training loop evaluates per-class F1 and rep-count MAE on the validation set. The best checkpoint (by validation rep-count MAE) is saved.

### Evaluation Metrics

Reported on a held-out test split (100 sequences per exercise, never seen during training):

| Exercise | Macro F1 | Idle F1 | Concentric F1 | Eccentric F1 | Rep MAE |
|----------|----------|---------|---------------|--------------|---------|
| pushup | 0.814 | 0.858 | 0.768 | 0.815 | 0.12 |
| squat | 0.775 | 0.813 | 0.739 | 0.772 | 0.16 |
| bicep_curl | 0.768 | 0.809 | 0.744 | 0.751 | 0.20 |
| jumping_jack | 0.849 | 0.891 | 0.799 | 0.857 | 0.20 |

These are **internal-consistency** numbers (synthetic features tested on held-out synthetic features), not real-world accuracy claims. They prove the model learns the calibrated per-phase distribution.

---

## 4. Component C: LLM Coaching Summary

### Files: `src/llm/coach.py` and `prompts/coach_v1.md`

### The RepStats Schema

All metrics are **computed in Python** before the LLM is called. The LLM rephrases and prioritizes — it never invents or recomputes a number.

```python
class RepStats(BaseModel):
    rep_count: int          # Total reps detected
    reps: list[RepTiming]   # Start/end timestamps per rep
    avg_tempo_s: float      # Average rep duration
    tempo_consistency: float # Coefficient of variation (std/mean) of rep durations
    concentric_avg_s: float # Average time in concentric phase
    eccentric_avg_s: float  # Average time in eccentric phase
    confidence: float       # Model confidence (0-1)
```

### The Coaching Pipeline

```
RepCountResult (from state machine)
        |
        v
Compute RepStats in Python (tempo, CV, phase durations, confidence)
        |
        v
Render prompt template (prompts/coach_v1.md) with computed stats
        |
        v
Call OpenAI with structured output (JSON schema)
        |
        v
Validate response with Pydantic (CoachFeedback model)
        |
   Success? ──Yes──▶ Return validated feedback
        │
       No
        │
        v
One repair retry (send back the invalid response + correction request)
        │
   Success? ──Yes──▶ Return validated feedback
        │
       No
        │
        v
Deterministic template fallback (_template_fallback)
```

### Structured Output with Validation

We use OpenAI's `response_format=json_schema` with strict mode. The schema is generated from the Pydantic `CoachFeedback` model, but OpenAI strict mode requires `additionalProperties: false` and a complete `required` list on every object — Pydantic's default schema doesn't include these. So we have a `_strict_json_schema()` function that walks the schema tree and adds them.

**The CoachFeedback model:**
```python
class CoachFeedback(BaseModel):
    summary: str           # 10-300 characters
    strengths: list[str]   # 1-3 items
    improvements: list[str] # 1-3 items
    safety_notes: list[str] # 0-2 items
```

### Retry Logic

1. **First attempt**: Call the LLM with the rendered prompt and JSON schema
2. **Validation failure**: One repair retry — send back the invalid response with an explicit "output ONLY valid JSON" instruction
3. **Final fallback**: Deterministic rule-based template that generates coaching from the same RepStats

The endpoint **never fails because the LLM misbehaved**. Timeouts, rate limits, missing API keys, and schema validation errors all lead to the template fallback.

### Rate Limiting and Resilience

- **Exponential backoff**: tenacity `wait_exponential(multiplier=1, min=1, max=10)` with `stop_after_attempt(3)`
- **Retryable errors**: `RateLimitError`, `APITimeoutError`, `APIConnectionError`
- **Timeout**: 10 seconds default (configurable)
- **Max tokens**: 300 (keeps responses short and costs low)
- **Missing API key**: Immediately returns template fallback

### The Prompt Template

`prompts/coach_v1.md` is a Jinja2 template that receives the computed RepStats and renders a coaching prompt. The prompt instructs the LLM to:
- Write a 1-2 sentence summary of the session
- Identify 1-3 strengths based on the metrics
- Suggest 1-3 specific improvements
- Add safety notes for risky patterns (fast reps, missing eccentric phase)
- Never invent numbers — all metrics come from the input

### Tests: No Network Calls

All coaching tests use a mock client. The test suite asserts against the deterministic template fallback, so it runs offline and never makes real API calls.

---

## 5. Component D: Serving Layer + Model Utilities

### Files: `src/serving/app.py`, `src/serving/registry.py`, `src/serving/stream.py`, `src/serving/schemas.py`, `src/serving/settings.py`

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/analyze` | POST | Upload video, get full analysis (batch) |
| `/v1/stream` | WebSocket | Real-time frame streaming with incremental results |
| `/v1/live/start` | POST | Start an HTTP live-camera session |
| `/v1/live/{id}/frame` | POST | Send one JPEG frame to a live session |
| `/v1/live/{id}/stop` | POST | Stop live session, get full results |
| `/healthz` | GET | Liveness + readiness (503 until model loaded) |
| `/v1/model` | GET | Model metadata (version, architecture, checkpoint hash) |

### Model Registry (`src/serving/registry.py`)

The `ModelRegistry` is a **thread-safe singleton** that manages all exercise models:

1. **Loads once at startup**: All four checkpoints are loaded into memory when the service starts.
2. **Warmup pass**: After loading each model, a dummy forward pass runs on random data to ensure the weights are fully initialized and compiled.
3. **Thread-safe access**: A lock protects the loading path; the singleton pattern ensures only one instance exists.
4. **Per-exercise models**: Each exercise has its own checkpoint and its own `ExerciseModel` entry in the registry.

**Key methods:**
- `extract_features(path)` → `(features, decode_time_ms)` — wraps Component A
- `predict(features, exercise_id)` → `(logits, inference_time_ms)` — batch inference
- `step(feature, exercise_id, hidden)` → `(logits, hidden, inference_ms)` — single-frame streaming inference
- `count_reps(logits)` → `RepCountResult` — wraps Component B's state machine
- `compute_stats(rep_result)` → `RepStats` — computes coaching statistics
- `get_coaching(stats, exercise_id)` → `(CoachFeedback, llm_time_ms)` — wraps Component C

### Configuration (`src/serving/settings.py`)

All configuration is driven by **environment variables** via Pydantic Settings. No hardcoded paths.

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHECKPOINT_PATH` | `checkpoints/lstm_pushup.pt` | Default checkpoint |
| `TARGET_FPS` | `15` | Video resampling rate |
| `MAX_SECONDS` | `30.0` | Max video duration |
| `MIN_CONCENTRIC_FRAMES` | `3` | Rep-counter debounce |
| `MIN_ECCENTRIC_FRAMES` | `3` | Rep-counter debounce |
| `MIN_IDLE_FRAMES` | `5` | Rep-counter debounce |
| `MAX_UPLOAD_MB` | `50` | Upload size limit |
| `OPENAI_API_KEY` | (required) | LLM API key |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | LLM model |
| `OPENAI_TIMEOUT_S` | `10.0` | LLM timeout |
| `LOG_LEVEL` | `INFO` | Logging level |

A `.env` file can be used for local development (gitignored).

### Blocking Work Off the Event Loop

All CPU-intensive operations (video decode, feature extraction, LSTM inference, LLM calls) run in a **thread pool** via FastAPI's `run_in_threadpool`. The async event loop never blocks.

### Structured Logging

Every request gets a unique **request ID** (UUID). Logs include this ID so you can trace a single request through all stages (decode, features, LSTM, LLM). Per-stage timings are recorded in milliseconds.

### Upload Guardrails

- **Content-type validation**: Only accepts video MIME types (mp4, mov, avi, mkv)
- **Size limit**: 50MB maximum (returns 413 if exceeded)
- **Temp file cleanup**: Uploaded files are written to a temp file that is deleted on both the success and failure paths

### Health Check

`GET /healthz` returns readiness that genuinely reflects whether models are loaded:
```json
{
  "status": "ok",
  "ready": true,
  "model_loaded": true,
  "timestamp": "2026-08-16T12:00:00Z"
}
```
`ready=false` until all models are loaded and the warmup pass completes. This is what Docker's health check and a load balancer would use.

### Real-Time Streaming (`src/serving/stream.py`)

The `StreamSession` class manages one live streaming session:

1. **Client sends config**: `{"type": "config", "exercise": "pushup", "source_fps": 30.0}`
2. **Server replies ready**: Client can now send frames
3. **Per frame**: Server decodes JPEG → extracts feature → runs LSTM step → updates rep counter → sends per-frame status
4. **Summary on demand**: Client sends `{"type": "summary"}` for cumulative results
5. **Disconnect**: Server sends final summary automatically

Each session is isolated: its own `StreamingFeatureExtractor`, `RepCounter`, and LSTM hidden state. Multiple clients can stream concurrently without interfering.

**Auto-detect mode** (`exercise=auto`):
- Accumulates features in a buffer
- Every ~1.5 seconds, runs all 4 models on the buffer
- Switches to the exercise with the highest non-idle confidence (threshold: 0.70, margin: 0.08 over runner-up, 5-second cooldown)
- Runs in a background daemon thread so the frame endpoint never blocks

---

## 6. Data Strategy: How We Trained Without Real Labels

### The Problem

No labeled exercise video dataset was available. We needed training data for a sequence model that classifies each frame as idle/concentric/eccentric.

### Our Solution: Calibrated Synthetic Data

**Step 1: Calibration**

We first measure the actual feature distribution of each exercise by rendering a known sequence and running the real feature extractor over it:

1. `scripts/make_demo_video.py` renders a synthetic exercise clip (e.g., a stick-figure doing 3 pushups)
2. The real `extract_features()` function processes this clip, producing actual 19-D feature vectors
3. We compute per-phase mean and standard deviation from these real features → `data/phase_stats_<exercise>.json`

**Step 2: Real-human calibration (MM-Fit) — pushup and squat only**

For pushup and squat, we used the **MM-Fit dataset** (21 subjects, real 3D motion capture poses):
1. `scripts/mmfit_pose_to_video.py` renders the real 3D poses back to video
2. The real feature extractor measures per-phase statistics from these real-human videos → `data/phase_stats_mmfit_<exercise>.json`
3. These MM-Fit statistics are **strictly preferred** in training data generation

This is the single biggest transfer improvement we could make. The synthetic render's idle phase is near-zero motion (perfect), but real humans are never perfectly still. By mixing both distributions, the model learns to handle both.

**Step 3: Training sequence generation**

`generate_synthetic_data.py` generates training sequences by:
1. Sampling per-phase feature vectors from a Gaussian with the calibrated mean and widened standard deviation (1.8×) — this creates natural variation
2. Adding a small per-sequence global shift — each "subject" is slightly different
3. Concatenating idle → concentric → eccentric → idle phases with randomized durations
4. Generating 600 train / 100 val / 100 test sequences per exercise (1-8 reps each)

### Why This Approach

- **The features are real**: Even though the labels are synthetic, the feature vectors come from the actual OpenCV pipeline (frame-difference grid, optical flow, centroid). The model learns the same distribution it sees at serving time.
- **Calibration bridges the sim-to-real gap**: By measuring features from real-human poses (MM-Fit) and mixing them with synthetic renders, the training distribution is closer to what a real webcam would produce.
- **Fast to generate**: No manual annotation needed. Generating 600+ sequences takes seconds.

### What We Rejected

The `exercises-dataset/` directory was evaluated and **rejected**. It contains single-rep looping GIFs without rep/phase annotations. Uselable for supervised training of a phase-sequence model. Documented rather than silently ignored.

---

## 7. Difficulties Faced and How We Solved Them

### Difficulty 1: Zero Reps on Demo Videos

**Problem**: The rep counter returned 0 reps even on demo videos with clearly visible exercise motion.

**Root causes (3 issues found):**

1. **Motion veto was too aggressive**: The `StreamSession` had a `_motion_energy_threshold` of 0.015 that forced the state machine into idle on 96% of frames. Demo videos and real webcam footage have natural motion even during "idle" moments (breathing, slight camera shake). This veto was killing all non-idle predictions.

   **Fix**: Removed the motion veto entirely. The confidence gate in the state machine is sufficient to suppress noise.

2. **Confidence gate was blocking idle predictions**: The confidence threshold (0.50) was applied to ALL predictions, including idle. When the model predicted idle with near-zero confidence (a near-tie between idle and another phase), the gate would force it back to the current state, preventing the idle run needed to complete a rep.

   **Fix**: Changed the confidence gate to only apply to **non-idle** predictions. Idle predictions are always trusted because ending a rep early is less harmful than never completing one.

3. **Stale cached bytecode**: After fixing the code, the running server still used old `.pyc` files. The `phase_id` field was added to the `LiveSessionResponse` model but the server process loaded the cached version without it.

   **Fix**: Clear all `__pycache__` directories and `.pyc` files before restarting uvicorn. Added `--reload` flag to uvicorn so it watches for file changes and reloads automatically.

### Difficulty 2: Auto-Detect Flapping

**Problem**: When `exercise=auto`, the system kept switching between exercises every ~2 seconds (pushup → squat → bicep_curl → pushup...), resetting the rep counter each time.

**Root cause**: The auto-detect logic ran every `_auto_window` frames (~1.5 seconds) and switched to the highest-confidence exercise without any stability checks. On ambiguous input, different exercises would take turns winning.

**Fix**: Added three stabilizing mechanisms:
- **Confidence threshold (0.70)**: Only switch if the best exercise's confidence is genuinely high
- **Margin requirement (0.08)**: The best exercise must beat the runner-up by at least 0.08. This prevents switching between two similar exercises.
- **Cooldown (5 seconds)**: After switching, don't re-evaluate for 5 seconds. This gives the rep counter time to accumulate a full rep cycle without interruption.

### Difficulty 3: 500 Errors on Frame Endpoint

**Problem**: The `/v1/live/{id}/frame` endpoint returned 500 errors with `ValidationError: phase_id Field required`.

**Root cause**: The `LiveSessionResponse` Pydantic model had `phase_id: int` as a field, but the endpoint code never passed `phase_id` when constructing the response. The model definition was added in one place but the three constructor calls at lines 515, 524, and 531 were never updated.

**Fix**: Added `phase_id=0` to the two idle fallback paths and `phase_id=result["phase_id"]` to the success path. Also removed a duplicate `rep_count: int` line in the model definition.

### Difficulty 4: Client Blocking on Exit

**Problem**: When pressing Ctrl+C or 'q' in the camera window, the client script hung and had to be killed via Task Manager.

**Root cause**: The original client used `concurrent.futures.wait(pending, timeout=10)` in the cleanup path, which blocked indefinitely if any upload thread was still running.

**Fix**: 
- Restructured the client into a `LiveCameraClient` class with explicit lifecycle management
- Used `pool.shutdown(wait=False, cancel_futures=True)` instead of waiting
- Added signal handlers for SIGINT and SIGTERM that set a shutdown event
- The camera loop checks the shutdown event and exits immediately
- All pending uploads are cancelled on shutdown

### Difficulty 5: Live Camera Lag

**Problem**: The camera feed was laggy because each frame upload blocked on the server round-trip before the next frame was captured.

**Fix**: Rewrote the client to use a `ThreadPoolExecutor` with 4 worker threads. Frames are submitted to the pool as fire-and-forget, and completed results are drained non-blocking. The camera loop never waits for the server.

### Difficulty 6: Content-Type Validation Blocking Valid Uploads

**Problem**: Some clients sent files with `application/octet-stream` content type even when the file was a valid video. The endpoint rejected these with 415.

**Fix**: Added extension-based fallback validation. If the content type is `application/octet-stream`, we check the file extension against the allowed list (`.mp4`, `.mov`, `.avi`, `.mkv`). If the extension matches, we accept the file regardless of content type.

---

## 8. What We Would Improve for Production

### Data Quality
- Collect 50-100 real labeled videos per exercise. This is the single biggest improvement.
- Add MediaPipe pose keypoints as additional features (better kinematic signal than frame-difference blobs).
- Add temporal smoothing (CRF/HMM) over per-frame phase predictions.

### Model Improvements
- Fine-tune the synthetic models on real data (few-shot domain adaptation).
- Experiment with a 1D CNN over the feature sequence before the LSTM (better local pattern detection).
- Add data augmentation (time warping, feature noise) during training.

### Serving Improvements
- Add authentication/API keys.
- Add a task queue (Celery/Redis) for async LLM calls — coaching is currently the slowest stage (~0.6s).
- Add request-level timeouts and circuit breakers for the LLM.
- Horizontal scaling with a shared model cache (models loaded once per worker, not per process).
- Add request queuing/batching for the LSTM inference.
- Add Prometheus metrics for latency percentiles, error rates, and model confidence distributions.

### Monitoring
- Log per-class confidence distributions to detect model drift.
- Alert when rep-count confidence drops below a threshold.
- Track LLM fallback rate (high fallback rate = LLM issues or bad input).

---

## 9. Quick Reference: Key Files and Their Roles

| File | Role | Key Classes/Functions |
|------|------|----------------------|
| `src/video/features.py` | Component A: Feature extraction | `extract_features()`, `StreamingFeatureExtractor`, `compute_frame_features()` |
| `src/model/lstm.py` | Component B: LSTM model | `PushupLSTM`, `PushupLSTMLoss`, `create_model()`, `load_checkpoint()`, `step()` |
| `src/model/reps.py` | Component B: Rep counter | `RepCounter`, `count_reps_from_logits()`, `evaluate_rep_counting()` |
| `src/llm/coach.py` | Component C: LLM coaching | `summarize()`, `RepStats`, `CoachFeedback`, `LLMClient`, `_template_fallback()` |
| `src/serving/app.py` | Component D: FastAPI app | `live_start()`, `live_frame()`, `live_stop()`, `/v1/analyze`, `/healthz` |
| `src/serving/registry.py` | Component D: Model management | `ModelRegistry`, `get_registry()`, `ExerciseModel` |
| `src/serving/stream.py` | Component D: Real-time streaming | `StreamSession`, `_maybe_auto_detect()` |
| `src/serving/schemas.py` | Component D: API schemas | `AnalyzeResponse`, `LiveSessionResponse`, `HealthResponse` |
| `src/serving/settings.py` | Component D: Configuration | `Settings`, `ExerciseSpec`, `DEFAULT_EXERCISES` |
| `train.py` | Training pipeline | `train_model()`, `evaluate_trained_model()`, `CollateWithLengths` |
| `generate_synthetic_data.py` | Data generation | `generate_synthetic_data()`, `load_dataset()` |
| `scripts/make_demo_video.py` | Demo video rendering | Renders synthetic exercise clips + measures real features |
| `scripts/mmfit_pose_to_video.py` | MM-Fit calibration | Renders real human poses to video for pushup/squat |
| `prompts/coach_v1.md` | LLM prompt template | Jinja2 template for coaching prompt |
| `Dockerfile` | Container build | Python 3.11-slim, health check, non-root user |
| `tests/test_features.py` | Feature extraction tests | Validates batch/stream feature parity |
| `tests/test_lstm.py` | Model tests | Forward pass, packed sequences, step() |
| `tests/test_reps.py` | Rep counter tests | State machine debounce, synthetic sequences |
| `tests/test_coach.py` | Coaching tests | Mock LLM, template fallback, validation |
| `tests/test_serving.py` | API tests | End-to-end through TestClient |

---

## How to Explain This in an Interview

### Opening (30 seconds)
"I built a complete ML pipeline for exercise analysis — video in, structured results out. It supports four exercises: pushup, squat, bicep curl, and jumping jack. The pipeline has four layers: OpenCV feature extraction, a causal LSTM for phase classification, an LLM coaching layer with deterministic fallback, and a FastAPI serving layer that supports both batch upload and real-time webcam streaming."

### If They Ask About Features (1 minute)
"I extracted 19 features per frame: 16 frame-difference energy values on a 4×4 spatial grid to capture where motion happens, a vertical centroid of motion to track direction, and optical flow magnitude and vertical component for speed and direction. All features are causal — computed from the current and previous frame only — which means batch processing and real-time streaming use the exact same feature distribution."

### If They Ask About the Model (1 minute)
"I used a unidirectional LSTM with 2 layers of 64 hidden units. Unidirectional is the key design choice — it makes the model causal, so the same weights serve both batch and real-time streaming. The model classifies each frame as idle, concentric, or eccentric. A separate state machine with minimum-duration debounce (3/3/5 frames) converts the phase sequence into rep counts with timestamps. The state machine absorbs noise and only transitions when a phase is consistently predicted."

### If They Ask About Data (1-2 minutes)
"This is where I had to be creative. No labeled exercise video dataset was available, so I used calibrated synthetic data. I rendered exercise clips and ran the real feature extractor over them to measure per-phase feature statistics. For pushup and squat, I went further — I used the MM-Fit dataset of real human 3D motion capture poses, rendered them back to video, and measured the real feature distribution. Training sequences are sampled from Gaussians centered on these calibrated statistics. It's honest about being synthetic, but the feature space is real."

### If They Ask About the LLM (1 minute)
"The coaching layer computes all metrics in Python — rep count, tempo, consistency, phase durations. The LLM only rephrases these numbers into natural language; it never invents metrics. I use OpenAI's structured output mode with JSON schema validation. If the response fails validation, one repair retry. If that fails too, a deterministic template fallback. The endpoint never fails because the LLM misbehaved."

### If They Ask About Serving (1 minute)
"FastAPI with a thread-safe singleton model registry that loads all four checkpoints once at startup with a warmup pass. All blocking work — decode, features, LSTM, LLM — runs in a thread pool. The health endpoint reflects actual model readiness. I support both batch upload via POST and real-time streaming via WebSocket, plus HTTP live sessions. Every response includes per-stage timing breakdowns and a request ID for tracing."

### If They Ask About Challenges (2 minutes)
"The biggest challenge was getting zero reps. Three things were wrong: the motion veto was too aggressive and killed all non-idle predictions, the confidence gate blocked idle transitions needed to complete reps, and stale bytecode hid my fixes. The auto-detect kept flapping between exercises — I fixed that with a confidence threshold, margin requirement, and cooldown. The frame endpoint returned 500s because I added a field to the response model but forgot to pass it in the constructor. And the client hung on exit because it was waiting on network threads that never finished. Each was a different root cause, but they all taught me something about defensive programming."

### If They Ask What You'd Improve (1 minute)
"Real labeled data is the obvious next step. I'd also add MediaPipe pose keypoints as features, temporal smoothing over predictions, and async LLM calls via a task queue. For production: auth, rate limiting, model versioning with A/B testing, and monitoring for model drift."
