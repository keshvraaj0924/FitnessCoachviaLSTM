# Assessment Notes — AerioneBharat Exercise Analysis

This file is the engineering journal behind the submission: *why* each choice was made, what the
honest caveats are, and how every component connects. Read [README.md](README.md) first for the
"how to run" story; this file is the "why it works / where it might not" story.

---

## 1. What We Built (one paragraph)

A vertical-slice fitness backend: **video in, structured result out**, for **four exercises**
(pushup, squat, bicep_curl, jumping_jack), over two consumption paths that share one model:

1. **Batch** — upload a video → `POST /v1/analyze` → reps with timestamps + coaching.
2. **Real-time** — stream live webcam (or replayed) JPEG frames → `WS /v1/stream` → per-frame
   phase + running rep count, then a final coaching summary.

The pipeline is OpenCV feature extraction (Component A) → causal LSTM phase classifier + state
machine rep counter (Component B) → LLM coaching with deterministic fallback (Component C) →
FastAPI serving with a per-exercise model registry (Component D). All four models train on a CPU
in ~10 minutes.

---

## 2. Multi-Exercise Design

The original brief named a single exercise of your choice; mid-project the requirement grew to
**several exercises working on real time**. Two design consequences followed:

- **One feature space, N models.** The 19-D feature vector is exercise-agnostic (it measures
  *where* motion happens and *how fast* it moves — see §3). Each exercise therefore gets its own
  LSTM checkpoint selected per request (`?exercise=` / `exercise` field / WS config frame), so the
  model only has to separate idle / concentric / eccentric for *that* movement.
- **Shared code, per-exercise calibration.** The same `generate_synthetic_data.py` samples
  per-phase feature statistics; only the statistics differ per exercise.

Exercise ids are centralized in `src/serving/settings.py` (`ExerciseSpec` + `DEFAULT_EXERCISES`),
so adding a 5th exercise is: add a spec, generate its stats, train, done.

---

## 3. Component A — Features (batch + streaming, one computation)

### The 19-D vector
| Index | Feature | Meaning |
|-------|---------|---------|
| 0–15 | Frame-difference energy on a 4×4 grid | *Where* motion happens in the frame |
| 16   | Vertical centroid of motion (0=top, 1=bottom) | Direction of the moving mass |
| 17   | Optical-flow mean magnitude | How much motion |
| 18   | Optical-flow mean vertical component | Up (−) vs down (+) motion |

### Preprocessing
- Letterbox resize to 160×160 (aspect preserved, black padding).
- Rotation metadata (`CAP_PROP_ORIENTATION_META`) handled.
- FPS estimated from frame count / duration, not trusted from container metadata.
- Graceful `VideoProcessingError` for corrupt / too-short videos.

### Causal, and shared with streaming (the key design decision)
Every feature at time *t* is computed from **current + previous frame only** — no clip-level
mean-centering, no look-ahead. That is what makes **batch and live identical**:

- `extract_features(path, ...)` → `(T, 19)` for uploads.
- `StreamingFeatureExtractor.process_frame(frame, source_fps)` → one `(19,)` vector per sampled
  frame for the WebSocket path.

The streaming extractor sub-samples by *index* (`round(sampled * source_fps/target_fps)`), which
reproduces exactly the `np.linspace` frames the batch path picks. A unit test
(`tests/test_stream.py::test_matches_batch_features`) asserts the two paths produce numerically
identical vectors on the same source frames. **Same features + same causal weights ⇒ live
predictions see the training distribution.**

---

## 4. Component B — Causal LSTM + Rep Counting

### Model
- **Unidirectional (causal) LSTM**: 2 layers × 64 hidden, dropout 0.2, 3-class head
  (idle / concentric / eccentric).
- `forward` uses `pack_padded_sequence` for variable-length sequences (masked via
  `CrossEntropyLoss(ignore_index=-1)`).
- `step(x, hidden)` runs the same weights **one frame at a time** — this is the streaming path.
  Because the LSTM is unidirectional, `step` and `forward` are numerically the same math, so the
  same checkpoint serves batch and real-time. (The old BiLSTM could not do this — that is why the
  model is causal now.)
- Loss: CrossEntropy + label smoothing 0.1.

### Rep counting (state machine, `src/model/reps.py`)
A rep is **CONCENTRIC → ECCENTRIC**, closed by an idle run:

```
IDLE ──≥3 concentric──▶ CONCENTRIC ──≥3 eccentric──▶ ECCENTRIC ──≥5 idle──▶ IDLE
                                                          │(abort if concentric→idle
                                                          │ without a validated idle)
```

- Debounce thresholds (3 / 3 / 5 frames @ 15 FPS ≈ 0.2 / 0.2 / 0.33 s) absorb single-frame noise.
- Consecutive reps with no real pause are absorbed into one continuous set (counted once per
  validated idle run) — matches how people actually do push-ups.
- `RepCounter` is used **live**: `rep_count` is a readable property while streaming, and
  `get_results()` replays + resets for the batch path.

### Training data — the honest story
There is no real labelled exercise video available, so data is synthetic **but calibrated**:

1. **Calibration clips**: `scripts/make_demo_video.py` renders a clip of each exercise and runs the
   *real* feature extractor over it, measuring per-phase mean/std of the 19-D vector →
   `data/phase_stats_<exercise>.json`.
2. **Real-human bridge (MM-Fit)**: for **pushup and squat**, the local `mm-fit/` dataset
   (21 subjects, real 3D motion capture pose) is rendered back to video
   (`scripts/mmfit_pose_to_video.py`) and the same real extractor measures per-phase statistics →
   `data/phase_stats_mmfit_<exercise>.json`. These are **strictly preferred** by
   `load_phase_stats`, so the push-up and squat models learn the feature distribution of *actual
   humans*, not just a stick-figure render. This is the single biggest transfer improvement we
   could make with the data available. `exercises-dataset/` was evaluated and rejected (see §9).
3. **Sampling**: `generate_synthetic_data.py` samples (mean, widened std ~1.8×) per phase and adds
   a small per-sequence global shift, so each training sequence is a slightly different "subject".

Lean dataset (chosen for ~10-minute CPU training): **600 / 100 / 100 train-val-test sequences,
1–8 reps each**, phase ranges concentric (5–10), eccentric (6–12), idle (5–15) frames @ 15 FPS —
mean sequence ≈ 117 frames, still ≥ the 3/3/5 state-machine debounce so the model learns to
produce the phase runs the counter expects.

### Metrics
See §10 — final per-exercise numbers from the held-out test split.

---

## 5. Component C — LLM Coaching

- `summarize(stats, *, timeout_s=10.0)` returns a pydantic `CoachFeedback`.
- **Metrics are computed in Python** (tempo, consistency/CV, phase balance, confidence) and handed
  to the LLM as numbers — the LLM writes the words, never invents numbers.
- **Structured output**: OpenAI `response_format=json_schema` + a `_strict_json_schema` walker
  (forces `additionalProperties:false`, required-field supersets) + pydantic validation. One
  repair retry on validation failure; tenacity exponential backoff on rate limits.
- **Deterministic fallback**: any LLM failure (timeout, no key, schema error) returns a
  rule-based template (`_template_fallback`) — coaching *never* 500s. The template is also what
  tests assert against, so the test suite needs no network.
- Prompt is exercise-aware (`prompts/coach_v1.md`), versioned.

### Prompt evaluation (Component C written answer)

To decide whether the coaching prompt is any good, I would build an **offline rubric-based evaluation suite**: create a fixed set of 20–30 `RepStats` fixtures covering good, mediocre, and poor sessions (high/low rep count, fast/slow tempo, consistent/erratic, high/low confidence). For each fixture, generate coaching output once (with temperature fixed at 0.3) and score it against a rubric with four dimensions — **accuracy** (no invented numbers; every metric in the output matches the input stats), **actionability** (contains at least one concrete improvement suggestion), **safety** (flags fast reps or missing eccentric phase when those conditions are present), and **conciseness** (summary ≤ 2 sentences, total ≤ 120 words). A prompt that scores > 0.8 on accuracy and safety across all fixtures and > 0.6 on actionability is good enough to ship. If actionability lags, the prompt needs more explicit instruction to name one specific tempo or form improvement. If safety misses cases, the prompt needs explicit "if-then" rules for risky patterns. This suite runs in CI with a mocked LLM client (no network), so prompt regressions are caught before deployment.

---

## 6. Component D — Serving

### Endpoints
| Endpoint | Kind | Purpose |
|----------|------|---------|
| `POST /v1/analyze` | HTTP | Batch: video → reps + timing + coaching + latency breakdown |
| `WS /v1/stream` | WebSocket | Real-time: config → ready → JPEG frames → per-frame msgs → summary |
| `GET /healthz` | HTTP | Readiness (503 until model actually loaded) |
| `GET /v1/model` | HTTP | Model metadata incl. checkpoint hash, architecture, available exercises |

### Design details
- **Registry**: thread-safe singleton `ModelRegistry` loads all four checkpoints at startup with a
  warmup pass; injectable via `Depends(get_model_registry)` so tests swap in a fake without the
  singleton trap.
- **Blocking work off the event loop** (`run_in_threadpool`) for decode, features, LSTM, LLM.
- **Streaming isolation**: `StreamSession` owns its own `StreamingFeatureExtractor` +
  `RepCounter` + LSTM hidden state per connection — two webcams don't interfere. Inference is
  step-by-step, low latency, no replay needed.
- **Upload guardrails**: content-type check (415), streaming size cap (413), temp-file cleanup on
  all paths, request-ID in logs/errors.

---

## 7. Code Standards (PEP8, readability)

The user-facing requirement was *"code standards, PEP8, human-understandable, meaningful docs"*.
How this is held:

- **Docstrings everywhere**: module-level (why this file exists), class-level (what it is), and
  public-method-level (args/returns). One voice, present tense.
- **Readable names**: `min_idle_frames`, `rep_start_frame`, `phase_sequence` — not abbreviations.
- **Small, single-purpose functions**; no god objects.
- **`black` for formatting, `ruff` for lint, `mypy` for types** — config in `pyproject.toml` if you
  want to run them:
  ```bash
  black src/ tests/
  ruff check src/ tests/
  mypy src/
  ```
- **Tests are the executable documentation** of the state machine and the streaming protocol
  (`tests/test_reps.py`, `tests/test_stream.py`).
- PEP8 line length (≈88/100) respected throughout.

---

## 8. Real-Time: How It Actually Works

1. Client connects to `ws://host/v1/stream`, sends `{"type":"config","exercise":"pushup","source_fps":30.0}`.
2. Server creates a `StreamSession` and replies `ready`.
3. Client sends each camera frame as **JPEG binary**. Server:
   - decodes → `process_frame` (sub-samples to 15 FPS) → maybe a feature vector
   - `LSTM.step(feature)` → logits → softmax argmax → phase + confidence
   - `RepCounter.process_frame(phase)` → updated `rep_count`
   - replies `{"type":"frame","t_s":..., "phase":..., "confidence":..., "rep_count":...}`
4. `{"type":"summary"}` → full results + coaching any time; `{"type":"reset"}` → zero the session.
5. Disconnect → server sends final summary.

Latency per sampled frame ≈ decode + feature + one LSTM step (a few ms on CPU) — comfortably
faster than a 30 FPS camera, so the client never outruns the server. `scripts/live_webcam.py` is
a ready-made client (webcam or `--file replay.mp4`).

**What makes real-time trustworthy**: the causal feature path (§3) and causal LSTM (§4) mean the
model that streams live was trained on exactly the same computation, and pushup/squat were
calibrated on real-human MM-Fit pose rather than only renders.

---

## 9. Honest Caveats & What Was Rejected

### Known limitations
1. **Still synthetic labels.** Even the MM-Fit bridge uses *pose → render → real extractor*; the
   per-phase labels come from motion (velocity zero-crossings), not human annotation. Good enough
   to prove the pipeline end-to-end; not a claim of production accuracy.
2. **exercises-dataset/ rejected**: inspected — single-rep looping GIFs without rep/phase
   annotations, unusable for supervised training of a phase-sequence model. Documented rather than
   silently ignored.
3. **No multiperson handling** — the features assume one subject.
4. **Camera / lighting transfer** — frame-difference and flow are fairly invariant, but extreme
   camera shake or a busy background will inflate the "motion" features.
5. **State-machine thresholds are heuristic** (3/3/5) — would be tuned on real data.
6. **LLM cost/latency** — coaching is the slowest stage (~0.6 s); the fallback template keeps it
   from being a failure point.

### What would improve real-world accuracy next
- Collect 50–100 real labelled videos per exercise (the only real fix).
- Fine-tune the synthetic models on them (few-shot domain adaptation).
- Add MediaPipe pose keypoints as extra features (better kinematic signal than blobs).
- Temporal smoothing (CRF/HMM) over per-frame phase predictions.

---

## 10. Final Results

All four causal/unidirectional LSTMs (2×64, dropout 0.2) trained on the lean calibrated
dataset (600/100/100 train-val-test, 1–8 reps, 20 epochs, CPU ~14 min for all four).
Metrics are on the **held-out synthetic test split** (100 sequences each).

**Domain mixing** (pushup/squat only): training sequences are randomly drawn from *both*
the MM-Fit real-human calibration and the synthetic-render calibration, so the model learns
both distributions. This improves real-world transfer — the MM-Fit idle is noisy (people never
freeze) while the render idle is near-zero, and the model must handle both at serving time.

| Exercise | Calibration | Macro F1 | Idle F1 | Concentric F1 | Eccentric F1 | Rep MAE |
|----------|-------------|----------|---------|---------------|--------------|---------|
| pushup    | MM-Fit + render (mixed) | 0.814 | 0.858 | 0.768 | 0.815 | 0.12 |
| squat     | MM-Fit + render (mixed) | 0.775 | 0.813 | 0.739 | 0.772 | 0.16 |
| bicep_curl | Synthetic render     | 0.768 | 0.809 | 0.744 | 0.751 | 0.20 |
| jumping_jack | Synthetic render   | 0.849 | 0.891 | 0.799 | 0.857 | 0.20 |

> **Trust level**: these are *internal-consistency* numbers (synthetic features → held-out
> synthetic features), not evidence of real-world accuracy. They prove the model learns the
> calibrated per-phase distribution and the state machine counts reps with MAE ≲ 0.20 on the
> test split. Real-world transfer is *improved* for pushup/squat by domain mixing (MM-Fit +
> render) and is verified end-to-end on the synthetic demo videos and the streaming path.
> Per-exercise metric files: `checkpoints/test_metrics_<exercise>.json`.

Also verified end-to-end in this repo:
- The demo videos (`demo_<exercise>.mp4`, 3 synthetic reps each) are counted correctly through
  the full `/v1/analyze` path with the trained models.
- The `/v1/stream` WebSocket path produces per-frame phase/confidence/rep_count and a final
  summary — 102 unit tests pass, no network calls.

---

## 11. Assumptions Made

1. One subject per video/frame; ceiling/floor camera view roughly constant.
2. OpenCV can decode the uploaded format (mp4/mov/avi/mkv).
3. The LSTM is causal so one checkpoint serves both batch and streaming.
4. Per-phase feature statistics are approximately Gaussian around the calibration means.
5. LLM provider: OpenAI, structured output (`response_format`) required; fallback covers outage.
6. CPU-only; the lean dataset is sized to train all four exercises in ~10 min on CPU.

---

## 12. Tool Usage Notes

Developed with assistance from Claude (Anthropic): architecture, OpenCV feature extraction,
causal LSTM + streaming, state machine, pydantic/OpenAI structured output, FastAPI serving,
test design, Dockerfile.

Human decisions: exercise set, feature selection, state-machine thresholds, which real datasets
to use (MM-Fit bridge vs rejecting exercises-dataset), evaluation methodology, and the honest
caveats above.
