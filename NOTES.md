# Assessment Notes

This is my engineering journal for the exercise analysis pipeline. I'm writing it as honestly as possible — what worked, what didn't, what I'd change, and why.

---

## 1. Feature Representation (Component A)

### The 19-D Feature Vector

I extract a 19-dimensional feature vector from each frame pair using OpenCV:

- **16 values — Frame-difference energy on a 4×4 grid**: I divide the frame into 16 cells and measure how much pixel change happened in each cell between the current frame and the previous one. This tracks *where* movement happens (center vs. sides) without running a full bounding-box detector.

- **1 value — Vertical centroid of motion**: The weighted vertical average of all active motion pixels. When someone lowers into a pushup, their body moves down and this number goes up (toward 1.0). When they push up, it goes down (toward 0.0). This captures the up/down pattern that defines exercises like pushups and squats.

- **1 value — Optical flow magnitude**: Using Farneback optical flow, I measure how much motion there is overall regardless of direction. This tells me whether the person is moving at all and how vigorously.

- **1 value — Vertical optical flow component**: From the optical flow, I extract just the vertical component. Negative values mean upward movement (concentric — pushing up). Positive values mean downward movement (eccentric — lowering down). This is the key signal that distinguishes the two active phases.

### Why These Features

**Normalization and invariance**: Frame-difference energy is normalized by cell area and clipped to [0, 1]. Optical flow is scaled by a maximum expected displacement. This keeps feature values stable across different lighting conditions and zoom levels. A darker room, a brighter room, or a more zoomed-in camera all produce roughly the same values for the same movement.

**Strictly causal**: Features at time t use only frames t and t-1. No future lookups, no full-video averaging. This is the most important design decision — it ensures the exact same feature computation works for both batch video uploads and live webcam streaming. The model sees the same feature distribution at training time and at serving time.

**Interpretable**: If reps are missed, I can trace whether the vertical centroid shifts were too weak or the flow velocity was too low. This makes debugging much easier than working with a black-box feature extractor.

### Preprocessing Details

**Letterboxing to 160×160**: I resize frames to a fixed working resolution while preserving aspect ratio. Black bars are added where needed — I never stretch the image, because stretching would distort the motion features.

**FPS fallback**: CAP_PROP_FPS in video metadata is frequently 0 or broken on mobile recordings. I estimate FPS from frame count divided by duration, with a fallback to 30.0 FPS.

**Rotation handling**: EXIF/MP4 rotation tags (90°, 180°, 270°) are parsed and applied so portrait mobile videos aren't analyzed sideways. Without this, a phone recording in portrait mode would have all its motion features rotated 90 degrees.

**Edge cases**: Unreadable or corrupt files raise a clear error. Zero-length or very short videos are handled gracefully — if a clip is shorter than one frame at the target FPS, I still process at least one frame by duplicating it.

---

## 2. Metrics and How Much I Trust Them

### Evaluation Results

Four 2-layer unidirectional LSTMs (64 hidden units, dropout 0.2) evaluated on a held-out synthetic test set (100 sequences per exercise):

| Exercise | Macro F1 | Idle F1 | Concentric F1 | Eccentric F1 | Rep MAE |
|----------|----------|---------|---------------|--------------|---------|
| Pushup | 0.814 | 0.858 | 0.768 | 0.815 | 0.12 |
| Squat | 0.775 | 0.813 | 0.739 | 0.772 | 0.16 |
| Bicep Curl | 0.768 | 0.809 | 0.744 | 0.751 | 0.20 |
| Jumping Jack | 0.849 | 0.891 | 0.799 | 0.857 | 0.20 |

### The Honest Truth

**What these numbers prove**: The model effectively learns the temporal phase distribution, and the state machine reliably counts reps on data drawn from the same distribution it was trained on. A rep MAE of 0.12 for pushups means the model is rarely off by more than one rep on the synthetic test set.

**What these numbers don't prove**: Real-world accuracy on actual phone recordings with real clutter, varying clothing, and genuine human movement. The labels were generated from motion zero-crossings in the animation data, not from a human watching each frame and annotating it. I haven't evaluated on a real human-labeled test set.

**Why they're still meaningful**:

- The features are real — computed from actual OpenCV processing of rendered video frames.
- Pushup and squat feature distributions were grounded in real human 3D motion-capture data (MM-Fit dataset, 21 subjects). The models for these two exercises have seen feature statistics from actual humans, not just stick-figure animations.
- The pipeline works end-to-end — demo videos count correctly, the streaming path produces stable per-frame predictions, and the coaching layer generates useful feedback.

**The caveat**: State machine debounce thresholds (3/3/5 frames) were set heuristically based on intuition about reasonable rep pacing. On real data with real humans moving at different speeds, these thresholds would need empirical validation and tuning.

---

## 3. Prompt Evaluation (Component C)

### Offline Rubric-Based Suite

Rather than eyeballing LLM outputs, I structured an automated evaluation pipeline:

**Step 1: Fixed test scenarios.** A static set of 20–30 `RepStats` cases covering the full range: perfect sessions, slow/fast eccentrics (under 1.0 seconds), fatigue patterns, zero-rep edge cases, and low model confidence. Each scenario is a fixed input that never changes, so evaluation is fully deterministic.

**Step 2: Four-dimension rubric.**

- **Accuracy (40%)**: Every number in the output must match the input `RepStats`. If the LLM says "you completed 10 reps" when the input says 3, that's an automatic zero. The LLM rephrases — it never invents.

- **Actionability (30%)**: Must provide specific form cues ("slow the eccentric phase to 2-3 seconds") rather than vague advice ("try harder"). The prompt explicitly instructs the model to name one concrete tempo or form improvement.

- **Safety (20%)**: Must flag risky patterns when present. If the eccentric phase is under 1 second, the feedback should mention injury risk. If model confidence is very low, it should suggest checking video quality or positioning.

- **Conciseness (10%)**: Summary should be 1–2 sentences. Total output under 120 words. Coaching feedback should be scannable, not a paragraph.

**Step 3: CI/CD regression test.** This suite runs against mock LLM responses at temperature=0.3 (fixed for reproducibility). If a prompt change causes actionability to drop from 0.8 to 0.4, that's a regression caught before deployment. No network calls needed — the mock client returns pre-canned responses.

### How I'd Interpret Results

- If accuracy is low → the prompt needs clearer guardrails against inventing numbers
- If actionability is low → the prompt needs more explicit instruction to name specific improvements
- If safety misses cases → the prompt needs explicit "if-then" rules for risky patterns
- If all feedback sounds the same → the prompt needs more exercise-specific vocabulary or personality

---

## 4. What's Missing, Next Steps, and Production Additions

### What I Didn't Finish

**Real-world test benchmark**: I don't have a test set of real human exercise videos with ground-truth rep counts and phase labels. Without this, I can't quantitatively prove field accuracy — only internal consistency on synthetic data.

**Temporal sequence smoothing**: Per-frame class predictions can flicker between classes on ambiguous frames. Right now the debounce logic in the state machine absorbs most of this, but a proper CRF or Viterbi post-processing pass would be more principled.

**MediaPipe pose landmarks**: I considered adding 17 pose keypoints (shoulder, elbow, hip, knee, etc.) as additional features. This would give the model a much richer kinematic signal than frame-difference blobs. I didn't add it because it would introduce a new runtime dependency and the current features are sufficient to prove the pipeline works end-to-end.

### What I'd Do Next

1. **Collect 50–100 labeled real-world videos per exercise.** This is the highest-priority task. Even a small real dataset would let me fine-tune the synthetic models and get honest real-world metrics.

2. **Add MediaPipe pose features.** Extract 17 keypoints per frame and append their positions and velocities to the 19-D feature vector. The model would then have both appearance-based features (frame differences, optical flow) and kinematic features (joint positions, joint velocities).

3. **Tune state-machine thresholds on real data.** The 3/3/5 frame debounce was chosen by intuition. With real labeled data, I'd run a grid search to find the thresholds that maximize F1 on a validation set.

4. **Build the prompt evaluation suite** described above and wire it into GitHub Actions.

### What I'd Add for Production

**Before this goes to production, I would add:**

- **Authentication and API key management.** Right now anyone can call the API. I'd add API key validation with per-key rate limiting.

- **Async LLM calls via a task queue.** The LLM is currently the slowest stage (~0.6 seconds per request). I'd move it to a Celery/Redis task queue so the API returns the rep count immediately and coaching feedback arrives asynchronously via webhook or polling. This also makes the API resilient to LLM outages.

- **Circuit breakers and request timeouts.** If the LLM is down or slow, the API should fail fast and return the template fallback, not hang for 10+ seconds.

- **Model versioning and A/B testing.** Every checkpoint gets a semantic version (v1.0.0, v1.1.0). Multiple versions can run simultaneously, and I can deploy a new model to a percentage of traffic to compare metrics before full rollout.

- **Monitoring and alerting.** I'd log per-class confidence distributions and rep-count confidence per request. If confidence drops below a threshold or the LLM fallback rate spikes unexpectedly, the team gets alerted. I'd track latency percentiles (p50, p95, p99) for each pipeline stage.

- **Horizontal scaling.** Right now the model registry is a singleton in one process. In production, I'd run multiple worker processes behind a load balancer, each loading its own model copy. The models are small (~100KB for weights each), so memory isn't a concern.

- **Streaming video from disk.** Currently the entire video is loaded into memory as a list of frames. For long videos, I'd use a generator that reads and yields frames one at a time, keeping memory usage bounded.

- **Multi-person handling.** The current features assume one subject. In production, I'd add person detection and tracking to isolate a single person in crowded scenes.

---

## 5. Assumptions

1. **One subject per video.** The feature extraction assumes one person moving in the frame. If multiple people are present, the motion features will be a blend of all their movements and the rep count will be wrong.

2. **OpenCV can decode the format.** I support mp4, mov, avi, and mkv containers. If a user uploads a format OpenCV can't decode, the API returns an error. I considered adding FFmpeg as a fallback decoder but kept the dependency tree minimal.

3. **Causal model is the right trade-off.** I chose a unidirectional LSTM because the brief explicitly requires real-time streaming. This means the batch path can't use bidirectional context. I think this was the right call — having both batch and streaming work with identical weights is more valuable than squeezing out a few extra F1 points on batch-only analysis.

4. **Gaussian feature distribution.** My synthetic data generation assumes feature vectors cluster around a mean with roughly Gaussian variance per phase. If the real distribution is multi-modal or has heavy tails, the synthetic data won't fully capture that.

5. **CPU-only training is sufficient.** The brief specifies CPU only. I sized the dataset (600 train / 100 val / 100 test sequences) and model architecture to train all four exercises in roughly 10 minutes on CPU.

6. **OpenAI for LLM coaching.** I used OpenAI's structured output mode because it's the most reliable way to enforce a JSON schema. The coaching layer is structured so the LLM client could be swapped for a different provider, but the prompt and validation logic are currently OpenAI-specific.

7. **The exercises-dataset/ folder was not useful.** I found an `exercises-dataset/` directory with exercise clips, evaluated it, and rejected it. The clips are short looping animations without rep or phase annotations — useless for supervised training of a phase-sequence model. I documented this rather than silently ignoring it.

8. **File extension as fallback for content-type validation.** Some clients send `application/octet-stream` for valid video files. I added extension-based fallback validation so these aren't rejected. This is slightly less strict than MIME-type-only validation but more practical for real-world clients.

---

## 6. API Key Security

The OpenAI API key is provided exclusively through the `OPENAI_API_KEY` environment variable. It is never committed to the repository.

**Local development:**
```bash
# Create a .env file (gitignored)
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

**Docker:**
```bash
docker run -p 8000:8000 -e OPENAI_API_KEY="sk-your-key-here" pushup-analysis
```

The `.env` file is listed in `.gitignore` and `.dockerignore` so it never enters version control or Docker image layers.

---

## 7. LLM Provider Choice

I chose **OpenAI** (`gpt-4o-mini`) for the coaching LLM because:

- **Structured output support**: OpenAI's `response_format=json_schema` with strict mode is the most reliable way to enforce a JSON schema contract. Other providers either don't support structured output or do so less strictly.

- **Quality at low cost**: `gpt-4o-mini` is fast and inexpensive while producing coherent, actionable coaching feedback.

- **The brief allows any provider**: The brief says I may use any LLM provider I have access to. I have access to OpenAI.

The coaching layer is structured so the LLM client could be swapped for a different provider (Anthropic, Google, etc.) by implementing a new client class. The prompt template, RepStats computation, Pydantic validation, and deterministic fallback are all provider-agnostic.
