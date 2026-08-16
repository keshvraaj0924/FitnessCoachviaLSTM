# Assessment Notes

This is my engineering journal for the exercise analysis pipeline. I'm writing it as honestly as possible — what worked, what didn't, what I'd change, and why.

---

## 1. Feature Representation (Component A)

### What I built

I extract a 19-dimensional feature vector from each frame pair using OpenCV. Here's what each part does:

**16 values — Frame-difference energy on a 4×4 grid**
I divide the frame into 16 cells (4 rows × 4 columns) and measure how much pixel change happened in each cell between the current frame and the previous one. This tells me *where* in the frame motion is occurring. If someone is doing pushups in the center of the frame, the middle cells light up. If they shift to the left, the left cells light up.

**1 value — Vertical centroid of motion**
I find the average vertical position of all the motion in the frame. When someone lowers into a pushup, their body moves down, so this number goes up (toward 1.0). When they push up, it goes down (toward 0.0). This is a simple but effective way to capture the up/down pattern of most exercises.

**1 value — Optical flow magnitude**
I use OpenCV's Farneback optical flow to measure how much motion there is overall, regardless of direction. This tells me whether the person is moving at all, and how vigorously.

**1 value — Optical flow vertical component**
From the optical flow, I extract just the vertical component. Upward motion gives negative values, downward gives positive. This is the key signal that tells me whether someone is in the concentric phase (pushing up) or eccentric phase (lowering down).

### Why these features

I picked features that are:

**Invariant to lighting and zoom.** Frame-difference energy is normalized by the cell area and clipped to [0, 1]. Optical flow is normalized by a maximum expected displacement. So a darker room, a brighter room, or a more zoomed-in camera all produce roughly the same feature values for the same movement.

**Causal — no peeking at the future.** Every feature at time t is computed from only the current frame and the previous frame. No averaging over the whole video, no look-ahead. This is the most important design decision because it means the exact same feature computation works for both batch video analysis and real-time webcam streaming. The model sees the same feature distribution at training time and at serving time.

**Interpretable.** I can look at the vertical centroid and vertical flow values and understand what the model is seeing. If the centroid isn't moving up and down, the model won't see a pushup pattern. This makes debugging easier.

### Preprocessing choices

**Letterbox resize to 160×160.** I resize frames to a fixed working resolution while preserving aspect ratio. If the input is 640×480 (4:3), it becomes 160×120 centered on a 160×160 canvas with black bars top and bottom. I don't stretch — stretching would distort the motion features.

**Don't trust CAP_PROP_FPS.** Phone videos often have incorrect or zero FPS in the file metadata. I estimate FPS from the frame count divided by duration, with a fallback to 30.0.

**Handle rotation metadata.** Phone videos recorded in portrait mode carry a rotation tag (90°, 180°, 270°). I read this tag and rotate the frames before processing. Without this, a portrait video would be analyzed sideways and the features would be wrong.

**Graceful error handling.** If the video is corrupt, zero-length, or unreadable, I raise a clear error. If it's shorter than one frame at the target FPS, I still process at least one frame (by duplicating it).

---

## 2. Metrics and How Much I Trust Them

### The numbers

I trained four unidirectional LSTM models (2 layers, 64 hidden units, dropout 0.2) on synthetic data. Each model was evaluated on a held-out test split of 100 sequences. Here are the results:

| Exercise | Macro F1 | Idle F1 | Concentric F1 | Eccentric F1 | Rep MAE |
|----------|----------|---------|---------------|--------------|---------|
| pushup | 0.814 | 0.858 | 0.768 | 0.815 | 0.12 |
| squat | 0.775 | 0.813 | 0.739 | 0.772 | 0.16 |
| bicep_curl | 0.768 | 0.809 | 0.744 | 0.751 | 0.20 |
| jumping_jack | 0.849 | 0.891 | 0.799 | 0.857 | 0.20 |

### How much do I trust these?

**Honest answer: these measure internal consistency, not real-world accuracy.**

Here's what the numbers actually prove: the model can learn the per-phase feature distribution I calibrated, and the state machine can count reps with reasonable accuracy on data drawn from the same distribution it was trained on.

Here's what they don't prove: that the model works on a real person doing real pushups in a real room with a real phone camera. The features are real (they come from actual OpenCV processing of actual rendered frames), but the labels are synthetic — they come from the animation timeline of a stick-figure render, not from human annotation.

**What makes me cautiously optimistic:**

1. **The features are real.** Even though the labels are synthetic, the feature vectors come from running the actual OpenCV pipeline (frame-difference grid, optical flow, centroid) over real images. The model learns the same distribution it sees at serving time.

2. **Pushup and squat are calibrated on real human data.** I used the MM-Fit dataset (21 subjects, real 3D motion capture) to generate feature statistics for these two exercises. I rendered the real 3D poses back to video and ran the real feature extractor over them. This means the pushup and squat models have seen feature distributions from actual humans, not just stick figures. This is the single biggest thing I did to improve real-world transfer.

3. **The rep MAE is low.** A rep MAE of 0.12 for pushups means the model typically gets the rep count right or off by one on the synthetic test set. That's a reasonable starting point.

**What worries me:**

1. **Synthetic labels.** The per-phase labels come from motion (velocity zero-crossings), not from a human watching the video and saying "this is the eccentric phase." The model might learn patterns in the synthetic data that don't exist in real videos.

2. **No real-world test set.** I haven't evaluated on real human videos. The demo videos I included are also synthetic renders. The only real-world test is the streaming webcam path, which works but isn't quantitatively measured.

3. **State-machine thresholds are heuristic.** The 3/3/5 frame debounce thresholds were chosen by intuition. On real data, they might need tuning.

**Bottom line:** I'd trust these metrics as proof that the pipeline works end-to-end and the model learns something meaningful. I wouldn't trust them as a prediction of real-world accuracy without a real labeled test set.

---

## 3. Prompt Evaluation (Component C)

### How would I decide if the coaching prompt is any good?

I would build an **offline rubric-based evaluation suite**. Here's exactly how it works:

**Step 1: Create a fixed test set of scenarios.**
I'd write 20-30 `RepStats` objects that cover the full range of possible sessions:
- Great sessions: high rep count, consistent tempo, good form
- Mediocre sessions: moderate reps, some tempo variation
- Poor sessions: very few reps, very fast reps, inconsistent tempo, low model confidence
- Edge cases: zero reps, exactly 1 rep, very fast eccentric phase

Each scenario is a fixed input that never changes, so evaluation is deterministic.

**Step 2: Define a scoring rubric with four dimensions.**

1. **Accuracy (weight: 40%)**: Does the output mention any numbers that don't match the input stats? Every metric in the feedback — rep count, tempo, consistency — must match the input `RepStats`. If the LLM says "you completed 10 reps" when the input says 3, that's a zero.

2. **Actionability (weight: 30%)**: Does the feedback contain at least one concrete, specific improvement suggestion? "Slow down the eccentric phase" is actionable. "Try harder" is not.

3. **Safety (weight: 20%)**: Does the feedback flag risky patterns when they're present? If the eccentric phase is under 1 second, the feedback should mention injury risk. If the confidence is very low, it should suggest checking video quality.

4. **Conciseness (weight: 10%)**: Is the feedback brief and readable? Summary should be 1-2 sentences. Total word count under 120.

**Step 3: Run the evaluation.**

For each scenario, I generate coaching output once (temperature fixed at 0.3 for reproducibility) and score it against the rubric. I'd aim for:
- Accuracy > 0.9 across all scenarios (the LLM must never invent numbers)
- Safety > 0.8 (risky patterns must be flagged consistently)
- Actionability > 0.6 (at least half the suggestions should be specific)

**Step 4: Use it as a regression test.**

This suite runs in CI with a mocked LLM client (no network calls). If a prompt change causes actionability to drop from 0.8 to 0.4, that's a regression that should be caught before deployment.

**How I'd interpret the results:**

- If accuracy is low → the prompt needs clearer instructions about not inventing numbers
- If actionability is low → the prompt needs more explicit instruction to name specific tempo or form improvements
- If safety misses cases → the prompt needs explicit "if-then" rules for risky patterns (e.g., "if eccentric_avg_s < 1.0, mention injury risk")
- If consistency is high but all feedback sounds the same → the prompt needs more personality or exercise-specific vocabulary

---

## 4. What I Didn't Finish, What I'd Do Next, and Production Additions

### What I didn't finish

1. **Real labeled data collection and model fine-tuning.** The models are trained on synthetic data. I calibrated the feature distribution as well as I could (especially with MM-Fit for pushup and squat), but the labels are still synthetic. The honest assessment is that the pipeline works end-to-end but real-world accuracy is unproven.

2. **Real-world evaluation.** I don't have a test set of real human exercise videos with ground-truth rep counts and phase labels. Without this, I can't quantify how well the system works on actual users.

3. **Temporal smoothing.** The model makes per-frame predictions that can flicker between classes. A simple conditional random field (CRF) or a short moving-average smoothing pass over the phase sequence would improve stability. I mention it in NOTES but didn't implement it.

4. **MediaPipe pose features.** I considered adding pose keypoints (shoulder, elbow, hip, knee positions) as additional features. This would give the model a much stronger kinematic signal than frame-difference blobs. I didn't add it because it would have required integrating a new dependency and the current features are sufficient to prove the pipeline works.

### What I'd do next

1. **Collect 50-100 real labeled videos per exercise.** This is the highest-priority task. Even a small real dataset would let me fine-tune the synthetic models and get honest real-world metrics.

2. **Fine-tune on real data.** Use the synthetic models as a starting point and fine-tune on the small real dataset. A few epochs of fine-tuning would adapt the model to real-world feature distributions.

3. **Add MediaPipe pose features.** Extract 17 pose keypoints per frame and append their positions and velocities to the 19-D feature vector. This would give the model a much richer signal.

4. **Tune state-machine thresholds on real data.** The 3/3/5 frame debounce was chosen by intuition. With real labeled data, I'd optimize these thresholds to maximize F1 on a validation set.

5. **Build the prompt evaluation suite** described in section 3 and integrate it into CI.

### What I'd add for production

**Before going to production, I would add:**

1. **Authentication and API key management.** Right now anyone can call the API. I'd add API key validation and rate limiting per key.

2. **Async LLM calls via a task queue.** The LLM is currently the slowest stage (~0.6s per request). I'd move it to a Celery/Redis task queue so the API can return the rep count immediately and the coaching feedback can arrive via webhook or polling. This also makes the API resilient to LLM outages.

3. **Request-level timeouts and circuit breakers.** If the LLM is down or slow, the API should fail fast and return the template fallback, not hang for 10+ seconds.

4. **Model versioning and A/B testing.** I'd version every checkpoint (e.g., `v1.0.0`, `v1.1.0`) and support running multiple versions simultaneously. This lets me deploy a new model to a percentage of traffic and compare metrics before full rollout.

5. **Monitoring and alerting.** I'd log per-class confidence distributions and rep-count confidence per request. If confidence drops below a threshold or the LLM fallback rate spikes, I'd alert the team. I'd also track latency percentiles (p50, p95, p99) for each pipeline stage.

6. **Horizontal scaling.** Right now the model registry is a singleton in one process. In production, I'd run multiple worker processes behind a load balancer, each loading its own model copy. Models are small (~100KB each for the weights), so memory isn't a concern.

7. **Video preprocessing optimizations.** Currently the entire video is loaded into memory as a list of frames. For long videos, I'd stream frames from disk using a generator to keep memory usage bounded.

8. **Multi-person handling.** The current features assume one subject. In production, I'd add person detection/tracking to handle videos with multiple people.

---

## 5. Assumptions

1. **One subject per video.** The feature extraction assumes one person moving in the frame. If there are multiple people, the motion features will be a blend of all their movements.

2. **OpenCV can decode the format.** I support mp4, mov, avi, and mkv containers. If a user uploads a format OpenCV can't decode, the API returns an error.

3. **The LSTM being causal is the right call.** I assumed that real-time streaming was a hard requirement (the brief explicitly mentions it), so I made the model causal/unidirectional. This means I can't use bidirectional context for the batch path. I think this was the right trade-off — the brief values having both batch and streaming work over squeezing out every last point of batch accuracy.

4. **Per-phase features are roughly Gaussian.** My synthetic data generation assumes feature vectors cluster around a mean with a roughly Gaussian distribution. If the real distribution is multi-modal or has heavy tails, the synthetic data won't capture that.

5. **LLM provider is OpenAI.** I used OpenAI's structured output mode (`response_format=json_schema`) because it's the most reliable way to enforce a JSON schema. The code is structured so the LLM client could be swapped for a different provider, but the prompt and validation logic are OpenAI-specific.

6. **CPU-only training is sufficient.** The brief specifies CPU only, and I sized the dataset (600 train / 100 val / 100 test sequences, lean model architecture) to train all four exercises in roughly 10 minutes on CPU.

7. **The exercises-dataset/ folder was not useful.** The brief said I could use any publicly available data. I found an `exercises-dataset/` directory with exercise GIFs, evaluated it, and rejected it because the clips are short looping animations without rep or phase annotations. Documenting this decision felt more honest than silently ignoring it.

8. **File extension as fallback for content-type validation.** Some clients send `application/octet-stream` for valid video files. I added extension-based fallback validation so these aren't rejected. This is slightly less strict than MIME-type-only validation but more practical for real-world clients.

---

## 6. API Key Security

The OpenAI API key is provided exclusively through the `OPENAI_API_KEY` environment variable. I never commit API keys to the repository.

**How to set it up locally:**
```bash
# Create a .env file (this file is gitignored)
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

**How to set it up in Docker:**
```bash
docker run -p 8000:8000 -e OPENAI_API_KEY="sk-your-key-here" pushup-analysis
```

The `.env` file is listed in `.gitignore` and `.dockerignore` so it never enters version control or Docker layers. The `OPENAI_API_KEY` environment variable is documented in the README.md setup instructions.

---

## 7. LLM Provider Choice

I chose **OpenAI** (specifically `gpt-4o-mini`) for the coaching LLM because:

1. **Structured output support**: OpenAI's `response_format=json_schema` with strict mode is the most reliable way to enforce a JSON schema. Other providers either don't support structured output or do so less strictly.

2. **Quality at low cost**: `gpt-4o-mini` is fast and inexpensive while producing coherent, actionable coaching feedback.

3. **The brief allows any provider**: The brief says "you may use any LLM provider you have access to." I have access to OpenAI.

The coaching layer is designed so the LLM client could be swapped for a different provider (Anthropic, Google, etc.) by implementing a new client class. The prompt template, RepStats computation, and fallback logic are all provider-agnostic.
