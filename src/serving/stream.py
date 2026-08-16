"""
Real-time streaming rep counting over WebSocket.

A client streams raw JPEG frames (or replayed video) to `/v1/stream`. The
server extracts causal features per sampled frame, runs the causal LSTM step
by step, and drives the same RepCounter state machine used by the batch path.
Progress is reported incrementally and a final summary is sent when the
client closes the stream.
"""
import logging
import time

import numpy as np

from src.llm.coach import CoachFeedback, RepStats, RepTiming, _template_fallback
from src.model.reps import RepCounter
from src.video.features import StreamingFeatureExtractor

logger = logging.getLogger(__name__)

# Frame is the classic JPEG; cv2.imencode produces this by default.
JPEG_CONTENT_TYPE = "image/jpeg"
_PHASE_NAME = {0: "idle", 1: "concentric", 2: "eccentric"}


def _stable_softmax(x, axis=-1):
    """Numeric-stable softmax for numpy arrays."""
    x = x - x.max(axis=axis, keepdims=True)
    exps = np.exp(x)
    return exps / exps.sum(axis=axis, keepdims=True)


class StreamSession:
    """One streaming session: features -> causal LSTM -> state machine.

    Owns a StreamingFeatureExtractor and a RepCounter so that every stream
    session is isolated (multiple clients can stream concurrently).

    When ``exercise_id`` is ``"auto"``, the session runs all four exercise
    models on a sliding window of recent features and automatically switches
    to the exercise with the highest non-idle confidence once it exceeds
    ``auto_confidence_threshold``.  The switch resets the rep counter and LSTM
    hidden state so only reps from the detected exercise are counted.
    """

    def __init__(
        self,
        registry,
        exercise_id: str = "pushup",
        source_fps: float = 30.0,
    ):
        self.registry = registry
        self._raw_exercise_id = exercise_id  # "auto" or a concrete id
        # When "auto", start with a default exercise so step() can always
        # resolve a real model. Auto-detect will switch exercise_id once
        # enough features are accumulated and confidence exceeds threshold.
        self.exercise_id = "pushup" if exercise_id == "auto" else exercise_id
        self.source_fps = source_fps
        self.target_fps = registry._settings_target_fps()
        self.extractor = StreamingFeatureExtractor(target_fps=self.target_fps)
        self.counter = RepCounter(
            min_concentric_frames=registry._settings_min_concentric_frames(),
            min_eccentric_frames=registry._settings_min_eccentric_frames(),
            min_idle_frames=registry._settings_min_idle_frames(),
            fps=self.target_fps,
            confidence_threshold=registry._settings_stream_confidence_threshold(),
        )
        self.hidden = None
        self.started = time.perf_counter()
        self.n_frames = 0
        self._non_idle_conf: list[float] = []
        # Minimum total grid energy (sum of first 16 feature dims) to accept a
        # non-idle prediction. Real webcam idle has subtle motion that the LSTM
        # can misclassify; this veto suppresses false concentric/eccentric
        # predictions when there is barely any frame-difference energy.
        self._motion_energy_threshold = 0.015
        # Auto-detect: how many features to accumulate before running all-4
        # models.  Set to ~2 seconds of target_fps samples.
        self._auto_window = int(self.target_fps * 1.5)
        # Minimum mean non-idle confidence (across the window) to declare a
        # detection and switch exercise.
        self._auto_confidence_threshold = 0.55
        # Ring buffer of recent features for auto-detect.
        self._auto_feature_buffer: list[np.ndarray] = []

    def _maybe_auto_detect(self, feature: np.ndarray) -> None:
        """If exercise is 'auto', accumulate features and trigger detection.

        The actual model evaluation runs in a background thread so the frame
        endpoint never blocks.  Detection is throttled to every
        ``_auto_window`` frames (~1.5s) to avoid overwhelming the CPU with
        repeated batch predictions.
        """
        if self._raw_exercise_id != "auto":
            return

        self._auto_feature_buffer.append(feature)
        # Only run detection when we have enough features and not already pending.
        if (len(self._auto_feature_buffer) < self._auto_window
                or getattr(self, "_auto_pending", False)):
            return

        # Mark as pending before submitting — prevents duplicate evaluations.
        self._auto_pending = True
        feats = np.stack(self._auto_feature_buffer)

        def _evaluate():
            try:
                best_exercise = self.exercise_id
                best_conf = 0.0
                for ex_id in self.registry.exercise_ids():
                    logits_arr, _ = self.registry.predict(feats, ex_id)
                    probs = _stable_softmax(logits_arr[0], axis=-1)
                    idle_mask = probs.argmax(axis=-1) != 0
                    mean_non_idle = (float(probs[idle_mask].max(axis=-1).mean())
                                     if idle_mask.any() else 0.0)
                    if mean_non_idle > best_conf:
                        best_conf = mean_non_idle
                        best_exercise = ex_id

                if best_conf >= self._auto_confidence_threshold and best_exercise != self.exercise_id:
                    logger.info(
                        f"Auto-detect: switching from {self.exercise_id} to "
                        f"{best_exercise} (conf={best_conf:.2f})"
                    )
                    self.exercise_id = best_exercise
                    self._switch_exercise()
            finally:
                self._auto_pending = False
                # Keep window fresh.
                if len(self._auto_feature_buffer) > self._auto_window:
                    self._auto_feature_buffer = self._auto_feature_buffer[-self._auto_window:]

        import threading
        threading.Thread(target=_evaluate, daemon=True).start()

    def _switch_exercise(self) -> None:
        """Reset counter, LSTM hidden state, and non-idle conf buffer."""
        self.counter.reset()
        self.hidden = None
        self._non_idle_conf = []

    def reset(self):
        """Start a fresh session: extractor, counter, hidden state, timing."""
        self.extractor.reset()
        self.counter.reset()
        self.hidden = None
        self.started = time.perf_counter()
        self.n_frames = 0
        self._non_idle_conf = []
        self._auto_feature_buffer = []

    def process_frame_bytes(self, jpeg_bytes: bytes) -> dict | None:
        """Decode one JPEG frame, extract a feature, run the LSTM step.

        When ``exercise_id`` is ``"auto"``, the session also accumulates
        features and periodically re-evaluates all 4 models, switching
        exercise when a confident detection is found.

        Returns None when the frame was skipped (below the target sample
        rate), otherwise a per-frame progress dict. Never raises on a bad
        frame — it is logged and skipped so one corrupt frame cannot kill
        the stream.
        """
        import cv2

        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
        except Exception:
            return None

        feature = self.extractor.process_frame(frame, source_fps=self.source_fps)
        if feature is None:
            return None

        self.n_frames += 1
        logits, self.hidden, _ = self.registry.step(feature, self.exercise_id, self.hidden)

        # Stable softmax -> prediction + confidence.
        logits = logits - logits.max()
        exps = np.exp(logits)
        probs = exps / exps.sum()
        phase = int(probs.argmax())
        confidence = float(probs[phase])

        # If we're in auto mode, check whether we should switch exercises.
        self._maybe_auto_detect(feature)

        # Motion veto: if the frame-difference grid energy is below threshold,
        # force idle regardless of what the LSTM predicted. Real webcam idle
        # (breathing, background flicker, slight camera shake) produces tiny
        # frame differences that the LSTM can misclassify as concentric.
        motion_energy = float(feature[:16].sum())
        if motion_energy < self._motion_energy_threshold:
            phase = 0  # idle
            confidence = max(confidence, 0.6)  # don't penalize the idle reading

        # Feed the state machine with confidence so low-confidence idle noise
        # is absorbed into the current state (prevents phantom rep counts).
        self.counter.process_frame(
            phase, self.extractor.sample_count - 1, confidence=confidence,
        )
        if phase != 0:
            self._non_idle_conf.append(confidence)

        return {
            "type": "frame",
            "t_s": round(self.extractor.sample_count / self.registry._settings_target_fps(), 3),
            "phase": _PHASE_NAME[phase],
            "phase_id": phase,
            "confidence": round(confidence, 4),
            "rep_count": self.counter.rep_count,
        }

    def summary(self) -> dict:
        """Final summary for the stream: rep count + per-rep timings + coaching.

        Reads the counter's committed state directly — `get_results()` would
        reset it — and reuses the LLM coaching path (with fallback).
        """
        fps = self.registry._settings_target_fps()
        reps = [
            {"start_s": round(start / fps, 3), "end_s": round(end / fps, 3)}
            for start, end in self.counter.reps
        ]
        rep_count = len(reps)
        durations = [r["end_s"] - r["start_s"] for r in reps]
        avg_tempo = float(np.mean(durations)) if durations else 0.0
        tempo_cv = float(np.std(durations) / avg_tempo) if avg_tempo > 0 else 0.0
        confidence = float(np.mean(self._non_idle_conf)) if self._non_idle_conf else 0.0

        # Coaching: reuse the LLM path (with fallback) for a consistent note.
        stats = RepStats(
            rep_count=rep_count,
            reps=[RepTiming(start_s=r["start_s"], end_s=r["end_s"]) for r in reps],
            avg_tempo_s=round(avg_tempo, 3),
            tempo_consistency=round(tempo_cv, 3),
            concentric_avg_s=0.0,
            eccentric_avg_s=0.0,
            confidence=round(confidence, 3),
        )
        feedback = self._coaching(stats)

        return {
            "type": "summary",
            "elapsed_s": round(time.perf_counter() - self.started, 2),
            "frames_processed": self.n_frames,
            "rep_count": rep_count,
            "reps": reps,
            "per_class_confidence": confidence,
            "coaching_feedback": feedback.model_dump(),
            "exercise": self.exercise_id,
        }

    def _coaching(self, stats: RepStats) -> CoachFeedback:
        """LLM coaching with the same timeout/fallback guarantees as batch."""
        try:
            return self.registry.get_coaching(stats, self.exercise_id)[0]
        except Exception as e:
            logger.warning(f"Stream coaching fallback: {e}")
            return _template_fallback(stats, exercise=self.exercise_id)
