"""
Component B: Rep Counting Logic

State machine to derive rep count from predicted phase sequence.
Classes: 0=idle, 1=concentric (pushing up), 2=eccentric (lowering down)
"""
import logging
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


class Phase(IntEnum):
    """Push-up phases."""
    IDLE = 0
    CONCENTRIC = 1  # Pushing up
    ECCENTRIC = 2   # Lowering down


@dataclass
class RepTiming:
    """Start and end time of a repetition in seconds."""
    start_s: float
    end_s: float

    def duration(self) -> float:
        return self.end_s - self.start_s


@dataclass
class RepCountResult:
    """Result of rep counting."""
    rep_count: int
    reps: list[RepTiming]
    phase_sequence: list[int]  # Predicted phase per frame
    confidence: float  # Average confidence of non-idle predictions


class RepCounter:
    """
    State machine for counting push-up repetitions from phase predictions.

    A rep is the sequence CONCENTRIC -> ECCENTRIC (push up, lower down).
    The machine tracks a committed phase and only switches to a different
    phase once that new phase has been predicted for a minimum number of
    *consecutive* frames (the debounce). Short/noisy excursions that do not
    meet the debounce are absorbed into the current state.

    Transition debounce (minimum consecutive frames of the target phase):
      IDLE        -> CONCENTRIC : min_concentric_frames  (validates the push-up
                                                           phase is real)
      CONCENTRIC  -> ECCENTRIC  : min_eccentric_frames   (validates lowering)
      ECCENTRIC   -> IDLE       : min_idle_frames        (validates the rep
                                                           really ended)
      CONCENTRIC  -> IDLE       : min_idle_frames        (abort, rep not counted)

    A CONCENTRIC prediction while in ECCENTRIC (with no validated idle run in
    between) is absorbed as part of the same continuous set of reps — the
    rep is only closed by a validated idle run. So push-ups performed back to
    back with no real pause are counted as one continuous set.

    Rep boundaries are recorded at the *start* of the triggering run, so a rep
    is timed [start of concentric run, start of the idle run that ends it).

    Invalid phase sequences (e.g. CONCENTRIC straight back to IDLE after a
    validated concentric) abort the current rep instead of counting it.
    """

    def __init__(
        self,
        min_concentric_frames: int = 3,
        min_eccentric_frames: int = 3,
        min_idle_frames: int = 5,
        fps: int = 15,
        confidence_threshold: float = 0.50,
    ):
        self.min_concentric_frames = min_concentric_frames
        self.min_eccentric_frames = min_eccentric_frames
        self.min_idle_frames = min_idle_frames
        self.fps = fps
        # Minimum softmax confidence to accept a predicted phase change.
        # Below this the frame is treated as "stay in current state",
        # suppressing the noisy transitions that cause phantom rep counts
        # when the model is uncertain (e.g. real webcam idle).
        self.confidence_threshold = confidence_threshold

        self.reset()

    @property
    def rep_count(self) -> int:
        """Number of completed reps so far (committed list length).

        Readable live during a stream, unlike ``get_results`` which replays
        the full sequence and then resets the counter.
        """
        return len(self.reps)

    def reset(self):
        """Reset the counter to initial state."""
        self.state = Phase.IDLE
        self.frames_in_state = 0
        self.rep_start_frame: int | None = None
        self.reps: list[tuple[int, int]] = []  # (start_frame, end_frame)
        self._run: int = 0          # consecutive frames of the candidate phase
        self._run_start: int = 0    # frame index where the current run began
        self._last_pred: Phase | None = None

    def _set_state(self, new_state: Phase, frame_idx: int):
        """Commit to a new state (already passed the debounce)."""
        self.state = new_state
        self.frames_in_state = 1
        self._run = 0
        self._last_pred = None

    def process_frame(self, predicted_phase: int, frame_idx: int, confidence: float | None = None):
        """Process a single frame's predicted phase.

        Args:
            predicted_phase: Predicted phase label (0=idle, 1=concentric, 2=eccentric).
            frame_idx: Zero-based frame index in the sequence.
            confidence: Optional softmax confidence for this prediction. When
                        provided and below ``confidence_threshold``, the frame
                        is treated as "stay in current state" to suppress
                        noisy transitions from uncertain predictions.
        """
        phase = Phase(predicted_phase)

        # Confidence gate: low-confidence predictions do not trigger a
        # transition — absorb them into the current state so a wobbly idle
        # does not spawn phantom concentric/eccentric runs.
        if confidence is not None and confidence < self.confidence_threshold:
            phase = self.state

        # Same as committed state: nothing to debounce. Clear the pending-run
        # bookkeeping so a committed-phase frame cannot chain into the next run.
        if phase == self.state:
            self.frames_in_state += 1
            self._run = 0
            self._last_pred = None
            return

        # A different phase: extend its consecutive run (or start a new one).
        if phase == self._last_pred:
            self._run += 1
        else:
            self._run = 1
            self._run_start = frame_idx
        self._last_pred = phase

        if self.state == Phase.IDLE:
            if phase == Phase.CONCENTRIC and self._run >= self.min_concentric_frames:
                self.rep_start_frame = self._run_start
                self._set_state(Phase.CONCENTRIC, frame_idx)
            # ECCENTRIC (or a too-short CONCENTRIC) while idle is ignored.

        elif self.state == Phase.CONCENTRIC:
            if phase == Phase.ECCENTRIC and self._run >= self.min_eccentric_frames:
                self._set_state(Phase.ECCENTRIC, frame_idx)
            elif phase == Phase.IDLE and self._run >= self.min_idle_frames:
                self.rep_start_frame = None  # abort, rep not counted
                self._set_state(Phase.IDLE, frame_idx)

        elif self.state == Phase.ECCENTRIC:
            if phase == Phase.IDLE and self._run >= self.min_idle_frames:
                if self.rep_start_frame is not None:
                    self.reps.append((self.rep_start_frame, self._run_start))
                self.rep_start_frame = None
                self._set_state(Phase.IDLE, frame_idx)
            # CONCENTRIC while ECCENTRIC (no validated idle in between) is part
            # of the same continuous set of reps: it is absorbed and the rep is
            # closed later by a validated idle run.

    def get_results(self, phase_sequence: list[int], confidences: list[float] | None = None) -> RepCountResult:
        """
        Get final results after processing all frames.

        Args:
            phase_sequence: List of predicted phases per frame
            confidences: Optional list of prediction confidences per frame

        Returns:
            RepCountResult with rep count and timings
        """
        self.reset()

        for frame_idx, (phase, conf) in enumerate(
            zip(phase_sequence, confidences or [None] * len(phase_sequence))
        ):
            self.process_frame(phase, frame_idx, confidence=conf)

        # Convert frame indices to timestamps
        reps = [
            RepTiming(start_s=start / self.fps, end_s=end / self.fps)
            for start, end in self.reps
        ]

        # Calculate average confidence for non-idle frames
        if confidences is not None:
            # Align confidences with phase_sequence
            non_idle_conf = []
            for p, c in zip(phase_sequence, confidences):
                if p != Phase.IDLE:
                    non_idle_conf.append(c)
            confidence = float(np.mean(non_idle_conf)) if non_idle_conf else 0.0
        else:
            confidence = 0.0

        return RepCountResult(
            rep_count=len(self.reps),
            reps=reps,
            phase_sequence=phase_sequence,
            confidence=confidence,
        )


def count_reps_from_logits(
    logits: np.ndarray,  # (T, 3) or (1, T, 3)
    min_concentric_frames: int = 3,
    min_eccentric_frames: int = 3,
    min_idle_frames: int = 5,
    fps: int = 15,
) -> RepCountResult:
    """
    Convenience function to count reps from model logits.

    Args:
        logits: Model output logits, shape (T, 3) or (1, T, 3)
        min_concentric_frames: Minimum frames in concentric phase
        min_eccentric_frames: Minimum frames in eccentric phase
        min_idle_frames: Minimum frames in idle phase
        fps: Frames per second for timestamp conversion

    Returns:
        RepCountResult
    """
    # Handle batch dimension
    if logits.ndim == 3:
        logits = logits[0]  # (T, 3)

    # Softmax with numerically-stable log-sum-exp
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(logits)
    probs = exps / exps.sum(axis=1, keepdims=True)
    predictions = probs.argmax(axis=1).tolist()
    confidences = probs.max(axis=1).tolist()

    counter = RepCounter(
        min_concentric_frames=min_concentric_frames,
        min_eccentric_frames=min_eccentric_frames,
        min_idle_frames=min_idle_frames,
        fps=fps,
    )

    return counter.get_results(predictions, confidences)


def evaluate_rep_counting(
    true_labels: list[list[int]],  # List of sequences, each is list of phase labels
    pred_labels: list[list[int]],  # List of predicted sequences
    fps: int = 15,
    min_concentric_frames: int = 3,
    min_eccentric_frames: int = 3,
    min_idle_frames: int = 5,
) -> dict:
    """
    Evaluate rep counting performance on a dataset.

    Args:
        true_labels: Ground truth phase labels per video
        pred_labels: Predicted phase labels per video
        fps: Frames per second
        min_concentric_frames: Min concentric frames threshold
        min_eccentric_frames: Min eccentric frames threshold
        min_idle_frames: Min idle frames threshold

    Returns:
        Dict with per-class F1, rep-count MAE, and other metrics
    """
    # Per-class precision/recall/F1 computed by hand over concatenated frames
    # (computed across all videos, then macro-averaged). No external metrics
    # dependency is required.
    per_class = {}
    for cls in (Phase.IDLE, Phase.CONCENTRIC, Phase.ECCENTRIC):
        tp = fp = fn = 0
        for t, p in zip(true_labels, pred_labels):
            min_len = min(len(t), len(p))
            for tt, pp in zip(t[:min_len], p[:min_len]):
                if pp == cls:
                    if tt == cls:
                        tp += 1
                    else:
                        fp += 1
                elif tt == cls:
                    fn += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[Phase(cls).name.lower()] = float(f1)

    per_class_f1 = per_class
    macro_f1 = float(np.mean(list(per_class.values())))

    # Rep count MAE
    counter = RepCounter(
        min_concentric_frames=min_concentric_frames,
        min_eccentric_frames=min_eccentric_frames,
        min_idle_frames=min_idle_frames,
        fps=fps,
    )

    true_rep_counts = []
    pred_rep_counts = []

    for t_seq, p_seq in zip(true_labels, pred_labels):
        true_result = counter.get_results(t_seq)
        pred_result = counter.get_results(p_seq)
        true_rep_counts.append(true_result.rep_count)
        pred_rep_counts.append(pred_result.rep_count)

    rep_mae = float(np.mean(np.abs(np.array(true_rep_counts) - np.array(pred_rep_counts))))

    return {
        "per_class_f1": per_class_f1,
        "macro_f1": macro_f1,
        "rep_count_mae": rep_mae,
        "true_rep_counts": true_rep_counts,
        "pred_rep_counts": pred_rep_counts,
    }


# Unit-testable synthetic label sequences for testing the state machine
def generate_synthetic_label_sequence(
    num_reps: int,
    fps: int = 15,
    concentric_frames_range: tuple[int, int] = (5, 12),
    eccentric_frames_range: tuple[int, int] = (6, 15),
    idle_frames_range: tuple[int, int] = (8, 30),
    noise_prob: float = 0.05,
) -> list[int]:
    """
    Generate a synthetic phase label sequence for testing.

    Args:
        num_reps: Number of repetitions
        fps: Frames per second
        concentric_frames_range: Range of frames for concentric phase
        eccentric_frames_range: Range of frames for eccentric phase
        idle_frames_range: Range of frames for idle between reps
        noise_prob: Probability of random label flip (simulating model errors)

    Returns:
        List of phase labels (0=idle, 1=concentric, 2=eccentric)
    """
    sequence = []

    for rep in range(num_reps):
        # Idle before rep (except first)
        if rep > 0:
            idle_frames = np.random.randint(*idle_frames_range)
            sequence.extend([Phase.IDLE] * idle_frames)

        # Concentric (pushing up)
        conc_frames = np.random.randint(*concentric_frames_range)
        sequence.extend([Phase.CONCENTRIC] * conc_frames)

        # Eccentric (lowering down)
        ecc_frames = np.random.randint(*eccentric_frames_range)
        sequence.extend([Phase.ECCENTRIC] * ecc_frames)

    # Final idle
    final_idle = np.random.randint(*idle_frames_range)
    sequence.extend([Phase.IDLE] * final_idle)

    # Add noise: uniformly random label flips (simulating model errors).
    # `int(...)` normalizes everything to plain Python ints so the returned
    # list matches the `list[int]` contract (np.random.choice yields np.int64).
    if noise_prob > 0:
        sequence = [
            int(np.random.choice([0, 1, 2])) if np.random.random() < noise_prob
            else int(label)
            for label in sequence
        ]

    return sequence
