"""
Model Registry - Thread-safe model loading and management.

Loads every exercise model once at startup, runs a warmup pass, and exposes
thread-safe inference. Each exercise has its own LSTM checkpoint; the
registry keeps a dict keyed by exercise id.
"""
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from src.llm.coach import CoachFeedback, RepStats, RepTiming, summarize
from src.model.lstm import PushupLSTM, create_model, load_checkpoint
from src.model.reps import RepCountResult, count_reps_from_logits
from src.serving.settings import settings
from src.video.features import extract_features

logger = logging.getLogger(__name__)


class ExerciseModel:
    """One loaded model + its metadata."""

    def __init__(self, exercise_id: str, checkpoint_path: Path):
        self.exercise_id = exercise_id
        self.checkpoint_path = checkpoint_path
        self.model: PushupLSTM | None = None
        self.checkpoint_hash: str = ""
        self.model_config: dict[str, Any] = {}
        self.ready: bool = False


class ModelRegistry:
    """
    Thread-safe singleton for model management.
    Loads all exercise models once, runs warmup, provides inference methods.
    """

    _instance: Optional['ModelRegistry'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._device: str = "cpu"
        self._ready: bool = False
        self._load_lock = threading.Lock()
        self._models: dict[str, ExerciseModel] = {}

        self._initialized = True
        logger.info("ModelRegistry initialized")

    def load(self, device: str = "cpu", exercises: list | None = None) -> bool:
        """
        Load all exercise models from checkpoints. Thread-safe.

        Args:
            device: Device to load models on
            exercises: Optional list of ExerciseSpec (defaults to settings.exercises)

        Returns:
            True if at least the default models loaded, False otherwise
        """
        with self._load_lock:
            if self._ready and self._models:
                logger.info("Models already loaded")
                return True

            self._device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
            exercises = exercises or settings.exercises

            loaded_any = False
            for spec in exercises:
                em = ExerciseModel(spec.id, spec.checkpoint)
                ok = self._load_exercise(em)
                if ok:
                    loaded_any = True
                self._models[spec.id] = em

            if loaded_any:
                self._ready = True
                logger.info(f"ModelRegistry loaded {sum(m.ready for m in self._models.values())}/{len(self._models)} exercise models")
                return True

            self._ready = False
            return False

    def _load_exercise(self, em: ExerciseModel) -> bool:
        """Load a single exercise model + checkpoint hash + warmup."""
        path = em.checkpoint_path
        if not path.exists():
            logger.error(f"Checkpoint not found for {em.exercise_id}: {path}")
            return False

        try:
            with open(path, 'rb') as f:
                em.checkpoint_hash = hashlib.sha256(f.read()).hexdigest()[:16]

            em.model = create_model(
                input_dim=settings.feature_dim,
                hidden_size=settings.lstm_hidden_size,
                num_layers=settings.lstm_num_layers,
                num_classes=3,
                dropout=settings.lstm_dropout,
                device=self._device,
                bidirectional=settings.lstm_bidirectional,
            )

            checkpoint = load_checkpoint(str(path), em.model, self._device)
            em.model_config = checkpoint.get('config', {})
            em.model.eval()

            if self._warmup(em.model):
                em.ready = True
                logger.info(f"Loaded {em.exercise_id} from {path} (hash: {em.checkpoint_hash})")
                return True
            return False

        except Exception:
            logger.exception(f"Failed to load {em.exercise_id} from {path}")
            em.model = None
            return False

    def _warmup(self, model: PushupLSTM) -> bool:
        """Run warmup inference on dummy data."""
        if model is None:
            return False
        try:
            dummy_input = torch.randn(1, 10, settings.feature_dim, device=self._device)
            dummy_lengths = torch.tensor([10], device=self._device)
            with torch.no_grad():
                _ = model(dummy_input, dummy_lengths)
            return True
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if at least one model is loaded and ready."""
        return self._ready and any(m.ready for m in self._models.values())

    # Streaming config helpers (avoid reaching into settings from stream.py)
    def _settings_target_fps(self) -> int:
        return settings.target_fps

    def _settings_min_concentric_frames(self) -> int:
        return settings.min_concentric_frames

    def _settings_min_eccentric_frames(self) -> int:
        return settings.min_eccentric_frames

    def _settings_min_idle_frames(self) -> int:
        return settings.min_idle_frames

    def _settings_stream_confidence_threshold(self) -> float:
        return settings.stream_confidence_threshold

    def get_model(self, exercise_id: str) -> ExerciseModel:
        """Get a loaded exercise model by id."""
        if exercise_id not in self._models:
            raise KeyError(f"Unknown exercise: {exercise_id}")
        em = self._models[exercise_id]
        if not em.ready:
            raise RuntimeError(f"Exercise {exercise_id} model not ready")
        return em

    def get_model_info(self, exercise_id: str = "pushup") -> dict[str, Any]:
        """Get model metadata for /v1/model endpoint.

        Works even before the exercise model is loaded (checkpoint_hash is then
        empty); the endpoint guards readiness separately.
        """
        em = self._models.get(exercise_id)
        checkpoint_hash = em.checkpoint_hash if em is not None else ""
        return {
            "exercise": exercise_id,
            "model_version": settings.model_version,
            "feature_config": {
                "target_fps": settings.target_fps,
                "max_seconds": settings.max_seconds,
                "feature_dim": settings.feature_dim,
                "working_resolution": [160, 160],
                "grid_size": [4, 4],
            },
            "checkpoint_hash": checkpoint_hash,
            "architecture": {
                "type": "BiLSTM" if settings.lstm_bidirectional else "LSTM (causal)",
                "input_dim": settings.feature_dim,
                "hidden_size": settings.lstm_hidden_size,
                "num_layers": settings.lstm_num_layers,
                "dropout": settings.lstm_dropout,
                "num_classes": 3,
                "classes": ["idle", "concentric", "eccentric"],
            },
            "rep_counter_config": {
                "min_concentric_frames": settings.min_concentric_frames,
                "min_eccentric_frames": settings.min_eccentric_frames,
                "min_idle_frames": settings.min_idle_frames,
            },
            "available_exercises": [e.id for e in settings.exercises],
        }

    def extract_features(self, video_path: str) -> tuple[np.ndarray, float]:
        """
        Extract features from video file.

        Returns:
            (features: (T, 19), decode_time_ms: float)
        """
        start = time.perf_counter()
        features = extract_features(
            video_path,
            target_fps=settings.target_fps,
            max_seconds=settings.max_seconds,
        )
        decode_time = (time.perf_counter() - start) * 1000
        return features, decode_time

    def predict(self, features: np.ndarray, exercise_id: str = "pushup") -> tuple[np.ndarray, float]:
        """
        Run LSTM inference on features for the given exercise.

        Returns:
            (logits: (1, T, 3), inference_time_ms: float)
        """
        em = self.get_model(exercise_id)
        start = time.perf_counter()

        input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(self._device)
        lengths = torch.tensor([features.shape[0]], device=self._device)

        with torch.no_grad():
            logits = em.model(input_tensor, lengths)

        inference_time = (time.perf_counter() - start) * 1000
        return logits.cpu().numpy(), inference_time

    def step(self, feature: np.ndarray, exercise_id: str,
             hidden: tuple[torch.Tensor, torch.Tensor] | None = None):
        """
        Streaming single-step inference for the given exercise.

        Args:
            feature: (19,) feature vector for one sampled frame.
            exercise_id: which exercise model to run.
            hidden: previous (h, c) hidden state from the prior step.

        Returns:
            (logits (3,), hidden, inference_ms)
        """
        em = self.get_model(exercise_id)
        start = time.perf_counter()
        x = torch.from_numpy(feature).float().view(1, 1, -1).to(self._device)
        with torch.no_grad():
            logits, hidden = em.model.step(x, hidden)
        logits = logits.cpu().numpy()[0, 0]  # (3,)
        ms = (time.perf_counter() - start) * 1000
        return logits, hidden, ms

    def count_reps(self, logits: np.ndarray) -> RepCountResult:
        """Count repetitions from logits."""
        return count_reps_from_logits(
            logits,
            min_concentric_frames=settings.min_concentric_frames,
            min_eccentric_frames=settings.min_eccentric_frames,
            min_idle_frames=settings.min_idle_frames,
            fps=settings.target_fps,
        )

    def compute_stats(self, rep_result: RepCountResult) -> RepStats:
        """Compute RepStats from RepCountResult."""
        if not rep_result.reps:
            return RepStats(
                rep_count=0,
                reps=[],
                avg_tempo_s=0.0,
                tempo_consistency=0.0,
                concentric_avg_s=0.0,
                eccentric_avg_s=0.0,
                confidence=rep_result.confidence,
            )

        durations = [r.duration() for r in rep_result.reps]
        avg_tempo = float(np.mean(durations))
        tempo_cv = float(np.std(durations) / avg_tempo) if avg_tempo > 0 else 0.0

        # Estimate concentric/eccentric from phase sequence
        phase_seq = rep_result.phase_sequence
        conc_frames = sum(1 for p in phase_seq if p == 1)
        ecc_frames = sum(1 for p in phase_seq if p == 2)
        total_active = conc_frames + ecc_frames

        if total_active > 0 and rep_result.rep_count > 0:
            concentric_avg = (conc_frames / total_active) * avg_tempo
            eccentric_avg = (ecc_frames / total_active) * avg_tempo
        else:
            concentric_avg = 0.0
            eccentric_avg = 0.0

        return RepStats(
            rep_count=rep_result.rep_count,
            reps=[RepTiming(start_s=r.start_s, end_s=r.end_s) for r in rep_result.reps],
            avg_tempo_s=avg_tempo,
            tempo_consistency=tempo_cv,
            concentric_avg_s=concentric_avg,
            eccentric_avg_s=eccentric_avg,
            confidence=rep_result.confidence,
        )

    def get_coaching(self, stats: RepStats, exercise_id: str = "pushup") -> tuple[CoachFeedback, float]:
        """Get coaching feedback from LLM (uses settings-resolved key/model)."""
        start = time.perf_counter()
        feedback = summarize(
            stats,
            timeout_s=settings.openai_timeout_s,
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            exercise=exercise_id,
        )
        llm_time = (time.perf_counter() - start) * 1000
        return feedback, llm_time


# Global registry instance
registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return registry
