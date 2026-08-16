"""
Synthetic Data Generation for Exercise LSTM Training

Generates synthetic feature sequences with known phase labels for training.

Since we don't have real labeled video, we simulate the feature patterns that
would be observed during each exercise. Crucially, the per-phase feature
statistics are MEASURED, not hand-tuned: scripts/make_demo_video.py renders a
synthetic clip of each exercise and runs the real feature extractor
(src/video/features.py) over it, producing data/phase_stats_<exercise>.json.
This generator samples from those measured means/stds, so each exercise model
trains on the same distribution the serving layer feeds it at inference.

Run once per exercise:
    python generate_synthetic_data.py --exercise pushup
    python generate_synthetic_data.py --exercise squat
    python generate_synthetic_data.py --exercise bicep_curl
    python generate_synthetic_data.py --exercise jumping_jack

See NOTES.md for the honest caveats.
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (kept for API compatibility with older checkpoints)

from src.model.reps import Phase, generate_synthetic_label_sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_DIM = 19  # 16 grid cells + centroid + flow_mag + flow_vert

# Fallback phase statistics used only when a measured stats JSON is absent.
# The measured values are preferred (see module docstring).
_FALLBACK_PHASE_FEATURE_MEANS = {
    Phase.IDLE: np.array(
        [*([0.001] * 16), 0.45, 0.0, 0.0], dtype=np.float32
    ),
    Phase.CONCENTRIC: np.array(
        [0.0025, 0.003, 0.003, 0.0025,
         0.004, 0.005, 0.005, 0.004,
         0.002, 0.0025, 0.0025, 0.002,
         0.001, 0.001, 0.001, 0.001,
         0.34, 0.0, -0.004], dtype=np.float32,
    ),
    Phase.ECCENTRIC: np.array(
        [0.0025, 0.0035, 0.0035, 0.0025,
         0.0045, 0.0055, 0.0055, 0.0045,
         0.002, 0.003, 0.003, 0.002,
         0.001, 0.001, 0.001, 0.001,
         0.36, 0.0, 0.004], dtype=np.float32,
    ),
}

_FALLBACK_PHASE_FEATURE_STDS = {
    Phase.IDLE: np.array(
        [*([0.0003] * 16), 0.08, 0.001, 0.001], dtype=np.float32
    ),
    Phase.CONCENTRIC: np.array(
        [*([0.001] * 16), 0.05, 0.001, 0.0015], dtype=np.float32
    ),
    Phase.ECCENTRIC: np.array(
        [*([0.001] * 16), 0.05, 0.001, 0.002], dtype=np.float32
    ),
}

# How much we widen the measured std when sampling. The measured variance comes
# from a single deterministic render; widening gives the model robustness to
# subjects/tempos that differ from the calibration clip.
STD_SCALE = 1.8

_EXERCISE_STATS_FILE = {
    "pushup": "data/phase_stats_pushup.json",
    "squat": "data/phase_stats_squat.json",
    "bicep_curl": "data/phase_stats_bicep_curl.json",
    "jumping_jack": "data/phase_stats_jumping_jack.json",
}

# Real-human calibration produced from MM-Fit (scripts/mmfit_pose_to_video.py).
# Push-up and squat have real, labelled human sessions in MM-Fit, so when these
# files exist they are STRICTLY PREFERRED over the synthetic render stats: the
# model then learns the feature distribution of actual humans performing the
# exercise, which transfers far better to the live webcam. The other exercises
# have no MM-Fit coverage and remain calibrated on the synthetic render.
_EXERCISE_MMFIT_STATS_FILE = {
    "pushup": "data/phase_stats_mmfit_pushup.json",
    "squat": "data/phase_stats_mmfit_squat.json",
}

_EXERCISE_DATA_PREFIX = {
    "pushup": "data/train_synthetic",
    "squat": "data/train_squat",
    "bicep_curl": "data/train_bicep_curl",
    "jumping_jack": "data/train_jumping_jack",
}


def _read_phase_stats_json(path: Path):
    """Parse a phase-stats JSON into (means, stds) dicts, or (None, None)."""
    if not path.exists():
        return None, None
    with open(path) as f:
        raw = json.load(f)
    name_to_phase = {"idle": Phase.IDLE, "concentric": Phase.CONCENTRIC,
                     "eccentric": Phase.ECCENTRIC}
    means = {
        name_to_phase[name]: np.asarray(v["mean"], dtype=np.float32)
        for name, v in raw.items()
    }
    stds = {
        name_to_phase[name]: np.asarray(v["std"], dtype=np.float32)
        for name, v in raw.items()
    }
    if all(p in means for p in Phase) and all(p in stds for p in Phase):
        return means, stds
    return None, None


def load_phase_stats(exercise: str = "pushup", stats_path: str | None = None):
    """Load the single preferred measured per-phase stats for an exercise.

    Preference order:
      1. Explicit `stats_path` (if given).
      2. MM-Fit real-human calibration (data/phase_stats_mmfit_<exercise>.json)
         for pushup/squat — real humans transfer far better to live webcam.
      3. Synthetic render calibration (data/phase_stats_<exercise>.json).
      4. Hard-coded fallback defaults.

    Returns (means: dict[Phase, np.ndarray], stds: dict[Phase, np.ndarray]).
    """
    sources = load_phase_stat_sources(exercise, stats_path)
    return sources[0][1], sources[0][2]


def load_phase_stat_sources(
    exercise: str = "pushup", stats_path: str | None = None
) -> list[tuple[str, dict, dict]]:
    """Load *all* available measured phase-stats sources for an exercise.

    Returns a list of ``(source_name, means, stds)`` tuples, most-preferred
    first. For pushup/squat this is the MM-Fit real-human calibration followed
    by the synthetic-render calibration; for the other exercises it is just the
    synthetic-render calibration (or the hard-coded fallback).

    The dataset generator mixes over every source so the model learns both
    distributions: real-human idle is noisy (people never freeze) while the
    synthetic render idle is near-zero. Training on *only* MM-Fit made the
    model see every low-motion frame as "real-human idle" and miss synthetic
    demos; training on *only* the render missed real webcam idle noise. Mixing
    covers both.
    """
    base = Path(__file__).resolve().parent
    candidates: list[tuple[str, Path]] = []
    if stats_path:
        candidates.append(("explicit", Path(stats_path)))
    mmfit_file = _EXERCISE_MMFIT_STATS_FILE.get(exercise)
    if mmfit_file:
        candidates.append(("MM-Fit real-human", base / mmfit_file))
    candidates.append(("synthetic-render", base / _EXERCISE_STATS_FILE[exercise]))

    sources: list[tuple[str, dict, dict]] = []
    for source, path in candidates:
        means, stds = _read_phase_stats_json(path)
        if means:
            logger.info(f"Loaded {source} phase stats for {exercise} from {path}")
            sources.append((source, means, stds))
    if not sources:
        logger.warning(f"No usable phase stats for {exercise}; using fallback defaults")
        sources.append(("fallback", _FALLBACK_PHASE_FEATURE_MEANS, _FALLBACK_PHASE_FEATURE_STDS))
    return sources


def generate_synthetic_sequence(
    num_reps: int,
    fps: int = 15,
    noise_level: float = 1.0,
    phase_means=None,
    phase_stds=None,
) -> tuple[np.ndarray, list[int]]:
    """
    Generate a synthetic feature sequence with labels.

    Args:
        num_reps: Number of repetitions
        fps: Frames per second
        noise_level: Multiplier for feature noise
        phase_means / phase_stds: Optional measured per-phase statistics.
                                  Defaults to the push-up calibration.

    Returns:
        (features: (T, 19), labels: List[int])
    """
    if phase_means is None or phase_stds is None:
        phase_means, phase_stds = load_phase_stats("pushup")

    labels = generate_synthetic_label_sequence(
        num_reps=num_reps,
        fps=fps,
        concentric_frames_range=(5, 10),   # 0.33-0.67s
        eccentric_frames_range=(6, 12),    # 0.4-0.8s
        idle_frames_range=(5, 15),         # 0.33-1.0s
        noise_prob=0.02,                   # Small label noise
    )

    # Per-sequence random offsets: subjects differ in size/contrast, so add a
    # small global shift per sequence. This is cheap data augmentation and
    # keeps the model from memorizing the exact calibration means.
    global_shift = np.random.normal(0.0, 0.0008, 16).astype(np.float32)

    features = []
    for label in labels:
        phase = Phase(label)
        mean = phase_means[phase]
        std = phase_stds[phase] * STD_SCALE * noise_level

        feat = np.random.normal(mean, std).astype(np.float32)
        feat[:16] = feat[:16] + global_shift

        # Clip to valid ranges (matches the feature extractor's normalization)
        feat[:16] = np.clip(feat[:16], 0.0, 1.0)   # Grid energies
        feat[16] = np.clip(feat[16], 0.0, 1.0)     # Centroid
        feat[17] = np.clip(feat[17], -2.0, 2.0)    # Flow magnitude
        feat[18] = np.clip(feat[18], -1.0, 1.0)    # Flow vertical

        features.append(feat)

    return np.stack(features), labels


def generate_dataset(
    num_sequences: int,
    min_reps: int = 1,
    max_reps: int = 8,
    fps: int = 15,
    noise_level: float = 1.0,
    phase_means=None,
    phase_stds=None,
    phase_stat_sources: list[tuple[str, dict, dict]] | None = None,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """
    Generate a dataset of synthetic sequences.

    Args:
        num_sequences: Number of sequences to generate
        min_reps: Minimum reps per sequence
        max_reps: Maximum reps per sequence
        fps: Frames per second
        noise_level: Feature noise multiplier
        phase_means / phase_stds: Optional single measured per-phase stats
                                  (ignored if `phase_stat_sources` is given).
        phase_stat_sources: Optional list of ``(name, means, stds)``. When
                            given, each sequence randomly samples one source,
                            so a model trained on pushup/squat sees BOTH the
                            MM-Fit real-human and synthetic-render domains.

    Returns:
        (features_list, labels_list)
    """
    sources = phase_stat_sources or (
        [("single", phase_means, phase_stds)] if phase_means else None
    )

    features_list = []
    labels_list = []

    for i in range(num_sequences):
        num_reps = np.random.randint(min_reps, max_reps + 1)
        if sources:
            _, means, stds = sources[np.random.randint(len(sources))]
        else:
            means, stds = load_phase_stats("pushup")
        feats, labels = generate_synthetic_sequence(
            num_reps, fps, noise_level, means, stds
        )
        features_list.append(feats)
        labels_list.append(labels)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_sequences} sequences")

    return features_list, labels_list


def save_dataset(
    features_list: list[np.ndarray],
    labels_list: list[list[int]],
    path: str,
    phase_means=None,
    phase_stds=None,
):
    """Save dataset to disk.

    `phase_means`/`phase_stds` are only metadata (for provenance in the pickle);
    when omitted they default to the pushup calibration. The generated features
    themselves already embed the (possibly mixed) source statistics.
    """
    if phase_means is None:
        phase_means, phase_stds = load_phase_stats("pushup")
    data = {
        'features': features_list,
        'labels': labels_list,
        'feature_dim': FEATURE_DIM,
        'phase_means': {int(k): v for k, v in phase_means.items()},
        'phase_stds': {int(k): v for k, v in phase_stds.items()},
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved dataset to {path}")


def load_dataset(path: str) -> tuple[list[np.ndarray], list[list[int]]]:
    """Load dataset from disk."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels']


def create_train_val_test_split(
    features_list: list[np.ndarray],
    labels_list: list[list[int]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[
    tuple[list[np.ndarray], list[list[int]]],  # train
    tuple[list[np.ndarray], list[list[int]]],  # val
    tuple[list[np.ndarray], list[list[int]]],  # test
]:
    """Split dataset by video (sequence), not by frame."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    np.random.seed(seed)
    indices = np.random.permutation(len(features_list))

    n_train = int(len(features_list) * train_ratio)
    n_val = int(len(features_list) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_data = ([features_list[i] for i in train_idx], [labels_list[i] for i in train_idx])
    val_data = ([features_list[i] for i in val_idx], [labels_list[i] for i in val_idx])
    test_data = ([features_list[i] for i in test_idx], [labels_list[i] for i in test_idx])

    logger.info(f"Split: train={len(train_data[0])}, val={len(val_data[0])}, test={len(test_data[0])}")

    return train_data, val_data, test_data


def generate_all_exercises():
    """Generate train/val/test splits for every supported exercise."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    for exercise, prefix in _EXERCISE_DATA_PREFIX.items():
        np.random.seed(42)  # deterministic per exercise
        sources = load_phase_stat_sources(exercise)
        source_names = ", ".join(s[0] for s in sources)

        logger.info(f"Generating dataset for {exercise} from sources: {source_names}")
        train_feats, train_labels = generate_dataset(
            600, min_reps=1, max_reps=8, noise_level=1.0, phase_stat_sources=sources,
        )
        save_dataset(train_feats, train_labels, f"{prefix}.pkl")

        val_feats, val_labels = generate_dataset(
            100, min_reps=1, max_reps=8, noise_level=1.0, phase_stat_sources=sources,
        )
        save_dataset(val_feats, val_labels, f"{prefix.replace('train', 'val')}.pkl")

        test_feats, test_labels = generate_dataset(
            100, min_reps=1, max_reps=8, noise_level=1.0, phase_stat_sources=sources,
        )
        save_dataset(test_feats, test_labels, f"{prefix.replace('train', 'test')}.pkl")

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic exercise datasets")
    parser.add_argument("--exercise", default="pushup",
                        choices=list(_EXERCISE_DATA_PREFIX.keys()),
                        help="which exercise's dataset to generate")
    parser.add_argument("--all", action="store_true",
                        help="generate datasets for all supported exercises")
    args = parser.parse_args()

    if args.all:
        generate_all_exercises()
    else:
        np.random.seed(42)
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        sources = load_phase_stat_sources(args.exercise)
        prefix = _EXERCISE_DATA_PREFIX[args.exercise]

        train_feats, train_labels = generate_dataset(
            600, min_reps=1, max_reps=8, noise_level=1.0, phase_stat_sources=sources,
        )
        save_dataset(train_feats, train_labels, f"{prefix}.pkl")

        val_feats, val_labels = generate_dataset(
            100, min_reps=1, max_reps=8, noise_level=1.0, phase_stat_sources=sources,
        )
        save_dataset(val_feats, val_labels, f"{prefix.replace('train', 'val')}.pkl")

        test_feats, test_labels = generate_dataset(
            100, min_reps=1, max_reps=8, noise_level=1.0, phase_stat_sources=sources,
        )
        save_dataset(test_feats, test_labels, f"{prefix.replace('train', 'test')}.pkl")

        logger.info("Done!")
