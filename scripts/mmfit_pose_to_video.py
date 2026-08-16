#!/usr/bin/env python3
"""
MM-Fit real-human calibration bridge.

The synthetic training data for an exercise is sampled from *measured*
per-phase feature statistics. For push-up and squat we can do much better than
the synthetic render: MM-Fit (a public multimodal fitness dataset) contains
real, labeled exercise sessions from 21 subjects, with 2D pose trajectories
recorded at 60 Hz. This script:

  1. Loads every subject's 2D pose sequence (view 0).
  2. Cuts out the labelled windows for "pushups" and "squats".
  3. Renders each window as a stick-figure video. We do NOT need to know which
     joint is which: the joints are drawn as filled blobs joined by a
     minimum-spanning-tree skeleton, so the video shows a real human body
     performing the exercise. Our pixel features only care about where motion
     happens, so this is enough to capture real motion statistics.
  4. Runs the REAL serving feature extractor over each rendered window (the
     exact same `extract_features` the API uses), which resamples to the
     serving target FPS.
  5. Derives per-frame phase labels from the body's vertical velocity:
     upward motion -> concentric, downward motion -> eccentric, near-zero
     speed between reps -> idle. (Even if a camera orientation inverts
     up/down, the model still learns a consistent concentric->eccentric->idle
     cycle, so rep counting is unaffected.)
  6. Aggregates the measured per-phase feature means/stds across every subject
     and writes data/phase_stats_mmfit_<exercise>.json, which
     generate_synthetic_data.py then uses as the source distribution.

Usage:
    python scripts/mmfit_pose_to_video.py --exercise pushup
    python scripts/mmfit_pose_to_video.py --exercise squat
    python scripts/mmfit_pose_to_video.py --all
    python scripts/mmfit_pose_to_video.py --all --limit-subjects 3   # quick dev
"""
import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MMFIT_DIR = Path("mm-fit")
STATS_OUT = "data/phase_stats_mmfit_{exercise}.json"
WINDOW_FPS = 60.0          # pose samples are written as a 60 fps video
CANVAS = (320, 480)        # width, height (matches make_demo_video)
JOINT_RADIUS = 11          # px
SKELETON_COLOR = (205, 205, 220)
JOINT_COLOR = (230, 235, 245)

# Each exercise is identified in MM-Fit by this label string.
_LABEL = {"pushup": "pushups", "squat": "squats"}


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def load_pose(subject_dir: Path) -> np.ndarray:
    """Return the 2D pose for a subject: shape (n_frames, 19)."""
    pose_file = next(subject_dir.glob("w*_pose_2d.npy"))
    pose = np.load(pose_file, allow_pickle=True)[0]  # view 0
    return pose


def load_windows(subject_dir: Path, exercise_label: str):
    """Yield (start_frame, end_frame, rep_count) for an exercise's labelled windows."""
    labels_file = next(subject_dir.glob("w*_labels.csv"))
    with open(labels_file, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 4 and row[3] == exercise_label:
                yield int(row[0]), int(row[1]), int(row[2])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _static_background(width, height, seed=7):
    """Reusable textured background (matches make_demo_video style)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), np.uint8)
    grad = np.linspace(40, 80, height, dtype=np.uint8)
    img[:] = grad[:, None, None]
    img = np.clip(
        img.astype(np.int16) + rng.integers(-4, 4, (height, width, 1)).astype(np.int16),
        0, 255,
    ).astype(np.uint8)
    return img


def _spanning_tree(points: np.ndarray):
    """Minimum-spanning-tree adjacency over joint positions (order-agnostic body)."""
    n = len(points)
    if n < 2:
        return []
    dist = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    in_tree = {0}
    edges = []
    while len(in_tree) < n:
        best = None
        for i in in_tree:
            for j in range(n):
                if j in in_tree:
                    continue
                if best is None or dist[i, j] < best[0]:
                    best = (dist[i, j], i, j)
        _, i, j = best
        edges.append((int(i), int(j)))
        in_tree.add(j)
    return edges


def render_pose_frame(joints: np.ndarray, canvas=(CANVAS[0], CANVAS[1])):
    """
    Draw one pose frame as blobs + MST skeleton.

    joints: (9, 2) pixel coordinates in the original video. The joint cloud is
    fitted into the canvas preserving aspect ratio, so body size and motion
    magnitude survive.
    """
    img = _static_background(canvas[0], canvas[1])
    w, h = canvas

    # Ignore zero/occluded joints when fitting the box.
    valid = joints[np.all(joints > 0, axis=1)]
    if len(valid) < 3:
        return img
    x0, y0 = valid.min(axis=0)
    x1, y1 = valid.max(axis=0)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)

    # Fit into 70% of the canvas with margin, preserving aspect ratio.
    scale = min(0.7 * w / bw, 0.7 * h / bh)
    ox = (w - bw * scale) / 2
    oy = (h - bh * scale) / 2

    pts = np.column_stack([(joints[:, 0] - x0) * scale + ox,
                           (joints[:, 1] - y0) * scale + oy])
    pts = pts.astype(np.int32)

    for i, j in _spanning_tree(pts):
        cv2.line(img, tuple(pts[i]), tuple(pts[j]), SKELETON_COLOR, 3)
    for px, py in pts:
        cv2.circle(img, (int(px), int(py)), JOINT_RADIUS, JOINT_COLOR, -1)
    return img


# ---------------------------------------------------------------------------
# Phase derivation from body velocity
# ---------------------------------------------------------------------------

def _moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def derive_phases(pose_segment: np.ndarray, up_is_concentric: bool = True) -> np.ndarray:
    """
    Phase label per frame from the body's vertical velocity.

    Uses the average y of the three most vertically-moving joints (robust to a
    single noisy tracker). Smoothed velocity sign -> phase; near-zero speed
    becomes idle.

    Returns: np.ndarray of ints in {0 idle, 1 concentric, 2 eccentric}.
    """
    joints = pose_segment[:, 1:]           # drop timestamp column
    joints = joints.reshape(joints.shape[0], -1, 2)  # (n, n_joints, 2)
    ys = joints[:, :, 1].astype(np.float64)
    ys[ys == 0] = np.nan                  # occluded joints

    # Vertically-moving joints: largest y variance (skip all-NaN columns).
    col_var = np.nanvar(ys, axis=0)
    top_idx = np.argsort(col_var)[-3:]
    body_y = np.nanmean(ys[:, top_idx], axis=1)
    body_y = np.nan_to_num(body_y, nan=np.nanmedian(body_y))

    vel = np.gradient(_moving_average(body_y, 5))
    speed_thresh = max(0.15 * np.nanstd(vel), 1e-3)

    phases = np.zeros(len(vel), dtype=np.int64)
    up = vel < -speed_thresh        # upward (smaller y) -> concentric
    down = vel > speed_thresh
    phases[up] = 1
    phases[down] = 2
    return phases


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def measure_phase_stats(exercise: str, limit_subjects: int | None = None,
                        max_windows: int | None = None) -> dict:
    """Render labelled windows for an exercise and measure per-phase stats."""
    from src.video.features import extract_features

    label = _LABEL[exercise]
    per_phase = {0: [], 1: [], 2: []}
    windows_used = 0

    subjects = sorted(MMFIT_DIR.glob("w*"))
    if limit_subjects:
        subjects = subjects[:limit_subjects]

    with tempfile.TemporaryDirectory(prefix="mmfit_") as tmp:
        for subject in subjects:
            try:
                pose = load_pose(subject)
            except Exception as e:
                print(f"  skip {subject.name}: {e}")
                continue
            ts = pose[:, 0].astype(np.int64)

            for start, end, _rep_count in load_windows(subject, label):
                if max_windows and windows_used >= max_windows:
                    break
                start = max(int(start), int(ts[0]))
                end = min(int(end), int(ts[-1]))
                seg = pose[start:end + 1]
                if len(seg) < 20:
                    continue

                # Render this window to a video at 60 fps, then extract the
                # exact features the serving path consumes.
                video_path = Path(tmp) / f"{subject.name}_{start}_{end}.mp4"
                writer = cv2.VideoWriter(
                    str(video_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    WINDOW_FPS,
                    CANVAS,
                )
                for joints in seg[:, 1:].reshape(len(seg), -1, 2):
                    writer.write(render_pose_frame(joints))
                writer.release()

                feats = extract_features(str(video_path),
                                         target_fps=15, max_seconds=30.0)
                T = feats.shape[0]

                # Phase labels at the 60 Hz source, then sampled to feature rows
                # exactly like make_demo_video does.
                phases = derive_phases(seg)
                src_fps = WINDOW_FPS
                labels = np.full(T, -1, dtype=np.int64)
                for t in range(T):
                    idx = min(int(t / 15.0 * src_fps), len(phases) - 1)
                    labels[t] = phases[idx]

                for ph in (0, 1, 2):
                    mask = labels == ph
                    if mask.sum() >= 5:
                        per_phase[ph].append(feats[mask])
                windows_used += 1

    # Aggregate means/stds across all windows (and subjects).
    stats = {}
    for ph, name in [(0, "idle"), (1, "concentric"), (2, "eccentric")]:
        arrays = per_phase[ph]
        if not arrays:
            print(f"  WARNING: no {name} frames measured for {exercise}")
            continue
        stacked = np.concatenate(arrays, axis=0)
        stats[name] = {
            "mean": stacked.mean(axis=0).tolist(),
            "std": stacked.std(axis=0).tolist(),
            "n": len(stacked),
            "source": "mmfit_real_humans",
            "n_windows": windows_used,
        }
        grid = stacked[:, :16].sum(1)
        print(f"{exercise:10s} {name:11s} n={len(stacked):5d} "
              f"grid_sum={grid.mean():.4f}±{grid.std():.4f} "
              f"cent={stats[name]['mean'][16]:.3f} "
              f"flowmag={stats[name]['mean'][17]:+.4f} "
              f"flowvert={stats[name]['mean'][18]:+.4f}")
    return stats


def write_stats_json(stats: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Phase stats written to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", choices=list(_LABEL.keys()), default="pushup")
    parser.add_argument("--all", action="store_true",
                        help="process every exercise in _LABEL")
    parser.add_argument("--limit-subjects", type=int, default=None,
                        help="only use the first N subjects (quick dev)")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="cap total windows used (quick dev)")
    args = parser.parse_args()

    exercises = list(_LABEL.keys()) if args.all else [args.exercise]
    for exercise in exercises:
        print(f"=== Measuring MM-Fit real-human stats: {exercise} ===")
        stats = measure_phase_stats(
            exercise,
            limit_subjects=args.limit_subjects,
            max_windows=args.max_windows,
        )
        write_stats_json(stats, STATS_OUT.format(exercise=exercise))


if __name__ == "__main__":
    main()
