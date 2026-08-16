#!/usr/bin/env python3
"""
Generate synthetic exercise demonstration videos + calibration statistics.

Renders a simple textured "body" performing an exercise on a textured static
background. Each exercise has its own motion signature (which shapes move, by
how much), so the per-phase feature statistics differ per exercise. The
statistics measured here (via the real serving feature-extractor path) are
written to JSON and used by generate_synthetic_data.py to train that
exercise's LSTM on the same distribution the API feeds it at inference.

Exercise convention (consistent across all exercises):
    concentric = upward/positive exertion motion
    eccentric  = downward/negative return motion
    idle       = rest between reps
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_STATIC_BG = None
_RNG = None


def _get_rng():
    global _RNG
    if _RNG is None:
        _RNG = np.random.default_rng(0)
    return _RNG


def _static_background(width=320, height=480):
    global _STATIC_BG
    if _STATIC_BG is None:
        img = np.zeros((height, width, 3), np.uint8)
        grad = np.linspace(50, 95, height, dtype=np.uint8)
        img[:] = grad[:, None, None]
        img = np.clip(
            img.astype(np.int16) + _get_rng().integers(-4, 4, (height, width, 1)).astype(np.int16),
            0, 255,
        ).astype(np.uint8)
        cv2.line(img, (0, height - 70), (width, height - 70), (200, 200, 210), 3)
        _STATIC_BG = img
    return _STATIC_BG


def _gradient_rect(base_h, base_w):
    """A (h, w, 3) gradient-shaded rectangle (darker top -> lighter bottom)."""
    grad = np.tile(np.linspace(130, 190, base_h, dtype=np.uint8), (base_w, 1)).T
    return np.dstack([grad] * 3)


# ---------------------------------------------------------------------------
# Exercise renderers. Each renders a single frame with the body at vertical
# offset pos_y (concentric moves pos_y negative/up, eccentric returns to 0).
# ---------------------------------------------------------------------------

def render_pushup(pos_y, width=320, height=480):
    """Body (torso + arms + head) bobbing up/down ~25px."""
    img = _static_background(width, height).copy()
    y = int(pos_y)
    torso = _gradient_rect(180, 40)  # (180, 40, 3)
    img[150 + y:330 + y, 140:180] = torso
    cv2.rectangle(img, (105, 170 + y), (140, 260 + y), (150, 160, 175), -1)
    cv2.rectangle(img, (180, 170 + y), (215, 260 + y), (150, 160, 175), -1)
    cv2.circle(img, (160, 115 + y), 30, (180, 190, 205), -1)
    return img


def render_squat(pos_y, width=320, height=480):
    """Tall torso + head moving up/down ~40px (bigger range than push-up)."""
    img = _static_background(width, height).copy()
    y = int(pos_y)
    torso = _gradient_rect(220, 70)  # taller torso
    img[140 + y:360 + y, 120:190] = torso
    cv2.circle(img, (155, 105 + y), 28, (180, 190, 205), -1)
    # legs stay planted (only torso/head move)
    cv2.rectangle(img, (130, 360), (155, 420), (90, 95, 105), -1)
    cv2.rectangle(img, (165, 360), (190, 420), (90, 95, 105), -1)
    return img


def render_bicep_curl(pos_y, width=320, height=480):
    """Forearms curling up/down ~30px; torso and upper arms static."""
    img = _static_background(width, height).copy()
    y = int(pos_y)
    # torso (static)
    torso = _gradient_rect(180, 55)
    img[150:330, 130:185] = torso
    cv2.circle(img, (160, 120), 28, (180, 190, 205), -1)
    # upper arms (static, vertical)
    cv2.rectangle(img, (110, 150), (140, 250), (150, 160, 175), -1)
    cv2.rectangle(img, (180, 150), (210, 250), (150, 160, 175), -1)
    # forearms (moving) - curl up rotates forearm upward
    cv2.rectangle(img, (105, 250 + y), (145, 300 + y), (170, 180, 195), -1)
    cv2.rectangle(img, (175, 250 + y), (215, 300 + y), (170, 180, 195), -1)
    return img


def render_jumping_jack(pos_y, width=320, height=480):
    """Whole body (torso + arms spread) jumping up/down ~50px."""
    img = _static_background(width, height).copy()
    y = int(pos_y)
    torso = _gradient_rect(150, 50)
    img[150 + y:300 + y, 130:180] = torso
    cv2.circle(img, (155, 115 + y), 26, (180, 190, 205), -1)
    # arms spread outward, move with body
    spread = 20 if y < 0 else 0  # arms wider when up
    cv2.rectangle(img, (90 - spread, 170 + y), (125 - spread, 210 + y), (150, 160, 175), -1)
    cv2.rectangle(img, (185 + spread, 170 + y), (220 + spread, 210 + y), (150, 160, 175), -1)
    return img


EXERCISE_RENDERERS = {
    "pushup": render_pushup,
    "squat": render_squat,
    "bicep_curl": render_bicep_curl,
    "jumping_jack": render_jumping_jack,
}

EXERCISE_AMP = {
    "pushup": 25,
    "squat": 40,
    "bicep_curl": 30,
    "jumping_jack": 50,
}


# ---------------------------------------------------------------------------
# Timeline + rendering
# ---------------------------------------------------------------------------

def build_timeline(num_reps=3, fps=30, amp=25, idle_between=0.8,
                   con_dur=0.8, ecc_dur=0.8, lead_in=0.6, lead_out=0.6):
    """Return (ys, phase_names) sampled at `fps`.

    Concentric moves the body to -amp (up); eccentric returns to 0 (down).
    """
    timeline = [("idle", lead_in, 0)]
    for _ in range(num_reps):
        timeline += [("con", con_dur, -amp), ("ecc", ecc_dur, 0)]
        if _ < num_reps - 1:
            timeline.append(("idle", idle_between, 0))
    timeline.append(("idle", lead_out, 0))

    ys, names = [], []
    for name, dur, y_end in timeline:
        y0 = ys[-1] if ys else 0
        ys.extend(np.linspace(y0, y_end, int(dur * fps)))
        names.extend([name] * int(dur * fps))
    return ys, names


def render_video(path, exercise="pushup", num_reps=3, fps=30, width=320, height=480):
    """Render an exercise demo video to `path`. Returns (frames, phase_names)."""
    global _last_exercise, _last_num_reps, _last_fps, _last_amp
    amp = EXERCISE_AMP[exercise]
    _last_exercise, _last_num_reps, _last_fps, _last_amp = exercise, num_reps, fps, amp

    render = EXERCISE_RENDERERS[exercise]
    ys, names = build_timeline(num_reps=num_reps, fps=fps, amp=amp)
    # Keep the static background for this exercise across all frames.
    global _STATIC_BG
    _STATIC_BG = None  # regenerate per exercise (different background seed is fine)
    frames = [render(y, width, height) for y in ys]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for f in frames:
        writer.write(f)
    writer.release()
    return frames, names


_last_exercise = "pushup"
_last_num_reps = 3
_last_fps = 30
_last_amp = 25


# ---------------------------------------------------------------------------
# Calibration: measure per-phase feature stats through the serving path
# ---------------------------------------------------------------------------

def measure_phase_stats(video_path, exercise="pushup", target_fps=15, max_seconds=30.0):
    """Compute per-phase feature means/stds through the real serving path.

    Runs the on-disk `extract_features` (which resamples to `target_fps`
    exactly as the API does) and aligns each feature row to the phase timeline
    by timestamp, so the measured statistics reflect the exact distribution
    the trained model receives at inference.
    """
    from src.video.features import extract_features

    feats = extract_features(str(video_path), target_fps=target_fps, max_seconds=max_seconds)
    T = feats.shape[0]

    amp = EXERCISE_AMP[exercise]
    _, names = build_timeline(num_reps=_last_num_reps, fps=_last_fps, amp=amp)
    total_frames = len(names)

    label_map = {"idle": 0, "con": 1, "ecc": 2}
    labels = np.full(T, -1, dtype=np.int64)
    for t in range(T):
        tm = t / target_fps
        idx = min(int(tm * _last_fps), total_frames - 1)
        labels[t] = label_map[names[idx]]

    stats = {}
    for ph, name in [(0, "idle"), (1, "concentric"), (2, "eccentric")]:
        mask = labels == ph
        if mask.sum() == 0:
            continue
        mean = feats[mask].mean(axis=0)
        std = feats[mask].std(axis=0)
        stats[name] = {"mean": mean, "std": std}
        grid_sum = feats[mask, :16].sum(1)
        print(
            f"{exercise:12s} {name:11s} n={mask.sum():3d} "
            f"grid_sum={grid_sum.mean():.4f}±{grid_sum.std():.4f} "
            f"cent={mean[16]:.3f}±{std[16]:.3f} "
            f"flowmag={mean[17]:+.4f} flowvert={mean[18]:+.4f}"
        )
    return stats


def write_stats_json(stats, path):
    out = {name: {"mean": v["mean"].tolist(), "std": v["std"].tolist()}
           for name, v in stats.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Phase stats written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render synthetic exercise demo videos")
    parser.add_argument("--exercise", default="pushup",
                        choices=list(EXERCISE_RENDERERS.keys()))
    parser.add_argument("--output", default=None,
                        help="output video path (default: demo_<exercise>.mp4)")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--measure", action="store_true",
                        help="also print per-phase feature stats (calibration)")
    parser.add_argument("--stats-out", default=None,
                        help="write measured per-phase stats to this JSON path")
    args = parser.parse_args()

    output = args.output or f"demo_{args.exercise}.mp4"
    frames, names = render_video(output, exercise=args.exercise, num_reps=args.reps)
    print(f"Rendered {len(frames)} frames ({args.exercise}) -> {output}")
    if args.measure:
        stats = measure_phase_stats(output, exercise=args.exercise)
        if args.stats_out:
            write_stats_json(stats, args.stats_out)
