"""Debug segmentation: show motion energy and model scores per frame."""
import numpy as np
import sys
sys.path.insert(0, r"D:\AerioneBharat")

from src.serving.registry import ModelRegistry
from src.video.features import extract_features

# Load features from your video
features, _ = extract_features(r"D:\AerioneBharat\WIN_20260816_12_54_25_Pro.mp4", target_fps=15)
print(f"Video: {features.shape[0]} frames, fps=15")
print(f"Duration: {features.shape[0]/15:.1f}s")

# Motion energy per frame
motion = features[:, :16].sum(axis=1)
print(f"\nMotion energy: min={motion.min():.4f}, max={motion.max():.4f}, mean={motion.mean():.4f}")
print(f"Active frames (energy > 0.02): {(motion > 0.02).sum()}/{len(motion)}")

# Show motion energy timeline
print("\nMotion timeline (every 15 frames = 1 second):")
for t in range(0, len(motion), 15):
    chunk = motion[t:t+15]
    bars = "".join("█" if v > 0.02 else "░" for v in chunk)
    print(f"  {t:3d}-{t+len(chunk):3d}s: {bars}")

# Run all 4 models and show per-frame winner
registry = ModelRegistry()
registry.load(device="cpu")

candidates = ["pushup", "squat", "bicep_curl", "jumping_jack"]
all_probs = {}
for ex_id in candidates:
    logits, _ = registry.predict(features, ex_id)
    lg = np.asarray(logits[0], dtype=np.float64)
    lg = lg - lg.max(axis=1, keepdims=True)
    exps = np.exp(lg)
    all_probs[ex_id] = exps / exps.sum(axis=1, keepdims=True)

print("\nPer-frame winner (every 15 frames):")
for t in range(0, len(features), 15):
    winners = []
    for ex_id in candidates:
        probs = all_probs[ex_id][t]
        active = float(max(probs[1], probs[2]))
        winners.append(f"{ex_id[:4]}:{active:.2f}")
    print(f"  {t:3d}s: {', '.join(winners)}")

# Find transition candidates
print("\nLooking for transitions (frames where certainty < 0.12):")
certainty = np.zeros(len(features))
for t in range(len(features)):
    best = 0.0
    for ex_id in candidates:
        probs = all_probs[ex_id][t]
        active = float(max(probs[1], probs[2]))
        if active > best:
            best = active
    certainty[t] = best

valleys = np.where(certainty < 0.12)[0]
if len(valleys) > 0:
    # Group consecutive valley frames
    groups = []
    start = valleys[0]
    prev = valleys[0]
    for v in valleys[1:]:
        if v == prev + 1:
            prev = v
        else:
            groups.append((start, prev))
            start = v
            prev = v
    groups.append((start, prev))

    for gs, ge in groups:
        dur = (ge - gs + 1) / 15
        print(f"  Valley at {gs/15:.1f}s-{ge/15:.1f}s ({dur:.1f}s, {ge-gs+1} frames)")
else:
    print("  No valleys found — all frames have certainty >= 0.12")
    print(f"  Min certainty: {certainty.min():.4f} at frame {np.argmin(certainty)}")
