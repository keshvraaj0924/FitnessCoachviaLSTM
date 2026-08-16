"""Test the HTTP live-session endpoints with a demo video (no camera needed)."""
import urllib.request
import urllib.error
import json
import os

BASE = "http://localhost:8000"

# 1. Start a live session (auto)
print("1. Starting live session (auto)...")
body = b"exercise=auto&source_fps=30.0"
req = urllib.request.Request(f"{BASE}/v1/live/start", data=body, method="POST")
req.add_header("Content-Type", "application/x-www-form-urlencoded")
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        start_result = json.loads(resp.read())
    print(f"   Session ID: {start_result.get('session_id')}")
    print(f"   Exercise:   {start_result.get('exercise')}")
    session_id = start_result.get("session_id")
except urllib.error.HTTPError as e:
    print(f"   ERROR {e.code}: {e.read().decode()[:200]}")
    exit(1)

if not session_id:
    print("No session_id returned")
    exit(1)

# 2. Send frames from demo_pushup.mp4
print("\n2. Sending frames from demo_pushup.mp4...")
import cv2
import numpy as np

cap = cv2.VideoCapture(r"D:\AerioneBharat\demo_pushup.mp4")
frame_idx = 0
rep_counts = []
for _ in range(120):  # send up to 120 frames (~4 seconds)
    ret, frame = cap.read()
    if not ret:
        break
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        continue

    req = urllib.request.Request(
        f"{BASE}/v1/live/{session_id}/frame",
        data=jpeg.tobytes(),
        method="POST",
    )
    req.add_header("Content-Type", "image/jpeg")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            frame_result = json.loads(resp.read())
        if frame_idx % 30 == 0:
            print(f"   Frame {frame_idx}: reps={frame_result.get('rep_count')} "
                  f"phase={frame_result.get('phase')}")
            rep_counts.append(frame_result.get("rep_count"))
    except urllib.error.HTTPError as e:
        print(f"   Frame {frame_idx} error: {e.code}")
    frame_idx += 1

cap.release()
print(f"   Total frames sent: {frame_idx}")
print(f"   Max reps seen:     {max(rep_counts) if rep_counts else 0}")

# 3. Stop session and get final result
print("\n3. Stopping session...")
req = urllib.request.Request(f"{BASE}/v1/live/{session_id}/stop", data=b"", method="POST")
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        final = json.loads(resp.read())
    print(f"\nFinal result:")
    print(f"  exercise:   {final.get('exercise')}")
    print(f"  rep_count:  {final.get('rep_count')}")
    reps = final.get("reps", [])
    print(f"  reps found: {len(reps)}")
    for i, r in enumerate(reps, 1):
        print(f"    Rep {i}: {r['start_s']}s - {r['end_s']}s")
    conf = final.get("per_class_confidence", {})
    print(f"  confidence: idle={conf.get('idle', 0):.2f} "
          f"concentric={conf.get('concentric', 0):.2f} "
          f"eccentric={conf.get('eccentric', 0):.2f}")
    coach = final.get("coaching_feedback", {})
    print(f"  coach:      {coach.get('summary', 'N/A')[:80]}")
    print(f"  latency_ms: {final.get('latency_ms')}")
    print("\nSUCCESS: Live HTTP session works!")
except urllib.error.HTTPError as e:
    print(f"ERROR {e.code}: {e.read().decode()[:200]}")
