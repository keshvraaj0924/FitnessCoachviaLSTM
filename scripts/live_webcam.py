#!/usr/bin/env python3
"""
Real-time streaming client for the /v1/stream WebSocket endpoint.

Streams live webcam frames (or replays a video file at its native rate) to the
running API and prints per-frame phase predictions plus a final summary.

Usage:
    python scripts/live_webcam.py --exercise pushup
    python scripts/live_webcam.py --exercise squat   --url ws://localhost:8000/v1/stream
    python scripts/live_webcam.py --exercise bicep_curl --file demo_bicep_curl.mp4
    python scripts/live_webcam.py --exercise jumping_jack --file demo_jumping_jack.mp4 --no-display

Start the API first:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
    (or: python -m src.serving.app)

Frame protocol (see src/serving/app.py::stream_video):
    1. {"type": "config", "exercise": ..., "source_fps": ...}
    2. server -> {"type": "ready", ...}
    3. binary JPEG frames (client -> server)
    4. server -> {"type": "frame", phase, confidence, rep_count, ...} per sample
    5. {"type": "summary"} (optional, any time) or on disconnect

Note: without a webcam the file-replay mode is what the demo pipeline uses.
"""
import argparse
import json
import time

import cv2
import numpy as np
from websockets.sync.client import connect


def _read_fps(cap, fallback=30.0):
    """Return the capture's real FPS, falling back to `fallback`."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or not np.isfinite(fps) or fps <= 0:
        return fallback
    return fps


def _next_frame_sleep(frame_interval):
    """Small helper to pace replay mode to the source frame rate."""
    return frame_interval


def stream_from_capture(uri, exercise, cap, source_fps, display):
    """
    Stream frames from an opened cv2.VideoCapture to the WebSocket.

    Paces to `source_fps` so the server's sub-sampler sees the same cadence a
    real camera would. Runs until the capture ends or Ctrl-C.
    """
    frame_interval = 1.0 / source_fps
    frame_idx = 0
    started = time.perf_counter()

    with connect(uri) as ws:
        # 1. Config handshake.
        ws.send(json.dumps({"type": "config", "exercise": exercise,
                            "source_fps": source_fps}))
        ready = json.loads(ws.recv())
        if ready.get("type") != "ready":
            print(f"[!] Server did not accept: {ready}")
            return
        print(f"[ready] exercise={ready.get('exercise')} "
              f"target_fps={ready.get('target_fps')}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[end] capture finished")
                break

            # Encode as JPEG (same format the server's cv2.imdecode expects).
            ok, buf = cv2.imencode(".jpg", frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            ws.send(buf.tobytes())

            # Drain any pending per-frame messages WITHOUT blocking the send
            # cadence. The server only replies on *sampled* frames (e.g. every
            # 2nd frame at 30fps -> 15fps), so a blocking recv here would stall
            # the client ~1s per unsampled frame and it would fall behind a
            # real webcam. A non-blocking read keeps sends at source_fps.
            while True:
                try:
                    msg = ws.recv(timeout=0.0)
                except TimeoutError:
                    break
                except Exception:
                    break
                if not msg:
                    break
                data = json.loads(msg)
                if data.get("type") == "frame":
                    print(
                        f"  t={data['t_s']:6.2f}s  "
                        f"{data['phase']:10s}  conf={data['confidence']:.2f}  "
                        f"reps={data['rep_count']}"
                    )
                    if display:
                        cv2.putText(frame, f"phase={data['phase']} "
                                   f"reps={data['rep_count']}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                   (0, 255, 0), 2)
                        cv2.imshow("live_webcam", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                elif data.get("type") == "summary":
                    print(f"[summary] {json.dumps(data, indent=2)}")

            # Pace to source rate.
            elapsed = time.perf_counter() - started
            target = frame_idx * frame_interval
            frame_idx += 1
            if target > elapsed:
                time.sleep(target - elapsed)

        # 2. Ask for the final summary, print it.
        ws.send(json.dumps({"type": "summary"}))
        summary = json.loads(ws.recv(timeout=5.0))
        print(f"[final summary] {json.dumps(summary, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://localhost:8000/v1/stream",
                        help="WebSocket URL of the /v1/stream endpoint")
    parser.add_argument("--exercise", default="pushup",
                        help="exercise id (pushup, squat, bicep_curl, jumping_jack)")
    parser.add_argument("--file", default=None,
                        help="replay this video file instead of the webcam")
    parser.add_argument("--no-display", action="store_true",
                        help="disable the live preview window")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.file or 0)
    if not cap.isOpened():
        print(f"[!] Could not open capture source: {args.file or 'webcam(0)'}")
        return

    source_fps = _read_fps(cap)
    display = args.file is None and not args.no_display

    try:
        stream_from_capture(args.url, args.exercise, cap, source_fps, display)
    except KeyboardInterrupt:
        print("\n[ctrl-c] stopping")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
