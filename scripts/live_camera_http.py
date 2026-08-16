"""Live camera test using the HTTP live-session endpoints.

Usage:
    python scripts/live_camera_http.py --exercise pushup
    python scripts/live_camera_http.py --exercise auto --source-fps 30

Press 'q' in the OpenCV window to stop and get the final result.
"""
import argparse
import concurrent.futures
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

import cv2

DEFAULT_BASE_URL = "http://localhost:8000"

# Pool of background workers — each frame is POSTed in its own thread
# so the camera loop never blocks on a round-trip.
_FRAME_WORKERS = 4


def _post_json(url: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:200]}


def _post_raw(url: str, raw_bytes: bytes) -> dict:
    req = urllib.request.Request(url, data=raw_bytes, method="POST")
    req.add_header("Content-Type", "image/jpeg")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:200]}


def start_session(base_url: str, exercise: str, source_fps: float) -> str | None:
    """Start a live session, return session_id or None on failure."""
    body = (
        f"exercise={urllib.parse.quote(exercise)}"
        f"&source_fps={source_fps}"
    ).encode()
    req = urllib.request.Request(f"{base_url}/v1/live/start", data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
        if "session_id" in result:
            print(f"Session started: {result['session_id']} (exercise={result['exercise']})")
            return result["session_id"]
        print(f"Failed to start session: {result}")
        return None
    except urllib.error.HTTPError as e:
        print(f"Start error: HTTP {e.code}: {e.read().decode()[:200]}")
        return None


def stop_session(base_url: str, session_id: str) -> dict:
    """Stop session and return the final AnalyzeResponse."""
    try:
        req = urllib.request.Request(
            f"{base_url}/v1/live/{session_id}/stop",
            data=b"",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:200]}


def main():
    parser = argparse.ArgumentParser(description="Live camera rep counter (HTTP)")
    parser.add_argument("--exercise", default="pushup", help="Exercise id (or 'auto')")
    parser.add_argument("--source-fps", type=float, default=30.0, help="Camera FPS")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Server base URL")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # Start session
    session_id = start_session(base_url, args.exercise, args.source_fps)
    if session_id is None:
        sys.exit(1)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        stop_session(base_url, session_id)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    window = f"Live {args.exercise} — press q to stop"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Thread pool for concurrent frame uploads — camera never waits for
    # the server round-trip.  Results are collected as futures complete.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=_FRAME_WORKERS)
    pending: list[concurrent.futures.Future] = []

    last_rep_count = 0
    last_phase = "idle"
    last_conf = 0.0
    frames_sent = 0

    def _submit_frame(jpeg_bytes: bytes):
        """Submit a frame for async upload, update display state on completion."""
        fut = pool.submit(_post_raw,
                          f"{base_url}/v1/live/{session_id}/frame",
                          jpeg_bytes)
        pending.append(fut)

    def _drain_results():
        """Collect any completed futures and update display state."""
        nonlocal last_rep_count, last_phase, last_conf
        still_pending = []
        for fut in pending:
            if fut.done():
                try:
                    r = fut.result()
                    if r and "rep_count" in r:
                        last_rep_count = r["rep_count"]
                        last_phase = r.get("phase", "?")
                        last_conf = r.get("confidence", 0.0)
                except Exception:
                    pass
            else:
                still_pending.append(fut)
        pending[:] = still_pending

    print("\nCamera active. Press 'q' to stop and get results.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            # Encode to JPEG
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue

            # Fire-and-forget: submit frame to background pool
            _submit_frame(jpeg.tobytes())
            frames_sent += 1

            # Drain any completed results (non-blocking)
            _drain_results()

            # Overlay status on frame
            display = frame.copy()
            cv2.putText(display, f"Reps: {last_rep_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(display, f"Phase: {last_phase}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display, f"Conf: {last_conf:.2f}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(display, f"Sent: {frames_sent}", (10, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(display, f"Pending: {len(pending)}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Wait for all pending frames to complete (with timeout)
        print(f"\nWaiting for {len(pending)} pending frames...")
        if pending:
            concurrent.futures.wait(pending, timeout=10)
            _drain_results()

        pool.shutdown(wait=False)

    # Stop session and print results
    print(f"Stopping session {session_id}...")
    result = stop_session(base_url, session_id)

    if "error" in result:
        print(f"Error: {result['error']} — {result.get('detail', '')}")
    else:
        print("\n" + "=" * 50)
        print("FINAL RESULT")
        print("=" * 50)
        print(f"Exercise:   {result.get('exercise', 'N/A')}")
        print(f"Reps:       {result.get('rep_count', 0)}")
        reps = result.get("reps", [])
        for i, r in enumerate(reps, 1):
            print(f"  Rep {i}: {r['start_s']}s - {r['end_s']}s")
        conf = result.get("per_class_confidence", {})
        print(f"Confidence: idle={conf.get('idle', 0):.2f} "
              f"concentric={conf.get('concentric', 0):.2f} "
              f"eccentric={conf.get('eccentric', 0):.2f}")
        coach = result.get("coaching_feedback", {})
        print(f"\nCoach: {coach.get('summary', 'N/A')}")
        print(f"\nLatency: {result.get('latency_ms', 0)}ms")


if __name__ == "__main__":
    main()
