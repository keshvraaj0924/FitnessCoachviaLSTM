"""Live camera test using the HTTP live-session endpoints.

Usage:
    python scripts/live_camera_http.py --exercise pushup
    python scripts/live_camera_http.py --exercise auto --source-fps 30

Press 'q' in the OpenCV window or Ctrl+C in the terminal to stop cleanly.
"""
import argparse
import concurrent.futures
import json
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import cv2

DEFAULT_BASE_URL = "http://localhost:8000"
_FRAME_WORKERS = 4
_STOP_TIMEOUT_S = 8


def _post_raw(url: str, raw_bytes: bytes, timeout: float = 5.0) -> dict:
    """POST raw JPEG bytes, return JSON response or error dict."""
    req = urllib.request.Request(url, data=raw_bytes, method="POST")
    req.add_header("Content-Type", "image/jpeg")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:200]}
    except Exception as e:
        return {"error": str(e)}


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
    req = urllib.request.Request(
        f"{base_url}/v1/live/{session_id}/stop",
        data=b"",
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_STOP_TIMEOUT_S) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode()[:200]}


class LiveCameraClient:
    """Encapsulates the live-camera session lifecycle."""

    def __init__(self, base_url: str, exercise: str, source_fps: float,
                 camera: int = 0):
        self.base_url = base_url
        self.exercise = exercise
        self.source_fps = source_fps
        self.camera_idx = camera

        self.session_id: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.pool: concurrent.futures.ThreadPoolExecutor | None = None
        self.pending: list[concurrent.futures.Future] = []
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        # Display state (updated from completed futures)
        self.last_rep_count = 0
        self.last_phase = "idle"
        self.last_conf = 0.0
        self.frames_sent = 0

    # ── session lifecycle ──────────────────────────────────────────────

    def start(self) -> bool:
        """Open camera, start session, launch worker pool. Returns False on failure."""
        self.session_id = start_session(self.base_url, self.exercise, self.source_fps)
        if self.session_id is None:
            return False

        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            print("Cannot open camera")
            self._cleanup_session()
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_FRAME_WORKERS,
            thread_name_prefix="frame_upload",
        )
        return True

    def stop(self):
        """Clean shutdown: stop camera, flush pool, stop session, print results."""
        print("\nShutting down...")
        self._shutdown_event.set()

        # 1. Release camera and destroy windows
        self._release_camera()

        # 2. Cancel pending futures and shutdown pool (non-blocking)
        if self.pool is not None:
            for fut in self.pending:
                fut.cancel()
            self.pending.clear()
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.pool = None

        # 3. Stop session on server
        if self.session_id:
            sid = self.session_id
            self.session_id = None
            print(f"Stopping session {sid}...")
            result = stop_session(self.base_url, sid)
            self._print_result(result)

    # ── frame loop ─────────────────────────────────────────────────────

    def run_loop(self):
        """Main camera loop. Returns when user quits."""
        window = f"Live {self.exercise} — press q to stop"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        print("\nCamera active. Press 'q' to stop and get results.\n")

        try:
            while not self._shutdown_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break

                # Encode to JPEG
                ok, jpeg = cv2.imencode(".jpg", frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    continue

                # Fire-and-forget: submit frame to background pool
                self._submit_frame(jpeg.tobytes())
                self.frames_sent += 1

                # Drain completed results (non-blocking)
                self._drain_results()

                # Overlay status on frame
                display = frame.copy()
                cv2.putText(display, f"Reps: {self.last_rep_count}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(display, f"Phase: {self.last_phase}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.putText(display, f"Conf: {self.last_conf:.2f}", (10, 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(display, f"Sent: {self.frames_sent}", (10, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.putText(display, f"Pending: {len(self.pending)}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self._release_camera()

    # ── internal helpers ───────────────────────────────────────────────

    def _submit_frame(self, jpeg_bytes: bytes):
        """Submit a frame for async upload."""
        if self.pool is None:
            return
        url = f"{self.base_url}/v1/live/{self.session_id}/frame"
        fut = self.pool.submit(_post_raw, url, jpeg_bytes)
        with self._lock:
            self.pending.append(fut)

    def _drain_results(self):
        """Collect any completed futures and update display state."""
        with self._lock:
            still_pending = []
            for fut in self.pending:
                if fut.done():
                    try:
                        r = fut.result()
                        if r and "rep_count" in r:
                            self.last_rep_count = r["rep_count"]
                            self.last_phase = r.get("phase", "?")
                            self.last_conf = r.get("confidence", 0.0)
                    except Exception:
                        pass
                else:
                    still_pending.append(fut)
            self.pending = still_pending

    def _release_camera(self):
        """Release camera and close all OpenCV windows (idempotent)."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _cleanup_session(self):
        """Best-effort session cleanup if start fails partway."""
        self._release_camera()
        if self.session_id and self.base_url:
            try:
                stop_session(self.base_url, self.session_id)
            except Exception:
                pass

    @staticmethod
    def _print_result(result: dict):
        """Print the final analysis result."""
        if "error" in result:
            print(f"Error: {result['error']} — {result.get('detail', '')}")
            return

        print("=" * 50)
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


def _setup_signal_handlers(client: LiveCameraClient):
    """Handle SIGINT and SIGTERM for graceful shutdown."""
    def handler(signum, frame):
        signame = signal.Signals(signum).name
        print(f"\nReceived {signame} — shutting down gracefully...")
        client._shutdown_event.set()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main():
    parser = argparse.ArgumentParser(
        description="Live camera rep counter (HTTP)"
    )
    parser.add_argument("--exercise", default="pushup",
                        help="Exercise id (or 'auto')")
    parser.add_argument("--source-fps", type=float, default=30.0,
                        help="Camera FPS")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")
    parser.add_argument("--url", default=DEFAULT_BASE_URL,
                        help="Server base URL")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    client = LiveCameraClient(
        base_url=base_url,
        exercise=args.exercise,
        source_fps=args.source_fps,
        camera=args.camera,
    )
    _setup_signal_handlers(client)

    if not client.start():
        sys.exit(1)

    try:
        client.run_loop()
    finally:
        client.stop()

    print("Done.")


if __name__ == "__main__":
    main()
