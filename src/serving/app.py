"""
FastAPI Application for Exercise Analysis Service.

Endpoints:
- POST /v1/analyze          - Upload video, get rep count and coaching
- POST /v1/live/start       - Start a live-camera session, get session_id
- POST /v1/live/{id}/frame  - Send one JPEG frame, get per-frame status
- POST /v1/live/{id}/stop   - Stop session, get full AnalyzeResponse
- GET  /healthz             - Health and readiness check
- GET  /v1/model            - Model information

Supports multiple exercises (pushup, squat, bicep_curl, jumping_jack). Each
exercise has its own trained LSTM checkpoint; the exercise is selected per
request via the `exercise` field/query (default: pushup).
"""
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.serving.registry import ModelRegistry, get_registry
from src.serving.schemas import (
    AnalyzeResponse,
    CoachFeedbackResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PerClassConfidence,
    RepTimingResponse,
)
from src.serving.settings import settings
from src.serving.stream import StreamSession

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - load model on startup."""
    logger.info("Starting Push-up Analysis API...")

    # Load model
    registry = get_registry()
    success = registry.load(device="cpu")

    if not success:
        logger.warning("Model failed to load - service will not be ready")
    else:
        logger.info("Model loaded and warmed up - service ready")

    yield

    logger.info("Shutting down Push-up Analysis API...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Exercise Analysis API",
    description="Analyze exercise videos (pushup, squat, bicep_curl, jumping_jack) "
                "for rep counting and coaching feedback, in batch or in real time.",
    version=settings.model_version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Dependency Injection
# =============================================================================

def get_model_registry() -> ModelRegistry:
    """Dependency to get model registry."""
    return get_registry()


def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file (content type).

    Accepts the standard video MIME types directly, or `application/octet-stream`
    when the filename carries a known video extension. Many `curl`/browser
    clients send the generic octet-stream MIME for any file, so being strict
    would reject perfectly good uploads; the extension is a strong-enough signal.
    """
    if file.content_type in settings.allowed_content_types:
        return
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext in settings.allowed_extensions:
        return
    raise HTTPException(
        status_code=415,
        detail=f"Unsupported content type: {file.content_type}. "
               f"Allowed: {settings.allowed_content_types}",
    )


def resolve_exercise(exercise_id: str) -> None:
    """Validate an exercise id against the configured exercise set."""
    if settings.exercise_spec(exercise_id) is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown exercise '{exercise_id}'. "
                   f"Supported: {[e.id for e in settings.exercises]}",
        )


async def _read_upload_limited(file: UploadFile, max_size: int) -> bytes:
    """Read the upload in chunks, aborting (413) once it exceeds max_size."""
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)  # 1 MiB at a time
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large (max {max_size} bytes)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


# =============================================================================
# Request ID Middleware
# =============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all responses and logs."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.request_id = request_id

    # Add to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail if isinstance(exc.detail, str) else "HTTP error",
            detail=str(exc.detail) if not isinstance(exc.detail, str) else None,
            request_id=getattr(request.state, 'request_id', None),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.log_level == "DEBUG" else None,
            request_id=getattr(request.state, 'request_id', None),
        ).model_dump(),
    )


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request, registry: ModelRegistry = Depends(get_model_registry)):
    """
    Liveness and readiness probe.

    Returns:
    - status: "ok" if service is running
    - ready: True if model weights loaded and warmup complete
    - model_loaded: True if model weights are loaded

    HTTP 200 when ready; 503 (Service Unavailable) while the model is still
    loading or failed to load, so liveness/readiness probes (and the Docker
    HEALTHCHECK) reflect genuine readiness.
    """
    ready = registry.is_ready()
    response = HealthResponse(
        status="ok",
        ready=ready,
        model_loaded=ready,
    )
    if not ready:
        # Raise through the HTTPException handler so the body keeps the
        # structured schema (including request_id).
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="Not ready",
                detail="Model not loaded yet",
                request_id=getattr(request.state, 'request_id', None),
            ).model_dump(),
        )
    return response


@app.get("/v1/model", response_model=ModelInfoResponse)
async def model_info(
    exercise: str = Query("pushup", description="Exercise id to inspect"),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Get model metadata.

    Returns:
    - model_version: Version string
    - feature_config: Feature extraction configuration
    - checkpoint_hash: SHA256 hash of checkpoint (first 16 chars)
    - architecture: Model architecture details
    """
    if not registry.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")
    resolve_exercise(exercise)

    info = registry.get_model_info(exercise)
    return ModelInfoResponse(**info)


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze_video(
    request: Request,
    video: UploadFile = File(..., description="Video file (mp4, mov, avi, mkv)"),
    exercise: str = Form("auto",
                         description="Exercise id (pushup, squat, bicep_curl, jumping_jack) "
                                     "or 'auto' to detect from the video"),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Analyze an exercise video (batch path).

    Upload a video file (max 50MB) and receive:
    - Repetition count with start/end timestamps
    - Per-class confidence scores
    - Coaching feedback
    - Processing latency breakdown

    Pass ``exercise=auto`` (the default) to let the server run all four
    exercise models and pick the one with the highest non-idle confidence.
    Pass an explicit exercise id to skip detection and use that model directly.
    Returns 415 for unsupported content types, 413 for oversized files.
    """
    request_id = request.state.request_id
    logger.info(f"[{request_id}] Received video: {video.filename}, content_type: {video.content_type}, exercise={exercise!r}")

    # Check readiness
    if not registry.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Model not ready. Check /healthz for status."
        )

    if exercise != "auto":
        resolve_exercise(exercise)
    validate_video_file(video)

    # Read and save to temp file (streamed so we never buffer more than the cap)
    max_size = settings.max_upload_mb * 1024 * 1024
    content = await _read_upload_limited(video, max_size)

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Save to temp file
    suffix = ".mp4"
    if video.filename:
        ext = os.path.splitext(video.filename)[1].lower()
        if ext in [".mp4", ".mov", ".avi", ".mkv"]:
            suffix = ext

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        logger.info(f"[{request_id}] Saved temp file: {temp_path} ({len(content)} bytes)")

        # =========================================================================
        # STAGE 1: Video Decode & Feature Extraction
        # (blocking CPU work; never on the event loop)
        # =========================================================================
        stage_start = time.perf_counter()
        features, decode_ms = await run_in_threadpool(registry.extract_features, temp_path)
        features_ms = (time.perf_counter() - stage_start) * 1000 - decode_ms

        logger.info(f"[{request_id}] Features extracted: {features.shape} (decode: {decode_ms:.0f}ms, features: {features_ms:.0f}ms)")

        # =========================================================================
        # STAGE 2: Exercise Detection (when exercise=auto)
        # =========================================================================
        detected_exercise = exercise
        if exercise == "auto":
            detected_exercise = await run_in_threadpool(
                _detect_exercise, registry, features,
            )
            logger.info(f"[{request_id}] Auto-detected exercise: {detected_exercise}")

        # =========================================================================
        # STAGE 3: LSTM Inference
        # =========================================================================
        stage_start = time.perf_counter()
        logits, lstm_ms = await run_in_threadpool(registry.predict, features, detected_exercise)

        logger.info(f"[{request_id}] LSTM inference: {logits.shape} ({lstm_ms:.0f}ms)")

        # =========================================================================
        # STAGE 4: Rep Counting
        # =========================================================================
        rep_result = await run_in_threadpool(registry.count_reps, logits)

        # =========================================================================
        # STAGE 5: Statistics & Coaching
        # =========================================================================
        stats = await run_in_threadpool(registry.compute_stats, rep_result)

        stage_start = time.perf_counter()
        feedback, llm_ms = await run_in_threadpool(registry.get_coaching, stats, detected_exercise)
        logger.info(f"[{request_id}] Coaching generated ({llm_ms:.0f}ms)")

        # =========================================================================
        # Build Response
        # =========================================================================
        total_ms = int(decode_ms + features_ms + lstm_ms + llm_ms)

        # Per-class confidence from logits (stable softmax)
        logits_0 = np.asarray(logits[0], dtype=np.float64)
        logits_0 = logits_0 - logits_0.max(axis=1, keepdims=True)
        exps = np.exp(logits_0)
        probs = exps / exps.sum(axis=1, keepdims=True)
        per_class_conf = PerClassConfidence(
            idle=float(probs[:, 0].mean()),
            concentric=float(probs[:, 1].mean()),
            eccentric=float(probs[:, 2].mean()),
        )

        response = AnalyzeResponse(
            exercise=detected_exercise,
            rep_count=rep_result.rep_count,
            reps=[RepTimingResponse(start_s=r.start_s, end_s=r.end_s) for r in rep_result.reps],
            per_class_confidence=per_class_conf,
            coaching_feedback=CoachFeedbackResponse(
                summary=feedback.summary,
                strengths=feedback.strengths,
                improvements=feedback.improvements,
                safety_notes=feedback.safety_notes,
            ),
            model_version=settings.model_version,
            latency_ms=total_ms,
            stage_timings={
                "decode_ms": int(decode_ms),
                "features_ms": int(features_ms),
                "lstm_ms": int(lstm_ms),
                "llm_ms": int(llm_ms),
            },
        )

        logger.info(f"[{request_id}] Analysis complete ({detected_exercise}): {rep_result.rep_count} reps, {total_ms}ms total")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e!s}") from e

    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                logger.debug(f"[{request_id}] Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to cleanup temp file: {e}")


# =============================================================================
# Live Camera Sessions (HTTP polling, same StreamSession engine)
# =============================================================================

# In-memory store for live camera sessions.  Each session owns its own
# StreamSession so concurrent clients are fully isolated.
_live_sessions: dict[str, StreamSession] = {}


def _new_session_id() -> str:
    return str(uuid.uuid4())[:8]


class LiveSessionResponse(BaseModel):
    """Lightweight per-frame status for the live camera path."""
    type: Literal["frame"] = "frame"
    t_s: float
    phase: str
    confidence: float
    rep_count: int


class LiveStartResponse(BaseModel):
    """Response to POST /v1/live/start."""
    session_id: str
    exercise: str
    target_fps: int
    feature_dim: int


@app.post("/v1/live/start", response_model=LiveStartResponse)
async def live_start(
    exercise: str = Form("pushup"),
    source_fps: float = Form(30.0),
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Start a live-camera session.

    Returns a ``session_id``.  Send JPEG frames to
    ``POST /v1/live/{session_id}/frame`` and stop with
    ``POST /v1/live/{session_id}/stop``.

    The ``/v1/live`` path uses the same ``StreamSession`` engine as the
    WebSocket endpoint, so the per-frame results and final summary are
    identical - only the transport differs.
    """
    if not registry.is_ready():
        raise HTTPException(status_code=503, detail="Model not ready")

    if exercise != "auto" and settings.exercise_spec(exercise) is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown exercise '{exercise}'. Supported: {[e.id for e in settings.exercises]} or 'auto'",
        )

    session_id = _new_session_id()
    _live_sessions[session_id] = StreamSession(
        registry, exercise_id=exercise, source_fps=source_fps
    )
    logger.info(f"Live session {session_id} started for {exercise} @ {source_fps} fps")
    return LiveStartResponse(
        session_id=session_id,
        exercise=exercise,
        target_fps=registry._settings_target_fps(),
        feature_dim=settings.feature_dim,
    )


@app.post("/v1/live/{session_id}/frame", response_model=LiveSessionResponse)
async def live_frame(
    session_id: str,
    request: Request,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Send one camera frame to a live session.

    Accepts raw JPEG bytes in the request body (Content-Type: image/jpeg).
    Returns the same per-frame message the WebSocket path would send:
    phase, confidence, and current rep count.
    """
    session = _live_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /v1/live/start first.")

    jpeg_bytes = await request.body()
    if not jpeg_bytes:
        return LiveSessionResponse(
            t_s=session.extractor.sample_count / session.registry._settings_target_fps(),
            phase="idle",
            confidence=0.0,
            rep_count=session.counter.rep_count,
        )

    result = await run_in_threadpool(session.process_frame_bytes, jpeg_bytes)
    if result is None:
        return LiveSessionResponse(
            t_s=session.extractor.sample_count / session.registry._settings_target_fps(),
            phase="idle",
            confidence=0.0,
            rep_count=session.counter.rep_count,
        )

    return LiveSessionResponse(
        t_s=result["t_s"],
        phase=result["phase"],
        confidence=result["confidence"],
        rep_count=result["rep_count"],
    )


@app.post("/v1/live/{session_id}/stop", response_model=AnalyzeResponse)
async def live_stop(
    session_id: str,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """Stop a live session and return the full analysis result.

    The response has the **same shape** as ``POST /v1/analyze``:
    ``rep_count``, ``reps``, ``per_class_confidence``, ``coaching_feedback``,
    ``model_version``, ``latency_ms``, ``stage_timings``.
    """
    session = _live_sessions.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    summary = session.summary()

    # Map StreamSession.summary() keys to AnalyzeResponse fields.
    # summary has per_class_confidence as a float; AnalyzeResponse needs
    # a PerClassConfidence object.  We approximate by using the overall
    # confidence as all three phases equally (honest limitation noted in
    # NOTES.md - per-class confidence requires running the LSTM again,
    # which the WebSocket path doesn't do for performance).
    conf = summary.get("per_class_confidence", 0.0)
    per_class_conf = PerClassConfidence(
        idle=conf, concentric=conf, eccentric=conf
    )

    elapsed_ms = int(summary.get("elapsed_s", 0.0) * 1000)

    return AnalyzeResponse(
        exercise=session.exercise_id,
        rep_count=summary["rep_count"],
        reps=[RepTimingResponse(**r) for r in summary["reps"]],
        per_class_confidence=per_class_conf,
        coaching_feedback=CoachFeedbackResponse(**summary["coaching_feedback"]),
        model_version=settings.model_version,
        latency_ms=elapsed_ms,
        stage_timings={
            "decode_ms": 0,
            "features_ms": 0,
            "lstm_ms": 0,
            "llm_ms": 0,
        },
    )


# =============================================================================
# Real-Time Streaming (WebSocket)
# =============================================================================

@app.websocket("/v1/stream")
async def stream_video(
    websocket: WebSocket,
    registry: ModelRegistry = Depends(get_model_registry),
):
    """
    Real-time streaming rep counting over WebSocket.

    Protocol (all messages JSON text unless noted):
      1. Client connects and sends a config frame first:
             {"type": "config", "exercise": "pushup", "source_fps": 30.0}
      2. Server replies {"type": "ready", "exercise": ...}.
      3. Client streams raw JPEG frames as BINARY messages (one per camera
         frame). Server replies per sampled frame with a progress dict:
             {"type": "frame", "t_s": ..., "phase": ..., "confidence": ...,
              "rep_count": ...}
      4. Client may send {"type": "summary"} any time to get the current
         cumulative summary, or {"type": "reset"} to restart the session.
      5. On disconnect the server sends a final summary.

    Notes:
      - Frames are decoded with cv2.imdecode; a corrupt frame is skipped and
        logged, never fatal.
      - Inference runs in a thread pool so the event loop stays responsive.
      - The model is the causal (unidirectional) LSTM; step-by-step inference
        shares the exact same weights as the batch /v1/analyze path.
    """
    session = None
    try:
        await websocket.accept()

        # 1. Config
        config = await websocket.receive_json()
        if config.get("type") != "config":
            await websocket.close(code=4001, reason="First message must be a config frame")
            return
        exercise_id = config.get("exercise", "pushup")
        source_fps = float(config.get("source_fps", 30.0))

        if settings.exercise_spec(exercise_id) is None:
            await websocket.send_json({
                "type": "error",
                "detail": f"Unknown exercise '{exercise_id}'. "
                          f"Supported: {[e.id for e in settings.exercises]}",
            })
            await websocket.close(code=4002, reason="Unknown exercise")
            return

        if not registry.is_ready():
            await websocket.send_json({"type": "error", "detail": "Model not ready"})
            await websocket.close(code=4003, reason="Model not ready")
            return

        session = StreamSession(registry, exercise_id=exercise_id, source_fps=source_fps)
        await websocket.send_json({
            "type": "ready",
            "exercise": exercise_id,
            "target_fps": settings.target_fps,
            "feature_dim": settings.feature_dim,
        })

        # 2. Frame loop
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if message["type"] == "websocket.receive":
                if isinstance(message.get("text"), str):
                    text = message["text"].strip()
                    if not text:
                        continue
                    try:
                        import json as _json
                        control = _json.loads(text)
                    except Exception:
                        await websocket.send_json({"type": "error", "detail": "Invalid JSON control message"})
                        continue
                    ctype = control.get("type")
                    if ctype == "summary":
                        await websocket.send_json(session.summary())
                    elif ctype == "reset":
                        session.reset()
                        await websocket.send_json({"type": "reset_ack"})
                    else:
                        await websocket.send_json({"type": "error", "detail": f"Unknown control message '{ctype}'"})
                    continue

                if isinstance(message.get("bytes"), bytes) and message["bytes"]:
                    frame_bytes = message["bytes"]
                    result = await run_in_threadpool(session.process_frame_bytes, frame_bytes)
                    if result is not None:
                        await websocket.send_json(result)

        # 3. Final summary on disconnect
        if session is not None:
            try:
                await websocket.send_json(session.summary())
            except Exception:
                pass

    except WebSocketDisconnect:
        logger.info("Stream client disconnected")
        if session is not None:
            try:
                await websocket.send_json(session.summary())
            except Exception:
                pass
    except Exception as e:
        logger.exception("Stream error")
        try:
            await websocket.send_json({"type": "error", "detail": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# =============================================================================
# Main Entry Point
# =============================================================================

def _detect_exercise(registry, features: np.ndarray) -> str:
    """Run all four exercise models and pick the one with the highest
    average non-idle confidence.

    This is a cheap heuristic: we do a single forward pass per exercise
    (all unidirectional, all on CPU, 30 sampled frames ≈ 30 steps each
    at most), then average the softmax confidence over frames where the
    predicted phase is *not* idle. The exercise whose model is most
    confident about active motion wins.

    Args:
        registry: Loaded ``ModelRegistry``.
        features: ``(T, 19)`` feature array from the uploaded video.

    Returns:
        Exercise id string (one of ``settings.exercises``).
    """
    candidates = [e.id for e in settings.exercises]
    best_exercise = candidates[0]
    best_score = -1.0

    for ex_id in candidates:
        logits, _ = registry.predict(features, ex_id)
        logits = np.asarray(logits[0], dtype=np.float64)
        logits = logits - logits.max(axis=1, keepdims=True)
        exps = np.exp(logits)
        probs = exps / exps.sum(axis=1, keepdims=True)
        # Confidence on non-idle frames (phase != 0).
        pred_phases = probs.argmax(axis=1)
        active = probs[pred_phases != 0]
        score = float(active.max(axis=1).mean()) if len(active) > 0 else 0.0
        logger.debug(f"Auto-detect {ex_id}: score={score:.4f}")
        if score > best_score:
            best_score = score
            best_exercise = ex_id

    return best_exercise


def _window_exercise_scores(registry, features: np.ndarray,
                             window_size: int = 45, stride: int = 15
                             ) -> np.ndarray:
    """Score every sliding window of ``features`` with all four exercise models.

    Each window is classified by running all four models and picking the one
    with the highest average non-idle confidence (same heuristic as
    ``_detect_exercise``). Windows that are mostly idle get a special
    ``-1`` label so they don't pollute segment boundaries.

    Args:
        registry: Loaded ``ModelRegistry``.
        features: ``(T, 19)`` feature array.
        window_size: Number of frames per window (default ~3s at 15fps).
        stride: Hop between windows (default ~1s, gives 3x overlap).

    Returns:
        ``(n_windows,)`` int array of exercise indices (0-3) or ``-1`` for idle.
    """
    candidates = [e.id for e in settings.exercises]
    T = features.shape[0]
    starts = list(range(0, max(1, T - window_size + 1), stride))
    if not starts or starts[-1] + window_size < T:
        starts.append(max(0, T - window_size))

    labels = np.full(len(starts), -1, dtype=int)

    for wi, start in enumerate(starts):
        win = features[start:start + window_size]
        best_idx = 0
        best_score = -1.0
        scores = {}
        for ci, ex_id in enumerate(candidates):
            logits, _ = registry.predict(win, ex_id)
            logits = np.asarray(logits[0], dtype=np.float64)
            logits = logits - logits.max(axis=1, keepdims=True)
            exps = np.exp(logits)
            probs = exps / exps.sum(axis=1, keepdims=True)
            pred_phases = probs.argmax(axis=1)
            active = probs[pred_phases != 0]
            score = float(active.max(axis=1).mean()) if len(active) > 0 else 0.0
            scores[ex_id] = round(score, 4)
            if score > best_score:
                best_score = score
                best_idx = ci
        # Use a lower threshold with high overlap so transitions are detected.
        # Also require the winner to be at least 2x the runner-up to avoid
        # near-ties collapsing both exercises into one.
        runner_up = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        margin = best_score - runner_up
        labels[wi] = best_idx if (best_score > 0.08 and margin > 0.02) else -1
        logger.debug(
            f"Window {wi} [frame {start}-{start+window_size}]: "
            f"scores={scores} -> {candidates[best_idx] if labels[wi] >= 0 else 'idle'}"
        )

    return labels


def _smooth_labels(labels: np.ndarray, window: int = 3) -> np.ndarray:
    """Majority-vote smoothing over a sliding window.

    Idle labels (-1) are preserved unless a clear majority of the window
    is a non-idle exercise.
    """
    if len(labels) <= window:
        return labels
    out = np.empty_like(labels)
    half = window // 2
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        window_slice = labels[lo:hi]
        # Count non-idle occurrences
        counts: dict[int, int] = {}
        for v in window_slice:
            if v != -1:
                counts[v] = counts.get(v, 0) + 1
        if counts:
            out[i] = max(counts, key=counts.get)  # type: ignore[arg-type]
        else:
            out[i] = -1
    return out


def _labels_to_segments(
    labels: np.ndarray,
    starts: list[int],
    window_size: int,
    fps: float,
) -> list[dict]:
    """Merge consecutive same-label windows into segments.

    Args:
        labels: ``(n_windows,)`` int array (0-3 or -1).
        starts: Start frame for each window.
        window_size: Frames per window.
        fps: Feature extraction frame rate.

    Returns:
        List of ``{"exercise": str, "start_frame": int, "end_frame": int}``.
        Idle windows (-1) are skipped.
    """
    segments = []
    current_label = None
    current_start = None

    for i, lbl in enumerate(labels):
        if lbl != current_label:
            if current_label is not None and current_label != -1:
                seg_start_frame = starts[current_start]
                seg_end_frame = starts[i - 1] + window_size
                segments.append({
                    "exercise_idx": int(current_label),
                    "start_frame": int(seg_start_frame),
                    "end_frame": int(seg_end_frame),
                })
            current_label = lbl if lbl != -1 else None
            current_start = i if lbl != -1 else None

    if current_label is not None and current_start is not None:
        seg_start_frame = starts[current_start]
        seg_end_frame = starts[-1] + window_size
        segments.append({
            "exercise_idx": int(current_label),
            "start_frame": int(seg_start_frame),
            "end_frame": int(seg_end_frame),
        })

    return segments


def _segment_exercises(registry, features: np.ndarray) -> list[dict]:
    """Segment a video into exercise segments by detecting transitions.

    Uses a two-pass approach:
      1. Per-frame: for each frame, score all 4 models and pick the winner.
         Apply a majority-vote temporal filter (2 s window) to smooth noise.
      2. Transition detection: find frame ranges where the smoothed label
         flips from one exercise to another. Require each segment to be at
         least 3 seconds (45 frames @ 15 fps) to avoid spurious splits.

    If the video doesn't contain clear transitions (all frames get the same
    label, or segments are too short), fall back to global detection and
    return a single segment.

    Args:
        registry: Loaded ``ModelRegistry``.
        features: ``(T, 19)`` feature array from the uploaded video.

    Returns:
        List of ``{"exercise": str, "start_frame": int, "end_frame": int}``
        dicts, one per detected exercise segment.
    """
    fps = registry._settings_target_fps()
    candidates = [e.id for e in settings.exercises]
    T = features.shape[0]

    if T < 15:
        ex = _detect_exercise(registry, features)
        return [{"exercise": ex, "start_frame": 0, "end_frame": T}]

    # --- Pass 1: per-frame winner (which exercise model is most confident). ---
    all_probs = {}
    for ex_id in candidates:
        logits, _ = registry.predict(features, ex_id)
        lg = np.asarray(logits[0], dtype=np.float64)
        lg = lg - lg.max(axis=1, keepdims=True)
        exps = np.exp(lg)
        all_probs[ex_id] = exps / exps.sum(axis=1, keepdims=True)

    # Winner per frame: exercise whose model has the highest non-idle confidence.
    frame_winner = np.full(T, -1, dtype=int)
    frame_conf = np.zeros(T)
    for t in range(T):
        best_ci, best_active = -1, 0.0
        for ci, ex_id in enumerate(candidates):
            probs = all_probs[ex_id][t]
            active_conf = float(max(probs[1], probs[2]))
            if active_conf > best_active:
                best_active = active_conf
                best_ci = ci
        frame_winner[t] = best_ci
        frame_conf[t] = best_active

    # --- Pass 2: temporal smoothing (2-second majority window). ---
    smooth_win = int(fps * 2)
    if smooth_win < T:
        half = smooth_win // 2
        smooth = np.empty_like(frame_winner)
        for i in range(T):
            lo, hi = max(0, i - half), min(T, i + half + 1)
            window_slice = frame_winner[lo:hi]
            counts: dict[int, int] = {}
            for v in window_slice:
                if v != -1:
                    counts[v] = counts.get(v, 0) + 1
            smooth[i] = max(counts, key=counts.get) if counts else -1  # type: ignore[arg-type]
    else:
        smooth = frame_winner

    # --- Pass 3: merge consecutive same-label runs into segments. ---
    min_seg = int(fps * 3)  # minimum 3 seconds per segment
    segments = []
    cur_label, cur_start = None, None

    for i in range(T):
        lbl = int(smooth[i])
        if lbl != cur_label:
            if cur_label is not None and cur_label != -1:
                seg_len = i - cur_start
                if seg_len >= min_seg:
                    segments.append({
                        "exercise_idx": cur_label,
                        "start_frame": cur_start,
                        "end_frame": i,
                    })
            cur_label = lbl if lbl != -1 else None
            cur_start = i

    if cur_label is not None and cur_start is not None:
        seg_len = T - cur_start
        if seg_len >= min_seg:
            segments.append({
                "exercise_idx": cur_label,
                "start_frame": cur_start,
                "end_frame": T,
            })

    # --- Pass 4: convert indices → exercise ids; merge same-exercise neighbors. ---
    if not segments:
        ex = _detect_exercise(registry, features)
        return [{"exercise": ex, "start_frame": 0, "end_frame": T}]

    merged = [segments[0]]
    for seg in segments[1:]:
        if seg["exercise_idx"] == merged[-1]["exercise_idx"]:
            merged[-1]["end_frame"] = seg["end_frame"]
        else:
            merged.append(seg)

    for seg in merged:
        seg["exercise"] = candidates[seg.pop("exercise_idx")]

    return merged


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
