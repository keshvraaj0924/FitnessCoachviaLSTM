"""
Tests for the real-time streaming path:

  * StreamSession / StreamingFeatureExtractor (feature path shared with batch)
  * the /v1/stream WebSocket endpoint (config, frames, summary, reset)
  * multi-exercise selection on /v1/analyze

The model weights are not required: the registry's ``step`` / ``predict`` are
mocked so the tests exercise the feature extraction, state machine, protocol
and exercise routing rather than the trained LSTM itself.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """TestClient with a registry whose model calls are mocked.

    NOTE: the fake must NOT subclass ModelRegistry — its singleton __new__
    would return the real global registry instead of this object.
    """
    from src.serving.app import app, get_model_registry

    class FakeRegistry:
        """Stand-in registry that reports ready and returns deterministic steps."""

        def is_ready(self):
            return True

        def step(self, feature, exercise_id, hidden=None):
            # Causal phase pattern: predict based on a monotone counter in the
            # feature vector so a moving object is recognized as active.
            motion = float(np.abs(feature[17]) + feature[:16].sum())
            if motion > 0.05:
                # alternate concentric/eccentric so the state machine counts reps
                phase = 2 if (hidden or 0) and int(hidden[0]) % 2 else 1
                hidden = (int(hidden[0]) + 1,) if hidden else (1,)
                logits = np.zeros(3)
                logits[phase] = 1.0
                return logits, hidden, 0.5
            return np.zeros(3), hidden, 0.5  # idle

        def predict(self, features, exercise_id="pushup"):
            # Mock batch inference: idle everywhere so /v1/analyze returns 0 reps.
            return np.zeros((1, features.shape[0], 3)), 1.0

        def extract_features(self, video_path):
            # The fake upload bytes are not a decodable video; return a
            # plausible (T, 19) feature array instead.
            return np.zeros((30, 19), dtype=np.float32), 10.0

        def count_reps(self, logits):
            from src.model.reps import count_reps_from_logits
            return count_reps_from_logits(
                logits,
                min_concentric_frames=3,
                min_eccentric_frames=3,
                min_idle_frames=5,
                fps=15,
            )

        def compute_stats(self, rep_result):
            from src.llm.coach import RepStats
            from src.model.reps import RepTiming
            durations = [r.duration() for r in rep_result.reps]
            avg = float(np.mean(durations)) if durations else 0.0
            cv_ = float(np.std(durations) / avg) if avg > 0 else 0.0
            return RepStats(
                rep_count=rep_result.rep_count,
                reps=[RepTiming(start_s=r.start_s, end_s=r.end_s) for r in rep_result.reps],
                avg_tempo_s=avg,
                tempo_consistency=cv_,
                concentric_avg_s=0.0,
                eccentric_avg_s=0.0,
                confidence=rep_result.confidence,
            )

        def _settings_target_fps(self):
            return 15

        def _settings_min_concentric_frames(self):
            return 3

        def _settings_min_eccentric_frames(self):
            return 3

        def _settings_min_idle_frames(self):
            return 5

        def _settings_stream_confidence_threshold(self):
            return 0.50

        def get_coaching(self, stats, exercise_id):
            from src.llm.coach import _template_fallback
            return _template_fallback(stats, exercise=exercise_id), 0.0

    fake = FakeRegistry()
    app.dependency_overrides[get_model_registry] = lambda: fake

    # Deliberately NOT a context manager: entering it would run the lifespan,
    # which calls the real registry.load() and would fail on the old BiLSTM
    # checkpoints. The dependency override supplies the fake registry instead.
    yield TestClient(app)
    app.dependency_overrides.clear()


def _jpeg(frame_bgr):
    """Encode a BGR numpy frame to JPEG bytes (the stream wire format)."""
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buf.tobytes()


def _moving_frame(t):
    """A frame with a bright block moving with time, so features are active."""
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    x = int(20 + (t * 3) % 100)
    y = int(20 + (t * 5) % 100)
    cv2.rectangle(img, (x, y), (x + 40, y + 40), (255, 255, 255), -1)
    return img


class TestStreamingFeatureExtractor:
    """The streaming feature path must be causal and identical to batch."""

    def test_subsamples_frames(self):
        from src.video.features import StreamingFeatureExtractor
        ex = StreamingFeatureExtractor(target_fps=15)
        # 30 source frames at 30 fps -> 15 sampled frames.
        sampled = 0
        for t in range(30):
            feat = ex.process_frame(_moving_frame(t), source_fps=30.0)
            if feat is not None:
                sampled += 1
                assert feat.shape == (19,)
        assert sampled == 15

    def test_first_sampled_frame_is_zero(self):
        from src.video.features import StreamingFeatureExtractor
        ex = StreamingFeatureExtractor(target_fps=15)
        first = ex.process_frame(_moving_frame(0), source_fps=30.0)
        assert first is not None
        assert np.allclose(first, 0.0)

    def test_matches_batch_features(self):
        """Stream samples the same source frames as batch and computes the same vectors."""
        from src.video.features import (
            StreamingFeatureExtractor,
            _prepare_gray,
            compute_frame_features,
        )
        ex = StreamingFeatureExtractor(target_fps=15)
        frames = [_moving_frame(t) for t in range(4)]  # 4 source frames @ 30fps
        stream_feats = [
            f for f in (ex.process_frame(fr, 30.0) for fr in frames) if f is not None
        ]
        # ratio = 30/15 = 2 -> stream samples source frames 0 and 2.
        assert len(stream_feats) == 2
        grays = [_prepare_gray(f) for f in frames]
        expected = compute_frame_features(grays[0], grays[2])  # 2nd sample's pair
        assert np.allclose(stream_feats[1], expected, atol=1e-6)


class TestStreamSession:
    """StreamSession integrates extractor + counter + coaching."""

    def _session(self, registry):
        from src.serving.stream import StreamSession
        return StreamSession(registry, exercise_id="pushup", source_fps=30.0)

    def test_process_frame_bytes_returns_frame_message(self):
        from src.serving.stream import StreamSession
        class R:
            def _settings_target_fps(self): return 15
            def _settings_min_concentric_frames(self): return 3
            def _settings_min_eccentric_frames(self): return 3
            def _settings_min_idle_frames(self): return 5
            def _settings_stream_confidence_threshold(self): return 0.50
            def step(self, f, ex, h):
                return np.array([0.5, 0.25, 0.25]), (1,), 0.5
            def get_coaching(self, stats, ex):
                from src.llm.coach import _template_fallback
                return _template_fallback(stats, exercise=ex), 0.0
        s = StreamSession(R(), exercise_id="pushup", source_fps=30.0)
        msgs = [s.process_frame_bytes(_jpeg(_moving_frame(t))) for t in range(30)]
        msgs = [m for m in msgs if m is not None]
        assert msgs, "expected at least one per-frame message"
        assert msgs[0]["type"] == "frame"
        assert "rep_count" in msgs[0]
        assert msgs[0]["rep_count"] == 0

    def test_summary_reads_counter_state(self):
        from src.serving.stream import StreamSession
        class R:
            def _settings_target_fps(self): return 15
            def _settings_min_concentric_frames(self): return 3
            def _settings_min_eccentric_frames(self): return 3
            def _settings_min_idle_frames(self): return 5
            def _settings_stream_confidence_threshold(self): return 0.50
            def step(self, f, ex, h):
                return np.array([0.0, 0.0, 0.0]), None, 0.5  # always idle
            def get_coaching(self, stats, ex):
                from src.llm.coach import _template_fallback
                return _template_fallback(stats, exercise=ex), 0.0
        s = StreamSession(R(), exercise_id="squat", source_fps=30.0)
        for t in range(30):
            s.process_frame_bytes(_jpeg(_moving_frame(t)))
        summ = s.summary()
        assert summ["type"] == "summary"
        assert summ["rep_count"] == 0
        assert summ["exercise"] == "squat"
        assert "coaching_feedback" in summ

    def test_reset_clears_state(self):
        from src.serving.stream import StreamSession
        class R:
            def _settings_target_fps(self): return 15
            def _settings_min_concentric_frames(self): return 3
            def _settings_min_eccentric_frames(self): return 3
            def _settings_min_idle_frames(self): return 5
            def _settings_stream_confidence_threshold(self): return 0.50
            def step(self, f, ex, h):
                return np.array([0.0, 0.0, 0.0]), None, 0.5
            def get_coaching(self, stats, ex):
                from src.llm.coach import _template_fallback
                return _template_fallback(stats, exercise=ex), 0.0
        s = StreamSession(R(), exercise_id="pushup", source_fps=30.0)
        for t in range(30):
            s.process_frame_bytes(_jpeg(_moving_frame(t)))
        s.reset()
        summ = s.summary()
        assert summ["rep_count"] == 0
        assert summ["frames_processed"] == 0


class TestStreamEndpoint:
    """End-to-end WebSocket protocol tests."""

    def test_stream_handshake_and_frames(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "config", "exercise": "pushup", "source_fps": 30.0})
            ready = ws.receive_json()
            assert ready["type"] == "ready"
            assert ready["exercise"] == "pushup"

            # Send a burst of moving frames -> expect per-frame messages.
            got_frame = False
            for t in range(45):
                ws.send_bytes(_jpeg(_moving_frame(t)))

            ws.send_json({"type": "summary"})

            # The server emits one frame message per sampled frame (23 for
            # 45 source frames at 30fps->15fps) *before* the summary reply.
            # Read until the summary arrives, tracking that we saw a frame.
            summary = None
            for _ in range(60):
                msg = ws.receive_json()
                if msg.get("type") == "frame":
                    got_frame = True
                elif msg.get("type") == "summary":
                    summary = msg
                    break
            assert got_frame
            assert summary is not None
            assert "rep_count" in summary

    def test_stream_unknown_exercise_rejected(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "config", "exercise": "nope", "source_fps": 30.0})
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "nope" in err["detail"]

    def test_stream_reset_control(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"type": "config", "exercise": "squat", "source_fps": 30.0})
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "reset"})
            ack = ws.receive_json()
            assert ack["type"] == "reset_ack"


class TestMultiExerciseAnalyze:
    """Exercise routing on the batch /v1/analyze endpoint."""

    def test_analyze_with_exercise_field(self, client):
        files = {"video": ("test.mp4", b"dummy video bytes", "video/mp4")}
        data = {"exercise": "squat"}
        resp = client.post("/v1/analyze", files=files, data=data)
        assert resp.status_code == 200
        assert resp.json()["rep_count"] == 0  # model mocked to idle

    def test_analyze_unknown_exercise(self, client):
        files = {"video": ("test.mp4", b"dummy video bytes", "video/mp4")}
        data = {"exercise": "plank"}
        resp = client.post("/v1/analyze", files=files, data=data)
        assert resp.status_code == 422
        body = resp.json()
        # The exception handler puts the message in `error`.
        assert "plank" in (body.get("error") or body.get("detail") or "")
