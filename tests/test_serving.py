"""
Unit tests for Component D: Serving API (app.py, registry.py, schemas.py)
"""
# Import after setting up mocks
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_rep_timing_response(self):
        from src.serving.schemas import RepTimingResponse
        rep = RepTimingResponse(start_s=1.0, end_s=3.5)
        assert rep.start_s == 1.0
        assert rep.end_s == 3.5

    def test_per_class_confidence(self):
        from src.serving.schemas import PerClassConfidence
        conf = PerClassConfidence(idle=0.9, concentric=0.8, eccentric=0.85)
        assert conf.idle == 0.9

    def test_confidence_validation(self):
        from src.serving.schemas import PerClassConfidence
        with pytest.raises(Exception):
            PerClassConfidence(idle=1.5, concentric=0.8, eccentric=0.85)  # > 1.0

    def test_analyze_response(self):
        from src.serving.schemas import (
            AnalyzeResponse,
            CoachFeedbackResponse,
            PerClassConfidence,
            RepTimingResponse,
        )
        response = AnalyzeResponse(
            exercise="pushup",
            rep_count=10,
            reps=[RepTimingResponse(start_s=1.0, end_s=3.0)],
            per_class_confidence=PerClassConfidence(idle=0.9, concentric=0.8, eccentric=0.85),
            coaching_feedback=CoachFeedbackResponse(
                summary="Good job!",
                strengths=["Consistent"],
                improvements=["Slow down"],
                safety_notes=[],
            ),
            model_version="1.0.0",
            latency_ms=1000,
            stage_timings={"decode_ms": 100, "features_ms": 200, "lstm_ms": 50, "llm_ms": 650},
        )
        assert response.rep_count == 10

    def test_health_response(self):
        from src.serving.schemas import HealthResponse
        health = HealthResponse(ready=True, model_loaded=True)
        assert health.status == "ok"
        assert health.ready is True


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_registry_singleton(self):
        from src.serving.registry import ModelRegistry, get_registry
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        r3 = get_registry()
        assert r1 is r2
        assert r2 is r3

    def test_registry_not_ready_initially(self):
        from src.serving.registry import ModelRegistry
        # Create fresh registry
        r = ModelRegistry.__new__(ModelRegistry)
        r._initialized = False
        r.__init__()
        assert r.is_ready() is False

    def test_get_model_info_not_ready(self):
        from src.serving.registry import ModelRegistry
        r = ModelRegistry.__new__(ModelRegistry)
        r._initialized = False
        r.__init__()
        info = r.get_model_info()
        assert "model_version" in info
        assert "feature_config" in info


@pytest.fixture
def client():
    """Create test client with mocked registry (module-scoped fixture)."""
    from src.serving.app import app

    # Mock the registry
    mock_registry = MagicMock()
    mock_registry.is_ready.return_value = True
    mock_registry.get_model_info.return_value = {
        "model_version": "1.0.0",
        "feature_config": {"target_fps": 15, "feature_dim": 19},
        "checkpoint_hash": "abc123",
        "architecture": {"type": "BiLSTM"},
        "rep_counter_config": {},
    }
    mock_registry.extract_features.return_value = (np.zeros((30, 19), dtype=np.float32), 100.0)
    mock_registry.predict.return_value = (np.zeros((1, 30, 3)), 50.0)

    # Mock rep counting
    from src.model.reps import RepCountResult, RepTiming
    mock_registry.count_reps.return_value = RepCountResult(
        rep_count=10,
        reps=[RepTiming(start_s=1.0, end_s=3.0), RepTiming(start_s=4.0, end_s=6.0)],
        phase_sequence=[0]*10 + [1]*5 + [2]*5 + [0]*10,
        confidence=0.9,
    )

    # Mock compute_stats
    from src.llm.coach import RepStats
    from src.llm.coach import RepTiming as CoachRepTiming
    mock_registry.compute_stats.return_value = RepStats(
        rep_count=10,
        reps=[CoachRepTiming(start_s=1.0, end_s=3.0), CoachRepTiming(start_s=4.0, end_s=6.0)],
        avg_tempo_s=2.0,
        tempo_consistency=0.1,
        concentric_avg_s=1.0,
        eccentric_avg_s=1.0,
        confidence=0.9,
    )

    # Mock coaching
    from src.llm.coach import CoachFeedback
    mock_registry.get_coaching.return_value = (
        CoachFeedback(
            summary="Great job!",
            strengths=["Consistent tempo"],
            improvements=["Slow down eccentric"],
            safety_notes=[],
        ),
        500.0,
    )

    # Override dependency
    from src.serving.app import get_model_registry
    app.dependency_overrides[get_model_registry] = lambda: mock_registry

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestAppEndpoints:
    """Tests for FastAPI endpoints using TestClient."""

    def test_healthz_ready(self, client):
        """Test /healthz when model is ready."""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["ready"] is True
        assert data["model_loaded"] is True

    def test_healthz_not_ready(self, client):
        """Test /healthz returns 503 while the model is not ready."""
        from src.serving.app import get_model_registry
        mock_registry = MagicMock()
        mock_registry.is_ready.return_value = False
        client.app.dependency_overrides[get_model_registry] = lambda: mock_registry

        response = client.get("/healthz")
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "Not ready"

    def test_model_info(self, client):
        """Test /v1/model endpoint."""
        response = client.get("/v1/model")
        assert response.status_code == 200
        data = response.json()
        assert data["model_version"] == "1.0.0"
        assert "feature_config" in data
        assert "checkpoint_hash" in data

    def test_analyze_success(self, client):
        """Test successful video analysis."""
        # Create a dummy video file
        video_content = b"fake video content"
        files = {"video": ("test.mp4", BytesIO(video_content), "video/mp4")}

        response = client.post("/v1/analyze", files=files)
        assert response.status_code == 200
        data = response.json()

        assert data["rep_count"] == 10
        assert len(data["reps"]) == 2
        assert "coaching_feedback" in data
        assert "stage_timings" in data
        assert data["model_version"] == "1.0.0"

    def test_analyze_invalid_content_type(self, client):
        """Test rejection of invalid content type."""
        files = {"video": ("test.txt", BytesIO(b"not a video"), "text/plain")}
        response = client.post("/v1/analyze", files=files)
        assert response.status_code == 415

    def test_analyze_empty_file(self, client):
        """Test rejection of empty file."""
        files = {"video": ("test.mp4", BytesIO(b""), "video/mp4")}
        response = client.post("/v1/analyze", files=files)
        assert response.status_code == 400

    def test_analyze_oversized_file(self, client):
        """Test rejection of oversized file."""
        # Create content larger than 50MB
        large_content = b"x" * (51 * 1024 * 1024)
        files = {"video": ("test.mp4", BytesIO(large_content), "video/mp4")}
        response = client.post("/v1/analyze", files=files)
        assert response.status_code == 413

    def test_analyze_model_not_ready(self, client):
        """Test analysis when model not ready."""
        from src.serving.app import get_model_registry

        mock_registry = MagicMock()
        mock_registry.is_ready.return_value = False
        client.app.dependency_overrides[get_model_registry] = lambda: mock_registry

        files = {"video": ("test.mp4", BytesIO(b"video"), "video/mp4")}
        response = client.post("/v1/analyze", files=files)
        assert response.status_code == 503


class TestRequestIDMiddleware:
    """Test request ID middleware."""

    def test_request_id_header(self, client):
        """Test that request ID is added to response headers."""
        response = client.get("/healthz")
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
