"""
API Request/Response Schemas for Push-up Analysis Service.
"""
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RepTimingResponse(BaseModel):
    """Start and end time of a repetition."""
    start_s: float = Field(..., description="Start timestamp in seconds")
    end_s: float = Field(..., description="End timestamp in seconds")

    model_config = ConfigDict(json_schema_extra={
        "example": {"start_s": 1.2, "end_s": 3.5}
    })


class CoachFeedbackResponse(BaseModel):
    """Coaching feedback from LLM."""
    summary: str
    strengths: list[str]
    improvements: list[str]
    safety_notes: list[str] = []


class PerClassConfidence(BaseModel):
    """Per-class prediction confidence."""
    idle: float = Field(..., ge=0.0, le=1.0)
    concentric: float = Field(..., ge=0.0, le=1.0)
    eccentric: float = Field(..., ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    """Response from video analysis."""
    exercise: str = Field(..., description="Exercise id (resolved when 'auto')")
    rep_count: int = Field(..., ge=0, description="Total repetition count")
    reps: list[RepTimingResponse] = Field(..., description="Per-rep timings")
    per_class_confidence: PerClassConfidence = Field(..., description="Average per-class confidence")
    coaching_feedback: CoachFeedbackResponse = Field(..., description="Coaching feedback")
    model_version: str = Field(..., description="Model version")
    latency_ms: int = Field(..., ge=0, description="Total processing latency in milliseconds")
    stage_timings: dict = Field(..., description="Per-stage processing times in ms")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "exercise": "pushup",
            "rep_count": 10,
            "reps": [
                {"start_s": 1.2, "end_s": 3.5},
                {"start_s": 4.1, "end_s": 6.3}
            ],
            "per_class_confidence": {
                "idle": 0.92,
                "concentric": 0.87,
                "eccentric": 0.89
            },
            "coaching_feedback": {
                "summary": "Great work completing 10 push-ups with consistent tempo!",
                "strengths": ["Excellent rep consistency", "Well-controlled eccentric phase"],
                "improvements": ["Try slowing the concentric phase slightly"],
                "safety_notes": []
            },
            "model_version": "1.0.0",
            "latency_ms": 6734,
            "stage_timings": {
                "decode_ms": 3255,
                "features_ms": 0,
                "lstm_ms": 2,
                "llm_ms": 2579
            }
        }
    })


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["ok"] = "ok"
    ready: bool = Field(..., description="Whether model weights are loaded and service is ready")
    model_loaded: bool
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ModelInfoResponse(BaseModel):
    """Model information endpoint response."""
    model_version: str
    feature_config: dict
    checkpoint_hash: str
    architecture: dict


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str | None = None
    request_id: str | None = None
