"""
Unit tests for Component C: Coaching Summary (coach.py)

All tests use mock responses - no network calls.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.llm.coach import (
    CoachFeedback,
    RepStats,
    RepTiming,
    _template_fallback,
    summarize,
    summarize_with_mock,
)


class TestRepStats:
    """Tests for RepStats schema."""

    def test_valid_stats(self):
        """Test valid RepStats creation."""
        stats = RepStats(
            rep_count=10,
            reps=[RepTiming(start_s=1.0, end_s=3.0), RepTiming(start_s=4.0, end_s=6.0)],
            avg_tempo_s=2.0,
            tempo_consistency=0.1,
            concentric_avg_s=1.0,
            eccentric_avg_s=1.0,
            confidence=0.9,
        )
        assert stats.rep_count == 10
        assert len(stats.reps) == 2

    def test_zero_reps(self):
        """Test zero reps stats."""
        stats = RepStats(
            rep_count=0,
            reps=[],
            avg_tempo_s=0.0,
            tempo_consistency=0.0,
            concentric_avg_s=0.0,
            eccentric_avg_s=0.0,
            confidence=0.5,
        )
        assert stats.rep_count == 0
        assert stats.reps == []


class TestCoachFeedback:
    """Tests for CoachFeedback schema."""

    def test_valid_feedback(self):
        """Test valid CoachFeedback creation."""
        feedback = CoachFeedback(
            summary="Great work on 10 push-ups!",
            strengths=["Consistent tempo", "Good form"],
            improvements=["Slow down eccentric phase"],
            safety_notes=[],
        )
        assert feedback.summary == "Great work on 10 push-ups!"
        assert len(feedback.strengths) == 2

    def test_validation_min_max_items(self):
        """Test validation of list lengths."""
        # Too many strengths
        with pytest.raises(Exception):
            CoachFeedback(
                summary="Test",
                strengths=["a", "b", "c", "d"],  # max 3
                improvements=["a"],
                safety_notes=[],
            )

        # Too many improvements
        with pytest.raises(Exception):
            CoachFeedback(
                summary="Test",
                strengths=["a"],
                improvements=["a", "b", "c", "d"],  # max 3
                safety_notes=[],
            )

        # Too many safety notes
        with pytest.raises(Exception):
            CoachFeedback(
                summary="Test",
                strengths=["a"],
                improvements=["a"],
                safety_notes=["a", "b", "c"],  # max 2
            )


class TestTemplateFallback:
    """Tests for deterministic template fallback."""

    def test_zero_reps_fallback(self):
        """Fallback for zero reps."""
        stats = RepStats(
            rep_count=0,
            reps=[],
            avg_tempo_s=0.0,
            tempo_consistency=0.0,
            concentric_avg_s=0.0,
            eccentric_avg_s=0.0,
            confidence=0.5,
        )
        feedback = _template_fallback(stats)

        assert "No repetitions detected" in feedback.summary
        assert len(feedback.strengths) >= 1

    def test_low_reps_fallback(self):
        """Fallback for low rep count."""
        stats = RepStats(
            rep_count=3,
            reps=[RepTiming(start_s=1.0, end_s=3.0)] * 3,
            avg_tempo_s=2.0,
            tempo_consistency=0.1,
            concentric_avg_s=1.0,
            eccentric_avg_s=1.0,
            confidence=0.8,
        )
        feedback = _template_fallback(stats)

        assert "3 push-up" in feedback.summary
        assert len(feedback.strengths) >= 1

    def test_high_reps_fallback(self):
        """Fallback for high rep count."""
        stats = RepStats(
            rep_count=15,
            reps=[RepTiming(start_s=1.0, end_s=2.5)] * 15,
            avg_tempo_s=2.5,
            tempo_consistency=0.1,
            concentric_avg_s=1.2,
            eccentric_avg_s=1.3,
            confidence=0.9,
        )
        feedback = _template_fallback(stats)

        assert "15 push-ups" in feedback.summary

    def test_consistency_strength(self):
        """Consistent tempo should be noted as strength."""
        stats = RepStats(
            rep_count=10,
            reps=[RepTiming(start_s=i*2.0, end_s=i*2.0+2.0) for i in range(10)],
            avg_tempo_s=2.0,
            tempo_consistency=0.05,  # Very consistent
            concentric_avg_s=1.0,
            eccentric_avg_s=1.0,
            confidence=0.9,
        )
        feedback = _template_fallback(stats)

        assert any("consistent" in s.lower() for s in feedback.strengths)

    def test_inconsistent_improvement(self):
        """Inconsistent tempo should be noted as improvement."""
        stats = RepStats(
            rep_count=10,
            reps=[RepTiming(start_s=1.0, end_s=3.0)] * 10,
            avg_tempo_s=2.0,
            tempo_consistency=0.35,  # Inconsistent
            concentric_avg_s=1.0,
            eccentric_avg_s=1.0,
            confidence=0.8,
        )
        feedback = _template_fallback(stats)

        assert any("consistent" in s.lower() or "pacing" in s.lower() for s in feedback.improvements)

    def test_fast_eccentric_safety(self):
        """Fast eccentric should trigger safety note."""
        stats = RepStats(
            rep_count=10,
            reps=[RepTiming(start_s=1.0, end_s=2.0)] * 10,
            avg_tempo_s=2.0,
            tempo_consistency=0.1,
            concentric_avg_s=1.0,
            eccentric_avg_s=0.5,  # Too fast
            confidence=0.8,
        )
        feedback = _template_fallback(stats)

        assert any("injury risk" in s.lower() or "rapid lowering" in s.lower() for s in feedback.safety_notes)

    def test_fast_concentric_safety(self):
        """Very fast concentric should trigger safety note."""
        stats = RepStats(
            rep_count=10,
            reps=[RepTiming(start_s=1.0, end_s=1.5)] * 10,
            avg_tempo_s=1.5,
            tempo_consistency=0.1,
            concentric_avg_s=0.3,  # Very fast
            eccentric_avg_s=1.2,
            confidence=0.8,
        )
        feedback = _template_fallback(stats)

        assert any("explosive" in s.lower() or "compromise form" in s.lower() for s in feedback.safety_notes)


class TestSummarizeWithMock:
    """Tests for summarize_with_mock (testing helper)."""

    def test_mock_response_valid(self):
        """Test with valid mock response."""
        stats = RepStats(
            rep_count=10,
            reps=[RepTiming(start_s=1.0, end_s=3.0)],
            avg_tempo_s=2.0,
            tempo_consistency=0.1,
            concentric_avg_s=1.0,
            eccentric_avg_s=1.0,
            confidence=0.9,
        )

        mock_response = {
            "summary": "Mock summary",
            "strengths": ["Mock strength"],
            "improvements": ["Mock improvement"],
            "safety_notes": [],
        }

        feedback = summarize_with_mock(stats, mock_response=mock_response)

        assert feedback.summary == "Mock summary"
        assert feedback.strengths == ["Mock strength"]

    def test_mock_response_invalid_fallback(self):
        """Invalid mock response should trigger fallback."""
        stats = RepStats(
            rep_count=10,
            reps=[RepTiming(start_s=1.0, end_s=3.0)],
            avg_tempo_s=2.0,
            tempo_consistency=0.1,
            concentric_avg_s=1.0,
            eccentric_avg_s=1.0,
            confidence=0.9,
        )

        # Invalid: missing required fields
        mock_response = {"summary": "Only summary"}

        feedback = summarize_with_mock(stats, mock_response=mock_response)

        # Should fall back to template
        assert "push-up" in feedback.summary.lower()

    def test_no_mock_uses_fallback(self):
        """No mock response should use fallback."""
        stats = RepStats(
            rep_count=5,
            reps=[RepTiming(start_s=1.0, end_s=3.0)] * 5,
            avg_tempo_s=2.0,
            tempo_consistency=0.1,
            concentric_avg_s=1.0,
            eccentric_avg_s=1.0,
            confidence=0.8,
        )

        feedback = summarize_with_mock(stats, mock_response=None)

        assert "5 push-up" in feedback.summary


class TestSummarizeCore:
    """Tests for _summarize_with_client using an injected stub client.

    All tests run with no network access (the OpenAI client is never
    constructed and never called).
    """

    @staticmethod
    def _make_stats():
        return RepStats(
            rep_count=10,
            reps=[RepTiming(start_s=1.0, end_s=3.0)],
            avg_tempo_s=2.0,
            tempo_consistency=0.1,
            concentric_avg_s=1.0,
            eccentric_avg_s=1.0,
            confidence=0.9,
        )

    @staticmethod
    def _response(content: str):
        r = MagicMock()
        r.choices = [MagicMock(message=MagicMock(content=content))]
        r.usage = MagicMock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        return r

    def test_summarize_success(self):
        """Valid JSON from the model is parsed and returned."""
        from src.llm.coach import LLMClient, _summarize_with_client
        stub = MagicMock(spec=LLMClient)
        stub._call_with_retry.return_value = self._response(json.dumps({
            "summary": "LLM summary",
            "strengths": ["LLM strength"],
            "improvements": ["LLM improvement"],
            "safety_notes": [],
        }))

        feedback = _summarize_with_client(self._make_stats(), stub)
        assert feedback.summary == "LLM summary"
        assert stub._call_with_retry.call_count == 1

    def test_validation_failure_triggers_one_repair(self):
        """Invalid JSON triggers exactly one repair retry, then success."""
        from src.llm.coach import LLMClient, _summarize_with_client
        stub = MagicMock(spec=LLMClient)
        stub._call_with_retry.side_effect = [
            self._response("invalid json"),
            self._response(json.dumps({
                "summary": "Retry summary",
                "strengths": ["Retry strength"],
                "improvements": ["Retry improvement"],
                "safety_notes": [],
            })),
        ]

        feedback = _summarize_with_client(self._make_stats(), stub)
        assert feedback.summary == "Retry summary"
        assert stub._call_with_retry.call_count == 2

    def test_fallback_on_validation_failure_both_attempts(self):
        """Two bad responses fall back to the template."""
        from src.llm.coach import LLMClient, _summarize_with_client
        stub = MagicMock(spec=LLMClient)
        stub._call_with_retry.side_effect = [
            self._response("invalid json"),
            self._response("also invalid"),
        ]

        feedback = _summarize_with_client(self._make_stats(), stub)
        assert "push-up" in feedback.summary.lower()
        assert stub._call_with_retry.call_count == 2

    def test_fallback_when_model_raises(self):
        """An exception from the model falls back to the template."""
        from src.llm.coach import LLMClient, _summarize_with_client
        stub = MagicMock(spec=LLMClient)
        stub._call_with_retry.side_effect = RuntimeError("model exploded")

        feedback = _summarize_with_client(self._make_stats(), stub)
        assert "push-up" in feedback.summary.lower()

    def test_fallback_when_missing_api_key(self):
        """No API key short-circuits to the template fallback."""
        stats = self._make_stats()
        feedback = summarize(stats, timeout_s=10.0)
        assert "push-up" in feedback.summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
