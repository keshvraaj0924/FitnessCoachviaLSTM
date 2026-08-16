"""
Unit tests for Component B: Rep Counting Logic (reps.py)
"""
import numpy as np
import pytest

from src.model.reps import (
    Phase,
    RepCounter,
    RepTiming,
    count_reps_from_logits,
    evaluate_rep_counting,
    generate_synthetic_label_sequence,
)


class TestPhaseEnum:
    """Tests for Phase enum."""

    def test_phase_values(self):
        assert Phase.IDLE == 0
        assert Phase.CONCENTRIC == 1
        assert Phase.ECCENTRIC == 2


class TestRepTiming:
    """Tests for RepTiming dataclass."""

    def test_duration(self):
        rep = RepTiming(start_s=1.0, end_s=3.5)
        assert rep.duration() == 2.5


class TestRepCounter:
    """Tests for RepCounter state machine."""

    def test_single_rep(self):
        """Test counting a single clean rep."""
        # Sequence: idle -> concentric -> eccentric -> idle
        sequence = (
            [Phase.IDLE] * 10 +
            [Phase.CONCENTRIC] * 5 +
            [Phase.ECCENTRIC] * 5 +
            [Phase.IDLE] * 10
        )

        counter = RepCounter(min_concentric_frames=3, min_eccentric_frames=3, min_idle_frames=5, fps=15)
        result = counter.get_results(sequence)

        assert result.rep_count == 1
        assert len(result.reps) == 1
        assert result.reps[0].start_s == pytest.approx(10/15, rel=0.1)
        assert result.reps[0].end_s == pytest.approx(20/15, rel=0.1)

    def test_multiple_reps(self):
        """Test counting multiple reps."""
        sequence = (
            [Phase.IDLE] * 10 +
            [Phase.CONCENTRIC] * 5 + [Phase.ECCENTRIC] * 5 + [Phase.IDLE] * 8 +
            [Phase.CONCENTRIC] * 5 + [Phase.ECCENTRIC] * 5 + [Phase.IDLE] * 8 +
            [Phase.CONCENTRIC] * 5 + [Phase.ECCENTRIC] * 5 +
            [Phase.IDLE] * 10
        )

        counter = RepCounter(min_concentric_frames=3, min_eccentric_frames=3, min_idle_frames=5, fps=15)
        result = counter.get_results(sequence)

        assert result.rep_count == 3
        assert len(result.reps) == 3

    def test_insufficient_concentric_frames(self):
        """Too few concentric frames should not count as rep."""
        sequence = (
            [Phase.IDLE] * 10 +
            [Phase.CONCENTRIC] * 2 +  # Only 2 frames, need 3
            [Phase.ECCENTRIC] * 5 +
            [Phase.IDLE] * 10
        )

        counter = RepCounter(min_concentric_frames=3, min_eccentric_frames=3, min_idle_frames=5, fps=15)
        result = counter.get_results(sequence)

        assert result.rep_count == 0

    def test_insufficient_eccentric_frames(self):
        """Too few eccentric frames should not count as rep."""
        sequence = (
            [Phase.IDLE] * 10 +
            [Phase.CONCENTRIC] * 5 +
            [Phase.ECCENTRIC] * 2 +  # Only 2 frames, need 3
            [Phase.IDLE] * 10
        )

        counter = RepCounter(min_concentric_frames=3, min_eccentric_frames=3, min_idle_frames=5, fps=15)
        result = counter.get_results(sequence)

        assert result.rep_count == 0

    def test_insufficient_idle_between_reps(self):
        """Too few idle frames between reps should not separate reps.

        Both push-ups happen with no validated idle run between them, so they
        form one continuous set. The state machine only counts a rep that
        reaches the committed ECCENTRIC state with a real (validated) return to
        IDLE, so both are absorbed into the single ongoing movement and counted
        as one rep.
        """
        sequence = (
            [Phase.IDLE] * 10 +
            [Phase.CONCENTRIC] * 5 + [Phase.ECCENTRIC] * 5 +
            [Phase.IDLE] * 3 +  # Only 3 idle frames, need 5
            [Phase.CONCENTRIC] * 5 + [Phase.ECCENTRIC] * 5 +
            [Phase.IDLE] * 10
        )

        counter = RepCounter(min_concentric_frames=3, min_eccentric_frames=3, min_idle_frames=5, fps=15)
        result = counter.get_results(sequence)

        # One continuous set, not two separately-timed reps
        assert result.rep_count == 1

    def test_invalid_transition_resets(self):
        """Invalid transitions (e.g., concentric -> idle) should reset."""
        sequence = (
            [Phase.IDLE] * 10 +
            [Phase.CONCENTRIC] * 5 +
            [Phase.IDLE] * 5 +  # Invalid: concentric -> idle without eccentric
            [Phase.CONCENTRIC] * 5 + [Phase.ECCENTRIC] * 5 +
            [Phase.IDLE] * 10
        )

        counter = RepCounter(min_concentric_frames=3, min_eccentric_frames=3, min_idle_frames=5, fps=15)
        result = counter.get_results(sequence)

        # First attempt invalid, second should count
        assert result.rep_count == 1

    def test_eccentric_without_concentric(self):
        """Eccentric without prior concentric should not count."""
        sequence = (
            [Phase.IDLE] * 10 +
            [Phase.ECCENTRIC] * 5 +  # No concentric before
            [Phase.IDLE] * 10
        )

        counter = RepCounter(min_concentric_frames=3, min_eccentric_frames=3, min_idle_frames=5, fps=15)
        result = counter.get_results(sequence)

        assert result.rep_count == 0

    def test_noisy_sequence(self):
        """Test with some noise in predictions."""
        # Clean sequence with a few wrong predictions
        sequence = (
            [Phase.IDLE] * 10 +
            [Phase.CONCENTRIC] * 4 + [Phase.IDLE] + [Phase.CONCENTRIC] +  # One idle in middle
            [Phase.ECCENTRIC] * 5 +
            [Phase.IDLE] * 10
        )

        counter = RepCounter(min_concentric_frames=3, min_eccentric_frames=3, min_idle_frames=5, fps=15)
        result = counter.get_results(sequence)

        # Should still count the rep (noise filtered by min frame requirements)
        assert result.rep_count == 1

    def test_confidence_calculation(self):
        """Test confidence is computed from non-idle frames."""
        sequence = [Phase.IDLE] * 5 + [Phase.CONCENTRIC] * 5 + [Phase.ECCENTRIC] * 5 + [Phase.IDLE] * 5
        confidences = [0.9] * 5 + [0.8] * 5 + [0.85] * 5 + [0.95] * 5

        counter = RepCounter(fps=15)
        result = counter.get_results(sequence, confidences)

        # Non-idle frames: 10 frames with confidences [0.8]*5 + [0.85]*5
        expected_conf = (0.8 * 5 + 0.85 * 5) / 10
        assert result.confidence == pytest.approx(expected_conf)


class TestCountRepsFromLogits:
    """Tests for count_reps_from_logits convenience function."""

    def test_from_logits_2d(self):
        """Test with 2D logits (T, 3)."""
        T = 30
        # Create logits that clearly predict the phases
        logits = np.zeros((T, 3))
        logits[:10, 0] = 10  # Idle
        logits[10:15, 1] = 10  # Concentric
        logits[15:20, 2] = 10  # Eccentric
        logits[20:, 0] = 10  # Idle

        result = count_reps_from_logits(logits, fps=15)
        assert result.rep_count == 1

    def test_from_logits_3d(self):
        """Test with 3D logits (1, T, 3)."""
        T = 30
        logits = np.zeros((1, T, 3))
        logits[0, :10, 0] = 10
        logits[0, 10:15, 1] = 10
        logits[0, 15:20, 2] = 10
        logits[0, 20:, 0] = 10

        result = count_reps_from_logits(logits, fps=15)
        assert result.rep_count == 1


class TestEvaluateRepCounting:
    """Tests for evaluate_rep_counting function."""

    def test_perfect_predictions(self):
        """Perfect predictions should give F1=1, MAE=0."""
        true = [
            [0]*10 + [1]*5 + [2]*5 + [0]*10,
            [0]*8 + [1]*6 + [2]*6 + [0]*8,
        ]
        pred = [list(t) for t in true]

        metrics = evaluate_rep_counting(true, pred)

        # Use pytest.approx for floating point comparison
        assert metrics['macro_f1'] == pytest.approx(1.0)
        assert metrics['rep_count_mae'] == pytest.approx(0.0)
        for f1 in metrics['per_class_f1'].values():
            assert f1 == pytest.approx(1.0)

    def test_all_idle_predictions(self):
        """All idle predictions should give F1=0 for other classes."""
        true = [
            [0]*10 + [1]*5 + [2]*5 + [0]*10,
        ]
        pred = [
            [0]*20,  # All idle
        ]

        metrics = evaluate_rep_counting(true, pred)

        assert metrics['per_class_f1']['idle'] > 0
        assert metrics['per_class_f1']['concentric'] == 0.0
        assert metrics['per_class_f1']['eccentric'] == 0.0
        assert metrics['rep_count_mae'] == 1.0  # True has 1 rep, pred has 0


class TestGenerateSyntheticLabelSequence:
    """Tests for synthetic label sequence generation."""

    def test_basic_generation(self):
        """Test basic sequence generation."""
        seq = generate_synthetic_label_sequence(num_reps=3, fps=15)

        assert len(seq) > 0
        # Should contain all three phases
        assert Phase.IDLE in seq
        assert Phase.CONCENTRIC in seq
        assert Phase.ECCENTRIC in seq

    def test_rep_count_matches(self):
        """Generated sequence should have correct number of reps."""
        for num_reps in [1, 3, 5, 10]:
            seq = generate_synthetic_label_sequence(num_reps=num_reps, fps=15, noise_prob=0.0)

            counter = RepCounter(fps=15)
            result = counter.get_results(seq)

            # With no noise, should match exactly
            assert result.rep_count == num_reps

    def test_noise_added(self):
        """Noise should be added when requested."""
        seq_clean = generate_synthetic_label_sequence(num_reps=5, fps=15, noise_prob=0.0)
        seq_noisy = generate_synthetic_label_sequence(num_reps=5, fps=15, noise_prob=0.5)

        # Noisy should have some different labels
        assert seq_clean != seq_noisy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
