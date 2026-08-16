"""
Unit tests for Component A: Video Preprocessing (features.py)
"""
import os
import tempfile

import cv2
import numpy as np
import pytest

from src.video.features import (
    VideoProcessingError,
    _compute_frame_difference_grid,
    _compute_motion_centroid,
    _compute_optical_flow_features,
    _letterbox_resize,
    extract_features,
    extract_features_from_frames,
)


class TestLetterboxResize:
    """Tests for letterbox resize function."""

    def test_letterbox_preserves_aspect_ratio(self):
        """Test that aspect ratio is preserved."""
        # Wide frame
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _letterbox_resize(frame, (160, 160))
        assert result.shape == (160, 160, 3)

        # Tall frame
        frame = np.zeros((200, 100, 3), dtype=np.uint8)
        result = _letterbox_resize(frame, (160, 160))
        assert result.shape == (160, 160, 3)

    def test_letterbox_padding_is_black(self):
        """Test that padding is black (zeros) for non-square frames."""
        # Wide frame - should have vertical padding
        frame = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White frame
        result = _letterbox_resize(frame, (160, 160))
        # Top and bottom should have black padding
        assert result[0, 80].sum() == 0  # Top padding
        assert result[-1, 80].sum() == 0  # Bottom padding

        # Tall frame - should have horizontal padding
        frame = np.ones((200, 100, 3), dtype=np.uint8) * 255
        result = _letterbox_resize(frame, (160, 160))
        # Left and right should have black padding
        assert result[80, 0].sum() == 0  # Left padding
        assert result[80, -1].sum() == 0  # Right padding

    def test_square_frame_no_padding(self):
        """Test square frame fits exactly."""
        frame = np.ones((160, 160, 3), dtype=np.uint8) * 128
        result = _letterbox_resize(frame, (160, 160))
        assert np.allclose(result, 128)


class TestFrameDifferenceGrid:
    """Tests for frame difference grid computation."""

    def test_identical_frames_zero_energy(self):
        """Identical frames should have zero energy."""
        frame = np.random.randint(0, 255, (160, 160), dtype=np.uint8)
        energy = _compute_frame_difference_grid(frame, frame)
        assert np.allclose(energy, 0.0)

    def test_different_frames_positive_energy(self):
        """Different frames should have positive energy."""
        frame1 = np.zeros((160, 160), dtype=np.uint8)
        frame2 = np.ones((160, 160), dtype=np.uint8) * 255
        energy = _compute_frame_difference_grid(frame1, frame2)
        assert np.all(energy > 0)
        assert energy.shape == (16,)

    def test_energy_normalized(self):
        """Energy should be normalized to [0, 1]."""
        frame1 = np.zeros((160, 160), dtype=np.uint8)
        frame2 = np.ones((160, 160), dtype=np.uint8) * 255
        energy = _compute_frame_difference_grid(frame1, frame2)
        assert np.all(energy <= 1.0)
        assert np.all(energy >= 0.0)


class TestMotionCentroid:
    """Tests for motion centroid computation."""

    def test_no_motion_returns_center(self):
        """No motion should return center (0.5)."""
        frame = np.ones((160, 160), dtype=np.uint8) * 128
        centroid = _compute_motion_centroid(frame, frame)
        assert centroid == 0.5

    def test_motion_at_top_returns_low_value(self):
        """Motion at top of frame should return low centroid."""
        frame1 = np.zeros((160, 160), dtype=np.uint8)
        frame2 = np.zeros((160, 160), dtype=np.uint8)
        # Add motion at top
        frame2[10:30, 50:110] = 255
        centroid = _compute_motion_centroid(frame1, frame2)
        assert centroid < 0.5

    def test_motion_at_bottom_returns_high_value(self):
        """Motion at bottom of frame should return high centroid."""
        frame1 = np.zeros((160, 160), dtype=np.uint8)
        frame2 = np.zeros((160, 160), dtype=np.uint8)
        # Add motion at bottom
        frame2[130:150, 50:110] = 255
        centroid = _compute_motion_centroid(frame1, frame2)
        assert centroid > 0.5


class TestOpticalFlowFeatures:
    """Tests for optical flow feature computation."""

    def test_identical_frames_zero_flow(self):
        """Identical frames should have zero flow."""
        frame = np.random.randint(0, 255, (160, 160), dtype=np.uint8)
        mag, vert = _compute_optical_flow_features(frame, frame)
        assert mag < 0.01  # Near zero
        assert abs(vert) < 0.01

    def test_vertical_motion_detected(self):
        """Vertical motion should be detected in vertical component."""
        frame1 = np.zeros((160, 160), dtype=np.uint8)
        frame2 = np.zeros((160, 160), dtype=np.uint8)
        # Create vertical translation pattern
        frame1[50:110, 50:110] = 200
        frame2[70:130, 50:110] = 200  # Moved down
        mag, vert = _compute_optical_flow_features(frame1, frame2)
        assert mag > 0
        assert vert > 0  # Positive = downward

    def test_upward_motion_negative_vertical(self):
        """Upward motion should give negative vertical component."""
        frame1 = np.zeros((160, 160), dtype=np.uint8)
        frame2 = np.zeros((160, 160), dtype=np.uint8)
        frame1[70:130, 50:110] = 200
        frame2[50:110, 50:110] = 200  # Moved up
        mag, vert = _compute_optical_flow_features(frame1, frame2)
        assert mag > 0
        assert vert < 0  # Negative = upward


class TestExtractFeaturesFromFrames:
    """Tests for feature extraction from frame list."""

    def test_single_frame(self):
        """Single frame should return zeros."""
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        features = extract_features_from_frames([frame])
        assert features.shape == (1, 19)
        assert np.allclose(features[0], 0.0)

    def test_two_frames(self):
        """Two frames should return features for second frame."""
        frame1 = np.zeros((240, 320, 3), dtype=np.uint8)
        frame2 = np.ones((240, 320, 3), dtype=np.uint8) * 255
        features = extract_features_from_frames([frame1, frame2])
        assert features.shape == (2, 19)
        # First frame zeros
        assert np.allclose(features[0], 0.0)
        # Second frame has features
        assert np.any(features[1] > 0)

    def test_feature_ranges(self):
        """Features should be in expected ranges."""
        frames = [
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        features = extract_features_from_frames(frames)
        assert features.shape == (10, 19)
        # Grid energies [0, 1]
        assert np.all(features[:, :16] >= 0.0)
        assert np.all(features[:, :16] <= 1.0)
        # Centroid [0, 1]
        assert np.all(features[:, 16] >= 0.0)
        assert np.all(features[:, 16] <= 1.0)
        # Flow channels are clipped to [-2, 2] (causal, no clip-level
        # mean-centering — see module docstring)
        assert np.all(features[:, 17:] >= -2.0)
        assert np.all(features[:, 17:] <= 2.0)


class TestExtractFeatures:
    """Tests for main extract_features function."""

    def test_nonexistent_file(self):
        """Should raise VideoProcessingError for nonexistent file."""
        with pytest.raises(VideoProcessingError, match="not found"):
            extract_features("/nonexistent/path/video.mp4")

    def test_corrupt_file(self):
        """Should raise VideoProcessingError for corrupt file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"not a video file")
            temp_path = f.name
        try:
            with pytest.raises(VideoProcessingError, match="Cannot open"):
                extract_features(temp_path)
        finally:
            os.unlink(temp_path)

    def test_empty_file(self):
        """Should raise VideoProcessingError for empty file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(VideoProcessingError):
                extract_features(temp_path)
        finally:
            os.unlink(temp_path)

    def test_synthetic_video(self):
        """Test with a synthetic video created in memory."""
        # Create a simple test video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            # Write a simple video using OpenCV
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 15.0, (320, 240))

            # Write 30 frames with some motion
            for i in range(30):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                # Moving square
                y = 100 + i * 2
                x = 100 + i
                frame[y:y+40, x:x+40] = [0, 255, 0]
                out.write(frame)
            out.release()

            # Extract features
            features = extract_features(temp_path, target_fps=15, max_seconds=2.0)
            assert features.shape[1] == 19
            assert features.shape[0] > 0
            assert features.dtype == np.float32
            # Should have 30 frames at 15fps for 2 seconds = 30 frames
            # But we limit to max_seconds=2.0 so 30 frames
            assert features.shape[0] <= 30

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_brightness_invariance(self):
        """A global brightness shift should barely change the features.

        Frame differences and Farneback flow are invariant to a constant
        intensity offset, so a darker copy of the same clip should produce a
        near-identical feature vector.
        """
        def _make_clip(motion_fn):
            import os as _os
            import tempfile as _tf
            fd, p = _tf.mkstemp(suffix=".mp4")
            _os.close(fd)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(p, fourcc, 15.0, (320, 240))
            for i in range(20):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                frame[100:140, 100:140] = motion_fn(i)
                out.write(frame)
            out.release()
            return p

        def _del(p):
            if os.path.exists(p):
                os.unlink(p)

        p1 = _make_clip(lambda i: (30 + i * 4, 60, 90))      # brighter movement
        p2 = _make_clip(lambda i: (10 + i * 4, 40, 70))      # darker offset copy
        try:
            f1 = extract_features(p1, target_fps=15, max_seconds=2.0)
            f2 = extract_features(p2, target_fps=15, max_seconds=2.0)
            # Motion-driven features must track the same movement closely.
            assert np.max(np.abs(f1 - f2)) < 0.05
        finally:
            _del(p1)
            _del(p2)

    def test_deterministic(self):
        """Same video should always yield same features."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_path, fourcc, 15.0, (320, 240))
            for _ in range(20):
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                frame[100:140, 100:140] = [0, 255, 0]
                out.write(frame)
            out.release()

            features1 = extract_features(temp_path, target_fps=15, max_seconds=2.0)
            features2 = extract_features(temp_path, target_fps=15, max_seconds=2.0)
            assert np.allclose(features1, features2)

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
