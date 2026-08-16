"""
Component A: Video Preprocessing with OpenCV

Extracts fixed-rate feature vectors from video files for push-up analysis.
Features: frame-difference energy (4x4 grid), vertical centroid of motion,
optical flow magnitude and vertical component.

Two consumption modes share the exact same per-frame computation so that the
feature distribution at serving time equals the one the model was trained on:

* `extract_features`        - batch: a whole video file -> (T, D) array.
* `StreamingFeatureExtractor` - live: frame-by-frame -> per-frame vectors for
                               the real-time WebSocket endpoint.

Every feature is *causal*: it is computed from at most the current and
previous sampled frame, with no look-ahead. In particular optical-flow
features are not mean-centred over the clip (that would leak future frames
into the present), so batch and stream are numerically identical for the
same frame pair.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Feature configuration
TARGET_RESOLUTION = (160, 160)  # width, height
GRID_SIZE = (4, 4)  # 4x4 spatial grid for frame differences
MAX_DISPLACEMENT = 30.0  # Max expected optical flow displacement for normalization
FLOW_CLIP = 2.0  # clip bounds applied to flow channels after normalization


class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""


# OpenCV's CAP_PROP_ORIENTATION_META returns a VideoCaptureOrientations enum
# value, NOT raw degrees:
#   0 = UNKNOWN, 1 = 0 deg, 2 = 90 deg, 3 = 180 deg, 4 = 270 deg
_ORIENTATION_ENUM_TO_DEG = {0: 0, 1: 0, 2: 90, 3: 180, 4: 270}


def _get_rotation_metadata(cap: cv2.VideoCapture) -> int:
    """
    Check for rotation metadata in the video container.

    Phone videos often carry a rotation tag (e.g. a portrait clip recorded with
    a landscape sensor). Returns the rotation angle in degrees (0, 90, 180,
    270), mapping OpenCV's orientation enum to degrees. Falls back to 0 when
    the tag is absent or unreadable.
    """
    try:
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        return _ORIENTATION_ENUM_TO_DEG.get(rotation, 0)
    except Exception:
        return 0


def _apply_rotation(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply rotation to frame if needed."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _letterbox_resize(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize frame preserving aspect ratio with letterboxing (black padding).
    target_size: (width, height)
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size

    # Calculate scale to fit within target while preserving aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create padded canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def _compute_frame_difference_grid(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    grid_size: tuple[int, int] = GRID_SIZE
) -> np.ndarray:
    """
    Compute frame difference energy over a spatial grid.
    Returns normalized energy per grid cell (16 values for 4x4).
    """
    h, w = prev_gray.shape
    grid_h, grid_w = grid_size
    cell_h, cell_w = h // grid_h, w // grid_w

    # Compute absolute difference
    diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)

    # Compute energy per grid cell
    energies = np.zeros(grid_h * grid_w, dtype=np.float32)
    for i in range(grid_h):
        for j in range(grid_w):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = diff[y1:y2, x1:x2]
            # Normalize by max possible (255) and cell area
            energies[i * grid_w + j] = cell.sum() / (255.0 * cell.size)

    return energies


def _compute_motion_centroid(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Compute vertical centroid of motion (frame difference).
    Returns normalized Y coordinate (0=top, 1=bottom).
    """
    diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
    h = diff.shape[0]

    # Threshold to get motion regions
    _, motion_mask = cv2.threshold(diff, 10.0, 255.0, cv2.THRESH_BINARY)

    # Find moments
    moments = cv2.moments(motion_mask)
    if moments['m00'] > 0:
        cy = moments['m01'] / moments['m00']
        return cy / h  # Normalize to [0, 1]
    return 0.5  # Default to center if no motion


def _compute_optical_flow_features(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float]:
    """
    Compute optical flow magnitude and vertical component.
    Returns (mean_magnitude, mean_vertical_component) normalized.
    """
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    # Magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Mean magnitude (normalized by max expected displacement)
    mean_mag = magnitude.mean() / MAX_DISPLACEMENT

    # Mean vertical component (sin of angle * magnitude)
    vertical_component = (magnitude * np.sin(angle)).mean() / MAX_DISPLACEMENT

    return float(mean_mag), float(vertical_component)


def compute_frame_features(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Compute the 19-dim feature vector for a frame pair (causal).

    Used by both batch extraction and the streaming extractor so the two
    modes are numerically identical.

    Returns a float32 vector of length 19:
      [0:16]  frame-difference grid energy (4x4, normalized to [0, 1])
      [16]    vertical centroid of motion (normalized 0-1)
      [17]    optical flow mean magnitude (normalized)
      [18]    optical flow mean vertical component (normalized)
    """
    grid_energy = _compute_frame_difference_grid(prev_gray, curr_gray)
    centroid_y = _compute_motion_centroid(prev_gray, curr_gray)
    flow_mag, flow_vert = _compute_optical_flow_features(prev_gray, curr_gray)

    feat = np.zeros(19, dtype=np.float32)
    feat[:16] = grid_energy
    feat[16] = centroid_y
    feat[17] = flow_mag
    feat[18] = flow_vert

    # Final normalization: grid energy and centroid already live in [0, 1];
    # flow features are brightness-invariant by construction (Farneback flow is
    # a Taylor-series velocity estimator) and are clipped to a fixed range.
    feat[:16] = np.clip(feat[:16], 0.0, 1.0)
    feat[16] = np.clip(feat[16], 0.0, 1.0)
    feat[17:] = np.clip(feat[17:], -FLOW_CLIP, FLOW_CLIP)

    return feat


def _prepare_gray(frame: np.ndarray) -> np.ndarray:
    """Letterbox to working resolution and convert to grayscale."""
    frame = _letterbox_resize(frame, TARGET_RESOLUTION)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def extract_features(
    path: str,
    target_fps: int = 15,
    max_seconds: float = 30.0,
) -> np.ndarray:
    """
    Extract feature vectors from a video file.

    Args:
        path: Path to video file
        target_fps: Target sampling rate (frames per second)
        max_seconds: Maximum duration to process

    Returns:
        Float32 array of shape (T, 19) where T = target_fps * max_seconds

    Raises:
        VideoProcessingError: If video cannot be read or is corrupt
    """
    # Validate input
    video_path = Path(path)
    if not video_path.exists():
        raise VideoProcessingError(f"Video file not found: {path}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"Cannot open video file: {path}")

    try:
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Handle invalid FPS (common in phone videos)
        if fps <= 0 or fps > 120 or np.isnan(fps):
            logger.warning(f"Invalid FPS ({fps}) for {path}, estimating from frame count")
            fps = 30.0

        # Check for rotation metadata
        rotation = _get_rotation_metadata(cap)
        if rotation != 0:
            logger.info(f"Video has rotation metadata: {rotation} degrees")
            # Swap width/height for 90/270 rotation
            if rotation in (90, 270):
                width, height = height, width

        # Handle zero-length or corrupt video
        if total_frames <= 0 or width <= 0 or height <= 0:
            raise VideoProcessingError(f"Invalid video dimensions: frames={total_frames}, {width}x{height}")

        # Calculate actual duration and target frame count
        actual_duration = total_frames / fps
        process_duration = min(actual_duration, max_seconds)
        target_frame_count = int(process_duration * target_fps)

        # If video is shorter than 1 frame at target_fps, still process at least 1 frame
        target_frame_count = max(target_frame_count, 1)

        # Read all frames (or up to max_seconds)
        frames_to_read = min(total_frames, int(max_seconds * fps) + 1)
        frames = []

        for _ in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                break
            # Apply rotation if needed
            if rotation != 0:
                frame = _apply_rotation(frame, rotation)
            frames.append(frame)

        if not frames:
            raise VideoProcessingError("No frames could be read from video")

        gray_frames = [_prepare_gray(f) for f in frames]

        # Uniformly sample to target_fps
        # We have len(frames) frames at original fps, need target_frame_count frames
        if len(gray_frames) == 1:
            # Only one frame - duplicate it
            sampled_indices = [0] * target_frame_count
        else:
            # Uniform sampling indices
            sampled_indices = np.linspace(
                0, len(gray_frames) - 1, target_frame_count, dtype=int
            )

        sampled_gray = [gray_frames[i] for i in sampled_indices]

        # Extract features for each sampled frame
        # First frame has no previous, so use zero features
        features = np.zeros((target_frame_count, 19), dtype=np.float32)

        for t in range(target_frame_count):
            if t == 0:
                features[t] = 0.0
            else:
                features[t] = compute_frame_features(
                    sampled_gray[t - 1], sampled_gray[t]
                )

        return features.astype(np.float32)

    finally:
        cap.release()


def extract_features_from_frames(frames: list[np.ndarray], target_fps: int = 15) -> np.ndarray:
    """
    Extract features from a list of pre-loaded frames (for testing).
    Frames should be BGR numpy arrays.
    """
    if not frames:
        return np.zeros((1, 19), dtype=np.float32)

    gray_frames = [_prepare_gray(f) for f in frames]

    target_frame_count = len(gray_frames)
    features = np.zeros((target_frame_count, 19), dtype=np.float32)

    for t in range(target_frame_count):
        if t == 0:
            features[t] = 0.0
        else:
            features[t] = compute_frame_features(gray_frames[t - 1], gray_frames[t])

    return features.astype(np.float32)


class StreamingFeatureExtractor:
    """
    Causal, frame-by-frame feature extraction for live video.

    Frames arrive at some source rate (e.g. 30 fps from a webcam). The
    extractor sub-samples to `target_fps` and produces one feature vector per
    sampled frame, computed from the previous sampled frame — exactly the same
    computation the batch path performs, so live and batch features share a
    distribution.

    Thread-safety: a single instance must be owned by one stream session.
    """

    def __init__(self, target_fps: int = 15):
        self.target_fps = target_fps
        self._prev_gray: np.ndarray | None = None
        self.frame_count = 0
        self._sampled = 0           # how many feature vectors emitted
        self._source_fps = 30.0
        self._ratio = 2.0           # source_fps / target_fps

    def reset(self):
        """Start a new stream session (clears previous frame state)."""
        self._prev_gray = None
        self.frame_count = 0
        self._sampled = 0
        self._source_fps = 30.0
        self._ratio = 2.0

    @property
    def sample_count(self) -> int:
        """Number of feature vectors emitted so far (0-indexed as frame indices)."""
        return self._sampled

    def _is_sampling_frame(self) -> bool:
        """
        Should this incoming source frame produce a feature vector?

        Matches the batch path, which uniformly samples the clip's frames to
        `target_fps` via ``np.linspace`` — i.e. it samples source frame
        ``round(k * source_fps / target_fps)`` for k = 0, 1, 2, ...
        Sampling the same indices here keeps live features numerically
        identical to batch features for the same frame pair.
        """
        target_frame = round(self._sampled * self._ratio)
        return self.frame_count == target_frame

    def process_frame(self, frame_bgr: np.ndarray, source_fps: float = 30.0) -> np.ndarray | None:
        """
        Feed one frame. Returns a (19,) feature vector when this frame is a
        sampling point, otherwise None (frame skipped).

        The first sampled frame yields zeros (no previous frame), matching the
        batch path.
        """
        if source_fps is None or source_fps <= 0 or np.isnan(source_fps):
            source_fps = 30.0
        self._source_fps = source_fps
        self._ratio = source_fps / self.target_fps

        if not self._is_sampling_frame():
            self.frame_count += 1
            return None

        self.frame_count += 1
        self._sampled += 1

        gray = _prepare_gray(frame_bgr)
        if self._prev_gray is None:
            feat = np.zeros(19, dtype=np.float32)
        else:
            feat = compute_frame_features(self._prev_gray, gray)

        self._prev_gray = gray
        return feat.astype(np.float32)
