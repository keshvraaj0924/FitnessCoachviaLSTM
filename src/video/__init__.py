"""Video preprocessing module."""
from .features import (
    VideoProcessingError,
    extract_features,
    extract_features_from_frames,
)

__all__ = ["VideoProcessingError", "extract_features", "extract_features_from_frames"]
