"""Configuration management for the exercise analysis API."""
from pathlib import Path

from pydantic import AliasChoices, ConfigDict, Field
from pydantic_settings import BaseSettings


class ExerciseSpec(BaseSettings):
    """Per-exercise configuration (checkpoint + display name)."""

    id: str
    name: str
    checkpoint: Path


# The set of exercises the system supports. Each has its own trained LSTM
# checkpoint (same architecture, same feature vector, distinct weights tuned
# to that exercise's motion signature). Synthetic training data and coaching
# prompts are also per-exercise.
DEFAULT_EXERCISES = [
    ExerciseSpec(id="pushup", name="Push-up", checkpoint=Path("checkpoints/lstm_pushup.pt")),
    ExerciseSpec(id="squat", name="Squat", checkpoint=Path("checkpoints/lstm_squat.pt")),
    ExerciseSpec(id="bicep_curl", name="Bicep Curl", checkpoint=Path("checkpoints/lstm_bicep_curl.pt")),
    ExerciseSpec(id="jumping_jack", name="Jumping Jack", checkpoint=Path("checkpoints/lstm_jumping_jack.pt")),
]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model settings
    checkpoint_path: Path = Field(
        default=Path("checkpoints/lstm_pushup.pt"),
        description="Path to the default (push-up) LSTM checkpoint"
    )
    model_version: str = Field(
        default="1.0.0",
        description="Model version identifier"
    )
    lstm_bidirectional: bool = Field(
        default=False,
        description="Use bidirectional LSTM (offline only; real-time streaming "
                    "requires the causal/unidirectional model)"
    )
    target_fps: int = Field(
        default=15,
        description="Target FPS for video resampling"
    )
    max_seconds: float = Field(
        default=30.0,
        description="Maximum video duration to process (seconds)"
    )
    feature_dim: int = Field(
        default=19,
        description="Feature vector dimension"
    )
    lstm_hidden_size: int = Field(
        default=64,
        description="LSTM hidden size"
    )
    lstm_num_layers: int = Field(
        default=2,
        description="Number of LSTM layers"
    )
    lstm_dropout: float = Field(
        default=0.2,
        description="LSTM dropout rate"
    )

    # Rep counting thresholds
    min_concentric_frames: int = Field(
        default=3,
        description="Minimum frames in concentric phase"
    )
    min_eccentric_frames: int = Field(
        default=3,
        description="Minimum frames in eccentric phase"
    )
    min_idle_frames: int = Field(
        default=5,
        description="Minimum frames in idle phase"
    )
    stream_confidence_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Minimum softmax confidence to accept a phase transition "
                    "in the streaming state machine (frames below this are "
                    "treated as 'stay in current state').",
    )

    # API settings
    max_upload_mb: int = Field(
        default=50,
        description="Maximum upload size in MB"
    )
    allowed_content_types: list[str] = Field(
        default=["video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska"],
        description="Allowed video MIME types"
    )
    allowed_extensions: list[str] = Field(
        default=[".mp4", ".mov", ".avi", ".mkv"],
        description="Accepted video extensions (fallback when MIME is octet-stream)"
    )
    host: str = Field(
        default="0.0.0.0",
        description="API host"
    )
    port: int = Field(
        default=8000,
        description="API port"
    )

    # LLM settings
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (set via environment)"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use",
        validation_alias=AliasChoices("OPENAI_CHAT_MODEL", "OPENAI_MODEL"),
    )
    openai_timeout_s: float = Field(
        default=10.0,
        description="OpenAI request timeout in seconds"
    )
    openai_max_tokens: int = Field(
        default=300,
        description="Maximum tokens for LLM response"
    )
    openai_temperature: float = Field(
        default=0.3,
        description="LLM temperature"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    @property
    def exercises(self) -> list[ExerciseSpec]:
        """The full set of supported exercises (id, name, checkpoint)."""
        return DEFAULT_EXERCISES

    def exercise_spec(self, exercise_id: str) -> ExerciseSpec | None:
        """Look up an exercise by id, case-insensitively."""
        return next((e for e in self.exercises if e.id == exercise_id), None)

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Global settings instance
settings = Settings()
