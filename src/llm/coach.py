"""
Component C: Coaching Summary with LLM

Turns computed rep statistics into structured coaching feedback using OpenAI.
Includes validation, retry logic, and deterministic fallback.
"""
import logging
import os
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Schemas
# =============================================================================

class RepTiming(BaseModel):
    """Single repetition timing."""
    start_s: float
    end_s: float


class RepStats(BaseModel):
    """Computed statistics from the pipeline (never invented by LLM)."""
    rep_count: int
    reps: list[RepTiming]
    avg_tempo_s: float
    tempo_consistency: float  # Coefficient of variation of rep durations
    concentric_avg_s: float
    eccentric_avg_s: float
    confidence: float  # Model confidence (0-1)


class CoachFeedback(BaseModel):
    """Structured coaching feedback from LLM."""
    summary: str = Field(..., min_length=10, max_length=300)
    strengths: list[str] = Field(..., min_length=1, max_length=3)
    improvements: list[str] = Field(..., min_length=1, max_length=3)
    safety_notes: list[str] = Field(default_factory=list, max_length=2)

    def to_dict(self) -> dict:
        return self.model_dump()


# =============================================================================
# Template Fallback (Deterministic)
# =============================================================================

_EXERCISE_VERB = {
    "pushup": ("push-up", "pushing up", "lowering down"),
    "squat": ("squat", "standing up", "lowering down"),
    "bicep_curl": ("bicep curl", "curling up", "lowering down"),
    "jumping_jack": ("jumping jack", "jumping up", "returning down"),
}


def _exercise_name(exercise: str) -> str:
    """Human-readable name for an exercise id."""
    spec = {
        "pushup": "push-up",
        "squat": "squat",
        "bicep_curl": "bicep curl",
        "jumping_jack": "jumping jack",
    }
    return spec.get(exercise, "exercise")


def _template_fallback(stats: RepStats, exercise: str = "pushup") -> CoachFeedback:
    """
    Deterministic template-based fallback when LLM fails.
    Always returns valid CoachFeedback.
    """
    movement_noun = _EXERCISE_VERB.get(exercise, ("exercise", "up", "down"))[0]
    up_verb = _EXERCISE_VERB.get(exercise, ("exercise", "up", "down"))[1]
    down_verb = _EXERCISE_VERB.get(exercise, ("exercise", "up", "down"))[2]

    # Summary
    if stats.rep_count == 0:
        summary = "No repetitions detected. Please check video quality and positioning."
    elif stats.rep_count < 5:
        summary = f"Completed {stats.rep_count} {movement_noun}{'s' if stats.rep_count != 1 else ''} - a good start!"
    else:
        summary = f"Solid set of {stats.rep_count} {movement_noun}s completed."

    # Strengths
    strengths = []
    if stats.tempo_consistency < 0.2:
        strengths.append("Consistent tempo across repetitions")
    if stats.eccentric_avg_s >= 1.5:
        strengths.append("Well-controlled lowering phase")
    if stats.concentric_avg_s >= 1.0:
        strengths.append(f"Controlled {up_verb} phase")
    if stats.confidence > 0.85:
        strengths.append("Clear movement pattern detected")
    if not strengths:
        strengths.append("Completed the set")

    # Improvements
    improvements = []
    if stats.tempo_consistency >= 0.25:
        improvements.append("Work on consistent pacing between reps")
    if stats.eccentric_avg_s < 1.5:
        improvements.append(f"Slow down the {down_verb} phase to 2-3 seconds")
    if stats.concentric_avg_s < 1.0:
        improvements.append(f"Control the {up_verb} phase - aim for 1-2 seconds")
    if stats.rep_count < 8 and stats.rep_count > 0:
        improvements.append("Gradually increase rep count over sessions")
    if not improvements:
        improvements.append("Maintain current form and build volume")

    # Safety notes
    safety_notes = []
    if stats.eccentric_avg_s < 1.0:
        safety_notes.append("Rapid lowering increases injury risk")
    if stats.concentric_avg_s < 0.5:
        safety_notes.append("Explosive movement may compromise form - control the motion")

    return CoachFeedback(
        summary=summary,
        strengths=strengths[:3],
        improvements=improvements[:3],
        safety_notes=safety_notes[:2],
    )


# =============================================================================
# Prompt Loading
# =============================================================================

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(PROMPT_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _round(value, precision=0):
    """Jinja filter: round a numeric value to `precision` decimals."""
    return round(float(value), precision)


_jinja_env.filters['round'] = _round


def _load_prompt(version: str = "v1"):
    """Load the raw prompt template (not yet rendered)."""
    template_path = PROMPT_DIR / f"coach_{version}.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    return _jinja_env.get_template(f"coach_{version}.md")


def _render_prompt(template, stats: RepStats, exercise: str = "pushup") -> str:
    """Render prompt with statistics (all metrics are computed in Python)."""
    return template.render(
        **stats.model_dump(),
        exercise=exercise,
        exercise_name=_exercise_name(exercise),
    )


def _strict_json_schema(model: type[BaseModel]) -> dict:
    """
    Return a JSON schema OpenAI's `json_schema` (strict) mode accepts.

    OpenAI strict mode requires, on every object node: `additionalProperties:
    false`, and `required` listing *every* property (including optionals with
    defaults). Pydantic's default schema omits both, so we walk the schema and
    fix them.
    """
    schema = model.model_json_schema()

    def _add(sp):
        if isinstance(sp, dict):
            if sp.get("type") == "object":
                sp.setdefault("additionalProperties", False)
                props = sp.get("properties", {})
                # Rebuild required as a superset: existing + all property keys.
                required = list(sp.get("required", []))
                for key in props:
                    if key not in required:
                        required.append(key)
                sp["required"] = required
                for v in props.values():
                    _add(v)
            for ref in sp.get("definitions", {}).values():
                _add(ref)
        elif isinstance(sp, list):
            for item in sp:
                _add(item)

    _add(schema)
    return schema


# =============================================================================
# OpenAI Client with Retry Logic
# =============================================================================

class LLMClient:
    """OpenAI client with retry, timeout, and token logging."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout_s: float = 10.0,
        max_tokens: int = 300,
        temperature: float = 0.3,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            logger.warning("OpenAI API key not set - LLM calls will use fallback")

        self._client = None
        if self.api_key:
            self._client = OpenAI(
                api_key=self.api_key,
                timeout=timeout_s,
                max_retries=0,  # We handle retries ourselves
            )

    def _log_token_usage(self, response, prompt_tokens: int = 0):
        """Log token usage from response."""
        usage = response.usage
        if usage:
            logger.info(
                f"LLM tokens - prompt: {usage.prompt_tokens}, "
                f"completion: {usage.completion_tokens}, "
                f"total: {usage.total_tokens}"
            )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_with_retry(self, messages: list[dict], response_format: dict) -> dict:
        """Call OpenAI with retry logic."""
        if not self._client:
            raise RuntimeError("OpenAI client not initialized (no API key)")

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format=response_format,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response


# =============================================================================
# Main Summarize Function
# =============================================================================

def _summarize_with_client(
    stats: RepStats,
    client: LLMClient,
    prompt_version: str = "v1",
    exercise: str = "pushup",
) -> CoachFeedback:
    """
    Core summarization logic, split out so tests can inject a stub client.

    Returns a validated CoachFeedback. Never raises: on any LLM or validation
    failure it performs exactly one repair retry, then falls back to the
    deterministic template.
    """
    start_time = time.time()

    # Load and render prompt
    try:
        prompt_template = _load_prompt(prompt_version)
        prompt = _render_prompt(prompt_template, stats, exercise=exercise)
    except Exception:
        logger.exception("Failed to load/render prompt")
        return _template_fallback(stats, exercise=exercise)

    messages = [
        {"role": "system", "content": f"You are an expert fitness coach for {_exercise_name(exercise)}. Output ONLY valid JSON matching the schema."},
        {"role": "user", "content": prompt},
    ]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "coach_feedback",
            "schema": _strict_json_schema(CoachFeedback),
            "strict": True,
        },
    }

    content = ""
    # Attempt 1: Normal call
    try:
        response = client._call_with_retry(messages, response_format)
        content = response.choices[0].message.content
        client._log_token_usage(response)
        feedback = CoachFeedback.model_validate_json(content)
        logger.info(f"LLM coaching generated in {time.time() - start_time:.2f}s")
        return feedback

    except ValidationError as e:
        logger.warning(f"LLM output validation failed: {e}. Attempting repair...")
    except (RateLimitError, APITimeoutError, APIConnectionError) as e:
        logger.warning(f"LLM API error: {type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"LLM call failed: {type(e).__name__}: {e}")

    # Attempt 2: one repair retry with explicit instruction
    try:
        repair_messages = [
            *messages,
            {"role": "assistant", "content": content},
            {"role": "user", "content": "The previous response was invalid. Output ONLY valid JSON matching the exact schema. No explanations."},
        ]
        response = client._call_with_retry(repair_messages, response_format)
        content = response.choices[0].message.content
        client._log_token_usage(response)
        feedback = CoachFeedback.model_validate_json(content)
        logger.info(f"LLM coaching repaired successfully in {time.time() - start_time:.2f}s")
        return feedback

    except Exception as e:
        logger.warning(f"LLM repair failed: {type(e).__name__}: {e}")

    # Fallback: Deterministic template. The endpoint must never fail because
    # the model misbehaved.
    logger.info("Using template fallback")
    return _template_fallback(stats, exercise=exercise)


def _default_api_key() -> str | None:
    """Resolve the OpenAI API key from the environment, then app settings."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        from src.serving.settings import (
            settings,  # lazy import, avoids llm->serving coupling at module load
        )
        return settings.openai_api_key or None
    except Exception:
        return None


def summarize(
    stats: RepStats,
    *,
    timeout_s: float = 10.0,
    model: str = "gpt-4o-mini",
    prompt_version: str = "v1",
    api_key: str | None = None,
    exercise: str = "pushup",
) -> CoachFeedback:
    """
    Generate coaching feedback from rep statistics.

    Args:
        stats: Computed RepStats from the pipeline
        timeout_s: Request timeout in seconds
        model: OpenAI model to use
        prompt_version: Prompt template version
        api_key: Optional explicit API key. Defaults to OPENAI_API_KEY env var
                 or the app settings value.
        exercise: Exercise id (e.g. "pushup", "squat") so the prompt and the
                  deterministic fallback use the right movement vocabulary.

    Returns:
        CoachFeedback (validated, never raises)

    The function handles:
    - Missing API key -> template fallback
    - Timeouts -> template fallback
    - Rate limits -> retry with backoff, then fallback
    - Invalid JSON -> one repair retry, then fallback
    - Validation failure -> one repair retry, then fallback
    """
    resolved_key = api_key or _default_api_key()
    if not resolved_key:
        logger.info("No OpenAI API key - using template fallback")
        return _template_fallback(stats, exercise=exercise)

    client = LLMClient(
        api_key=resolved_key,
        model=model,
        timeout_s=timeout_s,
    )
    return _summarize_with_client(stats, client, prompt_version=prompt_version, exercise=exercise)


# =============================================================================
# Convenience Function for Testing
# =============================================================================

def summarize_with_mock(
    stats: RepStats,
    mock_response: dict | None = None,
) -> CoachFeedback:
    """
    Testing version that uses a mock response instead of calling OpenAI.
    """
    if mock_response:
        try:
            return CoachFeedback.model_validate(mock_response)
        except ValidationError:
            return _template_fallback(stats)
    return _template_fallback(stats)
