"""Typed success and failure contracts for external model calls."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ModelCallResult(BaseModel):
    """One stable return shape, whether the provider succeeds or fails."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "error"]
    content: str
    model: str
    latency_seconds: float = Field(ge=0.0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    error: ModelError | None = None
    timeout_seconds: float = Field(gt=0.0)
    max_retries_configured: int = Field(ge=0)
    max_output_tokens_configured: int = Field(ge=1)

    def to_runtime_dict(self) -> dict:
        result = self.model_dump()
        # Compatibility alias used by the existing Agent usage output.
        result["latency"] = result["latency_seconds"]
        return result


def classify_model_error(error: Exception) -> ModelError:
    """Classify without copying possibly sensitive exception text."""
    error_type = type(error).__name__
    normalized_type = error_type.lower()
    status_code = getattr(error, "status_code", None)
    details = {"exception_type": error_type}
    if isinstance(status_code, int):
        details["status_code"] = status_code

    if isinstance(error, TimeoutError) or "timeout" in normalized_type:
        return ModelError(
            code="model_timeout",
            message="The model request timed out after the configured limit.",
            retryable=True,
            details=details,
        )
    if isinstance(error, ConnectionError) or "connection" in normalized_type:
        return ModelError(
            code="model_connection_error",
            message="The model provider could not be reached.",
            retryable=True,
            details=details,
        )
    if "ratelimit" in normalized_type or status_code == 429:
        return ModelError(
            code="model_rate_limited",
            message="The model provider rate limit was reached.",
            retryable=True,
            details=details,
        )
    if status_code in {401, 403} or any(
        marker in normalized_type
        for marker in ["authentication", "permissiondenied"]
    ):
        return ModelError(
            code="model_authentication_error",
            message="The model provider rejected the configured credentials.",
            retryable=False,
            details=details,
        )
    if isinstance(status_code, int) and status_code >= 500:
        return ModelError(
            code="model_provider_error",
            message="The model provider returned a server error.",
            retryable=True,
            details=details,
        )

    return ModelError(
        code="model_call_error",
        message=f"The model call failed with {error_type}.",
        retryable=False,
        details=details,
    )
