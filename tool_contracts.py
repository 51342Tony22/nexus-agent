"""Typed contracts at the Planner -> Tool -> Observation boundary.

Runtime dictionaries remain available through ``to_runtime_dict`` while the
rest of the learning project gradually adopts canonical field names.
"""

from collections.abc import Callable
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)


class ToolInputValidationError(ValueError):
    """Raised before execution when tool arguments violate their schema."""

    def __init__(self, tool_name: str, validation_error: ValidationError):
        self.tool_name = tool_name
        self.validation_errors = validation_error.errors(
            include_url=False,
            include_input=False,
        )
        super().__init__(
            f"Invalid arguments for tool '{tool_name}': "
            f"{validation_error.error_count()} validation error(s)."
        )


class ToolError(BaseModel):
    """A stable error shape that the Agent loop can reason about."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    """Canonical result of one planned tool step."""

    model_config = ConfigDict(extra="forbid")

    step_id: int | str
    description: str
    tool_name: str | None
    status: Literal["success", "error", "skipped"]
    arguments: dict[str, Any] | None = None
    result: Any = None
    error: ToolError | None = None
    permission: dict[str, Any] | None = None
    latency_seconds: float = Field(default=0.0, ge=0.0)

    def to_runtime_dict(self) -> dict:
        """Expose canonical fields plus old aliases during the migration."""
        canonical = self.model_dump()
        return {
            **canonical,
            # Compatibility aliases consumed by the current Context Builder,
            # retrieval evaluator, Debug output, and existing tests.
            "tool_used": canonical["tool_name"],
            "tool_args": canonical["arguments"],
            "tool_result": canonical["result"],
        }


class RetrieveFromFileInput(BaseModel):
    """Validated arguments for the first tool migrated in v1.6.1."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    file_path: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class NoArgumentsInput(BaseModel):
    """Reject accidental parameters for tools that take no arguments."""

    model_config = ConfigDict(extra="forbid")


class CompareDatasetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_names: list[str] = Field(min_length=1, max_length=3)

    @field_validator("dataset_names", mode="before")
    @classmethod
    def normalize_dataset_names(cls, value):
        if not isinstance(value, list):
            return value

        canonical = {
            "gsm8k": "GSM8K",
            "svamp": "SVAMP",
            "strategyqa": "StrategyQA",
            "strategy qa": "StrategyQA",
        }
        normalized = []
        for item in value:
            key = str(item).strip().lower()
            if key not in canonical:
                raise ValueError(f"Unsupported dataset: {item}")
            name = canonical[key]
            if name not in normalized:
                normalized.append(name)
        return normalized


class StudentModelInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    task_type: Literal["generation", "classification"] = "generation"


class ReadTextFileInput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    file_path: str = Field(min_length=1)


class ReadPdfFileInput(ReadTextFileInput):
    max_pages: int | None = Field(default=5, ge=1, le=100)
    max_characters: int | None = Field(
        default=8000,
        ge=100,
        le=120000,
    )


class PaperSearchInput(BaseModel):
    """Arguments accepted by the Semantic Scholar search adapter."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=2, max_length=300)
    limit: int = Field(default=5, ge=1, le=10)
    year: str | None = Field(
        default=None,
        pattern=r"^\d{4}(?:-\d{4}|-)?$",
    )


class MCPLibrarySearchInput(BaseModel):
    """Arguments for the read-only local FYP Library MCP tool."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=6, ge=1, le=6)


class SemanticScholarPaperInput(BaseModel):
    """One validated Semantic Scholar paper identifier."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    paper_id: str = Field(pattern=r"^[0-9a-fA-F]{40}$")

    @field_validator("paper_id")
    @classmethod
    def normalize_paper_id(cls, value: str) -> str:
        return value.lower()


class ToolSpec(BaseModel):
    """Description, callable, input contract, and permission metadata."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
    )

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    function: Callable[..., Any]
    input_schema: type[BaseModel] | None = None
    risk_level: Literal["read_only", "local_write", "external_write"]
    requires_confirmation: bool = False

    def validate_arguments(self, arguments: dict | None) -> dict:
        raw_arguments = arguments or {}
        if self.input_schema is None:
            return dict(raw_arguments)

        try:
            validated = self.input_schema.model_validate(raw_arguments)
        except ValidationError as error:
            raise ToolInputValidationError(self.name, error) from error

        return validated.model_dump()


def error_from_tool_result(result: Any) -> ToolError | None:
    """Normalize a tool's own error payload without changing its result."""
    if not isinstance(result, dict) or not result.get("error"):
        return None

    raw_error = result["error"]
    if isinstance(raw_error, dict):
        code = str(raw_error.get("code") or "tool_reported_error")
        message = str(raw_error.get("message") or raw_error)
        details = {
            key: value
            for key, value in raw_error.items()
            if key not in {"code", "message", "retryable"}
        }
        retryable = bool(raw_error.get("retryable", False))
    else:
        code = "tool_reported_error"
        message = str(raw_error)
        details = {}
        retryable = False

    return ToolError(
        code=code,
        message=message,
        retryable=retryable,
        details=details,
    )
