"""Validated final-output contracts for v1.7 structured responses."""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    chunk_id: int | str
    claim: str = Field(min_length=1)


class PaperAnalysisOutput(BaseModel):
    """Machine-readable paper analysis grounded in retrieved evidence."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    method: str = Field(min_length=1)
    datasets: list[str]
    results: list[str]
    limitations: list[str]
    fyp_relevance: str = Field(min_length=1)
    evidence_references: list[EvidenceReference]


class StructuredOutputError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)


class StructuredOutputResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["not_requested", "valid", "fallback"]
    schema_name: str | None = None
    data: dict[str, Any] | None = None
    error: StructuredOutputError | None = None
    attempt_count: int = Field(default=0, ge=0)


PAPER_ANALYSIS_SKILLS = {
    "paper_analysis_skill",
    "file_analysis_skill",
    "paper_import_skill",
}


def build_output_schema_context(
    output_mode: str,
    skill: dict,
) -> dict | None:
    """Select one explicit output contract; default text stays unchanged."""
    if output_mode == "text":
        return None
    if output_mode != "paper_analysis":
        raise ValueError(f"Unsupported output mode: {output_mode}")
    if skill.get("name") not in PAPER_ANALYSIS_SKILLS:
        raise ValueError(
            "paper_analysis output mode is only available for paper or file "
            "analysis skills in v1.7.1."
        )

    return {
        "name": "PaperAnalysisOutput",
        "json_schema": PaperAnalysisOutput.model_json_schema(),
        "rules": [
            "Return exactly one JSON object with no Markdown code fence.",
            "Write explanatory field values in Chinese.",
            "Use empty lists when datasets, results, or limitations are not supported by evidence.",
            "Every evidence reference must copy a source and chunk_id that appears in Retrieved Evidence.",
            "Never invent a source, chunk_id, result, metric, dataset, or limitation.",
        ],
    }


def _strip_json_fence(raw_content: str) -> str:
    text = raw_content.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1).strip() if fenced else text


def validate_paper_analysis_output(
    raw_content: str,
    evidence: list[dict],
) -> StructuredOutputResult:
    """Parse the model JSON, then verify references against runtime evidence."""
    try:
        payload = json.loads(_strip_json_fence(raw_content))
    except json.JSONDecodeError as error:
        return StructuredOutputResult(
            status="fallback",
            schema_name="PaperAnalysisOutput",
            error=StructuredOutputError(
                code="invalid_json",
                message="The model response was not one valid JSON object.",
                details={
                    "line": error.lineno,
                    "column": error.colno,
                },
            ),
        )

    try:
        output = PaperAnalysisOutput.model_validate(payload)
    except ValidationError as error:
        return StructuredOutputResult(
            status="fallback",
            schema_name="PaperAnalysisOutput",
            error=StructuredOutputError(
                code="schema_validation_failed",
                message="The JSON did not match PaperAnalysisOutput.",
                details={
                    "validation_errors": error.errors(
                        include_url=False,
                        include_input=False,
                    ),
                },
            ),
        )

    allowed_references = {
        (str(item.get("source")), str(item.get("chunk_id")))
        for item in evidence
    }
    references = output.evidence_references
    if evidence and not references:
        return StructuredOutputResult(
            status="fallback",
            schema_name="PaperAnalysisOutput",
            error=StructuredOutputError(
                code="missing_evidence_references",
                message="Retrieved evidence exists, but the output cited none of it.",
            ),
        )

    invalid_references = [
        {
            "source": reference.source,
            "chunk_id": reference.chunk_id,
        }
        for reference in references
        if (reference.source, str(reference.chunk_id)) not in allowed_references
    ]
    if invalid_references:
        return StructuredOutputResult(
            status="fallback",
            schema_name="PaperAnalysisOutput",
            error=StructuredOutputError(
                code="invalid_evidence_reference",
                message="The output cited evidence that was not present in this turn.",
                details={"invalid_references": invalid_references},
            ),
        )

    return StructuredOutputResult(
        status="valid",
        schema_name="PaperAnalysisOutput",
        data=output.model_dump(),
    )


def build_structured_output_repair(
    raw_content: str,
    validation_result: StructuredOutputResult,
) -> dict:
    """Build bounded repair data without changing the original evidence."""
    return {
        "instruction": (
            "Correct the previous candidate into one complete JSON object. "
            "Fix only the reported contract errors, keep claims grounded in "
            "Retrieved Evidence, and do not add commentary or Markdown."
        ),
        "validation_error": (
            validation_result.error.model_dump()
            if validation_result.error
            else None
        ),
        "previous_candidate": raw_content,
    }


def format_paper_analysis(data: dict) -> str:
    """Render validated data for humans without asking the model again."""
    def render_list(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- 证据中未说明"

    references = data.get("evidence_references", [])
    rendered_references = (
        "\n".join(
            f"- {item['source']}，chunk {item['chunk_id']}：{item['claim']}"
            for item in references
        )
        if references
        else "- 本轮没有可引用的检索片段"
    )
    return (
        f"摘要\n{data['summary']}\n\n"
        f"Problem\n{data['problem']}\n\n"
        f"Method\n{data['method']}\n\n"
        f"Datasets\n{render_list(data['datasets'])}\n\n"
        f"Results\n{render_list(data['results'])}\n\n"
        f"Limitations\n{render_list(data['limitations'])}\n\n"
        f"FYP Relevance\n{data['fyp_relevance']}\n\n"
        f"Evidence References\n{rendered_references}"
    )
