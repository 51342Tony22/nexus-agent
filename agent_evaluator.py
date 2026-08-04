"""Offline-testable metrics for the v1.8 live Agent evaluation harness."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = (
    PROJECT_DIR / "evaluations" / "paper_analysis_cases.json"
)


class PaperAnalysisEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    paper: str = Field(min_length=1, pattern=r"^[^/]+\.pdf$")
    question: str = Field(min_length=1)
    evaluation_goal: str = Field(min_length=1)


def load_evaluation_cases(
    path: str | Path = DEFAULT_CASES_PATH,
) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [
        PaperAnalysisEvaluationCase.model_validate(item).model_dump()
        for item in payload
    ]
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case_id values must be unique.")
    return cases


def _runtime_evidence_references(result: dict) -> set[tuple[str, str]]:
    references = set()
    for observation in result.get("tool_observations", []):
        tool_result = observation.get("tool_result")
        if not isinstance(tool_result, dict):
            continue
        retrieval = tool_result.get("retrieval")
        if not isinstance(retrieval, dict):
            continue
        source = str(tool_result.get("file_path"))
        for chunk in retrieval.get("retrieved_chunks", []):
            references.add((source, str(chunk.get("chunk_id"))))
    return references


def evaluate_agent_result(case: dict, result: dict) -> dict:
    """Measure observable contracts; do not claim semantic correctness."""
    structured = result.get("structured_output", {})
    repair = result.get("structured_output_repair", {})
    status = structured.get("status")
    structured_valid = status == "valid"
    output_data = structured.get("data") or {}
    cited_references = output_data.get("evidence_references", [])
    allowed_references = _runtime_evidence_references(result)
    citation_identity_valid = bool(cited_references) and all(
        (str(reference.get("source")), str(reference.get("chunk_id")))
        in allowed_references
        for reference in cited_references
    )
    context_evaluation = result.get("context_evaluation", {})
    usage = result.get("usage", {})
    model_call = result.get("model_call", {})
    debug = result.get("debug", {})
    retrieval_debug = debug.get("retrievals", [])
    context_stats = debug.get("context_stats", {})
    total_context = context_stats.get("total", {})

    return {
        "case_id": case["case_id"],
        "paper": case["paper"],
        "evaluation_goal": case["evaluation_goal"],
        "completed": True,
        "structured_status": status,
        "structured_valid": structured_valid,
        "first_pass_valid": (
            structured_valid and structured.get("attempt_count") == 1
        ),
        "repair_attempted": repair.get("attempted") is True,
        "repair_success": (
            repair.get("attempted") is True and structured_valid
        ),
        "fallback": status == "fallback",
        "citation_count": len(cited_references),
        "citation_identity_valid": citation_identity_valid,
        "retrieval_sufficient": (
            context_evaluation.get("is_sufficient") is True
        ),
        "retrieval_method": (
            retrieval_debug[-1].get("retrieval_method")
            if retrieval_debug
            else None
        ),
        "model_call_count": model_call.get("call_count", 1),
        "prompt_characters": total_context.get("characters"),
        "prompt_estimated_tokens": total_context.get("estimated_tokens"),
        "latency_seconds": usage.get("latency"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "error": structured.get("error"),
    }


def failed_case_result(case: dict, error: Exception) -> dict:
    """Keep one failed case from aborting the remaining benchmark."""
    return {
        "case_id": case["case_id"],
        "paper": case["paper"],
        "evaluation_goal": case["evaluation_goal"],
        "completed": False,
        "structured_status": "execution_error",
        "structured_valid": False,
        "first_pass_valid": False,
        "repair_attempted": False,
        "repair_success": False,
        "fallback": False,
        "citation_count": 0,
        "citation_identity_valid": False,
        "retrieval_sufficient": False,
        "retrieval_method": None,
        "model_call_count": 0,
        "prompt_characters": None,
        "prompt_estimated_tokens": None,
        "latency_seconds": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "error": {
            "code": "evaluation_case_failed",
            "exception_type": type(error).__name__,
        },
    }


def aggregate_evaluation(case_results: list[dict]) -> dict:
    case_count = len(case_results)

    def rate(predicate: Callable[[dict], bool]) -> float:
        if case_count == 0:
            return 0.0
        return sum(1 for item in case_results if predicate(item)) / case_count

    repair_cases = [
        item for item in case_results if item["repair_attempted"]
    ]
    valid_cases = [
        item for item in case_results if item["structured_valid"]
    ]
    known_usage = [
        item for item in case_results
        if isinstance(item.get("total_tokens"), int)
    ]
    known_latency = [
        item for item in case_results
        if isinstance(item.get("latency_seconds"), (int, float))
    ]

    return {
        "case_count": case_count,
        "completed_count": sum(item["completed"] for item in case_results),
        "structured_valid_rate": rate(lambda item: item["structured_valid"]),
        "first_pass_valid_rate": rate(lambda item: item["first_pass_valid"]),
        "repair_attempt_rate": rate(lambda item: item["repair_attempted"]),
        "repair_success_rate": (
            sum(item["repair_success"] for item in repair_cases)
            / len(repair_cases)
            if repair_cases
            else None
        ),
        "fallback_rate": rate(lambda item: item["fallback"]),
        "citation_identity_valid_rate": (
            sum(item["citation_identity_valid"] for item in valid_cases)
            / len(valid_cases)
            if valid_cases
            else None
        ),
        "retrieval_sufficient_rate": rate(
            lambda item: item["retrieval_sufficient"]
        ),
        "total_model_calls": sum(
            item["model_call_count"] for item in case_results
        ),
        "usage_known_cases": len(known_usage),
        "total_tokens": sum(item["total_tokens"] for item in known_usage),
        "average_tokens": (
            sum(item["total_tokens"] for item in known_usage)
            / len(known_usage)
            if known_usage
            else None
        ),
        "total_latency_seconds": sum(
            item["latency_seconds"] for item in known_latency
        ),
        "average_latency_seconds": (
            sum(item["latency_seconds"] for item in known_latency)
            / len(known_latency)
            if known_latency
            else None
        ),
    }


def run_live_evaluation(
    cases: list[dict],
    agent: Any,
) -> dict:
    case_results = []
    for case in cases:
        try:
            result = agent.run(
                case["question"],
                output_mode="paper_analysis",
            )
            case_results.append(evaluate_agent_result(case, result))
        except Exception as error:
            case_results.append(failed_case_result(case, error))
        finally:
            agent.clear_conversation()

    return {
        "evaluation": "paper_analysis_structured_output_v1",
        "case_results": case_results,
        "aggregate": aggregate_evaluation(case_results),
        "metric_scope": (
            "Schema validity, citation identity, retrieval sufficiency and "
            "runtime cost. Semantic claim correctness is not measured."
        ),
    }
