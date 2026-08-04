# context_builder.py
# Select and normalize runtime information before it is rendered for the LLM.

from config import (
    CONTEXT_AGENT_LOOP_MAX_CHARS,
    CONTEXT_DECISION_MAX_CHARS,
    CONTEXT_EVALUATION_MAX_CHARS,
    CONTEXT_EVIDENCE_MAX_CHARS,
    CONTEXT_EXTERNAL_DATA_MAX_CHARS,
    CONTEXT_HISTORY_MAX_CHARS,
    CONTEXT_MEMORY_MAX_CHARS,
    CONTEXT_OUTPUT_REPAIR_MAX_CHARS,
    CONTEXT_OUTPUT_SCHEMA_MAX_CHARS,
    CONTEXT_PERSISTENT_MEMORY_MAX_CHARS,
    CONTEXT_PLAN_MAX_CHARS,
    CONTEXT_SUMMARY_MAX_CHARS,
    CONTEXT_TOOL_MAX_CHARS,
)
from context_cleaner import clean_memory_for_skill


def _build_plan_context(execution_plan: list[dict]) -> list[dict]:
    """Keep actions, but remove repeated tool arguments such as the query."""
    return [
        {
            "step_id": step.get("step_id"),
            "description": step.get("description"),
            "tool_name": step.get("tool_name"),
        }
        for step in execution_plan
    ]


def _compact_permission(permission: dict | None) -> dict | None:
    """Keep approval control-plane identifiers and arguments out of prompts."""
    if not isinstance(permission, dict):
        return None
    return {
        "allowed": permission.get("allowed"),
        "reason_code": permission.get("reason_code"),
        "risk_level": permission.get("risk_level"),
        "requires_confirmation": permission.get("requires_confirmation"),
        "approval_required": bool(permission.get("approval_request")),
    }


def _build_tool_context(tool_observations: list[dict]) -> list[dict]:
    """Build a compact tool status view without duplicating retrieved text."""
    compact_observations = []

    for observation in tool_observations:
        tool_result = observation.get("tool_result")
        compact_result = tool_result

        if isinstance(tool_result, dict) and isinstance(
            tool_result.get("retrieval"),
            dict,
        ):
            retrieval = tool_result["retrieval"]
            compact_result = {
                "file_path": tool_result.get("file_path"),
                "file_type": tool_result.get("file_type"),
                "retrieval_method": retrieval.get("retrieval_method"),
                "fusion_method": retrieval.get("fusion_method"),
                "chunks_returned": len(retrieval.get("retrieved_chunks", [])),
                "fallback_reason": retrieval.get("fallback_reason"),
            }
        elif (
            observation.get("tool_used") in {
                "search_papers",
                "get_paper_details",
                "import_paper_pdf",
                "mcp_search_fyp_library",
            }
            and isinstance(tool_result, dict)
        ):
            compact_result = {
                "provider": tool_result.get("provider"),
                "query": tool_result.get("query"),
                "paper_id": tool_result.get("paper_id"),
                "year_filter": tool_result.get("year_filter"),
                "result_count": tool_result.get("result_count", 0),
                "imported": tool_result.get("imported"),
                "file_path": tool_result.get("file_path"),
                "error": tool_result.get("error"),
            }
            if observation.get("tool_used") == "mcp_search_fyp_library":
                data = tool_result.get("data") or {}
                compact_result = {
                    "provider": "MCP",
                    "server": (tool_result.get("mcp") or {}).get("server"),
                    "protocol_version": (
                        tool_result.get("mcp") or {}
                    ).get("protocol_version"),
                    "transport": (
                        tool_result.get("mcp") or {}
                    ).get("transport"),
                    "query": data.get("query"),
                    "result_count": data.get("result_count", 0),
                    "error": tool_result.get("error"),
                }

        compact_observations.append({
            "step_id": observation.get("step_id"),
            "tool_used": observation.get("tool_used"),
            "status": observation.get("status"),
            "error": observation.get("error"),
            "permission": _compact_permission(observation.get("permission")),
            "latency_seconds": observation.get("latency_seconds"),
            "result": compact_result,
        })

    return compact_observations


def _extract_evidence(tool_observations: list[dict]) -> list[dict]:
    evidence = []

    for observation in tool_observations:
        tool_result = observation.get("tool_result")
        if not isinstance(tool_result, dict):
            continue

        retrieval = tool_result.get("retrieval")
        if not isinstance(retrieval, dict):
            continue

        for chunk in retrieval.get("retrieved_chunks", []):
            evidence.append({
                "source": tool_result.get("file_path", "unknown"),
                "file_type": tool_result.get("file_type"),
                "chunk_id": chunk.get("chunk_id"),
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "retrieval_sources": chunk.get("retrieval_sources", []),
                "source_ranks": chunk.get("source_ranks", {}),
                "trust_level": "untrusted_document_content",
                "content": chunk.get("content", ""),
            })

    return evidence


def _extract_external_tool_data(tool_observations: list[dict]) -> list[dict]:
    """Keep external provider data separate from local-document evidence."""
    external_data = []
    for observation in tool_observations:
        tool_used = observation.get("tool_used")
        if tool_used not in {
            "search_papers",
            "get_paper_details",
            "import_paper_pdf",
            "mcp_search_fyp_library",
        }:
            continue
        tool_result = observation.get("tool_result")
        if not isinstance(tool_result, dict):
            continue
        if tool_used == "mcp_search_fyp_library":
            external_data.append({
                "tool": tool_used,
                "provider": "MCP",
                "mcp": tool_result.get("mcp"),
                "data": tool_result.get("data"),
                "error": tool_result.get("error"),
                "trust_level": "untrusted_mcp_server_data",
            })
        elif tool_used == "search_papers":
            external_data.append({
                "tool": tool_used,
                "provider": tool_result.get("provider"),
                "query": tool_result.get("query"),
                "year_filter": tool_result.get("year_filter"),
                "result_count": tool_result.get("result_count", 0),
                "error": tool_result.get("error"),
                "attribution": tool_result.get("attribution"),
                "papers": tool_result.get("papers", []),
            })
        else:
            external_data.append({
                "tool": tool_used,
                "provider": tool_result.get("provider"),
                "paper_id": tool_result.get("paper_id"),
                "paper": tool_result.get("paper"),
                "imported": tool_result.get("imported"),
                "already_existed": tool_result.get("already_existed"),
                "file_path": tool_result.get("file_path"),
                "bytes_written": tool_result.get("bytes_written"),
                "error": tool_result.get("error"),
                "attribution": tool_result.get("attribution"),
            })
    return external_data


def build_context(
    user_input: str,
    memory: dict,
    skill: dict,
    retrieval_decision: dict | None = None,
    context_evaluation: dict | None = None,
    conversation_summary: str = "",
    conversation_history: str = "",
    execution_plan: list[dict] | None = None,
    tool_observations: list[dict] | None = None,
    persistent_memories: list[dict] | None = None,
    agent_tool_loop: dict | None = None,
    structured_output_schema: dict | None = None,
) -> dict:
    execution_plan = execution_plan or []
    tool_observations = tool_observations or []
    persistent_memories = persistent_memories or []
    agent_tool_loop = agent_tool_loop or {}
    structured_output_schema = structured_output_schema or {}

    relevant_memory = clean_memory_for_skill(
        memory=memory,
        skill=skill,
    )
    external_tool_data = _extract_external_tool_data(tool_observations)

    retrieval_decision = retrieval_decision or {}
    context_evaluation = context_evaluation or {}
    decision_context = {
        key: retrieval_decision.get(key)
        for key in [
            "needs_retrieval",
            "reason_code",
            "reason",
            "strategy",
            "source_type",
            "source",
        ]
        if key in retrieval_decision
    }
    evaluation_context = {
        key: context_evaluation.get(key)
        for key in [
            "status",
            "is_sufficient",
            "reason_code",
            "reason",
            "recommended_action",
        ]
        if key in context_evaluation
    }
    evaluation_signals = context_evaluation.get("signals", {})
    if evaluation_signals:
        evaluation_context["signals"] = {
            key: evaluation_signals.get(key)
            for key in [
                "chunks_returned",
                "attempt_count",
                "evidence_characters",
                "dual_source_chunks",
                "maximum_embedding_score",
                "semantic_score_query_aligned",
                "query_term_coverage",
                "missing_query_terms",
            ]
            if key in evaluation_signals
        }

    return {
        "role": "You are an FYP research assistant for a Computer Science and AI student.",
        "user_input": user_input,
        "selected_skill": skill,
        "relevant_memory": relevant_memory,
        "persistent_memories": persistent_memories,
        "conversation_summary": conversation_summary,
        "conversation_history": conversation_history,
        "retrieval_decision_context": decision_context,
        "context_evaluation": evaluation_context,
        "plan_context": _build_plan_context(execution_plan),
        "tool_context": _build_tool_context(tool_observations),
        "evidence": _extract_evidence(tool_observations),
        "external_tool_data": external_tool_data,
        "agent_tool_loop": agent_tool_loop,
        "structured_output_schema": structured_output_schema,
        "structured_output_repair": {},
        "context_budget": {
            "agent_tool_loop": CONTEXT_AGENT_LOOP_MAX_CHARS,
            "retrieval_decision": CONTEXT_DECISION_MAX_CHARS,
            "context_evaluation": CONTEXT_EVALUATION_MAX_CHARS,
            "conversation_summary": CONTEXT_SUMMARY_MAX_CHARS,
            "conversation_history": CONTEXT_HISTORY_MAX_CHARS,
            "memory": CONTEXT_MEMORY_MAX_CHARS,
            "persistent_memory": CONTEXT_PERSISTENT_MEMORY_MAX_CHARS,
            "plan": CONTEXT_PLAN_MAX_CHARS,
            "tools": CONTEXT_TOOL_MAX_CHARS,
            "evidence": CONTEXT_EVIDENCE_MAX_CHARS,
            "external_data": CONTEXT_EXTERNAL_DATA_MAX_CHARS,
            "output_schema": CONTEXT_OUTPUT_SCHEMA_MAX_CHARS,
            "output_repair": CONTEXT_OUTPUT_REPAIR_MAX_CHARS,
        },
        "output_requirements": [
            "Answer in Chinese.",
            "Relate the answer to the user's FYP when relevant.",
            "Give concrete next steps.",
            "Avoid vague suggestions.",
            "Use simple explanations suitable for a learner in Agent development.",
            *(
                [
                    "For external paper results, include the paper title and URL, and do not claim to have read the full paper from metadata alone."
                ]
                if external_tool_data
                else []
            ),
        ],
    }
