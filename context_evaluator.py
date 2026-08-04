# context_evaluator.py
# Estimate whether retrieved evidence is sufficient at runtime.

from config import (
    CONTEXT_MIN_EVIDENCE_CHARS,
    CONTEXT_MODERATE_QUERY_COVERAGE,
    CONTEXT_MODERATE_SEMANTIC_SCORE,
    CONTEXT_STRONG_QUERY_COVERAGE,
    CONTEXT_STRONG_SEMANTIC_SCORE,
)
from rag_store import tokenize


def _normalize_token(token: str) -> str:
    token = token.lower()

    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"

    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]

    return token


def _query_coverage(query: str, chunks: list[dict]) -> dict:
    query_tokens = {
        _normalize_token(token)
        for token in tokenize(query)
    }
    evidence_tokens = {
        _normalize_token(token)
        for chunk in chunks
        for token in tokenize(chunk.get("content", ""))
    }
    matched_tokens = query_tokens.intersection(evidence_tokens)
    missing_tokens = query_tokens.difference(evidence_tokens)
    coverage = (
        len(matched_tokens) / len(query_tokens)
        if query_tokens
        else 0.0
    )

    return {
        "query_term_count": len(query_tokens),
        "matched_query_terms": sorted(matched_tokens),
        "missing_query_terms": sorted(missing_tokens),
        "query_term_coverage": coverage,
    }


def _base_result(
    status: str,
    is_sufficient: bool | None,
    reason_code: str,
    reason: str,
    recommended_action: str,
    signals: dict | None = None,
) -> dict:
    return {
        "status": status,
        "is_sufficient": is_sufficient,
        "reason_code": reason_code,
        "reason": reason,
        "recommended_action": recommended_action,
        "signals": signals or {},
    }


def evaluate_retrieval_context(
    retrieval_decision: dict,
    tool_observations: list[dict],
) -> dict:
    if not retrieval_decision.get("needs_retrieval"):
        return _base_result(
            status="not_required",
            is_sufficient=None,
            reason_code="retrieval_not_required",
            reason="The retrieval decision did not require document evidence.",
            recommended_action="answer_without_retrieval",
        )

    retrieval_tool_result = None
    for observation in tool_observations:
        if observation.get("tool_used") != "retrieve_from_file":
            continue
        retrieval_tool_result = observation.get("tool_result")
        break

    if not isinstance(retrieval_tool_result, dict):
        return _base_result(
            status="insufficient",
            is_sufficient=False,
            reason_code="missing_retrieval_observation",
            reason="Retrieval was required, but no retrieval tool result was produced.",
            recommended_action="retry_retrieval",
        )

    if retrieval_tool_result.get("error"):
        return _base_result(
            status="insufficient",
            is_sufficient=False,
            reason_code="retrieval_tool_error",
            reason=retrieval_tool_result["error"],
            recommended_action="fix_source",
            signals={
                "source": retrieval_decision.get("source"),
            },
        )

    retrieval = retrieval_tool_result.get("retrieval")
    if not isinstance(retrieval, dict):
        return _base_result(
            status="insufficient",
            is_sufficient=False,
            reason_code="missing_retrieval_payload",
            reason="The tool result did not contain a retrieval payload.",
            recommended_action="retry_retrieval",
        )

    chunks = retrieval.get("retrieved_chunks", [])
    if not chunks:
        return _base_result(
            status="insufficient",
            is_sufficient=False,
            reason_code="no_retrieved_chunks",
            reason="The retriever returned no evidence chunks.",
            recommended_action="rewrite_query",
            signals={
                "retrieval_method": retrieval.get("retrieval_method"),
                "chunks_returned": 0,
            },
        )

    # Query expansion is a recall technique, not proof that the evidence
    # answers the user's information need. Generic expansion terms such as
    # "method" or "framework" appear in almost every paper and would inflate
    # lexical coverage. Evaluate against the original retrieval query first.
    evaluation_query = (
        retrieval_decision.get("query")
        or retrieval.get("query")
        or retrieval.get("expanded_query")
        or ""
    )
    coverage_signals = _query_coverage(evaluation_query, chunks)
    evidence_characters = sum(
        len(chunk.get("content", ""))
        for chunk in chunks
    )
    dual_source_chunks = sum(
        1
        for chunk in chunks
        if {"bm25", "embedding"}.issubset(
            set(chunk.get("retrieval_sources", []))
        )
    )
    embedding_scores = [
        chunk.get("source_scores", {}).get("embedding")
        for chunk in chunks
        if chunk.get("source_scores", {}).get("embedding") is not None
    ]
    maximum_embedding_score = (
        max(embedding_scores)
        if embedding_scores
        else None
    )
    semantic_score_query_aligned = retrieval.get(
        "semantic_score_query_aligned",
        retrieval.get("query") == evaluation_query,
    )
    query_coverage = coverage_signals["query_term_coverage"]
    enough_text = evidence_characters >= CONTEXT_MIN_EVIDENCE_CHARS
    has_dual_support = dual_source_chunks > 0

    signals = {
        "evaluation_query": evaluation_query,
        "attempt_count": retrieval.get("attempt_count", 1),
        "retrieval_method": retrieval.get("retrieval_method"),
        "chunks_returned": len(chunks),
        "evidence_characters": evidence_characters,
        "dual_source_chunks": dual_source_chunks,
        "maximum_embedding_score": maximum_embedding_score,
        "semantic_score_query_aligned": semantic_score_query_aligned,
        **coverage_signals,
        "thresholds": {
            "minimum_evidence_characters": CONTEXT_MIN_EVIDENCE_CHARS,
            "strong_query_coverage": CONTEXT_STRONG_QUERY_COVERAGE,
            "moderate_query_coverage": CONTEXT_MODERATE_QUERY_COVERAGE,
            "strong_semantic_score": CONTEXT_STRONG_SEMANTIC_SCORE,
            "moderate_semantic_score": CONTEXT_MODERATE_SEMANTIC_SCORE,
        },
    }

    if enough_text and query_coverage >= CONTEXT_STRONG_QUERY_COVERAGE:
        return _base_result(
            status="sufficient",
            is_sufficient=True,
            reason_code="strong_query_coverage",
            reason="The retrieved evidence covers most of the normalized query terms.",
            recommended_action="answer",
            signals=signals,
        )

    if (
        enough_text
        and has_dual_support
        and semantic_score_query_aligned
        and maximum_embedding_score is not None
        and maximum_embedding_score >= CONTEXT_STRONG_SEMANTIC_SCORE
    ):
        return _base_result(
            status="sufficient",
            is_sufficient=True,
            reason_code="strong_semantic_support",
            reason="Hybrid retrieval provides strong semantic support despite limited lexical overlap.",
            recommended_action="answer",
            signals=signals,
        )

    if (
        enough_text
        and has_dual_support
        and query_coverage >= CONTEXT_MODERATE_QUERY_COVERAGE
        and maximum_embedding_score is not None
        and maximum_embedding_score >= CONTEXT_MODERATE_SEMANTIC_SCORE
    ):
        return _base_result(
            status="sufficient",
            is_sufficient=True,
            reason_code="combined_lexical_semantic_support",
            reason="Moderate lexical coverage and semantic support agree on the evidence.",
            recommended_action="answer",
            signals=signals,
        )

    return _base_result(
        status="insufficient",
        is_sufficient=False,
        reason_code="weak_evidence_support",
        reason="The retrieved chunks do not provide enough lexical or semantic support.",
        recommended_action="rewrite_query",
        signals=signals,
    )
