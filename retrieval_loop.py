# Pure helpers for the bounded Agentic RAG retrieval loop.

from copy import deepcopy

from rag_store import tokenize


INTENT_HINTS = {
    "mechanism": {
        "markers": ["how", "why", "如何", "怎么", "为什么", "机制"],
        "terms": ["mechanism", "method", "process"],
    },
    "results": {
        "markers": ["result", "performance", "accuracy", "结果", "性能", "效果"],
        "terms": ["results", "evaluation", "findings"],
    },
    "datasets": {
        "markers": ["dataset", "benchmark", "数据集", "基准"],
        "terms": ["dataset", "benchmark", "experiment"],
    },
    "comparison": {
        "markers": ["compare", "difference", "versus", "比较", "区别", "差异"],
        "terms": ["comparison", "difference", "analysis"],
    },
}


def _deduplicate_terms(terms: list[str]) -> list[str]:
    unique_terms = []
    seen = set()

    for term in terms:
        normalized = term.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_terms.append(term.strip())

    return unique_terms


def _intent_hints(query: str) -> tuple[str, list[str]]:
    lower_query = query.lower()

    for intent_name, intent in INTENT_HINTS.items():
        if any(marker in lower_query for marker in intent["markers"]):
            return intent_name, intent["terms"]

    return "explanation", ["definition", "explanation", "evidence"]


def build_refined_query(original_query: str, evaluation: dict) -> dict:
    """Focus a retry on concepts missing from the first evidence set."""
    signals = evaluation.get("signals", {})
    missing_terms = list(signals.get("missing_query_terms", []))
    matched_terms = list(signals.get("matched_query_terms", []))
    intent, hints = _intent_hints(original_query)

    # Missing concepts receive most of the retry query's weight. A small
    # number of matched terms remain as anchors to reduce query drift.
    focus_terms = missing_terms[:6]
    if focus_terms:
        focus_terms.extend(matched_terms[:2])
    else:
        focus_terms.extend(tokenize(original_query)[:8])

    refined_terms = _deduplicate_terms(focus_terms + hints)
    refined_query = " ".join(refined_terms).strip()

    if not refined_query:
        refined_query = original_query.strip()

    return {
        "strategy": "missing_terms_with_intent_hints",
        "original_query": original_query,
        "refined_query": refined_query,
        "missing_terms": missing_terms,
        "anchor_terms": matched_terms[:2],
        "intent": intent,
        "added_hints": hints,
    }


def should_retry(evaluation: dict, retries_used: int, max_retries: int) -> bool:
    return (
        retries_used < max_retries
        and evaluation.get("status") == "insufficient"
        and evaluation.get("recommended_action") == "rewrite_query"
    )


def build_retry_step(
    source: str,
    refined_query: str,
    step_id: int,
) -> dict:
    return {
        "step_id": step_id,
        "description": (
            "Retry document retrieval with a query focused on concepts "
            "missing from the previous evidence."
        ),
        "tool_name": "retrieve_from_file",
        "tool_args": {
            "file_path": source,
            "query": refined_query,
            "top_k": 3,
        },
    }


def _merge_chunk(existing: dict, incoming: dict, attempt_number: int):
    existing["retrieval_attempts"] = _deduplicate_terms(
        [
            *[str(item) for item in existing.get("retrieval_attempts", [])],
            str(attempt_number),
        ]
    )
    existing["retrieval_attempts"] = [
        int(item)
        for item in existing["retrieval_attempts"]
    ]
    existing["retrieval_sources"] = _deduplicate_terms(
        existing.get("retrieval_sources", [])
        + incoming.get("retrieval_sources", [])
    )

    existing_ranks = existing.setdefault("source_ranks", {})
    for source, rank in incoming.get("source_ranks", {}).items():
        if source not in existing_ranks:
            existing_ranks[source] = rank
        else:
            existing_ranks[source] = min(existing_ranks[source], rank)

    existing_scores = existing.setdefault("source_scores", {})
    for source, score in incoming.get("source_scores", {}).items():
        if source not in existing_scores:
            existing_scores[source] = score
        else:
            existing_scores[source] = max(existing_scores[source], score)


def merge_retrieval_observations(
    observations: list[dict],
    original_query: str,
    max_chunks: int,
) -> dict | None:
    """Deduplicate evidence from all successful retrieval attempts."""
    successful = []

    for observation in observations:
        tool_result = observation.get("tool_result")
        if not isinstance(tool_result, dict) or tool_result.get("error"):
            continue
        retrieval = tool_result.get("retrieval")
        if not isinstance(retrieval, dict):
            continue
        successful.append((observation, tool_result, retrieval))

    if not successful:
        return None

    chunks_by_key = {}
    ordered_keys = []
    attempt_queries = []
    index_cache_attempts = []
    fallback_reasons = []

    for attempt_number, (_, tool_result, retrieval) in enumerate(
        successful,
        start=1,
    ):
        attempt_queries.append(retrieval.get("query"))
        if retrieval.get("index_cache"):
            index_cache_attempts.append(retrieval["index_cache"])
        if retrieval.get("fallback_reason"):
            fallback_reasons.append(retrieval["fallback_reason"])

        for chunk in retrieval.get("retrieved_chunks", []):
            key = (
                tool_result.get("file_path"),
                chunk.get("chunk_id"),
            )
            if key not in chunks_by_key:
                merged_chunk = deepcopy(chunk)
                merged_chunk["retrieval_attempts"] = [attempt_number]
                chunks_by_key[key] = merged_chunk
                ordered_keys.append(key)
                continue

            _merge_chunk(
                chunks_by_key[key],
                chunk,
                attempt_number,
            )

    merged_chunks = [
        chunks_by_key[key]
        for key in ordered_keys[:max_chunks]
    ]
    first_observation, first_tool_result, first_retrieval = successful[0]

    return {
        "step_id": "retrieval_merge",
        "description": "Merge and deduplicate evidence from retrieval attempts.",
        "tool_used": "retrieve_from_file",
        "tool_args": {
            "file_path": first_tool_result.get("file_path"),
            "query": original_query,
            "attempt_queries": attempt_queries,
        },
        "tool_result": {
            "file_path": first_tool_result.get("file_path"),
            "file_type": first_tool_result.get("file_type"),
            "character_count": first_tool_result.get("character_count"),
            "retrieval": {
                "query": original_query,
                "expanded_query": first_retrieval.get("expanded_query"),
                "retrieval_method": (
                    f"Agentic Hybrid ({len(successful)} attempts)"
                ),
                "fusion_method": "RRF per attempt + evidence union",
                "embedding_model": first_retrieval.get("embedding_model"),
                "fallback_reason": (
                    " | ".join(fallback_reasons)
                    if fallback_reasons
                    else None
                ),
                "attempt_count": len(successful),
                "attempt_queries": attempt_queries,
                "index_cache": {
                    "attempts": index_cache_attempts,
                    "hit_sequence": [
                        item.get("hit")
                        for item in index_cache_attempts
                    ],
                },
                # Source embedding scores came from different rewritten
                # queries. They are useful supporting signals, but no longer
                # a direct similarity measurement for the original query.
                "semantic_score_query_aligned": False,
                "candidate_k": sum(
                    retrieval.get("candidate_k", 0) or 0
                    for _, _, retrieval in successful
                ),
                "bm25_candidates": sum(
                    retrieval.get("bm25_candidates", 0) or 0
                    for _, _, retrieval in successful
                ),
                "embedding_candidates": sum(
                    retrieval.get("embedding_candidates", 0) or 0
                    for _, _, retrieval in successful
                ),
                "rank_constant": first_retrieval.get("rank_constant"),
                "retrieved_chunks": merged_chunks,
            },
        },
    }
