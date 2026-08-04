"""Dependency-free persistent-memory retrieval for v1.4.2."""

import json
import math
import re
from datetime import datetime, timezone


GLOBAL_MEMORY_TAGS = {"always", "global", "preference", "profile"}


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _lexical_units(value: str) -> set[str]:
    """Return English terms and CJK unigrams/bigrams for mixed-language text."""
    normalized = _normalize_text(value)
    units = set(re.findall(r"[a-z0-9][a-z0-9_+.-]*", normalized))

    for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
        units.update(sequence)
        units.update(
            sequence[index:index + 2]
            for index in range(len(sequence) - 1)
        )

    return units


def _record_search_text(record: dict) -> str:
    metadata = json.dumps(
        record.get("metadata", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    return " ".join([
        str(record.get("content", "")),
        " ".join(record.get("tags", [])),
        metadata,
    ])


def _relevance_score(
    query_units: set[str],
    record_units: set[str],
) -> tuple[float, list[str]]:
    if not query_units or not record_units:
        return 0.0, []

    matched_units = query_units & record_units
    score = len(matched_units) / math.sqrt(
        len(query_units) * len(record_units)
    )
    return min(score, 1.0), sorted(matched_units)


def _recency_score(record: dict, now: datetime) -> float:
    timestamp = record.get("last_used_at") or record.get("updated_at")
    if not timestamp:
        return 0.0

    try:
        parsed = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return 0.0

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - parsed).total_seconds() / 86400)
    return math.exp(-age_days / 90.0)


def retrieve_memories(
    query: str,
    records: list[dict],
    *,
    top_k: int = 3,
    min_relevance: float = 0.08,
    now: datetime | None = None,
) -> dict:
    """Select memories using relevance, importance, confidence and recency.

    Relevance is the admission gate. The other signals only rank admitted
    memories and cannot make an unrelated memory relevant. A record tagged as
    a global preference/profile is explicitly exempt from the relevance gate.
    """
    if top_k < 0:
        raise ValueError("top_k cannot be negative.")
    if not 0.0 <= min_relevance <= 1.0:
        raise ValueError("min_relevance must be between 0.0 and 1.0.")

    current_time = now or datetime.now(timezone.utc)
    query_units = _lexical_units(query)
    ranked = []

    for record in records:
        relevance, matched_units = _relevance_score(
            query_units,
            _lexical_units(_record_search_text(record)),
        )
        tags = set(record.get("tags", []))
        is_global = bool(tags & GLOBAL_MEMORY_TAGS)

        if relevance < min_relevance and not is_global:
            continue

        importance = float(record.get("importance", 0.5))
        confidence = float(record.get("confidence", 1.0))
        recency = _recency_score(record, current_time)
        global_bonus = 0.08 if is_global else 0.0
        retrieval_score = min(
            1.0,
            0.62 * relevance
            + 0.18 * importance
            + 0.10 * confidence
            + 0.10 * recency
            + global_bonus,
        )

        ranked.append({
            "memory_id": record["memory_id"],
            "memory_type": record["memory_type"],
            "content": record["content"],
            "source": record.get("source"),
            "tags": list(record.get("tags", [])),
            "retrieval_score": round(retrieval_score, 4),
            "signals": {
                "relevance": round(relevance, 4),
                "importance": round(importance, 4),
                "confidence": round(confidence, 4),
                "recency": round(recency, 4),
                "is_global": is_global,
                "matched_units": matched_units[:12],
            },
        })

    ranked.sort(
        key=lambda item: (
            item["retrieval_score"],
            item["signals"]["relevance"],
            item["memory_id"],
        ),
        reverse=True,
    )
    selected = ranked[:top_k]

    return {
        "query": query,
        "retrieval_method": "Lexical+Importance+Recency",
        "candidate_count": len(records),
        "eligible_count": len(ranked),
        "selected_count": len(selected),
        "top_k": top_k,
        "min_relevance": min_relevance,
        "memories": selected,
    }
