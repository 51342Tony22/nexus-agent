# retrieval_evaluator.py

from collections.abc import Callable


def calculate_retrieval_metrics(
    retrieved_chunk_ids: list[int],
    relevant_chunk_ids: list[int],
) -> dict:
    """Calculate Precision, Recall, and Hit for one retrieval query."""
    if not relevant_chunk_ids:
        raise ValueError("Each evaluation case needs at least one relevant chunk.")

    # A retriever should not return the same chunk twice. Deduplicating here
    # also prevents a duplicate from artificially improving the metrics.
    retrieved_ids = list(dict.fromkeys(retrieved_chunk_ids))
    relevant_ids = set(relevant_chunk_ids)
    relevant_retrieved = relevant_ids.intersection(retrieved_ids)

    precision = (
        len(relevant_retrieved) / len(retrieved_ids)
        if retrieved_ids
        else 0.0
    )
    recall = len(relevant_retrieved) / len(relevant_ids)
    hit = 1.0 if relevant_retrieved else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "hit": hit,
    }


def evaluate_retriever(
    retriever_name: str,
    cases: list[dict],
    retrieve: Callable[[str, int], list[dict]],
    top_k: int,
) -> dict:
    """Evaluate one retriever against human-labelled query/chunk pairs."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    case_results = []

    for case in cases:
        retrieved_chunks = retrieve(case["query"], top_k)
        retrieved_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        metrics = calculate_retrieval_metrics(
            retrieved_chunk_ids=retrieved_ids,
            relevant_chunk_ids=case["relevant_chunk_ids"],
        )

        case_results.append({
            "case_name": case["name"],
            "query": case["query"],
            "relevant_chunk_ids": case["relevant_chunk_ids"],
            "retrieved_chunk_ids": retrieved_ids,
            **metrics,
        })

    case_count = len(case_results)
    aggregate = {
        "precision": sum(item["precision"] for item in case_results) / case_count,
        "recall": sum(item["recall"] for item in case_results) / case_count,
        "hit_rate": sum(item["hit"] for item in case_results) / case_count,
    } if case_count else {
        "precision": 0.0,
        "recall": 0.0,
        "hit_rate": 0.0,
    }

    return {
        "retriever": retriever_name,
        "top_k": top_k,
        "case_results": case_results,
        "aggregate": aggregate,
    }


def print_evaluation_report(reports: list[dict]):
    print("\nRetrieval Evaluation")
    print("=" * 72)

    for report in reports:
        aggregate = report["aggregate"]
        print(
            f"{report['retriever']:<28} "
            f"P@{report['top_k']}: {aggregate['precision']:.3f}  "
            f"R@{report['top_k']}: {aggregate['recall']:.3f}  "
            f"Hit@{report['top_k']}: {aggregate['hit_rate']:.3f}"
        )

        for case in report["case_results"]:
            print(
                f"  - {case['case_name']:<10} "
                f"expected={case['relevant_chunk_ids']} "
                f"retrieved={case['retrieved_chunk_ids']}"
            )

        print()
