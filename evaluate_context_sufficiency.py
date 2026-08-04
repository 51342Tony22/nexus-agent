# Reproducible runtime context-sufficiency experiment using the six FYP papers.

from context_evaluator import evaluate_retrieval_context
from embedding_store import EmbeddingStore
from query_expander import expand_query
from rag_store import bm25_retrieve, chunk_text, reciprocal_rank_fusion
from tools import read_pdf_file


POSITIVE_CASES = [
    (
        "1_CotBasic_FromLLM->LM_DIstill.pdf",
        "teaching small language models reasoning chain of thought distillation",
    ),
    (
        "2_SocraticCot.pdf",
        "Socratic method rationale generation distilling reasoning capabilities into smaller language models",
    ),
    (
        "3_Disitill_Stepbystep_multitask->Siglerationale.pdf",
        "distilling step by step rationales multitask training smaller models less data",
    ),
    (
        "4_SymblicCoT_Multi_rationales.pdf",
        "symbolic chain of thought distillation multiple rationales",
    ),
    (
        "5_MCC-KD_Mutiltationales_SameAnswer.pdf",
        "bidirectional KL divergence consistency multiple rationales same answer",
    ),
    (
        "6_Anaylsis_Cot.pdf",
        "key factors chain of thought distillation rationale quality diversity",
    ),
]

NEGATIVE_QUERY = "quantum computing hardware superconducting qubits"


def build_paper_index(filename: str) -> tuple[list[dict], EmbeddingStore]:
    file_result = read_pdf_file(
        file_path=filename,
        max_pages=None,
        max_characters=120000,
    )
    if file_result.get("error"):
        raise RuntimeError(file_result["error"])

    chunks = chunk_text(file_result["content"])
    embedding_store = EmbeddingStore()
    embedding_store.build_index(chunks)
    return chunks, embedding_store


def retrieve_with_existing_index(
    query: str,
    chunks: list[dict],
    embedding_store: EmbeddingStore,
    top_k: int = 3,
) -> dict:
    """Run the same Hybrid+RRF logic without rebuilding the paper index."""
    expanded_query = expand_query(query)
    candidate_k = min(top_k * 3, len(chunks))
    bm25_candidates = bm25_retrieve(
        query=expanded_query,
        chunks=chunks,
        top_k=candidate_k,
    )
    embedding_candidates = embedding_store.search(
        query=expanded_query,
        top_k=candidate_k,
    )
    retrieved_chunks = reciprocal_rank_fusion(
        ranked_results={
            "bm25": bm25_candidates,
            "embedding": embedding_candidates,
        },
        top_k=top_k,
    )

    return {
        "query": query,
        "expanded_query": expanded_query,
        "retrieval_method": "Hybrid (BM25+Embedding)",
        "fusion_method": "RRF",
        "embedding_model": embedding_store.model_source,
        "candidate_k": candidate_k,
        "bm25_candidates": len(bm25_candidates),
        "embedding_candidates": len(embedding_candidates),
        "retrieved_chunks": retrieved_chunks,
    }


def evaluate_case(
    filename: str,
    query: str,
    chunks: list[dict],
    embedding_store: EmbeddingStore,
) -> dict:
    decision = {
        "needs_retrieval": True,
        "source": filename,
        "query": query,
    }
    retrieval = retrieve_with_existing_index(
        query=query,
        chunks=chunks,
        embedding_store=embedding_store,
        top_k=3,
    )
    return evaluate_retrieval_context(
        retrieval_decision=decision,
        tool_observations=[{
            "tool_used": "retrieve_from_file",
            "tool_result": {
                "file_path": filename,
                "file_type": ".pdf",
                "retrieval": retrieval,
            },
        }],
    )


def compact_result(result: dict) -> dict:
    signals = result.get("signals", {})
    embedding_score = signals.get("maximum_embedding_score")

    return {
        "status": result["status"],
        "reason": result["reason_code"],
        "action": result["recommended_action"],
        "query_coverage": round(
            signals.get("query_term_coverage", 0.0),
            3,
        ),
        "embedding_score": (
            round(embedding_score, 3)
            if embedding_score is not None
            else None
        ),
        "dual_source_chunks": signals.get("dual_source_chunks", 0),
        "retrieval_method": signals.get("retrieval_method"),
    }


def run_evaluation() -> dict:
    results = []

    for filename, positive_query in POSITIVE_CASES:
        # Indexing is document work, not query work. Reuse one FAISS index for
        # both cases to make the experiment faster and more stable.
        chunks, embedding_store = build_paper_index(filename)

        for label, query, expected_sufficient in [
            ("positive", positive_query, True),
            ("negative", NEGATIVE_QUERY, False),
        ]:
            evaluation = evaluate_case(
                filename=filename,
                query=query,
                chunks=chunks,
                embedding_store=embedding_store,
            )
            actual_sufficient = evaluation["is_sufficient"] is True
            results.append({
                "paper": filename,
                "label": label,
                "expected_sufficient": expected_sufficient,
                "actual_sufficient": actual_sufficient,
                "passed": actual_sufficient == expected_sufficient,
                **compact_result(evaluation),
            })

    return {
        "cases": results,
        "passed": sum(item["passed"] for item in results),
        "total": len(results),
    }


if __name__ == "__main__":
    report = run_evaluation()
    for item in report["cases"]:
        print(item)

    print(
        f"Context sufficiency cases passed: "
        f"{report['passed']}/{report['total']}"
    )
