# evaluate_retrieval.py

from pathlib import Path

from embedding_store import EmbeddingStore
from query_expander import expand_query
from rag_store import bm25_retrieve, chunk_text, hybrid_retrieve
from retrieval_evaluator import evaluate_retriever, print_evaluation_report


DOCUMENT_PATH = Path("files/ReAct.md")
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40

# These labels were created by reading the chunks, not by asking a retriever.
# That independence is what makes them usable as ground truth.
EVALUATION_CASES = [
    {
        "name": "method",
        "query": "How does ReAct combine reasoning and acting?",
        "relevant_chunk_ids": [2],
    },
    {
        "name": "datasets",
        "query": "Which datasets are used to evaluate ReAct?",
        "relevant_chunk_ids": [3],
    },
    {
        "name": "result",
        "query": "What result does ReAct achieve compared with Chain-of-Thought?",
        "relevant_chunk_ids": [4],
    },
]


def run_evaluation() -> list[dict]:
    text = DOCUMENT_PATH.read_text(encoding="utf-8")
    chunks = chunk_text(
        text=text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    embedding_store = EmbeddingStore()
    embedding_store.build_index(chunks)

    def bm25_search(query: str, top_k: int) -> list[dict]:
        return bm25_retrieve(
            query=expand_query(query),
            chunks=chunks,
            top_k=top_k,
        )

    def embedding_search(query: str, top_k: int) -> list[dict]:
        return embedding_store.search(
            query=expand_query(query),
            top_k=top_k,
        )

    def hybrid_search(query: str, top_k: int) -> list[dict]:
        results, _ = hybrid_retrieve(
            query=expand_query(query),
            chunks=chunks,
            embedding_store=embedding_store,
            top_k=top_k,
        )
        return results

    retrievers = {
        "BM25": bm25_search,
        "Embedding+FAISS": embedding_search,
        "Hybrid+RRF": hybrid_search,
    }

    reports = []
    for top_k in [1, 2]:
        for name, retrieve in retrievers.items():
            reports.append(
                evaluate_retriever(
                    retriever_name=name,
                    cases=EVALUATION_CASES,
                    retrieve=retrieve,
                    top_k=top_k,
                )
            )

    return reports


if __name__ == "__main__":
    print_evaluation_report(run_evaluation())
