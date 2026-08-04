# A reusable sparse+dense index for one document.

from time import perf_counter

from rank_bm25 import BM25Okapi

from embedding_store import EmbeddingStore
from query_expander import expand_query
from rag_store import chunk_text, reciprocal_rank_fusion, tokenize


class DocumentIndex:
    """Chunk and index one document once, then serve multiple queries."""

    def __init__(
        self,
        text: str,
        chunk_size: int = 1200,
        overlap: int = 200,
        embedding_store: EmbeddingStore | None = None,
    ):
        started_at = perf_counter()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        tokenized_documents = [
            tokenize(chunk["content"])
            for chunk in self.chunks
        ]
        self.bm25 = (
            BM25Okapi(tokenized_documents)
            if tokenized_documents and any(tokenized_documents)
            else None
        )

        self.embedding_store = None
        self.embedding_error = None
        self.embedding_model = EmbeddingStore.configured_model_source()

        if self.chunks:
            try:
                self.embedding_store = embedding_store or EmbeddingStore()
                self.embedding_store.build_index(self.chunks)
                self.embedding_model = self.embedding_store.model_source
            except Exception as error:
                # BM25 remains available if model loading or FAISS indexing
                # fails. Cache the failure with the document so every query
                # does not repeat the same expensive failing initialization.
                self.embedding_store = None
                self.embedding_error = f"{type(error).__name__}: {error}"

        self.build_seconds = perf_counter() - started_at

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        if self.bm25 is None or top_k <= 0:
            return []

        scores = self.bm25.get_scores(tokenize(query))
        ranked_chunks = []

        for index, score in enumerate(scores):
            chunk = self.chunks[index]
            ranked_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "score": float(score),
                "retrieval_source": "bm25",
                "content": chunk["content"],
            })

        ranked_chunks.sort(
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked_chunks[:top_k]

    def search(self, query: str, top_k: int = 3) -> dict:
        expanded_query = expand_query(query)

        if not self.chunks or top_k <= 0:
            return {
                "query": query,
                "expanded_query": expanded_query,
                "retrieval_method": "None",
                "fusion_method": None,
                "embedding_model": self.embedding_model,
                "fallback_reason": "The document produced no chunks.",
                "candidate_k": 0,
                "rank_constant": 60,
                "bm25_candidates": 0,
                "embedding_candidates": 0,
                "retrieved_chunks": [],
            }

        candidate_k = min(max(top_k * 3, top_k), len(self.chunks))
        bm25_candidates = self._bm25_search(
            query=expanded_query,
            top_k=candidate_k,
        )

        if self.embedding_store is None:
            return {
                "query": query,
                "expanded_query": expanded_query,
                "retrieval_method": "BM25 fallback",
                "fusion_method": None,
                "embedding_model": self.embedding_model,
                "fallback_reason": self.embedding_error,
                "candidate_k": candidate_k,
                "rank_constant": None,
                "bm25_candidates": len(bm25_candidates),
                "embedding_candidates": 0,
                "retrieved_chunks": bm25_candidates[:top_k],
            }

        embedding_candidates = self.embedding_store.search(
            query=expanded_query,
            top_k=candidate_k,
        )
        retrieved_chunks = reciprocal_rank_fusion(
            ranked_results={
                "bm25": bm25_candidates,
                "embedding": embedding_candidates,
            },
            top_k=top_k,
            rank_constant=60,
        )

        return {
            "query": query,
            "expanded_query": expanded_query,
            "retrieval_method": "Hybrid (BM25+Embedding)",
            "fusion_method": "RRF",
            "embedding_model": self.embedding_model,
            "fallback_reason": None,
            "candidate_k": candidate_k,
            "rank_constant": 60,
            "bm25_candidates": len(bm25_candidates),
            "embedding_candidates": len(embedding_candidates),
            "retrieved_chunks": retrieved_chunks,
        }
