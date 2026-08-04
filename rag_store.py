#rag_store.py

import re
from query_expander import expand_query
from rank_bm25 import BM25Okapi
from embedding_store import EmbeddingStore

def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[\w']+", text)

    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with",
        "is", "are", "was", "were", "be", "by", "as", "at", "from", "that",
        "this", "it", "its", "into", "we", "our", "you", "your",
        "什么", "怎么", "如何", "分析", "总结", "文件", "论文"
    }

    # Keep duplicate tokens. BM25 uses term frequency, so returning a set here
    # would silently remove part of the signal it needs.
    return [
        token
        for token in tokens
        if token not in stopwords and len(token) > 1
    ]

def chunk_text(
        text:str,
        chunk_size:int= 1200,
        overlap:int= 200,

)->list[dict]:
    chunks = []

    if not text:
        return chunks

    start = 0
    chunk_id = 1

    while start < len(text):
        end=start +chunk_size
        chunk_content = text[start:end]

        chunks.append({
            "chunk_id": chunk_id,
            "start": start,
            "end": min(end, len(text)),
            "content": chunk_content
        })

        chunk_id += 1

        if end >= len(text):
            break

        start = end - overlap

    return chunks

def score_chunk(query: str, chunk_content: str) -> int:
    query_tokens = tokenize(query)
    chunk_tokens = tokenize(chunk_content)

    if not query_tokens or not chunk_tokens:
        return 0

    overlap = set(query_tokens).intersection(chunk_tokens)

    return len(overlap)

def retrieve_relevant_chunks(
    query: str,
    text: str,
    top_k: int = 3,
    chunk_size: int = 1200,
    overlap: int = 200
):
    expanded_query = expand_query(query)
    chunks = chunk_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap
    )

    if not chunks:
        return {
            "query": query,
            "expanded_query": expanded_query,
            "retrieval_method": "None",
            "retrieved_chunks": [],
            "fallback_reason": "The document produced no chunks."
        }

    try:
        embedding_store = EmbeddingStore()
        top_chunks, hybrid_debug = hybrid_retrieve(
            query=expanded_query,
            chunks=chunks,
            embedding_store=embedding_store,
            top_k=top_k,
        )

        return {
            "query": query,
            "expanded_query": expanded_query,
            "retrieval_method": "Hybrid (BM25+Embedding)",
            "fusion_method": "RRF",
            "embedding_model": embedding_store.model_source,
            **hybrid_debug,
            "retrieved_chunks": top_chunks
        }
    except Exception as error:
        # Retrieval is an external-dependency boundary. If the model or FAISS
        # fails, BM25 keeps the Agent useful while Debug preserves the reason.
        top_chunks = bm25_retrieve(expanded_query, chunks, top_k)
        return {
            "query": query,
            "expanded_query": expanded_query,
            "retrieval_method": "BM25 fallback",
            "fusion_method": None,
            "embedding_model": EmbeddingStore.configured_model_source(),
            "fallback_reason": f"{type(error).__name__}: {error}",
            "retrieved_chunks": top_chunks
        }

def bm25_retrieve(
    query: str,
    chunks: list[dict],
    top_k: int = 3
):

    if not chunks or top_k <= 0:
        return []

    documents = []

    for chunk in chunks:
        tokens = list(
            tokenize(
                chunk["content"]
            )
        )

        documents.append(tokens)


    if not any(documents):
        return []

    bm25 = BM25Okapi(documents)


    query_tokens = list(
        tokenize(query)
    )


    scores = bm25.get_scores(
        query_tokens
    )


    ranked_chunks = []


    for index, score in enumerate(scores):

        ranked_chunks.append(
            {
                "chunk_id": chunks[index]["chunk_id"],
                "start": chunks[index].get("start"),
                "end": chunks[index].get("end"),
                "score": float(score),
                "retrieval_source": "bm25",
                "content": chunks[index]["content"]
            }
        )


    ranked_chunks.sort(
        key=lambda x:x["score"],
        reverse=True
    )


    return ranked_chunks[:top_k]


def reciprocal_rank_fusion(
    ranked_results: dict[str, list[dict]],
    top_k: int = 3,
    rank_constant: int = 60,
) -> list[dict]:
    """Fuse independent rankings without comparing their raw score scales."""
    if top_k <= 0 or not ranked_results:
        return []

    if rank_constant < 1:
        raise ValueError("rank_constant must be at least 1.")

    fused_by_chunk = {}

    for source_name, results in ranked_results.items():
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]

            if chunk_id not in fused_by_chunk:
                fused_by_chunk[chunk_id] = {
                    "chunk_id": chunk_id,
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "content": item["content"],
                    "rrf_score": 0.0,
                    "retrieval_source": "hybrid",
                    "retrieval_sources": [],
                    "source_ranks": {},
                    "source_scores": {},
                }

            fused_item = fused_by_chunk[chunk_id]
            fused_item["rrf_score"] += 1.0 / (rank_constant + rank)
            fused_item["retrieval_sources"].append(source_name)
            fused_item["source_ranks"][source_name] = rank
            fused_item["source_scores"][source_name] = item["score"]

    # With two retrievers, rank 1 in both is the maximum possible RRF score.
    # Normalize it to 1.0 so Debug and prompts get an easier-to-read score.
    maximum_score = len(ranked_results) / (rank_constant + 1)

    fused_results = list(fused_by_chunk.values())
    for item in fused_results:
        item["score"] = item["rrf_score"] / maximum_score

    fused_results.sort(
        key=lambda item: (-item["rrf_score"], item["chunk_id"])
    )

    return fused_results[:top_k]


def hybrid_retrieve(
    query: str,
    chunks: list[dict],
    embedding_store: EmbeddingStore,
    top_k: int = 3,
    candidate_k: int | None = None,
    rank_constant: int = 60,
) -> tuple[list[dict], dict]:
    """Run sparse and dense retrieval, then combine them with RRF."""
    if not chunks or top_k <= 0:
        return [], {
            "candidate_k": 0,
            "rank_constant": rank_constant,
            "bm25_candidates": 0,
            "embedding_candidates": 0,
        }

    if candidate_k is None:
        candidate_k = top_k * 3

    candidate_k = min(max(candidate_k, top_k), len(chunks))

    bm25_candidates = bm25_retrieve(
        query=query,
        chunks=chunks,
        top_k=candidate_k,
    )

    embedding_store.build_index(chunks)
    embedding_candidates = embedding_store.search(
        query=query,
        top_k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        ranked_results={
            "bm25": bm25_candidates,
            "embedding": embedding_candidates,
        },
        top_k=top_k,
        rank_constant=rank_constant,
    )

    return fused_results, {
        "candidate_k": candidate_k,
        "rank_constant": rank_constant,
        "bm25_candidates": len(bm25_candidates),
        "embedding_candidates": len(embedding_candidates),
    }
