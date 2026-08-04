import unittest
from unittest.mock import call, patch

import numpy as np

from embedding_store import EmbeddingStore
from rag_store import chunk_text, reciprocal_rank_fusion, tokenize


class FakeEmbeddingModel:
    """Small deterministic model for testing FAISS without network access."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []

        for text in texts:
            lower_text = text.lower()
            if "reasoning" in lower_text:
                vector = np.array([1.0, 0.0], dtype="float32")
            elif "dataset" in lower_text:
                vector = np.array([0.0, 1.0], dtype="float32")
            else:
                vector = np.array([0.5, 0.5], dtype="float32")

            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)

            vectors.append(vector)

        return np.stack(vectors)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        EmbeddingStore._model_cache.clear()

    def test_chunk_overlap_repeats_boundary_text(self):
        chunks = chunk_text(
            text="0123456789",
            chunk_size=6,
            overlap=2,
        )

        self.assertEqual(chunks[0]["content"], "012345")
        self.assertEqual(chunks[1]["content"], "456789")
        self.assertEqual(
            chunks[0]["content"][-2:],
            chunks[1]["content"][:2],
        )

    def test_bm25_tokenization_keeps_term_frequency(self):
        tokens = tokenize("reasoning reasoning action")

        self.assertEqual(tokens.count("reasoning"), 2)
        self.assertEqual(tokens.count("action"), 1)

    def test_rrf_promotes_a_chunk_supported_by_both_retrievers(self):
        bm25_results = [
            {"chunk_id": 1, "content": "BM25 first", "score": 8.2},
            {"chunk_id": 2, "content": "Shared result", "score": 6.0},
        ]
        embedding_results = [
            {"chunk_id": 2, "content": "Shared result", "score": 0.91},
            {"chunk_id": 3, "content": "Embedding second", "score": 0.87},
        ]

        fused = reciprocal_rank_fusion(
            ranked_results={
                "bm25": bm25_results,
                "embedding": embedding_results,
            },
            top_k=3,
        )

        self.assertEqual(fused[0]["chunk_id"], 2)
        self.assertEqual(
            fused[0]["retrieval_sources"],
            ["bm25", "embedding"],
        )

    def test_embedding_search_caps_top_k_to_chunk_count(self):
        chunks = [
            {"chunk_id": 1, "content": "reasoning and action"},
            {"chunk_id": 2, "content": "dataset benchmark"},
        ]
        store = EmbeddingStore(model=FakeEmbeddingModel())
        store.build_index(chunks)

        results = store.search("reasoning process", top_k=10)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], 1)

    @patch("embedding_store.SentenceTransformer")
    def test_embedding_loader_uses_cache_before_network(self, mock_loader):
        cached_model = FakeEmbeddingModel()
        mock_loader.return_value = cached_model

        store = EmbeddingStore(local_files_only=False)

        self.assertIs(store.model, cached_model)
        mock_loader.assert_called_once_with(
            store.model_source,
            local_files_only=True,
        )

    @patch("embedding_store.SentenceTransformer")
    def test_embedding_loader_uses_network_after_cache_miss(self, mock_loader):
        downloaded_model = FakeEmbeddingModel()
        mock_loader.side_effect = [OSError("cache miss"), downloaded_model]

        store = EmbeddingStore(local_files_only=False)

        self.assertIs(store.model, downloaded_model)
        self.assertEqual(
            mock_loader.call_args_list,
            [
                call(store.model_source, local_files_only=True),
                call(store.model_source, local_files_only=False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
