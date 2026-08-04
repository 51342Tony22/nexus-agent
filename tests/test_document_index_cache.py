import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from document_index import DocumentIndex
from document_index_cache import DocumentIndexCache
from embedding_store import EmbeddingStore
from tools import retrieve_from_file


class CountingEmbeddingModel:
    def __init__(self):
        self.batch_sizes = []

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        self.batch_sizes.append(len(texts))
        vectors = []

        for text in texts:
            lower_text = text.lower()
            vector = np.array(
                [
                    1.0 if "reasoning" in lower_text else 0.1,
                    1.0 if "dataset" in lower_text else 0.1,
                ],
                dtype="float32",
            )
            if normalize_embeddings:
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)

        return np.stack(vectors)


class FakeDocumentIndex:
    build_count = 0

    def __init__(self, text, chunk_size, overlap):
        type(self).build_count += 1
        self.build_seconds = 0.01
        self.text = text

    def search(self, query, top_k=3):
        return {
            "query": query,
            "expanded_query": query,
            "retrieval_method": "BM25 fallback",
            "fusion_method": None,
            "embedding_model": "fake",
            "fallback_reason": None,
            "candidate_k": 1,
            "rank_constant": None,
            "bm25_candidates": 1,
            "embedding_candidates": 0,
            "retrieved_chunks": [{
                "chunk_id": 1,
                "content": self.text[:100],
                "score": 1.0,
                "retrieval_source": "bm25",
            }],
        }


class DocumentIndexCacheTests(unittest.TestCase):
    def test_cache_hit_reuses_the_same_value(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.txt"
            path.write_text("first version", encoding="utf-8")
            cache = DocumentIndexCache(max_entries=2)
            build_calls = []

            def builder():
                build_calls.append(True)
                return object()

            first, first_debug = cache.get_or_build(path, (1200, 200), builder)
            second, second_debug = cache.get_or_build(path, (1200, 200), builder)

            self.assertIs(first, second)
            self.assertFalse(first_debug["hit"])
            self.assertTrue(second_debug["hit"])
            self.assertEqual(len(build_calls), 1)
            self.assertEqual(cache.build_count, 1)

    def test_file_change_invalidates_cached_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.txt"
            path.write_text("version one", encoding="utf-8")
            cache = DocumentIndexCache(max_entries=2)
            values = iter(["index-v1", "index-v2"])

            first, _ = cache.get_or_build(path, (1200, 200), lambda: next(values))
            path.write_text("version two is longer", encoding="utf-8")
            second, debug = cache.get_or_build(
                path,
                (1200, 200),
                lambda: next(values),
            )

            self.assertEqual(first, "index-v1")
            self.assertEqual(second, "index-v2")
            self.assertFalse(debug["hit"])
            self.assertEqual(cache.invalidation_count, 1)

    def test_lru_cache_evicts_oldest_document(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = DocumentIndexCache(max_entries=2)
            paths = []

            for index in range(3):
                path = Path(directory) / f"paper-{index}.txt"
                path.write_text(str(index), encoding="utf-8")
                paths.append(path)
                cache.get_or_build(path, (1200, 200), lambda i=index: i)

            self.assertEqual(len(cache), 2)
            self.assertEqual(cache.eviction_count, 1)

    def test_document_index_embeds_chunks_once_for_multiple_queries(self):
        model = CountingEmbeddingModel()
        embedding_store = EmbeddingStore(model=model)
        index = DocumentIndex(
            text=(
                "Reasoning methods explain intermediate steps. "
                "Dataset benchmarks measure model performance. " * 4
            ),
            chunk_size=90,
            overlap=20,
            embedding_store=embedding_store,
        )

        first = index.search("reasoning method", top_k=2)
        second = index.search("dataset benchmark", top_k=2)

        self.assertEqual(first["retrieval_method"], "Hybrid (BM25+Embedding)")
        self.assertEqual(second["retrieval_method"], "Hybrid (BM25+Embedding)")
        self.assertGreater(model.batch_sizes[0], 1)
        self.assertEqual(model.batch_sizes[1:], [1, 1])

    def test_file_tool_reports_miss_then_hit(self):
        cache = DocumentIndexCache(max_entries=2)
        FakeDocumentIndex.build_count = 0

        with (
            patch("tools.DOCUMENT_INDEX_CACHE", cache),
            patch("tools.DocumentIndex", FakeDocumentIndex),
        ):
            first = retrieve_from_file(
                file_path="ReAct.md",
                query="reasoning",
            )
            second = retrieve_from_file(
                file_path="ReAct.md",
                query="acting",
            )

        self.assertFalse(first["retrieval"]["index_cache"]["hit"])
        self.assertTrue(second["retrieval"]["index_cache"]["hit"])
        self.assertEqual(FakeDocumentIndex.build_count, 1)


if __name__ == "__main__":
    unittest.main()
