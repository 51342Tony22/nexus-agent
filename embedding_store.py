# embedding_store.py

from pathlib import Path

# Import config first. config.py loads .env, including HF_ENDPOINT, before
# sentence-transformers imports huggingface_hub.
from config import (
    EMBEDDING_CPU_THREADS,
    EMBEDDING_DEVICE,
    EMBEDDING_LOCAL_FILES_ONLY,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_PATH,
    HF_ENDPOINT,
)

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# The current macOS arm64 + Python 3.14 runtime can crash inside PyTorch's
# multi-threaded CPU encoder. Keep this configurable and default to the stable
# single-threaded setting for this small learning project.
torch.set_num_threads(EMBEDDING_CPU_THREADS)


class EmbeddingModelLoadError(RuntimeError):
    """Raised when the configured embedding model cannot be loaded."""


class EmbeddingStore:
    """Build and search an in-memory FAISS index for document chunks."""

    # Loading a transformer is expensive. Stores created later reuse the same
    # model when their source and settings are identical.
    _model_cache = {}

    def __init__(
        self,
        model_name: str | None = None,
        local_model_path: str | None = None,
        local_files_only: bool | None = None,
        model=None,
    ):
        self.model_name = model_name or EMBEDDING_MODEL
        self.local_model_path = local_model_path or EMBEDDING_MODEL_PATH
        self.local_files_only = (
            EMBEDDING_LOCAL_FILES_ONLY
            if local_files_only is None
            else local_files_only
        )
        self.model_source = self._resolve_model_source()

        # Model injection keeps the FAISS logic testable without downloading a
        # real model.
        self.model = model or self._get_or_load_model()
        self.index = None
        self.chunks = []

    @classmethod
    def configured_model_source(cls) -> str:
        if EMBEDDING_MODEL_PATH:
            return str(Path(EMBEDDING_MODEL_PATH).expanduser())
        return EMBEDDING_MODEL

    def _resolve_model_source(self) -> str:
        if not self.local_model_path:
            return self.model_name

        local_path = Path(self.local_model_path).expanduser()
        if not local_path.exists():
            raise EmbeddingModelLoadError(
                f"Local embedding model path does not exist: {local_path}"
            )

        # A local path must never trigger an accidental network request.
        self.local_files_only = True
        return str(local_path.resolve())

    def _get_or_load_model(self):
        cache_key = (
            self.model_source,
            self.local_files_only,
            EMBEDDING_DEVICE,
        )

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model_kwargs = {}
        if EMBEDDING_DEVICE:
            model_kwargs["device"] = EMBEDDING_DEVICE

        # When online mode is allowed, still try the local Hugging Face cache
        # first. This prevents a complete cached model from failing merely
        # because an optional remote metadata check cannot reach the network.
        # A cache miss then falls through to the configured HF endpoint.
        load_modes = (
            [True]
            if self.local_files_only
            else [True, False]
        )
        load_errors = []

        for local_only in load_modes:
            try:
                model = SentenceTransformer(
                    self.model_source,
                    local_files_only=local_only,
                    **model_kwargs,
                )
                break
            except Exception as error:
                load_errors.append((local_only, error))
        else:
            final_local_only, final_error = load_errors[-1]
            mode = "local-only" if final_local_only else "online/cache"
            raise EmbeddingModelLoadError(
                "Failed to load embedding model "
                f"'{self.model_source}' in {mode} mode via {HF_ENDPOINT}. "
                "Configure EMBEDDING_MODEL_PATH with a complete local model, "
                "or configure HF_ENDPOINT and retry the download. "
                f"Original error: {type(final_error).__name__}: {final_error}"
            ) from final_error

        self._model_cache[cache_key] = model
        return model

    def build_index(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("Cannot build a FAISS index from an empty chunk list.")

        self.chunks = chunks
        texts = [chunk["content"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding model returned an unexpected matrix shape.")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None:
            raise ValueError("FAISS index has not been built. Call build_index first.")

        if not query.strip() or top_k <= 0:
            return []

        actual_top_k = min(top_k, len(self.chunks))
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.asarray(query_embedding, dtype="float32")

        scores, indices = self.index.search(
            query_embedding,
            actual_top_k,
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "start": chunk.get("start"),
                "end": chunk.get("end"),
                "score": float(score),
                "retrieval_source": "embedding",
                "content": chunk["content"],
            })

        return results
