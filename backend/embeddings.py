"""
backend/embeddings.py
======================
Wraps Sentence-Transformers for embedding generation and FAISS for
similarity search / persistence.

Pipeline stage: Chunking -> Embeddings -> FAISS -> Retriever
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.document_loader import Chunk, build_knowledge_chunks
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_model_lock = threading.Lock()
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load (and cache) the sentence-transformers model — expensive to init."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
                _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a list of strings into L2-normalized embedding vectors (for cosine sim via inner product)."""
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.astype("float32")


class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP (inner product == cosine similarity,
    since embeddings are L2-normalized) plus a parallel metadata store.
    """

    def __init__(self, index_name: str | None = None):
        self.index_name = index_name or settings.FAISS_INDEX_NAME
        self.index_path = settings.VECTORSTORE_DIR / f"{self.index_name}.faiss"
        self.meta_path = settings.VECTORSTORE_DIR / f"{self.index_name}.meta.json"
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []  # parallel array: metadata[i] <-> index vector i

    # ---------------- build / rebuild ----------------
    def build_from_documents(self, documents_dir: Path | None = None) -> int:
        """Full pipeline: load docs -> chunk -> embed -> build FAISS index. Returns #chunks indexed."""
        chunks: list[Chunk] = build_knowledge_chunks(documents_dir)
        if not chunks:
            logger.warning("No chunks produced; vector store will be empty.")
            self.index = faiss.IndexFlatIP(settings.EMBEDDING_DIM)
            self.metadata = []
            self._save()
            return 0

        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        self.index = index
        self.metadata = [
            {"text": c.text, **c.metadata()} for c in chunks
        ]
        self._save()
        logger.info("Built FAISS index with %d vectors (dim=%d)", index.ntotal, dim)
        return index.ntotal

    # ---------------- persistence ----------------
    def _save(self) -> None:
        settings.VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        """Load a previously built index from disk. Returns True if successful."""
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        self.index = faiss.read_index(str(self.index_path))
        self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
        return True

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    # ---------------- search ----------------
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        Search the index for the closest chunks to `query`.
        Returns a list of dicts: {text, source, page, chunk_id, score} sorted by score desc.
        `score` is a cosine similarity in [-1, 1] (typically [0, 1] for normalized text embeddings).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        top_k = top_k or settings.TOP_K
        query_vec = embed_texts([query])
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            item = dict(self.metadata[idx])
            item["score"] = float(score)
            results.append(item)
        return results


def get_or_build_vector_store(force_rebuild: bool = False) -> VectorStore:
    """Convenience helper used by the app at startup / admin rebuild action."""
    store = VectorStore()
    if not force_rebuild and store.load():
        return store
    store.build_from_documents()
    return store
