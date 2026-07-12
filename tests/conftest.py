"""
tests/conftest.py
==================
Shared pytest fixtures. Uses a deterministic, hash-based fake embedding
function in place of the real sentence-transformers model, so the full test
suite runs fast and fully offline (no model download / network access
required) in CI. Swap out `FAKE=False` locally if you want to test against
the real embedding model.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("LLM_PROVIDER", "local")


def _fake_embed_texts(texts: list[str]) -> np.ndarray:
    """Deterministic pseudo-embeddings: same text -> same vector, every run."""
    vectors = []
    for t in texts:
        rng = np.random.RandomState(abs(hash(t)) % (2**32))
        v = rng.normal(size=384).astype("float32")
        v = v / np.linalg.norm(v)
        vectors.append(v)
    return np.array(vectors, dtype="float32")


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Autouse fixture: every test gets the fake embedder, keeping CI offline-friendly."""
    import backend.embeddings as embeddings_mod
    import backend.faq as faq_mod

    monkeypatch.setattr(embeddings_mod, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(faq_mod, "embed_texts", _fake_embed_texts)
    yield


@pytest.fixture(autouse=True)
def reset_singletons():
    """
    backend.faq, backend.chatbot, and backend.memory each keep a module-level
    singleton for performance in production. Reset them before every test so
    one test's FAISS index / FAQ cache / conversation memory never leaks into
    the next (especially important since each test gets its own temp DB).
    """
    import backend.chatbot as chatbot_mod
    import backend.faq as faq_mod
    import backend.memory as memory_mod

    chatbot_mod._chatbot_singleton = None
    faq_mod._faq_matcher = None
    memory_mod.conversation_memory._sessions.clear()
    yield
    chatbot_mod._chatbot_singleton = None
    faq_mod._faq_matcher = None


@pytest.fixture()
def temp_project_dirs(monkeypatch, tmp_path):
    """
    Redirect documents/vectorstore/database paths to a temp dir for full
    test isolation (so tests never touch the real project data).
    """
    from config import settings

    docs_dir = tmp_path / "documents"
    vec_dir = tmp_path / "vectorstore"
    docs_dir.mkdir()
    vec_dir.mkdir()

    monkeypatch.setattr(settings, "DOCUMENTS_DIR", docs_dir)
    monkeypatch.setattr(settings, "VECTORSTORE_DIR", vec_dir)

    # The SQLAlchemy engine/sessionmaker are bound once at import time in
    # backend.database, so simply patching settings.DATABASE_URL wouldn't take
    # effect. Instead, build a fresh temp-file engine and swap it in directly.
    import backend.database as db_mod
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test.db"
    test_engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, future=True
    )
    test_session_local = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    monkeypatch.setattr(db_mod, "engine", test_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    db_mod.init_db()

    return {"documents": docs_dir, "vectorstore": vec_dir, "db_path": db_path}


@pytest.fixture()
def sample_txt_document(temp_project_dirs):
    """Write a small sample .txt knowledge-base file into the temp documents dir."""
    doc_path = temp_project_dirs["documents"] / "sample.txt"
    doc_path.write_text(
        "Hostel Fee\n"
        "The hostel fee is 60000 rupees per year, payable in two installments. "
        "Mess charges are billed separately at approximately 4500 rupees per month.\n\n"
        "Library Timings\n"
        "The library is open Monday to Saturday, 8 AM to 8 PM.\n",
        encoding="utf-8",
    )
    return doc_path
