"""Unit tests for backend.rag"""

from backend.embeddings import VectorStore
from backend.rag import answer_question, compute_confidence, dedupe_sources, format_context


def test_compute_confidence_empty_returns_zero():
    assert compute_confidence([]) == 0.0


def test_compute_confidence_uses_top_score():
    chunks = [{"score": 0.3}, {"score": 0.9}, {"score": 0.5}]
    assert compute_confidence(chunks) == 0.9


def test_compute_confidence_clamped_to_unit_interval():
    assert compute_confidence([{"score": 1.5}]) == 1.0
    assert compute_confidence([{"score": -0.5}]) == 0.0


def test_dedupe_sources_removes_duplicate_page_hits():
    chunks = [
        {"source": "a.pdf", "page": 1},
        {"source": "a.pdf", "page": 1},
        {"source": "a.pdf", "page": 2},
        {"source": "b.pdf", "page": None},
    ]
    sources = dedupe_sources(chunks)
    assert len(sources) == 3


def test_format_context_includes_source_labels():
    chunks = [{"source": "fee_structure.pdf", "page": 5, "text": "The fee is 1000."}]
    ctx = format_context(chunks)
    assert "fee_structure.pdf" in ctx
    assert "page 5" in ctx
    assert "The fee is 1000." in ctx


def test_answer_question_low_confidence_triggers_fallback(temp_project_dirs):
    store = VectorStore()
    store.build_from_documents(temp_project_dirs["documents"])  # empty KB -> no chunks
    result = answer_question("What is the hostel fee?", store)
    assert result.is_fallback is True
    assert result.confidence == 0.0
    assert result.sources == []


def test_answer_question_local_provider_returns_extractive_answer(
    temp_project_dirs, sample_txt_document, monkeypatch
):
    from config import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "local")
    monkeypatch.setattr(settings, "CONFIDENCE_THRESHOLD", -1.0)  # force pass-through for this test

    store = VectorStore()
    store.build_from_documents(temp_project_dirs["documents"])
    result = answer_question("What is the hostel fee?", store)

    assert result.is_fallback is False
    assert result.confidence >= -1.0
    assert len(result.sources) >= 1
    assert result.sources[0]["source"] == "sample.txt"
