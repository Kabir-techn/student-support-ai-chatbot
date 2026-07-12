"""Unit tests for backend.embeddings (VectorStore / FAISS integration)"""

from backend.embeddings import VectorStore


def test_build_from_documents_indexes_all_chunks(temp_project_dirs, sample_txt_document):
    store = VectorStore()
    count = store.build_from_documents(temp_project_dirs["documents"])
    assert count > 0
    assert store.index is not None
    assert store.index.ntotal == count


def test_build_with_no_documents_yields_empty_index(temp_project_dirs):
    store = VectorStore()
    count = store.build_from_documents(temp_project_dirs["documents"])
    assert count == 0
    assert store.search("anything") == []


def test_search_returns_scored_results(temp_project_dirs, sample_txt_document):
    store = VectorStore()
    store.build_from_documents(temp_project_dirs["documents"])
    results = store.search("hostel fee", top_k=2)
    assert len(results) <= 2
    for r in results:
        assert "score" in r
        assert "text" in r
        assert r["source"] == "sample.txt"


def test_persistence_roundtrip(temp_project_dirs, sample_txt_document):
    store = VectorStore()
    store.build_from_documents(temp_project_dirs["documents"])

    reloaded = VectorStore()
    assert reloaded.load() is True
    assert reloaded.index.ntotal == store.index.ntotal
    assert len(reloaded.metadata) == len(store.metadata)
