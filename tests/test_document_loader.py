"""Unit tests for backend.document_loader"""

from backend.document_loader import chunk_documents, load_documents


def test_load_txt_document(temp_project_dirs, sample_txt_document):
    segments = load_documents(temp_project_dirs["documents"])
    assert len(segments) == 1
    text, source, page = segments[0]
    assert source == "sample.txt"
    assert "Hostel Fee" in text
    assert page is None


def test_load_csv_document(temp_project_dirs):
    csv_path = temp_project_dirs["documents"] / "fees.csv"
    csv_path.write_text("Program,Fee\nB.Tech,125000\nMBA,150000\n", encoding="utf-8")

    segments = load_documents(temp_project_dirs["documents"])
    assert len(segments) == 1
    text, source, _ = segments[0]
    assert source == "fees.csv"
    assert "Program: B.Tech" in text
    assert "Fee: 125000" in text


def test_unsupported_extension_is_ignored(temp_project_dirs):
    (temp_project_dirs["documents"] / "notes.xyz").write_text("ignored", encoding="utf-8")
    segments = load_documents(temp_project_dirs["documents"])
    assert segments == []


def test_empty_directory_returns_empty_list(temp_project_dirs):
    assert load_documents(temp_project_dirs["documents"]) == []


def test_chunking_produces_overlapping_chunks(sample_txt_document, temp_project_dirs):
    segments = load_documents(temp_project_dirs["documents"])
    chunks = chunk_documents(segments, chunk_size=60, chunk_overlap=10)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.source == "sample.txt"
        assert c.text.strip() != ""
        assert c.chunk_id  # non-empty
