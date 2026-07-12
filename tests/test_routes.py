"""Unit tests for backend.routes / app.py (FastAPI endpoints)"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(temp_project_dirs, sample_txt_document):
    from app import app

    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_endpoint_returns_answer_with_confidence(client):
    resp = client.post("/chat", json={"message": "What is hostel fee?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["session_id"]


def test_chat_endpoint_rejects_empty_message(client):
    resp = client.post("/chat", json={"message": ""})
    assert resp.status_code == 422  # pydantic min_length validation


def test_feedback_endpoint(client):
    chat_resp = client.post("/chat", json={"message": "What is hostel fee?"}).json()
    resp = client.post(
        "/chat/feedback", json={"message_id": chat_resp["message_id"], "is_helpful": True}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_history_endpoint_returns_prior_turns(client):
    chat_resp = client.post("/chat", json={"message": "What is library timing?"}).json()
    resp = client.get(f"/chat/history/{chat_resp['session_id']}")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_admin_upload_rejects_unsupported_extension(client):
    resp = client.post(
        "/admin/upload", files={"file": ("notes.xyz", b"hello world", "text/plain")}
    )
    assert resp.status_code == 400


def test_admin_upload_and_rebuild_flow(client):
    upload_resp = client.post(
        "/admin/upload", files={"file": ("extra_notes.txt", b"Exam fee is 1500 rupees.", "text/plain")}
    )
    assert upload_resp.status_code == 200

    rebuild_resp = client.post("/admin/rebuild-index")
    assert rebuild_resp.status_code == 200
    assert rebuild_resp.json()["chunks_indexed"] >= 1

    docs_resp = client.get("/admin/documents")
    filenames = [d["filename"] for d in docs_resp.json()]
    assert "extra_notes.txt" in filenames


def test_admin_analytics_endpoint(client):
    client.post("/chat", json={"message": "What is hostel fee?"})
    resp = client.get("/admin/analytics")
    assert resp.status_code == 200
    assert resp.json()["total_messages"] >= 1


def test_admin_export_logs_returns_csv(client):
    client.post("/chat", json={"message": "What is hostel fee?"})
    resp = client.get("/admin/export-logs")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert b"session_id" in resp.content
