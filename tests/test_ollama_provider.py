"""
Unit tests for backend.rag.OllamaProvider — verifies the real HTTP request/
response contract against a lightweight mock server (no actual Ollama
installation required), including the three failure modes surfaced during
integration testing: connection refused, model not found, and success.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


class _MockOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence default request logging
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        model = body.get("model")

        if model == "does-not-exist":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error":"model not found"}')
            return

        payload = json.dumps({"model": model, "response": "mock generated answer", "done": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def mock_ollama_server():
    server = HTTPServer(("localhost", 0), _MockOllamaHandler)  # port 0 = OS picks a free port
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://localhost:{port}"
    server.shutdown()


def test_ollama_provider_success(mock_ollama_server, monkeypatch):
    from config import settings
    from backend.rag import OllamaProvider

    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", mock_ollama_server)
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "llama3")

    provider = OllamaProvider()
    assert provider.generate("hello") == "mock generated answer"


def test_ollama_provider_model_not_found(mock_ollama_server, monkeypatch):
    from config import settings
    from backend.rag import OllamaProvider

    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", mock_ollama_server)
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "does-not-exist")

    provider = OllamaProvider()
    with pytest.raises(RuntimeError, match="not found"):
        provider.generate("hello")


def test_ollama_provider_connection_refused(monkeypatch):
    from config import settings
    from backend.rag import OllamaProvider

    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://localhost:1")  # nothing listens here
    monkeypatch.setattr(settings, "OLLAMA_MODEL", "llama3")

    provider = OllamaProvider()
    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        provider.generate("hello")


def test_answer_question_falls_back_gracefully_when_ollama_unreachable(
    temp_project_dirs, sample_txt_document, monkeypatch
):
    """If Ollama is configured but unreachable, the pipeline should degrade to
    the extractive answer rather than raising an error up to the caller."""
    from config import settings
    from backend.embeddings import VectorStore
    from backend.rag import answer_question

    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://localhost:1")
    monkeypatch.setattr(settings, "CONFIDENCE_THRESHOLD", -1.0)  # force pass-through for this test

    store = VectorStore()
    store.build_from_documents(temp_project_dirs["documents"])
    result = answer_question("What is the hostel fee?", store)

    assert result.is_fallback is False  # confidence was high enough to attempt an answer
    assert len(result.sources) >= 1  # still grounded/cited even though the LLM call failed
