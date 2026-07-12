"""
backend/routes.py
==================
FastAPI route definitions, split into:
  - /chat/*   student-facing chat endpoints
  - /admin/*  document management, analytics, log export

All routes are thin: they validate input via Pydantic models and delegate to
backend.chatbot / backend.database / backend.embeddings for actual logic.
"""

from __future__ import annotations

import csv
import io
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.chatbot import Chatbot, get_chatbot
from backend.database import (
    delete_document_record,
    get_analytics_summary,
    get_chat_history,
    get_common_questions,
    list_documents,
    record_feedback,
    register_document,
)
from backend.document_loader import SUPPORTED_EXTENSIONS
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["chat"])
admin_router = APIRouter(prefix="/admin", tags=["admin"])


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class SourceOut(BaseModel):
    source: str
    page: int | None = None


class ChatResponseOut(BaseModel):
    session_id: str
    answer: str
    confidence: float
    sources: list[SourceOut]
    intent: str
    answered_by: str
    message_id: int | None


class FeedbackRequest(BaseModel):
    message_id: int
    is_helpful: bool
    comment: str | None = None


class HistoryItemOut(BaseModel):
    question: str
    answer: str
    confidence: float | None
    timestamp: str


# --------------------------------------------------------------------------
# Chat endpoints
# --------------------------------------------------------------------------
@chat_router.post("", response_model=ChatResponseOut)
def chat(payload: ChatRequest, bot: Chatbot = Depends(get_chatbot)) -> ChatResponseOut:
    """Send a message and receive a grounded, cited, confidence-scored answer."""
    try:
        result = bot.chat(question=payload.message, session_id=payload.session_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat pipeline failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    return ChatResponseOut(
        session_id=result.session_id,
        answer=result.answer,
        confidence=result.confidence,
        sources=[SourceOut(**s) for s in result.sources],
        intent=result.intent,
        answered_by=result.answered_by,
        message_id=result.message_id,
    )


@chat_router.get("/history/{session_id}", response_model=list[HistoryItemOut])
def history(session_id: str) -> list[HistoryItemOut]:
    rows = get_chat_history(session_id)
    return [
        HistoryItemOut(
            question=r.question,
            answer=r.answer,
            confidence=r.confidence,
            timestamp=r.timestamp.isoformat(),
        )
        for r in rows
    ]


@chat_router.post("/feedback")
def feedback(payload: FeedbackRequest) -> dict:
    record_feedback(payload.message_id, payload.is_helpful, payload.comment)
    return {"status": "ok"}


@chat_router.get("/suggested-questions", response_model=list[str])
def suggested_questions() -> list[str]:
    return [
        "What is admission process?",
        "How can I apply for scholarship?",
        "What is hostel fee?",
        "What is library timing?",
        "What is examination fee?",
        "Is there transport facility?",
        "What is minimum attendance?",
        "Can I pay fees online?",
    ]


# --------------------------------------------------------------------------
# Admin endpoints
# --------------------------------------------------------------------------
@admin_router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    dest = settings.DOCUMENTS_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    register_document(file.filename, indexed=False)
    logger.info("Uploaded document: %s", file.filename)
    return {"status": "uploaded", "filename": file.filename}


@admin_router.get("/documents")
def documents() -> list[dict]:
    return [
        {"filename": d.filename, "uploaded_at": d.uploaded_at.isoformat(), "indexed": d.indexed}
        for d in list_documents()
    ]


@admin_router.delete("/documents/{filename}")
def delete_document(filename: str) -> dict:
    path = settings.DOCUMENTS_DIR / filename
    if path.exists():
        path.unlink()
    delete_document_record(filename)
    logger.info("Deleted document: %s", filename)
    return {"status": "deleted", "filename": filename}


@admin_router.post("/rebuild-index")
def rebuild_index(bot: Chatbot = Depends(get_chatbot)) -> dict:
    count = bot.rebuild_index()
    for d in list_documents():
        register_document(d.filename, indexed=True)
    return {"status": "rebuilt", "chunks_indexed": count}


@admin_router.get("/analytics")
def analytics() -> dict:
    return get_analytics_summary()


@admin_router.get("/common-questions")
def common_questions(limit: int = 10) -> list[dict]:
    return [{"question": q, "count": c} for q, c in get_common_questions(limit)]


@admin_router.get("/export-logs")
def export_logs() -> StreamingResponse:
    """Export the full chat history table as a downloadable CSV."""
    from backend.database import get_session, ChatMessage

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["session_id", "question", "answer", "confidence", "source", "intent", "answered_by", "timestamp"])

    with get_session() as db:
        rows = db.query(ChatMessage).order_by(ChatMessage.timestamp.asc()).all()
        for r in rows:
            writer.writerow(
                [r.session_id, r.question, r.answer, r.confidence, r.source, r.intent, r.answered_by, r.timestamp.isoformat()]
            )

    buffer.seek(0)
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=chat_logs.csv"},
    )
