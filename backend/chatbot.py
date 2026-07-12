"""
backend/chatbot.py
===================
Top-level orchestrator invoked by the API routes / Streamlit frontend.

Flow for every incoming message:
  1. Resolve / create a session_id (conversation continuity)
  2. Lightweight intent detection (for analytics + FAQ category hinting)
  3. FAQ Mode: exact / semantic cache lookup -> instant answer if matched
  4. Otherwise: RAG pipeline (retrieve -> ground -> generate -> cite -> score)
  5. Update short-term conversation memory
  6. Persist the exchange to SQLite (chat_history)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from backend.database import new_session_id, save_chat_message
from backend.embeddings import VectorStore, get_or_build_vector_store
from backend.faq import get_faq_matcher
from backend.memory import conversation_memory
from backend.rag import RAGResult, answer_question
from utils.logger import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Lightweight intent / entity detection
# --------------------------------------------------------------------------
INTENT_KEYWORDS: dict[str, list[str]] = {
    "admissions": ["admission", "apply", "enroll", "eligibility", "documents required"],
    "fees": ["fee", "fees", "payment", "pay", "installment", "tuition"],
    "scholarship": ["scholarship", "financial aid", "grant"],
    "hostel": ["hostel", "mess", "accommodation", "room"],
    "library": ["library", "book", "e-book", "reading room"],
    "placements": ["placement", "job", "company", "recruit", "salary", "package"],
    "examinations": ["exam", "semester", "result", "marksheet", "revaluation"],
    "attendance": ["attendance", "present", "absent"],
    "academic_calendar": ["calendar", "holiday", "semester start", "semester end"],
    "events": ["event", "fest", "workshop", "seminar"],
    "faculty": ["hod", "faculty", "professor", "department head"],
    "transportation": ["bus", "transport", "shuttle"],
    "clubs": ["club", "society", "extracurricular"],
    "grievance": ["grievance", "complaint", "harassment", "issue"],
}


def detect_intent(question: str) -> str:
    q = question.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return intent
    return "general"


def format_sources_text(sources: list[dict]) -> str | None:
    if not sources:
        return None
    parts = []
    for s in sources:
        if s.get("page"):
            parts.append(f"{s['source']} (page {s['page']})")
        else:
            parts.append(s["source"])
    return "; ".join(parts)


@dataclass
class ChatResponse:
    session_id: str
    question: str
    answer: str
    confidence: float
    sources: list[dict] = field(default_factory=list)
    intent: str = "general"
    answered_by: str = "rag"  # "faq" | "rag" | "fallback"
    message_id: int | None = None


class Chatbot:
    """Stateful-per-store, stateless-per-call façade used by the API layer."""

    def __init__(self):
        self.vector_store: VectorStore = get_or_build_vector_store()
        self.faq_matcher = get_faq_matcher()

    def rebuild_index(self) -> int:
        """Admin action: re-scan documents/, re-embed, rebuild FAISS index."""
        count = self.vector_store.build_from_documents()
        logger.info("Vector store rebuilt: %d chunks indexed", count)
        return count

    def refresh_faq(self) -> None:
        self.faq_matcher.refresh()

    def chat(self, question: str, session_id: str | None = None) -> ChatResponse:
        session_id = session_id or new_session_id()
        question = question.strip()
        intent = detect_intent(question)

        # ---- 1. FAQ Mode (exact/semantic cache) ----
        faq_hit = self.faq_matcher.match(question)
        if faq_hit is not None:
            answer = faq_hit["answer"]
            confidence = faq_hit["score"]
            answered_by = "faq"
            sources: list[dict] = []
            logger.info(
                "FAQ %s match (score=%.2f): %r -> %r",
                faq_hit["match_type"], faq_hit["score"], question, faq_hit["matched_question"],
            )
        else:
            # ---- 2. RAG pipeline ----
            history_text = conversation_memory.get_history_text(session_id)
            result: RAGResult = answer_question(question, self.vector_store, history_text=history_text)
            answer = result.answer
            confidence = result.confidence
            sources = result.sources
            answered_by = "fallback" if result.is_fallback else "rag"

        # ---- 3. Update memory + persistence ----
        conversation_memory.add_turn(session_id, question, answer)
        source_text = format_sources_text(sources)
        message_id = save_chat_message(
            session_id=session_id,
            question=question,
            answer=answer,
            confidence=confidence,
            source=source_text,
            intent=intent,
            answered_by=answered_by,
        )

        return ChatResponse(
            session_id=session_id,
            question=question,
            answer=answer,
            confidence=round(confidence, 3),
            sources=sources,
            intent=intent,
            answered_by=answered_by,
            message_id=message_id,
        )


_chatbot_singleton: Chatbot | None = None


def get_chatbot() -> Chatbot:
    """FastAPI dependency: lazily construct a single shared Chatbot instance."""
    global _chatbot_singleton
    if _chatbot_singleton is None:
        _chatbot_singleton = Chatbot()
    return _chatbot_singleton
