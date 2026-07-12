"""
backend/database.py
====================
SQLite persistence layer (via SQLAlchemy ORM) for:
  - chat_history: every question/answer/timestamp/confidence exchanged
  - feedback: thumbs up / down ratings per chat message
  - faq_cache: pre-approved question -> answer pairs for instant responses
  - documents: metadata about uploaded knowledge-base files (admin panel)

This is intentionally simple (SQLite) per the PRD; swapping to Postgres later
only requires changing DATABASE_URL in config.py.
"""

from __future__ import annotations

import datetime as dt
import uuid
from contextlib import contextmanager

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _utcnow() -> dt.datetime:
    """Timezone-aware UTC now, converted to naive for consistent SQLite storage/comparison."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class ChatMessage(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    source = Column(String(255), nullable=True)   # e.g. "fee_structure.pdf (p.5)"
    intent = Column(String(64), nullable=True)     # detected intent / category
    answered_by = Column(String(32), nullable=True)  # "faq" | "rag" | "fallback"
    timestamp = Column(DateTime, default=_utcnow, nullable=False)

    feedback = relationship("Feedback", back_populates="message", uselist=False)


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("chat_history.id"), nullable=False)
    is_helpful = Column(Boolean, nullable=False)  # True = 👍, False = 👎
    comment = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_utcnow, nullable=False)

    message = relationship("ChatMessage", back_populates="feedback")


class FAQCache(Base):
    __tablename__ = "faq_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(Text, nullable=False, unique=True)
    answer = Column(Text, nullable=False)
    category = Column(String(64), nullable=True)
    hit_count = Column(Integer, default=0)


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, unique=True)
    uploaded_at = Column(DateTime, default=_utcnow, nullable=False)
    indexed = Column(Boolean, default=False)


# --------------------------------------------------------------------------
# Engine / session management
# --------------------------------------------------------------------------
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI thread pool
    future=True,
)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
)


def init_db() -> None:
    """Create all tables if they don't already exist. Called on app startup."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized at %s", settings.DATABASE_URL)


@contextmanager
def get_session():
    """Context-managed DB session: `with get_session() as db: ...`"""
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def new_session_id() -> str:
    return str(uuid.uuid4())


# --------------------------------------------------------------------------
# CRUD helpers
# --------------------------------------------------------------------------
def save_chat_message(
    session_id: str,
    question: str,
    answer: str,
    confidence: float | None = None,
    source: str | None = None,
    intent: str | None = None,
    answered_by: str | None = None,
) -> int:
    with get_session() as db:
        msg = ChatMessage(
            session_id=session_id,
            question=question,
            answer=answer,
            confidence=confidence,
            source=source,
            intent=intent,
            answered_by=answered_by,
        )
        db.add(msg)
        db.flush()  # populate msg.id before commit
        return msg.id


def get_chat_history(session_id: str, limit: int = 50) -> list[ChatMessage]:
    with get_session() as db:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.timestamp.asc())
            .limit(limit)
            .all()
        )


def record_feedback(message_id: int, is_helpful: bool, comment: str | None = None) -> None:
    with get_session() as db:
        db.add(Feedback(message_id=message_id, is_helpful=is_helpful, comment=comment))


def get_common_questions(limit: int = 10) -> list[tuple[str, int]]:
    """Return the most frequently asked questions (by exact text match) for analytics."""
    with get_session() as db:
        rows = (
            db.query(ChatMessage.question, func.count(ChatMessage.id).label("cnt"))
            .group_by(ChatMessage.question)
            .order_by(func.count(ChatMessage.id).desc())
            .limit(limit)
            .all()
        )
        return [(q, c) for q, c in rows]


def get_analytics_summary() -> dict:
    with get_session() as db:
        total_messages = db.query(func.count(ChatMessage.id)).scalar() or 0
        total_sessions = db.query(func.count(func.distinct(ChatMessage.session_id))).scalar() or 0
        helpful = db.query(func.count(Feedback.id)).filter(Feedback.is_helpful.is_(True)).scalar() or 0
        not_helpful = db.query(func.count(Feedback.id)).filter(Feedback.is_helpful.is_(False)).scalar() or 0
        avg_confidence = db.query(func.avg(ChatMessage.confidence)).scalar()
        return {
            "total_messages": total_messages,
            "total_sessions": total_sessions,
            "helpful_feedback": helpful,
            "not_helpful_feedback": not_helpful,
            "average_confidence": round(avg_confidence, 3) if avg_confidence else None,
        }


def upsert_faq(question: str, answer: str, category: str | None = None) -> None:
    with get_session() as db:
        existing = db.query(FAQCache).filter(FAQCache.question == question).first()
        if existing:
            existing.answer = answer
            existing.category = category
        else:
            db.add(FAQCache(question=question, answer=answer, category=category))


def get_all_faqs() -> list[FAQCache]:
    with get_session() as db:
        return db.query(FAQCache).all()


def register_document(filename: str, indexed: bool = False) -> None:
    with get_session() as db:
        existing = db.query(DocumentRecord).filter(DocumentRecord.filename == filename).first()
        if existing:
            existing.indexed = indexed
        else:
            db.add(DocumentRecord(filename=filename, indexed=indexed))


def list_documents() -> list[DocumentRecord]:
    with get_session() as db:
        return db.query(DocumentRecord).all()


def delete_document_record(filename: str) -> None:
    with get_session() as db:
        db.query(DocumentRecord).filter(DocumentRecord.filename == filename).delete()
