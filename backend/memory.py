"""
backend/memory.py
==================
In-process conversation memory, keyed by session_id, so the chatbot can
resolve follow-up questions like "does that include mess?" after
"What is hostel fee?".

For a single-process deployment (FastAPI + Streamlit talking to one backend)
an in-memory store keeps things simple and fast. Full transcripts are still
persisted to SQLite (see database.py) for history/analytics; this module only
holds the *working* short-term memory used to build LLM context.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Turn:
    question: str
    answer: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """Thread-safe, in-memory store of recent turns per session."""

    def __init__(self, max_turns: int | None = None):
        self.max_turns = max_turns or settings.MAX_HISTORY_TURNS
        self._sessions: dict[str, list[Turn]] = {}
        self._lock = threading.Lock()

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append(Turn(question=question, answer=answer))
            # Keep only the most recent N turns to bound prompt size
            if len(turns) > self.max_turns:
                self._sessions[session_id] = turns[-self.max_turns :]

    def get_history(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_history_text(self, session_id: str) -> str:
        """Render history as plain text for prompt injection."""
        turns = self.get_history(session_id)
        if not turns:
            return ""
        lines = []
        for t in turns:
            lines.append(f"Student: {t.question}")
            lines.append(f"Assistant: {t.answer}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)


# Module-level singleton shared across the FastAPI app (single-process deployment)
conversation_memory = ConversationMemory()
