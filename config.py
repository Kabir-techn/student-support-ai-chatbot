"""
config.py
==========
Centralized configuration for the AI Student Support Services Chatbot.

All tunable parameters (paths, model names, thresholds, API keys) are defined
here and loaded from environment variables / a `.env` file, so nothing is
hard-coded inside business logic. This keeps the app 12-factor friendly and
easy to deploy across dev / docker / production environments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------
# Base paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
DATABASE_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for _dir in (DOCUMENTS_DIR, VECTORSTORE_DIR, DATABASE_DIR, MODELS_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application-wide settings, overridable via environment variables or .env"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---------------- Filesystem paths (exposed on the settings object too,
    # so modules can simply do `settings.DOCUMENTS_DIR` etc.) ----------------
    DOCUMENTS_DIR: Path = DOCUMENTS_DIR
    VECTORSTORE_DIR: Path = VECTORSTORE_DIR
    DATABASE_DIR: Path = DATABASE_DIR
    MODELS_DIR: Path = MODELS_DIR
    LOGS_DIR: Path = LOGS_DIR

    # ---------------- App metadata ----------------
    APP_NAME: str = "AI Student Support Services Chatbot"
    APP_VERSION: str = "1.0.0"
    ENV: Literal["dev", "staging", "prod"] = "dev"
    DEBUG: bool = True

    # ---------------- Server ----------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501

    # ---------------- LLM Provider ----------------
    # "openai" | "ollama" | "local" (local = extractive fallback, no external calls)
    LLM_PROVIDER: Literal["openai", "ollama", "local"] = Field(
        default_factory=lambda: "openai" if os.getenv("OPENAI_API_KEY") else "local"
    )
    OPENAI_API_KEY: str | None = Field(default=None)
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.1"
    OLLAMA_TIMEOUT_SECONDS: int = 90  # local inference is much slower than cloud APIs, especially on CPU
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 512

    # ---------------- Embeddings ----------------
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ---------------- RAG / Retrieval ----------------
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 60
    TOP_K: int = 4
    CONFIDENCE_THRESHOLD: float = 0.45  # below this -> "not confident" fallback
    FAISS_INDEX_NAME: str = "college_index"

    # ---------------- FAQ ----------------
    FAQ_EXACT_MATCH_THRESHOLD: float = 0.97  # cosine similarity for "exact" cached match
    FAQ_SEMANTIC_THRESHOLD: float = 0.80

    # ---------------- Memory ----------------
    MAX_HISTORY_TURNS: int = 8  # number of prior turns kept in the LLM context window

    # ---------------- Database ----------------
    DATABASE_URL: str = f"sqlite:///{DATABASE_DIR / 'student.db'}"

    # ---------------- Response behaviour ----------------
    RESPONSE_TIMEOUT_SECONDS: int = 3
    NOT_CONFIDENT_MESSAGE: str = (
        "I couldn't find anything about that in the college's documents, so I don't "
        "want to guess. I can help with admissions, fees, scholarships, hostel, "
        "library, placements, exams, attendance, academic calendar, and similar "
        "student support topics — try rephrasing, or contact Student Support at "
        "support@college.edu for anything else."
    )

    # ---------------- Logging ----------------
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = LOGS_DIR / "app.log"


settings = Settings()
