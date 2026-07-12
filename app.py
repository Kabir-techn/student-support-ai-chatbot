"""
app.py
======
FastAPI application entry point for the AI Student Support Services Chatbot.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Docs:
    http://localhost:8000/docs
"""

from __future__ import annotations

import time

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routes import admin_router, chat_router
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting %s v%s (env=%s)", settings.APP_NAME, settings.APP_VERSION, settings.ENV)
    init_db()
    # NOTE: the vector store / chatbot singleton is built lazily on first request
    # (see backend.chatbot.get_chatbot) so app startup itself stays fast.
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready RAG-powered chatbot for student support services.",
    lifespan=lifespan,
)

# CORS: allow the Streamlit frontend (and other local dev tools) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Attach response latency to headers and log slow (>3s) responses per the NFR."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    response.headers["X-Process-Time"] = f"{duration:.3f}"
    if duration > settings.RESPONSE_TIMEOUT_SECONDS:
        logger.warning("Slow response (%.2fs) for %s %s", duration, request.method, request.url.path)
    return response


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}


app.include_router(chat_router)
app.include_router(admin_router)
