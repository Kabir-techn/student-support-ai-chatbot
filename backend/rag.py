"""
backend/rag.py
===============
Retrieval-Augmented Generation pipeline: Retriever -> LLM -> Answer.

Responsibilities:
  - Retrieve top-K relevant chunks from the FAISS vector store
  - Compute a confidence score from retrieval similarity
  - Build a grounded prompt and call the configured LLM provider
    (OpenAI / Ollama / local extractive fallback — pluggable)
  - Attach source citations (filename + page) to the final answer
  - Reduce hallucination by refusing to answer when confidence is low
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.embeddings import VectorStore
from backend.prompt import EXTRACTIVE_FALLBACK_TEMPLATE, build_rag_prompt
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RAGResult:
    answer: str
    confidence: float
    sources: list[dict] = field(default_factory=list)  # [{"source": ..., "page": ...}]
    is_fallback: bool = False


# --------------------------------------------------------------------------
# LLM provider abstraction
# --------------------------------------------------------------------------
class LLMProvider:
    """Base interface. Subclasses implement `generate(prompt) -> str`."""

    def generate(self, prompt: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI  # local import: optional dependency

        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content.strip()


class OllamaProvider(LLMProvider):
    def __init__(self):
        import requests  # local import: only needed for this provider

        self._requests = requests
        self.base_url = settings.OLLAMA_BASE_URL

    def generate(self, prompt: str) -> str:
        resp = self._requests.post(
            f"{self.base_url}/api/generate",
            json={"model": settings.OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=settings.RESPONSE_TIMEOUT_SECONDS * 5,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


class LocalExtractiveProvider(LLMProvider):
    """
    Zero-dependency fallback: no external LLM call. Returns the highest-scoring
    retrieved chunk(s) directly, so the system remains fully functional (and
    hallucination-free, if less fluent) with no API key / no Ollama server.
    """

    def generate(self, prompt: str) -> str:
        # The caller (answer_question) builds a full RAG prompt for real LLMs;
        # for the local provider we instead handle context formatting directly
        # in answer_question(), so this path is only reached if invoked directly.
        return prompt


def get_llm_provider() -> LLMProvider:
    provider = settings.LLM_PROVIDER
    try:
        if provider == "openai":
            return OpenAIProvider()
        if provider == "ollama":
            return OllamaProvider()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falling back to local extractive provider: %s", exc)
    return LocalExtractiveProvider()


# --------------------------------------------------------------------------
# Confidence scoring
# --------------------------------------------------------------------------
def compute_confidence(retrieved_chunks: list[dict]) -> float:
    """
    Simple, explainable confidence heuristic: the top retrieval similarity
    score (cosine similarity, roughly 0..1), optionally damped if very few
    chunks were retrieved. Kept transparent rather than a black-box model,
    per the PRD's "Responsible AI / reduce hallucination" goal.
    """
    if not retrieved_chunks:
        return 0.0
    top_score = max(c["score"] for c in retrieved_chunks)
    # Clamp into [0, 1] since cosine sim on normalized vectors can dip slightly negative
    return max(0.0, min(1.0, top_score))


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        page_str = f", page {c['page']}" if c.get("page") else ""
        blocks.append(f"[Source: {c['source']}{page_str}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def dedupe_sources(chunks: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for c in chunks:
        key = (c["source"], c.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({"source": c["source"], "page": c.get("page")})
    return sources


def answer_question(
    question: str,
    vector_store: VectorStore,
    history_text: str = "",
    top_k: int | None = None,
) -> RAGResult:
    """Run the full RAG pipeline for a single question and return a grounded, cited answer."""

    retrieved = vector_store.search(question, top_k=top_k)
    confidence = compute_confidence(retrieved)

    if confidence < settings.CONFIDENCE_THRESHOLD or not retrieved:
        logger.info("Low confidence (%.2f) for question: %r", confidence, question)
        return RAGResult(
            answer=settings.NOT_CONFIDENT_MESSAGE,
            confidence=confidence,
            sources=[],
            is_fallback=True,
        )

    context = format_context(retrieved)
    sources = dedupe_sources(retrieved)

    if settings.LLM_PROVIDER == "local":
        # No external LLM: return a clearly-labelled extractive answer built
        # directly from the top chunk(s), still grounded and cited.
        best = retrieved[0]
        answer_text = EXTRACTIVE_FALLBACK_TEMPLATE.format(context=best["text"])
    else:
        prompt = build_rag_prompt(history=history_text, context=context, question=question)
        provider = get_llm_provider()
        try:
            answer_text = provider.generate(prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM generation failed (%s); using extractive fallback", exc)
            answer_text = EXTRACTIVE_FALLBACK_TEMPLATE.format(context=retrieved[0]["text"])

    return RAGResult(answer=answer_text, confidence=confidence, sources=sources, is_fallback=False)
