"""
backend/faq.py
===============
FAQ Mode: if an incoming question matches a curated FAQ entry — either as an
exact string match or via semantic (embedding) similarity — return the cached
answer instantly instead of running the full RAG pipeline. This both speeds
up common queries and guarantees a vetted, consistent answer for them.
"""

from __future__ import annotations

import numpy as np

from backend.database import get_all_faqs, upsert_faq
from backend.embeddings import embed_texts
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


DEFAULT_FAQS: list[dict] = [
    {
        "question": "What is admission process?",
        "answer": (
            "Admissions open every year in June. Steps:\n"
            "1. Fill the online application form.\n"
            "2. Upload required documents (10th/12th mark sheets, ID proof, photo).\n"
            "3. Appear for the entrance test / merit review.\n"
            "4. Pay the admission fee to confirm your seat."
        ),
        "category": "admissions",
    },
    {
        "question": "How can I apply for scholarship?",
        "answer": (
            "Scholarships can be applied for through the Student Portal under "
            "'Financial Aid'. You will need your latest mark sheet, income "
            "certificate, and a filled application form. Applications for the "
            "merit-based scholarship close on 30th September each year."
        ),
        "category": "scholarship",
    },
    {
        "question": "What is hostel fee?",
        "answer": "The hostel fee is ₹60,000 per year, payable in two installments.",
        "category": "hostel",
    },
    {
        "question": "Who is HOD of CSE?",
        "answer": (
            "Please check the Department page on the college website for the "
            "current Head of Department, as this can change each academic year."
        ),
        "category": "faculty",
    },
    {
        "question": "What is library timing?",
        "answer": "The library is open Monday to Saturday, 8:00 AM to 8:00 PM.",
        "category": "library",
    },
    {
        "question": "Is hostel available?",
        "answer": "Yes, separate hostel facilities are available for boys and girls on campus.",
        "category": "hostel",
    },
    {
        "question": "What is minimum attendance?",
        "answer": "A minimum of 75% attendance is required to be eligible to sit for semester exams.",
        "category": "attendance",
    },
    {
        "question": "Can I pay fees online?",
        "answer": (
            "Yes, fees can be paid online through the Student Portal using "
            "net banking, UPI, or debit/credit card."
        ),
        "category": "fees",
    },
]


def seed_default_faqs() -> None:
    """Populate the faq_cache table with default entries if empty (idempotent)."""
    existing = {f.question for f in get_all_faqs()}
    for item in DEFAULT_FAQS:
        if item["question"] not in existing:
            upsert_faq(item["question"], item["answer"], item["category"])
    logger.info("FAQ cache seeded with %d default entries", len(DEFAULT_FAQS))


class FAQMatcher:
    """
    Holds an in-memory embedding index of all FAQ questions so lookups are
    O(1) vector comparisons rather than a DB hit on every message.
    Call `.refresh()` after any FAQ is added/edited via the admin panel.
    """

    def __init__(self):
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.vectors: np.ndarray | None = None
        self.refresh()

    def refresh(self) -> None:
        faqs = get_all_faqs()
        self.questions = [f.question for f in faqs]
        self.answers = [f.answer for f in faqs]
        self.vectors = embed_texts(self.questions) if self.questions else None
        logger.info("FAQMatcher refreshed with %d entries", len(self.questions))

    def match(self, user_question: str) -> dict | None:
        """
        Returns {"answer": str, "matched_question": str, "score": float, "match_type": "exact"|"semantic"}
        or None if nothing crosses the semantic threshold.
        """
        if not self.questions or self.vectors is None:
            return None

        query_vec = embed_texts([user_question])[0]
        scores = self.vectors @ query_vec  # cosine similarity (vectors are normalized)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= settings.FAQ_EXACT_MATCH_THRESHOLD:
            match_type = "exact"
        elif best_score >= settings.FAQ_SEMANTIC_THRESHOLD:
            match_type = "semantic"
        else:
            return None

        return {
            "answer": self.answers[best_idx],
            "matched_question": self.questions[best_idx],
            "score": best_score,
            "match_type": match_type,
        }


_faq_matcher: FAQMatcher | None = None


def get_faq_matcher() -> FAQMatcher:
    global _faq_matcher
    if _faq_matcher is None:
        seed_default_faqs()
        _faq_matcher = FAQMatcher()
    return _faq_matcher
