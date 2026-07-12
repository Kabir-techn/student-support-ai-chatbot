"""
utils/build_index.py
=====================
One-off / cron-friendly CLI to (re)build the FAISS vector index from
whatever is currently in `documents/`, and initialize the SQLite database
with tables + seeded default FAQs.

Usage:
    python -m utils.build_index
"""

from __future__ import annotations

from backend.database import init_db, register_document
from backend.embeddings import VectorStore
from backend.faq import seed_default_faqs
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    logger.info("Initializing database...")
    init_db()

    logger.info("Seeding default FAQ cache...")
    seed_default_faqs()

    logger.info("Building FAISS vector index from %s ...", settings.DOCUMENTS_DIR)
    store = VectorStore()
    count = store.build_from_documents()

    for path in settings.DOCUMENTS_DIR.iterdir():
        if path.is_file():
            register_document(path.name, indexed=True)

    logger.info("Done. Indexed %d chunks from %s.", count, settings.DOCUMENTS_DIR)


if __name__ == "__main__":
    main()
