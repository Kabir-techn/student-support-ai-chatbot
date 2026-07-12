"""
utils/logger.py
================
Provides a single, consistently-configured logger for the whole project.
Every module should call `get_logger(__name__)` instead of using `print()`.

Logs are written both to stdout (for docker/container visibility) and to a
rotating file under /logs so history is preserved between restarts.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from config import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root() -> None:
    """Configure the root logger exactly once."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if reloaded (e.g. Streamlit hot-reload)
    if root.handlers:
        _configured = True
        return

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    try:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # Filesystem may be read-only in some deployment targets; stdout logging still works.
        pass

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, e.g. `get_logger(__name__)`."""
    _configure_root()
    return logging.getLogger(name)
