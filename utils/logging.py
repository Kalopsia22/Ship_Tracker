"""
Centralized logging configuration for the application.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with console + file handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (avoids duplicate handlers on Streamlit reruns)
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(LOG_DIR / "app.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        # File system may be read-only in some deployment environments
        pass

    logger.propagate = False
    return logger
