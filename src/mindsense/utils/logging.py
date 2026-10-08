"""Consistent logging for pipeline, training and app code."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "mindsense") -> logging.Logger:
    """Return a configured logger (idempotent, stderr, INFO by default)."""
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root = logging.getLogger("mindsense")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logger
