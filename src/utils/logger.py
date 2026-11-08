"""Logging utilities."""

from __future__ import annotations

import logging
import sys

__all__ = ["get_logger"]

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Return a logger with a single stderr handler.

    Args:
        name: Logger name, typically ``__name__``.
        level: Logging level as an integer or a level name such as ``"DEBUG"``.

    Returns:
        A configured logger. Repeated calls with the same name reuse the same
        logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATE_FORMAT)
        )
        logger.addHandler(handler)
        logger.propagate = False

    return logger
