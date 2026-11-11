"""Logging utilities."""

from __future__ import annotations

import logging
import sys
import time

__all__ = ["ProgressReporter", "get_logger"]

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


class ProgressReporter:
    """Log periodic progress with elapsed time and ETA."""

    def __init__(
        self,
        log: logging.Logger,
        label: str,
        total: int,
        every: int = 0,
    ) -> None:
        """Initialize the reporter.

        Args:
            log: Logger to write to.
            label: Prefix for progress lines, usually the model name.
            total: Total number of items.
            every: Log every N items; ``0`` picks about 20 updates.
        """
        self._log = log
        self._label = label
        self._total = max(1, total)
        self._every = every if every > 0 else max(1, min(500, total // 20))
        self._start = time.perf_counter()

    def update(self, done: int) -> None:
        """Emit a progress line when due.

        Args:
            done: Number of completed items.
        """
        if done % self._every and done < self._total:
            return
        elapsed = time.perf_counter() - self._start
        rate = elapsed / max(1, done)
        remaining = rate * max(0, self._total - done)
        self._log.info(
            "%s %d/%d (%.1f%%) | elapsed %s | ETA %s",
            self._label,
            done,
            self._total,
            100.0 * done / self._total,
            _format_seconds(elapsed),
            _format_seconds(remaining),
        )


def _format_seconds(seconds: float) -> str:
    """Format a duration compactly.

    Args:
        seconds: Duration in seconds.

    Returns:
        A string such as ``45s``, ``12m03s``, or ``1h07m``.
    """
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"
