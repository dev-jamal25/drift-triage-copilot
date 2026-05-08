"""structlog configuration for the worker service.

Mirrors ``platform/app/core/logging.py`` so log output across services is
shaped identically (JSON, ISO-UTC timestamps, contextvars merged in).
"""

from __future__ import annotations

import logging

import structlog

_LEVEL_NAMES: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with JSON output and the requested log level.

    Unknown ``level`` strings fall back to INFO rather than raising — matches
    platform behaviour and avoids a startup crash on a typo.
    """
    numeric_level = _LEVEL_NAMES.get(level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )
