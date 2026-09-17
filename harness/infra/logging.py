"""Logging setup: structlog when available, stdlib otherwise."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(level: str = "INFO", *, logfile: str | None = None, json: bool = False) -> None:
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))

    try:  # pragma: no cover - optional dependency
        import structlog

        logging.basicConfig(format="%(message)s", level=numeric, handlers=handlers, force=True)
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
            if json
            else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ]
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(numeric),
            logger_factory=structlog.PrintLoggerFactory(sys.stderr),
            cache_logger_on_first_use=True,
        )
    except ImportError:
        logging.basicConfig(
            level=numeric,
            handlers=handlers,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            force=True,
        )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def get_logger(name: str):
    try:  # pragma: no cover - optional dependency
        import structlog

        return structlog.get_logger(name)
    except ImportError:
        return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]
