"""Structured JSON logging.

One JSON object per line on stdout, easy to parse by Render/CloudWatch and
trivially greppable during experiments. Falls back to a colourised pretty
form in development for readability.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Production formatter: one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Attach any extras the caller passed via `extra={...}`.
        for key, value in record.__dict__.items():
            if key in _RESERVED:
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


class ColorFormatter(logging.Formatter):
    """Dev formatter: short, coloured, single-line."""

    COLORS = {
        "DEBUG": "\x1b[36m",
        "INFO": "\x1b[32m",
        "WARNING": "\x1b[33m",
        "ERROR": "\x1b[31m",
        "CRITICAL": "\x1b[35m",
    }
    RESET = "\x1b[0m"
    DIM = "\x1b[2m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        time = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED
        }
        tail = f" {self.DIM}{json.dumps(extras, default=str)}{self.RESET}" if extras else ""
        return (
            f"{self.DIM}{time}{self.RESET} "
            f"{color}{record.levelname:<5}{self.RESET} "
            f"{record.name}: {record.getMessage()}{tail}"
        )


_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
}


def configure_logging(level: str = "info", *, dev: bool = False) -> None:
    """Install our handler on the root logger and quiet down noisy libs."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(ColorFormatter() if dev else JsonFormatter())
    root.addHandler(handler)

    # Tame uvicorn's default access log; we'll log via middleware ourselves.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = [handler]
        logging.getLogger(name).propagate = False
