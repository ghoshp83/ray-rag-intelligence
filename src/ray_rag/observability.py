"""Structured run logging — one JSON object per line, with a documented schema.

Every significant step (an eval run, each served request) emits a single JSON
line to stdout so runs are greppable, machine-readable, and survive in the Ray
log capture without a logging backend to stand up. The schema is small and
fixed so a reader (or `jq`) can rely on it:

| field       | always | meaning                                              |
|-------------|--------|------------------------------------------------------|
| `ts`        | yes    | ISO-8601 UTC timestamp of the event                  |
| `level`     | yes    | log level (`INFO`, `WARNING`, ...)                   |
| `component` | yes    | subsystem: `eval`, `serve`, `ingest`, `train`        |
| `event`     | yes    | event name within the component (e.g. `reranker`)    |
| *(rest)*    | no     | event-specific fields — metrics, `latency_ms`, etc.  |

Reserved field names (`ts`, `level`, `component`, `event`) can't be overridden
by event fields — that keeps the schema's spine stable no matter what a caller
passes, so downstream parsing never breaks on a key collision.

Example line:

    {"ts": "2026-06-08T10:00:00+00:00", "level": "INFO", "component": "serve",
     "event": "ask", "intent": "factual", "latency_ms": 1231.6, "n_sources": 5}
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

_RESERVED = ("ts", "level", "component", "event")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", "app"),
            "event": record.getMessage(),
        }
        # Event fields are merged but never shadow the reserved spine above.
        for key, value in getattr(record, "fields", {}).items():
            if key not in _RESERVED:
                payload[key] = value
        return json.dumps(payload)


_configured = False


def _logger() -> logging.Logger:
    global _configured
    logger = logging.getLogger("ray_rag")
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False  # don't double-log through the root handler
        _configured = True
    return logger


def log_event(component: str, event: str, **fields: Any) -> None:
    """Emit one structured JSON log line. See the module docstring for the schema."""
    _logger().info(event, extra={"component": component, "fields": fields})
