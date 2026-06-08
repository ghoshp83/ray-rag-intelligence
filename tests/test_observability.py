"""Tests for the structured run-logging schema.

The schema is a contract other tools (jq, log scrapers) parse, so these pin the
parts that would silently break a reader: every line is valid JSON, the four
reserved fields are always present, and an event field can never overwrite one
of them (a caller passing `ts=...` must not blank out the timestamp).

We capture by attaching a temporary StringIO handler (with the real formatter)
to the `ray_rag` logger rather than via capsys/capfd: the module's stdout
handler binds to sys.stdout once, which pytest's capture fixtures swap per test,
so a StringIO handler is the robust way to read exactly what was emitted.
"""

from __future__ import annotations

import io
import json
import logging

from ray_rag.observability import _JsonFormatter, log_event


def _capture_event(component: str, event: str, **fields) -> dict:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    logger = logging.getLogger("ray_rag")
    logger.addHandler(handler)
    try:
        log_event(component, event, **fields)
    finally:
        logger.removeHandler(handler)
    return json.loads(buf.getvalue().strip().splitlines()[-1])


def test_log_event_emits_valid_json_with_reserved_spine():
    rec = _capture_event("serve", "ask", intent="factual", latency_ms=12.5, n_sources=5)
    assert rec["component"] == "serve"
    assert rec["event"] == "ask"
    assert rec["level"] == "INFO"
    assert "ts" in rec
    # event-specific fields are merged in
    assert rec["intent"] == "factual"
    assert rec["latency_ms"] == 12.5
    assert rec["n_sources"] == 5


def test_event_fields_cannot_shadow_reserved_keys():
    # `ts` and `level` are reserved but, unlike component/event, are not named
    # params, so a caller *can* pass them as fields — the formatter must drop
    # them rather than let them corrupt the schema spine.
    rec = _capture_event("eval", "reranker", level="DEBUG", ts="hacked", dense_ndcg=0.9)
    assert rec["event"] == "reranker"
    assert rec["level"] == "INFO"
    assert rec["ts"] != "hacked"
    assert rec["dense_ndcg"] == 0.9
