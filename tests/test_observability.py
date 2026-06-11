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
    # `ts` is reserved but, unlike component/event/level, is not a named param,
    # so a caller *can* pass it as a field — the formatter must drop it rather
    # than let it corrupt the schema spine.
    rec = _capture_event("eval", "reranker", ts="hacked", dense_ndcg=0.9)
    assert rec["event"] == "reranker"
    assert rec["ts"] != "hacked"
    assert rec["dense_ndcg"] == 0.9


def test_explicit_level_sets_the_spine():
    # A non-routine event (e.g. a skipped eval step) can be raised to WARNING so
    # it stands out; the schema's `level` column then reflects it.
    rec = _capture_event("eval", "grounding_skipped", level="WARNING", reason="no key")
    assert rec["level"] == "WARNING"
    assert rec["reason"] == "no key"
