"""Tests for the persisted eval report.

A saved report is the run artifact a reader trusts later, so these pin that it
is self-describing (carries the config that shaped the metrics), that a skipped
grounding eval is recorded as an explicit null rather than dropped, and that it
round-trips as valid JSON on disk.
"""

from __future__ import annotations

import json

from ray_rag.eval.harness import build_report, write_report


def test_build_report_carries_config_and_metrics():
    rr = {"dense_ndcg": 0.879, "reranked_ndcg": 0.854}
    ic = {"holdout_macro_f1": 0.774, "n_test": 23}
    report = build_report(rr, ic, grounding=None)

    assert report["reranker"] == rr
    assert report["intent"] == ic
    # config context must travel with the numbers — a metric is meaningless
    # without the models, depths, and sets that produced it: the grounding score's
    # llm_model, the intent F1's intents_path, and what the held-out reranker
    # number is held out from (eval_train_path), not just the reranker eval_path.
    assert set(report["config"]) == {
        "embed_model",
        "llm_model",
        "retrieve_top_k",
        "rerank_top_k",
        "eval_train_path",
        "eval_path",
        "intents_path",
    }
    assert "generated_at" in report


def test_skipped_grounding_is_explicit_null_not_dropped():
    report = build_report({}, {}, grounding=None)
    assert "grounding" in report
    assert report["grounding"] is None


def test_write_report_round_trips_valid_json(tmp_path):
    path = str(tmp_path / "nested" / "eval_report.json")  # dir created on write
    report = build_report({"dense_ndcg": 0.9}, {"holdout_macro_f1": 0.8}, grounding=None)
    write_report(report, path)

    with open(path) as f:
        loaded = json.load(f)
    assert loaded == report
