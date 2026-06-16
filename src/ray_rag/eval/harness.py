"""Evaluation harness: the quantitative verification for the whole system.

Reports (1) retrieval recall@k plus reranker nDCG@k / MRR vs dense-only retrieval
on a held-out test set (disjoint from the queries the ranker trained on) — recall
shows whether retrieval surfaced the right docs at all, nDCG/MRR whether the ranker
ordered them better; (2) a held-out intent macro-F1
(re-fit on a train split so the figure is honest generalisation, not training
accuracy); (3) citation-grounding faithfulness over in-scope queries — skipped
unless ANTHROPIC_API_KEY is set, and that skip is stated, never silent.

Run: `python -m ray_rag.eval.harness`
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from ray_rag.config import settings
from ray_rag.data.embed import Embedder
from ray_rag.data.index import VectorIndex
from ray_rag.eval.grounding import grounding_score
from ray_rag.eval.metrics import mrr, ndcg_at_k, recall_at_k
from ray_rag.models.reranker import Reranker
from ray_rag.models.train import load_jsonl
from ray_rag.observability import log_event


def _labels(candidates: list[dict], relevant: set[str]) -> list[float]:
    return [float(c["doc_id"] in relevant) for c in candidates]


def evaluate_reranker(index, embedder, reranker, labelled, k) -> dict:
    dense_ndcg, dense_mrr, dense_recall = [], [], []
    rr_ndcg, rr_mrr, rr_recall = [], [], []
    per_query = []
    for ex in labelled:
        relevant = set(ex["relevant_docs"])
        candidates = index.search(embedder.encode([ex["query"]]), settings.retrieve_top_k)[0]
        if not candidates:
            continue
        dense_lbls = _labels(candidates, relevant)
        d_ndcg = ndcg_at_k(dense_lbls, k)
        dense_ndcg.append(d_ndcg)
        dense_mrr.append(mrr(dense_lbls))
        dense_recall.append(recall_at_k([c["doc_id"] for c in candidates], relevant, k))
        reranked = reranker.rerank(ex["query"], candidates, len(candidates))
        rr_lbls = _labels(reranked, relevant)
        r_ndcg = ndcg_at_k(rr_lbls, k)
        rr_ndcg.append(r_ndcg)
        rr_mrr.append(mrr(rr_lbls))
        rr_recall.append(recall_at_k([c["doc_id"] for c in reranked], relevant, k))
        per_query.append(
            {
                "query": ex["query"],
                "dense_ndcg": d_ndcg,
                "reranked_ndcg": r_ndcg,
                "delta": r_ndcg - d_ndcg,
            }
        )
    if not per_query:
        # Every labelled query retrieved zero candidates — an empty/misconfigured
        # index, not a real score. Without this guard np.mean([]) yields nan, the
        # printed headline reads "nan -> nan", and json.dump writes a bare `NaN`
        # token (invalid JSON) into the report. Fail loud (Rule 10), like
        # train_reranker's "need >=2 ... queries" guard.
        raise ValueError(
            "no labelled query produced retrieval candidates — is the index empty or misbuilt?"
        )
    return {
        "dense_ndcg": float(np.mean(dense_ndcg)),
        "reranked_ndcg": float(np.mean(rr_ndcg)),
        "dense_mrr": float(np.mean(dense_mrr)),
        "reranked_mrr": float(np.mean(rr_mrr)),
        "dense_recall": float(np.mean(dense_recall)),
        "reranked_recall": float(np.mean(rr_recall)),
        **uplift_summary(per_query),
        "per_query": per_query,
    }


def uplift_summary(per_query: list[dict], tol: float = 1e-9) -> dict:
    """Count where reranking moved nDCG up, down, or not at all.

    The headline dense->reranked averages can read as flat "parity" while hiding
    that the reranker helped some queries and hurt others in equal measure. These
    counts make that visible: a reranker worth keeping should improve more queries
    than it regresses, not just match dense on average.
    """
    improved = sum(1 for r in per_query if r["delta"] > tol)
    regressed = sum(1 for r in per_query if r["delta"] < -tol)
    return {
        "n_improved": improved,
        "n_regressed": regressed,
        "n_tied": len(per_query) - improved - regressed,
    }


def evaluate_intent(labelled, embedder, clf) -> dict:
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    X = embedder.encode([ex["query"] for ex in labelled])
    y = np.array([ex["intent"] for ex in labelled])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    # Re-fit the *deployed* classifier's tuned hyperparameters (clone copies the
    # params but not the fitted state) on the train split, so the held-out figure
    # measures the model we actually ship rather than an untuned default — and
    # without leaking the test rows the saved model was fit on.
    model = clone(clf).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    labels = sorted(set(y))
    per_class = f1_score(y_te, pred, average=None, labels=labels, zero_division=0)
    return {
        "holdout_macro_f1": float(f1_score(y_te, pred, average="macro")),
        "holdout_accuracy": float(accuracy_score(y_te, pred)),
        # Per-intent F1 so a maintainer can see *which* route is weak — macro-F1
        # alone can't say where to add labelled examples (RUNBOOK leans on this).
        "per_class_f1": {name: float(f) for name, f in zip(labels, per_class, strict=True)},
        "n_test": int(len(y_te)),
    }


def evaluate_grounding(index, embedder, reranker, labelled, client=None) -> dict:
    from ray_rag.serve.generate import generate_answer

    # The Anthropic client is injectable (defaulted, like generate_answer's own
    # `client` arg) so this loop — retrieve, rerank, generate, score the answer
    # against the *reranked* ids — is unit-testable with a fake client, key-free.
    if client is None:
        from anthropic import Anthropic

        client = Anthropic()
    scores = []
    for ex in labelled:
        candidates = index.search(embedder.encode([ex["query"]]), settings.retrieve_top_k)[0]
        passages = reranker.rerank(ex["query"], candidates, settings.rerank_top_k)
        answer = generate_answer(client, settings.llm_model, ex["query"], passages)
        scores.append(grounding_score(answer, [p["chunk_id"] for p in passages]))
    return {
        "mean_valid_citation_fraction": float(np.mean([s["valid_fraction"] for s in scores])),
        "answers_with_citation": float(np.mean([s["has_citation"] for s in scores])),
        "n_queries": len(scores),
    }


def build_report(reranker: dict, intent: dict, grounding: dict | None) -> dict:
    """Assemble the persisted eval report: metrics plus the context to read them in.

    Carries the config that shaped the numbers (models, retrieval depths, which
    sets) so a saved report is self-describing — a metric is meaningless without
    the corpus, model, and depths that produced it. That includes `llm_model`
    (which model produced the grounding score), `intents_path` (the set behind the
    intent F1), and `eval_train_path` (what the held-out reranker number is held
    out *from*) — not just the reranker's `eval_path`. `grounding` is None when the
    LLM eval was skipped, recorded explicitly rather than omitted (never silent).
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "embed_model": settings.embed_model,
            "llm_model": settings.llm_model,
            "retrieve_top_k": settings.retrieve_top_k,
            "rerank_top_k": settings.rerank_top_k,
            "eval_train_path": settings.eval_train_path,
            "eval_path": settings.eval_path,
            "intents_path": settings.intents_path,
        },
        "reranker": reranker,
        "intent": intent,
        "grounding": grounding,
    }


def write_report(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def main() -> None:
    embedder = Embedder(settings.embed_model)
    index = VectorIndex.load(settings.index_path)
    reranker = Reranker.load(settings.reranker_path)
    labelled = load_jsonl(settings.eval_path)

    rr = evaluate_reranker(index, embedder, reranker, labelled, settings.rerank_top_k)
    print(
        f"reranker  nDCG@{settings.rerank_top_k}: dense={rr['dense_ndcg']:.3f} -> "
        f"reranked={rr['reranked_ndcg']:.3f}  "
        f"(MRR {rr['dense_mrr']:.3f} -> {rr['reranked_mrr']:.3f})"
    )
    print(
        f"retrieval recall@{settings.rerank_top_k}: dense={rr['dense_recall']:.3f} -> "
        f"reranked={rr['reranked_recall']:.3f}"
    )
    print(
        f"reranker uplift: improved={rr['n_improved']} regressed={rr['n_regressed']} "
        f"tied={rr['n_tied']} (n={len(rr['per_query'])})"
    )
    movers = sorted(rr["per_query"], key=lambda r: r["delta"])
    if movers and movers[0]["delta"] < 0:
        print(f"  worst regression (Δ{movers[0]['delta']:+.3f}): {movers[0]['query']!r}")
    if movers and movers[-1]["delta"] > 0:
        print(f"  best gain        (Δ{movers[-1]['delta']:+.3f}): {movers[-1]['query']!r}")
    # Persist the summary counts; drop the verbose per-query rows from the logged event.
    log_event("eval", "reranker", **{k: v for k, v in rr.items() if k != "per_query"})
    import joblib

    intent_clf = joblib.load(settings.intent_path)
    ic = evaluate_intent(load_jsonl(settings.intents_path), embedder, intent_clf)
    print(
        f"intent    holdout macro-F1={ic['holdout_macro_f1']:.3f}  "
        f"acc={ic['holdout_accuracy']:.3f}  (n={ic['n_test']})"
    )
    print(
        "  per-intent F1: "
        + "  ".join(f"{name}={f1:.3f}" for name, f1 in ic["per_class_f1"].items())
    )
    log_event("eval", "intent", **ic)

    g: dict | None = None
    if os.environ.get("ANTHROPIC_API_KEY"):
        g = evaluate_grounding(index, embedder, reranker, labelled)
        print(
            f"grounding valid-citation fraction={g['mean_valid_citation_fraction']:.3f}  "
            f"answers-with-citation={g['answers_with_citation']:.3f}  (n={g['n_queries']})"
        )
        log_event("eval", "grounding", **g)
    else:
        print("grounding SKIPPED: ANTHROPIC_API_KEY not set (generation eval needs the LLM API).")
        log_event("eval", "grounding_skipped", level="WARNING", reason="ANTHROPIC_API_KEY not set")

    write_report(build_report(rr, ic, g), settings.eval_report_path)
    print(f"wrote eval report -> {settings.eval_report_path}")


if __name__ == "__main__":
    main()
