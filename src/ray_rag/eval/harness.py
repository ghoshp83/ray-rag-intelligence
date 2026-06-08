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

import os

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
    for ex in labelled:
        relevant = set(ex["relevant_docs"])
        candidates = index.search(embedder.encode([ex["query"]]), settings.retrieve_top_k)[0]
        if not candidates:
            continue
        dense_ndcg.append(ndcg_at_k(_labels(candidates, relevant), k))
        dense_mrr.append(mrr(_labels(candidates, relevant)))
        dense_recall.append(recall_at_k([c["doc_id"] for c in candidates], relevant, k))
        reranked = reranker.rerank(ex["query"], candidates, len(candidates))
        rr_ndcg.append(ndcg_at_k(_labels(reranked, relevant), k))
        rr_mrr.append(mrr(_labels(reranked, relevant)))
        rr_recall.append(recall_at_k([c["doc_id"] for c in reranked], relevant, k))
    return {
        "dense_ndcg": float(np.mean(dense_ndcg)),
        "reranked_ndcg": float(np.mean(rr_ndcg)),
        "dense_mrr": float(np.mean(dense_mrr)),
        "reranked_mrr": float(np.mean(rr_mrr)),
        "dense_recall": float(np.mean(dense_recall)),
        "reranked_recall": float(np.mean(rr_recall)),
    }


def evaluate_intent(labelled, embedder) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    X = embedder.encode([ex["query"] for ex in labelled])
    y = np.array([ex["intent"] for ex in labelled])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "holdout_macro_f1": float(f1_score(y_te, pred, average="macro")),
        "holdout_accuracy": float(accuracy_score(y_te, pred)),
        "n_test": int(len(y_te)),
    }


def evaluate_grounding(index, embedder, reranker, labelled) -> dict:
    from anthropic import Anthropic

    from ray_rag.serve.generate import generate_answer

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
    log_event("eval", "reranker", **rr)
    ic = evaluate_intent(load_jsonl(settings.intents_path), embedder)
    print(
        f"intent    holdout macro-F1={ic['holdout_macro_f1']:.3f}  "
        f"acc={ic['holdout_accuracy']:.3f}  (n={ic['n_test']})"
    )
    log_event("eval", "intent", **ic)

    if os.environ.get("ANTHROPIC_API_KEY"):
        g = evaluate_grounding(index, embedder, reranker, labelled)
        print(
            f"grounding valid-citation fraction={g['mean_valid_citation_fraction']:.3f}  "
            f"answers-with-citation={g['answers_with_citation']:.3f}  (n={g['n_queries']})"
        )
        log_event("eval", "grounding", **g)
    else:
        print("grounding SKIPPED: ANTHROPIC_API_KEY not set (generation eval needs the LLM API).")
        log_event("eval", "grounding_skipped", reason="ANTHROPIC_API_KEY not set")


if __name__ == "__main__":
    main()
