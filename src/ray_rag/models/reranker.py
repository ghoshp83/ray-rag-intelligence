"""Learned-to-rank reranker: an XGBoost `rank:ndcg` model we train ourselves.

Retrieval favours recall; this model owns *precision at the top* — the ordering
the LLM actually reads. It is trained on labelled relevance and tuned with Ray
Tune to maximise validation nDCG, so its quality is a number we move, not a
black box. Training and inference share `eval.metrics.ndcg_at_k` and the same
`FeatureExtractor`, so the objective and the serving behaviour cannot drift.
"""

from __future__ import annotations

import numpy as np

from ray_rag.eval.metrics import ndcg_at_k
from ray_rag.models.features import FEATURE_NAMES, FeatureExtractor


def _mean_ndcg(preds: np.ndarray, labels: np.ndarray, groups: list[int], k: int) -> float:
    ndcgs, start = [], 0
    for g in groups:
        order = np.argsort(-preds[start : start + g])
        ranked = [float(labels[start : start + g][i]) for i in order]
        ndcgs.append(ndcg_at_k(ranked, k))
        start += g
    return float(np.mean(ndcgs)) if ndcgs else 0.0


class Reranker:
    """Inference wrapper: reorder retrieved candidates by learned relevance."""

    def __init__(self, booster, feature_extractor: FeatureExtractor):
        self._booster = booster
        self._features = feature_extractor

    @classmethod
    def load(cls, path: str) -> Reranker:
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(path)
        return cls(booster, FeatureExtractor())

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        import xgboost as xgb

        if not candidates:
            return []
        feats = self._features.features(query, candidates)
        scores = self._booster.predict(xgb.DMatrix(feats, feature_names=FEATURE_NAMES))
        ranked = sorted(
            ({**c, "rerank_score": float(s)} for c, s in zip(candidates, scores, strict=True)),
            key=lambda c: c["rerank_score"],
            reverse=True,
        )
        return ranked[:top_k]


# --- Training (Ray Tune drives parallel HPO) -------------------------------


def build_ranking_examples(
    index, embedder, feature_extractor: FeatureExtractor, labelled: list[dict], retrieve_n: int
) -> list[tuple[np.ndarray, list[int]]]:
    """Per labelled query: retrieve candidates, featurise, label by source doc."""
    examples = []
    for ex in labelled:
        query, relevant = ex["query"], set(ex["relevant_docs"])
        candidates = index.search(embedder.encode([query]), retrieve_n)[0]
        if not candidates:
            continue
        feats = feature_extractor.features(query, candidates)
        labels = [int(c["doc_id"] in relevant) for c in candidates]
        examples.append((feats, labels))
    return examples


def _assemble(examples: list[tuple[np.ndarray, list[int]]]):
    X = np.vstack([f for f, _ in examples])
    y = np.array([lbl for _, labels in examples for lbl in labels])
    groups = [len(labels) for _, labels in examples]
    return X, y, groups


def _train_booster(X, y, groups, config: dict):
    import xgboost as xgb

    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    dtrain.set_group(groups)
    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@5",
        "eta": config["eta"],
        "max_depth": config["max_depth"],
        "min_child_weight": config["min_child_weight"],
    }
    return xgb.train(params, dtrain, num_boost_round=config["num_boost_round"])


def train_reranker(
    index,
    embedder,
    labelled: list[dict],
    out_path: str,
    retrieve_n: int = 30,
    num_samples: int = 12,
    val_frac: float = 0.3,
    k: int = 5,
) -> dict:
    """Tune XGBoost ranking hyperparameters on a query-level split; save the best.

    Returns the best config + its validation nDCG so callers can log/verify.
    """
    import ray
    from ray import tune

    extractor = FeatureExtractor()
    examples = build_ranking_examples(index, embedder, extractor, labelled, retrieve_n)
    if len(examples) < 2:
        raise ValueError("need >=2 labelled queries with candidates to train+validate")

    n_val = max(1, int(len(examples) * val_frac))
    val_ex, train_ex = examples[:n_val], examples[n_val:]
    Xtr, ytr, gtr = _assemble(train_ex)
    Xv, yv, gv = _assemble(val_ex)
    train_ref = ray.put((Xtr, ytr, gtr, Xv, yv, gv))

    def trainable(config: dict) -> None:
        xtr, y_tr, g_tr, xv, y_v, g_v = ray.get(train_ref)
        booster = _train_booster(xtr, y_tr, g_tr, config)
        import xgboost as xgb

        preds = booster.predict(xgb.DMatrix(xv, feature_names=FEATURE_NAMES))
        ray.train.report({"ndcg": _mean_ndcg(preds, y_v, g_v, k)})

    param_space = {
        "eta": tune.loguniform(0.01, 0.3),
        "max_depth": tune.randint(2, 7),
        "min_child_weight": tune.uniform(0.5, 5.0),
        "num_boost_round": tune.choice([50, 100, 200]),
    }
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, metric="ndcg", mode="max"),
    )
    best = tuner.fit().get_best_result(metric="ndcg", mode="max")
    best_config, best_metrics = best.config, best.metrics
    if best_config is None or best_metrics is None:
        raise RuntimeError("no successful reranker tuning trial")

    # Refit the winning config on all labelled data and persist.
    X, y, groups = _assemble(examples)
    _train_booster(X, y, groups, best_config).save_model(out_path)
    return {"config": best_config, "val_ndcg": best_metrics["ndcg"]}
