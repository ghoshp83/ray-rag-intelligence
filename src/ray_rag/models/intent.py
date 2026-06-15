"""Query intent/routing classifier: sentence-embedding features -> logistic regression.

Routing is a prediction task, so a trained model owns it (not the LLM). The
class label decides how the pipeline treats the query — e.g. an `out_of_scope`
query is refused before any generation, which is a guardrail, so misrouting is a
correctness issue measured by macro-F1. Ray Tune searches the regularisation
strength in parallel; the model itself is small and CPU-instant.
"""

from __future__ import annotations

import numpy as np

# The label that triggers the Serve guardrail (refuse before any retrieval or
# generation). Named once here, the classifier's vocabulary, so the serve layer
# can import it instead of hard-coding the string — a rename then can't silently
# desync the refusal check from what the classifier actually emits.
OUT_OF_SCOPE = "out_of_scope"
INTENTS = ("factual", "summarize", OUT_OF_SCOPE)


class IntentClassifier:
    def __init__(self, clf, embedder):
        self._clf = clf
        self._embedder = embedder

    @classmethod
    def load(cls, path: str, embedder) -> IntentClassifier:
        import joblib

        return cls(joblib.load(path), embedder)

    def predict(self, query: str) -> tuple[str, float]:
        """Return (predicted_intent, confidence)."""
        emb = self._embedder.encode([query])
        proba = self._clf.predict_proba(emb)[0]
        idx = int(np.argmax(proba))
        return str(self._clf.classes_[idx]), float(proba[idx])


def _make_clf(config: dict):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=config["C"],
        class_weight=config["class_weight"],
        max_iter=1000,
    )


def train_intent(
    labelled: list[dict],
    embedder,
    out_path: str,
    num_samples: int = 10,
    cv: int = 3,
) -> dict:
    """Tune logistic-regression hyperparameters by cross-validated macro-F1; save best."""
    import joblib
    import ray
    from ray import tune
    from sklearn.model_selection import cross_val_score

    queries = [ex["query"] for ex in labelled]
    y = np.array([ex["intent"] for ex in labelled])
    X = embedder.encode(queries)
    data_ref = ray.put((X, y))

    def trainable(config: dict) -> None:
        feats, labels = ray.get(data_ref)
        n_splits = min(cv, int(np.min(np.bincount(_codes(labels)))))
        score = cross_val_score(
            _make_clf(config), feats, labels, cv=max(2, n_splits), scoring="f1_macro"
        ).mean()
        ray.train.report({"f1": float(score)})

    param_space = {
        "C": tune.loguniform(0.01, 100.0),
        "class_weight": tune.choice([None, "balanced"]),
    }
    tuner = tune.Tuner(
        trainable,
        param_space=param_space,
        tune_config=tune.TuneConfig(num_samples=num_samples, metric="f1", mode="max"),
    )
    best = tuner.fit().get_best_result(metric="f1", mode="max")
    best_config, best_metrics = best.config, best.metrics
    if best_config is None or best_metrics is None:
        raise RuntimeError("no successful intent tuning trial")

    clf = _make_clf(best_config).fit(X, y)
    joblib.dump(clf, out_path)
    return {"config": best_config, "cv_f1": best_metrics["f1"]}


def _codes(labels: np.ndarray) -> np.ndarray:
    _, codes = np.unique(labels, return_inverse=True)
    return codes
