"""Intent-classifier prediction: routing is a guardrail, so the predicted label
must map through the classifier's own `classes_` (sklearn sorts them, which is
*not* the INTENTS declaration order) — never a positional guess. A wrong mapping
would route a query to the wrong branch, e.g. let an out-of-scope query through
to generation. These tests pin the argmax-via-classes_ contract and confidence,
with the model and embedder injected so no model load or network is needed.
"""

import numpy as np

from ray_rag.models.intent import IntentClassifier


class _FakeClf:
    # sklearn sorts classes_ alphabetically -> deliberately NOT the INTENTS order,
    # so a positional shortcut would pick the wrong label and this test would catch it.
    classes_ = np.array(["factual", "out_of_scope", "summarize"])

    def __init__(self, proba):
        self._proba = proba

    def predict_proba(self, emb):
        return np.array([self._proba])


class _FakeEmbedder:
    def __init__(self):
        self.seen = None

    def encode(self, queries):
        self.seen = queries
        return np.zeros((len(queries), 4), dtype=np.float32)


def test_predict_returns_argmax_class_via_classes_and_its_confidence():
    embedder = _FakeEmbedder()
    clf = _FakeClf(proba=[0.1, 0.7, 0.2])  # argmax at idx 1 -> classes_[1]
    intent, confidence = IntentClassifier(clf, embedder).predict("is this in scope?")
    assert intent == "out_of_scope"  # mapped through classes_, not INTENTS order
    assert confidence == 0.7
    assert embedder.seen == ["is this in scope?"]  # query is wrapped in a batch


def test_predict_picks_a_different_class_when_probabilities_shift():
    clf = _FakeClf(proba=[0.8, 0.1, 0.1])  # argmax at idx 0
    intent, confidence = IntentClassifier(clf, _FakeEmbedder()).predict("what is ray?")
    assert intent == "factual"
    assert confidence == 0.8
