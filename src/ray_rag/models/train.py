"""Train both models end to end: ensure the index exists, then tune + fit the
reranker and the intent classifier, saving artifacts. Run: `python -m ray_rag.models.train`.

The embedding model is shared across the index build, the reranker's dense
feature, and the intent features so every vector comes from one model.
"""

from __future__ import annotations

import json
from pathlib import Path

import ray

from ray_rag.config import settings
from ray_rag.data.embed import Embedder, build_corpus_index
from ray_rag.data.index import VectorIndex
from ray_rag.models.intent import train_intent
from ray_rag.models.reranker import train_reranker


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    if not ray.is_initialized():
        ray.init()
    Path(settings.reranker_path).parent.mkdir(parents=True, exist_ok=True)

    if Path(settings.index_path).exists():
        index = VectorIndex.load(settings.index_path)
    else:
        index = build_corpus_index(settings.corpus_path, settings.embed_model, settings.index_path)
    embedder = Embedder(settings.embed_model)

    rr = train_reranker(index, embedder, load_jsonl(settings.eval_path), settings.reranker_path)
    ic = train_intent(load_jsonl(settings.intents_path), embedder, settings.intent_path)

    print(f"reranker: val_nDCG@5={rr['val_ndcg']:.3f}  -> {settings.reranker_path}")
    print(f"  best config: {rr['config']}")
    print(f"intent:   cv_macroF1={ic['cv_f1']:.3f}  -> {settings.intent_path}")
    print(f"  best config: {ic['config']}")


if __name__ == "__main__":
    main()
