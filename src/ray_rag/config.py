"""Central configuration, sourced from environment variables (Hard Rule 7).

All knobs live here as one Pydantic settings object so every Ray task, training
job, and Serve deployment reads the same source of truth. Secrets (the LLM API
key) are never defaulted to a real value — absent key fails loud at call time,
not silently.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAYRAG_", env_file=".env", extra="ignore")

    # --- GenAI generation (happy path). Key read separately, unprefixed. ---
    llm_model: str = "claude-opus-4-8"

    # --- Retrieval / models ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_path: str = "artifacts/reranker.json"
    intent_path: str = "artifacts/intent_clf.joblib"
    index_path: str = "artifacts/index.faiss"
    eval_report_path: str = "artifacts/eval_report.json"

    # --- Retrieval depths ---
    retrieve_top_k: int = 50
    rerank_top_k: int = 5

    # --- Data ---
    corpus_path: str = "data/corpus"
    # Reranker labels are split so the headline nDCG/MRR is held-out, not a
    # train-set number: train_path tunes+fits the ranker, eval_path is unseen.
    eval_train_path: str = "data/eval/relevance_train.jsonl"
    eval_path: str = "data/eval/relevance_test.jsonl"
    intents_path: str = "data/intents/intents.jsonl"


settings = Settings()
