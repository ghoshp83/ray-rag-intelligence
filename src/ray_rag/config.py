"""Central configuration, sourced from environment variables (Hard Rule 7).

All knobs live here as one Pydantic settings object so every Ray task, training
job, and Serve deployment reads the same source of truth. Secrets (the LLM API
key) are never defaulted to a real value — absent key fails loud at call time,
not silently.
"""

from __future__ import annotations

from pydantic import model_validator
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

    @model_validator(mode="after")
    def _check_retrieval_depths(self) -> Settings:
        # rerank selects its top-k from the retrieved candidates, so asking for
        # more reranked passages than were retrieved is a misconfiguration: the
        # pipeline would silently return fewer passages than rerank_top_k claims.
        # Enforce it at load time (env can override either knob) rather than let a
        # typo cap retrieval invisibly.
        if not 1 <= self.rerank_top_k <= self.retrieve_top_k:
            raise ValueError(
                f"rerank_top_k ({self.rerank_top_k}) must be in 1..retrieve_top_k "
                f"({self.retrieve_top_k}); cannot rerank more passages than retrieved"
            )
        return self


settings = Settings()
