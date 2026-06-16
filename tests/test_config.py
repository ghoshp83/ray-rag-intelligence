"""Smoke test: the package imports and config exposes sane defaults.

Intent: a broken package layout or a config regression (e.g. a default that
would silently break retrieval depth) fails CI here, not in production.
"""

import pytest
from pydantic import ValidationError

from ray_rag.config import Settings, settings


def test_config_defaults_are_sane():
    assert settings.retrieve_top_k >= settings.rerank_top_k > 0
    assert settings.embed_model
    assert settings.llm_model


def test_reranking_more_than_retrieved_fails_loud():
    # An env typo (e.g. RAYRAG_RERANK_TOP_K=100 with retrieve_top_k=50) must not
    # load silently and then cap retrieval invisibly — it must fail at construction.
    with pytest.raises(ValidationError, match="cannot rerank more passages than retrieved"):
        Settings(retrieve_top_k=50, rerank_top_k=100)


def test_rerank_top_k_must_be_at_least_one():
    with pytest.raises(ValidationError):
        Settings(rerank_top_k=0)
