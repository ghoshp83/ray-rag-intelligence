"""Smoke test: the package imports and config exposes sane defaults.

Intent: a broken package layout or a config regression (e.g. a default that
would silently break retrieval depth) fails CI here, not in production.
"""

from ray_rag.config import settings


def test_config_defaults_are_sane():
    assert settings.retrieve_top_k >= settings.rerank_top_k > 0
    assert settings.embed_model
    assert settings.llm_model
