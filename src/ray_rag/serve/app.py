"""Bind the deployment graph. Run with:  serve run ray_rag.serve.app:app

Each stage is bound as its own deployment and handed to the ingress, so Serve
manages them as independently scalable replicas behind one HTTP endpoint.
"""

from __future__ import annotations

from ray_rag.serve.deployments import (
    Generator,
    Ingress,
    RerankerDeployment,
    Retriever,
    Router,
)

app = Ingress.bind(
    Retriever.bind(),
    RerankerDeployment.bind(),
    Router.bind(),
    Generator.bind(),
)
