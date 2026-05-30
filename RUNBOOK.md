# Runbook

Operational guide for running and recovering the RAG service. Commands assume
the local Docker Ray cluster (`make up`); prefix with `docker compose exec ray-head`
to run inside the cluster, or run directly in a venv after `make install`.

## Normal startup order

1. `make up` — start the Ray head + worker.
2. `make ingest` — build the vector index (`artifacts/index.faiss` + sidecar).
3. `make train` — tune + fit the reranker and intent classifier into `artifacts/`.
4. `make serve` — start the Serve graph on `:8000`.
5. `make eval` — sanity-check metrics.

Health check: `curl localhost:8000/health` → `{"status": "ok"}`.
Ray dashboard: http://localhost:8265.

## Failure handling

| Symptom | Likely cause | Action |
|---------|-------------|--------|
| Serve replicas fail to start: `FileNotFoundError` on index/model | step 2/3 not run, or `artifacts/` not visible to workers | Run `make ingest && make train`; ensure artifacts are on shared/cluster storage. |
| `index/sidecar desync` error on load | index and `.chunks.jsonl` out of sync (partial write) | Delete `artifacts/index.faiss*` and re-run `make ingest`. |
| `/ask` returns the out-of-scope refusal for a valid question | intent classifier misrouted | Inspect with `make eval` (intent F1); add labelled examples to `data/intents/intents.jsonl` and retrain. |
| Generation errors / 401 | `ANTHROPIC_API_KEY` unset or invalid | Set it in `.env` (never commit it). The generator fails loud rather than returning an ungrounded answer. |
| Empty / irrelevant answers | retrieval miss or weak reranking | Check `make eval` nDCG uplift; raise `RAYRAG_RETRIEVE_TOP_K`; confirm the corpus actually covers the question. |
| Reranker training raises "need >=2 labelled queries" | too few rows in `relevance.jsonl` | Add labelled queries; each must retrieve at least one candidate. |
| Tune trials all fail | dependency/version mismatch in the cluster env | Rebuild the image (`make up`); check `pyproject.toml` pins. |

## Re-runs are safe

`make ingest` rebuilds the index from scratch (deterministic chunk ids).
`make train` overwrites the artifacts. Neither mutates the corpus or labels.
