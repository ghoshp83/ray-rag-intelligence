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
| Reranker training raises "need >=2 labelled queries" | too few rows in `data/eval/relevance_train.jsonl` | Add labelled queries; each must retrieve at least one candidate. |
| Tune trials all fail | dependency/version mismatch in the cluster env | Rebuild the image (`make up`); check `pyproject.toml` pins. |

## Re-runs are safe

`make ingest` rebuilds the index from scratch (deterministic chunk ids).
`make train` overwrites the artifacts. Neither mutates the corpus or labels.

## Reading the reranker eval honestly

`make eval` prints, alongside the dense→reranked nDCG/MRR averages, a per-query
uplift line — `improved / regressed / tied` — and the single worst regression
and best gain. Read the averages *with* that breakdown: a flat average can be a
genuine tie or it can be wins and regressions cancelling out, and only the
per-query counts tell them apart. A reranker worth shipping improves more
queries than it regresses.

If the reranker shows no uplift over dense (counts skew to `tied`), that is
usually not a model bug — it means the corpus gives dense retrieval no ordering
to fix. Dense embeddings separate topically-distinct documents cleanly, so on a
small clean corpus `recall@5` saturates at 1.000 and dense already orders the top
near-perfectly. The lever is the **corpus**, not the hyperparameters: add
*lexically-confusable hard negatives* — sibling documents that share surface
vocabulary with a query but answer a different question — then label queries over
that confusable region (`data/eval/relevance_*.jsonl`), `make ingest`, `make
train`, `make eval`. That is how the bundled corpus earns its uplift. Change the
data, re-run **once**, and report the result as-is; do not sweep seeds or trial
counts to chase a number.
