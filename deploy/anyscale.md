# Production scale-out on Anyscale

> **Honest disclaimer:** this is the documented production path. The repo's happy
> path runs on the local Docker Ray cluster (CPU, free). The Anyscale steps below
> are real and idiomatic but are not continuously running in this project.

The whole point of building on Ray is that the *same code* moves to a cluster by
changing where it runs, not what it does. Nothing in `src/ray_rag/` is local-only.

## What changes vs. local

| Concern | Local (this repo) | Anyscale |
|---------|-------------------|----------|
| Cluster | `docker compose` head + worker | Anyscale workspace / service, autoscaling worker groups |
| Embedding (Ray Data) | CPU worker group | scale CPU workers; throughput grows ~linearly |
| Tuning (Ray Tune) | trials across local CPUs | trials across the autoscaling cluster |
| Serving (Ray Serve) | `serve run ...` on :8000 | Anyscale Service (managed ingress, autoscaling, zero-downtime rollout) |
| Generation | Claude API (CPU) | Claude API **or** a GPU worker group running the vLLM `Generator` (see `vllm.md`) |

## Steps (sketch)

1. Define the cluster environment from `pyproject.toml` (an Anyscale cluster env
   pins the same dependencies the Docker image installs).
2. Build the index and train the models as Anyscale **jobs**:
   `python -m ray_rag.data.embed`, then `python -m ray_rag.models.train`.
   Persist `artifacts/` to cluster storage (or object storage) so Serve can load it.
3. Deploy the graph as an Anyscale **Service** from `ray_rag.serve.app:app`.
   Add a GPU worker group only if you switch the generator to vLLM.
4. Set `ANTHROPIC_API_KEY` as a service secret (never in the repo — Hard Rule 7).

## Why not make Anyscale the only target

A hiring reviewer must be able to `git clone` and run this without a paid
account. Local-first keeps it reproducible; Anyscale is the scale story.
