# Contributing

Thanks for taking a look. The bar for any change is simple: **the gate stays
green and the change earns its place.**

## Development setup

CPU-only, Python 3.10. Install the package with its dev tooling:

```bash
pip install -e ".[dev]"     # or: uv pip install -e ".[dev]"
```

You can run everything directly (Ray runs in-process) without Docker — see the
[README](README.md#run-it-locally) for the full local loop.

## The gate

Every change must pass the same checks CI runs on each push and PR
(`.github/workflows/ci.yml`), plus a Ray Serve smoke test:

```bash
make lint        # ruff check + ruff format --check (src tests scripts)
make typecheck   # mypy src scripts
make test        # pytest unit + e2e smoke
```

## Conventions

- **Tests encode intent.** A test states *why* a constraint matters and fails if
  the logic it guards changes — not merely exercises a code path. ML eval encodes
  why the metric matters (nDCG guards ranking, macro-F1 guards routing).
- **Fail loud.** No silent fallbacks or swallowed errors — surface them. See the
  empty-input guards in [`src/ray_rag/eval/harness.py`](src/ray_rag/eval/harness.py)
  for the house style.
- **The LLM only does grounded generation.** Ranking and routing are owned by
  trained models with measured accuracy; keep the LLM out of that path.
- Keep changes surgical and match the surrounding style.

See [RUNBOOK.md](RUNBOOK.md) for operating the running system and failure handling.
