.PHONY: install lint typecheck test ingest train eval bench latency serve up down

install:  ## install package + dev tooling
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

typecheck:
	mypy src

test:
	pytest -q

ingest:   ## build the vector index from the corpus (Ray Data)
	python -m ray_rag.data.embed

train:    ## train + tune the reranker and intent classifier (Ray Tune)
	python -m ray_rag.models.train

eval:     ## print recall@k, nDCG/MRR, intent F1, and grounding score
	python -m ray_rag.eval.harness

bench:    ## measure Ray Data embedding throughput (chunks/sec)
	python scripts/measure_throughput.py

latency:  ## measure request-path latency per stage (route/retrieve/rerank)
	python scripts/measure_latency.py

serve:    ## run the Ray Serve deployment graph on :8000
	serve run ray_rag.serve.app:app

up:       ## build + start the local Ray cluster (head + worker)
	docker compose up -d --build

down:
	docker compose down
