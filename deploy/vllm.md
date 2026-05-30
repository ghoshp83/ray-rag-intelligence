# GPU generation with vLLM on Ray Serve

> **Honest disclaimer:** the reference machine has no GPU, so the happy path
> generates with the Anthropic Claude API. This document is the architected
> scale-out for self-hosted generation; it is not running in this project.

The deployment graph isolates generation in one deployment (`Generator` in
`src/ray_rag/serve/deployments.py`). Self-hosting an open model means replacing
*only* that deployment — `Retriever`, `RerankerDeployment`, and `Router` are
unchanged, which is the reason the graph is split by responsibility.

## The swap

Replace the Claude-backed `Generator` with a vLLM-backed one that keeps the same
`generate(query, passages) -> {answer, sources}` interface:

```python
@serve.deployment(ray_actor_options={"num_gpus": 1})
class VLLMGenerator:
    def __init__(self) -> None:
        from vllm import LLM
        self._llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")  # example open model

    def generate(self, query: str, passages: list[dict]) -> dict:
        system, messages = build_messages(query, passages)   # same grounding prompt
        prompt = render_chat(system, messages)                # model-specific template
        out = self._llm.generate([prompt])[0].outputs[0].text
        sources = [{"chunk_id": p["chunk_id"], "source": p["source"]} for p in passages]
        return {"answer": out, "sources": sources}
```

Then bind `VLLMGenerator` instead of `Generator` in `app.py`. The grounding
contract (`build_messages`) and the citation-faithfulness eval are reused as-is,
so the trustworthiness checks still apply to the self-hosted model.

## Notes
- Needs a GPU worker group (e.g. on Anyscale, see `anyscale.md`).
- vLLM has a CPU backend, but generation throughput/quality on CPU is poor —
  hence Claude API for the local happy path.
