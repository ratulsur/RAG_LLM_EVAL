# llm_optimization — making the RAG cheap, fast, reliable, and observable

Optimization + production-operations layer over the reference pipeline
`practice/rag_langchain_pinecone.py`. The interview framing: *"your RAG works —
now make it 10× cheaper, 2× faster, and tell me what it's doing under real
traffic."* **LLM monitoring/observability and RAG hallucination recur constantly
in your interviews**, so files `03` and `../eval_suite/hallucination_suite.py`
are the two to know cold.

Every file runs **offline** (pure stdlib / hand-rolled, no keys) or **live**
(`--live`, real OpenAI + Pinecone).

## Files → interview themes → your projects

| File | Interview theme | Maps to your work |
|---|---|---|
| `01_prompt_and_output_optimization.py` | "Make the prompt reliable and cheap" | structured output → Synapse grounding grader reads typed fields, not prose |
| `02_inference_cost_latency.py` | "Cut cost/latency without hurting quality" | quantization → **AP Audit** QLoRA fine-tune of Phi-3.5 |
| `03_llm_monitoring_observability.py` | "How do you monitor an LLM app in prod?" (emphasized) | groundedness guardrail = Synapse grounding grader inline; feedback loop → eval set |

## 01 — prompt & output optimization

Maximize **answer quality per token**. Runnable/measured blocks:

- **zero-shot vs few-shot**, costed in tokens (few-shot = recurring per-call cost;
  use it only to teach a behavior like abstention).
- **structured output** — Pydantic schema + `with_structured_output` →
  `{answer, cited_order_ids, grounded}` typed object, so the downstream grader
  reads fields instead of regexing prose.
- **validation + repair** — validate against the schema; on failure send **one
  bounded** repair prompt, then fail closed (never spin an unbounded loop).
- **prompt compression** — extractive trim that keeps evidence (verified: the
  status/restock sentences survive a 55% cut) — always re-check faithfulness.
- **token budgeting** — `tiktoken` exact counts (hand-rolled ~4-chars/token
  fallback); fit context to a budget by dropping whole low-rank chunks, never
  truncating mid-document.

## 02 — inference cost & latency

The levers, in the order you reach for them:

1. **exact cache** (normalize → hash → memoize, LRU).
2. **semantic cache** — hand-rolled embedding + cosine ≥ threshold; catches
   paraphrases the exact cache misses (demo hits at sim 1.00).
3. **model cascade / routing** — cheap model answers; escalate to the strong model
   only on low confidence (demo: strong-model price paid on 1 of 3 queries).
4. **streaming** — `chain.stream()` cuts *perceived* latency (time-to-first-token).
5. **batching** — `chain.batch()` amortizes overhead for bulk/offline work.
6. **context trimming** — fewer retrieved tokens → lower cost + latency (file 01).
7. **quantization** — fp16→int4 ≈ 4× less memory; **QLoRA = frozen 4-bit base +
   LoRA adapters = your AP Audit Phi-3.5 fine-tune**. Re-eval faithfulness after
   quantizing.

`--live` measures real time-to-first-token, cold-vs-warm cache latency, and a real
OpenAI-embedding semantic-cache hit.

## 03 — monitoring & observability (the emphasized file)

A **runnable in-memory `RAGMonitor`** that wraps the orders chain and prints a
trace + fleet-metrics summary. Seven pillars:

1. **tracing** — per request: prompt, response, latency, tokens, cost, retrieved
   doc ids, model/prompt version.
2. **online evals** — faithfulness + relevance on a **sample** of live traffic
   (no ground truth → works in prod; the CI-vs-online distinction operationalized).
3. **drift detection** — PSI over the score distribution + question-embedding
   centroid drift.
4. **guardrails** — input (PII, prompt-injection → fail closed) + output (PII leak,
   toxicity, **groundedness gate** = Synapse grounding grader inline).
5. **metrics + alerting** — p50/p95 latency, cost, error rate, abstention rate,
   faithfulness, with thresholds that fire alerts.
6. **feedback loops** — thumbs-down trace → tomorrow's eval set (loop back to
   `../eval_suite/`).
7. **A/B + canary** — route a fraction to a canary variant, compare, promote/roll
   back. Never flip 100% blind.

**LangSmith** patterns (`@traceable`, tracing env vars, server-side online
evaluators) are included as **clearly-labelled reference** — no account/keys
needed to read them. The hand-rolled monitor captures the same signals so you can
explain what LangSmith does for you.

## Dependencies

All already installed — **no new deps required**: `tiktoken`, `pydantic`,
`langchain-openai`. `tiktoken` is used for exact token counts with a hand-rolled
`~4-chars/token` fallback if it were absent. All offline paths are pure stdlib.
