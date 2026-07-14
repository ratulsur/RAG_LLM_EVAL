# RAG_LLM_EVAL

A hands-on reference codebase for **production-grade Retrieval-Augmented Generation, RAG evaluation, and LLM optimization** — built around one runnable pipeline and progressively extended into advanced retrieval, exhaustive evaluation, hallucination testing, and LLM monitoring.

Everything runs against a single, coherent example: a **customer-orders** knowledge base served by **LangChain + Pinecone + OpenAI**. Each module is a self-contained, heavily-commented file you can read, run, and adapt.

**Stack:** Python 3.12 · LangChain · Pinecone (serverless) · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`) · RAGAS · DeepEval · FastAPI

---

## The reference pipeline

```
CSV orders ─▶ Documents ─▶ OpenAIEmbeddings ─▶ Pinecone ─▶ retriever ─▶ LCEL chain ─▶ ChatOpenAI ─▶ grounded, cited answer
                                                                 │
                                              (grounding + abstention: "I don't know" on empty retrieval)
```

`practice/rag_langchain_pinecone.py` is the runnable core. Every other module evaluates, extends, optimizes, or monitors it.

---

## Module map

| Area | Files | What it covers |
|------|-------|----------------|
| **Base RAG** | `practice/rag_langchain_pinecone.py` · `rag_with_langchain_semantic.py` · `tue_rag_core_reference.py` · `rag_fastapi_service.py` | End-to-end RAG: ingestion, chunking, embeddings, retrieval, generation. From-scratch (no framework), LangChain + semantic chunking, the runnable Pinecone pipeline, and a FastAPI service. |
| **Advanced retrieval** | `practice/advanced_rag/` | Multi-query, HyDE, query routing, hybrid (dense + BM25 + RRF), Graph RAG, vector-DB tradeoffs (Pinecone / Weaviate / pgvector), retrieval-quality metrics, and unified structured + unstructured retrieval. |
| **Evaluation** | `practice/eval_suite/` | Exhaustive RAGAS (with CI gate + failure export), exhaustive DeepEval (+ custom GEval + pytest harness), and a dedicated 7-check hallucination suite. |
| **LLM optimization** | `practice/llm_optimization/` | Prompt & structured-output optimization, inference cost/latency (caching, batching, model cascade, quantization), and LLM monitoring/observability. |

Each subfolder has its own `README.md` with a deeper breakdown.

---

## Repository layout

```
practice/
├── rag_langchain_pinecone.py        # ★ runnable reference pipeline (OpenAI + Pinecone)
├── rag_with_langchain_semantic.py   # LangChain + semantic chunking (study material)
├── tue_rag_core_reference.py        # RAG from scratch, no framework (OOP)
├── rag_eval_ragas_deepeval.py       # intro eval (RAGAS + DeepEval)
├── rag_fastapi_service.py           # RAG as a FastAPI service
├── data/customer_orders.csv         # sample knowledge base (synthetic)
├── advanced_rag/
│   ├── 01_advanced_retrieval.py     # multi-query, HyDE, routing, hybrid + RRF
│   ├── 02_graph_rag.py              # in-memory knowledge graph + graph∪vector
│   ├── 03_vector_dbs.py             # Pinecone vs Weaviate vs pgvector
│   ├── 04_retrieval_metrics.py      # hit@k, MRR, precision@k, recall@k
│   └── 05_unified_structured_unstructured.py  # router: SQL vs vector RAG
├── eval_suite/
│   ├── ragas_production.py          # exhaustive RAGAS + CI gate + failure export
│   ├── deepeval_production.py       # exhaustive DeepEval + GEval + pytest harness
│   └── hallucination_suite.py       # 7 dedicated hallucination checks
├── llm_optimization/
│   ├── 01_prompt_and_output_optimization.py
│   ├── 02_inference_cost_latency.py
│   └── 03_llm_monitoring_observability.py
└── *.ipynb                          # exploratory notebooks (chunking, reranking, etc.)
```

---

## Setup

```bash
# 1. Python 3.12 + virtual environment
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide API keys (see below)
cp .env.example .env                 # then fill in your keys
```

Create a `.env` in the repo root with:

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

The pipeline reads these via `python-dotenv`. The live paths call OpenAI and Pinecone; offline stub paths in the eval files run without keys.

---

## Quickstart

```bash
# Ingest the sample orders into Pinecone (creates the serverless index)
python practice/rag_langchain_pinecone.py ingest

# Ask grounded, cited questions
python practice/rag_langchain_pinecone.py query "What happened with order 1011?"

# Exhaustive / filtered retrieval (fixes dense top-k recall misses)
python practice/rag_langchain_pinecone.py query "Which orders were returned?" --filter status=Returned --top-k 16

# Ingest + a few demo questions in one go
python practice/rag_langchain_pinecone.py demo
```

Run the API service:

```bash
cd practice
uvicorn rag_fastapi_service:app --reload   # then POST /ingest, POST /query, GET /health
```

---

## Running the modules

All commands are run from the repo root (the module files resolve the shared pipeline import internally).

```bash
# Advanced retrieval — compare strategies with real metrics
python practice/advanced_rag/04_retrieval_metrics.py
python practice/advanced_rag/01_advanced_retrieval.py
python practice/advanced_rag/03_vector_dbs.py          # prints the DB decision matrix

# Evaluation
python practice/eval_suite/ragas_production.py          # metrics + CI gate (exit code)
python practice/eval_suite/deepeval_production.py
python practice/eval_suite/hallucination_suite.py       # 7 hallucination checks

# LLM optimization & monitoring
python practice/llm_optimization/02_inference_cost_latency.py
python practice/llm_optimization/03_llm_monitoring_observability.py
```

Each eval file has an **offline stub** path (no keys/network) so you can see the metric shapes before spending API calls.

---

## Design principles

- **Grounding first.** The prompt answers *only* from retrieved context and abstains ("I don't know") on empty retrieval — the cheapest hallucination control, framework or not.
- **Right tool per query.** Tabular records are indexed one document per row; semantic chunking is reserved for prose. Aggregation/filter questions use metadata filters (or SQL), not dense top-k luck.
- **Measure, don't eyeball.** Retrieval quality (hit-rate, MRR, precision/recall) is separated from generation quality (faithfulness, answer relevance) so a regression can be localized to the retriever or the generator.
- **Self-contained, expert-level.** No modular scaffolding — each file is a tight, readable reference you can reason about end to end.

---

## Notes

- The `data/customer_orders.csv` dataset is **synthetic** — 16 fictional orders used purely for demonstration.
- Live runs incur small OpenAI + Pinecone costs. Pinecone uses a **serverless** index (`customer-orders-rag`, 1536-dim, cosine).
- The Weaviate and pgvector sections in `advanced_rag/03_vector_dbs.py` are **reference config** (no local server required); the Pinecone paths are fully live.

---

*A learning + reference repository for RAG and LLM evaluation engineering.*
