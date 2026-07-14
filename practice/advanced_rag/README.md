# Advanced RAG — practice module

Builds directly on the base harness `practice/rag_langchain_pinecone.py`
(OpenAI `text-embedding-3-small` + Pinecone serverless index `customer-orders-rag`
+ `gpt-4o-mini`, one Document per order over `practice/data/customer_orders.csv`).
Every file here **reuses that harness's plumbing** (`load_env`, `get_embeddings`,
`get_vectorstore`, `build_rag_chain`, `load_documents`) — nothing is re-plumbed.
The base module is core RAG; this module is the advanced layer an interviewer for
a mid-senior GenAI role probes once you've shown the basics.

## The five files

| # | File | What it teaches | Live vs reference |
|---|------|-----------------|-------------------|
| 1 | `01_advanced_retrieval.py` | Multi-Query, HyDE, Query Routing, Hybrid (dense + hand-rolled BM25, hand-rolled RRF) | **Live** (Pinecone + OpenAI) |
| 2 | `02_graph_rag.py` | In-memory knowledge graph (networkx) + traversal UNION vector retrieval; when Graph RAG beats vector RAG | **Live** graph offline; vector union uses the base index |
| 3 | `03_vector_dbs.py` | Pinecone vs Weaviate vs pgvector: config, HNSW knobs, tenancy, decision matrix | Pinecone **live**; Weaviate + pgvector **reference config only** (no server) |
| 4 | `04_retrieval_metrics.py` | hit-rate@k, MRR, precision@k, recall@k — LLM-free; tabulates which strategy from file 1 retrieves best | **Live** (Pinecone) |
| 5 | `05_unified_structured_unstructured.py` | Router → SQL/pandas for aggregation, Pinecone RAG for semantic; the aggregation-recall trap | SQL path **live offline**; semantic path uses the base index |

Run any file directly, e.g. `python practice/advanced_rag/01_advanced_retrieval.py`.
Keys load from repo-root `.env` via the base `load_env()`. Files degrade
gracefully (graph-only / SQL-only / matrix-only) when keys are absent.

## Maps to the 5 JD bullets

1. **Advanced retrieval strategies** (multi-query, HyDE, routing, hybrid) → file 1.
2. **Graph RAG / multi-hop relational retrieval** → file 2.
3. **Vector database selection & tuning at enterprise scale** → file 3.
4. **Retrieval evaluation (offline, deterministic) distinct from generation eval** → file 4
   (complements the LLM-judge metrics in `practice/rag_eval_ragas_deepeval.py`).
5. **Unified retrieval over structured + unstructured data** → file 5.

## Maps to Ratul's projects (say these in the room)

- **Synapse source-grader** = retrieval-failure handling — the safety net under
  hybrid/multi-query (files 1, 2) and the runtime version of "did retrieval
  surface evidence?" (file 4).
- **Synapse grounding-grader** = faithfulness — the generation-side metric that
  file 4's retrieval metrics deliberately sit *before*.
- **Synapse ReAct agent tool-choice** = query routing (file 1) and the
  structured/unstructured router (file 5).
- **Universal Document Ingestor** (multi-source, DeepEval) = the hybrid + tenancy
  story (files 1, 3) and the unified structured+unstructured architecture (file 5).

## The four things this module drills into muscle memory

1. Name the retrieval **failure mode → the fix** (file 1's opening table).
2. **Graphs for multi-hop/relational, vectors for fuzzy prose — union both** (file 2).
3. Pick a vector DB from **operational constraints, not benchmarks** (file 3).
4. **Debug retrieval first with cheap deterministic metrics; aggregation never
   goes to a vector store** (files 4 + 5).
