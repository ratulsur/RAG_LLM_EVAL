"""
RETRIEVAL METRICS — hit-rate@k, MRR, precision@k, recall@k (NO LLM judge).
Advanced-RAG module, file 4. Runs the strategies from `01_advanced_retrieval.py`
against the live Pinecone index and TABULATES which retriever finds the gold
orders best. Requires the base index to be ingested.

WHY A SEPARATE, LLM-FREE EVAL (the sentence that shows you get it)
-----------------------------------------------------------------
Your RAGAS/DeepEval module scores GENERATION quality (faithfulness,
answer-relevance) with an LLM judge — expensive, non-deterministic, and it can't
tell you WHERE a failure started. These metrics score RETRIEVAL ONLY, with pure
set math against known-relevant order ids: deterministic, free, and CI-cheap. The
split is the whole diagnostic story:

    RETRIEVAL quality (this file)     GENERATION quality (eval module)
    ----------------------------      --------------------------------
    hit-rate@k, MRR, recall@k,        faithfulness (= grounding grader),
    precision@k  -> IS THE EVIDENCE   answer-relevance -> DID THE MODEL USE
    IN THE TOP-K AT ALL?               THE EVIDENCE WELL?

If recall@k here is low, no prompt or model change can fix the answer — the
evidence never arrived. So you debug retrieval FIRST, with cheap deterministic
metrics, and only reach for the LLM judge once retrieval clears the bar. That
ordering is the senior workflow. It also maps to your Synapse source-grader:
the grader is the runtime version of "did retrieval actually surface evidence?"

THE FOUR METRICS (be able to define each in one breath)
  hit-rate@k : fraction of queries with AT LEAST ONE relevant doc in top-k.
               "Did we get anything right?" The floor metric.
  MRR        : mean of 1/rank of the FIRST relevant doc. Rewards ranking the
               right answer HIGH, not just including it. RANK-sensitive.
  precision@k: of the k retrieved, what fraction are relevant. Signal-to-noise;
               low precision = the LLM wades through junk (context dilution).
  recall@k   : of the relevant docs that EXIST, what fraction are in top-k.
               The completeness metric — the one aggregation questions fail.

RUN IT
    python practice/advanced_rag/04_retrieval_metrics.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))


def _load_strategies():
    """File 1 is named `01_advanced_retrieval.py` — a leading digit, so it can't
    be `import`ed by name. Load it by PATH via importlib. This is the clean way
    to reuse a numbered practice file; worth knowing for exactly this situation."""
    spec = importlib.util.spec_from_file_location(
        "advanced_retrieval", HERE / "01_advanced_retrieval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# GOLD LABELS — question -> the set of order_ids that SHOULD be retrieved
# ===========================================================================
# Labelling retrieval is cheaper than labelling answers: you only assert WHICH
# rows are relevant, not the prose. That's why retrieval eval scales — you can
# hand-label dozens of these in minutes. Keep ids as strings (Pinecone metadata).
GOLD: list[tuple[str, list[str]]] = [
    ("What happened with the faulty wireless mouse?", ["1011"]),
    ("Which orders were returned?", ["1011", "1016"]),
    ("Which orders were cancelled?", ["1004", "1014"]),
    ("Tell me about the 27-inch monitor orders", ["1003", "1013"]),
    ("What is the status of the ergonomic chair orders?", ["1008", "1016"]),
    ("Which orders shipped to APAC-North?", ["1003", "1009", "1013"]),
]


# ===========================================================================
# METRIC MATH — pure set operations over ranked order_id lists (no LLM)
# ===========================================================================
def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / (rank of first relevant hit). 0 if none. Ranks are 1-based here so a
    top-1 hit scores 1.0 — the textbook MRR convention."""
    for i, rid in enumerate(retrieved, start=1):
        if rid in relevant:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    topk = retrieved[:k]
    return len(set(topk) & relevant) / max(len(topk), 1)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / max(len(relevant), 1)


# ===========================================================================
# RUN EACH STRATEGY OVER THE GOLD SET, AVERAGE THE METRICS
# ===========================================================================
def ids_of(docs, S) -> list[str]:
    """Ranked list of order_ids from a strategy's Document output, dedup-preserved."""
    out, seen = [], set()
    for d in docs:
        rid = S.doc_id(d)
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def evaluate_strategy(name: str, retrieve_fn, S, k: int) -> dict:
    """retrieve_fn(query) -> list[Document]. Average each metric over GOLD."""
    hr = mrr = prec = rec = 0.0
    for question, relevant_ids in GOLD:
        relevant = set(relevant_ids)
        retrieved = ids_of(retrieve_fn(question), S)
        hr += hit_rate_at_k(retrieved, relevant, k)
        mrr += reciprocal_rank(retrieved, relevant)
        prec += precision_at_k(retrieved, relevant, k)
        rec += recall_at_k(retrieved, relevant, k)
    n = len(GOLD)
    return {"strategy": name, f"hit@{k}": hr / n, "MRR": mrr / n,
            f"P@{k}": prec / n, f"R@{k}": rec / n}


def main() -> None:
    S = _load_strategies()
    S.load_env()
    from langchain_openai import ChatOpenAI

    embeddings = S.get_embeddings()
    vs = S.get_vectorstore(embeddings)
    llm = ChatOpenAI(model=S.CHAT_MODEL, temperature=0)
    # k=3 (not a larger k) on purpose: with only 16 docs a big k lets EVERY
    # strategy scoop up the answers and the table flattens to a tie. Tightening
    # k surfaces the real separation — the hybrid RRF fusion edges out plain
    # dense on the multi-relevant 'which orders' queries. Always eval at the k
    # you'll actually serve, not the k that flatters your retriever.
    K = 3

    # Wrap each strategy as retrieve_fn(query) -> list[Document]. We deliberately
    # give multi-query/hybrid a larger candidate budget (that's their point:
    # trade a few extra calls for recall), then measure at the SAME top-k.
    strategies = {
        "dense":        lambda q: S.dense_retrieve(vs, q, k=K),
        "multi_query":  lambda q: S.multi_query_retrieve(vs, llm, q, n=3, k=K),
        "hyde":         lambda q: S.hyde_retrieve(vs, embeddings, llm, q, k=K),
        "hybrid_rrf":   lambda q: S.hybrid_retrieve(vs, q, k=K),
    }

    print(f"Evaluating {len(strategies)} strategies over {len(GOLD)} labelled "
          f"queries at k={K} (metrics are LLM-free set math).\n")
    results = [evaluate_strategy(name, fn, S, K) for name, fn in strategies.items()]

    # tabulate
    cols = ["strategy", f"hit@{K}", "MRR", f"P@{K}", f"R@{K}"]
    widths = [12, 8, 7, 7, 7]
    print("  " + "".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  " + "-" * sum(widths))
    for r in sorted(results, key=lambda x: x[f"R@{K}"], reverse=True):
        cells = [r["strategy"]] + [f"{r[c]:.3f}" for c in cols[1:]]
        print("  " + "".join(c.ljust(w) for c, w in zip(cells, widths)))

    best = max(results, key=lambda x: x[f"R@{K}"])
    print(f"\n  Best recall@{K}: {best['strategy']}. Read the table as: hit-rate is the\n"
          f"  floor, MRR rewards ranking the answer high, precision is signal-to-noise,\n"
          f"  recall is completeness. For the 'which orders' aggregation queries, watch\n"
          f"  recall — that's where a single dense top-k quietly loses rows.")


if __name__ == "__main__":
    main()
