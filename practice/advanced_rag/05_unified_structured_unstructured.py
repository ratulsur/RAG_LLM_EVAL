"""
UNIFIED RETRIEVAL — structured (SQL/pandas) + unstructured (Pinecone RAG).
Advanced-RAG module, file 5. A router classifies each question and sends it to
the RIGHT engine: aggregation/filter/analytics -> SQL over a table; semantic/
explanatory -> vector RAG over the prose notes; then it fuses the answer.

THE LESSON THIS FILE DRILLS (the aggregation-recall trap)
---------------------------------------------------------
"Total revenue by region" or "how many orders were returned" must NOT go to a
vector store. A vector DB retrieves the top-k most SIMILAR rows — it has no
notion of SUM, COUNT, or GROUP BY. Ask it "how many returned?" and it returns
~k returned-ish rows and the LLM guesses a count from a partial view. That's the
aggregation-recall failure: the store can't be exhaustive over a computation.
The fix is not a better prompt or a bigger k — it's ROUTING the computation to a
system that computes: SQL/pandas. Say exactly that in the room; it's the
single most common RAG design mistake, and naming it is a senior signal.

    question shape                          engine            why
    --------------                          ------            ---
    aggregate / filter / count / "how many" SQL or pandas     needs EXACT computation
    "why / explain / what happened"          vector RAG        needs SEMANTIC prose
    both ("returned orders and why")         SQL then RAG      compute set, explain it

MAPS TO YOUR PROJECTS
  - This is your Synapse ReAct agent's tool-choice made concrete: SQL is one tool,
    vector search is another, and the router is the policy. Your source-grader
    still guards the semantic path (retrieval-failure handling); the SQL path
    doesn't need it because it's exact.
  - Universal Document Ingestor mixes structured records and prose — this is the
    honest architecture for "unified retrieval over both."

DESIGN NOTE (self-contained on purpose): structured store = SQLite built in
memory from the same CSV; unstructured store = the live Pinecone index. No new
services. SQLite is stdlib, so the structured path runs even with no API keys.

RUN IT
    python practice/advanced_rag/05_unified_structured_unstructured.py
"""

from __future__ import annotations

import csv
import re
import sqlite3
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from rag_langchain_pinecone import DATA_CSV  # noqa: E402


# ===========================================================================
# STRUCTURED STORE — SQLite table built from the CSV (in memory)
# ===========================================================================
def build_sqlite(csv_path: Path = DATA_CSV) -> sqlite3.Connection:
    """One row per order, typed columns. In-memory so it's disposable and needs
    no file. This is the 'right tool' for anything countable/aggregable."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE orders (
            order_id TEXT, order_date TEXT, customer_name TEXT, region TEXT,
            product TEXT, category TEXT, quantity INTEGER, unit_price REAL,
            total REAL, status TEXT, notes TEXT)"""
    )
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        rows = [
            (r["order_id"], r["order_date"], r["customer_name"], r["region"],
             r["product"], r["category"], int(r["quantity"]), float(r["unit_price"]),
             float(r["total"]), r["status"], r["notes"])
            for r in csv.DictReader(f)
        ]
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    return conn


# ===========================================================================
# THE ROUTER — classify the question, pick the engine
# ===========================================================================
# We show a RULES router (deterministic, free, auditable — ship this first). An
# LLM router is a drop-in upgrade for the tail; the interview point is that the
# CLASSIFICATION exists at all, not which classifier you use.
AGG_SIGNALS = ("how many", "count", "total", "sum", "average", "avg", "revenue",
               "by region", "by status", "by category", "most", "least", "number of")
SEMANTIC_SIGNALS = ("why", "explain", "what happened", "reason", "describe", "tell me about")


def route(question: str) -> str:
    q = question.lower()
    if any(s in q for s in AGG_SIGNALS):
        return "structured"
    if any(s in q for s in SEMANTIC_SIGNALS):
        return "unstructured"
    return "unstructured"  # default: prose RAG is the safer fallback for free text


# ===========================================================================
# STRUCTURED PATH — deterministic SQL for the aggregation questions
# ===========================================================================
# In a full system an LLM would write the SQL (text-to-SQL) behind a strict
# allow-list. Here we map a few intents to VETTED SQL so the file is runnable and
# safe — and so you can talk about text-to-SQL guardrails (never execute raw LLM
# SQL against prod without validation) without needing the LLM to run.
def answer_structured(conn: sqlite3.Connection, question: str) -> str:
    q = question.lower()
    cur = conn.cursor()
    if "revenue" in q or ("total" in q and "region" in q):
        cur.execute("SELECT region, ROUND(SUM(total),2) rev FROM orders "
                    "GROUP BY region ORDER BY rev DESC")
        return "Revenue by region: " + ", ".join(f"{r['region']}=${r['rev']}" for r in cur)
    if "how many" in q and "return" in q:
        n = cur.execute("SELECT COUNT(*) c FROM orders WHERE status='Returned'").fetchone()["c"]
        return f"{n} orders were returned."
    if "how many" in q and "cancel" in q:
        n = cur.execute("SELECT COUNT(*) c FROM orders WHERE status='Cancelled'").fetchone()["c"]
        return f"{n} orders were cancelled."
    if "by status" in q or ("how many" in q and "status" in q):
        cur.execute("SELECT status, COUNT(*) c FROM orders GROUP BY status ORDER BY c DESC")
        return "Orders by status: " + ", ".join(f"{r['status']}={r['c']}" for r in cur)
    if "by category" in q or "revenue by category" in q:
        cur.execute("SELECT category, ROUND(SUM(total),2) rev FROM orders "
                    "GROUP BY category ORDER BY rev DESC")
        return "Revenue by category: " + ", ".join(f"{r['category']}=${r['rev']}" for r in cur)
    # generic count fallback
    n = cur.execute("SELECT COUNT(*) c FROM orders").fetchone()["c"]
    return f"(no specific aggregation matched) There are {n} orders in total."


# ===========================================================================
# UNSTRUCTURED PATH — vector RAG over the prose notes (Pinecone)
# ===========================================================================
def answer_unstructured(question: str) -> str:
    """Semantic path: retrieve prose from Pinecone and let the LLM explain. Reuses
    the base RAG chain verbatim — the grounding/abstention prompt comes with it."""
    from langchain_openai import ChatOpenAI
    from rag_langchain_pinecone import (
        CHAT_MODEL, build_rag_chain, get_embeddings, get_vectorstore, load_env,
    )

    load_env()
    vs = get_vectorstore(get_embeddings())
    chain = build_rag_chain(vs.as_retriever(search_kwargs={"k": 4}),
                            ChatOpenAI(model=CHAT_MODEL, temperature=0))
    return chain.invoke(question)


# ===========================================================================
# UNIFIED ENTRYPOINT — route, dispatch, (optionally) fuse
# ===========================================================================
def unified_answer(conn, question: str, allow_vector: bool = True) -> dict:
    engine = route(question)
    if engine == "structured":
        return {"question": question, "engine": "structured (SQL)",
                "answer": answer_structured(conn, question)}
    if not allow_vector:
        return {"question": question, "engine": "unstructured (skipped: no keys)",
                "answer": "[vector path unavailable offline]"}
    return {"question": question, "engine": "unstructured (vector RAG)",
            "answer": answer_unstructured(question)}


def main() -> None:
    conn = build_sqlite()

    # Is the vector path available? Probe keys once; degrade to structured-only.
    allow_vector = True
    try:
        from rag_langchain_pinecone import load_env
        load_env()
    except SystemExit:
        allow_vector = False
        print("(no API keys -> structured/SQL path only; semantic path skipped)\n")

    questions = [
        "What is the total revenue by region?",          # -> structured
        "How many orders were returned?",                # -> structured
        "How many orders by status?",                    # -> structured
        "Why was the wireless mouse order returned?",    # -> unstructured
        "What happened with the 27-inch monitor order?", # -> unstructured
    ]
    for q in questions:
        print("=" * 74)
        res = unified_answer(conn, q, allow_vector=allow_vector)
        print(f"Q: {res['question']}")
        print(f"   route -> {res['engine']}")
        print(f"A: {res['answer']}\n")


if __name__ == "__main__":
    main()
