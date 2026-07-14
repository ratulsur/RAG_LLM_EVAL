"""
RAG WITH LANGCHAIN + PINECONE + OpenAI — runnable practice harness (Part 2b).

This is the *runnable* sibling of `rag_with_langchain_semantic.py`. That file is
study material (stub embeddings, FAISS, no network). This one actually runs:

    OpenAIEmbeddings  ->  Pinecone (serverless)  ->  ChatOpenAI  via an LCEL chain

over a sample customer-orders CSV, so you can practice real ingest + query.

WHAT MAPS TO WHAT (say this in the room)
    hand-written piece          this file
    ----------------            ---------
    Embedder / .embed           OpenAIEmbeddings(model="text-embedding-3-small")
    FAISSStore (in-memory)      PineconeVectorStore (managed, serverless)
    parallel id->text list      Pinecone's own vector<->metadata store
    RAGPipeline.answer          retriever | prompt | ChatOpenAI | StrOutputParser

WHY PINECONE OVER FAISS (the senior point)
  FAISS is an in-process library: the index dies with your process and lives in
  one machine's RAM. Pinecone is a managed vector DATABASE: persistent, remote,
  horizontally scaled, with metadata filtering and namespaces for multi-tenancy.
  You reach for FAISS in a notebook / single-node prototype; you reach for
  Pinecone (or pgvector, Weaviate, ...) when the index must survive restarts,
  scale past RAM, and be queried by many services. Same LangChain VectorStore
  interface, so the pipeline code barely changes — that's the whole point of
  programming against the interface.

CHUNKING CHOICE FOR THIS DATA (an interview trap — get it right)
  These are TABULAR order records. The correct default is ONE Document PER ROW:
  each order is an atomic retrievable fact, and row-level docs give clean
  per-order citations ("order 1013"). Semantic chunking shines on LONG PROSE
  where topic boundaries are fuzzy — it is the WRONG tool for short structured
  rows, where it can split one order across chunks or merge two orders and wreck
  precision. So this harness defaults to --chunk row, and exposes
  --chunk semantic only to *demonstrate* the splitter on the concatenated corpus.
  Stating that tradeoff unprompted is the senior signal.

RUN IT
    # one-time (or after editing the CSV): embed + upsert into Pinecone
    python practice/rag_langchain_pinecone.py ingest

    # ask questions grounded in the orders
    python practice/rag_langchain_pinecone.py query "What happened with order 1011?"
    python practice/rag_langchain_pinecone.py query "Which orders were returned and why?"

    # ingest + a couple of demo queries in one go
    python practice/rag_langchain_pinecone.py demo

    # evaluate the pipeline with DeepEval (Faithfulness / Answer Relevancy /
    # Contextual Precision & Recall); add --gate for CI (non-zero exit on fail)
    python practice/rag_langchain_pinecone.py eval
    python practice/rag_langchain_pinecone.py eval --threshold 0.8 --gate

Requires OPENAI_API_KEY and PINECONE_API_KEY in .env (already present).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# --- Config -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT.parent / ".env"
DATA_CSV = ROOT / "data" / "customer_orders.csv"

INDEX_NAME = "customer-orders-rag"
EMBED_MODEL = "text-embedding-3-small"   # 1536 dims
EMBED_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"            # serverless free-tier region


# --- Env --------------------------------------------------------------------
def load_env() -> None:
    """Load .env and fail loudly if a required key is missing. python-dotenv
    strips whitespace around keys, so `PINECONE_API_KEY = ...` (with the space
    in the .env) still resolves correctly."""
    load_dotenv(ENV_PATH)
    import os

    missing = [k for k in ("OPENAI_API_KEY", "PINECONE_API_KEY") if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing required env var(s): {', '.join(missing)} (check {ENV_PATH})")


# --- Data -> Documents ------------------------------------------------------
def order_to_text(row: dict) -> str:
    """Render one CSV row as a natural-language paragraph. Embeddings model prose
    far better than raw comma-separated fields, so we verbalize the record."""
    return (
        f"Order {row['order_id']} was placed on {row['order_date']} by "
        f"{row['customer_name']} in region {row['region']}. "
        f"It is for {row['quantity']} x {row['product']} ({row['category']}) at "
        f"${row['unit_price']} each, totalling ${row['total']}. "
        f"Current status: {row['status']}. "
        f"Notes: {row['notes']}"
    )


def load_documents(csv_path: Path):
    """CSV rows -> LangChain Documents (one per order). Metadata is kept simple
    (str / number) because Pinecone only accepts scalar or list-of-string
    metadata, and we want order_id/status/region available for citations and
    future metadata-filtered retrieval."""
    from langchain_core.documents import Document

    docs = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append(
                Document(
                    page_content=order_to_text(row),
                    metadata={
                        "order_id": row["order_id"],
                        "customer_name": row["customer_name"],
                        "product": row["product"],
                        "status": row["status"],
                        "region": row["region"],
                        "total": float(row["total"]),
                    },
                )
            )
    return docs


def semantic_chunk(documents, embeddings):
    """DEMO PATH ONLY. Concatenate all orders and let SemanticChunker place
    breakpoints. Included so you can practice the semantic splitter; NOT the
    recommended path for tabular records (see module docstring)."""
    from langchain_experimental.text_splitter import SemanticChunker

    corpus = "\n".join(d.page_content for d in documents)
    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90,
    )
    chunks = splitter.create_documents([corpus])
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    return chunks


# --- OpenAI + Pinecone plumbing --------------------------------------------
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBED_MODEL)


def ensure_index() -> None:
    """Create the Pinecone serverless index if it doesn't exist, and block until
    it is ready. Idempotent — safe to call before every ingest."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone()  # reads PINECONE_API_KEY from env
    existing = {idx["name"] for idx in pc.list_indexes()}
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{INDEX_NAME}' (dim={EMBED_DIM}, cosine)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)
        print("Index ready.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


def get_vectorstore(embeddings):
    """Bind LangChain to the existing Pinecone index. PineconeVectorStore reads
    PINECONE_API_KEY from env and speaks the standard VectorStore interface."""
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)


# --- LCEL chain -------------------------------------------------------------
def build_rag_chain(retriever, llm):
    """retriever | prompt | llm | parser. The prompt carries the same grounding
    + abstention discipline as the from-scratch pipeline: answer ONLY from
    context, else say you don't know, and cite the order id."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a customer-orders assistant. Answer using ONLY the "
                "context below. If the answer isn't in it, reply exactly: "
                "'I don't know based on the order records.' "
                "Cite the relevant order id(s) in brackets, e.g. [order 1011].\n\n"
                "CONTEXT:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs) -> str:
        return "\n\n".join(
            f"[order {d.metadata.get('order_id', '?')}] {d.page_content}" for d in docs
        )

    setup = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    return setup | prompt | llm | StrOutputParser()


# --- Commands ---------------------------------------------------------------
def cmd_ingest(chunk_mode: str) -> None:
    load_env()
    embeddings = get_embeddings()
    docs = load_documents(DATA_CSV)
    if chunk_mode == "semantic":
        docs = semantic_chunk(docs, embeddings)
        print(f"Semantic chunking produced {len(docs)} chunks from the corpus.")
    else:
        print(f"Loaded {len(docs)} order documents (one per row).")

    ensure_index()
    vs = get_vectorstore(embeddings)
    vs.add_documents(docs)
    print(f"Upserted {len(docs)} vectors into '{INDEX_NAME}'.")


def parse_filter(spec: str | None) -> dict | None:
    """'status=Returned' -> {'status': {'$eq': 'Returned'}} (Pinecone filter).

    This is the fix for aggregation/filter questions ('which orders were
    returned?'). Pure dense top-k retrieval only surfaces the nearest few
    vectors, so it silently misses matching rows that rank below k — a recall
    failure, not a code bug. A metadata filter makes retrieval EXHAUSTIVE over
    the matching subset, so combined with a large enough k the LLM sees every
    returned order and can answer completely."""
    if not spec:
        return None
    if "=" not in spec:
        sys.exit("--filter must look like key=value, e.g. status=Returned")
    key, value = spec.split("=", 1)
    return {key.strip(): {"$eq": value.strip()}}


def cmd_query(question: str, top_k: int, meta_filter: str | None = None) -> None:
    load_env()
    from langchain_openai import ChatOpenAI

    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    search_kwargs: dict = {"k": top_k}
    flt = parse_filter(meta_filter)
    if flt:
        search_kwargs["filter"] = flt
    retriever = vs.as_retriever(search_kwargs=search_kwargs)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = build_rag_chain(retriever, llm)

    print(f"\nQ: {question}" + (f"   [filter: {meta_filter}]" if flt else ""))
    print(f"A: {chain.invoke(question)}\n")


def cmd_demo(chunk_mode: str) -> None:
    cmd_ingest(chunk_mode)
    # Plain semantic query.
    cmd_query("What happened with order 1011?", top_k=4)
    # Aggregation query WITHOUT a filter — dense top-k misses some returns.
    cmd_query("Which orders were returned, and why?", top_k=6)
    # Same question WITH a metadata filter — now retrieval is exhaustive.
    cmd_query("Which orders were returned, and why?", top_k=16, meta_filter="status=Returned")


def cmd_eval(top_k: int, threshold: float, gate: bool) -> None:
    """Evaluate THIS pipeline with DeepEval — the default practice harness.

    Runs the real pipeline over the gold set, then judges each answer on
    DeepEval's threshold metrics (Faithfulness, Answer Relevancy, Contextual
    Precision, Contextual Recall). DeepEval's score+threshold+pass/fail model is
    a CI quality gate: `--gate` makes a failing check exit non-zero (for CI),
    while the default just reports (for practice)."""
    load_env()
    from rag_eval_ragas_deepeval import run_deepeval_over_orders

    print(f"DeepEval over the pipeline (top_k={top_k}, threshold={threshold})...\n")
    _samples, report = run_deepeval_over_orders(threshold=threshold, top_k=top_k)

    total = passed = 0
    per_metric: dict[str, list[int]] = {}
    for row in report:
        print(f"Q: {row['question']}")
        for name, res in row.items():
            if name == "question":
                continue
            ok = bool(res["passed"])
            total += 1
            passed += int(ok)
            agg = per_metric.setdefault(name, [0, 0])
            agg[0] += int(ok)
            agg[1] += 1
            print(f"   {'PASS' if ok else 'FAIL'}  {name:<26} {res['score']:.3f}")
        print()

    print("Per-metric pass rate:")
    for name, (p, n) in per_metric.items():
        print(f"   {name:<26} {p}/{n}")
    print(f"\nOverall: {passed}/{total} checks passed ({passed / max(total, 1):.0%}).")
    if gate and passed < total:
        sys.exit(1)  # CI gate: fail the build on any metric below threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG over customer orders (Pinecone + OpenAI).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="embed the CSV and upsert into Pinecone")
    p_ing.add_argument("--chunk", choices=["row", "semantic"], default="row")

    p_q = sub.add_parser("query", help="ask a question grounded in the orders")
    p_q.add_argument("question")
    p_q.add_argument("--top-k", type=int, default=6)
    p_q.add_argument(
        "--filter",
        dest="meta_filter",
        default=None,
        help="metadata filter for exhaustive retrieval, e.g. status=Returned",
    )

    p_d = sub.add_parser("demo", help="ingest + a few demo questions")
    p_d.add_argument("--chunk", choices=["row", "semantic"], default="row")

    p_e = sub.add_parser("eval", help="evaluate the pipeline with DeepEval")
    p_e.add_argument("--top-k", type=int, default=6)
    p_e.add_argument("--threshold", type=float, default=0.7)
    p_e.add_argument("--gate", action="store_true", help="exit non-zero on any metric below threshold (CI mode)")

    args = parser.parse_args()
    if args.command == "ingest":
        cmd_ingest(args.chunk)
    elif args.command == "query":
        cmd_query(args.question, args.top_k, args.meta_filter)
    elif args.command == "demo":
        cmd_demo(args.chunk)
    elif args.command == "eval":
        cmd_eval(args.top_k, args.threshold, args.gate)


if __name__ == "__main__":
    main()
