"""
VECTOR DATABASES — Pinecone vs Weaviate vs pgvector (config + decision matrix).
Advanced-RAG module, file 3. The Pinecone section is LIVE (reuses the base
`customer-orders-rag` index). Weaviate + pgvector are clearly-labelled REFERENCE
config — importable, never executed against a server, so the file always runs
and prints the decision matrix.

WHY THIS FILE EXISTS FOR *YOUR* INTERVIEWS
------------------------------------------
You target enterprise / scalable GenAI roles (EY, PwC, Barclays, Accenture). The
question "why Pinecone and not X?" is a near-certainty. The wrong answer is a
feature list. The right answer is a DECISION with tradeoffs: pick the store from
the operational constraints — where your data already lives, who runs the
infra, tenancy model, scale, and cost — not from a benchmark. This file gives
you the three you must be able to compare and the knobs you must be able to tune.

THE 30-SECOND VERDICT (say this, then defend it)
  - pgvector   : you already run Postgres and want vectors NEXT TO your relational
                 data (one DB, real JOINs, transactions). Best default for most
                 enterprises. Scales to ~1-10M vectors comfortably; beyond that,
                 tuning HNSW and hardware becomes the job.
  - Pinecone   : you want a fully-managed, serverless vector DB and don't want to
                 run infra. Fastest to production, scales horizontally, strong
                 metadata filtering + namespaces for multi-tenancy. You pay for it.
  - Weaviate   : you want an open-source, self-hostable engine with first-class
                 HYBRID search (BM25 + dense built in), modules, and GraphQL, and
                 you're willing to operate it (or use their cloud).

THE KNOBS EVERY ONE OF THEM EXPOSES (understand these, not the vendor UI)
  - metric: cosine vs dot vs L2. Use COSINE for normalized text embeddings
    (OpenAI's are ~normalized), so cosine≈dot. Mismatch here silently wrecks
    recall — a classic bug.
  - HNSW graph params: M (edges per node; higher = better recall, more memory) and
    ef_construction (build-time search breadth; higher = better graph, slower
    build) and ef_search / ef (query-time breadth; the recall<->latency dial you
    tune at query time). This is the single most-asked ANN-tuning question.
  - metadata filtering: pre-filter (filter THEN search — exact, can be slow) vs
    post-filter (search THEN drop — fast, can under-return). Pinecone/Weaviate
    do metadata-aware ANN; pgvector filters in SQL WHERE alongside the ANN scan.
  - tenancy: namespaces (Pinecone) / tenants (Weaviate) / a tenant_id column +
    partial index (pgvector). This is how you isolate customers in a shared index.
  - serverless vs pod/self-host: managed elasticity + pay-per-use vs dedicated
    capacity you size and pay for continuously.

MAPS TO YOUR PROJECTS
  Universal Document Ingestor is multi-source and multi-tenant-shaped: namespaces
  (Pinecone) or a tenant_id filter (pgvector) is exactly how you'd keep one
  client's docs from leaking into another's retrieval. State the isolation model
  unprompted — it's the compliance-doc-QA concern MNCs care about.

RUN IT
    python practice/advanced_rag/03_vector_dbs.py            # matrix + live Pinecone
    python practice/advanced_rag/03_vector_dbs.py --no-live  # matrix only, offline
"""

from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


# ===========================================================================
# SECTION A — PINECONE (LIVE; reuses the base harness)
# ===========================================================================
def pinecone_reference_and_live(run_live: bool = True) -> None:
    """Pinecone serverless: create/describe/query. The base module already wraps
    index creation (ensure_index) and querying (get_vectorstore); here we show
    the raw client knobs an interviewer probes — spec, metric, metadata filter,
    namespaces — and then hit the live index."""
    print("--- PINECONE (serverless, managed) ---")
    print(
        """
    # CREATE (serverless — no capacity to size; you pay per usage)
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=...)
    pc.create_index(
        name="customer-orders-rag",
        dimension=1536,                 # MUST match the embedding model
        metric="cosine",                # normalized OpenAI embeds -> cosine
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    # Pinecone hides HNSW M/ef from you (managed) — you trade tuning for ops-free.

    # UPSERT with metadata (scalars/str-lists only) for filtered retrieval:
    idx.upsert([(id, vector, {"status": "Returned", "region": "US-East"})],
               namespace="tenant_acme")          # <-- multi-tenancy isolation

    # QUERY with a metadata pre-filter + namespace:
    idx.query(vector=q, top_k=6, include_metadata=True,
              filter={"status": {"$eq": "Returned"}}, namespace="tenant_acme")
    """
    )
    if not run_live:
        print("    [--no-live: skipped the live query]")
        return
    try:
        from rag_langchain_pinecone import INDEX_NAME, load_env

        load_env()
        from pinecone import Pinecone

        pc = Pinecone()
        stats = pc.Index(INDEX_NAME).describe_index_stats()
        desc = pc.describe_index(INDEX_NAME)
        print(f"    LIVE: index '{INDEX_NAME}' metric={desc.metric} dim={desc.dimension} "
              f"vectors={stats['total_vector_count']} namespaces={list(stats['namespaces'])}")
    except SystemExit:
        print("    [no API keys -> skipped live query]")
    except Exception as e:  # noqa: BLE001
        print(f"    [live query unavailable: {type(e).__name__}]")


# ===========================================================================
# SECTION B — WEAVIATE (REFERENCE ONLY — not executed; no server here)
# ===========================================================================
def weaviate_reference() -> None:
    """Weaviate: open-source, self-hostable, HYBRID search is first-class (you get
    BM25 + dense fusion built in — the alpha param blends them). Note the explicit
    vectorIndexConfig: this is where you SET the HNSW knobs Pinecone hides."""
    print("--- WEAVIATE (self-host / cloud; native hybrid) — reference ---")
    print(
        """
    import weaviate
    client = weaviate.connect_to_local()      # or connect_to_weaviate_cloud(...)

    client.collections.create(
        name="Order",
        vectorizer_config=Configure.Vectorizer.none(),   # bring your own vectors
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
            ef_construction=128,      # build-time breadth (recall vs build time)
            max_connections=32,       # == HNSW 'M' (recall vs memory)
            ef=64,                    # query-time breadth (recall vs latency)
        ),
        # multi-tenancy is a first-class toggle:
        multi_tenancy_config=Configure.multi_tenancy(enabled=True),
    )

    # NATIVE HYBRID — no hand-rolled RRF needed; alpha blends dense<->sparse:
    coll.query.hybrid(query="faulty mouse", alpha=0.5, limit=6,
                      filters=Filter.by_property("status").equal("Returned"))
    """
    )


# ===========================================================================
# SECTION C — PGVECTOR (REFERENCE ONLY — not executed; no Postgres here)
# ===========================================================================
def pgvector_reference() -> None:
    """pgvector: vectors AS A COLUMN in Postgres. The killer feature is that your
    embeddings sit next to relational data, so filtering/joining/tenancy are just
    SQL — no separate system to sync. This is often the right enterprise default."""
    print("--- PGVECTOR (vectors inside Postgres; SQL-native) — reference ---")
    print(
        """
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE TABLE orders (
        order_id   text PRIMARY KEY,
        tenant_id  text,                    -- multi-tenancy = a WHERE clause
        status     text,
        embedding  vector(1536)             -- dim MUST match the model
    );

    -- HNSW index with the SAME knobs, exposed in SQL:
    CREATE INDEX ON orders USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
    SET hnsw.ef_search = 40;                -- query-time recall<->latency dial

    -- QUERY = ANN + relational filter in one statement (the big win):
    SELECT order_id FROM orders
    WHERE tenant_id = 'acme' AND status = 'Returned'   -- exact SQL pre-filter
    ORDER BY embedding <=> :query_vec                  -- <=> is cosine distance
    LIMIT 6;
    """
    )


# ===========================================================================
# THE DECISION MATRIX (the artifact you reproduce on the whiteboard)
# ===========================================================================
def print_decision_matrix() -> None:
    rows = [
        ("Deployment",      "Managed serverless",   "Self-host / cloud",     "Postgres extension"),
        ("Ops burden",      "None (vendor runs it)","You run it",            "Your existing DBA"),
        ("Hybrid search",   "Sparse-vector index",  "Native (alpha blend)",  "Hand-roll / FTS+RRF"),
        ("HNSW tuning",     "Hidden (managed)",      "ef/M exposed",         "m/ef_* in SQL"),
        ("Metadata filter", "Pre-filter + $ops",    "Where filters",         "SQL WHERE (any op)"),
        ("Multi-tenancy",   "Namespaces",            "Tenants (native)",     "tenant_id column"),
        ("Joins to rel data","No (separate store)",  "No (separate store)",  "Yes (same DB)"),
        ("Scale sweet spot","10M–1B+ (elastic)",     "1M–100M (self-sized)", "10K–10M (then tune)"),
        ("Cost model",      "Pay-per-use",           "Infra you provision",  "Free ext; your PG"),
    ]
    w = (18, 22, 22, 20)
    header = ("Dimension", "Pinecone", "Weaviate", "pgvector")
    line = "  " + " | ".join(h.ljust(wi) for h, wi in zip(header, w))
    print("\n" + "=" * len(line))
    print("DECISION MATRIX")
    print("=" * len(line))
    print(line)
    print("  " + "-+-".join("-" * wi for wi in w))
    for r in rows:
        print("  " + " | ".join(c.ljust(wi) for c, wi in zip(r, w)))
    print(
        "\n  VERDICT for enterprise doc-QA (your target): default to pgvector if the\n"
        "  org already runs Postgres and wants vectors beside relational data; reach\n"
        "  for Pinecone when 'no infra / elastic scale' outweighs cost; pick Weaviate\n"
        "  when you want native hybrid + open-source and can operate it."
    )


def main() -> None:
    run_live = "--no-live" not in sys.argv
    pinecone_reference_and_live(run_live)
    weaviate_reference()
    pgvector_reference()
    print_decision_matrix()


if __name__ == "__main__":
    main()
