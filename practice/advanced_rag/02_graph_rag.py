"""
GRAPH RAG — knowledge-graph traversal UNIONed with vector retrieval.
Advanced-RAG module, file 2. Offline-runnable (builds the graph from the CSV);
the vector-union half reuses the base Pinecone index when keys are present.

THE QUESTION AN INTERVIEWER IS REALLY ASKING (answer this before any code)
-------------------------------------------------------------------------
"When does Graph RAG beat vector RAG?"  ->  When the answer requires MULTI-HOP
RELATIONAL reasoning that no single chunk contains. Vector RAG retrieves the
top-k chunks most SIMILAR to the query and stuffs them in the prompt. That's
perfect for "why was order 1011 returned?" (the answer lives in one chunk) and
terrible for "which customers in APAC-North bought monitors, and what else did
they order?" — because:
  - the answer is a JOIN across many rows, not a similarity match;
  - the relevant rows aren't textually similar to each other or to the query;
  - top-k silently truncates: ask for k=4 and you miss the 5th matching customer.
A graph makes the RELATIONSHIPS first-class. Traversal is EXHAUSTIVE and exact
over the edges you modelled — no similarity threshold, no k cutoff. That is the
recall property vector search cannot give you on relational questions.

THE SENIOR NUANCE (don't oversell graphs)
  Graph RAG is not a replacement — it's the other half. Graphs win on structure
  and multi-hop; vectors win on fuzzy semantics and unstructured prose. Real
  systems do BOTH and union: traverse the graph for the exact related entities,
  then vector-retrieve the prose that explains them. That hybrid is what we build.

MAPS TO YOUR PROJECTS
  - Synapse's parallel subgraphs already reason over structured state; a KG is
    the same instinct applied to the corpus. Your source-grader would catch the
    case where traversal returns nothing (relationship not modelled) and fall
    back to vector search — retrieval-failure handling again.
  - This is also the honest answer to "how would you handle aggregation-style
    questions?": model the entities, don't ask a vector store to do a JOIN.

We use networkx (already installed). If it weren't, a dict-of-adjacency-lists
does the same job — the concept is edges, not the library.

RUN IT (graph half is fully offline; vector union needs the base index)
    python practice/advanced_rag/02_graph_rag.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import networkx as nx

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from rag_langchain_pinecone import DATA_CSV  # noqa: E402


# ===========================================================================
# BUILD THE KNOWLEDGE GRAPH
# ===========================================================================
# NODE TYPES: customer, order, product, region
# EDGE TYPES: placed_by (order->customer), contains_product (order->product),
#             in_region (order->region)
# Modelling choice worth stating: the ORDER is the hub node. Everything else
# hangs off it, so a customer's products are reachable as customer<-order->product
# (a 2-hop path). Choosing the hub is the schema decision that makes traversals
# short — the equivalent of choosing a good primary key.
def build_order_graph(csv_path: Path = DATA_CSV) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            oid = f"order:{row['order_id']}"
            cust = f"customer:{row['customer_name']}"
            prod = f"product:{row['product']}"
            region = f"region:{row['region']}"

            # typed nodes (keep the row on the order node for cheap lookups)
            g.add_node(oid, kind="order", **row)
            g.add_node(cust, kind="customer", name=row["customer_name"])
            g.add_node(prod, kind="product", name=row["product"], category=row["category"])
            g.add_node(region, kind="region", name=row["region"])

            # typed edges out of the order hub
            g.add_edge(oid, cust, rel="placed_by")
            g.add_edge(oid, prod, rel="contains_product")
            g.add_edge(oid, region, rel="in_region")
    return g


# ===========================================================================
# TRAVERSAL QUERIES — the multi-hop questions vector RAG cannot answer well
# ===========================================================================
def _orders_with(g: nx.MultiDiGraph, kind_prefix: str, name: str) -> set[str]:
    """All order nodes linked to a given region/product/customer node. We walk
    the graph's incoming edges to the entity node — that's the JOIN, done as a
    traversal instead of a SQL WHERE."""
    target = f"{kind_prefix}:{name}"
    if target not in g:
        return set()
    return {u for u, _, _ in g.in_edges(target, keys=True) if g.nodes[u]["kind"] == "order"}


def customers_in_region_buying_product(g, region: str, product: str) -> list[str]:
    """THE flagship multi-hop query: customer <- placed_by <- order -> contains
    product AND order -> in region. Intersect two order-sets, then hop to the
    customers. Exhaustive by construction — no top-k truncation."""
    orders = _orders_with(g, "region", region) & _orders_with(g, "product", product)
    customers = set()
    for oid in orders:
        for _, v, data in g.out_edges(oid, data=True):
            if data["rel"] == "placed_by":
                customers.add(g.nodes[v]["name"])
    return sorted(customers)


def other_orders_by_customer(g, customer_name: str) -> list[dict]:
    """2-hop: from a customer, back to all their orders (the 'what else did they
    buy' follow-up). This is the relationship expansion that makes graphs shine."""
    cust = f"customer:{customer_name}"
    if cust not in g:
        return []
    out = []
    for oid in _orders_with(g, "customer", customer_name):
        n = g.nodes[oid]
        out.append({"order_id": n["order_id"], "product": n["product"], "status": n["status"]})
    return sorted(out, key=lambda r: r["order_id"])


def _orders_with_customer(g, name):  # symmetry helper used above
    return _orders_with(g, "customer", name)


# ===========================================================================
# GRAPH RAG = graph traversal  UNION  vector retrieval
# ===========================================================================
# The traversal gives you the EXACT related entities; the vector store gives you
# the PROSE that explains them (the notes field). Union = structure + semantics.
def graph_rag_answer(g, question: str, region: str, product: str, vs=None, k: int = 4) -> dict:
    """Answer a relational question by:
      1) traversing the graph for the exact matching orders/customers (recall),
      2) OPTIONALLY vector-retrieving the explanatory prose for those orders.
    Returns the structured result plus any retrieved context. In a full pipeline
    you'd feed both into the LLM prompt; here we return them so the mechanics
    are visible."""
    customers = customers_in_region_buying_product(g, region, product)
    orders = sorted(_orders_with(g, "region", region) & _orders_with(g, "product", product))

    context_docs = []
    if vs is not None and orders:
        # vector-retrieve prose for the traversal result. Union, not replace.
        hits = vs.similarity_search(question, k=k)
        # keep only hits whose order_id is in the traversal set -> grounded context
        wanted = {o.split(":")[1] for o in orders}
        context_docs = [d for d in hits if str(d.metadata.get("order_id")) in wanted]

    return {
        "question": question,
        "matched_orders": [o.split(":")[1] for o in orders],
        "customers": customers,
        "vector_context": [d.page_content[:80] + "..." for d in context_docs],
    }


# ===========================================================================
# DEMO
# ===========================================================================
def main() -> None:
    g = build_order_graph()
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges "
          f"(customers/orders/products/regions).")

    print("\n" + "=" * 74)
    print("MULTI-HOP Q1: which customers in APAC-North bought 27-inch Monitors?")
    print("   ->", customers_in_region_buying_product(g, "APAC-North", "27-inch Monitor"))
    print("   (vector top-k would truncate/miss; traversal is exhaustive)")

    print("\n" + "=" * 74)
    print("MULTI-HOP Q2: everything Kenji Tanaka ordered")
    for r in other_orders_by_customer(g, "Kenji Tanaka"):
        print("  ", r)

    print("\n" + "=" * 74)
    print("GRAPH RAG (traversal UNION vector prose)")
    # Vector union only if keys/index are available; degrade gracefully offline.
    vs = None
    try:
        from rag_langchain_pinecone import get_embeddings, get_vectorstore, load_env

        load_env()
        vs = get_vectorstore(get_embeddings())
    except SystemExit:
        print("   (no API keys -> running graph-only; vector union skipped)")
    except Exception as e:  # noqa: BLE001
        print(f"   (vector store unavailable: {type(e).__name__}; graph-only)")

    result = graph_rag_answer(
        g,
        "Which APAC-North customers bought 27-inch monitors and what were the orders?",
        region="APAC-North",
        product="27-inch Monitor",
        vs=vs,
    )
    for key, val in result.items():
        print(f"   {key}: {val}")


if __name__ == "__main__":
    main()
