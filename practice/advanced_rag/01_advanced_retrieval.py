"""
ADVANCED RETRIEVAL — Multi-Query, HyDE, Query Routing, Hybrid (RRF).
Advanced-RAG module, file 1. BUILDS ON `practice/rag_langchain_pinecone.py`
(OpenAI text-embedding-3-small + Pinecone `customer-orders-rag` + gpt-4o-mini,
one Document per order). We reuse that harness's plumbing and only add the four
retrieval upgrades an interviewer for a mid-senior GenAI role expects you to name.

THE ONE FRAME TO CARRY IN (say this first, unprompted)
------------------------------------------------------
Plain top-k dense retrieval fails in three specific ways, and each advanced
technique fixes ONE of them. Naming the failure -> the fix is the senior move:

    failure mode                         the fix
    ------------                         -------
    query is phrased unlike the docs     MULTI-QUERY (rewrite the query N ways)
    query is a QUESTION, docs are        HyDE (embed a hypothetical ANSWER, not
      ANSWERS (asymmetric embedding)       the question — answer~answer is closer)
    one pipeline can't serve every       QUERY ROUTING (classify, then dispatch
      question shape                       to the right strategy/subset)
    dense misses exact tokens (IDs,      HYBRID (dense + BM25 sparse, fused with
      SKUs, names, rare words)            reciprocal-rank fusion)

MAPS TO YOUR PROJECTS
  - Synapse's ReAct retrieval agent already *chooses* how to retrieve — routing
    is that choice made explicit and cheap. Your source-grader (retrieval-failure
    handling) is the safety net UNDER these: when even hybrid+multi-query comes
    back thin, the grader detects it and re-queries or abstains.
  - Universal Document Ingestor is multi-source; hybrid + routing is exactly how
    you'd stop a keyword-heavy query (an invoice number) from silently missing.

WHY HAND-ROLL RRF AND BM25 (the interview signal)
  LangChain's MultiQueryRetriever is fine to *use*, but reciprocal-rank fusion
  and BM25 are three lines of math each. If you can't write RRF on a whiteboard
  you don't understand hybrid search — so we hand-roll both here. RRF needs no
  score calibration between the dense and sparse lists (it fuses RANKS, not raw
  scores), which is precisely why it's the default fusion in production.

RUN IT (needs the base index already ingested; keys from repo-root .env)
    python practice/advanced_rag/01_advanced_retrieval.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from math import log
from pathlib import Path

# The base harness lives one directory up. Add it to the path so we can reuse
# its Pinecone/OpenAI plumbing verbatim instead of re-implementing it.
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from rag_langchain_pinecone import (  # noqa: E402  (import after sys.path edit)
    CHAT_MODEL,
    DATA_CSV,
    get_embeddings,
    get_vectorstore,
    load_documents,
    load_env,
)


# ===========================================================================
# SHARED HELPERS — keying results by order_id so every strategy is comparable
# ===========================================================================
def doc_id(doc) -> str:
    """The stable key for fusion/dedupe. Every order is one Document with an
    `order_id` in metadata, so that's our identity — NOT page_content (which
    could differ trivially between the CSV and Pinecone's stored copy)."""
    return str(doc.metadata.get("order_id", doc.page_content[:40]))


def short(doc) -> str:
    return f"[{doc_id(doc)}] {doc.page_content[:70]}..."


# ===========================================================================
# STRATEGY 0 — DENSE (the baseline every upgrade is measured against)
# ===========================================================================
def dense_retrieve(vs, query: str, k: int = 4):
    """Vanilla similarity search over Pinecone. This is what plain RAG does."""
    return vs.similarity_search(query, k=k)


# ===========================================================================
# STRATEGY 1 — MULTI-QUERY (rewrite the query N ways, union the hits)
# ===========================================================================
# WHY: a user asks "faulty mouse?" but the doc says "three units stopped
# working ... possible batch defect". One embedding of one phrasing can miss it.
# Generating several paraphrases and UNIONING the retrieved sets raises recall:
# if ANY phrasing surfaces the right doc, it's in the candidate pool.
def generate_query_variants(llm, query: str, n: int = 3) -> list[str]:
    """Ask the LLM for n alternative phrasings. We keep the original too, so the
    union is never WORSE than the baseline — a cheap guarantee worth stating."""
    prompt = (
        f"Generate {n} alternative search queries that capture the same intent "
        f"as the user question, using different wording, synonyms, and phrasings. "
        f"Return ONE per line, no numbering.\n\nUser question: {query}"
    )
    text = llm.invoke(prompt).content
    variants = [line.strip("-• ").strip() for line in text.splitlines() if line.strip()]
    return [query, *variants][: n + 1]


def multi_query_retrieve(vs, llm, query: str, n: int = 3, k: int = 3):
    """Retrieve k for EACH variant, then dedupe by order_id (first occurrence
    wins its rank). Union of recall, at the cost of n+1 embedding+query calls —
    that latency/cost tradeoff is the thing to mention in the room."""
    seen: dict[str, object] = {}
    for variant in generate_query_variants(llm, query, n):
        for doc in vs.similarity_search(variant, k=k):
            seen.setdefault(doc_id(doc), doc)  # dedupe, keep first
    return list(seen.values())


def multi_query_retrieve_langchain(vs, llm, query: str, k: int = 3):
    """The batteries-included path: LangChain's MultiQueryRetriever does the
    generate->retrieve->union for you. Use this in real code; hand-roll above to
    PROVE you understand it. Same idea, less control over the dedupe/rank."""
    from langchain.retrievers.multi_query import MultiQueryRetriever

    mqr = MultiQueryRetriever.from_llm(retriever=vs.as_retriever(search_kwargs={"k": k}), llm=llm)
    return mqr.invoke(query)


# ===========================================================================
# STRATEGY 2 — HyDE (Hypothetical Document Embeddings)
# ===========================================================================
# WHY (the asymmetry point that makes you sound senior): a QUESTION and its
# ANSWER live in different regions of embedding space ("Why was it returned?"
# vs "three units failed within a week"). Embedding the question and matching it
# to answer-shaped docs is an asymmetric-search problem. HyDE sidesteps it:
# have the LLM HALLUCINATE a plausible answer, embed THAT, and search — answer
# text sits near answer text. The hallucination is fine here: we never show it to
# the user, we only use it as a better query vector. Cost: one extra LLM call.
def hyde_retrieve(vs, embeddings, llm, query: str, k: int = 4):
    """Generate a hypothetical answer -> embed it -> similarity_search_by_vector.
    Note we bypass vs.similarity_search (which would re-embed the raw query) and
    hand Pinecone the HYDE vector directly."""
    hypothetical = llm.invoke(
        f"Write a short, factual-sounding answer to this question as if it were "
        f"an entry in a customer-orders database. Do not hedge.\n\nQuestion: {query}"
    ).content
    vector = embeddings.embed_query(hypothetical)  # embed the ANSWER, not the question
    return vs.similarity_search_by_vector(vector, k=k)


# ===========================================================================
# STRATEGY 3 — QUERY ROUTING (classify the question, dispatch a strategy)
# ===========================================================================
# WHY: no single retriever is best for every question. An ID lookup wants exact
# match (hybrid/sparse); a fuzzy "why" wants HyDE; a broad browse wants
# multi-query. Routing picks the tool. We show BOTH a rules router (fast, free,
# auditable — what you ship first) and an LLM router (flexible, costs a call).
def route_rules(query: str) -> str:
    """Cheap deterministic router. In production you START here: it's free,
    testable, and covers the 80%. Escalate to an LLM router only for the tail."""
    q = query.lower()
    # strip punctuation so "order 1013?" -> the token "1013" is still detected
    tokens = _tokenize(q)
    if any(tok.isdigit() and len(tok) == 4 for tok in tokens):
        return "hybrid"          # a 4-digit order id -> exact-token match matters
    if q.startswith(("why", "how", "explain")):
        return "hyde"            # explanatory -> question/answer asymmetry
    if any(w in q for w in ("which", "all", "list", "every")):
        return "multi_query"     # broad/enumerating -> maximize recall
    return "dense"


def route_llm(llm, query: str) -> str:
    """LLM router: classify into one label from a fixed menu. The trick is a
    CONSTRAINED output — one word from a known set — so the router is parseable
    and can't wander. Defaults to dense on any surprise."""
    label = llm.invoke(
        "Classify the retrieval strategy for this question. Reply with EXACTLY one "
        "of: dense, multi_query, hyde, hybrid.\n"
        "- dense: simple factual lookup\n"
        "- multi_query: broad/enumerating ('which', 'all')\n"
        "- hyde: explanatory 'why/how'\n"
        "- hybrid: mentions an id/code/exact token\n\n"
        f"Question: {query}"
    ).content.strip().lower()
    return label if label in {"dense", "multi_query", "hyde", "hybrid"} else "dense"


def routed_retrieve(vs, embeddings, llm, query: str, k: int = 4):
    """Dispatch. Returns (strategy_name, docs) so the demo can show the choice."""
    strategy = route_rules(query)
    if strategy == "multi_query":
        return strategy, multi_query_retrieve(vs, llm, query, n=3, k=k)
    if strategy == "hyde":
        return strategy, hyde_retrieve(vs, embeddings, llm, query, k=k)
    if strategy == "hybrid":
        return strategy, hybrid_retrieve(vs, query, k=k)
    return strategy, dense_retrieve(vs, query, k=k)


# ===========================================================================
# STRATEGY 4 — HYBRID (dense Pinecone + in-memory BM25, fused with RRF)
# ===========================================================================
# WHY: dense embeddings are semantically strong but weak on EXACT tokens — order
# ids, SKUs, rare product names, negations. BM25 (lexical) nails those and misses
# semantics. Fuse them and you get both. This is the single biggest real-world
# recall win, which is why every serious vector DB now ships hybrid.
#
# We hand-roll BM25 over the 16 orders so there's no server and no `rank_bm25`
# dependency — and so you can WRITE the formula in an interview.
class BM25:
    """Textbook Okapi BM25. Scores a query against a fixed corpus of tokenized
    docs. k1 controls term-frequency saturation; b controls length normalization.

    score(D,Q) = sum_t IDF(t) * ( f(t,D)*(k1+1) ) / ( f(t,D) + k1*(1-b+b*|D|/avgdl) )
    IDF(t) = log( (N - n_t + 0.5) / (n_t + 0.5) + 1 )
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus_tokens
        self.N = len(corpus_tokens)
        self.avgdl = sum(len(d) for d in corpus_tokens) / max(self.N, 1)
        # document frequency n_t = number of docs containing term t
        self.df: dict[str, int] = defaultdict(int)
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] += 1
        # term frequency per doc, precomputed
        self.tf: list[dict[str, int]] = []
        for doc in corpus_tokens:
            counts: dict[str, int] = defaultdict(int)
            for term in doc:
                counts[term] += 1
            self.tf.append(counts)

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query_tokens: list[str], index: int) -> float:
        counts, dl, s = self.tf[index], len(self.corpus[index]), 0.0
        for term in query_tokens:
            f = counts.get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s += self.idf(term) * (f * (self.k1 + 1)) / denom
        return s

    def top_k(self, query_tokens: list[str], k: int) -> list[int]:
        scored = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scored = [(i, sc) for i, sc in scored if sc > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scored[:k]]


def _tokenize(text: str) -> list[str]:
    """Lowercase alnum tokens. Deliberately simple — good enough to show BM25;
    a real deploy would add stemming/stopwords, but that's not the lesson."""
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split()]


def reciprocal_rank_fusion(ranked_lists: list[list], key_fn, k: int = 60):
    """RRF: fuse multiple RANKED lists without needing comparable scores.

        rrf_score(d) = sum over lists of  1 / (k + rank_in_list(d))

    rank is 0-based here. The constant k (~60 in the paper) damps the influence
    of top ranks so no single list dominates. Because it uses only ranks, dense
    cosine similarities and BM25 scores — which live on totally different scales —
    fuse cleanly with ZERO calibration. That last sentence is the whole reason
    RRF is the production default; say it out loud.

    Returns items ordered by fused score. `key_fn` maps an item to its identity,
    and we keep the first-seen item object per key for the output.
    """
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, object] = {}
    for lst in ranked_lists:
        for rank, item in enumerate(lst):
            key = key_fn(item)
            scores[key] += 1.0 / (k + rank)
            items.setdefault(key, item)
    ordered_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [items[kk] for kk in ordered_keys]


def hybrid_retrieve(vs, query: str, k: int = 4, bm25_docs=None):
    """Dense (Pinecone) + sparse (BM25 over the 16 orders), fused with RRF.

    bm25_docs is the in-memory corpus of order Documents; we rebuild BM25 each
    call for teaching clarity — in prod you'd build the index ONCE and reuse it
    (BM25 over 16 rows is free; over millions you'd use a real sparse index)."""
    if bm25_docs is None:
        bm25_docs = load_documents(DATA_CSV)

    # dense list (ranked by cosine)
    dense = vs.similarity_search(query, k=k)

    # sparse list (ranked by BM25)
    corpus_tokens = [_tokenize(d.page_content) for d in bm25_docs]
    bm25 = BM25(corpus_tokens)
    sparse_idx = bm25.top_k(_tokenize(query), k=k)
    sparse = [bm25_docs[i] for i in sparse_idx]

    # fuse ranks — no score calibration needed
    return reciprocal_rank_fusion([dense, sparse], key_fn=doc_id)[:k]


# ===========================================================================
# DEMO
# ===========================================================================
def main() -> None:
    load_env()
    from langchain_openai import ChatOpenAI

    embeddings = get_embeddings()
    vs = get_vectorstore(embeddings)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    print("=" * 74)
    print("STRATEGY 0 — DENSE baseline")
    for d in dense_retrieve(vs, "faulty wireless mouse", k=3):
        print("  ", short(d))

    print("\n" + "=" * 74)
    print("STRATEGY 1 — MULTI-QUERY (hand-rolled union)")
    variants = generate_query_variants(llm, "faulty wireless mouse", n=3)
    print("   variants:", variants)
    for d in multi_query_retrieve(vs, llm, "faulty wireless mouse", n=3, k=2):
        print("  ", short(d))

    print("\n" + "=" * 74)
    print("STRATEGY 2 — HyDE (embed a hypothetical answer)")
    for d in hyde_retrieve(vs, embeddings, llm, "Why did a customer send back their order?", k=3):
        print("  ", short(d))

    print("\n" + "=" * 74)
    print("STRATEGY 3 — QUERY ROUTING (rules vs llm)")
    for q in ["What is order 1013?", "Why was an order returned?", "Which orders were cancelled?"]:
        print(f"   rules-> {route_rules(q):<12} llm-> {route_llm(llm, q):<12} | {q}")
    strat, docs = routed_retrieve(vs, embeddings, llm, "Which orders were cancelled?", k=3)
    print(f"   routed to '{strat}':")
    for d in docs:
        print("  ", short(d))

    print("\n" + "=" * 74)
    print("STRATEGY 4 — HYBRID (dense + BM25, RRF-fused)")
    for d in hybrid_retrieve(vs, "order 1011 mouse defect", k=4):
        print("  ", short(d))


if __name__ == "__main__":
    main()
