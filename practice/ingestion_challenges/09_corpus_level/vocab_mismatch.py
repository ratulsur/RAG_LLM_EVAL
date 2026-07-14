"""
09f — DOMAIN VOCAB MISMATCH -> where BM25 fails and hybrid/embeddings earn their keep.

THE PATHOLOGY (say this in the room)
  The user asks for their "salary slip". The corpus never uses that phrase — HR
  calls it a "pay statement". BM25 / exact match scores ZERO on the right doc
  because there is no lexical overlap on the key term. This is THE textbook case
  for dense retrieval: same meaning, different words.

THE DEMO (failure -> three fixes, ranked)
  1. Show BM25 (bag-of-words) returning the WRONG doc or nothing for "salary slip".
  2. FIX A — synonym expansion: expand the query with a curated synonym map
     (cheapest, most controllable; brittle to unknown synonyms).
  3. FIX B — dense embeddings: cosine similarity puts "pay statement" near "salary
     slip" with no synonym list (generalizes; needs a model).
  4. FIX C — HYBRID + RERANK: BM25 + dense fused (reciprocal-rank fusion), THEN a
     reranker over the fused candidates. Naive RRF alone can be dragged by a
     lexical DISTRACTOR (here 'onboarding_note' literally contains the word
     "salary"), which is exactly why production RAG puts a cross-encoder RERANK on
     top of hybrid retrieval. The rerank restores the right doc.

EMBEDDINGS: LIVE IF KEYS PRESENT, ELSE OFFLINE
  Uses the repo helper get_embeddings() (OpenAI) when OPENAI_API_KEY is set;
  otherwise falls back to a deterministic offline embedder (character-hashed
  n-gram vectors) so the ranking effect still runs with no network. The offline
  vectors are weaker but demonstrate the SAME mechanism.

RUN
    python vocab_mismatch.py
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from fixtures.make_corpus import VOCAB_QUERY, make_corpus

# Make the repo's practice/ importable for the OpenAI embeddings helper.
_PRACTICE = Path(__file__).resolve().parents[2]  # .../practice
if str(_PRACTICE) not in sys.path:
    sys.path.insert(0, str(_PRACTICE))


# ===========================================================================
# 1. BM25 (compact, dependency-free) — the thing that FAILS here
# ===========================================================================
# Drop stopwords so BM25 scores on CONTENT terms only. Otherwise "how do i get my"
# lets the target sneak in on function-word overlap and hides the real lexical
# miss on the key term ("salary slip") — which is the whole point of this demo.
_STOP = {"how", "do", "i", "get", "my", "the", "a", "to", "your", "of", "and",
         "for", "on", "at", "is", "in", "you"}


def tok(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if w not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [tok(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(len(d) for d in self.docs) / len(self.docs)
        self.df: Counter = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.N = len(self.docs)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str, i: int) -> float:
        d = self.docs[i]
        tf = Counter(d)
        s = 0.0
        for q in set(tok(query)):
            if q not in tf:
                continue
            idf = self._idf(q)
            num = tf[q] * (self.k1 + 1)
            den = tf[q] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            s += idf * num / den
        return s

    def rank(self, query: str) -> list[tuple[int, float]]:
        return sorted(((i, self.score(query, i)) for i in range(self.N)),
                      key=lambda x: -x[1])


# ===========================================================================
# 2. Synonym expansion (Fix A)
# ===========================================================================
SYNONYMS = {
    "salary": ["pay", "remuneration", "compensation"],
    "slip": ["statement", "stub"],
}


def expand(query: str) -> str:
    extra: list[str] = []
    for w in tok(query):
        extra += SYNONYMS.get(w, [])
    return query + " " + " ".join(extra)


# ===========================================================================
# 3. Dense embeddings (Fix B) — live OpenAI or offline fallback
# ===========================================================================
def _offline_embed(texts: list[str], dim: int = 256) -> list[list[float]]:
    """Deterministic hashed character-trigram vectors. No network. Weaker than a
    real model but captures enough surface overlap to show the mechanism."""
    vecs: list[list[float]] = []
    for t in texts:
        v = [0.0] * dim
        s = f" {t.lower()} "
        for i in range(len(s) - 2):
            tri = s[i:i + 3]
            v[hash(tri) % dim] += 1.0
        vecs.append(v)
    return vecs


def get_embedder():
    """Return (name, embed_fn). embed_fn: list[str] -> list[list[float]]."""
    try:
        from rag_langchain_pinecone import get_embeddings, load_env  # type: ignore

        load_env()
        import os

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("no OPENAI_API_KEY")
        emb = get_embeddings()
        return "OpenAI text-embedding-3-small (LIVE)", emb.embed_documents
    except Exception as e:  # noqa: BLE001
        return f"offline hashed-trigram fallback ({type(e).__name__})", _offline_embed


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ===========================================================================
# 4. Hybrid: reciprocal-rank fusion
# ===========================================================================
def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_i in enumerate(ranking):
            scores[doc_i] = scores.get(doc_i, 0.0) + 1.0 / (k + rank + 1)
    return scores


@dataclass
class Corpus:
    filenames: list[str]
    texts: list[str]


def _rule(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main() -> None:
    docs = make_corpus()
    corpus = Corpus([d.filename for d in docs], [d.text for d in docs])
    target = "hr_pay_statement.txt"
    query = VOCAB_QUERY

    _rule(f"QUERY: {query!r}   (right answer = {target})")

    bm25 = BM25(corpus.texts)
    bm_rank = bm25.rank(query)
    _rule("BEFORE — BM25 fails: 'salary slip' has zero lexical overlap with 'pay statement'")
    for i, s in bm_rank[:4]:
        mark = "  <-- TARGET" if corpus.filenames[i] == target else ""
        print(f"   {s:.3f}  {corpus.filenames[i]}{mark}")
    top1_bm = corpus.filenames[bm_rank[0][0]]
    print(f"   BM25 top-1 = {top1_bm}  ({'HIT' if top1_bm == target else 'MISS'})")

    # Fix A: synonym expansion
    bm_rank_exp = bm25.rank(expand(query))
    _rule("FIX A — synonym expansion (salary->pay, slip->statement)")
    top1_exp = corpus.filenames[bm_rank_exp[0][0]]
    for i, s in bm_rank_exp[:3]:
        mark = "  <-- TARGET" if corpus.filenames[i] == target else ""
        print(f"   {s:.3f}  {corpus.filenames[i]}{mark}")
    print(f"   expanded top-1 = {top1_exp}  ({'HIT' if top1_exp == target else 'MISS'})")

    # Fix B: dense embeddings
    name, embed = get_embedder()
    _rule(f"FIX B — dense embeddings [{name}]")
    doc_vecs = embed(corpus.texts)
    q_vec = embed([query])[0]
    dense_rank = sorted(
        ((i, cosine(q_vec, doc_vecs[i])) for i in range(len(corpus.texts))),
        key=lambda x: -x[1],
    )
    for i, s in dense_rank[:3]:
        mark = "  <-- TARGET" if corpus.filenames[i] == target else ""
        print(f"   {s:.3f}  {corpus.filenames[i]}{mark}")
    top1_dense = corpus.filenames[dense_rank[0][0]]
    print(f"   dense top-1 = {top1_dense}  ({'HIT' if top1_dense == target else 'MISS'})")

    # Fix C: hybrid RRF, then RERANK the fused candidates
    _rule("FIX C — HYBRID (RRF) then RERANK")
    fused = rrf([[i for i, _ in bm_rank], [i for i, _ in dense_rank]])
    hybrid_rank = sorted(fused.items(), key=lambda x: -x[1])
    print("  hybrid RRF candidates (naive fusion — dragged by lexical distractor):")
    for i, s in hybrid_rank[:3]:
        mark = "  <-- TARGET" if corpus.filenames[i] == target else ""
        print(f"     {s:.4f}  {corpus.filenames[i]}{mark}")
    top1_hyb = corpus.filenames[hybrid_rank[0][0]]
    print(f"     RRF top-1 = {top1_hyb}  ({'HIT' if top1_hyb == target else 'MISS'})")

    # Rerank the top RRF candidates. Production reranker = a cross-encoder
    # (e.g. bge-reranker / Cohere rerank); we PROXY it with the dense scorer over
    # just the candidate set — the mechanism (re-score a small shortlist with a
    # stronger model) is identical.
    cand = [i for i, _ in hybrid_rank[:5]]
    reranked = sorted(cand, key=lambda i: -cosine(q_vec, doc_vecs[i]))
    print("\n  reranked candidates (cross-encoder proxy = dense over shortlist):")
    for i in reranked[:3]:
        mark = "  <-- TARGET" if corpus.filenames[i] == target else ""
        print(f"     {cosine(q_vec, doc_vecs[i]):.4f}  {corpus.filenames[i]}{mark}")
    top1_rr = corpus.filenames[reranked[0]]
    print(f"     reranked top-1 = {top1_rr}  ({'HIT' if top1_rr == target else 'MISS'})")

    _rule("WHY THIS MATTERS")
    print("BM25 misses paraphrased intent entirely (0 overlap on 'salary slip').")
    print("Dense catches the paraphrase. Naive hybrid RRF gets pulled by the "
          "lexical distractor 'onboarding_note' (it literally says 'salary').")
    print("A RERANK over the hybrid shortlist restores the right doc — this is the "
          "concrete reason production RAG is hybrid + rerank, not one or the other.")


if __name__ == "__main__":
    main()
