"""
09d — EXTREMELY SKEWED DOC LENGTHS -> chunking + top-k unfairness.

THE PATHOLOGY (say this in the room)
  A corpus holds a 2-line memo next to a 400-page report. Fixed-size chunking
  produces 1 chunk for the memo and hundreds for the report. Two failures follow:
    1. TOP-K DOMINATION: the long doc floods top-k with many near-duplicate
       chunks, starving short docs even when the short doc is the right answer.
    2. SCORE INFLATION: a long doc has more chunks and thus more *chances* to
       match, so raw chunk scores over-represent it.

THE FIX
  A) LENGTH-AWARE CHUNKING: short docs stay whole (atomic fact); long docs are
     split into overlapping windows, but we cap and TAG chunks with their parent
     so we can group later (parent-document pattern).
  B) RETRIEVAL NORMALIZATION at query time:
       - per-document MAX-pooling: a doc's score = its best chunk, not the sum,
         so a 300-chunk doc can't win by volume.
       - per-document top-k CAP: at most `max_per_doc` chunks from any one source
         in the final context, preserving diversity/fairness.

SENIOR POINT
  This is the MMR / "source diversity" instinct applied at ingestion+retrieval.
  The interview trap is answering "just make chunks bigger" — wrong; the fix is
  normalizing how scores AGGREGATE per document and capping per-source
  contribution, independent of chunk size.

RUN
    python length_and_retrieval.py
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass

from fixtures.make_corpus import Doc, make_corpus


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    parent_id: str


# ===========================================================================
# A) Length-aware chunking
# ===========================================================================
def word_tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_doc(doc: Doc, size: int = 60, overlap: int = 15, short_threshold: int = 80) -> list[Chunk]:
    """Short docs -> a single atomic chunk (splitting a 2-line memo hurts recall).
    Long docs -> overlapping windows, each tagged with parent_id for regrouping."""
    toks = word_tokens(doc.text)
    if len(toks) <= short_threshold:
        return [Chunk(f"{doc.doc_id}#0", doc.doc_id, doc.filename, doc.text, doc.doc_id)]
    chunks: list[Chunk] = []
    step = size - overlap
    for i, start in enumerate(range(0, len(toks), step)):
        window = toks[start:start + size]
        if not window:
            break
        chunks.append(Chunk(
            f"{doc.doc_id}#{i}", doc.doc_id, doc.filename, " ".join(window), doc.doc_id,
        ))
        if start + size >= len(toks):
            break
    return chunks


# ===========================================================================
# Tiny BM25-ish scorer (no deps) just to demonstrate the ranking effect
# ===========================================================================
def score_chunk(query: str, chunk: Chunk) -> float:
    q = set(re.findall(r"\w+", query.lower()))
    words = re.findall(r"\w+", chunk.text.lower())
    if not words:
        return 0.0
    tf = sum(words.count(w) for w in q)
    # length-normalized tf so a longer chunk doesn't win purely on raw counts
    return tf / math.sqrt(len(words))


# ===========================================================================
# B) Retrieval normalization
# ===========================================================================
def naive_topk(chunks: list[Chunk], query: str, k: int = 5) -> list[tuple[Chunk, float]]:
    scored = [(c, score_chunk(query, c)) for c in chunks]
    return sorted(scored, key=lambda x: -x[1])[:k]


def fair_topk(chunks: list[Chunk], query: str, k: int = 5, max_per_doc: int = 1
              ) -> list[tuple[Chunk, float]]:
    """Per-doc MAX-pooling + per-source cap. Rank docs by their BEST chunk, then
    fill the final context round-robin so no single long doc dominates."""
    scored = sorted(((c, score_chunk(query, c)) for c in chunks), key=lambda x: -x[1])
    taken: dict[str, int] = defaultdict(int)
    out: list[tuple[Chunk, float]] = []
    for c, s in scored:
        if s <= 0:
            continue
        if taken[c.doc_id] >= max_per_doc:
            continue
        taken[c.doc_id] += 1
        out.append((c, s))
        if len(out) >= k:
            break
    return out


def _rule(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main() -> None:
    docs = make_corpus()
    # focus on the skewed pair + a couple others sharing the "ingestion" vocab
    chunks: list[Chunk] = []
    for d in docs:
        chunks.extend(chunk_doc(d))

    _rule("BEFORE — chunk counts show the skew")
    counts: dict[str, int] = defaultdict(int)
    for c in chunks:
        counts[c.filename] += 1
    for fn in ("office_memo.txt", "annual_report.txt"):
        print(f"  {fn:22s} -> {counts[fn]:4d} chunks")

    query = "when do normal office hours resume after the holiday"
    _rule(f"QUERY: {query!r}")

    naive = naive_topk(chunks, query, k=5)
    print("NAIVE top-5 (long report can flood the list):")
    for c, s in naive:
        print(f"   {s:.3f}  {c.filename:22s}  {c.text[:45]!r}")

    fair = fair_topk(chunks, query, k=5, max_per_doc=1)
    print("\nFAIR top-5 (per-doc max-pool + 1-per-source cap):")
    for c, s in fair:
        print(f"   {s:.3f}  {c.filename:22s}  {c.text[:45]!r}")

    _rule("WHY THIS MATTERS")
    rep_naive = sum(c.filename == "annual_report.txt" for c, _ in naive)
    rep_fair = sum(c.filename == "annual_report.txt" for c, _ in fair)
    src_naive = len({c.doc_id for c, _ in naive})
    src_fair = len({c.doc_id for c, _ in fair})
    print(f"report chunks in NAIVE top-5: {rep_naive}  (distinct sources: {src_naive})")
    print(f"report chunks in FAIR  top-5: {rep_fair}  (distinct sources: {src_fair})")
    print("The 67-chunk report floods NAIVE top-k with near-duplicate windows; the "
          "per-source cap collapses it to one and widens source diversity — fairness "
          "is a retrieval-time fix, not a bigger-chunk fix.")


if __name__ == "__main__":
    main()
