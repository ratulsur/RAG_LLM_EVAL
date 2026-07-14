"""
09a — NEAR-DUPLICATES & VERSIONING.

THE PATHOLOGY (say this in the room)
  A Drive dump has refund_policy_v1 / v2 / final / FINAL2 — 90% identical text.
  Embed all four and retrieval returns whichever scores highest, often a STALE
  version, and it cites it CONFIDENTLY. Exact-hash dedup misses them because a
  comma changed. This is the single most common reason a RAG system answers with
  an out-of-date policy.

THE FIX (detect -> pick the winner -> keep the rest as history)
  1. DETECT near-dups with MinHash over k-shingles + Jaccard. Shingling captures
     wording overlap; MinHash makes the pairwise Jaccard cheap at corpus scale.
  2. CLUSTER by a Jaccard threshold (union-find).
  3. PICK the authoritative version per cluster with a signal blend:
       filename markers (FINAL/final > v2 > v1)  +  recency (date)  +  an
       explicit "supersedes" phrase in the body.
  4. EMIT one CANONICAL doc per cluster; demote the rest to superseded (kept for
     audit, filtered out of the default retrieval set).

WHY MinHash NOT embeddings HERE
  Near-dup detection is a LEXICAL question ("is this the same document?"), not a
  SEMANTIC one. Two different policies can be semantically close; two versions of
  ONE policy are lexically near-identical. MinHash/Jaccard answers the right
  question, is deterministic, and needs no model. Embeddings are for the vocab-
  mismatch problem (see 09f), not this one.

RUN
    python near_dup_versioning.py
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from fixtures.make_corpus import Doc, make_corpus


# ===========================================================================
# MinHash over k-shingles
# ===========================================================================
def shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingles (k consecutive words). Word shingles are robust to
    whitespace/casing noise; char shingles would over-fire on short docs."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _hash_shingle(s: str, seed: int) -> int:
    h = hashlib.md5(f"{seed}:{s}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def minhash_signature(sh: set[str], num_perms: int = 64) -> list[int]:
    """One min-hash per permutation (simulated by seeded hashing). Two docs'
    signatures agree in ~Jaccard fraction of positions — that's the trick."""
    return [min(_hash_shingle(s, seed) for s in sh) for seed in range(num_perms)]


def estimated_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    agree = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return agree / len(sig_a)


def true_jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ===========================================================================
# Cluster (union-find) by Jaccard threshold
# ===========================================================================
class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def cluster_near_dups(docs: list[Doc], threshold: float = 0.5) -> list[list[int]]:
    sigs = [minhash_signature(shingles(d.text)) for d in docs]
    uf = _UF(len(docs))
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            if estimated_jaccard(sigs[i], sigs[j]) >= threshold:
                uf.union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(len(docs)):
        clusters.setdefault(uf.find(i), []).append(i)
    return list(clusters.values())


# ===========================================================================
# Version picker
# ===========================================================================
_VER_RE = re.compile(r"v(\d+)", re.I)


def version_score(doc: Doc) -> tuple:
    """Blend of signals, higher = more authoritative. Ordered tuple so date
    breaks ties. FINAL2 > FINAL > v2 > v1, recency, and explicit supersedes."""
    fn = doc.filename.lower()
    supersedes = 1 if "supersede" in doc.text.lower() else 0
    if "final2" in fn:
        marker = 4
    elif "final" in fn:
        marker = 3
    else:
        m = _VER_RE.search(fn)
        marker = int(m.group(1)) if m else 0
    return (supersedes, marker, doc.date)


@dataclass
class VersionDecision:
    canonical: Doc
    superseded: list[Doc]
    intra_cluster_jaccard: float


def resolve_versions(docs: list[Doc], threshold: float = 0.5) -> list[VersionDecision]:
    decisions: list[VersionDecision] = []
    for idxs in cluster_near_dups(docs, threshold):
        members = [docs[i] for i in idxs]
        winner = max(members, key=version_score)
        # report the min pairwise true Jaccard so we can see how alike they are
        sh = [shingles(m.text) for m in members]
        jac = min(
            (true_jaccard(sh[a], sh[b]) for a in range(len(sh)) for b in range(a + 1, len(sh))),
            default=1.0,
        )
        decisions.append(VersionDecision(
            canonical=winner,
            superseded=[m for m in members if m is not winner],
            intra_cluster_jaccard=round(jac, 3),
        ))
    return decisions


def _rule(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main() -> None:
    docs = make_corpus()
    _rule("BEFORE — 4 refund-policy versions all indexable; stale ones win queries")
    for d in docs:
        if "refund" in d.filename:
            print(f"  {d.filename:26s} {d.date}  '{d.text[-45:]}'")

    decisions = resolve_versions(docs)
    _rule("AFTER — near-dup clusters resolved to one canonical each")
    for dec in decisions:
        if len(dec.superseded) == 0:
            continue  # singletons aren't interesting here
        print(f"cluster (intra Jaccard ~{dec.intra_cluster_jaccard}):")
        print(f"  CANONICAL  -> {dec.canonical.filename}  ({dec.canonical.date})")
        for s in dec.superseded:
            print(f"  superseded -> {s.filename}  ({s.date})  [kept for audit, "
                  f"filtered from default retrieval]")

    _rule("WHY THIS MATTERS")
    print("Retrieval now defaults to refund_policy_FINAL2 (30-day window, "
          "'supersedes all prior'), not v1's stale 15-day rule.")


if __name__ == "__main__":
    main()
