"""
PATHOLOGY (d): duplicate / near-duplicate pages.

Crawls collect the same article many times: ?utm_source=... campaign variants,
print versions, AMP pages, trailing-slash and fragment variants. If they all land
in the vector index, retrieval returns five copies of one answer, wasting context
budget and skewing any similarity vote. Two defenses:

  1. URL canonicalization -- strip tracking params (utm_*, gclid, fbclid, ref),
     drop fragments, normalize scheme/host/trailing slash, sort remaining query
     keys. Exact-duplicate URLs collapse immediately.
  2. Near-duplicate CONTENT detection -- print/AMP variants have different chrome
     but the same prose. We shingle the text into k-word n-grams, hash them, take
     a MinHash signature, and estimate Jaccard similarity. Above a threshold ->
     near-duplicate, keep one canonical copy.

Interview line: "Canonicalization catches the cheap dupes; MinHash shingling
catches the expensive ones -- the print version whose URL and boilerplate differ
but whose body is identical. Both run before indexing so retrieval isn't poisoned
by redundancy." (In production I'd reach for datasketch's MinHashLSH; the
hand-rolled MinHash here shows I know what it's doing under the hood.)
"""
from __future__ import annotations

import hashlib
import os
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = re.compile(r"^(utm_|gclid|fbclid|mc_|ref$|ref_|igshid)", re.IGNORECASE)


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path.rstrip("/") or "/"
    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not _TRACKING.match(k)]
    query = urlencode(sorted(kept))
    return urlunsplit((scheme, netloc, path, query, ""))   # fragment dropped


# --------------------------------------------------------------------------- #
# MinHash near-duplicate detection (hand-rolled)
# --------------------------------------------------------------------------- #
def _shingles(text: str, k: int = 5) -> set[int]:
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < k:
        return {_h(" ".join(tokens))} if tokens else set()
    return {_h(" ".join(tokens[i:i + k])) for i in range(len(tokens) - k + 1)}


def _h(s: str) -> int:
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)


_MASK = (1 << 64) - 1


def minhash_signature(text: str, num_perm: int = 64, seed: int = 1) -> list[int]:
    """num_perm hash permutations; signature[i] = min over shingles of a permuted
    hash. Jaccard(A,B) ~= fraction of matching signature slots."""
    shingles = _shingles(text)
    if not shingles:
        return [0] * num_perm
    sig = []
    for p in range(num_perm):
        a = (seed + p) * 2654435761 & _MASK
        b = (p * 40503 + 12345) & _MASK
        sig.append(min(((a * s + b) & _MASK) for s in shingles))
    return sig


def estimate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    return sum(x == y for x, y in zip(sig_a, sig_b)) / len(sig_a)


def dedupe(docs: list[tuple[str, str]], threshold: float = 0.6):
    """docs = [(url, text)]. Returns (kept, dropped) where dropped notes the
    reason (exact URL or near-dup of which canonical)."""
    kept: list[dict] = []
    dropped: list[dict] = []
    seen_urls: dict[str, int] = {}   # canonical url -> index in kept
    for url, text in docs:
        curl = canonicalize_url(url)
        if curl in seen_urls:
            dropped.append({"url": url, "reason": f"exact URL dup of #{seen_urls[curl]}"})
            continue
        sig = minhash_signature(text)
        dup_of = None
        for i, k in enumerate(kept):
            j = estimate_jaccard(sig, k["sig"])
            if j >= threshold:
                dup_of = (i, round(j, 2))
                break
        if dup_of is not None:
            dropped.append({"url": url,
                            "reason": f"near-dup of #{dup_of[0]} (jaccard~{dup_of[1]})"})
        else:
            seen_urls[curl] = len(kept)
            kept.append({"url": url, "canonical": curl, "sig": sig,
                         "text": text[:60] + "..."})
    return kept, dropped


def _demo():
    from fixtures.make_fixtures import URLS
    from boilerplate import extract_main

    print("=" * 70)
    print("URL CANONICALIZATION + NEAR-DUP (MinHash shingling)")
    print("=" * 70)

    print("\n[BEFORE] raw URLs:")
    for u in URLS.values():
        print("   ", u)
    print("\n[AFTER] canonical URLs (tracking/query/fragment normalized):")
    for u in URLS.values():
        print("   ", u, "->", canonicalize_url(u))
    print("    note: exact_utm collapses to canonical by URL alone; print_variant "
          "keeps a distinct path\n    so only the content near-dup check can catch it.")

    # extract main text from the two page variants (chrome differs, body same)
    with open(os.path.join(os.path.dirname(__file__), "fixtures", "canonical.html")) as f:
        text_a, _ = extract_main(f.read())
    with open(os.path.join(os.path.dirname(__file__), "fixtures", "utm_dup.html")) as f:
        text_b, _ = extract_main(f.read())

    docs = [
        (URLS["canonical"], text_a),
        (URLS["exact_utm"], text_a),                     # collapses by URL alone
        (URLS["print_variant"], text_b),                 # caught only by MinHash
        ("https://ex.com/blog/reranking", "Reranking reorders retrieved passages "
         "with a cross-encoder to lift precision at k for the final context."),
    ]
    kept, dropped = dedupe(docs)
    sim = estimate_jaccard(minhash_signature(text_a), minhash_signature(text_b))
    print(f"\n[NEAR-DUP] jaccard(canonical, print-variant) ~= {sim:.2f}")
    print("\n[AFTER] kept:")
    for i, k in enumerate(kept):
        print(f"    #{i}  {k['canonical']}  :: {k['text']}")
    print("[AFTER] dropped:")
    for d in dropped:
        print(f"    {d['url']}  -- {d['reason']}")


if __name__ == "__main__":
    _demo()
