# 09 — Corpus-Level Pathologies (the ones that hurt retrieval most)

**Maps to:** Universal Document Ingestor + source-grading. These are the failures
that survive perfect per-document ingestion and only appear once documents sit
*together* in an index — so they're the ones that quietly wreck production RAG.

Each file: **detect the pathology → handle it → emit clean, retrieval-safe output**,
with a live before/after over a synthesized dirty corpus (`fixtures/make_corpus.py`).

| File | Pathology | Core technique |
|---|---|---|
| `near_dup_versioning.py` | (a) v1/v2/final/FINAL2 near-dups; stale version wins confidently | MinHash over k-shingles + Jaccard → union-find clusters → version picker (filename marker + recency + `supersedes`) |
| `authority_temporal.py` | (b) contradictory docs, no authority signal; (c) two "current" facts from different years | Numeric-claim contradiction detection → authority = supersedes > source-trust > recency; effective-date extraction + validity window |
| `length_and_retrieval.py` | (d) 2-line memo vs 400-page report skews chunking + top-k | Length-aware chunking (short=atomic, long=windowed+parent-tagged) + per-doc max-pooling + per-source cap |
| `pii_redaction.py` | (e) Aadhaar/PAN/account/card in free text pre-index | Regex + **validators** (Aadhaar **Verhoeff**, card **Luhn**) → typed salted-hash placeholders |
| `vocab_mismatch.py` | (f) "salary slip" vs corpus "pay statement"; BM25 fails | BM25 miss → synonym expansion → dense embeddings → hybrid RRF + **rerank** |

## The senior points to say out loud (one per file)
- **Near-dup ≠ semantic.** "Is this the same document?" is a **lexical** question →
  MinHash/Jaccard, deterministic, no model. Embeddings are the wrong tool here.
- **Recency is not authority.** A 2026 blog shouldn't beat a 2025 signed policy;
  authority is a blend — **explicit supersession > source trust > recency (tiebreak)**.
- **Length fairness is a retrieval-time fix, not a bigger-chunk fix.** Max-pool a
  doc's score to its best chunk and cap per-source contribution, so a 67-chunk
  report can't flood top-k against a 1-chunk memo.
- **Validation is what separates a redactor from a regex.** The fake `0000 0000 0000`
  matches the Aadhaar pattern but **fails Verhoeff**, so it isn't redacted — no false
  positive. Placeholders are typed + salted so retrievability survives and the index
  stays non-reversible.
- **Hybrid needs rerank.** Dense fixes the paraphrase; naive RRF gets dragged by a
  lexical distractor (`onboarding_note` literally says "salary"); a cross-encoder
  rerank over the shortlist restores the right doc. This is *why* production RAG is
  hybrid + rerank.

## Live vs reference
- **Live (offline, no deps beyond stdlib+numpy+dateutil):** near-dup MinHash/Jaccard/
  clustering, version picking, contradiction + authority ranking, temporal validity,
  length-aware chunking + fair retrieval, full PII detect/validate/redact (Verhoeff +
  Luhn hand-rolled), BM25 + synonym expansion + hybrid RRF + rerank.
- **Live IF keys present:** `vocab_mismatch.py` uses the repo's `get_embeddings()`
  (OpenAI `text-embedding-3-small`) when `OPENAI_API_KEY` is set — confirmed running
  live in this env — and **falls back to an offline hashed-trigram embedder** with no
  network. Same ranking mechanism either way.
- **Reference (guarded):** NLI model (`roberta-large-mnli`) as the production
  contradiction detector over the numeric-claim heuristic; cross-encoder reranker
  (`bge-reranker` / Cohere rerank) — proxied here by dense-over-shortlist.

## Run
```bash
python fixtures/make_corpus.py     # inspect the dirty corpus
python near_dup_versioning.py
python authority_temporal.py
python length_and_retrieval.py
python pii_redaction.py
python vocab_mismatch.py           # live OpenAI embeddings if key set, else offline
```

## Extra deps leaned on
`numpy`, `pandas` (available), `python-dateutil` (`authority_temporal.py`). Optional:
`transformers` (NLI / cross-encoder), OpenAI via existing repo helper (already in repo).
