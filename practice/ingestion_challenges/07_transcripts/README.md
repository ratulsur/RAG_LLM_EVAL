# 07 — Transcript Ingestion (audio-derived text)

**Maps to:** Universal Document Ingestor — the "ingest anything" source that most
teams get wrong. Transcripts are the hardest text source because they aren't prose.

## The pathology (what breaks)
ASR output is an **unpunctuated token stream** with fillers, homophone errors,
diarization noise, `[inaudible]` gaps, and — in the Indian market — **Hindi-English
code-switching**. A `RecursiveCharacterTextSplitter` has no sentence boundaries to
split on, so it cuts mid-thought, indexes `um`/`uh` as content, and wrecks recall.
**This is an ingestion problem, not a model problem.**

## Detect → Handle → Emit
| Stage | What it does |
|---|---|
| Parse turns | Speaker + `[HH:MM:SS]` become **metadata**, never body text. Big inter-turn timestamp gaps are pause boundaries. |
| Markers | `[inaudible]` → preserved `<INAUDIBLE>` placeholder (queryable gap, not silent drop); `[overlap]` → removed as a diarization artifact. |
| Fillers | `um, uh, like, you know, i mean, basically, sort of` stripped (multi-word first). |
| Normalize | Entity/drug/company fixes auto-applied **and flagged** (`metphormin→metformin`, `axenture→accenture`, `pharamcy→pharmacy`). |
| Homophones | `their/there/they're`, `to/too/two` **flagged, never auto-fixed** — the right choice is grammatical, not lexical. |
| Script detect | Devanagari vs Latin via unicode block ranges + romanized-Hindi function words → `code_switch` flag. |
| Segment | Heuristic sentence restoration (discourse markers + clause-length gate). **Production path:** a punctuation-restoration model (guarded import). |
| Chunk | Pack restored sentences to ~30 words, respecting turn + sentence boundaries; carry rich metadata. |

## Senior tradeoff to say out loud
> "I don't silently auto-correct homophones. In a clinical or financial transcript,
> `to high` vs `too high` changes meaning, so I surface corrections with a confidence
> flag and keep the original in metadata. Cleaning has to be **auditable**, not just clean."

## Live vs reference
- **Live (runs offline):** turn parsing, marker handling, filler strip, entity
  correction, homophone flagging, unicode script/code-switch detection, heuristic
  segmentation, chunking with metadata.
- **Reference (guarded):** `deepmultilingualpunctuation` punctuation-restoration model
  is the production segmentation path — the code uses it if importable, else the
  heuristic. `langdetect`/`fasttext` are the production language-ID upgrade over the
  romanized-Hindi wordlist.

## Run
```bash
python fixtures/make_transcript.py   # writes dirty_transcript.txt (optional)
python transcript_ingest.py          # fixture -> clean chunks + before/after
```

## Optional deps (not required)
`deepmultilingualpunctuation` (punctuation restoration), `langdetect` / `fasttext`
(language ID). Everything runs without them via offline fallbacks.
