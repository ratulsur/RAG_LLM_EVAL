# 03 — HTML ingestion challenges

> **Theme:** raw HTML is 80% chrome. If you embed `soup.get_text()`, the nav bar and cookie
> banner dominate every chunk and near-duplicate pages poison retrieval. Detect the pathology,
> extract the *content*, dedupe, emit clean chunk-ready text + metadata. Maps to the
> **Universal Document Ingestor** (multi-source RAG, source grading).

## Why naive extraction fails (the 30-second interview answer)

| Pathology | What breaks | The production fix (in this folder) |
|---|---|---|
| **(a)** Boilerplate contamination (nav, cookie banner, footer, "related") | every chunk is polluted with repeated chrome; distinct pages look identical to the index | main-content extraction: strip non-content tags + id/class boilerplate, score blocks by text-length × (1 − link-density) |
| **(b)** JS-rendered content + embedded state | crawler reads a near-empty `<body>`, gets nothing | **detect** the SPA shape (thin body + heavy `<script>` / `__NEXT_DATA__`), harvest embedded JSON *without a browser*; route to playwright only as last resort |
| **(c)** Tables built from `<div>` grids, not `<table>` | `pd.read_html` finds no table | detect the repeating row/cell class pattern, reconstruct a DataFrame |
| **(d)** Duplicate / near-dup pages (UTM variants, print versions) | retrieval returns 5 copies of one answer, wastes context, skews similarity votes | URL canonicalization (strip utm_*/gclid/ref, drop fragment, sort query) **+** MinHash shingling for content near-dups |

## Files

```
03_html/
├── fixtures/
│   ├── make_fixtures.py   # synthesizes: article (nav+cookie+footer+related), divtable,
│   │                      #  js_page (__NEXT_DATA__), canonical + print/UTM near-dups
│   └── *.html             # generated
├── boilerplate.py         # (a) hand-rolled density/tag main-content extractor (bs4)
├── structure.py           # (b) JS-render detection + embedded-JSON harvest  (c) div-grid → DataFrame
└── dedup.py               # (d) URL canonicalization + MinHash near-dup detection
```

## Run

```bash
python fixtures/make_fixtures.py
python boilerplate.py   # BEFORE (851 chars w/ nav+cookie) → AFTER (article only, no leak)
python structure.py     # SPA routed to __NEXT_DATA__ JSON; <div> grid → DataFrame
python dedup.py         # UTM variant collapses by URL; print variant caught by MinHash
```

Runs **live** on `bs4` + `lxml`. Two paths are **reference-only, guarded** (the file still runs
without them):

- `trafilatura` / `readability-lxml` — the production main-content extractors. `boilerplate.py`
  uses trafilatura when installed, else falls back to the hand-rolled heuristic. Install:
  `pip install trafilatura`.
- `playwright` — the headless renderer for genuinely client-rendered pages. `structure.py`
  references it but **prefers harvesting `__NEXT_DATA__`** first (100× cheaper). Install:
  `pip install playwright && playwright install chromium`.

## Senior tradeoffs to say out loud

- **Check for embedded JSON before you launch a browser.** Most "JS-only" pages ship their data
  in `__NEXT_DATA__` / `__NUXT__` / JSON-LD. Headless rendering is the expensive last resort, not
  the default.
- **Two dedup layers, on purpose.** Canonicalization is cheap and catches UTM/fragment dupes;
  MinHash shingling catches the *expensive* case — a print/AMP variant whose URL and chrome
  differ but whose body is identical. Production would use `datasketch`'s `MinHashLSH` for
  sublinear lookup; the hand-rolled MinHash here shows the mechanics.
- **The heuristic extractor is honest, not perfect.** trafilatura adds tag-path priors, comment
  stripping, and language detection. Know the library *and* the mechanics underneath it.
