# Ingestion Challenges — the hardest part of RAG

> **Thesis for the interview room:** in production RAG, the model is rarely the
> bottleneck — *ingestion* is. Retrieval quality, faithfulness, and hallucination
> rate are all capped by what your extractor did to the document *before* a single
> embedding was computed. Naive extraction (`page.get_text()`, `paragraph.text`,
> `msg.get_content()`) silently drops, duplicates, or scrambles content, and no
> reranker or grounding grader downstream can repair a chunk that was already
> corrupt at the source.

This module is a 9-codebase tour of real-world document **pathologies**. Each
codebase follows the same contract:

> **DETECT** a real pathology → **HANDLE** it → **EMIT** clean, metadata-rich,
> chunk-ready output for a RAG pipeline.

Every folder is self-contained: a `fixtures/` generator **synthesizes** the dirty
input (no external files needed), the handler files run live and print a
**before (naive) vs after (fixed)** comparison, and the README frames the
interview answer and the senior tradeoff.

Directly maps to Ratul's **Universal Document Ingestor** (multi-source RAG,
source grading, DeepEval): this is the extraction-and-normalization layer that
sits *upstream* of chunking, embedding, and the source/grounding graders.

---

## The 9 codebases

| # | Folder | Pathology cluster | Owner |
|---|--------|-------------------|-------|
| 01 | `01_pdfs/` | Scanned/OCR, multi-column reading order, tables across page breaks, mixed native+scanned, header/footer/watermark furniture, charts-as-pixels | **this agent** |
| 02 | `02_excel_csv/` | A spreadsheet is a canvas, not a table: merged cells, header/type inference, multi-sheet, date pathologies, dirty CSV | sibling |
| 03 | `03_html/` | Raw HTML is 80% chrome: boilerplate/nav stripping, main-content extraction, div-tables, JS-rendered, canonical dedup | sibling |
| 04 | `04_docx/` | Tracked changes + comments, textboxes/headers/footers/footnotes, style-generated numbering, legacy/renamed/encrypted format routing | **this agent** |
| 05 | `05_email/` | Quoted reply chains + signatures/disclaimers, inline images + nested attachments, thread reconstruction from RFC headers | **this agent** |
| 06 | `06_json_logs/` | Semi-structured drift: evolving schemas, null/absent/empty ambiguity, corrupt NDJSON lines, mixed log grammars | sibling |
| 07 | `07_transcripts/` | ASR output isn't prose: unpunctuated token streams, fillers, diarization, `[inaudible]` gaps, Hindi-English code-switching | sibling |
| 08 | `08_document_images/` | Photos, not clean scans: perspective/skew/glare/shadow correction before OCR, multiscript forms | sibling |
| 09 | `09_corpus_level/` | Cross-document pathologies: near-dup/versioning, PII redaction, authority/temporal conflicts, length skew, vocab mismatch | sibling |

> **Ownership:** `01_pdfs/`, `04_docx/`, and `05_email/` are built by this agent.
> Folders 02, 03, 06–09 are built by sibling agents (all present in this tree).

---

## How to run any codebase

```bash
cd 01_pdfs            # or 04_docx, 05_email
python fixtures/generate.py     # synthesize the dirty inputs
python multicolumn_layout.py    # run any handler -> prints before/after
```

Every handler also auto-generates its fixture on first run, so you can invoke a
single file cold.

## Dependency posture

- **Live** (installed, run for real): pdfplumber, pymupdf (`fitz`), pypdf,
  reportlab, Pillow, python-docx, lxml, pandas/numpy, stdlib `email`/`zipfile`.
- **Guarded-optional** (detected + routed, never required): OCR
  (`pytesseract` + tesseract binary), `olefile` (OLE2 disambiguation), LibreOffice
  `soffice`/`antiword` (legacy `.doc`). When absent, the code **detects the need,
  emits the exact install/route command, and degrades loudly** — it never fails
  silently or pretends. This is the honest production posture and the correct
  interview answer: *detect, route, degrade loudly.*
