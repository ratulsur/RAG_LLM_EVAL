# 04 — DOCX ingestion pathologies

A `.docx` is a ZIP of WordprocessingML parts. `python-docx`'s `paragraph.text`
walks only `<w:r>` runs in the main body — so tracked changes, comments,
footnotes, textboxes, headers/footers, and *computed* list numbers are all
invisible to it. Each gap is a place real content hides.

## Files

| File | Pathology | Live vs reference |
|------|-----------|-------------------|
| `fixtures/generate.py` | Hand-authors a dirty `.docx` (raw OOXML) + format decoys | **live** |
| `tracked_changes_comments.py` | **(a)** Separate ACCEPTED / REJECTED body streams + isolate comments | **live** (lxml over document.xml/comments.xml) |
| `hidden_content.py` | **(b)** Recover textboxes, headers/footers, footnotes naive parsers drop | **live** (lxml over the ZIP parts) |
| `numbering_reconstruction.py` | **(c)** Reconstruct style-generated "Section 2 / 2.1" numbering from numbering.xml | **live** |
| `format_detection.py` | **(d)** Route legacy `.doc` / encrypted / renamed / RTF / PDF by magic bytes | **live** sniffing; `.doc` conversion + `olefile` **guarded** (soffice/antiword absent → route + install note) |

## Why this is a strong RAG answer

Naive `python-docx` on the fixture returns *"The vendor shall deliver within
business days."* — the operative number (`30` deleted, `15` inserted) is gone,
the sidebar clause is gone, the liability-cap footnote is gone, and the numbered
clauses have no numbers. Any RAG answer about delivery terms, escrow, the
liability cap, or "Section 2.1" would be a confident hallucination.

- **Tracked changes:** produce three explicit streams — ACCEPTED (insertions kept,
  deletions removed = the final doc), REJECTED (the original), COMMENTS (isolated
  metadata, never inlined). *Which* stream you index is a policy call (signed
  contract → ACCEPTED; audit/diff → both with provenance). The failure is letting
  the default parser decide by accident.
- **Hidden content:** enumerate the ZIP parts and pull `txbxContent`, `header*.xml`,
  `footer*.xml`, `footnotes.xml` explicitly, tagging provenance. Headers are
  double-edged (a "CONFIDENTIAL" banner is furniture; a defined term is content) —
  surface with a tag and let the chunker's furniture filter decide.
- **Numbering:** the visible "2.1" is *computed* by Word and never stored in run
  text. Reconstruct it by walking list paragraphs with a per-level counter and
  rendering `numbering.xml`'s `lvlText`. Without this every "see Section 4.2"
  becomes a dangling pointer and clause-level citation dies.
- **Format detection:** extension ≠ format. Sniff container magic
  (`PK..` = OOXML zip, `D0CF11E0` = OLE2 legacy/encrypted, `{\rtf`, `%PDF`) and
  route. OLE2 needs an olefile probe to split "legacy .doc" from "encrypted
  OOXML"; legacy `.doc` has no clean pure-Python extractor, so route to
  `soffice --headless --convert-to txt` / `antiword` and degrade loudly.

## Run

```bash
python fixtures/generate.py
python tracked_changes_comments.py
python hidden_content.py
python numbering_reconstruction.py
python format_detection.py
```

Extra deps leaned on: **lxml** (pre-installed) for raw XML walking;
**olefile** guarded-optional (used only to disambiguate OLE2 containers — absent
here, code degrades to the install note). `.doc` conversion via soffice/antiword
is reference-only (binaries absent).
