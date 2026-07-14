# 01 — PDF ingestion pathologies

PDFs are a *layout* format, not a *content* format. They store glyphs and their
coordinates, not sentences and sections. Naive text extraction therefore fails in
six recurring ways — each handled here, each a live before/after demo.

## Files

| File | Pathology | Live vs reference |
|------|-----------|-------------------|
| `fixtures/generate.py` | Synthesizes all test PDFs with reportlab + Pillow | **live** |
| `multicolumn_layout.py` | **(b)** Multi-column reading order via projection-profile (X-Y cut) gutter detection | **live** (pdfplumber word bboxes) |
| `headers_footers_watermark.py` | **(e)** Repeating header/footer/page-number + watermark removal via frequency analysis | **live** (pdfplumber) |
| `tables_across_pages.py` | **(c)** Tables spanning page breaks; stitch by repeated-header match; emit row-records | **live** (pdfplumber table extraction) |
| `ocr_routing.py` | **(a)** image-only/scanned pages, **(d)** mixed native+scanned per-page routing, **(f)** charts-as-pixels | **live routing + rasterization** (pymupdf); **OCR guarded** (pytesseract/tesseract absent → stub + install note) |

## The interview one-liners

- **Multi-column:** "PDF stores glyphs in draw order, not reading order. I build a
  vertical projection profile of word x-spans, find the empty band = the gutter,
  split columns there, then read each top-to-bottom, columns left-to-right.
  x0-only clustering fails because a column's words fill a wide x-range — the gap
  is *between* columns, so you must project spans." Naive interleaves the columns;
  fixed recovers order.
- **Furniture:** "Running headers/footers/page-numbers/watermarks aren't content.
  I mask digits so `Page 1 of 4` and `Page 2 of 4` collapse to one signature, then
  drop any signature that recurs on ≥60% of pages *in a margin band*. Page numbers,
  banners, and watermarks go before chunking so they don't inflate term stats."
- **Tables across pages:** "Per-page extraction returns two fragments with a
  repeated header. I detect a continuation (same columns + first row == prior
  header), drop the phantom header, and emit *row-records* — header-keyed dicts —
  because a table row is only retrievable if it carries its own column keys."
- **OCR routing:** "I classify every page — text layer present? extract : OCR —
  keying on image *area fraction*, not just char count, so a full-page scan with a
  tiny caption still routes to OCR. Mixed docs get per-page dispatch. I OCR only
  the pages that need it (10–100× slower, error-prone on skew). Charts get flagged
  `data-in-pixels`: OCR recovers axis labels, not plotted values — that needs a
  vision model, and I flag rather than hallucinate."

## Senior tradeoffs (say these, don't just code them)

- Projection-profile column cuts assume a stable grid; escalate to a layout model
  (LayoutParser / Table Transformer) for magazine layouts and borderless tables.
- Frequency furniture detection needs several pages; on 1–2 pages fall back to a
  positional (top/bottom 8%) heuristic.
- OCR is the escape hatch, not the default — routing keeps it cheap and honest.

## Run

```bash
python fixtures/generate.py
python multicolumn_layout.py
python headers_footers_watermark.py
python tables_across_pages.py
python ocr_routing.py
```

Extra deps leaned on beyond the core set: none (reportlab, pdfplumber, pymupdf,
Pillow all pre-installed). OCR is guarded-optional.
