"""
PATHOLOGY (c): a table continues across a page break; the header row repeats on
the next page. Naive extraction yields two disconnected fragments -- and if you
chunk per page, half the rows lose their column meaning entirely.

WHY NAIVE EXTRACTION FAILS
--------------------------
- Per-page table extraction returns page 1's rows and page 2's rows as two
  separate tables. The continuation on page 2 often REPEATS the header, so you
  get a phantom header row mid-data.
- Flattening a table to plain text ("INV-1001 Northwind 12,400 Paid") loses the
  column semantics a RAG query needs ("which invoices are Disputed?").

THE PRODUCTION FIX
------------------
  1. Extract tables per page with pdfplumber.
  2. Detect a continuation: same column count AND the first row of table N+1
     matches the header of table N -> it's the same logical table.
  3. Stitch: drop the repeated header, concatenate rows.
  4. Emit ROW-oriented records (header-keyed dicts) so each row is a
     self-describing chunk: "Invoice=INV-1006, Vendor=Tailspin, ...".

SENIOR TRADEOFF
---------------
Header-match stitching is a heuristic. It breaks on borderless tables, on
merged/nested headers spanning columns, and on superscript footnote markers
that pdfplumber attaches to a cell. For those, escalate to a table-structure
model (e.g. Microsoft Table Transformer) that predicts cell grids directly.
The heuristic here is the right default for ruled, single-header enterprise
tables (AP registers, trial balances) which dominate finance/compliance corpora
-- the exact use case behind the Universal Document Ingestor.

Row-oriented emission is the key RAG insight: a table row is only retrievable if
it carries its own column keys. Do that at ingestion, not query time.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "table_across_pages.pdf")


class TableStitcher:
    def _page_tables(self, pdf) -> List[List[List[str]]]:
        tables = []
        for page in pdf.pages:
            for t in page.extract_tables():
                cleaned = [[(c or "").strip() for c in row] for row in t if any(row)]
                if cleaned:
                    tables.append(cleaned)
        return tables

    def _same_header(self, a: List[List[str]], b: List[List[str]]) -> bool:
        return bool(a) and bool(b) and a[0] == b[0] and len(a[0]) == len(b[0])

    def stitch(self, tables: List[List[List[str]]]) -> List[List[List[str]]]:
        """Merge consecutive tables that share a header (page continuations)."""
        merged: List[List[List[str]]] = []
        for t in tables:
            if merged and self._same_header(merged[-1], t):
                merged[-1].extend(t[1:])   # drop the repeated header row
            else:
                merged.append([r[:] for r in t])
        return merged

    def to_records(self, table: List[List[str]]) -> List[Dict[str, str]]:
        header, *rows = table
        return [dict(zip(header, r)) for r in rows]


def naive_per_page(pdf_path: str) -> List[List[List[str]]]:
    with pdfplumber.open(pdf_path) as pdf:
        return TableStitcher()._page_tables(pdf)


if __name__ == "__main__":
    if not os.path.exists(FIX):
        from fixtures.generate import make_table_across_pages
        make_table_across_pages()

    raw = naive_per_page(FIX)
    print("=" * 70)
    print(f"NAIVE per-page extraction: {len(raw)} separate tables "
          f"(rows: {[len(t) - 1 for t in raw]}), header repeated on page 2:")
    print("=" * 70)
    for i, t in enumerate(raw):
        print(f"  table {i}: header={t[0]}  #datarows={len(t) - 1}")

    stitched = TableStitcher().stitch(raw)
    print("\n" + "=" * 70)
    print(f"STITCHED: {len(stitched)} logical table(s); page-2 header dropped:")
    print("=" * 70)
    records = TableStitcher().to_records(stitched[0])
    print(f"  {len(records)} row-records emitted (each a self-describing chunk):")
    for rec in records:
        print("   ", ", ".join(f"{k}={v}" for k, v in rec.items()))
