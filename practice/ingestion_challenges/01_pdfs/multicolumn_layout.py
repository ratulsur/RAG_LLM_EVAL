"""
PATHOLOGY (b): multi-column PDF -> naive extraction interleaves the columns.

WHY NAIVE EXTRACTION FAILS
--------------------------
A PDF stores glyphs in *draw order*, not *reading order*. Most extractors emit
words roughly top-to-bottom, so on a two-column page they zig-zag across the
gutter: "line 1 left, line 1 right, line 2 left, line 2 right ...". The result
is grammatically shredded text that no reranker or LLM can un-scramble. In a
RAG pipeline this poisons chunks at the source.

THE PRODUCTION FIX
------------------
Extract words WITH bounding boxes, then recover layout geometrically via a
vertical projection profile (a 1-D X-Y cut):
  1. Paint every word's x-span [x0, x1] onto a horizontal occupancy histogram.
  2. Find the widest EMPTY x-band away from the margins -> that is the gutter.
  3. Split columns at the gutter, sort each top-to-bottom, read left-to-right.

Note: naive x0-only clustering does NOT work here -- a column's words fill a
wide range of x0 values, so there is no gap in the x0 histogram. The empty band
is between the columns, which is why we project word SPANS, not just starts.

SENIOR TRADEOFF
---------------
Projection-profile cuts are fast and library-free but assume a stable column
grid. Complex
magazine layouts, spanning figures, or sidebars need a full layout model
(LayoutParser / a detectron layout net). For 2-3 column reports -- invoices,
filings, contracts, the bulk of enterprise docs -- geometric clustering is the
right cost/accuracy point. Escalate to a layout model only when detected column
count is unstable across pages.

Maps to Universal Document Ingestor: this is the "reading-order normalizer"
that runs before chunking so clause boundaries stay intact.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "multicolumn.pdf")


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    top: float


class ColumnReorderer:
    """Recovers reading order on multi-column pages via a projection profile."""

    def __init__(self, page_width: float, bin_pt: float = 3.0,
                 min_gutter_pt: float = 30.0, margin_frac: float = 0.12):
        self.page_width = page_width
        self.bin_pt = bin_pt              # histogram resolution
        self.min_gutter_pt = min_gutter_pt  # narrowest band we'll call a gutter
        self.margin_frac = margin_frac    # ignore empty bands in the page margins

    def _occupancy(self, words: List[Word]) -> List[bool]:
        nbins = int(self.page_width / self.bin_pt) + 1
        occ = [False] * nbins
        for w in words:
            b0 = max(0, int(w.x0 / self.bin_pt))
            b1 = min(nbins - 1, int(w.x1 / self.bin_pt))
            for b in range(b0, b1 + 1):
                occ[b] = True
        return occ

    def _gutter_centers(self, occ: List[bool]) -> List[float]:
        """Empty runs wide enough, and not in the margins, are gutters."""
        lo = int(len(occ) * self.margin_frac)
        hi = int(len(occ) * (1 - self.margin_frac))
        centers, run_start = [], None
        for i in range(lo, hi + 1):
            empty = (i < len(occ)) and not occ[i]
            if empty and run_start is None:
                run_start = i
            elif not empty and run_start is not None:
                if (i - run_start) * self.bin_pt >= self.min_gutter_pt:
                    centers.append((run_start + i) / 2 * self.bin_pt)
                run_start = None
        return centers

    def reorder(self, words: List[Word]):
        if not words:
            return "", 1
        gutters = self._gutter_centers(self._occupancy(words))
        edges = [0.0] + gutters + [self.page_width]
        ncols = len(edges) - 1
        cols: List[List[Word]] = [[] for _ in range(ncols)]
        for w in words:
            mid = (w.x0 + w.x1) / 2
            for ci in range(ncols):
                if edges[ci] <= mid < edges[ci + 1]:
                    cols[ci].append(w)
                    break
        out_lines: List[str] = []
        for col in cols:  # left-to-right
            col.sort(key=lambda w: (round(w.top, 1), w.x0))  # top-to-bottom
            out_lines.extend(self._group_lines(col))
        return "\n".join(out_lines), ncols

    @staticmethod
    def _group_lines(col: List[Word], y_tol: float = 4.0) -> List[str]:
        lines, cur, cur_top = [], [], None
        for w in col:
            if cur_top is None or abs(w.top - cur_top) <= y_tol:
                cur.append(w.text)
                cur_top = w.top if cur_top is None else cur_top
            else:
                lines.append(" ".join(cur))
                cur, cur_top = [w.text], w.top
        if cur:
            lines.append(" ".join(cur))
        return lines


def load_words(pdf_path: str):
    """Return (words, page_width) for the first page."""
    words: List[Word] = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        for w in page.extract_words():
            words.append(Word(w["text"], float(w["x0"]), float(w["x1"]), float(w["top"])))
        return words, float(page.width)


def naive_read(pdf_path: str) -> str:
    """What a layout-blind extractor produces: raw text, draw/scan order."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join((p.extract_text() or "") for p in pdf.pages)


if __name__ == "__main__":
    if not os.path.exists(FIX):
        from fixtures.generate import make_multicolumn
        make_multicolumn()

    print("=" * 70)
    print("NAIVE (layout-blind) -- columns interleave:")
    print("=" * 70)
    print(naive_read(FIX))

    words, page_w = load_words(FIX)
    fixed, ncols = ColumnReorderer(page_width=page_w).reorder(words)
    print("\n" + "=" * 70)
    print(f"FIXED (x-clustered, detected {ncols} columns) -- correct reading order:")
    print("=" * 70)
    print(fixed)
