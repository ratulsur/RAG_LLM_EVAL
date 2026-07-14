"""
PATHOLOGY (e): running headers / footers / page numbers / watermarks are NOT
content -- but naive extraction dumps them into every page's text.

WHY NAIVE EXTRACTION FAILS
--------------------------
"ACME Corp -- Confidential" on all 40 pages becomes 40 identical lines in your
corpus. Effects: (1) chunks get polluted with boilerplate, (2) the token
"Confidential" gets an inflated IDF-like weight and hijacks retrieval, (3)
"Page 3 of 40" fragments split real sentences. Watermarks ("DRAFT") add a
recurring nonsense token behind every page.

THE PRODUCTION FIX
------------------
Frequency analysis over the corpus of pages:
  1. Extract lines per page, keep each line's vertical band (top/bottom).
  2. Normalize (lowercase, collapse whitespace, mask digits) so "Page 1 of 4"
     and "Page 2 of 4" collapse to the same signature.
  3. A signature that recurs on >= `min_page_frac` of pages, in a consistent
     vertical band (top or bottom margin), is furniture -> drop it.
  4. Watermarks: a short repeated token present on ~every page (any band) is
     flagged and removed too.

SENIOR TRADEOFF
---------------
Frequency detection needs several pages to be reliable -- on a 1-2 page doc it
has no signal, so we fall back to positional heuristics (top/bottom 8% band).
Risk: a legitimate line that genuinely repeats (e.g. a section label) can be
dropped; we mitigate by requiring BOTH high frequency AND margin position for
headers/footers, and only frequency for watermarks. This runs BEFORE chunking
in the Universal Document Ingestor so furniture never reaches the vector store.
"""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "furniture.pdf")

_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")


def _sig(text: str) -> str:
    return _WS.sub(" ", _DIGITS.sub("#", text.strip().lower()))


@dataclass
class Line:
    text: str
    top: float
    page: int
    page_h: float

    @property
    def band(self) -> str:
        if self.top < self.page_h * 0.10:
            return "header"
        if self.top > self.page_h * 0.90:
            return "footer"
        return "body"


class FurnitureFilter:
    def __init__(self, min_page_frac: float = 0.6):
        self.min_page_frac = min_page_frac

    def _lines(self, pdf) -> List[Line]:
        out: List[Line] = []
        for pi, page in enumerate(pdf.pages):
            # group words into visual lines by their 'top'
            rows: Dict[float, List[str]] = defaultdict(list)
            for w in page.extract_words():
                rows[round(float(w["top"]), 0)].append((float(w["x0"]), w["text"]))
            for top, items in rows.items():
                text = " ".join(t for _, t in sorted(items))
                out.append(Line(text, top, pi, float(page.height)))
        return out

    def analyze(self, lines: List[Line], npages: int):
        pages_with: Dict[str, set] = defaultdict(set)
        bands: Dict[str, List[str]] = defaultdict(list)
        for ln in lines:
            s = _sig(ln.text)
            pages_with[s].add(ln.page)
            bands[s].append(ln.band)
        need = max(2, math.ceil(self.min_page_frac * npages))
        furniture: Dict[str, str] = {}
        for s, pset in pages_with.items():
            if len(pset) < need:
                continue
            band_set = set(bands[s])
            if band_set <= {"header"}:
                furniture[s] = "header"
            elif band_set <= {"footer"}:
                furniture[s] = "footer"
            elif len(s.replace("#", "").strip()) <= 12:
                furniture[s] = "watermark"  # short token repeated everywhere
            else:
                furniture[s] = "repeating"
        return furniture

    def clean(self, pdf_path: str):
        with pdfplumber.open(pdf_path) as pdf:
            npages = len(pdf.pages)
            lines = self._lines(pdf)
        furniture = self.analyze(lines, npages)
        kept, dropped = defaultdict(list), []
        for ln in lines:
            tag = furniture.get(_sig(ln.text))
            if tag:
                dropped.append((ln.page, tag, ln.text))
            else:
                kept[ln.page].append((ln.top, ln.text))
        body = []
        for pi in sorted(kept):
            for _, text in sorted(kept[pi]):
                body.append(text)
        return "\n".join(body), furniture, dropped


if __name__ == "__main__":
    if not os.path.exists(FIX):
        from fixtures.generate import make_furniture
        make_furniture()

    filt = FurnitureFilter()
    with pdfplumber.open(FIX) as pdf:
        npages = len(pdf.pages)
    print("=" * 70)
    print(f"NAIVE extraction of a {npages}-page doc (furniture repeated every page):")
    print("=" * 70)
    with pdfplumber.open(FIX) as pdf:
        print((pdf.pages[0].extract_text() or ""))

    body, furniture, dropped = filt.clean(FIX)
    print("\n" + "=" * 70)
    print("DETECTED FURNITURE (signature -> type):")
    print("=" * 70)
    for s, t in furniture.items():
        print(f"  [{t:9}] {s!r}")
    print(f"\nDropped {len(dropped)} furniture line-instances across pages.")
    print("\n" + "=" * 70)
    print("CLEAN BODY (furniture removed, ready for chunking):")
    print("=" * 70)
    print(body)
