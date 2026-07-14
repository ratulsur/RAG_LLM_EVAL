"""
PATHOLOGIES (a) image-only/scanned pages, (d) mixed native+scanned in one doc,
(f) charts where the numbers live in pixels.

WHY NAIVE EXTRACTION FAILS
--------------------------
`page.get_text()` on a scanned page returns "" -- there is no text layer, every
glyph is a pixel. Pipelines that trust the extractor silently ingest EMPTY pages
and never know a whole appendix went missing. Worse is the MIXED doc: page 1 is
native text, page 2 is a scan; a one-size extractor drops half the content. And
a chart embedded as an image hides its numbers from text extraction entirely.

THE PRODUCTION FIX -- per-page routing
--------------------------------------
Classify EVERY page, then dispatch:
  NATIVE        text layer present            -> extract text (fast, exact)
  IMAGE_ONLY    ~no text + full-page image     -> OCR route (rasterize -> tesseract)
  MIXED_FIGURE  text present + large image     -> extract text AND flag the figure
                                                  region as "data-in-pixels" for a
                                                  vision/OCR pass
Signal used:  text_chars, image area fraction of the page.

OCR is a GUARDED-OPTIONAL dependency here (tesseract binary + pytesseract are not
installed in this env). The routing + rasterization pipeline still runs end-to-end
so you can see the decision; the OCR call is attempted and, if the binary is
absent, we surface the exact install note instead of failing. That is the honest
production posture: detect, route, and degrade loudly -- never silently.

SENIOR TRADEOFFS
----------------
- Threshold tuning: a page with a tiny caption but a full-page scan still needs
  OCR, so we key on the image AREA fraction, not raw char count alone.
- OCR cost: it is 10-100x slower than text extraction and error-prone on skew /
  low DPI / stamps, so we OCR ONLY pages that need it -- never the whole PDF.
- Charts: OCR recovers axis labels but not the plotted values; genuine
  chart-data recovery needs a vision model. We flag, we don't pretend.

Maps to Universal Document Ingestor: this is the ingestion "router" that decides
native-extract vs OCR vs vision per page, so no content is lost and no page is
OCR'd needlessly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import fitz  # pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
FIX_DIR = os.path.join(HERE, "fixtures")


class PageKind(str, Enum):
    NATIVE = "NATIVE"
    IMAGE_ONLY = "IMAGE_ONLY"
    MIXED_FIGURE = "MIXED_FIGURE"


@dataclass
class PageDecision:
    index: int
    kind: PageKind
    text_chars: int
    image_area_frac: float
    action: str


class PageRouter:
    def __init__(self, min_text_chars: int = 20,
                 image_dominant_frac: float = 0.55,
                 figure_frac: float = 0.15):
        self.min_text_chars = min_text_chars
        self.image_dominant_frac = image_dominant_frac  # full-page scan
        self.figure_frac = figure_frac                   # embedded figure

    def _image_area_frac(self, page) -> float:
        page_area = abs(page.rect.width * page.rect.height) or 1.0
        covered = 0.0
        for info in page.get_image_info():
            bbox = info.get("bbox")
            if bbox:
                x0, y0, x1, y1 = bbox
                covered += abs((x1 - x0) * (y1 - y0))
        return min(covered / page_area, 1.0)

    def classify(self, page, index: int) -> PageDecision:
        text = page.get_text().strip()
        nchars = len(text)
        img_frac = self._image_area_frac(page)
        if nchars < self.min_text_chars and img_frac >= self.image_dominant_frac:
            kind, action = PageKind.IMAGE_ONLY, "OCR route (rasterize -> tesseract)"
        elif nchars >= self.min_text_chars and img_frac >= self.figure_frac:
            kind, action = PageKind.MIXED_FIGURE, "extract text + flag figure as data-in-pixels"
        else:
            kind, action = PageKind.NATIVE, "extract text layer"
        return PageDecision(index, kind, nchars, round(img_frac, 3), action)

    def process(self, pdf_path: str):
        doc = fitz.open(pdf_path)
        out = []
        for i, page in enumerate(doc):
            d = self.classify(page, i)
            if d.kind is PageKind.NATIVE:
                content = page.get_text().strip()
            elif d.kind is PageKind.IMAGE_ONLY:
                content = ocr_page(page)
            else:  # MIXED_FIGURE
                content = page.get_text().strip() + "\n[FIGURE flagged for vision/OCR]"
            out.append((d, content))
        doc.close()
        return out


def ocr_page(page, dpi: int = 200) -> str:
    """Rasterize the page and OCR it. Guarded: works if pytesseract+tesseract
    are present; otherwise returns a labelled stub with the install note.
    The rasterization ALWAYS runs so the routing pipeline is demonstrated live.
    """
    pix = page.get_pixmap(dpi=dpi)  # this always works -> proves the OCR input is ready
    dims = f"{pix.width}x{pix.height}px @ {dpi}dpi"
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return f"[OCR-STUB] rasterized {dims}; pip install pytesseract Pillow to enable OCR"
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    try:
        text = pytesseract.image_to_string(img).strip()
        return text or f"[OCR ran on {dims} but returned no text]"
    except Exception as e:  # TesseractNotFoundError etc.
        return (f"[OCR-STUB] rasterized {dims}; tesseract binary not found "
                f"({type(e).__name__}). Install: brew install tesseract")


if __name__ == "__main__":
    from fixtures import generate as g
    if not os.path.exists(os.path.join(FIX_DIR, "image_only.pdf")):
        g.build_all()

    router = PageRouter()
    for name in ["image_only.pdf", "mixed.pdf", "chart.pdf", "multicolumn.pdf"]:
        path = os.path.join(FIX_DIR, name)
        if not os.path.exists(path):
            continue
        print("=" * 70)
        print(f"{name}")
        print("=" * 70)
        for d, content in router.process(path):
            print(f"  page {d.index}: {d.kind.value:12} "
                  f"chars={d.text_chars:<4} img_frac={d.image_area_frac:<5} "
                  f"-> {d.action}")
            snippet = content.replace("\n", " ")[:90]
            print(f"      content: {snippet}")
