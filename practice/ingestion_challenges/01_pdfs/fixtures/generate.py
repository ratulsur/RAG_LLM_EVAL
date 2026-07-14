"""
Synthetic dirty-PDF generator for the 01_pdfs ingestion challenges.

Everything here is produced with reportlab + Pillow so the handlers run with
NO external files. Each function returns the path it wrote. The pathologies are
deliberately baked in so detection code has something real to detect:

  multicolumn.pdf        two text columns (naive top-to-bottom read interleaves them)
  furniture.pdf          repeating header/footer/page-number + diagonal watermark, 4 pages
  table_across_pages.pdf a table whose rows continue past a page break, header repeats
  image_only.pdf         a page rasterised to an image -> NO text layer (needs OCR), skewed + a "stamp"
  mixed.pdf              page 1 native text, page 2 image-only (per-page routing)
  chart.pdf              a bar "chart" drawn as pixels -> numbers live in the image, not text

Run `python generate.py` to (re)build all of them into ./ (this dir).
"""
from __future__ import annotations

import io
import os

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE_W, PAGE_H = LETTER


def _out(name: str) -> str:
    return os.path.join(HERE, name)


# --------------------------------------------------------------------------
# (b) multi-column layout
# --------------------------------------------------------------------------
# kept narrow (~30 chars) so a real gutter exists between the columns
LEFT_COL = [
    "Retrieval quality is the",
    "biggest lever on RAG answer",
    "faithfulness. A wrong passage",
    "means the grounding grader",
    "can only reject, never repair.",
    "The Universal Document",
    "Ingestor spends its budget",
    "on extraction and chunking.",
]
RIGHT_COL = [
    "Multi-column PDFs break",
    "naive extraction: text is",
    "stored in draw order, not",
    "reading order. A glyph walk",
    "zig-zags across the gutter",
    "and yields interleaved",
    "nonsense no reranker can",
    "rescue. X-clustering fixes it.",
]


def make_multicolumn() -> str:
    path = _out("multicolumn.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1 * inch, PAGE_H - 1 * inch, "Two-Column Technical Note")
    c.setFont("Helvetica", 11)
    y0 = PAGE_H - 1.6 * inch
    lh = 0.28 * inch
    # left column starts at x=1in, right column at x=4.4in -> two clear x-clusters
    for i, line in enumerate(LEFT_COL):
        c.drawString(1.0 * inch, y0 - i * lh, line)
    for i, line in enumerate(RIGHT_COL):
        c.drawString(4.6 * inch, y0 - i * lh, line)
    c.showPage()
    c.save()
    return path


# --------------------------------------------------------------------------
# (e) repeating header/footer/page-number + watermark
# --------------------------------------------------------------------------
# Body varies per page (real docs do) so ONLY the header/footer/watermark
# recur -- that is exactly the signal the frequency detector must isolate.
BODY_PARAS = [
    "Production ingestion is where most RAG systems quietly fail. The model is fine; the "
    "pipeline fed it garbage. This is the opening argument on page one.",
    "Repeating page furniture -- running headers, footers, page numbers and watermarks -- is "
    "not content. Left in, it pollutes chunks and inflates term frequencies on page two.",
    "The fix is frequency analysis: any short line that recurs at the same vertical band on "
    "most pages is furniture, not prose, and is dropped before chunking. Page three explains.",
    "A watermark sits diagonally behind the text. Watermarks are often a separate content "
    "stream and can be filtered the same way once identified. Page four closes the note.",
]


def make_furniture(pages: int = 4) -> str:
    path = _out("furniture.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    for p in range(1, pages + 1):
        para = BODY_PARAS[(p - 1) % len(BODY_PARAS)]
        # running header (identical every page)
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(1 * inch, PAGE_H - 0.6 * inch, "ACME Corp -- Confidential -- Internal Use Only")
        # diagonal watermark
        c.saveState()
        c.setFont("Helvetica-Bold", 60)
        c.setFillGray(0.85)
        c.translate(PAGE_W / 2, PAGE_H / 2)
        c.rotate(45)
        c.drawCentredString(0, 0, "DRAFT")
        c.restoreState()
        # body (distinct per page)
        c.setFillGray(0.0)
        c.setFont("Helvetica", 11)
        y = PAGE_H - 1.4 * inch
        for chunk in _wrap(para, 92):
            c.drawString(1 * inch, y, chunk)
            y -= 0.24 * inch
        y -= 0.12 * inch
        # running footer + page number (footer identical, number varies)
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(1 * inch, 0.6 * inch, "(c) 2026 ACME Corp. All rights reserved.")
        c.drawRightString(PAGE_W - 1 * inch, 0.6 * inch, f"Page {p} of {pages}")
        c.showPage()
    c.save()
    return path


def _wrap(text: str, width: int):
    words, line, out = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


# --------------------------------------------------------------------------
# (c) table spanning a page break
# --------------------------------------------------------------------------
TABLE_HEADER = ["Invoice", "Vendor", "Amount", "Status"]
TABLE_ROWS = [
    ["INV-1001", "Northwind", "12,400.00", "Paid"],
    ["INV-1002", "Contoso", "8,900.50", "Pending"],
    ["INV-1003", "Fabrikam", "44,120.00", "Paid"],
    ["INV-1004", "Northwind", "1,050.00", "Disputed"],
    ["INV-1005", "Contoso", "76,000.00", "Paid"],
    ["INV-1006", "Tailspin", "3,300.25", "Pending"],
    ["INV-1007", "Fabrikam", "19,999.99", "Paid"],
    ["INV-1008", "Contoso", "60,500.00", "Paid"],
]


def _draw_grid_table(c, rows, x, y_top, col_w=1.5 * inch, row_h=0.32 * inch):
    """Draw a fully-ruled table (borders + gridlines) so pdfplumber's line
    strategy detects it. Returns the y of the bottom border."""
    ncols = max(len(r) for r in rows)
    nrows = len(rows)
    table_w = ncols * col_w
    y_bottom = y_top - nrows * row_h
    # horizontal rules
    for r in range(nrows + 1):
        yy = y_top - r * row_h
        c.line(x, yy, x + table_w, yy)
    # vertical rules
    for col in range(ncols + 1):
        xx = x + col * col_w
        c.line(xx, y_top, xx, y_bottom)
    # cell text (baseline padded inside the cell)
    for r, row in enumerate(rows):
        for i, cell in enumerate(row):
            c.drawString(x + i * col_w + 4, y_top - r * row_h - row_h + 8, str(cell))
    return y_bottom


def make_table_across_pages() -> str:
    path = _out("table_across_pages.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    # page 1: title + header + first 5 rows
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1 * inch, PAGE_H - 1 * inch, "Accounts Payable Register (continued table)")
    c.setFont("Helvetica", 10)
    _draw_grid_table(c, [TABLE_HEADER] + TABLE_ROWS[:5], 1 * inch, PAGE_H - 1.4 * inch)
    c.showPage()
    # page 2: header REPEATS (typical continuation), remaining rows -> stitch
    c.setFont("Helvetica", 10)
    _draw_grid_table(c, [TABLE_HEADER] + TABLE_ROWS[5:], 1 * inch, PAGE_H - 1.0 * inch)
    c.showPage()
    c.save()
    return path


# --------------------------------------------------------------------------
# (a) image-only page (no text layer) -- OCR required
# --------------------------------------------------------------------------
def _text_image(lines, skew=6, stamp=True, dpi_scale=1.0) -> Image.Image:
    W, H = int(1000 * dpi_scale), int(1300 * dpi_scale)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", int(26 * dpi_scale))
    except Exception:
        font = ImageFont.load_default()
    y = int(120 * dpi_scale)
    for ln in lines:
        d.text((int(90 * dpi_scale), y), ln, fill="black", font=font)
        y += int(46 * dpi_scale)
    if stamp:
        # a red "RECEIVED" stamp rectangle -- classic scanned-doc noise
        d.rectangle([int(600 * dpi_scale), int(80 * dpi_scale),
                     int(940 * dpi_scale), int(180 * dpi_scale)], outline="red", width=4)
        d.text((int(620 * dpi_scale), int(110 * dpi_scale)), "RECEIVED", fill="red", font=font)
    if skew:
        img = img.rotate(skew, expand=True, fillcolor="white")
    return img


def _image_page(c, img: Image.Image):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    from reportlab.lib.utils import ImageReader
    c.drawImage(ImageReader(buf), 0.5 * inch, 0.5 * inch,
                width=PAGE_W - 1 * inch, height=PAGE_H - 1 * inch, preserveAspectRatio=True)


def make_image_only() -> str:
    path = _out("image_only.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    img = _text_image([
        "SCANNED PURCHASE ORDER",
        "PO Number: 55231",
        "Vendor: Globex International",
        "Total Due: USD 42,000",
        "This page has NO text layer.",
        "Every glyph is a pixel -> OCR only.",
    ])
    _image_page(c, img)
    c.showPage()
    c.save()
    return path


# --------------------------------------------------------------------------
# (d) mixed: native text page + image-only page in one document
# --------------------------------------------------------------------------
def make_mixed() -> str:
    path = _out("mixed.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    # page 1 native
    c.setFont("Helvetica-Bold", 13)
    c.drawString(1 * inch, PAGE_H - 1 * inch, "Contract -- Section 1 (native text)")
    c.setFont("Helvetica", 11)
    y = PAGE_H - 1.5 * inch
    for para in BODY_PARAS[:3]:
        for chunk in _wrap(para, 92):
            c.drawString(1 * inch, y, chunk)
            y -= 0.24 * inch
        y -= 0.12 * inch
    c.showPage()
    # page 2 image-only (a scanned signature appendix)
    img = _text_image([
        "APPENDIX A -- SIGNED",
        "This page was scanned from paper.",
        "Signature: /s/ J. Rivera",
        "Dated: 2026-07-14",
    ], skew=3)
    _image_page(c, img)
    c.showPage()
    c.save()
    return path


# --------------------------------------------------------------------------
# (f) chart where the numbers are pixels
# --------------------------------------------------------------------------
def make_chart() -> str:
    path = _out("chart.pdf")
    c = canvas.Canvas(path, pagesize=LETTER)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(1 * inch, PAGE_H - 1 * inch, "Quarterly Revenue (figure below is an image)")
    # build a bar chart as a raster image
    W, H = 900, 500
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    bars = [("Q1", 120), ("Q2", 200), ("Q3", 160), ("Q4", 260)]
    base_y = 420
    for i, (label, val) in enumerate(bars):
        x = 120 + i * 180
        d.rectangle([x, base_y - val, x + 90, base_y], fill="steelblue")
        d.text((x + 20, base_y + 10), label, fill="black", font=font)
        d.text((x + 10, base_y - val - 30), f"${val}k", fill="black", font=font)  # value = pixels
    _image_page(c, img)
    c.showPage()
    c.save()
    return path


def build_all():
    paths = [
        make_multicolumn(),
        make_furniture(),
        make_table_across_pages(),
        make_image_only(),
        make_mixed(),
        make_chart(),
    ]
    for p in paths:
        print("wrote", os.path.relpath(p, HERE), f"({os.path.getsize(p)} bytes)")
    return paths


if __name__ == "__main__":
    build_all()
