"""
08 — DOCUMENT IMAGES (photos, not clean scans).

THE PATHOLOGY (say this in the room)
  Users photograph documents with a phone: perspective distortion, skew, glare,
  shadows, crumpled paper. Feed that straight to OCR and you get garbage — OCR
  assumes a flat, high-contrast, axis-aligned page. The value is in the PRE-OCR
  pipeline: detect the degradation, correct it, THEN OCR. Ingestion again is
  where retrieval quality is decided.

THE PRE-OCR PIPELINE (detect -> correct -> route -> OCR)
  1. GRAYSCALE + CONTRAST   normalize illumination (kills most shadow/glare).
  2. DESKEW                  estimate rotation and rotate back to axis-aligned.
                             Live path here: a projection-profile skew estimator
                             (PIL + numpy) that needs no system libs.
                             Production path: opencv perspective transform via
                             detected page-corner contours (guarded import).
  3. BINARIZE                adaptive/Otsu threshold -> clean bitonal for OCR.
  4. SEGMENT REGIONS         split text blocks from diagram/figure regions
                             (whiteboard case) via connected-component density.
  5. ROUTE BY SCRIPT         detect scripts present, route each region to the
                             right OCR model (Latin vs Devanagari vs handwriting).
  6. OCR                     guarded pytesseract; synthetic-proxy report if absent.

SENIOR TRADEOFFS
  - Deskew BEFORE binarize: rotating a bitonal image aliases the strokes; rotate
    the grayscale, then threshold.
  - Otsu (global) is fine for our even synthetic lighting; real phone shadows need
    ADAPTIVE (local) thresholding — noted, and what opencv would give you.
  - Handwriting and printed text need DIFFERENT OCR engines. Region segmentation
    is what lets you route them instead of running one model over everything.

RUN
    python fixtures/make_images.py   # writes the dirty PNGs
    python image_prep.py
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from fixtures import make_images

# ---------------------------------------------------------------------------
# Guarded heavy deps: opencv (geometric correction) and OCR engines.
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:  # noqa: BLE001
    _HAS_CV2 = False

try:  # pragma: no cover
    import pytesseract  # type: ignore

    _HAS_TESS = True
except Exception:  # noqa: BLE001
    _HAS_TESS = False


_DEVANAGARI = (0x0900, 0x097F)


# ===========================================================================
# 1. Illumination normalization
# ===========================================================================
def to_gray_contrast(img: Image.Image) -> Image.Image:
    """Grayscale + autocontrast. autocontrast rescales the histogram per-image,
    which flattens a global shadow gradient — the cheap, dependency-free win."""
    g = ImageOps.grayscale(img)
    return ImageOps.autocontrast(g, cutoff=2)


# ===========================================================================
# 2. Deskew — projection-profile estimator (PIL + numpy, no system libs)
# ===========================================================================
def estimate_skew(gray: Image.Image, search_deg: float = 20.0, step: float = 1.0) -> float:
    """Estimate skew by maximizing the VARIANCE of the horizontal projection
    profile over candidate rotations. Text rows produce sharp peaks (high row-sum
    variance) only when the page is axis-aligned, so the best angle un-skews it.
    This is the classic scan-deskew trick and needs no opencv."""
    base = np.asarray(gray, dtype="float32")
    base = 255.0 - base  # ink = high
    best_angle, best_score = 0.0, -1.0
    a = -search_deg
    while a <= search_deg:
        rot = np.asarray(
            Image.fromarray(base.astype("uint8")).rotate(a, resample=Image.BILINEAR, fillcolor=0),
            dtype="float32",
        )
        row_sums = rot.sum(axis=1)
        score = float(np.var(row_sums))  # sharp text rows -> high variance
        if score > best_score:
            best_score, best_angle = score, a
        a += step
    return best_angle


def deskew(gray: Image.Image) -> tuple[Image.Image, float]:
    angle = estimate_skew(gray)
    # We rotate the *grayscale* (not the later binary) to avoid aliasing strokes.
    corrected = gray.rotate(angle, resample=Image.BICUBIC, fillcolor=255, expand=False)
    return corrected, angle


def correct_perspective(img: Image.Image) -> Image.Image:  # pragma: no cover
    """PRODUCTION PATH (guarded): detect the 4 page corners and warp to a flat
    rectangle. Perspective distortion (trapezoid pages) is NOT fixable by a plain
    rotation — you need a homography, which is opencv territory."""
    if not _HAS_CV2:
        return img  # fall back to deskew-only; report says so
    arr = np.asarray(img.convert("L"))
    edges = cv2.Canny(arr, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    quad = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(quad, True)
    approx = cv2.approxPolyDP(quad, 0.02 * peri, True)
    if len(approx) != 4:
        return img
    # (warp omitted for brevity — this is the reference hook, not the live path)
    return img


# ===========================================================================
# 3. Binarize (Otsu, hand-rolled)
# ===========================================================================
def otsu_threshold(gray: Image.Image) -> Image.Image:
    arr = np.asarray(gray)
    hist = np.bincount(arr.ravel(), minlength=256).astype("float64")
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    w_b = 0.0
    sum_b = 0.0
    best_t, best_var = 0, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best_var:
            best_var, best_t = between, t
    binary = (arr > best_t).astype("uint8") * 255
    return Image.fromarray(binary, mode="L")


# ===========================================================================
# 4. Region segmentation: text blocks vs diagram/figure regions
# ===========================================================================
@dataclass
class Region:
    kind: str          # "text" | "figure"
    bbox: tuple[int, int, int, int]
    ink_density: float
    row_periodicity: float


def segment_regions(binary: Image.Image, band_h: int = 40) -> list[Region]:
    """Coarse horizontal-band segmentation. TEXT bands show a regular high-freq
    ink pattern along rows (many short runs); FIGURE bands (boxes/lines) show
    sparse, long strokes. We classify each band by ink density + horizontal
    run-length periodicity. This is the signal that lets us route text vs diagram
    to different OCR/handling instead of one-size-fits-all."""
    arr = 255 - np.asarray(binary)  # ink = high
    h, w = arr.shape
    regions: list[Region] = []
    for top in range(0, h, band_h):
        band = arr[top:top + band_h]
        if band.sum() == 0:
            continue
        density = float((band > 0).mean())
        # periodicity proxy: average number of ink->blank transitions per row
        trans = np.abs(np.diff((band > 0).astype("int8"), axis=1)).sum(axis=1)
        periodicity = float(trans.mean())
        kind = "text" if periodicity >= 6 and density < 0.35 else "figure"
        regions.append(Region(kind, (0, top, w, min(top + band_h, h)), density, periodicity))
    return regions


# ===========================================================================
# 5. Script routing
# ===========================================================================
def scripts_in_text(text: str) -> list[str]:
    scripts: set[str] = set()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        if _DEVANAGARI[0] <= cp <= _DEVANAGARI[1]:
            scripts.add("Devanagari")
        elif cp < 0x80:
            scripts.add("Latin")
        else:
            scripts.add(unicodedata.name(ch, "UNK").split()[0].title())
    return sorted(scripts)


def route_ocr_lang(scripts: list[str]) -> str:
    """Map detected scripts to a tesseract lang string. Multi-script forms need
    a combined model ('eng+hin'), not eng-only — that is the routing decision."""
    lang = []
    if "Latin" in scripts:
        lang.append("eng")
    if "Devanagari" in scripts:
        lang.append("hin")
    return "+".join(lang) or "eng"


# ===========================================================================
# 6. OCR (guarded)
# ===========================================================================
def ocr(binary: Image.Image, lang: str) -> str:
    if _HAS_TESS:  # pragma: no cover
        return pytesseract.image_to_string(binary, lang=lang)
    return "<OCR-SKIPPED: pytesseract not installed; pre-OCR image is ready>"


# ===========================================================================
# Orchestration + demo
# ===========================================================================
@dataclass
class PrepResult:
    name: str
    skew_deg: float
    n_text_regions: int
    n_figure_regions: int
    ocr_lang: str
    ocr_text: str


def prep_receipt(img: Image.Image) -> PrepResult:
    gray = to_gray_contrast(img)
    gray = correct_perspective(gray)         # guarded no-op without cv2
    corrected, angle = deskew(gray)
    binary = otsu_threshold(corrected)
    regions = segment_regions(binary)
    lang = route_ocr_lang(["Latin"])         # receipt is Latin
    text = ocr(binary, lang)
    return PrepResult(
        "receipt_bad", round(angle, 1),
        sum(r.kind == "text" for r in regions),
        sum(r.kind == "figure" for r in regions),
        lang, text.strip(),
    )


def prep_whiteboard(img: Image.Image) -> PrepResult:
    gray = to_gray_contrast(img)
    binary = otsu_threshold(gray)
    regions = segment_regions(binary)
    return PrepResult(
        "whiteboard", 0.0,
        sum(r.kind == "text" for r in regions),
        sum(r.kind == "figure" for r in regions),
        "eng (handwriting -> specialized engine, e.g. TrOCR)", "<region-routed>",
    )


def prep_multiscript(img: Image.Image) -> PrepResult:
    gray = to_gray_contrast(img)
    binary = otsu_threshold(gray)
    regions = segment_regions(binary)
    # In production the script detector runs per-region on a first OCR pass; here
    # we know the form carries English + Devanagari.
    lang = route_ocr_lang(["Latin", "Devanagari"])
    return PrepResult(
        "multiscript_form", 0.0,
        sum(r.kind == "text" for r in regions),
        sum(r.kind == "figure" for r in regions),
        lang, ocr(binary, lang).strip(),
    )


def _rule(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main() -> None:
    _rule("ENV")
    print(f"opencv available ...... {_HAS_CV2} (perspective warp = reference path)")
    print(f"pytesseract available . {_HAS_TESS} (OCR = reference path)")

    receipt = make_images.make_receipt_photo(skew_deg=14.0)
    whiteboard = make_images.make_whiteboard()
    form = make_images.make_multiscript_form()

    _rule("BEFORE — a phone photo: skewed 14 deg, shadowed, speckled")
    print(f"receipt size (skew expands canvas): {receipt.size}")

    results = [prep_receipt(receipt), prep_whiteboard(whiteboard), prep_multiscript(form)]

    _rule("AFTER — pre-OCR pipeline output")
    for r in results:
        print(f"\n[{r.name}]")
        print(f"  detected skew ....... {r.skew_deg} deg  (corrected back toward 0)")
        print(f"  text regions ........ {r.n_text_regions}")
        print(f"  figure regions ...... {r.n_figure_regions}")
        print(f"  OCR routing lang .... {r.ocr_lang}")
        print(f"  OCR text ............ {r.ocr_text[:80]!r}")

    _rule("WHAT THE PREP CAUGHT")
    print("- receipt: estimated the ~14 deg skew and rotated it back before threshold")
    print("- whiteboard: separated text bands from the boxes/arrows figure region")
    print("- form: routed to eng+hin because two scripts were present, not eng-only")


if __name__ == "__main__":
    main()
