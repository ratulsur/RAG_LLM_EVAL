# 08 — Document Images (photos, not clean scans)

**Maps to:** Universal Document Ingestor — the image source. Phone-captured docs
are the failure mode teams underestimate: OCR assumes a flat, high-contrast,
axis-aligned page, and a photo is none of those.

## The pathology (what breaks)
Perspective distortion, skew, glare, shadows, crumpled paper. Feed a raw phone
photo to OCR → garbage. **The value is the pre-OCR pipeline**, not the OCR call.

## Pre-OCR pipeline (detect → correct → route → OCR)
| Stage | What it does | Path |
|---|---|---|
| Grayscale + autocontrast | Rescales histogram per-image → flattens shadow gradient | **live** (PIL) |
| Deskew | Projection-profile skew estimator: maximizes horizontal row-sum variance over candidate rotations | **live** (PIL+numpy) |
| Perspective warp | 4-corner homography for trapezoid pages (rotation can't fix this) | reference (guarded `cv2`) |
| Binarize | Hand-rolled Otsu threshold → clean bitonal | **live** (numpy) |
| Region segmentation | Text bands vs figure/diagram bands via ink density + run-length periodicity | **live** |
| Script routing | Detect scripts present → tesseract lang (`eng`, `eng+hin`) | **live** |
| OCR | printed → tesseract; handwriting → specialized engine (TrOCR) | reference (guarded `pytesseract`) |

## Proof it works (from the live run)
Synthetic receipt is generated clean, then **skewed +14°**, shadowed, and speckled.
The projection-profile estimator recovers **−14°** and rotates it back before
thresholding. The whiteboard fixture's boxes/arrows are separated from its text
bands; the bilingual form routes to **`eng+hin`**, not `eng`-only.

## Senior tradeoffs to say out loud
- **Deskew on grayscale, threshold after.** Rotating a bitonal image aliases the
  strokes; rotate the grayscale, then binarize.
- **Otsu (global) suits even lighting; real phone shadows need adaptive (local)
  thresholding** — that's what opencv buys you.
- **Handwriting ≠ printed text** — different OCR engines. Region segmentation is
  what lets you route them instead of running one model over everything.

## Live vs reference
- **Live (runs offline, PIL+numpy):** contrast normalization, projection-profile
  deskew, Otsu binarization, text-vs-figure region segmentation, script routing.
- **Reference (guarded):** `cv2` perspective/homography warp; `pytesseract` /
  `easyocr` OCR; TrOCR-style handwriting OCR. Code detects them and reports when
  absent — the pre-OCR image is still produced.

## Run
```bash
python fixtures/make_images.py   # writes receipt_bad.png, whiteboard.png, multiscript_form.png
python image_prep.py             # runs the pre-OCR pipeline + before/after
```

## Optional deps (not required)
`opencv-python` (`cv2`), `pytesseract` or `easyocr`. Everything runs without them.
