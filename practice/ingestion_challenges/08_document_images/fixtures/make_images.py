"""
Fixture generator: SYNTHETIC bad-photo document images (PIL only).

We can't ship real receipt photos, so we synthesize the *degradations* a phone
camera introduces, then let the pre-OCR pipeline try to undo them:

  make_receipt_photo()   -> a "receipt" printed cleanly, then ROTATED (skew),
                            darkened on one side (SHADOW), and speckled (NOISE).
  make_whiteboard()      -> handwriting-ish text + a boxed "diagram" region, to
                            demonstrate text-vs-diagram region segmentation.
  make_multiscript_form()-> English + a Devanagari-glyph band on one form, to
                            demonstrate multi-script region routing.

Everything returns a PIL.Image so the prep code runs with zero external files.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

random.seed(7)

_W, _H = 600, 400


def _blank(color: int = 245) -> Image.Image:
    return Image.new("L", (_W, _H), color=color)


def _speckle(img: Image.Image, amount: float = 0.04) -> Image.Image:
    img = img.copy()  # arrays from np can be read-only
    px = img.load()
    for _ in range(int(_W * _H * amount)):
        x, y = random.randint(0, _W - 1), random.randint(0, _H - 1)
        px[x, y] = random.choice((0, 255))
    return img


def make_receipt_photo(skew_deg: float = 14.0) -> Image.Image:
    """Clean receipt -> skew + shadow + noise = a realistic bad phone capture."""
    img = _blank(248)
    d = ImageDraw.Draw(img)
    lines = [
        "SHOPRITE MART",
        "GST: 27ABCDE1234F1Z5",
        "-----------------------",
        "Rice 5kg        450.00",
        "Milk 1L          62.00",
        "Bread            40.00",
        "-----------------------",
        "TOTAL           552.00",
        "2026-07-14  14:32",
    ]
    for i, ln in enumerate(lines):
        d.text((90, 40 + i * 34), ln, fill=20)

    # shadow: subtract a blurred triangular mask
    shadow = Image.new("L", img.size, 0)
    ImageDraw.Draw(shadow).polygon([(0, 0), (int(_W * 0.85), 0), (0, _H)], fill=90)
    shadow = shadow.filter(ImageFilter.GaussianBlur(90))
    img = Image.eval(img, lambda p: p)  # ensure L
    import numpy as np

    arr = np.asarray(img, dtype="int16") - np.asarray(shadow, dtype="int16")
    img = Image.fromarray(arr.clip(0, 255).astype("uint8"), mode="L")

    img = _speckle(img, amount=0.03)
    # skew LAST so the prep pipeline has to detect + undo the rotation.
    img = img.rotate(skew_deg, expand=True, fillcolor=255, resample=Image.BICUBIC)
    return img


def make_whiteboard() -> Image.Image:
    img = _blank(250)
    d = ImageDraw.Draw(img)
    # "handwriting" text region (top)
    for i, ln in enumerate(["Sprint goals:", "- ship ingest v2", "- fix dedup"]):
        d.text((40, 30 + i * 30), ln, fill=15)
    # "diagram" region (bottom): boxes + arrows, NOT text
    d.rectangle([60, 200, 180, 260], outline=10, width=3)
    d.rectangle([360, 200, 480, 260], outline=10, width=3)
    d.line([180, 230, 360, 230], fill=10, width=3)
    d.polygon([(360, 230), (345, 222), (345, 238)], fill=10)
    return img


def make_multiscript_form() -> Image.Image:
    img = _blank(252)
    d = ImageDraw.Draw(img)
    d.text((40, 30), "APPLICATION FORM (English)", fill=15)
    d.text((40, 70), "Name: ____________________", fill=15)
    # Devanagari band (real glyphs; PIL default font may render as boxes but the
    # unicode codepoints are what the script router keys on if OCR'd).
    d.text((40, 160), "आवेदन पत्र", fill=15)
    d.text((40, 200), "नाम: ____________________", fill=15)
    return img


def write_all(dirpath: str | None = None) -> list[Path]:
    out_dir = Path(dirpath) if dirpath else Path(__file__).parent
    paths = []
    for name, fn in [
        ("receipt_bad.png", make_receipt_photo),
        ("whiteboard.png", make_whiteboard),
        ("multiscript_form.png", make_multiscript_form),
    ]:
        p = out_dir / name
        fn().save(p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in write_all():
        print(f"Wrote {p}")
