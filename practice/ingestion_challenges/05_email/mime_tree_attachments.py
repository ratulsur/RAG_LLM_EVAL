"""
PATHOLOGY (b): the real information is in an INLINE IMAGE (a pasted screenshot of
a table) and inside NESTED attachments (an .eml zipped inside the email). Naive
extraction reads the thin text body and misses everything that matters.

WHY NAIVE EXTRACTION FAILS
--------------------------
`msg.get_content()` returns only the top-level text part. It does not:
  - descend into multipart subtrees,
  - open a .zip attachment to find the .eml inside it,
  - notice that a screenshot PNG holds the actual numbers.
So "the real numbers are in the screenshot" ingests as a content-free sentence,
and a countersigned NDA sitting in a nested zip is invisible to retrieval.

THE PRODUCTION FIX
------------------
Walk the MIME tree RECURSIVELY, and recurse again INTO container attachments:
  - text/*      -> extract as content
  - image/*     -> flag "info-in-pixels", route to OCR (guarded pytesseract)
  - .zip        -> open in memory, recurse over each member; an .eml member is
                   re-parsed as a full message (nested MIME)
  - message/rfc822 or .eml attachment -> parse as a child email
Emit a flat list of typed parts with a path so provenance survives.

SENIOR TRADEOFFS
----------------
- Recursion needs a depth/size guard: a zip-bomb or deeply nested rfc822 can
  exhaust memory. Cap depth and total bytes.
- Inline vs attachment: Content-Disposition and Content-ID (cid:) tell you if an
  image is embedded in the body flow vs a true attachment; both may carry info,
  so we surface both and let the OCR stage decide.
- OCR is optional (tesseract not installed here) -> we detect the image, confirm
  it decoded, and emit the OCR route + install note rather than failing.

Maps to Universal Document Ingestor: the multi-source ingestor's recursive
"container cracker" -- email is itself a container format, and containers nest.
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "multipart_nested.eml")


@dataclass
class Part:
    path: str          # provenance breadcrumb, e.g. root/attachments.zip/nda_thread.eml
    kind: str          # text | image | container | nested-email
    content: str       # extracted text, or a routing note


class MimeWalker:
    def __init__(self, max_depth: int = 6, max_bytes: int = 20 * 1024 * 1024):
        self.max_depth = max_depth
        self.max_bytes = max_bytes

    def walk_bytes(self, raw: bytes, path: str = "root", depth: int = 0) -> List[Part]:
        if depth > self.max_depth:
            return [Part(path, "container", "[depth cap hit -- stopped recursing]")]
        msg = BytesParser(policy=policy.default).parse(io.BytesIO(raw))
        return self._walk_msg(msg, path, depth)

    def _walk_msg(self, msg, path: str, depth: int) -> List[Part]:
        out: List[Part] = []
        for part in msg.walk():
            if part.is_multipart():
                continue
            ctype = part.get_content_type()
            fname = part.get_filename() or ""
            payload = part.get_payload(decode=True) or b""
            child = f"{path}/{fname}" if fname else path

            if ctype.startswith("text/"):
                txt = part.get_content() if not fname else payload.decode("utf-8", "replace")
                if txt.strip():
                    out.append(Part(child, "text", txt.strip()))
            elif ctype.startswith("image/"):
                out.append(Part(child, "image", self._image_route(payload, ctype)))
            elif ctype == "application/zip" or fname.lower().endswith(".zip"):
                out.append(Part(child, "container", f"zip opened ({len(payload)} bytes)"))
                out += self._crack_zip(payload, child, depth)
            elif ctype == "message/rfc822" or fname.lower().endswith(".eml"):
                out.append(Part(child, "nested-email", "re-parsed as child message"))
                out += self.walk_bytes(payload, child, depth + 1)
            else:
                out.append(Part(child, "other", f"{ctype} ({len(payload)} bytes) -> route by type"))
        return out

    def _crack_zip(self, data: bytes, path: str, depth: int) -> List[Part]:
        out: List[Part] = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                total = sum(i.file_size for i in z.infolist())
                if total > self.max_bytes:
                    return [Part(path, "container", "[zip too large -- skipped]")]
                for name in z.namelist():
                    member = z.read(name)
                    child = f"{path}/{name}"
                    if name.lower().endswith(".eml"):
                        out.append(Part(child, "nested-email", "eml inside zip"))
                        out += self.walk_bytes(member, child, depth + 1)
                    elif name.lower().endswith((".png", ".jpg", ".jpeg")):
                        out.append(Part(child, "image", self._image_route(member, "image/*")))
                    else:
                        out.append(Part(child, "other",
                                        f"{len(member)} bytes -> route by extension"))
        except zipfile.BadZipFile:
            out.append(Part(path, "container", "[bad zip]"))
        return out

    def _image_route(self, data: bytes, ctype: str) -> str:
        # confirm the image decodes (proves the info-carrying bytes are intact),
        # then hand to OCR (guarded).
        dims = "?"
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                dims = f"{im.width}x{im.height}"
        except Exception:
            pass
        try:
            import pytesseract  # noqa: F401
            return f"[info-in-pixels {dims}] OCR-ready (pytesseract present)"
        except ImportError:
            return (f"[info-in-pixels {dims}] route to OCR; "
                    f"pip install pytesseract + brew install tesseract to enable")


def naive_body(eml_path: str) -> str:
    with open(eml_path, "rb") as f:
        msg = BytesParser(policy=policy.default).parse(f)
    b = msg.get_body(preferencelist=("plain",))
    return b.get_content().strip() if b else ""


if __name__ == "__main__":
    if not os.path.exists(FIX):
        from fixtures.generate import make_multipart_nested
        make_multipart_nested()

    print("=" * 70)
    print("NAIVE top-level body (misses image + nested zip/eml):")
    print("=" * 70)
    print(naive_body(FIX))

    with open(FIX, "rb") as f:
        parts = MimeWalker().walk_bytes(f.read())
    print("\n" + "=" * 70)
    print("RECURSIVE MIME walk (containers cracked, images flagged):")
    print("=" * 70)
    for p in parts:
        print(f"  [{p.kind:13}] {p.path}")
        print(f"      {p.content}")
