"""
PATHOLOGY (d): files lie about what they are. A ".docx" may be a legacy binary
.doc, an encrypted package, an RTF, or a PDF with the wrong extension. Trusting
the extension and handing it to python-docx throws deep in the parser -- or
worse, half-parses garbage.

WHY NAIVE EXTRACTION FAILS
--------------------------
Extension != format. Enterprise dumps are full of renamed and legacy files:
  - real OOXML (.docx) is a ZIP        -> magic 50 4B 03 04  ("PK..")
  - legacy .doc / encrypted OOXML is OLE2 -> magic D0 CF 11 E0 A1 B1 1A E1
  - RTF                                 -> "{\\rtf"
  - PDF                                 -> "%PDF"
python-docx only handles the ZIP/OOXML case; everything else must be DETECTED by
content and ROUTED, not fed in blindly.

THE PRODUCTION FIX
------------------
Sniff the leading bytes, then route:
  OOXML ZIP  -> peek inside: word/ = docx, xl/ = xlsx, ppt/ = pptx
               (and detect an encrypted OOXML masquerading as a plain zip)
  OLE2       -> could be legacy .doc OR an encrypted OOXML container; probe the
               directory for 'WordDocument' vs 'EncryptedPackage'
  legacy .doc-> route to LibreOffice headless / antiword (GUARDED optional deps)
  RTF/PDF    -> hand to the RTF/PDF path instead of the docx path

SENIOR TRADEOFF
---------------
Content sniffing is cheap and reliable for the container magic; distinguishing
"encrypted OOXML" from "legacy .doc" (both OLE2) needs an OLE directory read
(olefile) -- guarded here. Legacy .doc has no clean pure-Python extractor, so
production routes it through `soffice --headless --convert-to` or `antiword`;
neither binary is installed in this env, so we detect + emit the exact command
and degrade loudly. Never let a mislabeled file silently corrupt a corpus.

Maps to Universal Document Ingestor: the "front door" classifier that decides
which specialized extractor a file goes to, before any parser touches it.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from dataclasses import dataclass
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
FIX_DIR = os.path.join(HERE, "fixtures")

OLE2 = bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1])
ZIP = b"PK\x03\x04"


@dataclass
class Verdict:
    real_format: str
    handler: str
    note: str = ""


class FormatRouter:
    def sniff(self, path: str) -> Verdict:
        with open(path, "rb") as f:
            head = f.read(8)

        if head.startswith(ZIP):
            return self._classify_ooxml(path)
        if head.startswith(OLE2):
            return self._classify_ole2(path)
        if head.startswith(b"{\\rtf"):
            return Verdict("RTF", "rtf-parser (striprtf / pandoc)")
        if head.startswith(b"%PDF"):
            return Verdict("PDF", "route to 01_pdfs pipeline")
        return Verdict("UNKNOWN", "reject / manual triage",
                       f"leading bytes: {head[:4].hex()}")

    def _classify_ooxml(self, path: str) -> Verdict:
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except zipfile.BadZipFile:
            return Verdict("CORRUPT-ZIP", "reject", "PK magic but zip is invalid")
        # encrypted OOXML is actually OLE2, so a ZIP here is a real package
        if any(n.startswith("word/") for n in names):
            return Verdict("DOCX (OOXML)", "python-docx + XML recovery (this module set)")
        if any(n.startswith("xl/") for n in names):
            return Verdict("XLSX (OOXML)", "openpyxl / pandas")
        if any(n.startswith("ppt/") for n in names):
            return Verdict("PPTX (OOXML)", "python-pptx")
        return Verdict("ZIP (non-Office)", "generic archive walker")

    def _classify_ole2(self, path: str) -> Verdict:
        streams = self._ole_streams(path)
        if streams is None:
            return Verdict("OLE2 (legacy MS)", "soffice/antiword",
                           "olefile not installed -> pip install olefile to disambiguate; "
                           "route: soffice --headless --convert-to txt <file>  |  antiword <file>")
        if any("EncryptedPackage" in s for s in streams):
            return Verdict("ENCRYPTED-OOXML", "needs password -> msoffcrypto-tool then re-route",
                           "OLE2 wrapper around an encrypted OOXML package")
        if any("WordDocument" in s for s in streams):
            return Verdict("Legacy .DOC", self._doc_handler_hint(),
                           "true binary Word 97-2003 stream")
        return Verdict("OLE2 (unknown)", "olefile inspection required",
                       f"streams: {streams[:4]}")

    def _ole_streams(self, path: str):
        try:
            import olefile  # guarded optional
        except ImportError:
            return None
        try:
            with olefile.OleFileIO(path) as ole:
                return ["/".join(s) for s in ole.listdir()]
        except Exception:
            return None

    def _doc_handler_hint(self) -> str:
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        antiword = shutil.which("antiword")
        if soffice:
            return f"convert via {soffice} --headless --convert-to txt"
        if antiword:
            return f"extract via {antiword}"
        return ("no converter on PATH -> install LibreOffice (soffice) or antiword; "
                "then: soffice --headless --convert-to txt <file>")


if __name__ == "__main__":
    from fixtures import generate as g
    if not os.path.exists(os.path.join(FIX_DIR, "dirty.docx")):
        g.build_all()

    # synthesize two more in-memory proxies to show full routing coverage
    extra = {
        "proxy.rtf": b"{\\rtf1\\ansi this is rtf}",
        "proxy_pdf.bin": b"%PDF-1.7\n...",
    }
    for name, data in extra.items():
        with open(os.path.join(FIX_DIR, name), "wb") as f:
            f.write(data)

    router = FormatRouter()
    targets = [
        ("dirty.docx", "honest OOXML docx"),
        ("renamed.unknown", "docx with a bogus extension"),
        ("legacy_doc.bin", "OLE2 magic (legacy .doc / encrypted)"),
        ("proxy.rtf", "RTF"),
        ("proxy_pdf.bin", "PDF mislabeled .bin"),
    ]
    print("=" * 78)
    print(f"{'file':22} {'claimed':32} {'detected'}")
    print("=" * 78)
    for fname, claim in targets:
        path = os.path.join(FIX_DIR, fname)
        if not os.path.exists(path):
            continue
        v = router.sniff(path)
        print(f"{fname:22} {claim:32} {v.real_format}")
        print(f"{'':22} -> handler: {v.handler}")
        if v.note:
            print(f"{'':22}    note: {v.note}")
