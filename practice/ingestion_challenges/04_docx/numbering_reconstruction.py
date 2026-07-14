"""
PATHOLOGY (c): list numbers are STYLE-GENERATED, not literal text. Extractors
return "Deliverables" and "Acceptance testing" with no numbers -- so a
cross-reference like "as defined in Section 2.1" can never be resolved, and
clause-scoped retrieval collapses.

WHY NAIVE EXTRACTION FAILS
--------------------------
In OOXML a numbered paragraph stores only <w:numPr> (numId + level). The visible
"Section 2", "2.1" is COMPUTED by Word from numbering.xml at render time and is
never written into the run text. Body extraction therefore yields unlabeled
items. Chunk them and you lose the addressing scheme the document relies on;
every "see Section 4.2" becomes a dangling pointer.

THE PRODUCTION FIX
------------------
Reconstruct the numbering yourself:
  1. Parse numbering.xml: numId -> abstractNumId -> per-level {numFmt, lvlText}.
  2. Walk list paragraphs in document order maintaining a counter per level.
     At level L: increment counters[L], reset every deeper level.
  3. Render lvlText by substituting %1..%(L+1) with the running counters.
  4. Prefix each item -> "Section 2 Deliverables", "2.1 Acceptance testing".

SENIOR TRADEOFF
---------------
Full fidelity needs to honor lvlOverride, startOverride, restart rules,
numFmt beyond decimal (lowerRoman, bullet...), and numbering that continues
across sections. This engine covers decimal multi-level lists -- the 90% case
for contracts, SOPs, and regulatory filings. For the rest, fall back to
LibreOffice headless render or accept degraded numbering with a logged warning.
Getting numbering right is what makes clause-level citations trustworthy in the
Universal Document Ingestor -- the difference between "the contract says X" and
"section 4.2 of the contract says X".
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "dirty.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _qn(t: str) -> str:
    return f"{{{W}}}{t}"


@dataclass
class NumberedItem:
    numId: str
    ilvl: int
    text: str
    label: str = ""


class NumberingEngine:
    def __init__(self, path: str):
        self.path = path
        self.abstract_of: Dict[str, str] = {}          # numId -> abstractNumId
        self.lvl_text: Dict[str, Dict[int, str]] = {}   # absId -> {ilvl -> lvlText}
        self._load_numbering()

    def _root(self, member: str):
        with zipfile.ZipFile(self.path) as z:
            if member not in z.namelist():
                return None
            return etree.fromstring(z.read(member))

    def _load_numbering(self):
        root = self._root("word/numbering.xml")
        if root is None:
            return
        for absn in root.iter(_qn("abstractNum")):
            aid = absn.get(_qn("abstractNumId"))
            lvls = {}
            for lvl in absn.iter(_qn("lvl")):
                ilvl = int(lvl.get(_qn("ilvl")))
                lt = lvl.find(_qn("lvlText"))
                lvls[ilvl] = lt.get(_qn("val")) if lt is not None else "%{}".format(ilvl + 1)
            self.lvl_text[aid] = lvls
        for num in root.iter(_qn("num")):
            nid = num.get(_qn("numId"))
            ain = num.find(_qn("abstractNumId"))
            if ain is not None:
                self.abstract_of[nid] = ain.get(_qn("val"))

    def _para_items(self) -> List[NumberedItem]:
        doc = self._root("word/document.xml")
        items: List[NumberedItem] = []
        for p in doc.iter(_qn("p")):
            numPr = p.find(f".//{_qn('numPr')}")
            if numPr is None:
                continue
            ilvl_el = numPr.find(_qn("ilvl"))
            numid_el = numPr.find(_qn("numId"))
            if numid_el is None:
                continue
            ilvl = int(ilvl_el.get(_qn("val"))) if ilvl_el is not None else 0
            text = "".join(t.text or "" for t in p.iter(_qn("t"))).strip()
            items.append(NumberedItem(numid_el.get(_qn("val")), ilvl, text))
        return items

    def _render_label(self, absId: str, ilvl: int, counters: List[int]) -> str:
        template = self.lvl_text.get(absId, {}).get(ilvl, "%{}".format(ilvl + 1))
        out = template
        for i in range(ilvl + 1):
            out = out.replace(f"%{i + 1}", str(counters[i]))
        return out

    def reconstruct(self) -> List[NumberedItem]:
        items = self._para_items()
        counters: List[int] = [0] * 9
        for it in items:
            counters[it.ilvl] += 1
            for deeper in range(it.ilvl + 1, len(counters)):
                counters[deeper] = 0
            absId = self.abstract_of.get(it.numId, "0")
            it.label = self._render_label(absId, it.ilvl, counters)
        return items


def naive_list(path: str) -> List[str]:
    import docx
    doc = docx.Document(path)
    return [p.text for p in doc.paragraphs
            if p.text.strip() and ("List" in p.style.name or p._p.find(f".//{_qn('numPr')}") is not None)]


if __name__ == "__main__":
    if not os.path.exists(FIX):
        from fixtures.generate import build_all
        build_all()

    print("=" * 70)
    print("NAIVE extraction -- list items with NO numbers (refs break):")
    print("=" * 70)
    for t in naive_list(FIX):
        print(f"  - {t}")

    print("\n" + "=" * 70)
    print("RECONSTRUCTED numbering -- clause references now resolvable:")
    print("=" * 70)
    for it in NumberingEngine(FIX).reconstruct():
        indent = "  " * (it.ilvl + 1)
        print(f"{indent}{it.label}  {it.text}")
