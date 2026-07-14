"""
PATHOLOGY (a): tracked changes and comments leak into -- or vanish from -- the
extracted body, depending on the parser. Either way the chunk is wrong.

WHY NAIVE EXTRACTION FAILS
--------------------------
python-docx's `paragraph.text` walks only <w:r> runs. Text inside <w:ins>
(a tracked insertion) and <w:del> (a tracked deletion) lives in different
elements, so it is SILENTLY DROPPED. On our fixture, "deliver within 30/15
business days" comes out as "deliver within business days" -- the operative
number is gone. Other tools do the opposite and splice DELETED text and reviewer
COMMENTS straight into the body, so the chunk says both "30" and "15" and quotes
a reviewer's private note as if it were contract text.

THE PRODUCTION FIX
------------------
Walk the raw document.xml and produce THREE clean streams:
  ACCEPTED  = body with insertions kept, deletions removed   (the final doc)
  REJECTED  = body with deletions kept, insertions removed    (the original)
  COMMENTS  = reviewer notes from comments.xml, keyed to the anchored text
Chunk the ACCEPTED stream; keep COMMENTS as separate metadata, never inline.

SENIOR TRADEOFF
---------------
Which stream feeds RAG is a policy call: for a signed/executed contract you want
ACCEPTED (the final terms); for a diff/audit use case you may index both and tag
provenance. The one thing you must NOT do is let the default parser decide by
accident. Comments carry reviewer intent that is useful for audit but toxic in
answer context -- isolate them.

Maps to Universal Document Ingestor: this is the "revision normalizer" -- it
guarantees the chunk reflects a deliberate revision state, not parser luck.
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from typing import List

from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "dirty.docx")
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def _qn(tag: str) -> str:
    return f"{{{W}}}{tag}"


@dataclass
class Comment:
    cid: str
    author: str
    text: str
    anchor: str


class RevisionExtractor:
    def __init__(self, docx_path: str):
        self.path = docx_path

    def _xml(self, member: str):
        with zipfile.ZipFile(self.path) as z:
            if member not in z.namelist():
                return None
            return etree.fromstring(z.read(member))

    def _para_text(self, p, mode: str) -> str:
        """mode='accepted' keeps <w:ins>, drops <w:del>; 'rejected' vice-versa."""
        out: List[str] = []
        for el in p.iter():
            tag = etree.QName(el).localname
            if self._inside(el, "txbxContent"):  # textbox text handled elsewhere
                continue
            if tag == "t":                      # normal run text
                if not self._inside(el, "del") and not self._inside(el, "ins"):
                    out.append(el.text or "")
                elif self._inside(el, "ins") and mode == "accepted":
                    out.append(el.text or "")
            elif tag == "delText":              # deleted run text
                if mode == "rejected":
                    out.append(el.text or "")
        return "".join(out)

    @staticmethod
    def _inside(el, ancestor_local: str) -> bool:
        parent = el.getparent()
        while parent is not None:
            if etree.QName(parent).localname == ancestor_local:
                return True
            parent = parent.getparent()
        return False

    def body_streams(self):
        doc = self._xml("word/document.xml")
        accepted, rejected = [], []
        for p in doc.iter(_qn("p")):
            if p.getparent() is not None and etree.QName(p.getparent()).localname == "txbxContent":
                continue  # skip textbox paras here (handled in hidden_content.py)
            a = self._para_text(p, "accepted").strip()
            r = self._para_text(p, "rejected").strip()
            if a:
                accepted.append(a)
            if r:
                rejected.append(r)
        return "\n".join(accepted), "\n".join(rejected)

    def comments(self) -> List[Comment]:
        croot = self._xml("word/comments.xml")
        if croot is None:
            return []
        # map comment id -> anchored body text
        doc = self._xml("word/document.xml")
        anchors = self._comment_anchors(doc)
        out = []
        for cm in croot.iter(_qn("comment")):
            cid = cm.get(_qn("id"))
            author = cm.get(_qn("author")) or "?"
            text = "".join(t.text or "" for t in cm.iter(_qn("t")))
            out.append(Comment(cid, author, text.strip(), anchors.get(cid, "")))
        return out

    def _comment_anchors(self, doc):
        anchors = {}
        for start in doc.iter(_qn("commentRangeStart")):
            cid = start.get(_qn("id"))
            # collect sibling text until commentRangeEnd with same id
            buf, node = [], start.getnext()
            while node is not None:
                if etree.QName(node).localname == "commentRangeEnd" and node.get(_qn("id")) == cid:
                    break
                buf += [t.text or "" for t in node.iter(_qn("t"))]
                node = node.getnext()
            anchors[cid] = "".join(buf).strip()
        return anchors


def naive_python_docx(docx_path: str) -> str:
    import docx
    return "\n".join(p.text for p in docx.Document(docx_path).paragraphs if p.text.strip())


if __name__ == "__main__":
    if not os.path.exists(FIX):
        from fixtures.generate import build_all
        build_all()

    print("=" * 70)
    print("NAIVE python-docx .text -- tracked number silently dropped:")
    print("=" * 70)
    print(naive_python_docx(FIX))

    ex = RevisionExtractor(FIX)
    accepted, rejected = ex.body_streams()
    print("\n" + "=" * 70)
    print("ACCEPTED stream (insertions kept, deletions removed) -> chunk THIS:")
    print("=" * 70)
    print(accepted)
    print("\n" + "=" * 70)
    print("REJECTED stream (the original, deletions restored):")
    print("=" * 70)
    print(rejected)
    print("\n" + "=" * 70)
    print("COMMENTS (isolated metadata, NEVER inlined into the chunk):")
    print("=" * 70)
    for c in ex.comments():
        print(f"  [{c.author}] on {c.anchor!r}: {c.text}")
