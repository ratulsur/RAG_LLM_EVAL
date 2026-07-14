"""
PATHOLOGY (c): a thread arrives as loose .eml files, OUT OF ORDER, with subjects
mangled by "Re:", "RE:", "Fwd:", "[updated]". Sorting by subject or by file/
arrival order scrambles the conversation, and a branched reply (two messages
answering the same parent) is lost entirely.

WHY NAIVE EXTRACTION FAILS
--------------------------
Grouping by subject string fails: "Re: Q3 planning", "RE: Q3 planning [updated]",
and "Fwd: Q3 planning" look like three different subjects but are one thread.
Ordering by receive time flattens a tree into a line and destroys the
reply-to structure -- you can no longer tell which message answered which, so a
RAG answer about "the revised numbers" loses the question it responded to.

THE PRODUCTION FIX
------------------
Rebuild the tree from the RFC 5322 threading headers, not the subject:
  - Message-ID    uniquely identifies each message
  - In-Reply-To   points at the direct parent's Message-ID
  - References    the full ancestor chain (fallback when In-Reply-To is missing)
Algorithm (JWZ-style, simplified):
  1. index every message by Message-ID
  2. link each to its parent via In-Reply-To (or the last id in References)
  3. the message with no resolvable parent is the root
  4. order siblings by Date; walk depth-first for a stable reading order
Normalize the subject only to CONFIRM thread identity, never to group.

SENIOR TRADEOFFS
----------------
- Missing/foreign Message-IDs: some clients drop In-Reply-To; then References or
  a normalized-subject + time-window heuristic is the fallback (what JWZ does).
- Broken threads: a reply whose parent id we never received becomes a secondary
  root; surface it rather than silently attaching it to the wrong place.
- Branches matter: two replies to one message are siblings, not a sequence -- a
  linear "sort by time" would falsely imply m4 answered m3.

Maps to Universal Document Ingestor: correct thread structure lets you chunk a
conversation as a coherent unit (Q -> A -> revision) instead of shredding it
into orphaned messages -- essential for support/email RAG.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
THREAD_DIR = os.path.join(HERE, "fixtures", "thread")
_SUBJ_PREFIX = re.compile(r"^\s*(re|fwd|fw)\s*:\s*", re.IGNORECASE)


def norm_subject(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = _SUBJ_PREFIX.sub("", s)
    return re.sub(r"\[.*?\]", "", s).strip().lower()


@dataclass
class Node:
    mid: str
    subject: str
    date: object
    parent: Optional[str]
    body: str
    children: List["Node"] = field(default_factory=list)


class ThreadBuilder:
    def _parse(self, path: str) -> Node:
        with open(path, "rb") as f:
            m = BytesParser(policy=policy.default).parse(f)
        refs = (m["References"] or "").split()
        parent = m["In-Reply-To"] or (refs[-1] if refs else None)
        body = m.get_body(preferencelist=("plain",))
        return Node(
            mid=(m["Message-ID"] or path).strip(),
            subject=m["Subject"] or "",
            date=parsedate_to_datetime(m["Date"]) if m["Date"] else None,
            parent=parent.strip() if parent else None,
            body=body.get_content().strip() if body else "",
        )

    def build(self, eml_paths: List[str]):
        nodes: Dict[str, Node] = {}
        for p in eml_paths:
            n = self._parse(p)
            nodes[n.mid] = n
        roots: List[Node] = []
        for n in nodes.values():
            if n.parent and n.parent in nodes:
                nodes[n.parent].children.append(n)
            else:
                roots.append(n)  # true root OR orphan whose parent we never got
        for n in nodes.values():
            n.children.sort(key=lambda c: (c.date is None, c.date))
        roots.sort(key=lambda c: (c.date is None, c.date))
        return roots

    def render(self, roots: List[Node]) -> str:
        lines: List[str] = []

        def walk(n: Node, depth: int):
            pad = "  " * depth
            when = n.date.strftime("%H:%M") if n.date else "??:??"
            lines.append(f"{pad}- [{when}] {n.subject}  ({n.mid})")
            lines.append(f"{pad}    {n.body}")
            for c in n.children:
                walk(c, depth + 1)

        for r in roots:
            walk(r, 0)
        return "\n".join(lines)


if __name__ == "__main__":
    if not os.path.isdir(THREAD_DIR) or not os.listdir(THREAD_DIR):
        from fixtures.generate import make_thread
        make_thread()

    paths = sorted(os.path.join(THREAD_DIR, f)
                   for f in os.listdir(THREAD_DIR) if f.endswith(".eml"))

    print("=" * 70)
    print("NAIVE order (files as stored on disk) -- conversation scrambled:")
    print("=" * 70)
    for p in paths:
        with open(p, "rb") as f:
            m = BytesParser(policy=policy.default).parse(f)
        print(f"  {os.path.basename(p):12} subj={m['Subject']!r}")

    builder = ThreadBuilder()
    roots = builder.build(paths)
    subs = {norm_subject(r.subject) for r in roots} | {
        norm_subject(c.subject) for r in roots for c in r.children}
    print("\n" + "=" * 70)
    print(f"RECONSTRUCTED thread tree ({len(roots)} root; "
          f"normalized subject confirms one thread: {subs}):")
    print("=" * 70)
    print(builder.render(roots))
