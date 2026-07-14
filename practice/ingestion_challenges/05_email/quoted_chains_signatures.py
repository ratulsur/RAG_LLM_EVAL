"""
PATHOLOGY (a): a reply email carries the ENTIRE thread history quoted inline,
plus a signature block and a legal disclaimer. Ingest it raw and the same
sentences get embedded 3-5 times, disclaimers dominate the corpus, and dedup at
query time is hopeless.

WHY NAIVE EXTRACTION FAILS
--------------------------
`msg.get_content()` returns the whole visible body: the new reply AND every
quoted ancestor (">", ">>", ">>>") AND "On <date> X wrote:" attributions AND the
signature AND the confidentiality notice. If you ingest each message in a 5-deep
thread, the oldest message is embedded 5 times. Retrieval returns near-duplicate
chunks; the LLM sees "Cost is 12k" three times and may treat repetition as
emphasis. Disclaimers ("This email is confidential...") appear on every message
and swamp term statistics.

THE PRODUCTION FIX
------------------
Segment the body into: NEW content, QUOTED history, SIGNATURE, DISCLAIMER.
Heuristics that work on real mail:
  - quoted lines start with ">" (any depth)
  - an attribution line "On <date>, X wrote:" marks the start of quoted history
  - signature starts at a line that is exactly "-- " (RFC 3676 sig delimiter)
  - disclaimers match keyword patterns ("confidential", "intended recipient")
Keep only NEW content for chunking. Across a thread, DEDUPE by normalized line
hash so any quoted text that slipped through is collapsed to one instance.

SENIOR TRADEOFF
---------------
Quoting styles are a swamp: Outlook uses "-----Original Message-----" and no ">",
some clients bottom-post, mobile signatures omit the "-- " delimiter. No regex
wins everywhere; production stacks combine these heuristics with a trained
reply/quote splitter (e.g. the `talon` library from Mailgun). The heuristics
here cover the RFC-style majority and, crucially, DEGRADE to "keep everything"
rather than silently dropping the new reply. Getting this right is what stops a
support-ticket RAG system from drowning in its own quoted history.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures", "quoted_chain.eml")

ATTRIBUTION = re.compile(r"^\s*On\s.+\bwrote:\s*$", re.IGNORECASE)
OUTLOOK_HDR = re.compile(r"^-+\s*Original Message\s*-+", re.IGNORECASE)
SIG_DELIM = re.compile(r"^--\s?$")
DISCLAIMER = re.compile(r"confidential|intended recipient|legally privileged|no liability",
                        re.IGNORECASE)


@dataclass
class Segments:
    new: List[str] = field(default_factory=list)
    quoted: List[str] = field(default_factory=list)
    signature: List[str] = field(default_factory=list)
    disclaimer: List[str] = field(default_factory=list)


class ReplyCleaner:
    def load_body(self, eml_path: str) -> str:
        with open(eml_path, "rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        body = msg.get_body(preferencelist=("plain",))
        return body.get_content() if body else ""

    def segment(self, body: str) -> Segments:
        seg = Segments()
        state = "new"  # new -> (quoted | signature | disclaimer)
        for line in body.splitlines():
            if state in ("new",) and SIG_DELIM.match(line):
                state = "signature"
                continue
            if ATTRIBUTION.match(line) or OUTLOOK_HDR.match(line):
                state = "quoted"
                seg.quoted.append(line)
                continue
            if line.lstrip().startswith(">"):
                seg.quoted.append(line)
                continue
            if DISCLAIMER.search(line):
                state = "disclaimer"
            if state == "new":
                seg.new.append(line)
            elif state == "signature":
                # a disclaimer can follow a signature; re-route it
                (seg.disclaimer if DISCLAIMER.search(line) else seg.signature).append(line)
            elif state == "disclaimer":
                seg.disclaimer.append(line)
            else:  # quoted region, non-'>' continuation line
                seg.quoted.append(line)
        return seg

    def clean_new(self, seg: Segments) -> str:
        return "\n".join(l for l in seg.new).strip()

    @staticmethod
    def dedupe_lines(*bodies: str) -> List[str]:
        seen, out = set(), []
        for b in bodies:
            for line in b.splitlines():
                norm = re.sub(r"\s+", " ", line.strip().lower())
                if not norm:
                    continue
                h = hashlib.md5(norm.encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    out.append(line.strip())
        return out


if __name__ == "__main__":
    if not os.path.exists(FIX):
        from fixtures.generate import make_quoted_chain
        make_quoted_chain()

    cleaner = ReplyCleaner()
    raw = cleaner.load_body(FIX)
    print("=" * 70)
    print(f"NAIVE full body: {len(raw.splitlines())} lines "
          f"(new reply buried under quotes + sig + disclaimer)")
    print("=" * 70)
    print(raw)

    seg = cleaner.segment(raw)
    print("\n" + "=" * 70)
    print("SEGMENTED -> chunk ONLY the NEW content:")
    print("=" * 70)
    print("NEW:\n  " + (cleaner.clean_new(seg) or "(none)").replace("\n", "\n  "))
    print(f"\nquoted lines dropped : {len(seg.quoted)}")
    print(f"signature lines dropped: {len(seg.signature)}")
    print(f"disclaimer lines dropped: {len(seg.disclaimer)}")
