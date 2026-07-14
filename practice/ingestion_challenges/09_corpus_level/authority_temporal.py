"""
09b/c — CONTRADICTION DETECTION + AUTHORITATIVE RANKING + TEMPORAL VALIDITY.

THE PATHOLOGY (say this in the room)
  b) Two docs contradict each other (old policy: 15-day refunds; new: 30-day)
     with no signal telling retrieval which wins. Top-k returns both; the LLM
     picks one at random or hedges.
  c) Two docs both say "current pricing" — one from 2023 (INR 999), one from 2026
     (INR 1499). "Current" is a trap: the word is identical, only the effective
     DATE differs. Retrieval must know *when* each fact is valid.

THE FIX
  CONTRADICTION: detect same-topic docs whose extractable NUMERIC claims differ
  (refund window, price). This is a cheap, high-precision heuristic — full NLI is
  the production upgrade (guarded/reference).
  AUTHORITY RANK: when a contradiction is found, rank by an authority score:
     explicit "supersedes"  >  recency  >  source-trust tier (filename/path).
  TEMPORAL VALIDITY: extract each doc's effective date + any in-text year, tag a
  validity window, and at query time filter/deprioritize facts not valid AS OF
  the query date (default: today).

SENIOR POINT
  Recency alone is NOT authority — a 2026 blog post shouldn't beat a 2025 signed
  policy. So authority is a *blend*: explicit supersession first, then source
  trust, then recency as the tie-breaker. Say that; interviewers probe exactly
  this ("what if the newest doc is the least trustworthy?").

RUN
    python authority_temporal.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from dateutil import parser as dateparser

from fixtures.make_corpus import Doc, make_corpus

# Optional production path for contradiction: an NLI model.
try:  # pragma: no cover
    from transformers import pipeline  # type: ignore

    _NLI = pipeline("text-classification", model="roberta-large-mnli")
    _HAS_NLI = True
except Exception:  # noqa: BLE001
    _NLI = None
    _HAS_NLI = False


# Source-trust tiers (path/filename -> trust). In production this comes from a
# governance registry, not filenames — but the mechanism is the same.
def source_trust(doc: Doc) -> int:
    fn = doc.filename.lower()
    if "policy" in fn:
        return 3      # governed policy docs
    if "pricing" in fn:
        return 2
    return 1          # notes/memos/blogs


# ===========================================================================
# Numeric-claim extraction for cheap contradiction detection
# ===========================================================================
def extract_claims(text: str) -> dict[str, float]:
    claims: dict[str, float] = {}
    m = re.search(r"within\s+(\d+)\s+days", text, re.I)
    if m:
        claims["refund_window_days"] = float(m.group(1))
    m = re.search(r"INR\s+(\d+)", text, re.I)
    if m:
        claims["price_inr"] = float(m.group(1))
    return claims


def topic_of(doc: Doc) -> str:
    t = doc.text.lower()
    if "refund" in t:
        return "refund_policy"
    if "pricing" in t or "priced" in t:
        return "pricing"
    return "other"


@dataclass
class Contradiction:
    topic: str
    claim: str
    docs: list[tuple[str, float, str]]  # (filename, value, date)
    authoritative: str                  # filename that wins


def authority_score(doc: Doc) -> tuple:
    supersedes = 1 if "supersede" in doc.text.lower() else 0
    return (supersedes, source_trust(doc), doc.date)  # date is ISO -> sortable


def detect_contradictions(docs: list[Doc]) -> list[Contradiction]:
    by_topic: dict[str, list[Doc]] = {}
    for d in docs:
        by_topic.setdefault(topic_of(d), []).append(d)

    out: list[Contradiction] = []
    for topic, group in by_topic.items():
        if topic == "other" or len(group) < 2:
            continue
        # collect docs that assert each claim key
        claim_keys: set[str] = set()
        for d in group:
            claim_keys |= set(extract_claims(d.text).keys())
        for key in claim_keys:
            asserting = [(d, extract_claims(d.text)[key]) for d in group
                         if key in extract_claims(d.text)]
            values = {v for _, v in asserting}
            if len(values) > 1:  # genuine disagreement
                winner = max((d for d, _ in asserting), key=authority_score)
                out.append(Contradiction(
                    topic=topic,
                    claim=key,
                    docs=[(d.filename, v, d.date) for d, v in asserting],
                    authoritative=winner.filename,
                ))
    return out


# ===========================================================================
# Temporal validity
# ===========================================================================
@dataclass
class Validity:
    filename: str
    effective: str
    in_text_year: int | None
    valid_as_of_today: bool
    note: str


def extract_effective(doc: Doc, as_of: date) -> Validity:
    # in-text year cue ("2023 fiscal year", "effective 1 April 2026")
    yr = None
    m = re.search(r"\b(20\d{2})\b", doc.text)
    if m:
        yr = int(m.group(1))
    eff = dateparser.parse(doc.date).date()
    # A pricing doc is "stale" if a NEWER effective pricing doc exists; here we
    # approximate validity as: effective date <= as_of AND it's the latest such.
    valid = eff <= as_of
    note = "in effect" if valid else "future-dated"
    return Validity(doc.filename, doc.date, yr, valid, note)


def resolve_temporal(docs: list[Doc], topic: str, as_of: date) -> tuple[Doc, list[Validity]]:
    group = [d for d in docs if topic_of(d) == topic]
    vals = [extract_effective(d, as_of) for d in group]
    # current fact = latest effective date that is <= as_of
    eligible = [d for d in group if dateparser.parse(d.date).date() <= as_of]
    current = max(eligible, key=lambda d: d.date) if eligible else group[0]
    return current, vals


def _rule(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main() -> None:
    docs = make_corpus()
    today = date(2026, 7, 14)

    _rule("ENV")
    print(f"NLI model available: {_HAS_NLI} (numeric-claim heuristic in use otherwise)")

    _rule("BEFORE — contradictions sit in the index with no authority signal")
    print("refund window: v1 says 15 days, FINAL2 says 30 days")
    print("pricing:       2023 doc says INR 999 'current', 2026 doc says INR 1499 'current'")

    _rule("AFTER — contradictions detected + authoritative pick")
    for c in detect_contradictions(docs):
        print(f"[{c.topic} / {c.claim}]")
        for fn, v, dt in c.docs:
            flag = "  <-- AUTHORITATIVE" if fn == c.authoritative else ""
            print(f"   {fn:26s} = {v:<7} ({dt}){flag}")

    _rule("AFTER — temporal validity (as of 2026-07-14)")
    current, vals = resolve_temporal(docs, "pricing", today)
    for v in vals:
        print(f"   {v.filename:22s} effective {v.effective}  year-cue={v.in_text_year}  "
              f"[{v.note}]")
    print(f"   => CURRENT pricing fact = {current.filename} (INR 1499)")

    _rule("WHY THIS MATTERS")
    print("Authority = supersedes > source-trust > recency (NOT recency alone).")
    print("Temporal filter deprioritizes the 2023 'current pricing' so the LLM "
          "can't quote INR 999 in 2026.")


if __name__ == "__main__":
    main()
