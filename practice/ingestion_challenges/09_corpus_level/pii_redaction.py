"""
09e — PII SCATTERED IN FREE TEXT -> detect + redact BEFORE indexing.

THE PATHOLOGY (say this in the room)
  Free-text notes carry Aadhaar numbers, PAN, bank accounts, card numbers. If you
  embed and index raw text, PII lands in your vector DB, your logs, and every LLM
  prompt built from retrieved context — a compliance breach (DPDP Act / PCI). PII
  handling belongs at INGESTION, before a single vector is written.

THE FIX (detect -> validate -> redact-with-placeholder)
  1. DETECT with patterns tuned per PII type.
  2. VALIDATE to kill false positives — critically, Aadhaar uses the VERHOEFF
     checksum, so "0000 0000 0000" is rejected while a real Aadhaar passes; cards
     use LUHN. Validation is what separates a real redactor from a naive regex.
  3. REDACT with a TYPED, STABLE placeholder ([AADHAAR_a1b2]) instead of blanking.
     The placeholder PRESERVES RETRIEVABILITY (the chunk still says "an Aadhaar
     was provided") and the stable hash lets the same value map to the same token
     across docs without ever storing the value.

SENIOR POINT
  Redaction must be reversible ONLY via a separate, access-controlled token vault
  (not shown) — the index stores placeholders, the vault maps token->value behind
  auth. Never redact to a bare "[REDACTED]": you destroy the signal that PII was
  present and you merge distinct entities. Typed, salted-hash placeholders keep
  retrieval working and stay non-reversible from the index alone.

RUN
    python pii_redaction.py
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from fixtures.make_corpus import make_corpus

# A per-deployment salt so placeholder tokens can't be reversed by rainbow tables.
_SALT = "rag-ingest-salt-v1"


# ===========================================================================
# Checksums
# ===========================================================================
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    """Aadhaar's checksum. Rejects transpositions and most single-digit typos,
    which a length-12 regex alone cannot."""
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) != 12:
        return False
    c = 0
    for i, d in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


def luhn_valid(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# ===========================================================================
# Detectors: (name, regex, validator)
# ===========================================================================
def _always(_: str) -> bool:
    return True


DETECTORS = [
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), verhoeff_valid),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b"), luhn_valid),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), _always),
    ("ACCOUNT", re.compile(r"\baccount\s+(\d{9,18})\b", re.I), _always),
]


@dataclass
class Redaction:
    pii_type: str
    original: str
    placeholder: str


def _placeholder(pii_type: str, value: str) -> str:
    h = hashlib.sha256(f"{_SALT}:{value}".encode()).hexdigest()[:6]
    return f"[{pii_type}_{h}]"


def redact(text: str) -> tuple[str, list[Redaction]]:
    found: list[Redaction] = []
    out = text
    for pii_type, rx, validator in DETECTORS:
        for m in rx.finditer(text):
            raw = (m.group(1) if m.groups() else m.group(0)).strip()
            if not validator(raw):
                continue  # false positive killed by checksum/length
            ph = _placeholder(pii_type, re.sub(r"\D", "", raw))
            # replace the matched value (not the surrounding word like 'account')
            out = out.replace(raw, ph)
            found.append(Redaction(pii_type, raw, ph))
    return out, found


def _rule(t: str) -> None:
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main() -> None:
    doc = next(d for d in make_corpus() if d.doc_id == "d_pii")

    _rule("BEFORE — raw note (PII would be embedded + logged as-is)")
    print(doc.text)

    clean, found = redact(doc.text)

    _rule("AFTER — redacted, still retrievable")
    print(clean)

    _rule("WHAT VALIDATION CAUGHT")
    for r in found:
        print(f"   {r.pii_type:8s} {r.original!r:24s} -> {r.placeholder}")
    print("\nNote: '0000 0000 0000' matched the Aadhaar REGEX but FAILED the Verhoeff "
          "checksum, so it was correctly NOT redacted as Aadhaar — no false positive.")
    print("Placeholders are typed + salted-hashed: same value -> same token across "
          "docs, non-reversible from the index, retrievability preserved.")


if __name__ == "__main__":
    main()
