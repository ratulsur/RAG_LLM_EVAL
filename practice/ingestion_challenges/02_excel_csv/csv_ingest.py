"""
PATHOLOGY (e): CSV hell.

`pd.read_csv(path)` assumes UTF-8, a comma delimiter, and clean RFC-4180 quoting.
Real CSVs arrive with: a UTF-8 BOM that becomes a phantom prefix on the first
header ('\\ufeffClient'); non-UTF-8 bytes (Latin-1 / Windows-1252) that raise
UnicodeDecodeError or mojibake; embedded commas and doubled quotes inside fields;
inconsistent delimiters; and human trailer rows ('Total,,,', 'Signed off by...').

This module ingests bytes-first and defensively:
  1. detect encoding with chardet (fallback cp1252, then latin-1 which never
     fails) and strip a leading BOM,
  2. sniff the delimiter with csv.Sniffer on a sample,
  3. parse with the stdlib csv reader (RFC-4180 quote handling -- embedded commas
     and doubled quotes survive intact),
  4. strip trailing junk rows whose shape/keywords mark them as totals/sign-offs,
  5. return a clean DataFrame plus an ingest report.

Interview line: "read_csv is a happy-path convenience. Production ingest is
bytes -> detect encoding -> strip BOM -> sniff delimiter -> RFC-4180 parse ->
drop human trailers. Every one of those is a real ticket I've closed."
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from typing import Any

import chardet
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "fixtures", "dirty.csv")

_JUNK_KEYWORDS = re.compile(r"^\s*(total|subtotal|grand total|signed off|prepared by|"
                            r"generated on|notes?:)", re.IGNORECASE)


@dataclass
class IngestReport:
    encoding: str
    encoding_confidence: float
    had_bom: bool
    delimiter: str
    n_rows_raw: int
    n_rows_kept: int
    dropped_trailers: list[str] = field(default_factory=list)
    ragged_rows: list[tuple[int, int]] = field(default_factory=list)  # (row_idx, n_fields)


def _decode(raw: bytes) -> tuple[str, str, float, bool]:
    """Return (text, encoding, confidence, had_bom)."""
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    if had_bom:
        raw = raw[3:]
    guess = chardet.detect(raw)
    enc = guess.get("encoding") or "utf-8"
    conf = guess.get("confidence") or 0.0
    for candidate in (enc, "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(candidate), candidate, conf, had_bom
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace"), "latin-1", conf, had_bom


def _sniff_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _is_junk_row(cells: list[str], ncol: int) -> bool:
    non_empty = [c for c in cells if c.strip()]
    if not non_empty:
        return True
    # a totals/sign-off row: mostly empty AND/OR starts with a junk keyword
    if _JUNK_KEYWORDS.match(cells[0]):
        return True
    if len(non_empty) <= max(1, ncol // 2) and any(_JUNK_KEYWORDS.match(c) for c in cells):
        return True
    return False


def ingest_csv(path: str = CSV) -> tuple[pd.DataFrame, IngestReport]:
    with open(path, "rb") as f:
        raw = f.read()
    text, enc, conf, had_bom = _decode(raw)
    delim = _sniff_delimiter(text)

    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    rows = [r for r in rows if r]           # drop truly empty lines
    header, body = rows[0], rows[1:]
    ncol = len(header)

    kept, dropped, ragged = [], [], []
    for i, r in enumerate(body):
        if _is_junk_row(r, ncol):
            dropped.append(delim.join(r))
            continue
        if len(r) != ncol:
            # a genuinely ragged row (unescaped delimiter etc.) -- flag, don't
            # silently truncate. We still align it so the frame stays rectangular.
            ragged.append((i, len(r)))
        kept.append((r + [None] * ncol)[:ncol])

    df = pd.DataFrame(kept, columns=header)
    rep = IngestReport(encoding=enc, encoding_confidence=round(conf, 2),
                       had_bom=had_bom, delimiter=delim,
                       n_rows_raw=len(body), n_rows_kept=len(kept),
                       dropped_trailers=dropped, ragged_rows=ragged)
    return df, rep


def _demo():
    print("=" * 70)
    print("CSV INGEST: encoding + BOM + delimiter + RFC-4180 + trailer strip")
    print("=" * 70)

    with open(CSV, "rb") as f:
        raw = f.read()
    print("\n[BEFORE] first 90 raw bytes:")
    print("   ", raw[:90])
    print("\n[BEFORE] naive pd.read_csv (utf-8 default):")
    try:
        print(pd.read_csv(CSV).to_string(index=False))
    except Exception as e:  # noqa: BLE001
        print(f"    RAISED {type(e).__name__}: {e}")

    df, rep = ingest_csv()
    print("\n[AFTER] ingest report:")
    for k, v in rep.__dict__.items():
        print(f"    {k:22s}: {v}")
    print("\n[AFTER] clean frame (embedded commas/quotes + accent preserved):")
    print(df.to_string(index=False))
    print(f"\n    header[0] repr = {df.columns[0]!r}  (BOM stripped)")


if __name__ == "__main__":
    _demo()
