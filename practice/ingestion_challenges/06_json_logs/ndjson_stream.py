"""
PATHOLOGY (c): NDJSON with corrupt/truncated lines + timestamps in 3 formats / 2 tz.

A single bad line (a crash mid-write, a truncated payload) makes `json.load` or a
whole-file parse throw and lose the ENTIRE file. And the same stream carries
timestamps as epoch millis (int), ISO-8601 with a numeric offset, ISO with 'Z',
and a naive "YYYY-MM-DD HH:MM:SS" with no zone -- compare them raw and your event
ordering is wrong across timezones.

Fix:
  * Parse line by line; on a JSONDecodeError, SKIP + LOG the bad line (with its
    number and a snippet) and keep going. One poison line never sinks the batch.
  * Normalize every timestamp to UTC ISO-8601. Epoch millis -> UTC. Offset/Z ->
    convert to UTC. A naive stamp is assumed to be in a configured ASSUMED_TZ and
    FLAGGED so the assumption is visible (never silently treated as UTC).

Interview line: "Tolerant, line-oriented parsing with a dead-letter log is the
NDJSON discipline -- and timestamps get normalized to UTC with the naive ones
flagged, because a silent tz assumption is how you mis-order an incident timeline."
"""
from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass, field
from typing import Any

from dateutil import parser as du_parser

HERE = os.path.dirname(os.path.abspath(__file__))
NDJSON = os.path.join(HERE, "fixtures", "events.ndjson")

ASSUMED_TZ = dt.timezone(dt.timedelta(hours=5, minutes=30), name="Asia/Kolkata")
UTC = dt.timezone.utc


@dataclass
class BadLine:
    lineno: int
    error: str
    snippet: str


@dataclass
class ParseResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    bad_lines: list[BadLine] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return len(self.records)


def parse_ndjson(path: str = NDJSON) -> ParseResult:
    res = ParseResult()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                res.records.append(json.loads(line))
            except json.JSONDecodeError as e:
                res.bad_lines.append(BadLine(i, str(e), line[:40]))
    return res


@dataclass
class Timestamp:
    raw: Any
    utc_iso: str | None
    source_format: str          # 'epoch_ms' | 'iso_offset' | 'iso_z' | 'naive'
    tz_assumed: bool = False


def normalize_ts(value: Any) -> Timestamp:
    # 1) epoch millis / seconds as a number
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # heuristic: >1e11 => milliseconds
        secs = value / 1000.0 if value > 1e11 else float(value)
        d = dt.datetime.fromtimestamp(secs, tz=UTC)
        return Timestamp(value, d.isoformat(), "epoch_ms")
    if not isinstance(value, str):
        return Timestamp(value, None, "unknown")

    s = value.strip()
    try:
        d = du_parser.parse(s)
    except (ValueError, OverflowError):
        return Timestamp(value, None, "unparsed")

    if d.tzinfo is None:
        # naive: apply the configured assumed tz and FLAG it
        d = d.replace(tzinfo=ASSUMED_TZ)
        return Timestamp(value, d.astimezone(UTC).isoformat(), "naive", tz_assumed=True)
    fmt = "iso_z" if s.endswith("Z") else "iso_offset"
    return Timestamp(value, d.astimezone(UTC).isoformat(), fmt)


def _demo():
    from fixtures.make_fixtures import write_all
    write_all()

    print("=" * 70)
    print("TOLERANT NDJSON PARSE + TIMESTAMP NORMALIZE -> UTC")
    print("=" * 70)

    res = parse_ndjson()
    print(f"\n[AFTER] parsed OK: {res.n_ok}   bad lines skipped: {len(res.bad_lines)}")
    print("[DEAD-LETTER LOG] (batch survived the poison lines):")
    for b in res.bad_lines:
        print(f"    line {b.lineno}: {b.error}  :: {b.snippet!r}")

    print("\n[TIMESTAMP NORMALIZATION]")
    print(f"    {'raw':<32} {'source_format':<12} {'utc_iso':<28} tz_assumed")
    for rec in res.records:
        if "ts" not in rec:
            continue
        t = normalize_ts(rec["ts"])
        print(f"    {str(t.raw):<32} {t.source_format:<12} {str(t.utc_iso):<28} {t.tz_assumed}")
    print("\n[CHECK] all events now comparable on one UTC axis; the naive stamp is "
          "flagged\n    (assumed Asia/Kolkata), not silently treated as UTC.")


if __name__ == "__main__":
    _demo()
