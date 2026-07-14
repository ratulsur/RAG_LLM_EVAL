"""
PATHOLOGY (d): date ambiguity -- the silent corruptor.

Same file, same column: '03/04/2024' (is it 3 Apr or 4 Mar?), '13/05/2024'
(13 > 12, so DD/MM is certain), '07/22/23' (22 > 12 -> MM/DD, 2-digit year), and
45566 (an Excel serial that pandas leaves as an int). dateutil.parse with a fixed
dayfirst guesses ONE way for the whole column and silently mis-dates half of it.

Production approach -- decide format PER COLUMN from evidence, never per cell:
  1. Convert Excel serials (int/float in a plausible range) via the 1899-12-30
     epoch, correcting for Excel's fictional 1900-02-29 leap day.
  2. Scan the string dates and use the day>12 rule: any token whose first field
     is >12 forces DD/MM; any whose second field is >12 forces MM/DD. If both
     appear -> the column is internally inconsistent (flag it).
  3. Resolve 2-digit years with a pivot (<=cur_yy -> 2000s else 1900s).
  4. For cells that remain genuinely ambiguous (both fields <=12) FLAG them with
     the column-level format if one was established, else mark ambiguous=True and
     DO NOT guess.

Interview line: "Silently guessing dayfirst is the classic ingestion bug that
ships wrong quarter numbers to finance. The fix is column-level format inference
with an explicit ambiguity flag -- correctness you can audit."
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

_EXCEL_EPOCH = dt.date(1899, 12, 30)   # accounts for Excel's 1900 leap-year bug
_DMY_RE = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\s*$")


@dataclass
class DateCell:
    raw: Any
    iso: str | None            # normalized YYYY-MM-DD, or None
    format: str                # 'DD/MM' | 'MM/DD' | 'excel_serial' | 'unparsed'
    ambiguous: bool = False    # both fields <=12 and no column consensus
    note: str = ""


def _excel_serial_to_date(n: float) -> dt.date:
    return _EXCEL_EPOCH + dt.timedelta(days=int(n))


def _resolve_2digit_year(y: int, pivot: int = 70) -> int:
    return 2000 + y if y <= pivot else 1900 + y


def _infer_column_format(values: list[Any]) -> str | None:
    """Look across the column for a token that disambiguates. Returns 'DD/MM',
    'MM/DD', 'inconsistent', or None (no evidence -> can't tell)."""
    votes = set()
    for v in values:
        if not isinstance(v, str):
            continue
        m = _DMY_RE.match(v)
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 and b <= 12:
            votes.add("DD/MM")
        elif b > 12 and a <= 12:
            votes.add("MM/DD")
    if not votes:
        return None
    if len(votes) > 1:
        return "inconsistent"
    return votes.pop()


def normalize_column(values: list[Any]) -> list[DateCell]:
    fmt = _infer_column_format(values)
    out: list[DateCell] = []
    for v in values:
        # Excel serial (bare number in a plausible date range)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if 20000 <= v <= 60000:      # ~1954..2064
                d = _excel_serial_to_date(v)
                out.append(DateCell(v, d.isoformat(), "excel_serial"))
            else:
                out.append(DateCell(v, None, "unparsed", note="number out of date range"))
            continue

        m = _DMY_RE.match(str(v))
        if not m:
            out.append(DateCell(v, None, "unparsed"))
            continue
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if len(m.group(3)) == 2:
            y = _resolve_2digit_year(y)

        if a > 12 and b <= 12:                 # first field forces DD/MM
            day, mon, used, amb = a, b, "DD/MM", False
        elif b > 12 and a <= 12:               # second field forces MM/DD
            mon, day, used, amb = a, b, "MM/DD", False
        else:                                  # both <=12 -> genuinely ambiguous
            if fmt in ("DD/MM", "MM/DD"):
                if fmt == "DD/MM":
                    day, mon = a, b
                else:
                    mon, day = a, b
                used, amb = fmt + " (column-inferred)", True
            else:
                out.append(DateCell(v, None, "ambiguous", ambiguous=True,
                                    note="both fields <=12, no column consensus"))
                continue
        try:
            iso = dt.date(y, mon, day).isoformat()
            out.append(DateCell(v, iso, used, ambiguous=amb))
        except ValueError:
            out.append(DateCell(v, None, "unparsed", note="invalid calendar date"))
    return out


def _demo():
    from excel_tables import _read_grid, detect_regions, region_to_df
    from openpyxl import load_workbook
    import os

    print("=" * 70)
    print("DATE DISAMBIGUATION (DD/MM vs MM/DD, serials, 2-digit years)")
    print("=" * 70)

    xlsx = os.path.join(os.path.dirname(__file__), "fixtures", "dirty.xlsx")
    wb = load_workbook(xlsx, data_only=True)
    grid = _read_grid(wb["Q3_Report"])
    contracts = region_to_df(detect_regions(grid)[1])
    raw = list(contracts["Signed Date"])

    print(f"\n[BEFORE] raw column: {raw}")
    print(f"[BEFORE] column-level format inference -> {_infer_column_format(raw)!r}")

    print("\n[AFTER]")
    cells = normalize_column(raw)
    print(pd.DataFrame([c.__dict__ for c in cells]).to_string(index=False))
    flagged = [c for c in cells if c.ambiguous or c.iso is None]
    print(f"\n[FLAGGED] {len(flagged)} cell(s) need human review "
          "(ambiguous / unparsed) -- not silently guessed.")


if __name__ == "__main__":
    _demo()
