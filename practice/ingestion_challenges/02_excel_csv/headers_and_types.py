"""
PATHOLOGY (b) human-formatted headers + (c) mixed types in one column.

(b) Exports carry TWO-ROW headers (a group label merged over leaf labels), units
    baked into the name ("Revenue (INR Cr)"), and inconsistent casing. A retriever
    keys on column names, so "Revenue (INR Cr)" vs "revenue" fragments your schema.
    Fix: flatten multi-row headers into one, split the unit out into metadata, and
    normalize the name to snake_case.

(c) One column mixes a real value, a text-number ("1,240", "INR 1,110"), and a
    human sentinel ("N/A", "-", "TBD"). pd.to_numeric with errors='coerce' would
    silently turn BOTH the text-number and the junk into NaN, so you lose real data
    and can't tell a true-missing from a parse-failure. Fix: a per-column coercer
    that (i) maps known sentinels to explicit NULL, (ii) strips currency/thousands
    formatting before parsing, (iii) QUARANTINES anything still un-coercible with
    its original value, and (iv) reports coercion stats so the failure is visible.

Interview line: "errors='coerce' is a data-loss switch in disguise. Production
coercion reports what it dropped and quarantines the un-parseable -- silence here
is how bad numbers reach a dashboard."
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SENTINEL_NULLS = {"", "n/a", "na", "-", "--", "tbd", "none", "null", "nan", "#n/a"}
_UNIT_RE = re.compile(r"^(.*?)\s*[\(\[]([^)\]]+)[\)\]]\s*$")
_NUM_CLEAN_RE = re.compile(r"[,\s₹$€£]|inr|usd|rs\.?", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# (b) header handling
# --------------------------------------------------------------------------- #
def _looks_numeric(v: Any) -> bool:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return True
    if not isinstance(v, str):
        return False
    s = _NUM_CLEAN_RE.sub("", v).replace("%", "").strip()
    if s in ("", "-"):
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def detect_header_depth(rows: list[list[Any]], max_depth: int = 3) -> int:
    """A leading row is a header row iff none of its populated cells look numeric.
    Stop at the first row that contains any numeric-looking value (= data)."""
    depth = 0
    for row in rows[:max_depth]:
        populated = [v for v in row if v not in (None, "")]
        if populated and all(not _looks_numeric(v) for v in populated):
            depth += 1
        else:
            break
    return max(depth, 1)


def _to_snake(name: str) -> str:
    name = re.sub(r"[^\w\s]", " ", name)
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)   # camelCase boundary
    return re.sub(r"\s+", "_", name.strip().lower())


def flatten_header(rows: list[list[Any]]) -> tuple[list[str], dict[str, str], list[list[Any]]]:
    """Return (snake_case_columns, units_metadata, data_rows).

    Combines `depth` header rows per column, dropping a group label that is a
    prefix of the leaf label (so 'Revenue' + 'Revenue (INR Cr)' -> the leaf only),
    then extracts an embedded unit into metadata.
    """
    depth = detect_header_depth(rows)
    header_rows, data = rows[:depth], rows[depth:]
    ncol = max(len(r) for r in rows)

    raw_names: list[str] = []
    for j in range(ncol):
        parts: list[str] = []
        for hr in header_rows:
            val = hr[j] if j < len(hr) else None
            s = "" if val is None else str(val).strip()
            if s and (not parts or s not in parts):
                # drop an earlier part that the current leaf already contains
                parts = [p for p in parts if p not in s]
                parts.append(s)
        raw_names.append(" ".join(parts) if parts else f"col_{j}")

    units: dict[str, str] = {}
    cols: list[str] = []
    for raw in raw_names:
        m = _UNIT_RE.match(raw)
        if m:
            base, unit = m.group(1).strip(), m.group(2).strip()
        else:
            base, unit = raw, None
        col = _to_snake(base)
        # de-dupe collisions after normalization
        if col in cols:
            col = f"{col}_{cols.count(col)}"
        cols.append(col)
        if unit:
            units[col] = unit
    return cols, units, data


# --------------------------------------------------------------------------- #
# (c) per-column type coercion with stats + quarantine
# --------------------------------------------------------------------------- #
@dataclass
class ColumnReport:
    name: str
    inferred_type: str
    n: int
    n_ok: int = 0
    n_null: int = 0            # known sentinel / blank -> explicit NULL
    n_quarantined: int = 0     # un-coercible, original kept aside
    quarantine: dict[int, Any] = field(default_factory=dict)  # row_idx -> original

    def as_row(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "quarantine"}
        d["quarantine_rows"] = list(self.quarantine.keys())
        return d


def _infer_type(series: pd.Series) -> str:
    vals = [v for v in series if not _is_sentinel(v)]
    if not vals:
        return "string"
    numeric = sum(_looks_numeric(v) for v in vals)
    return "number" if numeric / len(vals) >= 0.6 else "string"


def _is_sentinel(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, str) and v.strip().lower() in SENTINEL_NULLS


def _coerce_number(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v)
    is_pct = s.strip().endswith("%")
    s = _NUM_CLEAN_RE.sub("", s).replace("%", "").strip()
    val = float(s)
    return val / 100.0 if is_pct else val


def coerce_column(series: pd.Series, name: str) -> tuple[pd.Series, ColumnReport]:
    """Coerce one column to its inferred type; report and quarantine failures."""
    ttype = _infer_type(series)
    rep = ColumnReport(name=name, inferred_type=ttype, n=len(series))
    out: list[Any] = []
    for idx, v in series.items():
        if _is_sentinel(v):
            rep.n_null += 1
            out.append(None)
            continue
        if ttype == "number":
            try:
                out.append(_coerce_number(v))
                rep.n_ok += 1
            except (ValueError, TypeError):
                rep.n_quarantined += 1
                rep.quarantine[idx] = v
                out.append(None)
        else:
            out.append(str(v).strip())
            rep.n_ok += 1
    dtype = "float64" if ttype == "number" else "object"
    return pd.Series(out, index=series.index, dtype=dtype), rep


def clean_frame(cols: list[str], data: list[list[Any]]) -> tuple[pd.DataFrame, list[ColumnReport]]:
    df = pd.DataFrame([r[:len(cols)] for r in data], columns=cols)
    reports = []
    for c in df.columns:
        df[c], rep = coerce_column(df[c], c)
        reports.append(rep)
    return df, reports


def _demo():
    from excel_tables import _read_grid, detect_regions
    from openpyxl import load_workbook
    import os

    print("=" * 70)
    print("HEADER FLATTENING + UNIT EXTRACTION + TYPE COERCION")
    print("=" * 70)

    xlsx = os.path.join(os.path.dirname(__file__), "fixtures", "dirty.xlsx")
    wb = load_workbook(xlsx, data_only=True)
    grid = _read_grid(wb["Q3_Report"])
    region = detect_regions(grid)[0]   # the two-row-header sales table

    print("\n[BEFORE] raw region rows (two-row header + dirty body):")
    for row in region.rows:
        print("   ", row)

    cols, units, data = flatten_header(region.rows)
    print(f"\n[AFTER] flattened columns : {cols}")
    print(f"        units -> metadata : {units}")

    df, reports = clean_frame(cols, data)
    print("\n[AFTER] coerced frame:")
    print(df.to_string(index=False))

    print("\n[COERCION REPORT]")
    print(pd.DataFrame([r.as_row() for r in reports]).to_string(index=False))
    for r in reports:
        if r.quarantine:
            print(f"   quarantined in '{r.name}': {r.quarantine}")


if __name__ == "__main__":
    _demo()
