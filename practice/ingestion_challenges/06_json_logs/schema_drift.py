"""
PATHOLOGY (a): schema drift across time.

Event streams evolve in place: a field gets renamed (user_id -> userId), a type
silently changes (amount "50" the string becomes 90 the int), and new nesting
appears mid-dataset (a "meta" object shows up in v3). Load that into one table
naively and you get half-null columns, mixed dtypes that break aggregation, and
records that don't align.

Fix -- a tolerant normalizer + a schema-diff reporter:
  * ALIASES fold renamed fields onto a canonical name.
  * A declared coercion map forces each canonical field to a stable type,
    quarantining values that won't coerce (never silently dropping).
  * The reporter tracks the first-seen type per field and LOGS every drift
    event (rename applied, type change, new field) so the evolution is auditable
    instead of invisible.

Interview line: "Ingestion has to be tolerant AND observable. I fold aliases to a
canonical schema and emit a drift log -- so when finance asks why amount is
sometimes a string, I hand them the exact record where the producer changed."
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))

# canonical schema: alias -> canonical name
ALIASES = {"userId": "user_id", "uid": "user_id", "user": "user_id"}
# canonical field -> target python type for coercion
COERCE = {"user_id": int, "amount": float}


@dataclass
class DriftEvent:
    record_idx: int
    kind: str          # 'rename' | 'type_change' | 'new_field' | 'quarantine'
    field: str
    detail: str


@dataclass
class NormalizeResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    drift: list[DriftEvent] = field(default_factory=list)
    unified_fields: list[str] = field(default_factory=list)


def _pytype(v: Any) -> str:
    return type(v).__name__


def normalize_records(records: list[dict[str, Any]]) -> NormalizeResult:
    res = NormalizeResult()
    first_type: dict[str, str] = {}      # canonical field -> first-seen type name
    seen_fields: set[str] = set()

    for idx, raw in enumerate(records):
        out: dict[str, Any] = {}
        for key, val in raw.items():
            canon = ALIASES.get(key, key)
            if canon != key:
                res.drift.append(DriftEvent(idx, "rename", canon,
                                            f"'{key}' -> '{canon}'"))
            # coerce to canonical type if declared
            if canon in COERCE and val is not None:
                try:
                    val = COERCE[canon](val)
                except (ValueError, TypeError):
                    res.drift.append(DriftEvent(idx, "quarantine", canon,
                                                f"un-coercible {val!r}"))
                    val = None
            # new-field + type-drift detection (against post-coercion type)
            if canon not in seen_fields:
                seen_fields.add(canon)
                first_type[canon] = _pytype(val)
                if idx > 0:              # a field absent in record 0 = later addition
                    res.drift.append(DriftEvent(idx, "new_field", canon,
                                                f"first appeared at record {idx}"))
            elif val is not None and _pytype(val) != first_type[canon] \
                    and first_type[canon] != "NoneType":
                res.drift.append(DriftEvent(
                    idx, "type_change", canon,
                    f"{first_type[canon]} -> {_pytype(val)}"))
            out[canon] = val
        res.records.append(out)

    # unified schema = union of all canonical fields, stable order by first sight
    order: dict[str, int] = {}
    for rec in res.records:
        for k in rec:
            order.setdefault(k, len(order))
    res.unified_fields = sorted(order, key=order.get)
    # backfill absent keys so every record aligns to the unified schema
    for rec in res.records:
        for k in res.unified_fields:
            rec.setdefault(k, None)
    return res


def _parse_valid_lines(path: str) -> list[dict[str, Any]]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue      # corrupt lines handled in ndjson_stream.py
    return recs


def _demo():
    from fixtures.make_fixtures import write_all, NDJSON
    write_all()
    print("=" * 70)
    print("SCHEMA DRIFT: tolerant normalizer + drift reporter")
    print("=" * 70)

    recs = _parse_valid_lines(NDJSON)
    print(f"\n[BEFORE] {len(recs)} valid records with drifting keys/types:")
    for r in recs:
        print("   ", {k: (v, _pytype(v)) for k, v in r.items()})

    res = normalize_records(recs)
    print(f"\n[AFTER] unified schema fields: {res.unified_fields}")
    print("[AFTER] aligned records:")
    for r in res.records:
        print("   ", r)

    print("\n[DRIFT LOG]")
    for d in res.drift:
        print(f"    rec#{d.record_idx:>2}  {d.kind:<12} {d.field:<10} {d.detail}")


if __name__ == "__main__":
    _demo()
