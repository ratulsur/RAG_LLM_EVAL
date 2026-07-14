"""
PATHOLOGY (b): null vs missing vs empty are THREE different facts.

In a nested payload:
  * "phone": null        -> the field exists and is known to be empty
  * "email": ""          -> the field exists and holds an empty string
  * "nickname" absent    -> we were never told anything about it

Downstream (feature stores, compliance, retrieval metadata) these mean different
things: null = asserted-unknown, "" = provided-but-blank, absent = never-supplied.
A naive flatten (json_normalize, or `d.get(k, "")`) collapses all three to "" or
NaN and destroys information you can't recover.

Fix -- a flattener that walks to dotted paths and encodes the distinction with
explicit sentinels, plus (given a schema of expected paths) reports which paths
were ABSENT vs present-but-null vs empty.

Interview line: "The three empties are not interchangeable. I flatten to dotted
paths and keep distinct sentinels -- because 'we don't have the user's phone' and
'the user has no phone' are different rows in an audit."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# distinguishable sentinels (objects, so they never collide with real data)
ABSENT = "<ABSENT>"    # key was not present at all
NULL = "<NULL>"        # key present, value was JSON null
EMPTY = "<EMPTY_STR>"  # key present, value was ""


def flatten(obj: Any, prefix: str = "", out: dict | None = None) -> dict[str, Any]:
    """Recursively flatten dicts/lists to dotted paths. Preserves null and empty
    string as explicit sentinels; list items get numeric indices."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        if not obj:
            out[prefix or "<root>"] = "<EMPTY_DICT>"
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            flatten(v, key, out)
    elif isinstance(obj, list):
        if not obj:
            out[prefix] = "<EMPTY_LIST>"
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}[{i}]", out)
    else:
        if obj is None:
            out[prefix] = NULL
        elif obj == "":
            out[prefix] = EMPTY
        else:
            out[prefix] = obj
    return out


@dataclass
class PresenceReport:
    present: list[str] = field(default_factory=list)
    null_valued: list[str] = field(default_factory=list)
    empty_valued: list[str] = field(default_factory=list)
    absent: list[str] = field(default_factory=list)


def presence_report(flat: dict[str, Any], expected_paths: list[str]) -> PresenceReport:
    """Classify each expected path as present / null / empty / absent."""
    rep = PresenceReport()
    for p in expected_paths:
        if p not in flat:
            rep.absent.append(p)
        elif flat[p] == NULL:
            rep.null_valued.append(p)
        elif flat[p] == EMPTY:
            rep.empty_valued.append(p)
        else:
            rep.present.append(p)
    return rep


def _demo():
    from fixtures.make_fixtures import RECORDS

    print("=" * 70)
    print("FLATTEN preserving NULL vs ABSENT vs EMPTY")
    print("=" * 70)

    rec = RECORDS[0]
    print("\n[BEFORE] nested record:")
    print("   ", rec)

    flat = flatten(rec)
    print("\n[AFTER] flattened (dotted paths, distinct sentinels):")
    for k, v in flat.items():
        print(f"    {k:22s} = {v!r}")

    expected = ["id", "profile.name", "profile.email", "profile.phone",
                "profile.nickname", "tags[0]", "tags[1]"]
    rep = presence_report(flat, expected)
    print("\n[PRESENCE REPORT] (why this matters for audits / features)")
    print(f"    present      : {rep.present}")
    print(f"    null_valued  : {rep.null_valued}   (asserted-unknown)")
    print(f"    empty_valued : {rep.empty_valued}   (provided-but-blank)")
    print(f"    absent       : {rep.absent}   (never-supplied)")
    print("\n[CHECK] a naive get(k,'') would have mapped all three empties to '' "
          "-- distinction lost.")


if __name__ == "__main__":
    _demo()
