# 06 — JSON / NDJSON / log ingestion challenges

> **Theme:** semi-structured data drifts and lies. Schemas evolve mid-stream, `null`/absent/`""`
> mean three different things, one corrupt NDJSON line sinks the whole batch, and log files mix
> two grammars. Tolerant, *observable* parsing is the job. Maps to the **Universal Document
> Ingestor** (multi-source RAG + source grading).

## Why naive parsing fails (the 30-second interview answer)

| Pathology | What breaks | The production fix (in this folder) |
|---|---|---|
| **(a)** Schema drift over time (field renamed, type changed, new nesting) | one table with half-null columns + mixed dtypes that break aggregation | tolerant normalizer: fold aliases → canonical, coerce declared types, **emit a drift log** (rename/type_change/new_field), align to a unified schema |
| **(b)** `null` vs missing key vs `""` — three different facts | naive `get(k, "")` collapses all three, destroying info you can't recover | flatten to dotted paths with **distinct sentinels** (`<NULL>` / `<ABSENT>` / `<EMPTY_STR>`) + presence report |
| **(c)** NDJSON with corrupt/truncated lines; 3 timestamp formats / 2 timezones | `json.load` throws → **entire file lost**; raw timestamps mis-order across tz | line-by-line parse with a **dead-letter log** (skip+record bad lines); normalize every stamp to UTC ISO-8601, **flag** naive-tz assumptions |
| **(d)** Log file mixing JSON lines with multi-line stack traces | a line-splitter shreds one traceback into 6 meaningless fragments | a small **state machine** groups a traceback into ONE event with exception_type/message/frame-count extracted |

## Files

```
06_json_logs/
├── fixtures/
│   ├── make_fixtures.py   # synthesizes events.ndjson (drift + corrupt + truncated + mixed tz)
│   │                      #  and app.log (JSON lines interleaved with a Python traceback)
│   ├── events.ndjson      # generated
│   └── app.log            # generated
├── schema_drift.py        # (a) tolerant normalizer + schema-diff reporter
├── flatten.py             # (b) dotted-path flattener preserving null/absent/empty
├── ndjson_stream.py       # (c) tolerant line parser + timestamp → UTC normalizer
└── log_split.py           # (d) JSON-vs-traceback classifier / trace grouper
```

## Run

```bash
python fixtures/make_fixtures.py
python schema_drift.py   # BEFORE (drifting keys/types) → AFTER (unified schema + drift log)
python flatten.py        # null/absent/empty kept distinct; presence report
python ndjson_stream.py  # 2 corrupt lines skipped, batch survives; all ts → UTC (naive flagged)
python log_split.py      # traceback grouped into ONE event with exception_type extracted
```

Everything runs **live** on the **standard library** + `python-dateutil` (timestamp parsing).
No external files, no network.

## Senior tradeoffs to say out loud

- **Tolerant AND observable.** The theme across all four: never abort on one bad record, and never
  handle drift silently. A dead-letter log and a drift log are what let you answer "why is `amount`
  sometimes a string?" with the exact offending record.
- **The three empties are not interchangeable.** "We don't have the user's phone" (`null`), "the
  user gave a blank phone" (`""`), and "we were never told about a nickname" (absent) are different
  rows in a compliance audit. Collapsing them is a real bug, not a style choice.
- **Never assume a naive timestamp is UTC.** Apply a *configured* assumed tz and flag it, so a
  wrong assumption is visible in the data instead of silently mis-ordering an incident timeline.
- **Logs need a grammar, not a splitter.** A stack trace is one logical event spanning many
  physical lines; the state machine is what makes error logs searchable and chunk-able.
