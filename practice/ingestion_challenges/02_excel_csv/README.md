# 02 — Excel / CSV ingestion challenges

> **Theme:** a spreadsheet is a *canvas*, not a table. `pd.read_excel` / `pd.read_csv`
> are happy-path conveniences; production ingestion has to detect the pathology, handle
> it, and emit clean, metadata-rich, chunk-ready output. Maps to the **Universal Document
> Ingestor** (multi-source RAG + source grading).

## Why naive extraction fails (the 30-second interview answer)

| Pathology | What breaks with the naive call | The production fix (in this folder) |
|---|---|---|
| **(a)** Multiple tables on one sheet + title/spacer rows + merged cells | `read_excel` returns one ragged frame; merged group labels become stray `None` | Segment the canvas into table **regions** on blank-row separators; forward-fill merged anchors; peel title rows into metadata |
| **(b)** Human headers — two-row, units-in-name, mixed casing | Column keys fragment (`Revenue (INR Cr)` ≠ `revenue`) | Flatten multi-row header, split unit → metadata, snake_case the name |
| **(c)** Mixed types in one column (`"1,240"`, `N/A`, `TBD`, `₹1,110`) | `to_numeric(errors="coerce")` silently nulls real data *and* junk alike | Per-column coercer: map sentinels→NULL, strip currency/thousands, **quarantine** un-coercible with stats |
| **(d)** Date ambiguity — DD/MM vs MM/DD in one file, Excel serials, 2-digit years | `dayfirst` guesses one way for the whole column → wrong quarters to finance | Column-level format inference via the **day>12 rule**; convert serials; resolve 2-digit years; **FLAG** genuine ambiguity, never guess |
| **(e)** CSV hell — BOM, Latin-1/cp1252, embedded commas/quotes, junk trailers | `read_csv` raises `UnicodeDecodeError` or mojibake; totals rows pollute the frame | bytes→chardet→strip BOM→sniff delimiter→RFC-4180 parse→drop human trailers |
| **(f)** Formula cells exported as `#REF!` / `#DIV/0!` | error strings poison a numeric column | openpyxl `data_only` vs formula load to classify **formula vs cached error**, quarantine errors |

## Files

```
02_excel_csv/
├── fixtures/
│   ├── make_fixtures.py   # synthesizes dirty.xlsx (multi-table, merged, 2-row header,
│   │                      #  error cells) and dirty.csv (BOM, Latin-1, embedded commas, junk)
│   ├── dirty.xlsx         # generated
│   └── dirty.csv          # generated
├── excel_tables.py        # (a) region detection + merged forward-fill  (f) error-cell quarantine
├── headers_and_types.py   # (b) header flatten + unit extraction  (c) type coercer + quarantine
├── dates.py               # (d) per-column date disambiguation + ambiguity flagging
└── csv_ingest.py          # (e) encoding/BOM/delimiter/RFC-4180/trailer strip
```

## Run

```bash
python fixtures/make_fixtures.py   # optional; each module regenerates as needed
python excel_tables.py             # BEFORE (one ragged frame) → AFTER (2 logical tables)
python headers_and_types.py        # BEFORE (two-row header, dirty body) → AFTER (typed frame + report)
python dates.py                    # shows an "inconsistent" column → ambiguous cell FLAGGED
python csv_ingest.py               # naive read_csv raises → clean frame + ingest report
```

Everything runs **live** on `pandas` + `openpyxl` + `chardet` — no external files, no network.

## Senior tradeoffs to say out loud

- **Region detection is heuristic.** Blank-row separation is robust for exports but fails on
  tables with intentional blank data rows — in production you'd combine it with header-run
  detection and a max-gap tolerance. Name the assumption; don't pretend it's exact.
- **Quarantine > coerce-to-NaN.** The whole point is *observability*: a coercion that drops
  data silently is a data-loss switch. Every module here emits a report of what it dropped.
- **Never guess an ambiguous date.** `03/04/2024` with no column consensus is flagged for
  human review, not silently resolved — correctness you can audit beats a confident wrong number.
- **Chunk-ready output:** each cleaned table carries its title, unit metadata, and per-column
  type — exactly the provenance a retriever needs to ground and cite.
