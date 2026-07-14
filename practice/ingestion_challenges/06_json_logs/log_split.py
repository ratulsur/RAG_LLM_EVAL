"""
PATHOLOGY (d): one log file, two grammars.

Application logs interleave structured JSON lines with free-text Python stack
traces. A stack trace is ONE logical error event spread over many physical lines
("Traceback (most recent call last):" ... indented frames ... "SomeError: msg").
Split the file line-by-line and you shred one error into eight meaningless
fragments; the exception type and message land in different chunks and retrieval
can never reunite them.

Fix -- a small state machine that:
  * emits each JSON line as its own structured event,
  * detects a "Traceback (most recent call last):" opener, then CONSUMES the
    indented frame lines and the terminating "ExceptionType: message" line,
    grouping the whole block into a SINGLE trace event with the exception type,
    message, and frame count extracted.

Interview line: "Log ingestion needs a line grammar, not a line splitter. I run a
tiny state machine so a multi-line traceback becomes one event with its exception
type pulled out -- that's what makes error logs searchable."
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
APPLOG = os.path.join(HERE, "fixtures", "app.log")

_TB_OPEN = re.compile(r"^\s*Traceback \(most recent call last\):")
_FRAME = re.compile(r'^\s+File ".*", line \d+, in ')
_EXC_LINE = re.compile(r"^(\w+(?:\.\w+)*Error|\w*Exception|\w+Warning)\b:?(.*)$")


@dataclass
class LogEvent:
    kind: str                       # 'json' | 'trace' | 'text'
    lineno: int
    payload: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


def split_log(path: str = APPLOG) -> list[LogEvent]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    events: list[LogEvent] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # structured JSON line?
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                events.append(LogEvent("json", i + 1, json.loads(stripped), stripped))
                i += 1
                continue
            except json.JSONDecodeError:
                pass  # fall through -> treat as text

        # start of a traceback? consume the whole block.
        if _TB_OPEN.match(line):
            start = i
            block = [line]
            i += 1
            # consume indented frame lines (File.../ code lines)
            while i < n and (lines[i].startswith("  ") or not lines[i].strip()):
                if lines[i].strip():
                    block.append(lines[i])
                i += 1
            exc_type, exc_msg = None, None
            # the terminating exception line (non-indented) if present
            if i < n:
                m = _EXC_LINE.match(lines[i].strip())
                if m:
                    exc_type = m.group(1)
                    exc_msg = m.group(2).lstrip(": ").strip()
                    block.append(lines[i])
                    i += 1
            frames = sum(1 for b in block if _FRAME.match(b))
            events.append(LogEvent(
                "trace", start + 1,
                {"exception_type": exc_type, "message": exc_msg,
                 "frames": frames, "lines": len(block)},
                "\n".join(block)))
            continue

        # anything else = free text
        events.append(LogEvent("text", i + 1, {}, line))
        i += 1
    return events


def _demo():
    from fixtures.make_fixtures import write_all
    write_all()

    print("=" * 70)
    print("LOG SPLIT: JSON lines vs multi-line stack traces")
    print("=" * 70)

    with open(APPLOG, encoding="utf-8") as f:
        raw = f.read()
    print(f"\n[BEFORE] raw file: {len(raw.splitlines())} physical lines; a naive "
          "line-splitter\n    would emit the traceback as 6 unrelated fragments.")

    events = split_log()
    print(f"\n[AFTER] {len(events)} logical events:")
    for e in events:
        if e.kind == "json":
            print(f"    [json ] line {e.lineno:>2}: {e.payload}")
        elif e.kind == "trace":
            p = e.payload
            print(f"    [TRACE] line {e.lineno:>2}: {p['exception_type']}: "
                  f"{p['message']!r}  ({p['frames']} frames, {p['lines']} lines "
                  "-> ONE event)")
        else:
            print(f"    [text ] line {e.lineno:>2}: {e.raw!r}")

    n_trace = sum(e.kind == "trace" for e in events)
    print(f"\n[CHECK] {n_trace} traceback grouped into a single searchable event "
          "with its\n    exception_type extracted -- not scattered across lines.")


if __name__ == "__main__":
    _demo()
