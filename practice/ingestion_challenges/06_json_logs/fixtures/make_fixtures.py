"""
Synthesize DIRTY JSON / NDJSON / log fixtures (all stdlib, no external files).

Writes:
  events.ndjson -- schema drift over time (user_id->userId, amount str->int,
                   added nesting), 3 timestamp formats across 2 timezones,
                   one CORRUPT line (bad JSON) and one TRUNCATED line.
  app.log       -- structured JSON log lines interleaved with a multi-line
                   Python stack trace (one logical error event spanning N lines).

Also exposes RECORDS (the intended objects) for the flatten demo.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
NDJSON = os.path.join(HERE, "events.ndjson")
APPLOG = os.path.join(HERE, "app.log")

# Raw NDJSON lines as strings (some deliberately not valid JSON).
NDJSON_LINES = [
    # --- v1 schema: snake_case id, amount as STRING, ts as epoch millis ---
    '{"event": "purchase", "user_id": 1, "amount": "50", "ts": 1704067200000}',
    '{"event": "purchase", "user_id": 2, "amount": "75", "ts": 1704070800000}',
    # --- CORRUPT line (missing closing brace) ---
    '{"event": "purchase", "user_id": 3, "amount": "20"',
    # --- v2 schema drift: camelCase userId, amount now INT, ISO ts w/ offset ---
    '{"event": "purchase", "userId": 4, "amount": 90, "ts": "2024-01-01T05:00:00+05:30"}',
    '{"event": "purchase", "userId": 5, "amount": 30, "ts": "2024-01-01T02:00:00Z"}',
    # --- v3: new nested "meta" object, ts as naive local string ---
    '{"event": "purchase", "userId": 6, "amount": 40, "ts": "2024-01-01 08:00:00", "meta": {"channel": "web"}}',
    # --- TRUNCATED line (cut off mid-token, e.g. writer crashed) ---
    '{"event": "purcha',
]

# app.log: JSON lines interleaved with a real multi-line Python traceback.
APPLOG_TEXT = """{"level": "INFO", "ts": "2024-01-01T00:00:01Z", "msg": "worker started"}
{"level": "INFO", "ts": "2024-01-01T00:00:02Z", "msg": "processing batch 42"}
Traceback (most recent call last):
  File "/app/worker.py", line 88, in run
    result = handler(payload)
  File "/app/handlers.py", line 31, in handler
    return payload["amount"] / payload["count"]
ZeroDivisionError: division by zero
{"level": "ERROR", "ts": "2024-01-01T00:00:03Z", "msg": "batch 42 failed"}
{"level": "INFO", "ts": "2024-01-01T00:00:04Z", "msg": "worker idle"}
"""

# Intended records for the flatten demo: null vs missing vs empty all present.
RECORDS = [
    {
        "id": 1,
        "profile": {"name": "Ada", "email": "", "phone": None},  # empty vs null
        "tags": ["vip", "beta"],
        # NOTE: 'nickname' key is ABSENT (not null, not empty)
    },
]


def write_all() -> tuple[str, str]:
    with open(NDJSON, "w", encoding="utf-8") as f:
        f.write("\n".join(NDJSON_LINES) + "\n")
    with open(APPLOG, "w", encoding="utf-8") as f:
        f.write(APPLOG_TEXT)
    return NDJSON, APPLOG


if __name__ == "__main__":
    a, b = write_all()
    print("wrote:", a)
    print("wrote:", b)
