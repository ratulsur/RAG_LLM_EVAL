"""
Fixture generator: a DIRTY audio-derived transcript.

Synthesizes exactly the pathologies an ASR/diarization pipeline dumps on you,
so the ingest code has something real to run against with zero external files:

  - NO punctuation, no reliable sentence boundaries (naive chunkers choke)
  - FILLER words: um, uh, like, you know, i mean, sort of
  - ASR HOMOPHONE errors: their/there/they're, mangled drug/company names
  - [inaudible] markers and OVERLAP markers ([overlap])
  - CODE-SWITCHING: a Hindi-English (Devanagari + Latin) mixed line
  - SPEAKER turns with (optional) start timestamps -> used as pause cues

Returns the raw transcript as a single string. Call `write()` to also drop a
.txt next to this file, but nothing downstream requires the file to exist.
"""

from __future__ import annotations

from pathlib import Path

# One canonical dirty transcript. Timestamps are [HH:MM:SS] at each turn start.
# Big inter-turn gaps (the jump from :09 to :21) are a *pause* cue a real ASR
# would expose and a heuristic segmenter should exploit as a boundary.
RAW_TRANSCRIPT = """\
[00:00:01] SPEAKER_00: um so yeah i think the the patient was started on metphormin last week and their blood pressure like you know was basically fine but we should recheck
[00:00:09] SPEAKER_01: [inaudible] their reports came back and there uh there was a note from dr mehta about the dosage i mean the dosage looked to high
[00:00:21] SPEAKER_00: haan theek hai lekin humein dosage phir se check karni hogi before we send it to the pharamcy
[00:00:29] SPEAKER_01: [overlap] right right so like the the plan is we titrate down and then their team follows up in two weeks sort of a standard protocol
[00:00:38] SPEAKER_00: yeah and can you loop in accenture uh i mean the vendor axenture on the billing side because their invoice was wrong again
"""


def get_raw() -> str:
    """Return the dirty transcript string."""
    return RAW_TRANSCRIPT


def write(path: str | None = None) -> Path:
    out = Path(path) if path else Path(__file__).with_name("dirty_transcript.txt")
    out.write_text(RAW_TRANSCRIPT, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = write()
    print(f"Wrote dirty transcript -> {p}")
    print("-" * 70)
    print(RAW_TRANSCRIPT)
