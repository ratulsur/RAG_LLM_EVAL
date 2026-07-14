"""
07 — TRANSCRIPT INGESTION (audio-derived text).

THE PATHOLOGY (say this in the room)
  ASR output is not prose. It arrives as an unpunctuated token stream with
  filler words, homophone errors, diarization noise, [inaudible] gaps and — in
  the Indian market — Hindi-English code-switching. A naive RecursiveCharacter
  splitter has NO sentence boundaries to split on, so it cuts mid-thought,
  destroys retrievability, and indexes "um" and "uh" as content. Ingestion, not
  the LLM, is where this is won or lost.

THE PIPELINE (detect -> handle -> emit chunk-ready)
  1. PARSE turns     -> speaker + timestamp become METADATA, never body text.
  2. CLEAN each turn -> strip fillers, resolve [inaudible]/[overlap] markers.
  3. NORMALIZE       -> homophone + entity correction against a small glossary,
                        with every correction FLAGGED (never silent) for review.
  4. DETECT script   -> Devanagari vs Latin unicode ranges -> code_switch flag.
  5. SEGMENT         -> restore sentence boundaries with a heuristic (pause gaps
                        from timestamps + discourse markers + length caps).
                        Production path: a punctuation-restoration model
                        (deepmultilingualpunctuation) — guarded import below.
  6. CHUNK           -> pack restored sentences into ~N-word chunks that respect
                        speaker-turn and sentence boundaries; carry rich metadata.

SENIOR TRADEOFF
  We DO NOT auto-apply homophone fixes silently — in a clinical/financial
  transcript "to high" vs "too high" or "metphormin" vs "metformin" changes
  meaning, so corrections are surfaced with a confidence flag and the original
  is preserved in metadata. Cleaning must be auditable, not just clean.

RUN
    python transcript_ingest.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Optional

from fixtures.make_transcript import get_raw

# ---------------------------------------------------------------------------
# Guarded production path: punctuation restoration model.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - heavy optional dep, not installed in this env
    from deepmultilingualpunctuation import PunctuationModel  # type: ignore

    _PUNCT_MODEL = PunctuationModel()
    _HAS_PUNCT_MODEL = True
except Exception:  # noqa: BLE001
    _PUNCT_MODEL = None
    _HAS_PUNCT_MODEL = False


# ---------------------------------------------------------------------------
# Small, domain-specific glossaries. In production these live in a reviewed
# resource file per domain (clinical, legal, finance), NOT hardcoded — the point
# is that ASR homophone/entity errors are corpus-specific and need a curated map.
# ---------------------------------------------------------------------------
FILLERS = {
    "um", "uh", "erm", "hmm",
    "like", "you know", "i mean", "sort of", "kind of", "basically",
}

# Entity / drug / company corrections: mangled_form -> canonical_form
ENTITY_GLOSSARY = {
    "metphormin": "metformin",
    "axenture": "accenture",
    "pharamcy": "pharmacy",
}

# Homophones we FLAG (not auto-fix) because the right choice is grammatical, not
# lexical. We record the alternatives so a reviewer / downstream LLM can decide.
HOMOPHONE_SETS = [
    {"their", "there", "they're"},
    {"to", "too", "two"},
    {"your", "you're"},
]
_HOMOPHONE_LOOKUP = {w: s for s in HOMOPHONE_SETS for w in s}

# Discourse markers that reliably open a new clause/sentence in speech.
DISCOURSE_MARKERS = {"so", "and", "but", "because", "then", "right", "okay", "well", "lekin"}

# Unicode block ranges for script detection.
_DEVANAGARI = (0x0900, 0x097F)


# ===========================================================================
# Data model
# ===========================================================================
@dataclass
class Turn:
    speaker: str
    start_sec: Optional[int]
    text: str


@dataclass
class Chunk:
    chunk_id: str
    text: str
    speaker: str
    start_sec: Optional[int]
    scripts: list[str]                       # e.g. ["Latin"] or ["Latin","Devanagari"]
    code_switch: bool
    inaudible_spans: int                     # how many [inaudible] gaps this chunk swallowed
    corrections: list[dict] = field(default_factory=list)  # applied entity fixes
    homophone_flags: list[dict] = field(default_factory=list)  # surfaced, not fixed


# ===========================================================================
# 1. Parse diarized turns -> metadata
# ===========================================================================
_TURN_RE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2})\]\s*(?P<spk>SPEAKER_\d+):\s*(?P<body>.*)$"
)


def _ts_to_sec(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def parse_turns(raw: str) -> list[Turn]:
    turns: list[Turn] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _TURN_RE.match(line)
        if not m:
            # Unlabeled continuation -> attach to previous turn.
            if turns:
                turns[-1].text += " " + line
            continue
        turns.append(Turn(m["spk"], _ts_to_sec(m["ts"]), m["body"].strip()))
    return turns


# ===========================================================================
# 2. Clean: markers + fillers
# ===========================================================================
def handle_markers(text: str) -> tuple[str, int]:
    """Resolve ASR markers. [inaudible] -> a preserved placeholder token so the
    gap is queryable ("what was said around X?") rather than silently dropped.
    [overlap] -> removed (it's a diarization artifact, not content)."""
    inaudible = len(re.findall(r"\[inaudible\]", text, flags=re.I))
    text = re.sub(r"\[inaudible\]", " <INAUDIBLE> ", text, flags=re.I)
    text = re.sub(r"\[overlap\]", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(), inaudible


def strip_fillers(text: str) -> str:
    """Remove fillers. Multi-word fillers first so 'you know' isn't half-left."""
    for f in sorted(FILLERS, key=lambda x: -len(x.split())):
        text = re.sub(rf"(?<!\w){re.escape(f)}(?!\w)", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


# ===========================================================================
# 3. Normalize: entity correction (auto, flagged) + homophone (flag only)
# ===========================================================================
def normalize(text: str) -> tuple[str, list[dict], list[dict]]:
    corrections: list[dict] = []
    homophones: list[dict] = []
    out_words: list[str] = []

    for tok in text.split():
        # Preserve our own placeholder verbatim.
        if tok == "<INAUDIBLE>":
            out_words.append(tok)
            continue
        low = tok.lower().strip(".,?!;:")
        if low in ENTITY_GLOSSARY:
            fixed = ENTITY_GLOSSARY[low]
            corrections.append({"from": low, "to": fixed, "type": "entity"})
            out_words.append(fixed)
        else:
            out_words.append(tok)
            if low in _HOMOPHONE_LOOKUP:
                homophones.append({
                    "token": low,
                    "alternatives": sorted(_HOMOPHONE_LOOKUP[low]),
                    "note": "ambiguous ASR homophone — resolve in context, not silently",
                })
    return " ".join(out_words), corrections, homophones


# ===========================================================================
# 4. Script / code-switch detection (unicode block ranges)
# ===========================================================================
def detect_scripts(text: str) -> list[str]:
    scripts: set[str] = set()
    for ch in text:
        if ch.isspace() or not ch.isalpha():
            continue
        cp = ord(ch)
        if _DEVANAGARI[0] <= cp <= _DEVANAGARI[1]:
            scripts.add("Devanagari")
        elif cp < 0x80:
            scripts.add("Latin")
        else:
            # Fall back to the unicode name family for anything else.
            scripts.add(unicodedata.name(ch, "UNKNOWN").split()[0].title())
    return sorted(scripts)


def is_code_switch(text: str) -> bool:
    """Heuristic: Latin script + known romanized-Hindi function words on one line
    is code-switching even when the source is transliterated (no Devanagari).
    This catches 'haan theek hai lekin humein' which is Latin-script Hindi."""
    scripts = detect_scripts(text)
    if len([s for s in scripts if s in {"Latin", "Devanagari"}]) > 1:
        return True
    romanized_hindi = {"haan", "theek", "hai", "lekin", "humein", "phir", "karni", "hogi", "se"}
    toks = {t.lower().strip(".,?!") for t in text.split()}
    return len(toks & romanized_hindi) >= 2


# ===========================================================================
# 5. Sentence segmentation (heuristic; model is the production path)
# ===========================================================================
def restore_sentences(turn_text: str) -> list[str]:
    """Split a cleaned, unpunctuated turn into sentence-like units.

    Production path: a punctuation-restoration model reinserts . ? ! and casing,
    then split on terminals. Offline heuristic fallback: start a new sentence at
    discourse markers once the current unit is long enough to be a real clause.
    """
    if _HAS_PUNCT_MODEL:  # pragma: no cover - not installed here
        restored = _PUNCT_MODEL.restore_punctuation(turn_text)
        return [s.strip() for s in re.split(r"(?<=[.?!])\s+", restored) if s.strip()]

    words = turn_text.split()
    sentences: list[str] = []
    cur: list[str] = []
    for w in words:
        low = w.lower()
        # Boundary: a discourse marker that opens a new clause, but only if the
        # current buffer is already a plausible clause (>=4 content words). This
        # stops "so", "and" from shattering text into fragments.
        if low in DISCOURSE_MARKERS and len(cur) >= 4:
            sentences.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        sentences.append(" ".join(cur))
    return sentences


# ===========================================================================
# 6. Chunk on restored boundaries, carrying metadata
# ===========================================================================
def build_chunks(turns: list[Turn], target_words: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    # A big inter-turn timestamp gap is a strong pause boundary; we already keep
    # turns separate, so pauses are respected by never merging across speakers.
    for turn in turns:
        cleaned, inaudible = handle_markers(turn.text)
        cleaned = strip_fillers(cleaned)
        cleaned, corrections, homophones = normalize(cleaned)
        sentences = restore_sentences(cleaned)

        buf: list[str] = []
        n = 0
        for sent in sentences:
            buf.append(sent)
            n += len(sent.split())
            if n >= target_words:
                chunks.append(_mk_chunk(turn, buf, corrections, homophones))
                buf, n = [], 0
        if buf:
            chunks.append(_mk_chunk(turn, buf, corrections, homophones))
    return chunks


def _mk_chunk(turn: Turn, buf: list[str], corrections, homophones) -> Chunk:
    text = " ".join(buf)
    return Chunk(
        chunk_id=f"{turn.speaker}@{turn.start_sec}s#{len(text)}",
        text=text,
        speaker=turn.speaker,
        start_sec=turn.start_sec,
        scripts=detect_scripts(text),
        code_switch=is_code_switch(text),
        inaudible_spans=text.count("<INAUDIBLE>"),
        corrections=corrections,
        homophone_flags=homophones,
    )


# ===========================================================================
# Demo
# ===========================================================================
def _rule(title: str) -> None:
    print("\n" + "=" * 74 + f"\n{title}\n" + "=" * 74)


def main() -> None:
    raw = get_raw()
    _rule("BEFORE — raw ASR dump (what a naive chunker would index)")
    print(raw)
    print(f"punctuation-restoration model available: {_HAS_PUNCT_MODEL} "
          f"(offline heuristic in use otherwise)")

    turns = parse_turns(raw)
    chunks = build_chunks(turns)

    _rule("AFTER — clean, metadata-rich, chunk-ready units")
    for c in chunks:
        print(json.dumps(asdict(c), ensure_ascii=False, indent=2))

    _rule("WHAT THE INGEST CAUGHT")
    print(f"turns parsed .............. {len(turns)}")
    print(f"chunks emitted ............ {len(chunks)}")
    print(f"code-switch chunks ........ {sum(c.code_switch for c in chunks)}")
    print(f"entity corrections ........ "
          f"{sum(len(c.corrections) for c in chunks)} (e.g. metphormin->metformin)")
    print(f"homophone flags surfaced .. {sum(len(c.homophone_flags) for c in chunks)}")
    print(f"inaudible spans preserved . {sum(c.inaudible_spans for c in chunks)}")


if __name__ == "__main__":
    main()
