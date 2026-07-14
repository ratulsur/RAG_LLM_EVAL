"""
PATHOLOGY (b) JS-rendered content + (c) <div>-based tables.

(b) A Single-Page-App ships a near-empty <body> (a mount div) plus a big JS
    bundle; the visible content is injected client-side or embedded in a state
    blob (Next.js __NEXT_DATA__, Nuxt __NUXT__, or a JSON-LD script). A crawler
    reading raw HTML gets nothing. Fix: DETECT the SPA shape (tiny visible text
    but heavy <script>, or a known state script present) and either (i) harvest
    the embedded JSON directly -- fast, no browser -- or (ii) route to a headless
    renderer (playwright) as a last resort.

(c) Tables built from <div> grids (CSS flex/grid) have no <table>/<tr>/<td>, so
    pd.read_html finds nothing. Fix: detect the repeating row/cell class pattern
    and reconstruct a DataFrame from it.

Interview line: "Before spinning up a headless browser -- which is 100x slower --
I check for __NEXT_DATA__/JSON-LD. Most 'JS-only' pages ship their data in the
HTML; you just have to know where to look."
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from playwright.sync_api import sync_playwright  # optional headless renderer
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False


@dataclass
class RenderDecision:
    needs_render: bool
    reason: str
    visible_chars: int
    script_chars: int
    state_script: str | None   # id of an embedded-state script if found


# --------------------------------------------------------------------------- #
# (b) JS-render detection + embedded-JSON harvesting
# --------------------------------------------------------------------------- #
_STATE_SCRIPT_IDS = ("__NEXT_DATA__", "__NUXT__")


def diagnose_render(html: str) -> RenderDecision:
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    for s in body.find_all("script"):
        pass  # keep scripts for size measurement below
    visible = body.get_text(" ", strip=True)
    # visible text with scripts removed
    tmp = BeautifulSoup(html, "lxml")
    for s in tmp.find_all(["script", "style"]):
        s.decompose()
    visible_text = (tmp.body or tmp).get_text(" ", strip=True)
    script_chars = sum(len(s.get_text()) for s in soup.find_all("script"))

    state = None
    for sid in _STATE_SCRIPT_IDS:
        if soup.find("script", id=sid):
            state = sid
            break
    if soup.find("script", type="application/ld+json"):
        state = state or "ld+json"

    needs = len(visible_text) < 200 and (script_chars > len(visible_text) or state is not None)
    reason = ("thin body + embedded state/heavy JS" if needs
              else "server-rendered content present")
    return RenderDecision(needs, reason, len(visible_text), script_chars, state)


def harvest_embedded_json(html: str) -> dict | None:
    """Pull structured content out of __NEXT_DATA__ / __NUXT__ / JSON-LD scripts
    without rendering. Returns the parsed object, or None."""
    soup = BeautifulSoup(html, "lxml")
    for sid in _STATE_SCRIPT_IDS:
        tag = soup.find("script", id=sid)
        if tag and tag.string:
            try:
                return json.loads(tag.string)
            except json.JSONDecodeError:
                continue
    ld = soup.find("script", type="application/ld+json")
    if ld and ld.string:
        try:
            return json.loads(ld.string)
        except json.JSONDecodeError:
            return None
    return None


def render_with_playwright(url: str) -> str:  # pragma: no cover - reference path
    """Reference last-resort path (requires: pip install playwright && playwright
    install chromium). Not exercised in the offline demo."""
    if not _HAS_PLAYWRIGHT:
        raise RuntimeError("playwright not installed; this is the reference path")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
        return html


# --------------------------------------------------------------------------- #
# (c) reconstruct a table from a <div> grid
# --------------------------------------------------------------------------- #
def _dominant_row_class(soup: BeautifulSoup) -> str | None:
    """The class that repeats most as a sibling group is the 'row' class."""
    counts = Counter()
    for div in soup.find_all("div"):
        for cls in (div.get("class") or []):
            counts[cls] += 1
    for cls, n in counts.most_common():
        if n >= 2:                       # a row class appears once per data row
            # require that such divs contain >=2 child divs (cells)
            sample = soup.find("div", class_=cls)
            if sample and len(sample.find_all("div", recursive=False)) >= 2:
                return cls
    return None


def divgrid_to_df(html: str) -> pd.DataFrame | None:
    soup = BeautifulSoup(html, "lxml")
    row_cls = _dominant_row_class(soup)
    if not row_cls:
        return None
    rows = []
    for rowdiv in soup.find_all("div", class_=row_cls):
        cells = [c.get_text(" ", strip=True)
                 for c in rowdiv.find_all("div", recursive=False)]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return None
    # first row is the header if it looks like labels (any non-numeric cell)
    header, body = rows[0], rows[1:]
    width = max(len(r) for r in rows)
    body = [(r + [None] * width)[:width] for r in body]
    return pd.DataFrame(body, columns=(header + [f"col_{i}" for i in range(width)])[:width])


def _demo():
    from fixtures.make_fixtures import JS_PAGE, DIVTABLE, ARTICLE

    print("=" * 70)
    print("JS-RENDER DETECTION + EMBEDDED-JSON HARVEST")
    print("=" * 70)
    for name, html in [("server-rendered article", ARTICLE), ("SPA js_page", JS_PAGE)]:
        d = diagnose_render(html)
        print(f"\n[{name}] needs_render={d.needs_render}  reason={d.reason}")
        print(f"    visible_chars={d.visible_chars}  script_chars={d.script_chars}"
              f"  state_script={d.state_script}")
    print(f"\n[playwright available? {_HAS_PLAYWRIGHT}]  -- SPA routed to embedded JSON:")
    data = harvest_embedded_json(JS_PAGE)
    art = data["props"]["pageProps"]["article"]
    print("    title:", art["title"])
    print("    body :", art["body"])
    print("    (no headless browser needed -- content was in __NEXT_DATA__)")

    print("\n" + "=" * 70)
    print("DIV-GRID -> DATAFRAME RECONSTRUCTION")
    print("=" * 70)
    print("\n[BEFORE] pd.read_html finds no <table>:")
    import io as _io
    try:
        pd.read_html(_io.StringIO(DIVTABLE), flavor="lxml")
        print("    (unexpectedly found a table)")
    except Exception as e:  # noqa: BLE001  -> ValueError('No tables found')
        print(f"    {type(e).__name__}: {e}")
    print("\n[AFTER] reconstructed from <div> grid:")
    print(divgrid_to_df(DIVTABLE).to_string(index=False))


if __name__ == "__main__":
    _demo()
