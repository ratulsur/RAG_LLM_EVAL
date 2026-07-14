"""
PATHOLOGY (a): boilerplate contamination.

Every crawled page repeats the same nav bar, cookie banner, footer, and "related
articles" rail. If you embed the raw page text, those boilerplate tokens dominate
every chunk -- retrieval returns the nav menu, and near-duplicate chrome makes
distinct pages look identical to the vector index.

Fix -- main-content extraction. Here we HAND-ROLL a density/tag heuristic with
bs4 so the mechanics are explicit (no black box):
  1. strip non-content tags outright (script/style/nav/header/footer/aside/form),
  2. also strip elements whose id/class scream boilerplate (cookie, banner, menu),
  3. score every remaining block-level node by text length and (1 - link_density)
     -- real article prose is long and lightly linked; nav/related rails are short
     and almost all links,
  4. return the highest-scoring subtree's text.

Production note: trafilatura or readability-lxml do this far more robustly
(they add tag-path priors, comment stripping, language detection). We import
trafilatura behind a guard and fall back to the hand-rolled extractor so the file
runs with or without it. Interview line: "I can explain the density heuristic AND
I know the library that productionizes it -- I don't ship a naive .get_text()."
"""
from __future__ import annotations

import os
import re

from bs4 import BeautifulSoup, Tag

try:
    import trafilatura  # optional; pip install trafilatura
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

HERE = os.path.dirname(os.path.abspath(__file__))

_STRIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form",
               "noscript", "svg", "button"}
_BOILERPLATE_HINT = re.compile(
    r"cookie|banner|consent|nav|menu|footer|header|sidebar|related|promo|social|"
    r"breadcrumb|newsletter|subscribe", re.IGNORECASE)


def _link_density(node: Tag) -> float:
    text = node.get_text(" ", strip=True)
    if not text:
        return 1.0
    link_text = " ".join(a.get_text(" ", strip=True) for a in node.find_all("a"))
    return len(link_text) / max(len(text), 1)


def _strip_chrome(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(list(_STRIP_TAGS)):
        tag.decompose()
    # id/class-based boilerplate not caught by tag name. Decomposing during
    # iteration detaches descendants (attrs -> None), so skip already-detached
    # nodes rather than crash on them.
    for el in soup.find_all(True):
        if el.attrs is None:      # detached by an earlier decompose()
            continue
        ident = " ".join(filter(None, [el.get("id", ""),
                                        " ".join(el.get("class", []) or [])]))
        if ident and _BOILERPLATE_HINT.search(ident):
            el.decompose()


def extract_main_bs4(html: str) -> str:
    """Hand-rolled density/tag heuristic main-content extractor."""
    soup = BeautifulSoup(html, "lxml")
    _strip_chrome(soup)

    # Prefer a semantic <article>/<main> if one survived stripping.
    for sel in ("article", "main"):
        node = soup.find(sel)
        if node and len(node.get_text(strip=True)) > 200:
            return _clean_text(node.get_text(" ", strip=True))

    # Otherwise score candidate blocks by length * (1 - link_density).
    best, best_score = None, 0.0
    for node in soup.find_all(["div", "section", "article", "main"]):
        text = node.get_text(" ", strip=True)
        score = len(text) * (1.0 - _link_density(node))
        if score > best_score:
            best, best_score = node, score
    body_text = best.get_text(" ", strip=True) if best else soup.get_text(" ", strip=True)
    return _clean_text(body_text)


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_main(html: str) -> tuple[str, str]:
    """Return (text, engine). Uses trafilatura when available, else the heuristic."""
    if _HAS_TRAFILATURA:
        out = trafilatura.extract(html, include_comments=False, include_tables=False)
        if out and len(out) > 100:
            return _clean_text(out), "trafilatura"
    return extract_main_bs4(html), "bs4_heuristic"


def _demo():
    from fixtures.make_fixtures import ARTICLE

    print("=" * 70)
    print("MAIN-CONTENT EXTRACTION (strip nav/cookie/footer/related)")
    print("=" * 70)

    raw = _clean_text(BeautifulSoup(ARTICLE, "lxml").get_text(" ", strip=True))
    print(f"\n[BEFORE] naive get_text() -- {len(raw)} chars, boilerplate included:")
    print("   ", raw[:320], "...")

    text, engine = extract_main(ARTICLE)
    print(f"\n[AFTER] engine={engine} ({'live' if engine=='trafilatura' else 'reference lib absent -> heuristic'})")
    print(f"[AFTER] {len(text)} chars, article only:")
    print("   ", text)

    contaminated = any(w in text.lower() for w in
                       ["accept all", "cookie", "related articles", "all rights reserved", "login"])
    print(f"\n[CHECK] boilerplate leaked into output? {contaminated}")


if __name__ == "__main__":
    _demo()
