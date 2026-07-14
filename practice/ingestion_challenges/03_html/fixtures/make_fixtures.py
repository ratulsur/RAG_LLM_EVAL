"""
Synthesize DIRTY HTML fixtures so the handlers run with zero network / no files.

Returns a dict of name -> html string (also written to *.html next to this file):
  article   -- real content buried in nav + cookie banner + footer + "related"
  divtable  -- a tabular grid built from <div>s (no <table>)
  js_page   -- near-empty body, content only inside __NEXT_DATA__ script JSON
  canonical -- clean article at a canonical URL
  utm_dup   -- the SAME article via a UTM-tagged URL + a print wrapper (near-dup)
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))

ARTICLE = """<!doctype html><html><head><title>RAG Chunking Guide</title>
<style>.x{color:red}</style><script>var a=1;</script></head>
<body>
<header id="site-header"><a href="/">Home</a><a href="/blog">Blog</a>
<a href="/pricing">Pricing</a><a href="/login">Login</a></header>
<nav class="breadcrumbs"><a href="/">Home</a> / <a href="/blog">Blog</a> / Chunking</nav>
<div class="cookie-banner">We use cookies to improve your experience.
<button>Accept all</button><a href="/privacy">Privacy policy</a></div>
<main>
  <article>
    <h1>How to Chunk Documents for RAG</h1>
    <p>Fixed-size chunking is simple but severs semantic units, which hurts
       retrieval precision. Recursive character splitting respects paragraph and
       sentence boundaries, so each chunk is a more self-contained idea.</p>
    <p>For long technical PDFs, parent-document retrieval indexes small child
       chunks for precision but returns the larger parent for context. This keeps
       embeddings tight while giving the LLM enough surrounding text to answer.</p>
    <p>Always attach metadata -- source, section heading, page -- to every chunk;
       retrieval without provenance cannot be grounded or audited.</p>
  </article>
</main>
<aside class="related"><h3>Related articles</h3><ul>
<li><a href="/a">Vector DBs compared</a></li><li><a href="/b">Reranking 101</a></li>
<li><a href="/c">HyDE explained</a></li></ul></aside>
<footer><a href="/terms">Terms</a><a href="/contact">Contact</a>
<span>(c) 2024 ExampleCorp. All rights reserved.</span></footer>
</body></html>"""

DIVTABLE = """<!doctype html><html><head><title>Pricing</title></head><body>
<div class="grid">
  <div class="row header"><div class="cell">Plan</div><div class="cell">Seats</div>
    <div class="cell">Price (USD)</div></div>
  <div class="row"><div class="cell">Starter</div><div class="cell">3</div>
    <div class="cell">49</div></div>
  <div class="row"><div class="cell">Team</div><div class="cell">10</div>
    <div class="cell">199</div></div>
  <div class="row"><div class="cell">Enterprise</div><div class="cell">50</div>
    <div class="cell">899</div></div>
</div>
</body></html>"""

# JS-rendered SPA: body is a near-empty mount point; content lives in Next.js JSON.
JS_PAGE = """<!doctype html><html><head><title>Report</title></head><body>
<div id="__next"></div>
<script id="__NEXT_DATA__" type="application/json">
{"props":{"pageProps":{"article":{"title":"Q3 Earnings Summary",
"body":"Revenue rose 14% YoY driven by cloud. Operating margin expanded to 22%.",
"tags":["earnings","cloud"]}}},"page":"/report","buildId":"abc123"}
</script>
<script src="/_next/static/chunks/main.js"></script>
</body></html>"""

_ARTICLE_BODY = """<main><article><h1>How to Chunk Documents for RAG</h1>
<p>Fixed-size chunking is simple but severs semantic units, which hurts retrieval
precision. Recursive character splitting respects paragraph and sentence
boundaries, so each chunk is a more self-contained idea.</p>
<p>For long technical PDFs, parent-document retrieval indexes small child chunks
for precision but returns the larger parent for context.</p></article></main>"""

CANONICAL = f"""<!doctype html><html><head><title>RAG Chunking Guide</title>
<link rel="canonical" href="https://ex.com/blog/chunking"/></head>
<body><nav>menu</nav>{_ARTICLE_BODY}<footer>(c) 2024</footer></body></html>"""

# same content, UTM-tagged URL + a "print" chrome difference => near-duplicate
UTM_DUP = f"""<!doctype html><html><head><title>RAG Chunking Guide (print)</title>
</head><body><div class="print-header">Printed from ExampleCorp</div>
{_ARTICLE_BODY}</body></html>"""

FIXTURES = {
    "article": ARTICLE, "divtable": DIVTABLE, "js_page": JS_PAGE,
    "canonical": CANONICAL, "utm_dup": UTM_DUP,
}

# URLs paired with the dedup fixtures.
#  - exact_utm collapses to `canonical` purely by URL canonicalization.
#  - print_variant has a DIFFERENT path, so only the CONTENT near-dup check
#    (MinHash) can catch it.
URLS = {
    "canonical": "https://ex.com/blog/chunking",
    "exact_utm": "https://ex.com/blog/chunking?utm_source=twitter&utm_campaign=x&ref=nl#top",
    "print_variant": "https://ex.com/print/blog/chunking?format=print",
}


def write_all() -> dict[str, str]:
    for name, html in FIXTURES.items():
        with open(os.path.join(HERE, f"{name}.html"), "w", encoding="utf-8") as f:
            f.write(html)
    return FIXTURES


if __name__ == "__main__":
    write_all()
    print("wrote:", ", ".join(f"{n}.html" for n in FIXTURES))
