# 05 — Email ingestion pathologies

Email is a *container* format, and containers nest. The real information often
sits in the parts naive extraction skips: quoted history buries the one new
sentence, the actual numbers are in a pasted screenshot, an NDA is zipped inside
an attachment, and a thread arrives as loose out-of-order files.

## Files

| File | Pathology | Live vs reference |
|------|-----------|-------------------|
| `fixtures/generate.py` | Synthesizes `.eml` fixtures (quoted chain, multipart+nested zip, out-of-order thread) | **live** (stdlib email + Pillow PNG) |
| `quoted_chains_signatures.py` | **(a)** Split new vs quoted history, strip signature + disclaimer, dedupe | **live** (stdlib email) |
| `mime_tree_attachments.py` | **(b)** Recursive MIME walk; crack nested zip→eml; flag inline images | **live** walk + zip crack; **OCR guarded** (image flagged, pytesseract absent) |
| `thread_reconstruction.py` | **(c)** Rebuild thread tree from Message-ID / In-Reply-To / References | **live** (RFC 5322 headers) |

## Why this is a strong RAG answer

- **Quoted chains:** `msg.get_content()` returns the whole visible body — the new
  reply *plus* every quoted ancestor (`>`, `>>`, `>>>`), the "On … wrote:"
  attributions, the `-- ` signature, and the confidentiality notice. Ingest a
  5-deep thread message-by-message and the oldest text is embedded 5×; retrieval
  returns near-duplicates and disclaimers swamp term stats. I segment into
  NEW / QUOTED / SIGNATURE / DISCLAIMER, chunk only NEW, and dedupe by normalized
  line hash. Heuristics cover the RFC-style majority; production adds a trained
  splitter (Mailgun `talon`) for Outlook `-----Original Message-----` and mobile
  sigs — and I degrade to "keep everything" rather than drop the new reply.
- **Nested MIME:** walk the tree recursively *and recurse into containers* —
  text extracted, images flagged `info-in-pixels` for OCR, `.zip` opened in
  memory and each member re-parsed (an `.eml` inside becomes a child message). On
  the fixture this recovers a countersigned NDA buried in `attachments.zip`
  →`nda_thread.eml` and a screenshot table the body only alludes to. Depth/byte
  caps guard against zip-bombs and rfc822 nesting.
- **Threading:** grouping by subject fails ("Re:", "RE: … [updated]", "Fwd:" look
  like three subjects but are one thread); sorting by time flattens a tree into a
  line and loses branches. I rebuild from Message-ID / In-Reply-To / References
  (JWZ-style): index by id, link to parent, root = no resolvable parent, order
  siblings by date, walk depth-first. On the fixture, two replies to the same
  message correctly render as *siblings*, not a false sequence. Missing ids fall
  back to References then normalized-subject + time window; orphans surface as
  secondary roots rather than attaching to the wrong parent.

## Run

```bash
python fixtures/generate.py
python quoted_chains_signatures.py
python mime_tree_attachments.py
python thread_reconstruction.py
```

Extra deps leaned on: none beyond the core set (stdlib `email`/`zipfile` +
Pillow, all pre-installed). OCR for inline images is guarded-optional.
