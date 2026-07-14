"""
Synthetic dirty-email generator for the 05_email challenges. Pure stdlib
(email, zipfile) + Pillow for a tiny inline PNG, so the handlers run with no
external files.

  quoted_chain.eml     a 3-deep reply: newest-first, quoted history repeated,
                       signature + legal disclaimer as trailing noise
  multipart_nested.eml multipart/mixed with an inline PNG (info-in-image) and a
                       .zip attachment that itself contains an .eml (nested MIME)
  thread/*.eml         4 messages of one thread saved OUT OF ORDER, with altered
                       subjects (Re:/Fwd:) -- reassemble via Message-ID headers
"""
from __future__ import annotations

import io
import os
import zipfile
from email.message import EmailMessage
from email.utils import format_datetime
from datetime import datetime, timezone, timedelta

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def _out(name: str) -> str:
    return os.path.join(HERE, name)


# --------------------------------------------------------------------------
# (a) quoted reply chain + signature + disclaimer
# --------------------------------------------------------------------------
QUOTED_BODY = """Thanks, approved. Ship it Friday.

--
Ratul Sur
Senior GenAI Engineer | ACME
Mobile: +1-555-0100

On Mon, 12 Jul 2026 at 09:00, Bob <bob@acme.com> wrote:
> Budget looks fine on my side. Any objection to Friday?
>
> --
> Bob Chen
> Finance
>
> On Sun, 11 Jul 2026 at 18:30, Ratul <ratul@acme.com> wrote:
>> Here is the revised plan. Cost is 12k, down from 20k.
>> Please confirm the Friday ship date works.
>>
>> On Sun, 11 Jul 2026 at 08:00, Bob <bob@acme.com> wrote:
>>> Can you send the revised Q3 plan with updated costs?

CONFIDENTIALITY NOTICE: This email and any attachments are confidential and may
be legally privileged. If you are not the intended recipient, delete it and
notify the sender. ACME Corp accepts no liability for the contents.
"""


def make_quoted_chain() -> str:
    m = EmailMessage()
    m["From"] = "Ratul <ratul@acme.com>"
    m["To"] = "Bob <bob@acme.com>"
    m["Subject"] = "Re: Re: Q3 plan"
    m["Date"] = format_datetime(datetime(2026, 7, 12, 9, 15, tzinfo=timezone.utc))
    m["Message-ID"] = "<final@acme.com>"
    m.set_content(QUOTED_BODY)
    path = _out("quoted_chain.eml")
    with open(path, "wb") as f:
        f.write(bytes(m))
    return path


# --------------------------------------------------------------------------
# (b) multipart with inline image + nested zip(.eml)
# --------------------------------------------------------------------------
def _png_bytes() -> bytes:
    img = Image.new("RGB", (240, 80), "white")
    d = ImageDraw.Draw(img)
    # a "screenshot table" -- the real info is pixels, not text
    d.rectangle([2, 2, 237, 77], outline="black")
    d.text((10, 10), "Q3 REV: 260k", fill="black")
    d.text((10, 40), "STATUS: APPROVED", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _inner_eml() -> bytes:
    inner = EmailMessage()
    inner["From"] = "legal@acme.com"
    inner["To"] = "ratul@acme.com"
    inner["Subject"] = "NDA countersigned"
    inner["Message-ID"] = "<nested-nda@acme.com>"
    inner.set_content("The countersigned NDA is attached to the parent thread. Effective 2026-07-01.")
    return bytes(inner)


def make_multipart_nested() -> str:
    m = EmailMessage()
    m["From"] = "ratul@acme.com"
    m["To"] = "team@acme.com"
    m["Subject"] = "Q3 numbers (see screenshot) + NDA"
    m["Message-ID"] = "<multipart@acme.com>"
    m["Date"] = format_datetime(datetime(2026, 7, 12, 10, 0, tzinfo=timezone.utc))
    m.set_content("Body text is thin; the real numbers are in the screenshot below.\n")

    # inline image
    m.add_attachment(_png_bytes(), maintype="image", subtype="png",
                     filename="q3_table.png", disposition="inline", cid="q3img")

    # nested: a zip that contains an .eml
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("nda_thread.eml", _inner_eml())
    m.add_attachment(zbuf.getvalue(), maintype="application", subtype="zip",
                     filename="attachments.zip")

    path = _out("multipart_nested.eml")
    with open(path, "wb") as f:
        f.write(bytes(m))
    return path


# --------------------------------------------------------------------------
# (c) thread, saved out of order, subjects altered
# --------------------------------------------------------------------------
def _msg(mid, subject, date, in_reply_to=None, references=None, body="...") -> bytes:
    m = EmailMessage()
    m["From"] = "someone@acme.com"
    m["To"] = "team@acme.com"
    m["Subject"] = subject
    m["Date"] = format_datetime(date)
    m["Message-ID"] = mid
    if in_reply_to:
        m["In-Reply-To"] = in_reply_to
    if references:
        m["References"] = " ".join(references)
    m.set_content(body)
    return bytes(m)


def make_thread():
    base = datetime(2026, 7, 10, 8, 0, tzinfo=timezone.utc)
    m1 = ("<t1@acme.com>", "Q3 planning", base, None, None, "Kicking off Q3 planning.")
    m2 = ("<t2@acme.com>", "Re: Q3 planning", base + timedelta(hours=2),
          "<t1@acme.com>", ["<t1@acme.com>"], "Draft budget attached.")
    m3 = ("<t3@acme.com>", "RE: Q3 planning [updated]", base + timedelta(hours=5),
          "<t2@acme.com>", ["<t1@acme.com>", "<t2@acme.com>"], "Revised numbers.")
    m4 = ("<t4@acme.com>", "Fwd: Q3 planning", base + timedelta(hours=6),
          "<t2@acme.com>", ["<t1@acme.com>", "<t2@acme.com>"], "Looping in finance.")
    specs = {"a_m3.eml": m3, "b_m1.eml": m1, "c_m4.eml": m4, "d_m2.eml": m2}  # out of order
    tdir = _out("thread")
    os.makedirs(tdir, exist_ok=True)
    paths = []
    for fname, spec in specs.items():
        p = os.path.join(tdir, fname)
        with open(p, "wb") as f:
            f.write(_msg(*spec))
        paths.append(p)
    return paths


def build_all():
    outs = [make_quoted_chain(), make_multipart_nested()]
    outs += make_thread()
    for p in outs:
        print("wrote", os.path.relpath(p, HERE), f"({os.path.getsize(p)} bytes)")


if __name__ == "__main__":
    build_all()
