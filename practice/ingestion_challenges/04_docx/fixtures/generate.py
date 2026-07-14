"""
Synthetic dirty-DOCX generator for the 04_docx challenges.

A .docx is a ZIP of WordprocessingML parts. python-docx does NOT create tracked
changes, comments, or footnotes for you, so we hand-author the OOXML to bake in
exactly the pathologies the handlers must recover:

  dirty.docx contains
    - a tracked INSERTION (<w:ins>) and a tracked DELETION (<w:del>)
    - a COMMENT anchored to body text (comments.xml + commentReference)
    - a FOOTNOTE (footnotes.xml + footnoteReference)
    - a TEXT BOX (VML <w:pict>) whose text naive parsers drop
    - a running HEADER and FOOTER (header1.xml / footer1.xml)
    - a style-generated NUMBERED list (numbering.xml) -> "Section 4.2" refs

Also emits two format-detection decoys:
    legacy_doc.bin   OLE2 magic bytes (a real legacy .doc container signature)
    renamed.docx     actually a ZIP/OOXML but the point is to detect by content

Run `python generate.py` to build them into ./ (this dir).
"""
from __future__ import annotations

import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
VML = ('xmlns:v="urn:schemas-microsoft-com:vml" '
       'xmlns:w10="urn:schemas-microsoft-com:office:word"')

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/comments.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml"/>
  <Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/header1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml"/>
  <Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>
</Types>"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

DOC_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdC" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments" Target="comments.xml"/>
  <Relationship Id="rIdF" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes" Target="footnotes.xml"/>
  <Relationship Id="rIdN" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
  <Relationship Id="rIdS" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rIdH" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/header" Target="header1.xml"/>
  <Relationship Id="rIdFt" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>
</Relationships>"""

DOCUMENT = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document {W} {VML} xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">
  <w:body>
    <w:p><w:r><w:t xml:space="preserve">The clause below is the operative agreement. </w:t></w:r></w:p>

    <!-- tracked changes: an accepted-insertion and a rejected-deletion -->
    <w:p>
      <w:r><w:t xml:space="preserve">The vendor shall deliver within </w:t></w:r>
      <w:del w:id="1" w:author="Legal" w:date="2026-07-10T00:00:00Z">
        <w:r><w:delText xml:space="preserve">30 </w:delText></w:r>
      </w:del>
      <w:ins w:id="2" w:author="Legal" w:date="2026-07-10T00:00:00Z">
        <w:r><w:t xml:space="preserve">15 </w:t></w:r>
      </w:ins>
      <w:r><w:t xml:space="preserve">business days.</w:t></w:r>
    </w:p>

    <!-- comment anchored to text -->
    <w:p>
      <w:commentRangeStart w:id="10"/>
      <w:r><w:t xml:space="preserve">Payment terms are net sixty.</w:t></w:r>
      <w:commentRangeEnd w:id="10"/>
      <w:r><w:commentReference w:id="10"/></w:r>
    </w:p>

    <!-- footnote reference -->
    <w:p>
      <w:r><w:t xml:space="preserve">Liability is capped as stated</w:t></w:r>
      <w:r><w:footnoteReference w:id="2"/></w:r>
      <w:r><w:t xml:space="preserve">.</w:t></w:r>
    </w:p>

    <!-- numbered list driven by numbering.xml (style-generated "Section" numbers) -->
    <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>Scope of work</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>Deliverables</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>Acceptance testing</w:t></w:r></w:p>
    <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="1"/></w:numPr></w:pPr>
      <w:r><w:t>Warranty period</w:t></w:r></w:p>

    <!-- text box: content naive parsers silently drop -->
    <w:p>
      <w:r>
        <w:pict>
          <v:shape type="#_x0000_t202" style="width:200pt;height:40pt">
            <v:textbox>
              <w:txbxContent>
                <w:p><w:r><w:t>SIDEBAR: escrow release requires dual sign-off.</w:t></w:r></w:p>
              </w:txbxContent>
            </v:textbox>
          </v:shape>
        </w:pict>
      </w:r>
    </w:p>

    <w:sectPr>
      <w:headerReference w:type="default" r:id="rIdH" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
      <w:footerReference w:type="default" r:id="rIdFt" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>
    </w:sectPr>
  </w:body>
</w:document>"""

COMMENTS = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments {W}>
  <w:comment w:id="10" w:author="Reviewer" w:date="2026-07-11T00:00:00Z" w:initials="RV">
    <w:p><w:r><w:t>Confirm net-60 is approved by finance before signing.</w:t></w:r></w:p>
  </w:comment>
</w:comments>"""

FOOTNOTES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:footnotes {W}>
  <w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>
  <w:footnote w:id="2">
    <w:p><w:r><w:t>Cap equals total fees paid in the trailing twelve months.</w:t></w:r></w:p>
  </w:footnote>
</w:footnotes>"""

NUMBERING = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering {W}>
  <w:abstractNum w:abstractNumId="0">
    <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="Section %1"/></w:lvl>
    <w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1.%2"/></w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""

STYLES = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles {W}><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style></w:styles>"""

HEADER = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:hdr {W}><w:p><w:r><w:t>ACME MASTER SERVICES AGREEMENT -- CONFIDENTIAL</w:t></w:r></w:p></w:hdr>"""

FOOTER = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr {W}><w:p><w:r><w:t>Page 1 -- (c) 2026 ACME Corp</w:t></w:r></w:p></w:ftr>"""


def make_dirty_docx() -> str:
    path = os.path.join(HERE, "dirty.docx")
    parts = {
        "[Content_Types].xml": CONTENT_TYPES,
        "_rels/.rels": RELS,
        "word/_rels/document.xml.rels": DOC_RELS,
        "word/document.xml": DOCUMENT,
        "word/comments.xml": COMMENTS,
        "word/footnotes.xml": FOOTNOTES,
        "word/numbering.xml": NUMBERING,
        "word/styles.xml": STYLES,
        "word/header1.xml": HEADER,
        "word/footer1.xml": FOOTER,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in parts.items():
            z.writestr(name, data)
    return path


def make_format_decoys():
    """A legacy OLE2 .doc (magic D0CF11E0) and a mislabelled file."""
    ole = os.path.join(HERE, "legacy_doc.bin")
    with open(ole, "wb") as f:
        f.write(bytes([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) + b"\x00" * 64)
    # a real OOXML zip but we'll pretend the extension was lost
    renamed = os.path.join(HERE, "renamed.unknown")
    src = os.path.join(HERE, "dirty.docx")
    if os.path.exists(src):
        with open(src, "rb") as a, open(renamed, "wb") as b:
            b.write(a.read())
    return ole, renamed


def build_all():
    p = make_dirty_docx()
    ole, renamed = make_format_decoys()
    for f in (p, ole, renamed):
        print("wrote", os.path.relpath(f, HERE), f"({os.path.getsize(f)} bytes)")


if __name__ == "__main__":
    build_all()
