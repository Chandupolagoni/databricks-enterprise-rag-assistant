"""Multi-format document parsing and text extraction.

One parser per format, all returning the same ``ParsedDocument`` shape so the rest of
the pipeline does not care whether a policy arrived as a PDF, a Word file or an
intranet HTML page.

  PDF   -> pdfplumber (layout-aware text), falling back to pypdf
  DOCX  -> python-docx (keeps heading structure)
  HTML  -> BeautifulSoup (drops nav/footer/script boilerplate)
  MD/TXT-> read as-is

On Databricks, scanned PDFs can instead go through the ``ai_parse_document`` SQL
function (the managed equivalent of an OCR service such as AWS Textract) — see
notebooks/02_parse_and_chunk.py.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    heading: str
    text: str


@dataclass
class ParsedDocument:
    doc_id: str
    title: str
    file_format: str
    sections: list[Section] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(f"{s.heading}\n{s.text}".strip() for s in self.sections)


_WS = re.compile(r"[ \t]+")


def clean_text(text: str) -> str:
    text = text.replace(" ", " ").replace("\r", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # de-hyphenate line-wrapped words
    text = _WS.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------- parsers
def parse_pdf(data: bytes) -> tuple[str, list[Section]]:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            title = (pdf.metadata or {}).get("Title", "") or ""
            pages = [p.extract_text() or "" for p in pdf.pages]
    except Exception:  # pragma: no cover - fallback path
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        title = (reader.metadata.title if reader.metadata else "") or ""
        pages = [p.extract_text() or "" for p in reader.pages]
    text = clean_text("\n".join(pages))
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not title and lines:
        title = lines[0]
    # PDF has no structural headings: treat short title-cased lines without a period as headings.
    sections, heading, buf = [], "", []
    for ln in lines:
        if ln == title:
            continue
        if len(ln) < 60 and not ln.endswith(".") and ln[:1].isupper() and len(ln.split()) <= 6:
            if buf:
                sections.append(Section(heading, clean_text(" ".join(buf))))
            heading, buf = ln, []
        else:
            buf.append(ln)
    if buf:
        sections.append(Section(heading, clean_text(" ".join(buf))))
    return title, sections


def parse_docx(data: bytes) -> tuple[str, list[Section]]:
    from docx import Document

    doc = Document(io.BytesIO(data))
    title = doc.core_properties.title or ""
    sections, heading, buf = [], "", []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if not txt:
            continue
        style = (p.style.name or "").lower()
        if style.startswith("heading 1") or style == "title":
            title = title or txt
            continue
        if style.startswith("heading"):
            if buf:
                sections.append(Section(heading, clean_text(" ".join(buf))))
            heading, buf = txt, []
        else:
            buf.append(txt)
    if buf:
        sections.append(Section(heading, clean_text(" ".join(buf))))
    return title, sections


def parse_html(data: bytes) -> tuple[str, list[Section]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    h1 = soup.find("h1")
    title = (h1.get_text(strip=True) if h1 else "") or (soup.title.get_text(strip=True) if soup.title else "")
    sections, heading, buf = [], "", []
    for el in soup.find_all(["h2", "h3", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if el.name in ("h2", "h3"):
            if buf:
                sections.append(Section(heading, clean_text(" ".join(buf))))
            heading, buf = txt, []
        elif txt:
            buf.append(txt)
    if buf:
        sections.append(Section(heading, clean_text(" ".join(buf))))
    return title, sections


def parse_markdown(data: bytes) -> tuple[str, list[Section]]:
    text = data.decode("utf-8", errors="replace")
    title, sections, heading, buf = "", [], "", []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if ln.startswith("# "):
            title = ln[2:].strip()
        elif ln.startswith("## ") or ln.startswith("### "):
            if buf:
                sections.append(Section(heading, clean_text(" ".join(buf))))
            heading, buf = ln.lstrip("#").strip(), []
        elif re.fullmatch(r"=+", ln.strip()) or re.fullmatch(r"-+", ln.strip()):
            continue
        elif nxt and re.fullmatch(r"=+", nxt.strip()):   # setext H1 (plain text files)
            title = ln.strip()
        elif nxt and re.fullmatch(r"-+", nxt.strip()):   # setext H2
            if buf:
                sections.append(Section(heading, clean_text(" ".join(buf))))
            heading, buf = ln.strip(), []
        elif ln.strip():
            buf.append(ln.strip())
    if buf:
        sections.append(Section(heading, clean_text(" ".join(buf))))
    return title, sections


PARSERS = {
    "pdf": parse_pdf,
    "docx": parse_docx,
    "html": parse_html,
    "htm": parse_html,
    "md": parse_markdown,
    "txt": parse_markdown,
}


def parse_bytes(data: bytes, file_format: str, doc_id: str = "") -> ParsedDocument:
    fmt = file_format.lower().lstrip(".")
    if fmt not in PARSERS:
        raise ValueError(f"unsupported format: {fmt}")
    title, sections = PARSERS[fmt](data)
    sections = [s for s in sections if s.text]
    return ParsedDocument(doc_id=doc_id, title=title, file_format=fmt, sections=sections)


def parse_file(path: str | Path, doc_id: str | None = None) -> ParsedDocument:
    path = Path(path)
    return parse_bytes(path.read_bytes(), path.suffix, doc_id or path.stem)
