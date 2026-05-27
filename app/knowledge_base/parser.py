import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"\n{2,}")


@dataclass(frozen=True)
class Section:
    section_id: str   # e.g. "p0000" (pdf page) or "s0001" (txt section)
    heading: str | None
    content: str


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    filename: str
    file_type: Literal["pdf", "txt"]
    sections: tuple[Section, ...]
    content_hash: str


class DocumentParser:
    def parse(self, path: Path, doc_id: str | None = None) -> ParsedDocument:
        suffix = path.suffix.lower().lstrip(".")
        if suffix == "pdf":
            return self._parse_pdf(path, doc_id or str(uuid.uuid4()))
        if suffix == "txt":
            return self._parse_txt(path, doc_id or str(uuid.uuid4()))
        raise ValueError(f"Unsupported file type: .{suffix}")

    def _parse_pdf(self, path: Path, doc_id: str) -> ParsedDocument:
        reader = PdfReader(str(path))
        sections: list[Section] = []
        full_parts: list[str] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            full_parts.append(text)
            if text.strip():
                sections.append(Section(
                    section_id=f"p{i:04d}",
                    heading=f"Page {i + 1}",
                    content=text,
                ))

        if not sections:
            sections = [Section(section_id="p0000", heading=None, content="")]

        full_text = "\n".join(full_parts)
        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type="pdf",
            sections=tuple(sections),
            content_hash=hashlib.sha256(full_text.encode()).hexdigest(),
        )

    def _parse_txt(self, path: Path, doc_id: str) -> ParsedDocument:
        content = path.read_text(encoding="utf-8", errors="replace")
        sections = self._split_sections(content)
        return ParsedDocument(
            doc_id=doc_id,
            filename=path.name,
            file_type="txt",
            sections=tuple(sections),
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
        )

    def _split_sections(self, content: str) -> list[Section]:
        paragraphs = _HEADING_RE.split(content)
        sections: list[Section] = []
        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
            lines = para.splitlines()
            heading: str | None = None
            body = para
            if len(lines) > 1:
                first = lines[0].strip()
                if first and len(first) < 80 and (first.istitle() or first.isupper()):
                    heading = first
                    body = "\n".join(lines[1:])
            sections.append(Section(section_id=f"s{i:04d}", heading=heading, content=body))

        if not sections:
            sections = [Section(section_id="s0000", heading=None, content=content)]
        return sections
