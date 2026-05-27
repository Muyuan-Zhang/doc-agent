"""Unit tests for DocumentParser."""
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.knowledge_base.parser import DocumentParser, ParsedDocument, Section


class TestDocumentParserTxt:
    def _write_txt(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "test.txt"
        p.write_text(content, encoding="utf-8")
        return p

    def test_parse_txt_returns_parsed_document(self, tmp_path):
        path = self._write_txt(tmp_path, "Hello world.")
        doc = DocumentParser().parse(path)
        assert isinstance(doc, ParsedDocument)

    def test_parse_txt_file_type_is_txt(self, tmp_path):
        path = self._write_txt(tmp_path, "Hello world.")
        doc = DocumentParser().parse(path)
        assert doc.file_type == "txt"

    def test_parse_txt_filename_matches(self, tmp_path):
        path = self._write_txt(tmp_path, "Hello world.")
        doc = DocumentParser().parse(path)
        assert doc.filename == "test.txt"

    def test_parse_txt_content_hash_is_sha256(self, tmp_path):
        content = "Hello world."
        path = self._write_txt(tmp_path, content)
        doc = DocumentParser().parse(path)
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert doc.content_hash == expected

    def test_parse_txt_has_at_least_one_section(self, tmp_path):
        path = self._write_txt(tmp_path, "Hello world.")
        doc = DocumentParser().parse(path)
        assert len(doc.sections) >= 1

    def test_parse_txt_sections_are_frozen(self, tmp_path):
        path = self._write_txt(tmp_path, "Hello world.")
        doc = DocumentParser().parse(path)
        for section in doc.sections:
            assert isinstance(section, Section)

    def test_parse_txt_section_ids_use_s_prefix(self, tmp_path):
        path = self._write_txt(tmp_path, "Hello world.\n\nAnother paragraph.")
        doc = DocumentParser().parse(path)
        for section in doc.sections:
            assert section.section_id.startswith("s")

    def test_parse_txt_detects_heading(self, tmp_path):
        # Single newline: heading + body in one paragraph block
        content = "Introduction\nThis is the body text of the section."
        path = self._write_txt(tmp_path, content)
        doc = DocumentParser().parse(path)
        headings = [s.heading for s in doc.sections if s.heading is not None]
        assert len(headings) >= 1

    def test_parse_txt_accepts_external_doc_id(self, tmp_path):
        path = self._write_txt(tmp_path, "Hello.")
        doc = DocumentParser().parse(path, doc_id="fixed-id-123")
        assert doc.doc_id == "fixed-id-123"

    def test_parse_txt_generates_uuid_doc_id_when_none(self, tmp_path):
        path = self._write_txt(tmp_path, "Hello.")
        doc = DocumentParser().parse(path)
        assert len(doc.doc_id) == 36  # UUID4 string length


class TestDocumentParserPdf:
    def test_parse_pdf_file_type_is_pdf(self, tmp_path):
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Page content."
        fake_reader = MagicMock()
        fake_reader.pages = [fake_page]

        with patch("app.knowledge_base.parser.PdfReader", return_value=fake_reader):
            p = tmp_path / "test.pdf"
            p.write_bytes(b"%PDF-1.4")
            doc = DocumentParser().parse(p)

        assert doc.file_type == "pdf"

    def test_parse_pdf_creates_page_sections(self, tmp_path):
        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "Page one."
        pages[1].extract_text.return_value = "Page two."
        fake_reader = MagicMock()
        fake_reader.pages = pages

        with patch("app.knowledge_base.parser.PdfReader", return_value=fake_reader):
            p = tmp_path / "test.pdf"
            p.write_bytes(b"%PDF-1.4")
            doc = DocumentParser().parse(p)

        assert len(doc.sections) == 2

    def test_parse_pdf_section_ids_use_p_prefix(self, tmp_path):
        fake_page = MagicMock()
        fake_page.extract_text.return_value = "Content."
        fake_reader = MagicMock()
        fake_reader.pages = [fake_page]

        with patch("app.knowledge_base.parser.PdfReader", return_value=fake_reader):
            p = tmp_path / "test.pdf"
            p.write_bytes(b"%PDF-1.4")
            doc = DocumentParser().parse(p)

        assert all(s.section_id.startswith("p") for s in doc.sections)

    def test_parse_pdf_empty_pages_skipped(self, tmp_path):
        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = ""
        pages[1].extract_text.return_value = "Actual content."
        fake_reader = MagicMock()
        fake_reader.pages = pages

        with patch("app.knowledge_base.parser.PdfReader", return_value=fake_reader):
            p = tmp_path / "test.pdf"
            p.write_bytes(b"%PDF-1.4")
            doc = DocumentParser().parse(p)

        assert len(doc.sections) == 1


class TestDocumentParserDispatch:
    def test_unsupported_extension_raises_value_error(self, tmp_path):
        p = tmp_path / "file.docx"
        p.write_bytes(b"data")
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentParser().parse(p)

    def test_document_is_frozen(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text("Hello.", encoding="utf-8")
        doc = DocumentParser().parse(p)
        with pytest.raises(Exception):
            doc.doc_id = "changed"  # type: ignore[misc]
