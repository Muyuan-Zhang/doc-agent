"""Unit tests for DocumentChunker."""
import hashlib

import pytest

from app.knowledge_base.chunker import DocumentChunker
from app.knowledge_base.parser import ParsedDocument, Section
from app.models.chunk import ChunkSchema


def _make_doc(content: str, doc_id: str = "doc-1") -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        filename="test.txt",
        file_type="txt",
        sections=(Section(section_id="s0000", heading=None, content=content),),
        content_hash="abc",
    )


class TestDocumentChunkerBasic:
    def test_returns_list_of_chunk_schemas(self):
        doc = _make_doc("Hello world.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert all(isinstance(c, ChunkSchema) for c in chunks)

    def test_empty_section_returns_no_chunks(self):
        doc = _make_doc("")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert chunks == []

    def test_whitespace_only_returns_no_chunks(self):
        doc = _make_doc("   \n  ")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert chunks == []

    def test_short_text_produces_one_chunk(self):
        doc = _make_doc("Short text.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert len(chunks) == 1

    def test_chunk_contains_original_text(self):
        doc = _make_doc("Hello world.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert "Hello world" in chunks[0].content

    def test_long_text_produces_multiple_chunks(self):
        # 600-char text, chunk_size=100
        text = "This sentence is roughly twenty chars. " * 20
        doc = _make_doc(text)
        chunks = DocumentChunker(chunk_size=100, chunk_overlap=10).chunk(doc, "v1")
        assert len(chunks) > 1

    def test_each_chunk_is_at_most_chunk_size_plus_tolerance(self):
        text = "word " * 300
        doc = _make_doc(text)
        chunks = DocumentChunker(chunk_size=50, chunk_overlap=5).chunk(doc, "v1")
        for c in chunks:
            assert len(c.content) <= 120  # allow reasonable boundary overrun


class TestDocumentChunkerFields:
    def test_doc_id_matches(self):
        doc = _make_doc("Hello.", doc_id="my-doc")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert all(c.doc_id == "my-doc" for c in chunks)

    def test_section_id_matches(self):
        doc = _make_doc("Hello.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert all(c.section_id == "s0000" for c in chunks)

    def test_version_matches(self):
        doc = _make_doc("Hello.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v2")
        assert all(c.version == "v2" for c in chunks)

    def test_embedding_is_none(self):
        doc = _make_doc("Hello.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert all(c.embedding is None for c in chunks)

    def test_parent_chunk_id_is_none(self):
        doc = _make_doc("Hello.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        assert all(c.parent_chunk_id is None for c in chunks)

    def test_chunk_indices_are_sequential(self):
        text = "word " * 300
        doc = _make_doc(text)
        chunks = DocumentChunker(chunk_size=50, chunk_overlap=5).chunk(doc, "v1")
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    def test_content_hash_is_sha256_of_content(self):
        doc = _make_doc("Hello world.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        for c in chunks:
            expected = hashlib.sha256(c.content.encode()).hexdigest()
            assert c.content_hash == expected

    def test_chunks_are_frozen(self):
        doc = _make_doc("Hello world.")
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        with pytest.raises(Exception):
            chunks[0].content = "changed"  # type: ignore[misc]


class TestDocumentChunkerMultipleSections:
    def test_chunks_from_multiple_sections_all_returned(self):
        doc = ParsedDocument(
            doc_id="d1",
            filename="f.txt",
            file_type="txt",
            sections=(
                Section(section_id="s0000", heading=None, content="Section one content."),
                Section(section_id="s0001", heading=None, content="Section two content."),
            ),
            content_hash="x",
        )
        chunks = DocumentChunker(chunk_size=512, chunk_overlap=64).chunk(doc, "v1")
        section_ids = {c.section_id for c in chunks}
        assert "s0000" in section_ids
        assert "s0001" in section_ids
