import hashlib

from app.knowledge_base.parser import ParsedDocument, Section
from app.models.chunk import ChunkSchema

_SENTENCE_ENDS = (".", "?", "!", "\n")


class DocumentChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, doc: ParsedDocument, version: str) -> list[ChunkSchema]:
        chunks: list[ChunkSchema] = []
        for section in doc.sections:
            chunks.extend(self._chunk_section(doc.doc_id, section, version))
        return chunks

    def _chunk_section(self, doc_id: str, section: Section, version: str) -> list[ChunkSchema]:
        text = section.content.strip()
        if not text:
            return []

        size = self._chunk_size
        overlap = self._chunk_overlap
        chunks: list[ChunkSchema] = []
        chunk_index = 0
        pos = 0

        while pos < len(text):
            end = min(pos + size, len(text))

            if end < len(text):
                boundary = -1
                for ch in _SENTENCE_ENDS:
                    idx = text.rfind(ch + " ", pos + size // 2, end)
                    if idx > boundary:
                        boundary = idx
                if boundary > pos + size // 2:
                    end = boundary + 1

            content = text[pos:end].strip()
            if content:
                chunks.append(ChunkSchema(
                    doc_id=doc_id,
                    section_id=section.section_id,
                    chunk_index=chunk_index,
                    parent_chunk_id=None,
                    content_hash=hashlib.sha256(content.encode()).hexdigest(),
                    version=version,
                    content=content,
                    embedding=None,
                ))
                chunk_index += 1

            if end >= len(text):
                break

            next_pos = end - overlap
            pos = next_pos if next_pos > pos else pos + 1

        return chunks
