-- M1 knowledge base schema
-- Run once against the target PostgreSQL database.

CREATE TABLE IF NOT EXISTS documents (
    doc_id       VARCHAR(36)  PRIMARY KEY,
    filename     VARCHAR(512) NOT NULL,
    file_type    VARCHAR(10)  NOT NULL CHECK (file_type IN ('pdf','txt')),
    status       VARCHAR(50)  NOT NULL DEFAULT 'pending',
    chunk_count  INT          NOT NULL DEFAULT 0,
    version      VARCHAR(100) NOT NULL,
    content_hash VARCHAR(64),
    created_at   TIMESTAMPTZ  DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks_metadata (
    chunk_id        VARCHAR(255) PRIMARY KEY,
    doc_id          VARCHAR(36)  NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    section_id      VARCHAR(255),
    chunk_index     INT          NOT NULL,
    parent_chunk_id VARCHAR(255),
    content_hash    VARCHAR(64)  NOT NULL UNIQUE,
    version         VARCHAR(100) NOT NULL,
    content         TEXT         NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks_metadata(doc_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
