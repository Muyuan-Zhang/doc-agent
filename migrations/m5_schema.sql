-- M5 memory schema
-- Run once against the target PostgreSQL database.

CREATE TABLE IF NOT EXISTS memory_summaries (
    summary_id   VARCHAR(36)  PRIMARY KEY,
    user_id      VARCHAR(36)  NOT NULL,
    session_id   VARCHAR(36)  NOT NULL,
    summary_text TEXT         NOT NULL,
    content_hash VARCHAR(64)  NOT NULL,
    created_at   TIMESTAMPTZ  DEFAULT NOW(),
    updated_at   TIMESTAMPTZ  DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_summaries_user_session
    ON memory_summaries(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_memory_summaries_user_id
    ON memory_summaries(user_id);

CREATE TABLE IF NOT EXISTS memory_static_facts (
    fact_id      VARCHAR(36)  PRIMARY KEY,
    user_id      VARCHAR(36)  NOT NULL,
    content      TEXT         NOT NULL,
    content_hash VARCHAR(64)  NOT NULL UNIQUE,
    created_at   TIMESTAMPTZ  DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_static_facts_user_id
    ON memory_static_facts(user_id);
