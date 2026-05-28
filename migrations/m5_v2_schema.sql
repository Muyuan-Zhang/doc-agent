-- M5 v2: importance scores, structured facts, session-scoped queries
-- Run after m5_schema.sql.

ALTER TABLE memory_summaries
    ADD COLUMN IF NOT EXISTS importance_score   FLOAT  DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS structured_facts   JSONB  DEFAULT '{}';

ALTER TABLE memory_static_facts
    ADD COLUMN IF NOT EXISTS importance_weight  FLOAT  DEFAULT 1.0;
