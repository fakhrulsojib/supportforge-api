-- Phase 6: Add moderation audit columns to messages table
-- Adds moderation_reason and moderation_matched_term for admin audit trail,
-- and an index on validation_status for efficient admin queries.
--
-- Run this migration against the PostgreSQL database:
--   psql -d supportforge -f migrations/006_add_moderation_audit_columns.sql

ALTER TABLE messages
    ADD COLUMN IF NOT EXISTS moderation_reason VARCHAR(100) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS moderation_matched_term VARCHAR(200) NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS ix_messages_validation_status
    ON messages (validation_status);
