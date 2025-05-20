DROP INDEX IF EXISTS "ix_message_partitions_by_message_and_partition";
DROP INDEX IF EXISTS "ix_message_partitions_by_agent_message_and_session";
DROP INDEX IF EXISTS "ix_message_partitions_by_embedding";
DROP TABLE IF EXISTS "message_partitions";

DROP INDEX IF EXISTS "ix_messages_by_agent_and_session";
DROP INDEX IF EXISTS "ix_messages_by_created_at";
DROP INDEX IF EXISTS "ix_messages_by_agent_session_deleted_and_created";
DROP TABLE IF EXISTS "messages";

DROP TABLE IF EXISTS "sessions";

DROP TYPE IF EXISTS "message_author_type";

DROP EXTENSION IF EXISTS vector;

