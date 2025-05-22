DROP INDEX IF EXISTS "ix_message_partitions_by_message_and_partition";
DROP INDEX IF EXISTS "ix_message_partitions_by_agent_message_and_session";
DROP INDEX IF EXISTS "ix_message_partitions_by_embedding";
DROP TABLE IF EXISTS "message_partitions";

DROP INDEX IF EXISTS "ix_memory_partitions_by_memory_and_partition";
DROP INDEX IF EXISTS "ix_memory_partitions_by_participant_and_memory";
DROP INDEX IF EXISTS "ix_memory_partitions_by_embedding";
DROP TABLE IF EXISTS "memory_partitions";

DROP INDEX IF EXISTS "ix_memories_by_created_at";
DROP INDEX IF EXISTS "ix_memories_namespace_tree_gist";
DROP INDEX IF EXISTS "ix_memories_by_type_participant_and_namespace";
DROP INDEX IF EXISTS "ix_memories_by_type_participant_namespace_deleted_created";
DROP TABLE IF EXISTS "memories";

DROP INDEX IF EXISTS "ix_messages_by_agent_and_session";
DROP INDEX IF EXISTS "ix_messages_by_created_at";
DROP INDEX IF EXISTS "ix_messages_by_agent_session_deleted_and_created";
DROP TABLE IF EXISTS "messages";

DROP TABLE IF EXISTS "sessions";

DROP TYPE IF EXISTS "message_author_type";
DROP TYPE IF EXISTS "memory_types";

DROP EXTENSION IF EXISTS ltree;
DROP EXTENSION IF EXISTS vector;

