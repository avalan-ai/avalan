CREATE EXTENSION IF NOT EXISTS vector;

CREATE TYPE "message_author_type" AS ENUM (
    'assistant',
    'system',
    'tool',
    'user'
);

CREATE TABLE IF NOT EXISTS "sessions" (
    "id" UUID NOT NULL,
    "agent_id" UUID NOT NULL,
    "participant_id" UUID NOT NULL,
    "messages" INT NOT NULL CHECK ("messages" >= 0),
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),

    PRIMARY KEY("id")
);

--
-- Each session can have multiple messages
--
CREATE TABLE IF NOT EXISTS "messages" (
    "id" UUID NOT NULL,
    "agent_id" UUID NOT NULL,
    "model_id" TEXT NOT NULL,
    "session_id" UUID default NULL,
    "author" message_author_type NOT NULL,
    "data" TEXT NOT NULL,
    "partitions" INT NOT NULL CHECK ("partitions" > 0),
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    "is_deleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "deleted_at" TIMESTAMP WITH TIME ZONE default NULL,

    PRIMARY KEY("id"),
    CONSTRAINT "fk_messages__sessions"
        FOREIGN KEY("session_id")
        REFERENCES "sessions"("id")
);

CREATE INDEX IF NOT EXISTS "ix_messages_by_agent_session_deleted_and_created"
    ON "messages"
    USING BTREE("agent_id", "session_id", "is_deleted", "created_at" DESC);

CREATE INDEX IF NOT EXISTS "ix_messages_by_agent_and_session"
    ON "messages"
    USING BTREE("agent_id", "session_id");

CREATE INDEX IF NOT EXISTS "ix_messages_by_created_at"
    ON "messages"
    USING BTREE("created_at" DESC);

--
-- Each message is split into multiple embedding partitions
--
CREATE TABLE IF NOT EXISTS "message_partitions" (
    "id" SERIAL,
    "agent_id" UUID NOT NULL,
    "session_id" UUID default NULL,
    "message_id" UUID NOT NULL,
    "partition" BIGINT NOT NULL CHECK ("partition" > 0),
    "data" TEXT NOT NULL,
    "embedding" VECTOR(384) NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),

    PRIMARY KEY("id"),
    CONSTRAINT "fk_message_partitions__sessions"
        FOREIGN KEY("session_id")
        REFERENCES "sessions"("id"),
    CONSTRAINT "fk_message_partitions__messages"
        FOREIGN KEY("message_id")
        REFERENCES "messages"("id")
);

-- InVerted File with FLAT quantization index (IVFFLAT)
-- with L2 (Euclidean) distance (VECTOR_L2_OPS)
-- partitioning vector space into 100 clusters (WITH LISTS=100)

CREATE INDEX IF NOT EXISTS "ix_message_partitions_by_embedding"
ON "message_partitions" USING IVFFLAT ("embedding" VECTOR_L2_OPS)
WITH (LISTS=100);

CREATE INDEX IF NOT EXISTS "ix_message_partitions_by_agent_message_and_session"
    ON "message_partitions"
    USING BTREE("agent_id", "message_id", "session_id");

CREATE INDEX IF NOT EXISTS "ix_message_partitions_by_message_and_partition"
    ON "message_partitions"
    USING BTREE("message_id", "partition" ASC);

