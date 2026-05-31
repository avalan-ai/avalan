from importlib import import_module
from typing import Any, cast

revision = "20260530_0001"
down_revision = None
branch_labels = ("memory",)
depends_on = None

MEMORY_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
CREATE EXTENSION IF NOT EXISTS vector;
""",
    """
CREATE EXTENSION IF NOT EXISTS ltree;
""",
    """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_type
        WHERE typname = 'message_author_type'
        AND typnamespace = current_schema()::regnamespace
    ) THEN
        CREATE TYPE "message_author_type" AS ENUM (
            'assistant',
            'system',
            'tool',
            'user'
        );
    END IF;
END
$$;
""",
    """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_type
        WHERE typname = 'memory_types'
        AND typnamespace = current_schema()::regnamespace
    ) THEN
        CREATE TYPE "memory_types" AS ENUM (
            'code',
            'file',
            'raw',
            'url'
        );
    END IF;
END
$$;
""",
    """
CREATE TABLE IF NOT EXISTS "sessions" (
    "id" UUID NOT NULL,
    "agent_id" UUID NOT NULL,
    "participant_id" UUID NOT NULL,
    "messages" INT NOT NULL CHECK ("messages" >= 0),
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),

    PRIMARY KEY("id")
);
""",
    """
CREATE TABLE IF NOT EXISTS "messages" (
    "id" UUID NOT NULL,
    "agent_id" UUID NOT NULL,
    "model_id" TEXT NOT NULL,
    "session_id" UUID DEFAULT NULL,
    "author" message_author_type NOT NULL,
    "data" TEXT NOT NULL,
    "partitions" INT NOT NULL CHECK ("partitions" > 0),
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    "is_deleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "deleted_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY("id"),
    CONSTRAINT "fk_messages__sessions"
        FOREIGN KEY("session_id")
        REFERENCES "sessions"("id")
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_messages_by_agent_session_deleted_and_created"
    ON "messages"
    USING BTREE("agent_id", "session_id", "is_deleted", "created_at" DESC);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_messages_by_agent_and_session"
    ON "messages"
    USING BTREE("agent_id", "session_id");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_messages_by_created_at"
    ON "messages"
    USING BTREE("created_at" DESC);
""",
    """
CREATE TABLE IF NOT EXISTS "message_partitions" (
    "id" SERIAL,
    "agent_id" UUID NOT NULL,
    "session_id" UUID DEFAULT NULL,
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
""",
    """
CREATE INDEX IF NOT EXISTS "ix_message_partitions_by_embedding"
ON "message_partitions" USING IVFFLAT ("embedding" VECTOR_L2_OPS)
WITH (LISTS=100);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_message_partitions_by_agent_message_and_session"
    ON "message_partitions"
    USING BTREE("agent_id", "message_id", "session_id");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_message_partitions_by_message_and_partition"
    ON "message_partitions"
    USING BTREE("message_id", "partition" ASC);
""",
    """
CREATE TABLE IF NOT EXISTS "memories" (
    "id" UUID NOT NULL,
    "model_id" TEXT NOT NULL,
    "participant_id" UUID NOT NULL,
    "memory_type" memory_types NOT NULL,
    "namespace" TEXT NOT NULL,
    "namespace_tree" LTREE GENERATED ALWAYS AS (
        text2ltree("namespace")
    ) STORED,
    "identifier" TEXT NOT NULL,
    "data" TEXT NOT NULL,
    "symbols" JSONB DEFAULT NULL,
    "partitions" INT NOT NULL CHECK ("partitions" > 0),
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    "title" TEXT DEFAULT NULL,
    "description" TEXT DEFAULT NULL,
    "is_deleted" BOOLEAN NOT NULL DEFAULT FALSE,
    "deleted_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY("id")
);
""",
    """
CREATE INDEX IF NOT EXISTS
    "ix_memories_by_type_participant_namespace_deleted_created"
    ON "memories"
    USING BTREE(
        "memory_type",
        "participant_id",
        "namespace_tree",
        "is_deleted",
        "created_at" DESC
    );
""",
    """
CREATE INDEX IF NOT EXISTS "ix_memories_by_type_participant_and_namespace"
    ON "memories"
    USING BTREE("memory_type", "participant_id", "namespace_tree");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_memories_namespace_tree_gist"
    ON "memories"
    USING GIST("namespace_tree");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_memories_by_created_at"
    ON "memories"
    USING BTREE("created_at" DESC);
""",
    """
CREATE TABLE IF NOT EXISTS "memory_partitions" (
    "id" SERIAL,
    "participant_id" UUID NOT NULL,
    "memory_id" UUID NOT NULL,
    "partition" BIGINT NOT NULL CHECK ("partition" > 0),
    "data" TEXT NOT NULL,
    "embedding" VECTOR(384) NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),

    PRIMARY KEY("id"),
    CONSTRAINT "fk_memory_partitions__memories"
        FOREIGN KEY("memory_id")
        REFERENCES "memories"("id")
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_memory_partitions_by_embedding"
ON "memory_partitions" USING IVFFLAT ("embedding" VECTOR_L2_OPS)
WITH (LISTS=100);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_memory_partitions_by_participant_and_memory"
    ON "memory_partitions"
    USING BTREE("participant_id", "memory_id");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_memory_partitions_by_memory_and_partition"
    ON "memory_partitions"
    USING BTREE("memory_id", "partition" ASC);
""",
)


def upgrade() -> None:
    _execute_statements(MEMORY_SCHEMA_STATEMENTS)


def downgrade() -> None:
    raise NotImplementedError("memory PostgreSQL migrations are forward-only")


def _execute_statements(statements: tuple[str, ...]) -> None:
    alembic = cast(Any, import_module("alembic"))
    bind = alembic.op.get_bind()
    for statement in statements:
        bind.exec_driver_sql(statement)
