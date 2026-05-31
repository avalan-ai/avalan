from importlib import import_module
from typing import Any, cast

revision = "20260530_0002"
down_revision = "20260530_0001"
branch_labels = None
depends_on = None

MEMORY_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
""",
    """
CREATE TABLE IF NOT EXISTS "hyperedges" (
    "id" UUID NOT NULL,
    "relation" TEXT NOT NULL,
    "surface_text" TEXT NOT NULL,
    "embedding" VECTOR(1024) NOT NULL,
    "symbols" JSONB DEFAULT NULL,
    "created_at" TIMESTAMPTZ NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    PRIMARY KEY("id")
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_hyperedges_embedding_ivfflat"
    ON "hyperedges" USING ivfflat ("embedding" vector_cosine_ops)
    WITH (lists = 100);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_hyperedges_by_created_at"
    ON "hyperedges" ("created_at" DESC);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_hyperedges_relation_lc"
    ON "hyperedges" (LOWER("relation"));
""",
    """
CREATE TABLE IF NOT EXISTS "hyperedges_memories" (
    "hyperedge_id" UUID NOT NULL
        REFERENCES "hyperedges"("id") ON DELETE CASCADE,
    "memory_id" UUID NOT NULL
        REFERENCES "memories"("id") ON DELETE CASCADE,
    "char_start" INT,
    "char_end" INT,
    PRIMARY KEY ("hyperedge_id", "memory_id")
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_hyperedges_memories_by_memory"
    ON "hyperedges_memories" ("memory_id");
""",
    """
CREATE OR REPLACE VIEW "v_live_hyperedges" AS
SELECT
  h."id" AS hyperedge_id,
  h."relation",
  h."surface_text",
  h."embedding",
  h."symbols" AS hyperedge_symbols,
  hm."memory_id",
  m."participant_id",
  m."namespace",
  m."namespace_tree",
  m."created_at" AS memory_created_at
FROM "hyperedges" h
JOIN "hyperedges_memories" hm ON hm."hyperedge_id" = h."id"
JOIN "memories" m ON m."id" = hm."memory_id"
WHERE m."is_deleted" = FALSE;
""",
    """
CREATE TABLE IF NOT EXISTS "entities" (
    "id" UUID NOT NULL,
    "name" TEXT NOT NULL,
    "name_lc" TEXT GENERATED ALWAYS AS (LOWER("name")) STORED,
    "type" TEXT,
    "embedding" VECTOR(1024) NOT NULL,
    "symbols" JSONB DEFAULT NULL,
    "participant_id" UUID,
    "namespace" TEXT,
    "namespace_tree" LTREE GENERATED ALWAYS AS (
        text2ltree("namespace")
    ) STORED,
    "created_at" TIMESTAMPTZ NOT NULL
                 DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC'),
    PRIMARY KEY("id")
);
""",
    """
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_entities_scope_name'
        AND conrelid = '"entities"'::regclass
    ) THEN
        ALTER TABLE "entities"
            ADD CONSTRAINT "uq_entities_scope_name"
            UNIQUE ("participant_id", "namespace_tree", "name_lc");
    END IF;
END
$$;
""",
    """
CREATE INDEX IF NOT EXISTS "ix_entities_embedding_ivfflat"
    ON "entities" USING ivfflat ("embedding" vector_cosine_ops)
    WITH (lists = 100);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_entities_name_trgm"
    ON "entities" USING GIN ("name" gin_trgm_ops);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_entities_scope"
    ON "entities" ("participant_id", "namespace_tree");
""",
    """
CREATE TABLE IF NOT EXISTS "hyperedge_entities" (
    "hyperedge_id" UUID NOT NULL
        REFERENCES "hyperedges"("id") ON DELETE CASCADE,
    "entity_id" UUID NOT NULL
        REFERENCES "entities"("id") ON DELETE CASCADE,
    "role_idx" INT NOT NULL CHECK ("role_idx" >= 1),
    "role_label" TEXT,
    PRIMARY KEY ("hyperedge_id", "role_idx")
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_hyperedge_entities_by_entity"
    ON "hyperedge_entities" ("entity_id");
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
