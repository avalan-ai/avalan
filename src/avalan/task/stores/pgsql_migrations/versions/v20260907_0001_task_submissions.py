"""Retain immutable task submission recovery evidence."""

from alembic import op

revision = "20260907_0001_task_submissions"
down_revision = "20260828_0001_patch_coordination"
branch_labels = None
depends_on = None

TASK_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
CREATE TABLE IF NOT EXISTS "task_submissions" (
    "owner_scope" TEXT NOT NULL,
    "submission_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "payload" JSONB NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY ("owner_scope", "submission_id"),
    CONSTRAINT "fk_task_submissions_run"
        FOREIGN KEY ("run_id") REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "ck_task_submissions_identity"
        CHECK (
            LENGTH(BTRIM("owner_scope")) > 0
            AND LENGTH(BTRIM("submission_id")) > 0
        ),
    CONSTRAINT "ck_task_submissions_envelope"
        CHECK (COALESCE((
            jsonb_typeof("payload") = 'object'
            AND "payload"->>'format' = 'avalan.task.submission'
            AND "payload"->'version' = '1'::jsonb
            AND jsonb_typeof("payload"->'payload') = 'object'
            AND "payload"->'payload'->>'owner_scope' = "owner_scope"
            AND "payload"->'payload'->>'submission_id' = "submission_id"
            AND "payload"->'payload'->>'run_id' = "run_id"
        ), FALSE))
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_submissions_run"
    ON "task_submissions" ("run_id");
""",
)


def upgrade() -> None:
    """Install durable evidence without reservation-based expiry."""
    bind = op.get_bind()
    for statement in TASK_SCHEMA_STATEMENTS:
        bind.exec_driver_sql(statement)


def downgrade() -> None:
    """Reject removal of durable submission recovery evidence."""
    raise NotImplementedError("task PostgreSQL migrations are forward-only")
