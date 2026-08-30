"""Persist one workspace-wide durable patch coordination admission."""

from collections.abc import Iterable

from alembic import op

revision = "20260828_0001_patch_coordination"
down_revision = "20260811_0002_patch_worker"
branch_labels = None
depends_on = None

TASK_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
CREATE TABLE IF NOT EXISTS "patch_durable_workspace_coordination" (
    "workspace_id" TEXT NOT NULL,
    "domain_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "tenant_id" TEXT NOT NULL,
    "principal_id" TEXT NOT NULL,
    "execution_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "task_id" TEXT NOT NULL,
    "agent_id" TEXT NOT NULL,
    "route_id" TEXT NOT NULL,
    "context_id" TEXT NOT NULL,
    "paths_digest" TEXT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("workspace_id"),
    CONSTRAINT "uq_patch_durable_workspace_coordination_request"
        UNIQUE ("request_id"),
    CONSTRAINT "fk_patch_durable_workspace_coordination_request"
        FOREIGN KEY ("request_id")
        REFERENCES "patch_durable_requests" ("request_id"),
    CONSTRAINT "ck_patch_durable_workspace_coordination_identifiers"
        CHECK (
            LENGTH(BTRIM("workspace_id")) > 0
            AND LENGTH(BTRIM("domain_id")) > 0
            AND LENGTH(BTRIM("request_id")) > 0
            AND LENGTH(BTRIM("tenant_id")) > 0
            AND LENGTH(BTRIM("principal_id")) > 0
            AND LENGTH(BTRIM("execution_id")) > 0
            AND LENGTH(BTRIM("run_id")) > 0
            AND LENGTH(BTRIM("session_id")) > 0
            AND LENGTH(BTRIM("task_id")) > 0
            AND LENGTH(BTRIM("agent_id")) > 0
            AND LENGTH(BTRIM("route_id")) > 0
            AND LENGTH(BTRIM("context_id")) > 0
        ),
    CONSTRAINT "ck_patch_durable_workspace_coordination_digest"
        CHECK ("paths_digest" ~ '^[0-9a-f]{64}$')
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_patch_durable_workspace_coordination_domain"
    ON "patch_durable_workspace_coordination" (
        "domain_id", "workspace_id"
    );
""",
)


def upgrade() -> None:
    """Apply one durable workspace admission table."""
    _execute_all(TASK_SCHEMA_STATEMENTS)


def downgrade() -> None:
    """Reject reverse migration of durable coordination evidence."""
    raise NotImplementedError("task PostgreSQL migrations are forward-only")


def _execute_all(statements: Iterable[str]) -> None:
    """Execute each reviewed schema statement through Alembic's connection."""
    bind = op.get_bind()
    for statement in statements:
        bind.exec_driver_sql(statement)
