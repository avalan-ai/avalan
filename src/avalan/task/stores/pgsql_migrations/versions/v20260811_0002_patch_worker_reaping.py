"""Bind durable patch ownership to write-capable worker reaping."""

from collections.abc import Iterable

from alembic import op

revision = "20260811_0002_patch_worker"
down_revision = "20260809_0001_patch_durable"
branch_labels = None
depends_on = None

TASK_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
ALTER TABLE "patch_durable_requests"
    ADD COLUMN IF NOT EXISTS "worker_binding_digest" TEXT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS "worker_reaped" BOOLEAN NOT NULL DEFAULT FALSE;
""",
    """
ALTER TABLE "patch_durable_requests"
    ADD CONSTRAINT "ck_patch_durable_requests_worker_binding"
    CHECK (
        ("worker_binding_digest" IS NULL)
        OR
        ("worker_binding_digest" ~ '^[0-9a-f]{64}$')
    ) NOT VALID;
""",
    """
ALTER TABLE "patch_durable_requests"
    VALIDATE CONSTRAINT "ck_patch_durable_requests_worker_binding";
""",
    """
CREATE INDEX IF NOT EXISTS "ix_patch_durable_requests_worker_reaping"
    ON "patch_durable_requests" (
        "domain_id", "worker_reaped", "lease_expires_at"
    )
    WHERE "worker_binding_digest" IS NOT NULL
      AND "lifecycle" IN ('commit_started', 'settlement_pending');
""",
)


def upgrade() -> None:
    """Add durable live-worker and reaping evidence."""
    _execute_all(TASK_SCHEMA_STATEMENTS)


def downgrade() -> None:
    """Reject reverse migration of durable worker evidence."""
    raise NotImplementedError("task PostgreSQL migrations are forward-only")


def _execute_all(statements: Iterable[str]) -> None:
    """Execute each reviewed schema statement through Alembic's connection."""
    bind = op.get_bind()
    for statement in statements:
        bind.exec_driver_sql(statement)
