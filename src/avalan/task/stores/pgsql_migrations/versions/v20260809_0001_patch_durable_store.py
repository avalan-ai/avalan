"""Add isolated durable semantic storage for dormant patch mutation."""

from collections.abc import Iterable

from alembic import op

revision = "20260809_0001_patch_durable"
down_revision = "20260801_0003"
branch_labels = None
depends_on = None

TASK_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
CREATE TABLE IF NOT EXISTS "patch_durable_domains" (
    "domain_id" TEXT NOT NULL,
    "current_fence" BIGINT NOT NULL DEFAULT 0,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("domain_id"),
    CONSTRAINT "ck_patch_durable_domains_identifier"
        CHECK (LENGTH(BTRIM("domain_id")) > 0),
    CONSTRAINT "ck_patch_durable_domains_current_fence"
        CHECK ("current_fence" >= 0)
);
""",
    """
CREATE TABLE IF NOT EXISTS "patch_durable_requests" (
    "request_id" TEXT NOT NULL,
    "tenant_id" TEXT NOT NULL,
    "principal_id" TEXT NOT NULL,
    "execution_id" TEXT NOT NULL,
    "route_id" TEXT NOT NULL,
    "retransmission_key" TEXT NOT NULL,
    "canonical_digest" TEXT NOT NULL,
    "plan_payload" BYTEA DEFAULT NULL,
    "lifecycle" TEXT NOT NULL DEFAULT 'received',
    "owner_id" TEXT DEFAULT NULL,
    "domain_id" TEXT DEFAULT NULL,
    "fence" BIGINT NOT NULL DEFAULT 0,
    "lease_expires_at" BIGINT DEFAULT NULL,
    "journal_revision" BIGINT NOT NULL DEFAULT 0,
    "pending_operation_id" TEXT DEFAULT NULL,
    "pending_correlation_id" TEXT DEFAULT NULL,
    "pending_next_check_after" BIGINT DEFAULT NULL,
    "pending_event_cursor" BIGINT DEFAULT NULL,
    "cancellation_requested" BOOLEAN NOT NULL DEFAULT FALSE,
    "event_cursor" BIGINT NOT NULL DEFAULT 0,
    "terminal_result" BYTEA DEFAULT NULL,
    "terminal_correlation_id" TEXT DEFAULT NULL,
    "terminal_pending_operation_id" TEXT DEFAULT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("request_id"),
    CONSTRAINT "uq_patch_durable_requests_retransmission"
        UNIQUE (
            "tenant_id", "principal_id", "execution_id", "route_id",
            "retransmission_key"
        ),
    CONSTRAINT "ck_patch_durable_requests_identifiers"
        CHECK (
            LENGTH(BTRIM("request_id")) > 0
            AND LENGTH(BTRIM("tenant_id")) > 0
            AND LENGTH(BTRIM("principal_id")) > 0
            AND LENGTH(BTRIM("execution_id")) > 0
            AND LENGTH(BTRIM("route_id")) > 0
            AND LENGTH(BTRIM("retransmission_key")) > 0
        ),
    CONSTRAINT "ck_patch_durable_requests_digest"
        CHECK ("canonical_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_patch_durable_requests_lifecycle"
        CHECK (
            "lifecycle" IN (
                'received', 'planned', 'commit_started',
                'settlement_pending', 'request_completed'
            )
        ),
    CONSTRAINT "ck_patch_durable_requests_fence"
        CHECK (
            "fence" >= 0
            AND (
                ("owner_id" IS NULL AND "domain_id" IS NULL
                 AND "fence" = 0 AND "lease_expires_at" IS NULL)
                OR
                ("owner_id" IS NOT NULL AND "domain_id" IS NOT NULL
                 AND "fence" > 0 AND "lease_expires_at" IS NOT NULL)
            )
        ),
    CONSTRAINT "ck_patch_durable_requests_journal_revision"
        CHECK ("journal_revision" >= 0),
    CONSTRAINT "ck_patch_durable_requests_event_cursor"
        CHECK ("event_cursor" >= 0),
    CONSTRAINT "ck_patch_durable_requests_pending_shape"
        CHECK (
            (
                "pending_operation_id" IS NULL
                AND "pending_correlation_id" IS NULL
                AND "pending_next_check_after" IS NULL
                AND "pending_event_cursor" IS NULL
            )
            OR
            (
                "pending_operation_id" IS NOT NULL
                AND "pending_correlation_id" IS NOT NULL
                AND "pending_next_check_after" > 0
                AND "pending_event_cursor" > 0
                AND "lifecycle" = 'settlement_pending'
            )
        ),
    CONSTRAINT "ck_patch_durable_requests_terminal_shape"
        CHECK (
            ("lifecycle" = 'request_completed'
             AND "terminal_result" IS NOT NULL
             AND "terminal_correlation_id" IS NOT NULL)
            OR
            ("lifecycle" <> 'request_completed' AND "terminal_result" IS NULL
             AND "terminal_correlation_id" IS NULL
             AND "terminal_pending_operation_id" IS NULL)
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "patch_durable_grant_consumptions" (
    "grant_id" TEXT NOT NULL,
    "approval_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "consumed_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("grant_id"),
    CONSTRAINT "uq_patch_durable_grant_consumptions_approval"
        UNIQUE ("approval_id"),
    CONSTRAINT "fk_patch_durable_grant_consumptions_request"
        FOREIGN KEY ("request_id")
        REFERENCES "patch_durable_requests" ("request_id"),
    CONSTRAINT "ck_patch_durable_grant_consumptions_identifiers"
        CHECK (
            LENGTH(BTRIM("grant_id")) > 0
            AND LENGTH(BTRIM("approval_id")) > 0
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "patch_durable_step_journal" (
    "request_id" TEXT NOT NULL,
    "revision" BIGINT NOT NULL,
    "step_id" TEXT NOT NULL,
    "lineage_id" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("request_id", "revision"),
    CONSTRAINT "fk_patch_durable_step_journal_request"
        FOREIGN KEY ("request_id")
        REFERENCES "patch_durable_requests" ("request_id"),
    CONSTRAINT "ck_patch_durable_step_journal_revision"
        CHECK ("revision" > 0),
    CONSTRAINT "ck_patch_durable_step_journal_state"
        CHECK (
            "state" IN ('planned', 'committed', 'not_committed', 'unknown')
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "patch_durable_artifact_journal" (
    "request_id" TEXT NOT NULL,
    "revision" BIGINT NOT NULL,
    "artifact_id" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("request_id", "revision"),
    CONSTRAINT "fk_patch_durable_artifact_journal_request"
        FOREIGN KEY ("request_id")
        REFERENCES "patch_durable_requests" ("request_id"),
    CONSTRAINT "ck_patch_durable_artifact_journal_revision"
        CHECK ("revision" > 0),
    CONSTRAINT "ck_patch_durable_artifact_journal_state"
        CHECK (
            "state" IN (
                'intended', 'not_created', 'present', 'removed', 'leaked',
                'unknown'
            )
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "patch_durable_outbox" (
    "event_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "sequence" BIGINT NOT NULL,
    "lifecycle" TEXT NOT NULL,
    "correlation_id" TEXT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("event_id"),
    CONSTRAINT "uq_patch_durable_outbox_request_sequence"
        UNIQUE ("request_id", "sequence"),
    CONSTRAINT "uq_patch_durable_outbox_terminal"
        UNIQUE ("request_id", "lifecycle")
        DEFERRABLE INITIALLY IMMEDIATE,
    CONSTRAINT "fk_patch_durable_outbox_request"
        FOREIGN KEY ("request_id")
        REFERENCES "patch_durable_requests" ("request_id"),
    CONSTRAINT "ck_patch_durable_outbox_sequence" CHECK ("sequence" > 0),
    CONSTRAINT "ck_patch_durable_outbox_lifecycle"
        CHECK ("lifecycle" IN ('settlement_pending', 'request_completed'))
);
""",
    """
CREATE TABLE IF NOT EXISTS "patch_durable_retention" (
    "retention_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "key_id" TEXT NOT NULL,
    "ciphertext" BYTEA NOT NULL,
    "ciphertext_digest" TEXT NOT NULL,
    "expires_at" BIGINT NOT NULL,
    "delete_on_terminal" BOOLEAN NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("retention_id"),
    CONSTRAINT "fk_patch_durable_retention_request"
        FOREIGN KEY ("request_id")
        REFERENCES "patch_durable_requests" ("request_id"),
    CONSTRAINT "ck_patch_durable_retention_kind"
        CHECK (
            "kind" IN ('sealed_plan', 'review_artifact', 'private_staging')
        ),
    CONSTRAINT "ck_patch_durable_retention_ciphertext"
        CHECK (OCTET_LENGTH("ciphertext") <= 1048576),
    CONSTRAINT "ck_patch_durable_retention_digest"
        CHECK ("ciphertext_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_patch_durable_retention_expiry" CHECK ("expires_at" > 0)
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_patch_durable_requests_recovery"
    ON "patch_durable_requests" (
        "domain_id", "lease_expires_at", "request_id"
    )
    WHERE "lifecycle" IN ('commit_started', 'settlement_pending');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_patch_durable_outbox_recovery"
    ON "patch_durable_outbox" ("request_id", "sequence");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_patch_durable_retention_expiry"
    ON "patch_durable_retention" ("expires_at", "retention_id");
""",
)


def upgrade() -> None:
    """Apply isolated durable patch semantic storage."""
    _execute_all(TASK_SCHEMA_STATEMENTS)


def downgrade() -> None:
    """Reject reverse migration of durable patch evidence."""
    raise NotImplementedError("task PostgreSQL migrations are forward-only")


def _execute_all(statements: Iterable[str]) -> None:
    """Execute each reviewed schema statement through Alembic's connection."""
    bind = op.get_bind()
    for statement in statements:
        bind.exec_driver_sql(statement)
