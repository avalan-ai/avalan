"""Add durable interaction checkpoints and task suspension boundaries."""

from collections.abc import Iterable
from importlib import import_module
from typing import Any, cast

revision = "20260723_0002"
down_revision = "20260530_0001"
branch_labels = None
depends_on = None

TASK_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
CREATE TABLE IF NOT EXISTS "interaction_store_metadata" (
    "singleton_id" SMALLINT NOT NULL DEFAULT 1,
    "store_generation" BIGINT NOT NULL DEFAULT 0,
    "schedule_revision" BIGINT NOT NULL DEFAULT 0,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("singleton_id"),
    CONSTRAINT "ck_interaction_store_metadata_singleton"
        CHECK ("singleton_id" = 1),
    CONSTRAINT "ck_interaction_store_metadata_generation_non_negative"
        CHECK ("store_generation" >= 0),
    CONSTRAINT "ck_interaction_store_metadata_schedule_non_negative"
        CHECK ("schedule_revision" >= 0)
);
""",
    """
INSERT INTO "interaction_store_metadata" (
    "singleton_id",
    "store_generation",
    "schedule_revision"
) VALUES (1, 0, 0)
ON CONFLICT ("singleton_id") DO NOTHING;
""",
    """
CREATE TABLE IF NOT EXISTS "interaction_records" (
    "request_id" TEXT NOT NULL,
    "continuation_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "turn_id" TEXT NOT NULL,
    "task_id" TEXT DEFAULT NULL,
    "agent_id" TEXT NOT NULL,
    "branch_id" TEXT NOT NULL,
    "model_call_id" TEXT NOT NULL,
    "scope_identity_digest" TEXT NOT NULL,
    "request_state" TEXT NOT NULL,
    "state_revision" BIGINT NOT NULL,
    "store_revision" BIGINT NOT NULL,
    "absolute_expires_at" TIMESTAMP WITH TIME ZONE NOT NULL,
    "retention_deadline_at" TIMESTAMP WITH TIME ZONE NOT NULL,
    "ciphertext" BYTEA NOT NULL,
    "encryption_key_id" TEXT NOT NULL,
    "encryption_algorithm" TEXT NOT NULL,
    "encryption_metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("request_id"),
    CONSTRAINT "uq_interaction_records_continuation"
        UNIQUE ("continuation_id"),
    CONSTRAINT "ck_interaction_records_identifiers_non_empty"
        CHECK (
            LENGTH(BTRIM("request_id")) > 0
            AND LENGTH(BTRIM("continuation_id")) > 0
            AND LENGTH(BTRIM("run_id")) > 0
            AND LENGTH(BTRIM("turn_id")) > 0
            AND LENGTH(BTRIM("agent_id")) > 0
            AND LENGTH(BTRIM("branch_id")) > 0
            AND LENGTH(BTRIM("model_call_id")) > 0
        ),
    CONSTRAINT "ck_interaction_records_task_id_non_empty"
        CHECK ("task_id" IS NULL OR LENGTH(BTRIM("task_id")) > 0),
    CONSTRAINT "ck_interaction_records_scope_identity_digest"
        CHECK ("scope_identity_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_interaction_records_request_state"
        CHECK (
            "request_state" IN (
                'pending',
                'answered',
                'declined',
                'cancelled',
                'timed_out',
                'unavailable',
                'expired',
                'superseded'
            )
        ),
    CONSTRAINT "ck_interaction_records_revisions_positive"
        CHECK ("state_revision" > 0 AND "store_revision" > 0),
    CONSTRAINT "ck_interaction_records_expiry_order"
        CHECK ("retention_deadline_at" >= "absolute_expires_at"),
    CONSTRAINT "ck_interaction_records_ciphertext_non_empty"
        CHECK (OCTET_LENGTH("ciphertext") > 0),
    CONSTRAINT "ck_interaction_records_encryption_non_empty"
        CHECK (
            LENGTH(BTRIM("encryption_key_id")) > 0
            AND LENGTH(BTRIM("encryption_algorithm")) > 0
        ),
    CONSTRAINT "ck_interaction_records_encryption_metadata_shape"
        CHECK (JSONB_TYPEOF("encryption_metadata") = 'object'),
    CONSTRAINT "ck_interaction_records_updated_at_not_before_created_at"
        CHECK ("updated_at" >= "created_at")
);
""",
    """
CREATE TABLE IF NOT EXISTS "interaction_branches" (
    "run_id" TEXT NOT NULL,
    "branch_id" TEXT NOT NULL,
    "parent_branch_id" TEXT NOT NULL,
    "root_branch_id" TEXT NOT NULL,
    "store_revision" BIGINT NOT NULL,
    "scope_identity_digest" TEXT NOT NULL,
    "ciphertext" BYTEA NOT NULL,
    "encryption_key_id" TEXT NOT NULL,
    "encryption_algorithm" TEXT NOT NULL,
    "encryption_metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("run_id", "branch_id", "scope_identity_digest"),
    CONSTRAINT "ck_interaction_branches_identifiers_non_empty"
        CHECK (
            LENGTH(BTRIM("run_id")) > 0
            AND LENGTH(BTRIM("branch_id")) > 0
            AND LENGTH(BTRIM("parent_branch_id")) > 0
            AND LENGTH(BTRIM("root_branch_id")) > 0
        ),
    CONSTRAINT "ck_interaction_branches_distinct_edge"
        CHECK ("branch_id" <> "parent_branch_id"),
    CONSTRAINT "ck_interaction_branches_revision_positive"
        CHECK ("store_revision" > 0),
    CONSTRAINT "ck_interaction_branches_scope_identity_digest"
        CHECK ("scope_identity_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_interaction_branches_ciphertext_non_empty"
        CHECK (OCTET_LENGTH("ciphertext") > 0),
    CONSTRAINT "ck_interaction_branches_encryption_non_empty"
        CHECK (
            LENGTH(BTRIM("encryption_key_id")) > 0
            AND LENGTH(BTRIM("encryption_algorithm")) > 0
        ),
    CONSTRAINT "ck_interaction_branches_encryption_metadata_shape"
        CHECK (JSONB_TYPEOF("encryption_metadata") = 'object'),
    CONSTRAINT "ck_interaction_branches_updated_at_not_before_created_at"
        CHECK ("updated_at" >= "created_at")
);
""",
    """
CREATE TABLE IF NOT EXISTS "interaction_continuations" (
    "continuation_id" TEXT NOT NULL,
    "checkpoint_id" TEXT DEFAULT NULL,
    "request_id" TEXT NOT NULL,
    "task_run_id" TEXT DEFAULT NULL,
    "lifecycle_state" TEXT NOT NULL DEFAULT 'pending',
    "state_revision" BIGINT NOT NULL,
    "store_revision" BIGINT NOT NULL,
    "claim_owner_id" TEXT DEFAULT NULL,
    "claim_lease_expires_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "fencing_token" BIGINT NOT NULL DEFAULT 0,
    "dispatch_id" TEXT DEFAULT NULL,
    "dispatch_started_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "dispatch_completed_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "dispatch_ambiguous" BOOLEAN NOT NULL DEFAULT FALSE,
    "invalid_reason" TEXT DEFAULT NULL,
    "ciphertext" BYTEA NOT NULL,
    "encryption_key_id" TEXT NOT NULL,
    "encryption_algorithm" TEXT NOT NULL,
    "encryption_metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "expires_at" TIMESTAMP WITH TIME ZONE NOT NULL,
    "retention_deadline_at" TIMESTAMP WITH TIME ZONE NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("continuation_id"),
    CONSTRAINT "fk_interaction_continuations__interaction_records"
        FOREIGN KEY ("request_id")
        REFERENCES "interaction_records" ("request_id")
        ON DELETE CASCADE,
    CONSTRAINT "fk_interaction_continuations__task_runs"
        FOREIGN KEY ("task_run_id")
        REFERENCES "task_runs" ("run_id")
        ON DELETE CASCADE,
    CONSTRAINT "uq_interaction_continuations_request"
        UNIQUE ("request_id"),
    CONSTRAINT "uq_interaction_continuations_checkpoint"
        UNIQUE ("checkpoint_id"),
    CONSTRAINT "uq_interaction_continuations_request_continuation"
        UNIQUE ("request_id", "continuation_id"),
    CONSTRAINT "uq_interaction_continuations_checkpoint_binding"
        UNIQUE ("request_id", "continuation_id", "checkpoint_id"),
    CONSTRAINT "ck_interaction_continuations_identifiers_non_empty"
        CHECK (
            LENGTH(BTRIM("continuation_id")) > 0
            AND LENGTH(BTRIM("request_id")) > 0
            AND (
                "task_run_id" IS NULL
                OR LENGTH(BTRIM("task_run_id")) > 0
            )
            AND (
                "checkpoint_id" IS NULL
                OR (
                    "task_run_id" IS NOT NULL
                    AND LENGTH(BTRIM("checkpoint_id")) > 0
                )
            )
        ),
    CONSTRAINT "ck_interaction_continuations_lifecycle"
        CHECK (
            "lifecycle_state" IN (
                'pending',
                'ready',
                'claimed',
                'dispatching',
                'completed',
                'invalidated'
            )
        ),
    CONSTRAINT "ck_interaction_continuations_revisions_positive"
        CHECK ("state_revision" >= 0 AND "store_revision" >= 0),
    CONSTRAINT "ck_interaction_continuations_fencing_non_negative"
        CHECK ("fencing_token" >= 0),
    CONSTRAINT "ck_interaction_continuations_claim_shape"
        CHECK (
            (
                "lifecycle_state" = 'claimed'
                AND "claim_owner_id" IS NOT NULL
                AND "claim_lease_expires_at" IS NOT NULL
            )
            OR (
                "lifecycle_state" = 'dispatching'
                AND "claim_owner_id" IS NOT NULL
                AND "claim_lease_expires_at" IS NULL
            )
            OR (
                "lifecycle_state" = 'completed'
                AND "claim_owner_id" IS NOT NULL
                AND "claim_lease_expires_at" IS NULL
            )
            OR (
                "lifecycle_state" = 'ready'
                AND "claim_lease_expires_at" IS NULL
            )
            OR (
                "lifecycle_state" NOT IN (
                    'ready',
                    'claimed',
                    'dispatching',
                    'completed'
                )
                AND "claim_owner_id" IS NULL
                AND "claim_lease_expires_at" IS NULL
            )
        ),
    CONSTRAINT "ck_interaction_continuations_dispatch_shape"
        CHECK (
            (
                "lifecycle_state" = 'dispatching'
                AND "dispatch_id" IS NOT NULL
                AND "dispatch_started_at" IS NOT NULL
                AND "dispatch_ambiguous"
            )
            OR "lifecycle_state" <> 'dispatching'
        ),
    CONSTRAINT "ck_interaction_continuations_invalid_reason"
        CHECK (
            (
                "lifecycle_state" = 'invalidated'
                AND "invalid_reason" IS NOT NULL
                AND LENGTH(BTRIM("invalid_reason")) > 0
            )
            OR (
                "lifecycle_state" <> 'invalidated'
                AND "invalid_reason" IS NULL
            )
        ),
    CONSTRAINT "ck_interaction_continuations_ciphertext_non_empty"
        CHECK (OCTET_LENGTH("ciphertext") > 0),
    CONSTRAINT "ck_interaction_continuations_encryption_non_empty"
        CHECK (
            LENGTH(BTRIM("encryption_key_id")) > 0
            AND LENGTH(BTRIM("encryption_algorithm")) > 0
        ),
    CONSTRAINT "ck_interaction_continuations_encryption_metadata_shape"
        CHECK (JSONB_TYPEOF("encryption_metadata") = 'object'),
    CONSTRAINT "ck_interaction_continuations_expiry_order"
        CHECK ("retention_deadline_at" >= "expires_at"),
    CONSTRAINT "ck_interaction_continuations_updated_at_not_before_created_at"
        CHECK ("updated_at" >= "created_at")
);
""",
    """
CREATE TABLE IF NOT EXISTS "interaction_resolution_keys" (
    "request_id" TEXT NOT NULL,
    "idempotency_key" TEXT NOT NULL,
    "resolution_digest" TEXT NOT NULL,
    "state_revision" BIGINT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("request_id", "idempotency_key"),
    CONSTRAINT "fk_interaction_resolution_keys__interaction_records"
        FOREIGN KEY ("request_id")
        REFERENCES "interaction_records" ("request_id")
        ON DELETE CASCADE,
    CONSTRAINT "ck_interaction_resolution_keys_values_non_empty"
        CHECK (
            LENGTH(BTRIM("request_id")) > 0
            AND LENGTH(BTRIM("idempotency_key")) > 0
            AND "resolution_digest" ~ '^[0-9a-f]{64}$'
        ),
    CONSTRAINT "ck_interaction_resolution_keys_revision_positive"
        CHECK ("state_revision" > 0)
);
""",
    """
CREATE TABLE IF NOT EXISTS "interaction_resumption_outbox" (
    "outbox_id" TEXT NOT NULL,
    "continuation_id" TEXT NOT NULL,
    "request_id" TEXT NOT NULL,
    "task_run_id" TEXT DEFAULT NULL,
    "resolution_revision" BIGINT NOT NULL,
    "status" TEXT NOT NULL DEFAULT 'pending',
    "claim_owner_id" TEXT DEFAULT NULL,
    "claim_lease_expires_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "fencing_token" BIGINT NOT NULL DEFAULT 0,
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "last_error_code" TEXT DEFAULT NULL,
    "available_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "delivered_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY ("outbox_id"),
    CONSTRAINT "fk_interaction_resumption_outbox__continuations"
        FOREIGN KEY ("continuation_id")
        REFERENCES "interaction_continuations" ("continuation_id")
        ON DELETE CASCADE,
    CONSTRAINT "fk_interaction_resumption_outbox__interaction_records"
        FOREIGN KEY ("request_id")
        REFERENCES "interaction_records" ("request_id")
        ON DELETE CASCADE,
    CONSTRAINT "fk_interaction_resumption_outbox__task_runs"
        FOREIGN KEY ("task_run_id")
        REFERENCES "task_runs" ("run_id")
        ON DELETE CASCADE,
    CONSTRAINT "uq_interaction_resumption_outbox_resolution"
        UNIQUE ("continuation_id", "resolution_revision"),
    CONSTRAINT "ck_interaction_resumption_outbox_status"
        CHECK ("status" IN ('pending', 'claimed', 'delivered', 'dead')),
    CONSTRAINT "ck_interaction_resumption_outbox_revision_positive"
        CHECK ("resolution_revision" > 0),
    CONSTRAINT "ck_interaction_resumption_outbox_counters_non_negative"
        CHECK ("fencing_token" >= 0 AND "attempts" >= 0),
    CONSTRAINT "ck_interaction_resumption_outbox_claim_shape"
        CHECK (
            (
                "status" = 'claimed'
                AND "claim_owner_id" IS NOT NULL
                AND "claim_lease_expires_at" IS NOT NULL
            )
            OR (
                "status" <> 'claimed'
                AND "claim_owner_id" IS NULL
                AND "claim_lease_expires_at" IS NULL
            )
        ),
    CONSTRAINT "ck_interaction_resumption_outbox_delivery_shape"
        CHECK (
            ("status" = 'delivered' AND "delivered_at" IS NOT NULL)
            OR ("status" <> 'delivered' AND "delivered_at" IS NULL)
        ),
    CONSTRAINT "ck_interaction_resumption_outbox_updated_at"
        CHECK ("updated_at" >= "created_at")
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_attempt_segments" (
    "segment_id" TEXT NOT NULL,
    "attempt_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "segment_number" INTEGER NOT NULL,
    "state" TEXT NOT NULL,
    "claim" JSONB DEFAULT NULL,
    "resumed_from_segment_id" TEXT DEFAULT NULL,
    "request_id" TEXT DEFAULT NULL,
    "continuation_id" TEXT DEFAULT NULL,
    "checkpoint_id" TEXT DEFAULT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("segment_id"),
    CONSTRAINT "fk_task_attempt_segments__attempts"
        FOREIGN KEY ("attempt_id")
        REFERENCES "task_attempts" ("attempt_id")
        ON DELETE CASCADE,
    CONSTRAINT "fk_task_attempt_segments__attempt_run"
        FOREIGN KEY ("run_id", "attempt_id")
        REFERENCES "task_attempts" ("run_id", "attempt_id")
        ON DELETE CASCADE,
    CONSTRAINT "fk_task_attempt_segments__previous"
        FOREIGN KEY ("resumed_from_segment_id")
        REFERENCES "task_attempt_segments" ("segment_id"),
    CONSTRAINT "fk_task_attempt_segments__interactions"
        FOREIGN KEY ("request_id", "continuation_id", "checkpoint_id")
        REFERENCES "interaction_continuations" (
            "request_id", "continuation_id", "checkpoint_id"
        )
        ON DELETE SET NULL,
    CONSTRAINT "uq_task_attempt_segments_attempt_order"
        UNIQUE ("attempt_id", "segment_number"),
    CONSTRAINT "ck_task_attempt_segments_number_positive"
        CHECK ("segment_number" > 0),
    CONSTRAINT "ck_task_attempt_segments_state"
        CHECK (
            "state" IN (
                'created',
                'running',
                'suspended',
                'succeeded',
                'failed',
                'abandoned'
            )
        ),
    CONSTRAINT "ck_task_attempt_segments_claim_shape"
        CHECK ("claim" IS NULL OR JSONB_TYPEOF("claim") = 'object'),
    CONSTRAINT "ck_task_attempt_segments_interaction_pair"
        CHECK (
            ("request_id" IS NULL) = ("continuation_id" IS NULL)
        ),
    CONSTRAINT "ck_task_attempt_segments_checkpoint_correlation"
        CHECK ("checkpoint_id" IS NULL OR "request_id" IS NOT NULL),
    CONSTRAINT "ck_task_attempt_segments_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object'),
    CONSTRAINT "ck_task_attempt_segments_updated_at"
        CHECK ("updated_at" >= "created_at")
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_attempt_segment_transitions" (
    "transition_id" TEXT NOT NULL,
    "segment_id" TEXT NOT NULL,
    "attempt_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "from_state" TEXT NOT NULL,
    "to_state" TEXT NOT NULL,
    "reason" TEXT NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("transition_id"),
    CONSTRAINT "fk_task_attempt_segment_transitions__segments"
        FOREIGN KEY ("segment_id")
        REFERENCES "task_attempt_segments" ("segment_id")
        ON DELETE CASCADE,
    CONSTRAINT "fk_task_attempt_segment_transitions__attempts"
        FOREIGN KEY ("attempt_id")
        REFERENCES "task_attempts" ("attempt_id")
        ON DELETE CASCADE,
    CONSTRAINT "fk_task_attempt_segment_transitions__attempt_run"
        FOREIGN KEY ("run_id", "attempt_id")
        REFERENCES "task_attempts" ("run_id", "attempt_id")
        ON DELETE CASCADE,
    CONSTRAINT "ck_task_attempt_segment_transitions_reason_non_empty"
        CHECK (LENGTH(BTRIM("reason")) > 0),
    CONSTRAINT "ck_task_attempt_segment_transitions_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
ALTER TABLE "task_runs"
    DROP CONSTRAINT IF EXISTS "ck_task_runs_state";
ALTER TABLE "task_runs"
    ADD CONSTRAINT "ck_task_runs_state"
    CHECK (
        "state" IN (
            'created',
            'validated',
            'queued',
            'claimed',
            'running',
            'input_required',
            'succeeded',
            'failed',
            'cancel_requested',
            'cancelled',
            'expired'
        )
    );
""",
    """
ALTER TABLE "task_attempts"
    DROP CONSTRAINT IF EXISTS "ck_task_attempts_state";
ALTER TABLE "task_attempts"
    ADD CONSTRAINT "ck_task_attempts_state"
    CHECK (
        "state" IN (
            'created',
            'running',
            'suspended',
            'succeeded',
            'failed',
            'abandoned'
        )
    );
""",
    """
ALTER TABLE "task_queue_items"
    ADD COLUMN IF NOT EXISTS "attempt_id" TEXT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS "segment_id" TEXT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS "request_id" TEXT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS "continuation_id" TEXT DEFAULT NULL;
ALTER TABLE "task_queue_items"
    DROP CONSTRAINT IF EXISTS "fk_task_queue_items__attempts";
ALTER TABLE "task_queue_items"
    ADD CONSTRAINT "fk_task_queue_items__attempts"
    FOREIGN KEY ("attempt_id")
    REFERENCES "task_attempts" ("attempt_id");
ALTER TABLE "task_queue_items"
    DROP CONSTRAINT IF EXISTS "fk_task_queue_items__segments";
ALTER TABLE "task_queue_items"
    ADD CONSTRAINT "fk_task_queue_items__segments"
    FOREIGN KEY ("segment_id")
    REFERENCES "task_attempt_segments" ("segment_id");
ALTER TABLE "task_queue_items"
    DROP CONSTRAINT IF EXISTS "fk_task_queue_items__interactions";
ALTER TABLE "task_queue_items"
    ADD CONSTRAINT "fk_task_queue_items__interactions"
    FOREIGN KEY ("request_id", "continuation_id")
    REFERENCES "interaction_continuations" ("request_id", "continuation_id")
    ON DELETE SET NULL;
ALTER TABLE "task_queue_items"
    DROP CONSTRAINT IF EXISTS "fk_task_queue_items__continuations";
ALTER TABLE "task_queue_items"
    DROP CONSTRAINT IF EXISTS "ck_task_queue_items_interaction_pair";
ALTER TABLE "task_queue_items"
    ADD CONSTRAINT "ck_task_queue_items_interaction_pair"
    CHECK (("request_id" IS NULL) = ("continuation_id" IS NULL));
ALTER TABLE "task_queue_items"
    DROP CONSTRAINT IF EXISTS "ck_task_queue_items_segment_attempt";
ALTER TABLE "task_queue_items"
    ADD CONSTRAINT "ck_task_queue_items_segment_attempt"
    CHECK ("segment_id" IS NULL OR "attempt_id" IS NOT NULL);
""",
    """
ALTER TABLE "task_usage_records"
    ADD COLUMN IF NOT EXISTS "segment_id" TEXT DEFAULT NULL;
ALTER TABLE "task_usage_records"
    DROP CONSTRAINT IF EXISTS "fk_task_usage_records__task_segments";
ALTER TABLE "task_usage_records"
    ADD CONSTRAINT "fk_task_usage_records__task_segments"
    FOREIGN KEY ("segment_id")
    REFERENCES "task_attempt_segments" ("segment_id")
    ON DELETE SET NULL;
ALTER TABLE "task_usage_records"
    DROP CONSTRAINT IF EXISTS "ck_task_usage_records_segment_attempt";
ALTER TABLE "task_usage_records"
    ADD CONSTRAINT "ck_task_usage_records_segment_attempt"
    CHECK ("segment_id" IS NULL OR "attempt_id" IS NOT NULL);
""",
    """
ALTER TABLE "task_queue_items"
    DROP CONSTRAINT IF EXISTS "ck_task_queue_items_state";
ALTER TABLE "task_queue_items"
    ADD CONSTRAINT "ck_task_queue_items_state"
    CHECK (
        "state" IN ('available', 'claimed', 'suspended', 'done', 'dead')
    );
""",
    """
ALTER TABLE "task_queue_items"
    DROP CONSTRAINT IF EXISTS "ck_task_queue_items_unclaimed_fields";
ALTER TABLE "task_queue_items"
    ADD CONSTRAINT "ck_task_queue_items_unclaimed_fields"
    CHECK (
        "state" = 'claimed'
        OR (
            "claimed_at" IS NULL
            AND "lease_expires_at" IS NULL
            AND "worker_id" IS NULL
            AND "claim_token" IS NULL
            AND "heartbeat_at" IS NULL
        )
    );
""",
    """
DROP INDEX IF EXISTS "uq_task_queue_items_one_active_per_run";
CREATE UNIQUE INDEX "uq_task_queue_items_one_active_per_run"
    ON "task_queue_items" ("run_id")
    WHERE "state" IN ('available', 'claimed', 'suspended');
""",
    """
CREATE OR REPLACE FUNCTION "interaction_lock_scope_before_record_delete"()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            jsonb_build_array(
                'avalan.interaction.retention.v1',
                OLD."run_id",
                OLD."scope_identity_digest"
            )::text,
            0
        )
    );
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "trg_interaction_lock_scope_before_record_delete"
    ON "interaction_records";
CREATE TRIGGER "trg_interaction_lock_scope_before_record_delete"
BEFORE DELETE ON "interaction_records"
FOR EACH ROW
EXECUTE FUNCTION "interaction_lock_scope_before_record_delete"();
""",
    """
CREATE OR REPLACE FUNCTION "interaction_delete_orphaned_branches"()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            jsonb_build_array(
                'avalan.interaction.retention.v1',
                OLD."run_id",
                OLD."scope_identity_digest"
            )::text,
            0
        )
    );
    IF NOT EXISTS (
        SELECT 1
        FROM "interaction_records"
        WHERE "run_id" = OLD."run_id"
          AND "scope_identity_digest" = OLD."scope_identity_digest"
    ) THEN
        DELETE FROM "interaction_branches"
        WHERE "run_id" = OLD."run_id"
          AND "scope_identity_digest" = OLD."scope_identity_digest";
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "trg_interaction_delete_orphaned_branches"
    ON "interaction_records";
CREATE TRIGGER "trg_interaction_delete_orphaned_branches"
AFTER DELETE ON "interaction_records"
FOR EACH ROW
EXECUTE FUNCTION "interaction_delete_orphaned_branches"();
""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS "uq_task_attempt_segments_active_attempt"
    ON "task_attempt_segments" ("attempt_id")
    WHERE "state" IN ('created', 'running');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_records_scope"
    ON "interaction_records" (
        "scope_identity_digest",
        "run_id",
        "branch_id",
        "request_state"
    );
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_records_correlation_scope"
    ON "interaction_records" (
        "request_id",
        "scope_identity_digest"
    );
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_branches_scope"
    ON "interaction_branches" (
        "scope_identity_digest",
        "run_id",
        "branch_id"
    );
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_records_expiry"
    ON "interaction_records" ("absolute_expires_at", "request_id")
    WHERE "request_state" = 'pending';
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_records_retention"
    ON "interaction_records" ("retention_deadline_at", "request_id");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_continuations_claimable"
    ON "interaction_continuations" (
        "lifecycle_state",
        "expires_at",
        "continuation_id"
    )
    WHERE "lifecycle_state" = 'ready';
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_continuations_lease_expiry"
    ON "interaction_continuations" (
        "claim_lease_expires_at",
        "continuation_id"
    )
    WHERE "lifecycle_state" = 'claimed';
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_continuations_retention"
    ON "interaction_continuations" (
        "retention_deadline_at",
        "continuation_id"
    );
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_resumption_outbox_claimable"
    ON "interaction_resumption_outbox" (
        "available_at",
        "outbox_id"
    )
    WHERE "status" = 'pending';
""",
    """
CREATE INDEX IF NOT EXISTS "ix_interaction_resumption_outbox_lease"
    ON "interaction_resumption_outbox" (
        "claim_lease_expires_at",
        "outbox_id"
    )
    WHERE "status" = 'claimed';
""",
)


def upgrade() -> None:
    """Apply the durable interaction schema."""
    _execute_all(TASK_SCHEMA_STATEMENTS)


def downgrade() -> None:
    """Reject reverse migration of durable continuation data."""
    raise NotImplementedError("task PostgreSQL migrations are forward-only")


def _execute_all(statements: Iterable[str]) -> None:
    alembic = cast(Any, import_module("alembic"))
    bind = alembic.op.get_bind()
    for statement in statements:
        bind.exec_driver_sql(statement)
