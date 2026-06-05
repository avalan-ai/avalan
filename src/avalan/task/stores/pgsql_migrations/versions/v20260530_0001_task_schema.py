from collections.abc import Iterable
from importlib import import_module
from typing import Any, cast

revision = "20260530_0001"
down_revision = None
branch_labels = ("task",)
depends_on = None

TASK_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
CREATE TABLE IF NOT EXISTS "task_definitions" (
    "definition_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "version" TEXT NOT NULL,
    "spec_hash" TEXT NOT NULL,
    "definition" JSONB NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("definition_id"),
    CONSTRAINT "ck_task_definitions_definition_id_non_empty"
        CHECK (LENGTH(BTRIM("definition_id")) > 0),
    CONSTRAINT "ck_task_definitions_name_non_empty"
        CHECK (LENGTH(BTRIM("name")) > 0),
    CONSTRAINT "ck_task_definitions_version_non_empty"
        CHECK (LENGTH(BTRIM("version")) > 0),
    CONSTRAINT "ck_task_definitions_spec_hash_non_empty"
        CHECK (LENGTH(BTRIM("spec_hash")) > 0),
    CONSTRAINT "ck_task_definitions_definition_shape"
        CHECK (JSONB_TYPEOF("definition") = 'object'),
    CONSTRAINT "ck_task_definitions_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object'),
    CONSTRAINT "uq_task_definitions_identity"
        UNIQUE ("name", "version", "spec_hash")
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_runs" (
    "run_id" TEXT NOT NULL,
    "definition_id" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "queue_name" TEXT DEFAULT NULL,
    "request" JSONB NOT NULL,
    "claim" JSONB DEFAULT NULL,
    "last_attempt_id" TEXT DEFAULT NULL,
    "result" JSONB DEFAULT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("run_id"),
    CONSTRAINT "fk_task_runs__task_definitions"
        FOREIGN KEY ("definition_id")
        REFERENCES "task_definitions" ("definition_id"),
    CONSTRAINT "ck_task_runs_run_id_non_empty"
        CHECK (LENGTH(BTRIM("run_id")) > 0),
    CONSTRAINT "ck_task_runs_state"
        CHECK (
            "state" IN (
                'created',
                'validated',
                'queued',
                'claimed',
                'running',
                'succeeded',
                'failed',
                'cancel_requested',
                'cancelled',
                'expired'
            )
        ),
    CONSTRAINT "ck_task_runs_updated_at_not_before_created_at"
        CHECK ("updated_at" >= "created_at"),
    CONSTRAINT "ck_task_runs_queue_name_non_empty"
        CHECK ("queue_name" IS NULL OR LENGTH(BTRIM("queue_name")) > 0),
    CONSTRAINT "ck_task_runs_request_shape"
        CHECK (JSONB_TYPEOF("request") = 'object'),
    CONSTRAINT "ck_task_runs_claim_shape"
        CHECK ("claim" IS NULL OR JSONB_TYPEOF("claim") = 'object'),
    CONSTRAINT "ck_task_runs_result_shape"
        CHECK ("result" IS NULL OR JSONB_TYPEOF("result") = 'object'),
    CONSTRAINT "ck_task_runs_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_run_transitions" (
    "transition_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "from_state" TEXT NOT NULL,
    "to_state" TEXT NOT NULL,
    "reason" TEXT NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("transition_id"),
    CONSTRAINT "fk_task_run_transitions__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "ck_task_run_transitions_reason_non_empty"
        CHECK (LENGTH(BTRIM("reason")) > 0),
    CONSTRAINT "ck_task_run_transitions_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_attempts" (
    "attempt_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "attempt_number" INTEGER NOT NULL,
    "state" TEXT NOT NULL,
    "context" JSONB NOT NULL,
    "result" JSONB DEFAULT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("attempt_id"),
    CONSTRAINT "fk_task_attempts__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "uq_task_attempts_run_order"
        UNIQUE ("run_id", "attempt_number"),
    CONSTRAINT "uq_task_attempts_run_attempt"
        UNIQUE ("run_id", "attempt_id"),
    CONSTRAINT "ck_task_attempts_attempt_number_positive"
        CHECK ("attempt_number" > 0),
    CONSTRAINT "ck_task_attempts_state"
        CHECK (
            "state" IN (
                'created',
                'running',
                'succeeded',
                'failed',
                'abandoned'
            )
        ),
    CONSTRAINT "ck_task_attempts_updated_at_not_before_created_at"
        CHECK ("updated_at" >= "created_at"),
    CONSTRAINT "ck_task_attempts_context_shape"
        CHECK (JSONB_TYPEOF("context") = 'object'),
    CONSTRAINT "ck_task_attempts_result_shape"
        CHECK ("result" IS NULL OR JSONB_TYPEOF("result") = 'object'),
    CONSTRAINT "ck_task_attempts_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_attempt_transitions" (
    "transition_id" TEXT NOT NULL,
    "attempt_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "from_state" TEXT NOT NULL,
    "to_state" TEXT NOT NULL,
    "reason" TEXT NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("transition_id"),
    CONSTRAINT "fk_task_attempt_transitions__task_attempts"
        FOREIGN KEY ("attempt_id")
        REFERENCES "task_attempts" ("attempt_id"),
    CONSTRAINT "fk_task_attempt_transitions__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "fk_task_attempt_transitions__task_attempt_run"
        FOREIGN KEY ("run_id", "attempt_id")
        REFERENCES "task_attempts" ("run_id", "attempt_id"),
    CONSTRAINT "ck_task_attempt_transitions_reason_non_empty"
        CHECK (LENGTH(BTRIM("reason")) > 0),
    CONSTRAINT "ck_task_attempt_transitions_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_artifacts" (
    "artifact_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "attempt_id" TEXT DEFAULT NULL,
    "purpose" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "ref" JSONB NOT NULL,
    "provenance" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "retention" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("artifact_id"),
    CONSTRAINT "fk_task_artifacts__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "fk_task_artifacts__task_attempts"
        FOREIGN KEY ("attempt_id")
        REFERENCES "task_attempts" ("attempt_id"),
    CONSTRAINT "fk_task_artifacts__task_attempt_run"
        FOREIGN KEY ("run_id", "attempt_id")
        REFERENCES "task_attempts" ("run_id", "attempt_id"),
    CONSTRAINT "ck_task_artifacts_purpose"
        CHECK (
            "purpose" IN (
                'input',
                'converted',
                'output',
                'intermediate'
            )
        ),
    CONSTRAINT "ck_task_artifacts_state"
        CHECK (
            "state" IN (
                'ready',
                'deleted',
                'lost'
            )
        ),
    CONSTRAINT "ck_task_artifacts_updated_at_not_before_created_at"
        CHECK ("updated_at" >= "created_at"),
    CONSTRAINT "ck_task_artifacts_ref_shape"
        CHECK (JSONB_TYPEOF("ref") = 'object'),
    CONSTRAINT "ck_task_artifacts_provenance_shape"
        CHECK (JSONB_TYPEOF("provenance") = 'object'),
    CONSTRAINT "ck_task_artifacts_retention_shape"
        CHECK (JSONB_TYPEOF("retention") = 'object'),
    CONSTRAINT "ck_task_artifacts_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_artifact_bytes" (
    "artifact_id" TEXT NOT NULL,
    "storage_key" TEXT NOT NULL,
    "media_type" TEXT DEFAULT NULL,
    "size_bytes" BIGINT NOT NULL,
    "sha256" TEXT NOT NULL,
    "ciphertext" BYTEA NOT NULL,
    "encryption_key_id" TEXT NOT NULL,
    "encryption_algorithm" TEXT NOT NULL,
    "encryption_metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "retention_days" INTEGER NOT NULL,
    "retention_deadline_at" TIMESTAMP WITH TIME ZONE NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "deleted_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY ("storage_key"),
    CONSTRAINT "ck_task_artifact_bytes_artifact_id_non_empty"
        CHECK (LENGTH(BTRIM("artifact_id")) > 0),
    CONSTRAINT "ck_task_artifact_bytes_storage_key_non_empty"
        CHECK (LENGTH(BTRIM("storage_key")) > 0),
    CONSTRAINT "ck_task_artifact_bytes_size_non_negative"
        CHECK ("size_bytes" >= 0),
    CONSTRAINT "ck_task_artifact_bytes_sha256"
        CHECK ("sha256" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_task_artifact_bytes_ciphertext_non_empty"
        CHECK (LENGTH("ciphertext") > 0),
    CONSTRAINT "ck_task_artifact_bytes_encryption_key_id_non_empty"
        CHECK (LENGTH(BTRIM("encryption_key_id")) > 0),
    CONSTRAINT "ck_task_artifact_bytes_encryption_algorithm_non_empty"
        CHECK (LENGTH(BTRIM("encryption_algorithm")) > 0),
    CONSTRAINT "ck_task_artifact_bytes_retention_positive"
        CHECK ("retention_days" > 0),
    CONSTRAINT "ck_task_artifact_bytes_retention_deadline_after_created"
        CHECK ("retention_deadline_at" > "created_at"),
    CONSTRAINT "ck_task_artifact_bytes_deleted_at_not_before_created_at"
        CHECK ("deleted_at" IS NULL OR "deleted_at" >= "created_at"),
    CONSTRAINT "ck_task_artifact_bytes_encryption_metadata_shape"
        CHECK (JSONB_TYPEOF("encryption_metadata") = 'object'),
    CONSTRAINT "ck_task_artifact_bytes_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_idempotency_keys" (
    "identity_key" TEXT NOT NULL,
    "task_name" TEXT NOT NULL,
    "task_version" TEXT NOT NULL,
    "spec_hash" TEXT NOT NULL,
    "owner_scope_hash" JSONB NOT NULL,
    "strategy" TEXT NOT NULL,
    "window_hash" JSONB DEFAULT NULL,
    "input_hash" JSONB DEFAULT NULL,
    "file_hash" JSONB DEFAULT NULL,
    "custom_hash" JSONB DEFAULT NULL,
    "run_id" TEXT NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "expires_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("identity_key"),
    CONSTRAINT "fk_task_idempotency_keys__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "ck_task_idempotency_keys_identity_key_non_empty"
        CHECK (LENGTH(BTRIM("identity_key")) > 0),
    CONSTRAINT "ck_task_idempotency_keys_task_name_non_empty"
        CHECK (LENGTH(BTRIM("task_name")) > 0),
    CONSTRAINT "ck_task_idempotency_keys_task_version_non_empty"
        CHECK (LENGTH(BTRIM("task_version")) > 0),
    CONSTRAINT "ck_task_idempotency_keys_spec_hash_non_empty"
        CHECK (LENGTH(BTRIM("spec_hash")) > 0),
    CONSTRAINT "ck_task_idempotency_keys_strategy"
        CHECK (
            "strategy" IN (
                'input_hash',
                'input_and_files_hash',
                'custom'
            )
        ),
    CONSTRAINT "ck_task_idempotency_keys_expires_after_created"
        CHECK ("expires_at" IS NULL OR "expires_at" > "created_at"),
    CONSTRAINT "ck_task_idempotency_keys_owner_scope_shape"
        CHECK (JSONB_TYPEOF("owner_scope_hash") = 'object'),
    CONSTRAINT "ck_task_idempotency_keys_window_shape"
        CHECK (
            "window_hash" IS NULL
            OR JSONB_TYPEOF("window_hash") = 'object'
        ),
    CONSTRAINT "ck_task_idempotency_keys_input_shape"
        CHECK ("input_hash" IS NULL OR JSONB_TYPEOF("input_hash") = 'object'),
    CONSTRAINT "ck_task_idempotency_keys_file_shape"
        CHECK ("file_hash" IS NULL OR JSONB_TYPEOF("file_hash") = 'object'),
    CONSTRAINT "ck_task_idempotency_keys_custom_shape"
        CHECK (
            "custom_hash" IS NULL
            OR JSONB_TYPEOF("custom_hash") = 'object'
        ),
    CONSTRAINT "ck_task_idempotency_keys_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_queue_items" (
    "queue_item_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "queue_name" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "priority" INTEGER NOT NULL DEFAULT 0,
    "available_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "claimed_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "lease_expires_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "worker_id" TEXT DEFAULT NULL,
    "claim_token" TEXT DEFAULT NULL,
    "heartbeat_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("queue_item_id"),
    CONSTRAINT "fk_task_queue_items__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "ck_task_queue_items_state"
        CHECK ("state" IN ('available', 'claimed', 'done', 'dead')),
    CONSTRAINT "ck_task_queue_items_queue_name_non_empty"
        CHECK (LENGTH(BTRIM("queue_name")) > 0),
    CONSTRAINT "ck_task_queue_items_attempts_non_negative"
        CHECK ("attempts" >= 0),
    CONSTRAINT "ck_task_queue_items_worker_id_non_empty"
        CHECK ("worker_id" IS NULL OR LENGTH(BTRIM("worker_id")) > 0),
    CONSTRAINT "ck_task_queue_items_claim_token_non_empty"
        CHECK ("claim_token" IS NULL OR LENGTH(BTRIM("claim_token")) > 0),
    CONSTRAINT "ck_task_queue_items_lease_after_claim"
        CHECK (
            "lease_expires_at" IS NULL
            OR "claimed_at" IS NOT NULL
        ),
    CONSTRAINT "ck_task_queue_items_lease_expires_after_claim"
        CHECK (
            "lease_expires_at" IS NULL
            OR "lease_expires_at" > "claimed_at"
        ),
    CONSTRAINT "ck_task_queue_items_heartbeat_after_claim"
        CHECK (
            "heartbeat_at" IS NULL
            OR (
                "claimed_at" IS NOT NULL
                AND "heartbeat_at" >= "claimed_at"
            )
        ),
    CONSTRAINT "ck_task_queue_items_claimed_fields"
        CHECK (
            "state" <> 'claimed'
            OR (
                "claimed_at" IS NOT NULL
                AND "lease_expires_at" IS NOT NULL
                AND "worker_id" IS NOT NULL
                AND "claim_token" IS NOT NULL
            )
        ),
    CONSTRAINT "ck_task_queue_items_updated_at_not_before_created_at"
        CHECK ("updated_at" >= "created_at"),
    CONSTRAINT "ck_task_queue_items_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_events" (
    "event_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "attempt_id" TEXT DEFAULT NULL,
    "sequence" BIGINT NOT NULL,
    "event_type" TEXT NOT NULL,
    "event_time" TIMESTAMP WITH TIME ZONE NOT NULL,
    "payload" JSONB NOT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("event_id"),
    CONSTRAINT "fk_task_events__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "fk_task_events__task_attempts"
        FOREIGN KEY ("attempt_id")
        REFERENCES "task_attempts" ("attempt_id"),
    CONSTRAINT "fk_task_events__task_attempt_run"
        FOREIGN KEY ("run_id", "attempt_id")
        REFERENCES "task_attempts" ("run_id", "attempt_id"),
    CONSTRAINT "uq_task_events_run_sequence"
        UNIQUE ("run_id", "sequence"),
    CONSTRAINT "ck_task_events_sequence_positive"
        CHECK ("sequence" > 0),
    CONSTRAINT "ck_task_events_event_type_non_empty"
        CHECK (LENGTH(BTRIM("event_type")) > 0),
    CONSTRAINT "ck_task_events_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_usage_records" (
    "usage_id" TEXT NOT NULL,
    "run_id" TEXT NOT NULL,
    "attempt_id" TEXT DEFAULT NULL,
    "sequence" BIGINT NOT NULL,
    "source" TEXT NOT NULL,
    "prompt_tokens" INTEGER DEFAULT NULL,
    "completion_tokens" INTEGER DEFAULT NULL,
    "total_tokens" INTEGER DEFAULT NULL,
    "cached_tokens" INTEGER DEFAULT NULL,
    "cache_creation_input_tokens" INTEGER DEFAULT NULL,
    "reasoning_tokens" INTEGER DEFAULT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("usage_id"),
    CONSTRAINT "fk_task_usage_records__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "fk_task_usage_records__task_attempts"
        FOREIGN KEY ("attempt_id")
        REFERENCES "task_attempts" ("attempt_id"),
    CONSTRAINT "fk_task_usage_records__task_attempt_run"
        FOREIGN KEY ("run_id", "attempt_id")
        REFERENCES "task_attempts" ("run_id", "attempt_id"),
    CONSTRAINT "uq_task_usage_records_run_sequence"
        UNIQUE ("run_id", "sequence"),
    CONSTRAINT "ck_task_usage_records_sequence_positive"
        CHECK ("sequence" > 0),
    CONSTRAINT "ck_task_usage_records_source_non_empty"
        CHECK (LENGTH(BTRIM("source")) > 0),
    CONSTRAINT "ck_task_usage_records_prompt_tokens_non_negative"
        CHECK ("prompt_tokens" IS NULL OR "prompt_tokens" >= 0),
    CONSTRAINT "ck_task_usage_records_completion_tokens_non_negative"
        CHECK ("completion_tokens" IS NULL OR "completion_tokens" >= 0),
    CONSTRAINT "ck_task_usage_records_total_tokens_non_negative"
        CHECK ("total_tokens" IS NULL OR "total_tokens" >= 0),
    CONSTRAINT "ck_task_usage_records_cached_tokens_non_negative"
        CHECK ("cached_tokens" IS NULL OR "cached_tokens" >= 0),
    CONSTRAINT "ck_task_usage_records_cache_creation_tokens_non_negative"
        CHECK (
            "cache_creation_input_tokens" IS NULL
            OR "cache_creation_input_tokens" >= 0
        ),
    CONSTRAINT "ck_task_usage_records_reasoning_tokens_non_negative"
        CHECK ("reasoning_tokens" IS NULL OR "reasoning_tokens" >= 0),
    CONSTRAINT "ck_task_usage_records_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE TABLE IF NOT EXISTS "task_run_rollups" (
    "run_id" TEXT NOT NULL,
    "event_count" BIGINT NOT NULL DEFAULT 0,
    "prompt_tokens" BIGINT DEFAULT NULL,
    "completion_tokens" BIGINT DEFAULT NULL,
    "total_tokens" BIGINT DEFAULT NULL,
    "cached_tokens" BIGINT DEFAULT NULL,
    "cache_creation_input_tokens" BIGINT DEFAULT NULL,
    "reasoning_tokens" BIGINT DEFAULT NULL,
    "metadata" JSONB NOT NULL DEFAULT '{}'::JSONB,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("run_id"),
    CONSTRAINT "fk_task_run_rollups__task_runs"
        FOREIGN KEY ("run_id")
        REFERENCES "task_runs" ("run_id"),
    CONSTRAINT "ck_task_run_rollups_event_count_non_negative"
        CHECK ("event_count" >= 0),
    CONSTRAINT "ck_task_run_rollups_prompt_tokens_non_negative"
        CHECK ("prompt_tokens" IS NULL OR "prompt_tokens" >= 0),
    CONSTRAINT "ck_task_run_rollups_completion_tokens_non_negative"
        CHECK ("completion_tokens" IS NULL OR "completion_tokens" >= 0),
    CONSTRAINT "ck_task_run_rollups_total_tokens_non_negative"
        CHECK ("total_tokens" IS NULL OR "total_tokens" >= 0),
    CONSTRAINT "ck_task_run_rollups_cached_tokens_non_negative"
        CHECK ("cached_tokens" IS NULL OR "cached_tokens" >= 0),
    CONSTRAINT "ck_task_run_rollups_cache_creation_tokens_non_negative"
        CHECK (
            "cache_creation_input_tokens" IS NULL
            OR "cache_creation_input_tokens" >= 0
        ),
    CONSTRAINT "ck_task_run_rollups_reasoning_tokens_non_negative"
        CHECK ("reasoning_tokens" IS NULL OR "reasoning_tokens" >= 0),
    CONSTRAINT "ck_task_run_rollups_metadata_shape"
        CHECK (JSONB_TYPEOF("metadata") = 'object')
);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_definitions_by_name_version"
    ON "task_definitions" ("name", "version");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_runs_by_definition_state_created"
    ON "task_runs" ("definition_id", "state", "created_at" DESC);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_runs_by_state_updated"
    ON "task_runs" ("state", "updated_at" DESC);
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_runs_by_queue_state_updated"
    ON "task_runs" ("queue_name", "state", "updated_at" DESC)
    WHERE "queue_name" IS NOT NULL;
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_run_transitions_by_run_created"
    ON "task_run_transitions" ("run_id", "created_at", "transition_id");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_attempts_by_run_order"
    ON "task_attempts" ("run_id", "attempt_number");
""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS "uq_task_attempts_one_active_per_run"
    ON "task_attempts" ("run_id")
    WHERE "state" NOT IN ('succeeded', 'failed', 'abandoned');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_attempt_transitions_by_attempt_created"
    ON "task_attempt_transitions"
    ("attempt_id", "created_at", "transition_id");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_artifacts_by_run_purpose_state"
    ON "task_artifacts" ("run_id", "purpose", "state");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_artifacts_by_attempt"
    ON "task_artifacts" ("attempt_id")
    WHERE "attempt_id" IS NOT NULL;
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_artifacts_retention_deadline"
    ON "task_artifacts" (
        "run_id",
        (("retention" ->> 'expires_at')),
        "artifact_id"
    )
    WHERE
        "state" = 'ready'
        AND ("retention" ? 'expires_at');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_artifacts_delete_after_days"
    ON "task_artifacts" (
        "run_id",
        (("retention" ->> 'delete_after_days')),
        "created_at"
    )
    WHERE
        "state" = 'ready'
        AND ("retention" ? 'delete_after_days');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_artifact_bytes_by_artifact"
    ON "task_artifact_bytes" ("artifact_id");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_artifact_bytes_retention_deadline"
    ON "task_artifact_bytes" ("retention_deadline_at", "storage_key")
    WHERE "deleted_at" IS NULL;
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_artifact_bytes_active_artifact"
    ON "task_artifact_bytes" ("artifact_id", "storage_key")
    WHERE "deleted_at" IS NULL;
""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS "uq_task_idempotency_keys_identity"
    ON "task_idempotency_keys" ("identity_key");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_idempotency_keys_by_run"
    ON "task_idempotency_keys" ("run_id");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_idempotency_keys_by_task_window"
    ON "task_idempotency_keys"
    ("task_name", "task_version", "spec_hash", "strategy", "expires_at");
""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS "uq_task_queue_items_one_active_per_run"
    ON "task_queue_items" ("run_id")
    WHERE "state" IN ('available', 'claimed');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_queue_items_claimable"
    ON "task_queue_items" (
        "queue_name",
        "state",
        "available_at",
        "priority" DESC,
        "queue_item_id"
    )
    WHERE "state" = 'available';
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_queue_items_lease_expiry"
    ON "task_queue_items" (
        "queue_name",
        "lease_expires_at",
        "queue_item_id"
    )
    WHERE "state" = 'claimed';
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_queue_items_retry_sweep"
    ON "task_queue_items" ("queue_name", "attempts", "available_at")
    WHERE "state" IN ('available', 'dead');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_events_by_run_sequence"
    ON "task_events" ("run_id", "sequence");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_events_by_attempt"
    ON "task_events" ("attempt_id", "sequence")
    WHERE "attempt_id" IS NOT NULL;
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_usage_records_by_run"
    ON "task_usage_records" ("run_id", "created_at");
""",
    """
CREATE INDEX IF NOT EXISTS "ix_task_usage_records_by_run_sequence"
    ON "task_usage_records" ("run_id", "sequence");
""",
    """
CREATE OR REPLACE FUNCTION "task_reject_terminal_run_state_change"()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD."state" IN ('succeeded', 'failed', 'cancelled', 'expired')
        AND NEW."state" IS DISTINCT FROM OLD."state" THEN
        RAISE EXCEPTION 'terminal task run state cannot be changed'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "tr_task_runs_terminal_state" ON "task_runs";
""",
    """
CREATE TRIGGER "tr_task_runs_terminal_state"
    BEFORE UPDATE ON "task_runs"
    FOR EACH ROW
    EXECUTE FUNCTION "task_reject_terminal_run_state_change"();
""",
)


def upgrade() -> None:
    _execute_all(TASK_SCHEMA_STATEMENTS)


def downgrade() -> None:
    raise NotImplementedError("task PostgreSQL migrations are forward-only")


def _execute_all(statements: Iterable[str]) -> None:
    alembic = cast(Any, import_module("alembic"))
    bind = alembic.op.get_bind()
    for statement in statements:
        bind.exec_driver_sql(statement)
