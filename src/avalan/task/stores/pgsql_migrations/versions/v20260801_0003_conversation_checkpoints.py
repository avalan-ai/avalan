"""Add encrypted durable conversation checkpoints and references."""

from collections.abc import Iterable
from importlib import import_module
from typing import Any, cast

revision = "20260801_0003"
down_revision = "20260723_0002"
branch_labels = None
depends_on = None

CONVERSATION_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
CREATE TABLE IF NOT EXISTS "conversation_store_metadata" (
    "singleton_id" SMALLINT NOT NULL DEFAULT 1,
    "schema_version" INTEGER NOT NULL DEFAULT 1,
    "minimum_reader_version" INTEGER NOT NULL DEFAULT 1,
    "maximum_reader_version" INTEGER NOT NULL DEFAULT 2,
    "minimum_writer_version" INTEGER NOT NULL DEFAULT 1,
    "maximum_writer_version" INTEGER NOT NULL DEFAULT 2,
    "checkpoint_codec_version" INTEGER NOT NULL DEFAULT 1,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("singleton_id"),
    CONSTRAINT "ck_conversation_store_metadata_singleton"
        CHECK ("singleton_id" = 1),
    CONSTRAINT "ck_conversation_store_metadata_versions_positive"
        CHECK (
            "schema_version" > 0
            AND "minimum_reader_version" > 0
            AND "maximum_reader_version" >= "minimum_reader_version"
            AND "minimum_writer_version" > 0
            AND "maximum_writer_version" >= "minimum_writer_version"
            AND "checkpoint_codec_version" > 0
        )
);
""",
    """
INSERT INTO "conversation_store_metadata" (
    "singleton_id",
    "schema_version",
    "minimum_reader_version",
    "maximum_reader_version",
    "minimum_writer_version",
    "maximum_writer_version",
    "checkpoint_codec_version"
) VALUES (1, 1, 1, 2, 1, 2, 1)
ON CONFLICT ("singleton_id") DO NOTHING;
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_key_revisions" (
    "authority_digest" TEXT NOT NULL,
    "key_id" TEXT NOT NULL,
    "key_revision" BIGINT NOT NULL,
    "key_status" TEXT NOT NULL,
    "algorithm" TEXT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "retired_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY ("authority_digest", "key_id", "key_revision"),
    CONSTRAINT "uq_conversation_key_revisions_generation"
        UNIQUE ("authority_digest", "key_revision"),
    CONSTRAINT "ck_conversation_key_revisions_authority_digest"
        CHECK ("authority_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_conversation_key_revisions_values"
        CHECK (
            LENGTH(BTRIM("key_id")) > 0
            AND "key_revision" > 0
            AND LENGTH(BTRIM("algorithm")) > 0
        ),
    CONSTRAINT "ck_conversation_key_revisions_status"
        CHECK ("key_status" IN ('current', 'grace', 'retired')),
    CONSTRAINT "ck_conversation_key_revisions_retirement_shape"
        CHECK (
            ("key_status" = 'retired' AND "retired_at" IS NOT NULL)
            OR ("key_status" <> 'retired' AND "retired_at" IS NULL)
        )
);
""",
    """
CREATE UNIQUE INDEX IF NOT EXISTS "uq_conversation_key_revisions_current"
    ON "conversation_key_revisions" ("authority_digest")
    WHERE "key_status" = 'current';
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_key_authorities" (
    "authority_digest" TEXT NOT NULL,
    "current_generation" BIGINT NOT NULL DEFAULT 0,
    "current_key_id" TEXT DEFAULT NULL,
    "current_key_revision" BIGINT DEFAULT NULL,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("authority_digest"),
    CONSTRAINT "ck_conversation_key_authorities_authority_digest"
        CHECK ("authority_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_conversation_key_authorities_current_shape"
        CHECK (
            (
                "current_generation" = 0
                AND "current_key_id" IS NULL
                AND "current_key_revision" IS NULL
            )
            OR (
                "current_generation" > 0
                AND LENGTH(BTRIM("current_key_id")) > 0
                AND "current_key_revision" = "current_generation"
            )
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversations" (
    "conversation_id" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "lifecycle_state" TEXT NOT NULL DEFAULT 'active',
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("conversation_id"),
    CONSTRAINT "uq_conversations_authority_identity"
        UNIQUE ("conversation_id", "authority_digest"),
    CONSTRAINT "ck_conversations_identifier_non_empty"
        CHECK (LENGTH(BTRIM("conversation_id")) > 0),
    CONSTRAINT "ck_conversations_authority_digest"
        CHECK ("authority_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_conversations_lifecycle"
        CHECK ("lifecycle_state" IN ('active', 'tombstoned', 'deleted')),
    CONSTRAINT "ck_conversations_updated_at"
        CHECK ("updated_at" >= "created_at")
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_checkpoints" (
    "checkpoint_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "logical_turn_id" TEXT NOT NULL,
    "execution_segment_id" TEXT NOT NULL,
    "branch_id" TEXT NOT NULL,
    "parent_checkpoint_id" TEXT DEFAULT NULL,
    "checkpoint_sequence" BIGINT NOT NULL,
    "parent_sequence" BIGINT DEFAULT NULL,
    "checkpoint_kind" TEXT NOT NULL,
    "lifecycle_state" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "checkpoint_codec_version" INTEGER NOT NULL,
    "payload_schema_version" INTEGER NOT NULL,
    "payload_count" INTEGER NOT NULL DEFAULT 0,
    "payload_bytes" BIGINT NOT NULL,
    "lane_count" INTEGER NOT NULL,
    "provider_item_count" INTEGER NOT NULL,
    "opaque_byte_count" BIGINT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL,
    "committed_at" TIMESTAMP WITH TIME ZONE NOT NULL,
    "expires_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "tombstoned_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "deleted_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY ("checkpoint_id"),
    CONSTRAINT "fk_conversation_checkpoints__conversations"
        FOREIGN KEY ("conversation_id", "authority_digest")
        REFERENCES "conversations" ("conversation_id", "authority_digest"),
    CONSTRAINT "fk_conversation_checkpoints__parent"
        FOREIGN KEY ("parent_checkpoint_id")
        REFERENCES "conversation_checkpoints" ("checkpoint_id"),
    CONSTRAINT "uq_conversation_checkpoints_segment"
        UNIQUE ("checkpoint_id", "execution_segment_id"),
    CONSTRAINT "uq_conversation_checkpoints_authority_identity"
        UNIQUE ("checkpoint_id", "conversation_id", "authority_digest"),
    CONSTRAINT "uq_conversation_checkpoints_sequence"
        UNIQUE ("conversation_id", "branch_id", "checkpoint_sequence"),
    CONSTRAINT "ck_conversation_checkpoints_identifiers_non_empty"
        CHECK (
            LENGTH(BTRIM("checkpoint_id")) > 0
            AND LENGTH(BTRIM("conversation_id")) > 0
            AND LENGTH(BTRIM("logical_turn_id")) > 0
            AND LENGTH(BTRIM("execution_segment_id")) > 0
            AND LENGTH(BTRIM("branch_id")) > 0
        ),
    CONSTRAINT "ck_conversation_checkpoints_authority_digest"
        CHECK ("authority_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_conversation_checkpoints_sequence_shape"
        CHECK (
            (
                "parent_checkpoint_id" IS NULL
                AND "parent_sequence" IS NULL
                AND "checkpoint_sequence" = 0
            )
            OR (
                "parent_checkpoint_id" IS NOT NULL
                AND "parent_sequence" IS NOT NULL
                AND "checkpoint_sequence" = "parent_sequence" + 1
            )
        ),
    CONSTRAINT "ck_conversation_checkpoints_kind"
        CHECK (
            "checkpoint_kind" IN (
                'internal_provider_boundary',
                'structured_input_suspension',
                'completed_outward_turn',
                'standalone_compact_result',
                'tombstone',
                'supersession'
            )
        ),
    CONSTRAINT "ck_conversation_checkpoints_lifecycle"
        CHECK (
            "lifecycle_state" IN (
                'committed', 'expired', 'tombstoned', 'deleted'
            )
        ),
    CONSTRAINT "ck_conversation_checkpoints_versions_positive"
        CHECK (
            "checkpoint_codec_version" > 0
            AND "payload_schema_version" > 0
        ),
    CONSTRAINT "ck_conversation_checkpoints_counts_bounded"
        CHECK (
            "payload_count" >= 0
            AND "payload_bytes" > 0
            AND "lane_count" > 0
            AND "provider_item_count" >= 0
            AND "opaque_byte_count" >= 0
        ),
    CONSTRAINT "ck_conversation_checkpoints_timestamp_order"
        CHECK (
            "committed_at" >= "created_at"
            AND ("expires_at" IS NULL OR "expires_at" > "committed_at")
            AND (
                "tombstoned_at" IS NULL
                OR "tombstoned_at" >= "committed_at"
            )
            AND (
                "deleted_at" IS NULL
                OR "deleted_at" >= COALESCE("tombstoned_at", "committed_at")
            )
        ),
    CONSTRAINT "ck_conversation_checkpoints_tombstone_shape"
        CHECK (
            (
                "lifecycle_state" = 'tombstoned'
                AND "tombstoned_at" IS NOT NULL
                AND "deleted_at" IS NULL
            )
            OR (
                "lifecycle_state" = 'deleted'
                AND "deleted_at" IS NOT NULL
            )
            OR (
                "lifecycle_state" IN ('committed', 'expired')
                AND "tombstoned_at" IS NULL
                AND "deleted_at" IS NULL
            )
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_lanes" (
    "checkpoint_id" TEXT NOT NULL,
    "lane_id" TEXT NOT NULL,
    "lane_sequence" INTEGER NOT NULL,
    "lane_mode" TEXT NOT NULL,
    "binding_digest" TEXT NOT NULL,
    "execution_digest" TEXT DEFAULT NULL,
    "provider_item_count" INTEGER NOT NULL,
    "opaque_byte_count" BIGINT NOT NULL,
    "upstream_deletion_state" TEXT NOT NULL DEFAULT 'not_applicable',

    PRIMARY KEY ("checkpoint_id", "lane_id"),
    CONSTRAINT "fk_conversation_lanes__checkpoints"
        FOREIGN KEY ("checkpoint_id")
        REFERENCES "conversation_checkpoints" ("checkpoint_id")
        ON DELETE CASCADE,
    CONSTRAINT "uq_conversation_lanes_order"
        UNIQUE ("checkpoint_id", "lane_sequence"),
    CONSTRAINT "ck_conversation_lanes_values"
        CHECK (
            LENGTH(BTRIM("lane_id")) > 0
            AND "lane_sequence" >= 0
            AND "binding_digest" ~ '^[0-9a-f]{64}$'
            AND (
                "execution_digest" IS NULL
                OR "execution_digest" ~ '^[0-9a-f]{64}$'
            )
            AND "provider_item_count" >= 0
            AND "opaque_byte_count" >= 0
        ),
    CONSTRAINT "ck_conversation_lanes_mode"
        CHECK ("lane_mode" IN ('stateless', 'stored')),
    CONSTRAINT "ck_conversation_lanes_upstream_deletion"
        CHECK (
            "upstream_deletion_state" IN (
                'not_applicable', 'pending', 'succeeded', 'failed',
                'unsupported'
            )
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_encrypted_payloads" (
    "payload_id" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "checkpoint_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "lane_id" TEXT NOT NULL,
    "payload_sequence" INTEGER NOT NULL,
    "payload_kind" TEXT NOT NULL,
    "payload_schema_version" INTEGER NOT NULL,
    "codec_version" INTEGER NOT NULL,
    "key_id" TEXT NOT NULL,
    "key_revision" BIGINT NOT NULL,
    "algorithm" TEXT NOT NULL,
    "nonce" BYTEA NOT NULL,
    "ciphertext" BYTEA NOT NULL,
    "authenticated_digest" TEXT NOT NULL,
    "reference_count" INTEGER NOT NULL DEFAULT 0,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("payload_id"),
    CONSTRAINT "uq_conversation_encrypted_payloads_reference_identity"
        UNIQUE (
            "payload_id", "checkpoint_id", "conversation_id",
            "authority_digest", "lane_id", "payload_sequence",
            "payload_kind", "payload_schema_version", "codec_version",
            "key_id", "key_revision", "algorithm",
            "authenticated_digest"
        ),
    CONSTRAINT "fk_conversation_encrypted_payloads__checkpoints"
        FOREIGN KEY (
            "checkpoint_id", "conversation_id", "authority_digest"
        )
        REFERENCES "conversation_checkpoints" (
            "checkpoint_id", "conversation_id", "authority_digest"
        )
        ON DELETE CASCADE,
    CONSTRAINT "fk_conversation_encrypted_payloads__keys"
        FOREIGN KEY ("authority_digest", "key_id", "key_revision")
        REFERENCES "conversation_key_revisions" (
            "authority_digest", "key_id", "key_revision"
        ),
    CONSTRAINT "ck_conversation_encrypted_payloads_values"
        CHECK (
            LENGTH(BTRIM("payload_id")) > 0
            AND "authority_digest" ~ '^[0-9a-f]{64}$'
            AND LENGTH(BTRIM("checkpoint_id")) > 0
            AND LENGTH(BTRIM("conversation_id")) > 0
            AND LENGTH(BTRIM("lane_id")) > 0
            AND "payload_sequence" >= 0
            AND "payload_schema_version" > 0
            AND "codec_version" > 0
            AND LENGTH(BTRIM("key_id")) > 0
            AND "key_revision" > 0
            AND LENGTH(BTRIM("algorithm")) > 0
            AND OCTET_LENGTH("nonce") >= 12
            AND OCTET_LENGTH("ciphertext") > 16
            AND "authenticated_digest" ~ '^[0-9a-f]{64}$'
            AND "reference_count" IN (0, 1)
        ),
    CONSTRAINT "ck_conversation_encrypted_payloads_kind"
        CHECK (
            "payload_kind" IN (
                'checkpoint', 'lane_output', 'continuation_reference',
                'deletion_target'
            )
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_checkpoint_payload_refs" (
    "checkpoint_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "lane_id" TEXT NOT NULL,
    "payload_sequence" INTEGER NOT NULL,
    "payload_kind" TEXT NOT NULL,
    "payload_schema_version" INTEGER NOT NULL,
    "codec_version" INTEGER NOT NULL,
    "key_id" TEXT NOT NULL,
    "key_revision" BIGINT NOT NULL,
    "algorithm" TEXT NOT NULL,
    "authenticated_digest" TEXT NOT NULL,
    "payload_id" TEXT NOT NULL,

    PRIMARY KEY (
        "checkpoint_id", "lane_id", "payload_sequence", "payload_kind"
    ),
    CONSTRAINT "fk_conversation_checkpoint_payload_refs__checkpoints"
        FOREIGN KEY (
            "checkpoint_id", "conversation_id", "authority_digest"
        )
        REFERENCES "conversation_checkpoints" (
            "checkpoint_id", "conversation_id", "authority_digest"
        )
        ON DELETE CASCADE,
    CONSTRAINT "fk_conversation_checkpoint_payload_refs__payloads"
        FOREIGN KEY (
            "payload_id", "checkpoint_id", "conversation_id",
            "authority_digest", "lane_id", "payload_sequence",
            "payload_kind", "payload_schema_version", "codec_version",
            "key_id", "key_revision", "algorithm",
            "authenticated_digest"
        )
        REFERENCES "conversation_encrypted_payloads" (
            "payload_id", "checkpoint_id", "conversation_id",
            "authority_digest", "lane_id", "payload_sequence",
            "payload_kind", "payload_schema_version", "codec_version",
            "key_id", "key_revision", "algorithm",
            "authenticated_digest"
        )
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT "uq_conversation_checkpoint_payload_refs_payload"
        UNIQUE ("payload_id"),
    CONSTRAINT "uq_conversation_checkpoint_payload_refs_exact"
        UNIQUE (
            "checkpoint_id", "conversation_id", "authority_digest",
            "lane_id", "payload_sequence", "payload_kind", "payload_id"
        ),
    CONSTRAINT "ck_conversation_checkpoint_payload_refs_values"
        CHECK (
            LENGTH(BTRIM("conversation_id")) > 0
            AND "authority_digest" ~ '^[0-9a-f]{64}$'
            AND LENGTH(BTRIM("lane_id")) > 0
            AND "payload_sequence" >= 0
            AND "payload_schema_version" > 0
            AND "codec_version" > 0
            AND LENGTH(BTRIM("key_id")) > 0
            AND "key_revision" > 0
            AND LENGTH(BTRIM("algorithm")) > 0
            AND "authenticated_digest" ~ '^[0-9a-f]{64}$'
            AND "payload_kind" IN (
                'checkpoint', 'lane_output', 'continuation_reference',
                'deletion_target'
            )
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_named_heads" (
    "authority_digest" TEXT NOT NULL,
    "head_id" TEXT NOT NULL,
    "head_revision" BIGINT NOT NULL,
    "checkpoint_id" TEXT NOT NULL,
    "lifecycle_state" TEXT NOT NULL DEFAULT 'active',
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("authority_digest", "head_id"),
    CONSTRAINT "fk_conversation_named_heads__checkpoints"
        FOREIGN KEY ("checkpoint_id")
        REFERENCES "conversation_checkpoints" ("checkpoint_id"),
    CONSTRAINT "ck_conversation_named_heads_values"
        CHECK (
            "authority_digest" ~ '^[0-9a-f]{64}$'
            AND LENGTH(BTRIM("head_id")) > 0
            AND "head_revision" >= 0
        ),
    CONSTRAINT "ck_conversation_named_heads_lifecycle"
        CHECK ("lifecycle_state" IN ('active', 'tombstoned'))
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_idempotency" (
    "authority_digest" TEXT NOT NULL,
    "operation" TEXT NOT NULL,
    "idempotency_key" TEXT NOT NULL,
    "request_digest" TEXT NOT NULL,
    "record_state" TEXT NOT NULL,
    "owner_token" TEXT NOT NULL,
    "lease_expires_at" TIMESTAMP WITH TIME ZONE NOT NULL,
    "execution_digest" TEXT DEFAULT NULL,
    "checkpoint_id" TEXT DEFAULT NULL,
    "public_response_id" TEXT DEFAULT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("authority_digest", "operation", "idempotency_key"),
    CONSTRAINT "fk_conversation_idempotency__checkpoints"
        FOREIGN KEY ("checkpoint_id")
        REFERENCES "conversation_checkpoints" ("checkpoint_id"),
    CONSTRAINT "ck_conversation_idempotency_values"
        CHECK (
            "authority_digest" ~ '^[0-9a-f]{64}$'
            AND LENGTH(BTRIM("idempotency_key")) > 0
            AND LENGTH(BTRIM("request_digest")) > 0
            AND LENGTH(BTRIM("owner_token")) > 0
            AND (
                "execution_digest" IS NULL
                OR "execution_digest" ~ '^[0-9a-f]{64}$'
            )
        ),
    CONSTRAINT "ck_conversation_idempotency_operation"
        CHECK (
            "operation" IN (
                'create', 'continue', 'branch', 'compact', 'retrieve',
                'delete'
            )
        ),
    CONSTRAINT "ck_conversation_idempotency_state"
        CHECK (
            "record_state" IN (
                'in_progress', 'committed', 'failed_no_dispatch', 'ambiguous'
            )
        ),
    CONSTRAINT "ck_conversation_idempotency_commit_shape"
        CHECK (
            ("record_state" = 'committed' AND "checkpoint_id" IS NOT NULL)
            OR ("record_state" <> 'committed' AND "checkpoint_id" IS NULL)
        ),
    CONSTRAINT "ck_conversation_idempotency_updated_at"
        CHECK ("updated_at" >= "created_at")
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_execution_staging" (
    "staging_id" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "operation" TEXT NOT NULL,
    "idempotency_key" TEXT NOT NULL,
    "request_digest" TEXT NOT NULL,
    "owner_token" TEXT NOT NULL,
    "checkpoint_id" TEXT NOT NULL,
    "lane_id" TEXT NOT NULL,
    "binding_digest" TEXT NOT NULL,
    "execution_digest" TEXT NOT NULL,
    "lane_mode" TEXT NOT NULL,
    "output_scope" TEXT NOT NULL,
    "item_count" INTEGER NOT NULL,
    "opaque_byte_count" BIGINT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("staging_id"),
    CONSTRAINT "fk_conversation_execution_staging__idempotency"
        FOREIGN KEY ("authority_digest", "operation", "idempotency_key")
        REFERENCES "conversation_idempotency" (
            "authority_digest", "operation", "idempotency_key"
        )
        ON DELETE CASCADE,
    CONSTRAINT "uq_conversation_execution_staging_owner_lane"
        UNIQUE ("owner_token", "checkpoint_id", "lane_id"),
    CONSTRAINT "ck_conversation_execution_staging_digests"
        CHECK (
            "authority_digest" ~ '^[0-9a-f]{64}$'
            AND LENGTH(BTRIM("request_digest")) > 0
            AND "binding_digest" ~ '^[0-9a-f]{64}$'
            AND "execution_digest" ~ '^[0-9a-f]{64}$'
        ),
    CONSTRAINT "ck_conversation_execution_staging_mode"
        CHECK ("lane_mode" IN ('stateless', 'stored')),
    CONSTRAINT "ck_conversation_execution_staging_scope"
        CHECK ("output_scope" IN ('current_call', 'cumulative')),
    CONSTRAINT "ck_conversation_execution_staging_counts"
        CHECK ("item_count" >= 0 AND "opaque_byte_count" >= 0)
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_execution_reservation_lanes" (
    "authority_digest" TEXT NOT NULL,
    "operation" TEXT NOT NULL,
    "idempotency_key" TEXT NOT NULL,
    "checkpoint_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "logical_turn_id" TEXT NOT NULL,
    "execution_segment_id" TEXT NOT NULL,
    "branch_id" TEXT NOT NULL,
    "checkpoint_sequence" BIGINT NOT NULL,
    "parent_checkpoint_id" TEXT DEFAULT NULL,
    "parent_sequence" BIGINT DEFAULT NULL,
    "lane_id" TEXT NOT NULL,
    "binding_digest" TEXT NOT NULL,
    "lane_mode" TEXT NOT NULL,
    "output_scope" TEXT NOT NULL,

    PRIMARY KEY (
        "authority_digest", "operation", "idempotency_key", "lane_id"
    ),
    CONSTRAINT "fk_conversation_execution_reservation_lanes__idempotency"
        FOREIGN KEY ("authority_digest", "operation", "idempotency_key")
        REFERENCES "conversation_idempotency" (
            "authority_digest", "operation", "idempotency_key"
        )
        ON DELETE CASCADE,
    CONSTRAINT "ck_conversation_execution_reservation_lanes_values"
        CHECK (
            "authority_digest" ~ '^[0-9a-f]{64}$'
            AND LENGTH(BTRIM("checkpoint_id")) > 0
            AND LENGTH(BTRIM("conversation_id")) > 0
            AND LENGTH(BTRIM("logical_turn_id")) > 0
            AND LENGTH(BTRIM("execution_segment_id")) > 0
            AND LENGTH(BTRIM("branch_id")) > 0
            AND LENGTH(BTRIM("lane_id")) > 0
            AND "binding_digest" ~ '^[0-9a-f]{64}$'
        ),
    CONSTRAINT "ck_conversation_execution_reservation_lanes_sequence"
        CHECK (
            (
                "parent_checkpoint_id" IS NULL
                AND "parent_sequence" IS NULL
                AND "checkpoint_sequence" = 0
            )
            OR (
                "parent_checkpoint_id" IS NOT NULL
                AND "parent_sequence" IS NOT NULL
                AND "checkpoint_sequence" = "parent_sequence" + 1
            )
        ),
    CONSTRAINT "ck_conversation_execution_reservation_lanes_mode"
        CHECK ("lane_mode" IN ('stateless', 'stored')),
    CONSTRAINT "ck_conversation_execution_reservation_lanes_scope"
        CHECK ("output_scope" IN ('current_call', 'cumulative'))
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_provisional_responses" (
    "provisional_response_id" TEXT NOT NULL,
    "public_response_id" TEXT NOT NULL,
    "owner_token" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("provisional_response_id"),
    CONSTRAINT "uq_conversation_provisional_responses_public"
        UNIQUE ("public_response_id"),
    CONSTRAINT "ck_conversation_provisional_responses_values"
        CHECK (
            LENGTH(BTRIM("provisional_response_id")) > 0
            AND LENGTH(BTRIM("public_response_id")) > 0
            AND LENGTH(BTRIM("owner_token")) > 0
            AND "authority_digest" ~ '^[0-9a-f]{64}$'
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_public_responses" (
    "public_response_id" TEXT NOT NULL,
    "checkpoint_id" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "tombstoned" BOOLEAN NOT NULL DEFAULT FALSE,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "tombstoned_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY ("public_response_id"),
    CONSTRAINT "fk_conversation_public_responses__checkpoints"
        FOREIGN KEY ("checkpoint_id")
        REFERENCES "conversation_checkpoints" ("checkpoint_id"),
    CONSTRAINT "uq_conversation_public_responses_checkpoint"
        UNIQUE ("checkpoint_id"),
    CONSTRAINT "ck_conversation_public_responses_authority_digest"
        CHECK ("authority_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_conversation_public_responses_tombstone_shape"
        CHECK ("tombstoned" = ("tombstoned_at" IS NOT NULL))
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_outbox" (
    "intent_id" TEXT NOT NULL,
    "checkpoint_id" TEXT NOT NULL,
    "public_response_id" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "outbox_state" TEXT NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "lease_owner" TEXT DEFAULT NULL,
    "lease_expires_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "available_sequence" BIGSERIAL NOT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "published_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY ("intent_id"),
    CONSTRAINT "fk_conversation_outbox__public_responses"
        FOREIGN KEY ("public_response_id")
        REFERENCES "conversation_public_responses" ("public_response_id")
        ON DELETE CASCADE,
    CONSTRAINT "uq_conversation_outbox_response"
        UNIQUE ("public_response_id"),
    CONSTRAINT "ck_conversation_outbox_authority_digest"
        CHECK ("authority_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_conversation_outbox_state"
        CHECK ("outbox_state" IN ('pending', 'claimed', 'published')),
    CONSTRAINT "ck_conversation_outbox_attempts"
        CHECK ("attempts" >= 0),
    CONSTRAINT "ck_conversation_outbox_claim_shape"
        CHECK (
            (
                "outbox_state" = 'claimed'
                AND "lease_owner" IS NOT NULL
                AND "lease_expires_at" IS NOT NULL
                AND "published_at" IS NULL
            )
            OR (
                "outbox_state" = 'published'
                AND "lease_owner" IS NULL
                AND "lease_expires_at" IS NULL
                AND "published_at" IS NOT NULL
            )
            OR (
                "outbox_state" = 'pending'
                AND "lease_owner" IS NULL
                AND "lease_expires_at" IS NULL
                AND "published_at" IS NULL
            )
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_reconciliation_outbox" (
    "reconciliation_id" TEXT NOT NULL,
    "checkpoint_id" TEXT NOT NULL,
    "lane_id" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "target_conversation_id" TEXT NOT NULL,
    "target_payload_sequence" INTEGER NOT NULL,
    "target_payload_kind" TEXT NOT NULL,
    "target_payload_id" TEXT NOT NULL,
    "work_kind" TEXT NOT NULL,
    "work_state" TEXT NOT NULL DEFAULT 'pending',
    "attempts" INTEGER NOT NULL DEFAULT 0,
    "lease_owner" TEXT DEFAULT NULL,
    "lease_expires_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,
    "created_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completed_at" TIMESTAMP WITH TIME ZONE DEFAULT NULL,

    PRIMARY KEY ("reconciliation_id"),
    CONSTRAINT "fk_conversation_reconciliation_outbox__lanes"
        FOREIGN KEY ("checkpoint_id", "lane_id")
        REFERENCES "conversation_lanes" ("checkpoint_id", "lane_id")
        ON DELETE CASCADE,
    CONSTRAINT "fk_conversation_reconciliation_outbox__target"
        FOREIGN KEY (
            "checkpoint_id", "target_conversation_id", "authority_digest",
            "lane_id", "target_payload_sequence", "target_payload_kind",
            "target_payload_id"
        )
        REFERENCES "conversation_checkpoint_payload_refs" (
            "checkpoint_id", "conversation_id", "authority_digest",
            "lane_id", "payload_sequence", "payload_kind", "payload_id"
        )
        ON DELETE CASCADE,
    CONSTRAINT "uq_conversation_reconciliation_outbox_lane"
        UNIQUE ("checkpoint_id", "lane_id", "work_kind"),
    CONSTRAINT "ck_conversation_reconciliation_outbox_authority_digest"
        CHECK ("authority_digest" ~ '^[0-9a-f]{64}$'),
    CONSTRAINT "ck_conversation_reconciliation_outbox_kind"
        CHECK ("work_kind" IN ('delete_upstream', 'rewrap_payload')),
    CONSTRAINT "ck_conversation_reconciliation_outbox_target"
        CHECK (
            LENGTH(BTRIM("target_conversation_id")) > 0
            AND "target_payload_sequence" = 0
            AND "target_payload_kind" = 'deletion_target'
            AND LENGTH(BTRIM("target_payload_id")) > 0
        ),
    CONSTRAINT "ck_conversation_reconciliation_outbox_state"
        CHECK ("work_state" IN ('pending', 'claimed', 'completed', 'failed')),
    CONSTRAINT "ck_conversation_reconciliation_outbox_attempts"
        CHECK ("attempts" >= 0),
    CONSTRAINT "ck_conversation_reconciliation_outbox_claim_shape"
        CHECK (
            (
                "work_state" = 'claimed'
                AND "lease_owner" IS NOT NULL
                AND "lease_expires_at" IS NOT NULL
                AND "completed_at" IS NULL
            )
            OR (
                "work_state" = 'completed'
                AND "lease_owner" IS NULL
                AND "lease_expires_at" IS NULL
                AND "completed_at" IS NOT NULL
            )
            OR (
                "work_state" IN ('pending', 'failed')
                AND "lease_owner" IS NULL
                AND "lease_expires_at" IS NULL
                AND "completed_at" IS NULL
            )
        )
);
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_checkpoint_continuations" (
    "checkpoint_id" TEXT NOT NULL,
    "conversation_id" TEXT NOT NULL,
    "authority_digest" TEXT NOT NULL,
    "execution_segment_id" TEXT NOT NULL,
    "continuation_id" TEXT NOT NULL,
    "continuation_state_revision" BIGINT NOT NULL,
    "continuation_digest" TEXT NOT NULL,
    "definition_digest" TEXT NOT NULL,
    "revision_binding_digest" TEXT NOT NULL,
    "payload_lane_id" TEXT NOT NULL,
    "payload_sequence" INTEGER NOT NULL,
    "payload_kind" TEXT NOT NULL,
    "payload_id" TEXT NOT NULL,

    PRIMARY KEY ("checkpoint_id"),
    CONSTRAINT "fk_conversation_checkpoint_continuations__checkpoint_segment"
        FOREIGN KEY ("checkpoint_id", "execution_segment_id")
        REFERENCES "conversation_checkpoints" (
            "checkpoint_id", "execution_segment_id"
        )
        ON DELETE CASCADE,
    CONSTRAINT "fk_conversation_checkpoint_continuations__payload_ref"
        FOREIGN KEY (
            "checkpoint_id", "conversation_id", "authority_digest",
            "payload_lane_id", "payload_sequence", "payload_kind",
            "payload_id"
        )
        REFERENCES "conversation_checkpoint_payload_refs" (
            "checkpoint_id", "conversation_id", "authority_digest",
            "lane_id", "payload_sequence", "payload_kind", "payload_id"
        )
        ON DELETE CASCADE,
    CONSTRAINT "uq_conversation_checkpoint_continuations_identity"
        UNIQUE ("continuation_id", "checkpoint_id"),
    CONSTRAINT "ck_conversation_checkpoint_continuations_values"
        CHECK (
            "continuation_state_revision" >= 0
            AND LENGTH(BTRIM("conversation_id")) > 0
            AND "authority_digest" ~ '^[0-9a-f]{64}$'
            AND LENGTH(BTRIM("continuation_digest")) > 0
            AND "definition_digest" ~ '^[0-9a-f]{64}$'
            AND "revision_binding_digest" ~ '^[0-9a-f]{64}$'
            AND "payload_lane_id" = 'structured-input'
            AND "payload_sequence" = 0
            AND "payload_kind" = 'continuation_reference'
        )
);
""",
    """
ALTER TABLE "interaction_continuations"
    ADD COLUMN IF NOT EXISTS "conversation_checkpoint_id" TEXT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS "conversation_execution_segment_id"
        TEXT DEFAULT NULL;
ALTER TABLE "interaction_continuations"
    DROP CONSTRAINT IF EXISTS "ck_interaction_continuations_conversation_pair";
ALTER TABLE "interaction_continuations"
    ADD CONSTRAINT "ck_interaction_continuations_conversation_pair"
    CHECK (
        ("conversation_checkpoint_id" IS NULL)
        = ("conversation_execution_segment_id" IS NULL)
    );
ALTER TABLE "interaction_continuations"
    DROP CONSTRAINT IF EXISTS "fk_interaction_continuations__conversation";
ALTER TABLE "interaction_continuations"
    ADD CONSTRAINT "fk_interaction_continuations__conversation"
    FOREIGN KEY (
        "conversation_checkpoint_id", "conversation_execution_segment_id"
    )
    REFERENCES "conversation_checkpoints" (
        "checkpoint_id", "execution_segment_id"
    );
""",
    """
CREATE TABLE IF NOT EXISTS "conversation_terminal_metadata" (
    "checkpoint_id" TEXT NOT NULL,
    "public_response_id" TEXT DEFAULT NULL,
    "terminal_state" TEXT NOT NULL,
    "terminal_at" TIMESTAMP WITH TIME ZONE NOT NULL,

    PRIMARY KEY ("checkpoint_id"),
    CONSTRAINT "ck_conversation_terminal_metadata_state"
        CHECK ("terminal_state" IN ('expired', 'tombstoned', 'deleted'))
);
""",
    """
CREATE OR REPLACE FUNCTION "conversation_payload_ref_increment"()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE "conversation_encrypted_payloads"
    SET "reference_count" = "reference_count" + 1
    WHERE "payload_id" = NEW."payload_id"
      AND "reference_count" = 0;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'conversation payload reference conflict'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
""",
    """
CREATE OR REPLACE FUNCTION "conversation_payload_ref_decrement"()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE "conversation_encrypted_payloads"
    SET "reference_count" = "reference_count" - 1
    WHERE "payload_id" = OLD."payload_id"
      AND "reference_count" = 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'conversation payload reference corruption'
            USING ERRCODE = '23514';
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "trg_conversation_payload_ref_increment"
    ON "conversation_checkpoint_payload_refs";
CREATE TRIGGER "trg_conversation_payload_ref_increment"
AFTER INSERT ON "conversation_checkpoint_payload_refs"
FOR EACH ROW EXECUTE FUNCTION "conversation_payload_ref_increment"();
""",
    """
DROP TRIGGER IF EXISTS "trg_conversation_payload_ref_decrement"
    ON "conversation_checkpoint_payload_refs";
CREATE TRIGGER "trg_conversation_payload_ref_decrement"
AFTER DELETE ON "conversation_checkpoint_payload_refs"
FOR EACH ROW EXECUTE FUNCTION "conversation_payload_ref_decrement"();
""",
    """
CREATE OR REPLACE FUNCTION "conversation_validate_payload_identity"()
RETURNS TRIGGER AS $$
DECLARE
    checkpoint_conversation_id TEXT;
    checkpoint_authority_digest TEXT;
    registered_algorithm TEXT;
    registered_lane_mode TEXT;
BEGIN
    SELECT c."conversation_id", c."authority_digest"
    INTO checkpoint_conversation_id, checkpoint_authority_digest
    FROM "conversation_checkpoints" AS c
    WHERE c."checkpoint_id" = NEW."checkpoint_id";
    IF NOT FOUND
       OR checkpoint_conversation_id <> NEW."conversation_id"
       OR checkpoint_authority_digest <> NEW."authority_digest" THEN
        RAISE EXCEPTION 'conversation payload checkpoint identity mismatch'
            USING ERRCODE = '23514';
    END IF;

    SELECT k."algorithm"
    INTO registered_algorithm
    FROM "conversation_key_revisions" AS k
    WHERE k."authority_digest" = NEW."authority_digest"
      AND k."key_id" = NEW."key_id"
      AND k."key_revision" = NEW."key_revision";
    IF NOT FOUND OR registered_algorithm <> NEW."algorithm" THEN
        RAISE EXCEPTION 'conversation payload key metadata mismatch'
            USING ERRCODE = '23514';
    END IF;

    IF NEW."payload_kind" = 'checkpoint' THEN
        IF NEW."lane_id" <> 'checkpoint-envelope'
           OR NEW."payload_sequence" <> 0 THEN
            RAISE EXCEPTION 'conversation checkpoint payload position invalid'
                USING ERRCODE = '23514';
        END IF;
    ELSIF NEW."payload_kind" = 'continuation_reference' THEN
        IF NEW."lane_id" <> 'structured-input'
           OR NEW."payload_sequence" <> 0 THEN
            RAISE EXCEPTION 'conversation continuation position invalid'
                USING ERRCODE = '23514';
        END IF;
    ELSE
        SELECT l."lane_mode"
        INTO registered_lane_mode
        FROM "conversation_lanes" AS l
        WHERE l."checkpoint_id" = NEW."checkpoint_id"
          AND l."lane_id" = NEW."lane_id";
        IF NOT FOUND THEN
            RAISE EXCEPTION 'conversation payload lane identity mismatch'
                USING ERRCODE = '23514';
        END IF;
        IF NEW."payload_kind" = 'deletion_target'
           AND (
               registered_lane_mode <> 'stored'
               OR NEW."payload_sequence" <> 0
           ) THEN
            RAISE EXCEPTION 'conversation deletion target position invalid'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "trg_conversation_payload_identity_valid"
    ON "conversation_encrypted_payloads";
CREATE TRIGGER "trg_conversation_payload_identity_valid"
BEFORE INSERT OR UPDATE ON "conversation_encrypted_payloads"
FOR EACH ROW EXECUTE FUNCTION "conversation_validate_payload_identity"();
""",
    """
CREATE OR REPLACE FUNCTION "conversation_reject_payload_identity_mutation"()
RETURNS TRIGGER AS $$
BEGIN
    IF ROW(
        OLD."payload_id", OLD."authority_digest", OLD."checkpoint_id",
        OLD."conversation_id", OLD."lane_id", OLD."payload_sequence",
        OLD."payload_kind", OLD."payload_schema_version",
        OLD."codec_version"
    ) IS DISTINCT FROM ROW(
        NEW."payload_id", NEW."authority_digest", NEW."checkpoint_id",
        NEW."conversation_id", NEW."lane_id", NEW."payload_sequence",
        NEW."payload_kind", NEW."payload_schema_version",
        NEW."codec_version"
    ) THEN
        RAISE EXCEPTION 'conversation payload identity is immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "trg_conversation_payload_identity_immutable"
    ON "conversation_encrypted_payloads";
CREATE TRIGGER "trg_conversation_payload_identity_immutable"
BEFORE UPDATE ON "conversation_encrypted_payloads"
FOR EACH ROW
EXECUTE FUNCTION "conversation_reject_payload_identity_mutation"();
""",
    """
CREATE OR REPLACE FUNCTION "conversation_reject_payload_ref_reassignment"()
RETURNS TRIGGER AS $$
BEGIN
    IF ROW(
        OLD."checkpoint_id", OLD."conversation_id",
        OLD."authority_digest", OLD."lane_id", OLD."payload_sequence",
        OLD."payload_kind", OLD."payload_schema_version",
        OLD."codec_version", OLD."payload_id"
    ) IS DISTINCT FROM ROW(
        NEW."checkpoint_id", NEW."conversation_id",
        NEW."authority_digest", NEW."lane_id", NEW."payload_sequence",
        NEW."payload_kind", NEW."payload_schema_version",
        NEW."codec_version", NEW."payload_id"
    ) THEN
        RAISE EXCEPTION 'conversation payload reference is immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "trg_conversation_payload_ref_immutable"
    ON "conversation_checkpoint_payload_refs";
CREATE TRIGGER "trg_conversation_payload_ref_immutable"
BEFORE UPDATE ON "conversation_checkpoint_payload_refs"
FOR EACH ROW
EXECUTE FUNCTION "conversation_reject_payload_ref_reassignment"();
""",
    """
CREATE OR REPLACE FUNCTION "conversation_retain_reconciliation_payload"()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM "conversation_reconciliation_outbox" AS o
        WHERE o."checkpoint_id" = OLD."checkpoint_id"
          AND o."work_state" <> 'completed'
    ) THEN
        RAISE EXCEPTION 'conversation reconciliation payload is retained'
            USING ERRCODE = '23514';
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "trg_conversation_payload_ref_retain"
    ON "conversation_checkpoint_payload_refs";
CREATE TRIGGER "trg_conversation_payload_ref_retain"
BEFORE DELETE ON "conversation_checkpoint_payload_refs"
FOR EACH ROW
EXECUTE FUNCTION "conversation_retain_reconciliation_payload"();
""",
    """
CREATE OR REPLACE FUNCTION "conversation_reject_checkpoint_identity_mutation"()
RETURNS TRIGGER AS $$
BEGIN
    IF ROW(
        OLD."checkpoint_id", OLD."conversation_id", OLD."logical_turn_id",
        OLD."execution_segment_id", OLD."branch_id",
        OLD."parent_checkpoint_id", OLD."checkpoint_sequence",
        OLD."parent_sequence", OLD."checkpoint_kind",
        OLD."authority_digest", OLD."checkpoint_codec_version",
        OLD."payload_schema_version", OLD."lane_count"
    ) IS DISTINCT FROM ROW(
        NEW."checkpoint_id", NEW."conversation_id", NEW."logical_turn_id",
        NEW."execution_segment_id", NEW."branch_id",
        NEW."parent_checkpoint_id", NEW."checkpoint_sequence",
        NEW."parent_sequence", NEW."checkpoint_kind",
        NEW."authority_digest", NEW."checkpoint_codec_version",
        NEW."payload_schema_version", NEW."lane_count"
    ) THEN
        RAISE EXCEPTION 'conversation checkpoint identity is immutable'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
""",
    """
DROP TRIGGER IF EXISTS "trg_conversation_checkpoint_identity_immutable"
    ON "conversation_checkpoints";
CREATE TRIGGER "trg_conversation_checkpoint_identity_immutable"
BEFORE UPDATE ON "conversation_checkpoints"
FOR EACH ROW
EXECUTE FUNCTION "conversation_reject_checkpoint_identity_mutation"();
""",
    """
CREATE INDEX IF NOT EXISTS "ix_conversation_checkpoints_authority_list"
    ON "conversation_checkpoints" (
        "authority_digest", "checkpoint_id"
    )
    WHERE "lifecycle_state" = 'committed';
""",
    """
CREATE INDEX IF NOT EXISTS "ix_conversation_checkpoints_parent"
    ON "conversation_checkpoints" (
        "parent_checkpoint_id", "checkpoint_id"
    )
    WHERE "parent_checkpoint_id" IS NOT NULL;
""",
    """
CREATE INDEX IF NOT EXISTS "ix_conversation_checkpoints_expiry"
    ON "conversation_checkpoints" ("expires_at", "checkpoint_id")
    WHERE "lifecycle_state" IN ('committed', 'expired');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_conversation_payloads_key_rotation"
    ON "conversation_encrypted_payloads" (
        "authority_digest", "key_id", "key_revision", "payload_id"
    );
""",
    """
CREATE INDEX IF NOT EXISTS "ix_conversation_payloads_gc"
    ON "conversation_encrypted_payloads" ("created_at", "payload_id")
    WHERE "reference_count" = 0;
""",
    """
CREATE INDEX IF NOT EXISTS "ix_conversation_idempotency_inflight"
    ON "conversation_idempotency" (
        "lease_expires_at", "authority_digest", "operation", "idempotency_key"
    )
    WHERE "record_state" IN ('in_progress', 'ambiguous');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_conversation_outbox_recovery"
    ON "conversation_outbox" ("available_sequence", "intent_id")
    WHERE "outbox_state" IN ('pending', 'claimed');
""",
    """
CREATE INDEX IF NOT EXISTS "ix_conversation_reconciliation_recovery"
    ON "conversation_reconciliation_outbox" (
        "created_at", "reconciliation_id"
    )
    WHERE "work_state" IN ('pending', 'claimed', 'failed');
""",
    """
CREATE OR REPLACE VIEW "conversation_store_readiness" AS
SELECT
    "schema_version",
    "minimum_reader_version",
    "maximum_reader_version",
    "minimum_writer_version",
    "maximum_writer_version",
    "checkpoint_codec_version"
FROM "conversation_store_metadata"
WHERE "singleton_id" = 1;
""",
)

TASK_SCHEMA_STATEMENTS = CONVERSATION_SCHEMA_STATEMENTS


def upgrade() -> None:
    """Apply the encrypted durable conversation schema."""
    _execute_all(TASK_SCHEMA_STATEMENTS)


def downgrade() -> None:
    """Reject reverse migration of encrypted conversation data."""
    raise NotImplementedError("task PostgreSQL migrations are forward-only")


def _execute_all(statements: Iterable[str]) -> None:
    alembic = cast(Any, import_module("alembic"))
    bind = alembic.op.get_bind()
    for statement in statements:
        bind.exec_driver_sql(statement)
