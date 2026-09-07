"""Install immutable trigger decisions and shared physical artifact owners."""

from alembic import op

revision = "20260907_0002_triggers"
down_revision = "20260907_0001_task_submissions"
branch_labels = None
depends_on = None

TASK_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
ALTER TABLE task_runs ADD CONSTRAINT uq_task_runs_definition
    UNIQUE (run_id, definition_id);
ALTER TABLE task_submissions ADD COLUMN created_run BOOLEAN
    GENERATED ALWAYS AS ((payload->'payload'->>'created')::boolean) STORED;
ALTER TABLE task_submissions ADD CONSTRAINT uq_task_submissions_created_run
    UNIQUE (owner_scope, submission_id, run_id, created_run);
""",
    """
CREATE TABLE trigger_definitions (
    owner_scope_id TEXT NOT NULL CHECK (length(btrim(owner_scope_id)) > 0),
    trigger_id TEXT NOT NULL CHECK (length(btrim(trigger_id)) > 0),
    revision BIGINT NOT NULL CHECK (revision > 0),
    task_definition_id TEXT NOT NULL REFERENCES
        task_definitions(definition_id),
    payload JSONB NOT NULL,
    registered_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    PRIMARY KEY (owner_scope_id, trigger_id, revision),
    UNIQUE (owner_scope_id, trigger_id, revision, task_definition_id),
    CHECK (closed_at IS NULL OR closed_at >= registered_at),
    CHECK (COALESCE(
        payload->>'format' = 'avalan.trigger.definition'
        AND payload->'version' = '1'::jsonb
        AND payload->'payload'->>'owner_scope_id' = owner_scope_id
        AND payload->'payload'->>'trigger_id' = trigger_id
        AND payload->'payload'->'revision' = to_jsonb(revision)
        AND payload->'payload'->>'task_definition_id' = task_definition_id,
        FALSE))
);
CREATE TABLE triggers (
    ordinal BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    owner_scope_id TEXT NOT NULL,
    trigger_id TEXT NOT NULL,
    name TEXT NOT NULL CHECK (name ~ '^[a-z][a-z0-9-]{0,62}$'),
    revision BIGINT NOT NULL CHECK (revision > 0),
    generation BIGINT NOT NULL CHECK (generation > 0),
    status TEXT NOT NULL CHECK (status IN ('active', 'paused', 'exhausted',
        'error')),
    next_at TIMESTAMPTZ,
    retry_after TIMESTAMPTZ,
    failure_count INTEGER NOT NULL CHECK (failure_count BETWEEN 0 AND 20),
    last_processed_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (owner_scope_id, trigger_id),
    UNIQUE (owner_scope_id, name),
    FOREIGN KEY (owner_scope_id, trigger_id, revision)
        REFERENCES trigger_definitions(owner_scope_id, trigger_id, revision)
        DEFERRABLE INITIALLY DEFERRED,
    CHECK ((status = 'exhausted') = (next_at IS NULL)),
    CHECK (retry_after IS NULL OR (failure_count > 0 AND status <>
        'exhausted')),
    CHECK (COALESCE(
        payload->>'format' = 'avalan.trigger.state'
        AND payload->'version' = '1'::jsonb
        AND payload->'payload'->>'owner_scope_id' = owner_scope_id
        AND payload->'payload'->>'trigger_id' = trigger_id
        AND payload->'payload'->'revision' = to_jsonb(revision)
        AND payload->'payload'->'generation' = to_jsonb(generation)
        AND payload->'payload'->>'status' = status,
        FALSE))
);
ALTER TABLE trigger_definitions ADD CONSTRAINT fk_trigger_definition_root
    FOREIGN KEY (owner_scope_id, trigger_id)
    REFERENCES triggers(owner_scope_id, trigger_id)
    DEFERRABLE INITIALLY DEFERRED;
CREATE INDEX ix_triggers_due ON triggers
    (owner_scope_id, last_processed_at, trigger_id, next_at, retry_after)
    WHERE status = 'active';
""",
    """
CREATE TABLE trigger_registration_requests (
    owner_scope_id TEXT NOT NULL,
    request_id UUID NOT NULL,
    fingerprint TEXT NOT NULL CHECK (fingerprint ~ '^[0-9a-f]{64}$'),
    trigger_id TEXT NOT NULL,
    revision BIGINT NOT NULL,
    definition JSONB NOT NULL,
    state JSONB NOT NULL,
    PRIMARY KEY (owner_scope_id, request_id),
    FOREIGN KEY (owner_scope_id, trigger_id, revision)
        REFERENCES trigger_definitions(owner_scope_id, trigger_id, revision)
);
CREATE TABLE trigger_occurrences (
    ordinal BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    owner_scope_id TEXT NOT NULL,
    trigger_id TEXT NOT NULL,
    revision BIGINT NOT NULL,
    occurrence_id UUID NOT NULL UNIQUE,
    scheduled_at TIMESTAMPTZ NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL CHECK (decided_at >= scheduled_at),
    disposition TEXT NOT NULL CHECK (disposition IN (
        'admitted', 'expired', 'skipped_misfire', 'skipped_overlap',
        'coalesced', 'superseded')),
    run_id TEXT UNIQUE,
    submission_id TEXT,
    task_definition_id TEXT NOT NULL,
    created_run BOOLEAN GENERATED ALWAYS AS (run_id IS NOT NULL) STORED,
    payload JSONB NOT NULL,
    PRIMARY KEY (owner_scope_id, trigger_id, revision, scheduled_at),
    FOREIGN KEY (owner_scope_id, trigger_id, revision, task_definition_id)
        REFERENCES trigger_definitions(owner_scope_id, trigger_id,
        revision, task_definition_id),
    FOREIGN KEY (run_id, task_definition_id)
        REFERENCES task_runs(run_id, definition_id),
    FOREIGN KEY (owner_scope_id, submission_id, run_id, created_run)
        REFERENCES task_submissions(owner_scope, submission_id, run_id,
        created_run),
    CHECK ((disposition = 'admitted') = (run_id IS NOT NULL)),
    CHECK ((run_id IS NULL) = (submission_id IS NULL)),
    CHECK (COALESCE(
        payload->>'format' = 'avalan.trigger.occurrence'
        AND payload->'version' = '1'::jsonb
        AND payload->'payload'->>'owner_scope_id' = owner_scope_id
        AND payload->'payload'->>'trigger_id' = trigger_id
        AND payload->'payload'->'revision' = to_jsonb(revision)
        AND payload->'payload'->>'occurrence_id' = occurrence_id::text
        AND payload->'payload'->>'disposition' = disposition
        AND (payload->'payload'->>'run_id') IS NOT DISTINCT FROM run_id,
        FALSE))
);
CREATE INDEX ix_trigger_occurrences_history ON trigger_occurrences
    (owner_scope_id, trigger_id, ordinal);
CREATE INDEX ix_trigger_occurrences_outstanding ON trigger_occurrences
    (owner_scope_id, trigger_id, run_id) WHERE run_id IS NOT NULL;
CREATE TABLE trigger_coverage_spans (
    ordinal BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    owner_scope_id TEXT NOT NULL,
    trigger_id TEXT NOT NULL,
    revision BIGINT NOT NULL,
    span_id UUID NOT NULL UNIQUE,
    first_at TIMESTAMPTZ NOT NULL,
    until_at TIMESTAMPTZ NOT NULL CHECK (until_at > first_at),
    decided_at TIMESTAMPTZ NOT NULL CHECK (decided_at >= first_at),
    disposition TEXT NOT NULL CHECK (disposition IN (
        'expired', 'skipped_misfire', 'skipped_overlap', 'coalesced',
        'superseded')),
    exact_count BIGINT CHECK (exact_count > 0),
    payload JSONB NOT NULL,
    PRIMARY KEY (owner_scope_id, trigger_id, revision, span_id),
    FOREIGN KEY (owner_scope_id, trigger_id, revision)
        REFERENCES trigger_definitions(owner_scope_id, trigger_id, revision),
    CHECK (COALESCE(
        payload->>'format' = 'avalan.trigger.span'
        AND payload->'version' = '1'::jsonb
        AND payload->'payload'->>'owner_scope_id' = owner_scope_id
        AND payload->'payload'->>'trigger_id' = trigger_id
        AND payload->'payload'->'revision' = to_jsonb(revision)
        AND payload->'payload'->>'span_id' = span_id::text
        AND payload->'payload'->>'disposition' = disposition,
        FALSE))
);
CREATE INDEX ix_trigger_spans_range ON trigger_coverage_spans
    (owner_scope_id, trigger_id, revision, first_at, until_at);
CREATE INDEX ix_trigger_spans_history ON trigger_coverage_spans
    (owner_scope_id, trigger_id, ordinal);
CREATE TABLE trigger_events (
    ordinal BIGINT GENERATED ALWAYS AS IDENTITY UNIQUE,
    owner_scope_id TEXT NOT NULL,
    trigger_id TEXT NOT NULL,
    revision BIGINT NOT NULL,
    event_id UUID PRIMARY KEY,
    recorded_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL,
    FOREIGN KEY (owner_scope_id, trigger_id, revision)
        REFERENCES trigger_definitions(owner_scope_id, trigger_id, revision),
    CHECK (COALESCE(
        payload->>'format' = 'avalan.trigger.event'
        AND payload->'version' = '1'::jsonb
        AND payload->'payload'->>'owner_scope_id' = owner_scope_id
        AND payload->'payload'->>'trigger_id' = trigger_id
        AND payload->'payload'->'revision' = to_jsonb(revision)
        AND payload->'payload'->>'event_id' = event_id::text,
        FALSE))
);
CREATE INDEX ix_trigger_events_history ON trigger_events
    (owner_scope_id, trigger_id, ordinal);
""",
    """
CREATE TABLE task_artifact_objects (
    object_id UUID PRIMARY KEY,
    store TEXT NOT NULL CHECK (length(btrim(store)) > 0),
    storage_key TEXT NOT NULL CHECK (length(btrim(storage_key)) > 0),
    sha256 TEXT NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    status TEXT NOT NULL CHECK (status IN ('live', 'deleting', 'deleted')),
    created_at TIMESTAMPTZ NOT NULL,
    cleanup_token UUID,
    staged_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    was_staged BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (store, storage_key),
    CHECK ((status = 'live') = (cleanup_token IS NULL))
);
CREATE TABLE task_artifact_staging_owners (
    staging_id UUID NOT NULL,
    object_id UUID NOT NULL REFERENCES task_artifact_objects(object_id),
    owner_scope_id TEXT NOT NULL CHECK (length(btrim(owner_scope_id)) > 0),
    recovery_id TEXT NOT NULL CHECK (length(btrim(recovery_id)) > 0),
    outcome TEXT NOT NULL CHECK (outcome IN ('unknown', 'committed',
        'not_committed')),
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (staging_id, object_id)
);
CREATE INDEX ix_task_artifact_staging_recovery ON task_artifact_staging_owners
    (owner_scope_id, recovery_id);
CREATE TABLE trigger_input_artifacts (
    owner_scope_id TEXT NOT NULL,
    trigger_id TEXT NOT NULL,
    revision BIGINT NOT NULL,
    object_id UUID NOT NULL REFERENCES task_artifact_objects(object_id),
    ref JSONB NOT NULL CHECK (jsonb_typeof(ref) = 'object'),
    released_at TIMESTAMPTZ,
    PRIMARY KEY (owner_scope_id, trigger_id, revision, object_id),
    FOREIGN KEY (owner_scope_id, trigger_id, revision)
        REFERENCES trigger_definitions(owner_scope_id, trigger_id, revision)
);
CREATE INDEX ix_trigger_input_live_owners ON trigger_input_artifacts(object_id)
    WHERE released_at IS NULL;
CREATE TABLE task_artifact_run_owners (
    artifact_id TEXT PRIMARY KEY REFERENCES task_artifacts(artifact_id),
    object_id UUID NOT NULL REFERENCES task_artifact_objects(object_id),
    released_at TIMESTAMPTZ
);
CREATE INDEX ix_task_artifact_live_owners ON
        task_artifact_run_owners(object_id)
    WHERE released_at IS NULL;
""",
    """
CREATE FUNCTION reject_trigger_history_mutation() RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'trigger history is immutable' USING ERRCODE = '23514';
END;
$$;
CREATE TRIGGER immutable_trigger_registration
    BEFORE UPDATE OR DELETE ON trigger_registration_requests
    FOR EACH ROW EXECUTE FUNCTION reject_trigger_history_mutation();
CREATE TRIGGER immutable_trigger_occurrence
    BEFORE UPDATE OR DELETE ON trigger_occurrences
    FOR EACH ROW EXECUTE FUNCTION reject_trigger_history_mutation();
CREATE TRIGGER immutable_trigger_span
    BEFORE UPDATE OR DELETE ON trigger_coverage_spans
    FOR EACH ROW EXECUTE FUNCTION reject_trigger_history_mutation();
CREATE TRIGGER immutable_trigger_event
    BEFORE UPDATE OR DELETE ON trigger_events
    FOR EACH ROW EXECUTE FUNCTION reject_trigger_history_mutation();
CREATE FUNCTION check_trigger_revision_closure() RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'trigger revision is retained' USING ERRCODE = '23514';
    END IF;
    IF OLD.closed_at IS NOT NULL OR NEW.closed_at IS NULL
        OR (to_jsonb(OLD) - 'closed_at') <> (to_jsonb(NEW) - 'closed_at') THEN
        RAISE EXCEPTION 'trigger revision is immutable' USING ERRCODE =
        '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER immutable_trigger_revision
    BEFORE UPDATE OR DELETE ON trigger_definitions
    FOR EACH ROW EXECUTE FUNCTION check_trigger_revision_closure();
CREATE FUNCTION check_trigger_generation() RETURNS TRIGGER
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.owner_scope_id <> OLD.owner_scope_id
        OR NEW.trigger_id <> OLD.trigger_id OR NEW.name <> OLD.name
        OR NEW.ordinal <> OLD.ordinal
        OR NEW.generation <> OLD.generation + 1
        OR NEW.revision NOT IN (OLD.revision, OLD.revision + 1) THEN
        RAISE EXCEPTION 'invalid trigger generation' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER checked_trigger_generation BEFORE UPDATE ON triggers
    FOR EACH ROW EXECUTE FUNCTION check_trigger_generation();
CREATE FUNCTION require_live_artifact_object() RETURNS TRIGGER
LANGUAGE plpgsql AS $$
DECLARE object_status TEXT;
BEGIN
    SELECT status INTO object_status FROM task_artifact_objects
        WHERE object_id = NEW.object_id FOR UPDATE;
    IF object_status IS DISTINCT FROM 'live' THEN
        RAISE EXCEPTION 'artifact is unavailable' USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER check_trigger_artifact_owner BEFORE INSERT ON
        trigger_input_artifacts
    FOR EACH ROW EXECUTE FUNCTION require_live_artifact_object();
CREATE TRIGGER check_run_artifact_owner BEFORE INSERT ON
        task_artifact_run_owners
    FOR EACH ROW EXECUTE FUNCTION require_live_artifact_object();
CREATE TRIGGER check_staging_artifact_owner BEFORE INSERT ON
        task_artifact_staging_owners
    FOR EACH ROW EXECUTE FUNCTION require_live_artifact_object();
""",
)


def upgrade() -> None:
    """Install the final trigger and physical ownership tables."""
    bind = op.get_bind()
    for statement in TASK_SCHEMA_STATEMENTS:
        bind.exec_driver_sql(statement)


def downgrade() -> None:
    """Reject removal of durable occurrence deduplication evidence."""
    raise NotImplementedError("task PostgreSQL migrations are forward-only")
