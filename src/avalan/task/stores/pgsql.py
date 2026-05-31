from ..store import TaskStoreConflictError, TaskStoreError

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import AsyncContextManager, Protocol


class PgsqlTaskMigrationCursor(Protocol):
    async def execute(
        self,
        query: str,
        parameters: tuple[object, ...] | None = None,
    ) -> None: ...

    async def fetchone(self) -> Mapping[str, object] | None: ...


class PgsqlTaskMigrationConnection(Protocol):
    def cursor(self) -> AsyncContextManager[PgsqlTaskMigrationCursor]: ...


class PgsqlTaskMigrationDatabase(Protocol):
    def connection(
        self,
    ) -> AsyncContextManager[PgsqlTaskMigrationConnection]: ...


class PgsqlTaskMigrationError(TaskStoreError):
    pass


def _assert_non_empty_string(value: str, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlTaskMigration:
    version: int
    name: str
    statements: tuple[str, ...]

    def __post_init__(self) -> None:
        assert isinstance(self.version, int)
        assert not isinstance(self.version, bool)
        assert self.version > 0
        _assert_non_empty_string(self.name, "name")
        assert isinstance(self.statements, tuple)
        assert self.statements
        for statement in self.statements:
            _assert_non_empty_string(statement, "statement")

    @property
    def checksum(self) -> str:
        return sha256("\n\n".join(self.statements).encode("utf-8")).hexdigest()


class PgsqlTaskMigrationRunner:
    def __init__(
        self,
        database: PgsqlTaskMigrationDatabase,
        *,
        migrations: tuple[PgsqlTaskMigration, ...] = (),
    ) -> None:
        assert hasattr(database, "connection")
        self._database = database
        self._migrations = migrations or TASK_PGSQL_MIGRATIONS
        assert self._migrations
        versions = [migration.version for migration in self._migrations]
        assert versions == sorted(versions)
        assert len(set(versions)) == len(versions)

    async def apply(self) -> tuple[PgsqlTaskMigration, ...]:
        applied: list[PgsqlTaskMigration] = []
        async with self._database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(_CREATE_MIGRATIONS_TABLE_SQL)
                for migration in self._migrations:
                    if await self._already_applied(cursor, migration):
                        continue
                    for statement in migration.statements:
                        await cursor.execute(statement)
                    await cursor.execute(
                        _INSERT_MIGRATION_SQL,
                        (
                            migration.version,
                            migration.name,
                            migration.checksum,
                        ),
                    )
                    applied.append(migration)
        return tuple(applied)

    async def _already_applied(
        self,
        cursor: PgsqlTaskMigrationCursor,
        migration: PgsqlTaskMigration,
    ) -> bool:
        await cursor.execute(_SELECT_MIGRATION_SQL, (migration.version,))
        row = await cursor.fetchone()
        if row is None:
            return False
        if row.get("checksum") != migration.checksum:
            raise TaskStoreConflictError(
                "task store migration checksum mismatch"
            )
        return True


TASK_PGSQL_MIGRATIONS: tuple[PgsqlTaskMigration, ...] = (
    PgsqlTaskMigration(
        version=1,
        name="task_lifecycle",
        statements=(
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
    CONSTRAINT "uq_task_definitions_identity"
        UNIQUE ("name", "version", "spec_hash")
);
""",
            """
CREATE TABLE IF NOT EXISTS "task_runs" (
    "run_id" TEXT NOT NULL,
    "definition_id" TEXT NOT NULL,
    "state" TEXT NOT NULL,
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
        CHECK ("updated_at" >= "created_at")
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
        CHECK (LENGTH(BTRIM("reason")) > 0)
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
        CHECK ("updated_at" >= "created_at")
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
    CONSTRAINT "ck_task_attempt_transitions_reason_non_empty"
        CHECK (LENGTH(BTRIM("reason")) > 0)
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
        CHECK ("updated_at" >= "created_at")
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
        ),
    ),
)

_CREATE_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS "task_schema_migrations" (
    "version" INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    "checksum" TEXT NOT NULL,
    "applied_at" TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY ("version"),
    CONSTRAINT "ck_task_schema_migrations_version_positive"
        CHECK ("version" > 0),
    CONSTRAINT "ck_task_schema_migrations_name_non_empty"
        CHECK (LENGTH(BTRIM("name")) > 0),
    CONSTRAINT "ck_task_schema_migrations_checksum_non_empty"
        CHECK (LENGTH(BTRIM("checksum")) > 0)
);
"""

_SELECT_MIGRATION_SQL = """
SELECT "version", "name", "checksum"
FROM "task_schema_migrations"
WHERE "version" = %s;
"""

_INSERT_MIGRATION_SQL = """
INSERT INTO "task_schema_migrations" ("version", "name", "checksum")
VALUES (%s, %s, %s);
"""


def task_pgsql_schema_statements() -> tuple[str, ...]:
    return (
        _CREATE_MIGRATIONS_TABLE_SQL,
        *(
            statement
            for migration in TASK_PGSQL_MIGRATIONS
            for statement in migration.statements
        ),
    )
