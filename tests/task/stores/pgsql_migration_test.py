from importlib import import_module
from os import environ
from sys import modules
from types import SimpleNamespace
from typing import cast
from unittest import TestCase, main
from uuid import uuid4

from pytest import importorskip

from avalan.task import TaskArtifactPurpose, TaskAttemptState, TaskRunState
from avalan.task.stores import (
    TASK_PGSQL_ALEMBIC_VERSION_TABLE,
    TASK_PGSQL_HEAD_REVISION,
    PgsqlTaskMigrationError,
    PgsqlTaskMigrationSettings,
    task_pgsql_alembic_config,
    task_pgsql_check,
    task_pgsql_claim_token_predicate,
    task_pgsql_current,
    task_pgsql_schema_statements,
    task_pgsql_script_location,
    task_pgsql_stamp,
    task_pgsql_state_predicate,
    task_pgsql_upgrade,
)


class FakeAlembicConfig:
    def __init__(self) -> None:
        self.options: dict[str, str] = {}
        self.attributes: dict[str, object] = {}

    def set_main_option(self, name: str, value: str) -> None:
        self.options[name] = value


class FakeAlembicConfigModule:
    def __init__(self) -> None:
        self.configs: list[FakeAlembicConfig] = []

    def Config(self) -> FakeAlembicConfig:
        config = FakeAlembicConfig()
        self.configs.append(config)
        return config


class FakeAlembicCommandModule:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = (
            []
        )

    def upgrade(self, config: object, revision: str) -> None:
        self.calls.append(("upgrade", (config, revision), {}))

    def current(self, config: object, **kwargs: object) -> None:
        self.calls.append(("current", (config,), kwargs))

    def stamp(self, config: object, revision: str) -> None:
        self.calls.append(("stamp", (config, revision), {}))


class FakeAlembicModules:
    def __init__(self) -> None:
        self.config = FakeAlembicConfigModule()
        self.command = FakeAlembicCommandModule()

    def module_finder(self, module: str) -> object | None:
        if module in {"alembic", "sqlalchemy"}:
            return object()
        return None

    def module_importer(self, module: str) -> object:
        if module == "alembic.config":
            return self.config
        if module == "alembic.command":
            return self.command
        raise AssertionError(f"unexpected module import: {module}")


class FakeAlembicEnvironmentConfig:
    config_ini_section = "alembic"

    def __init__(
        self,
        *,
        options: dict[str, str] | None = None,
        attributes: dict[str, object] | None = None,
    ) -> None:
        self.options = options or {}
        self.attributes = attributes or {}

    def get_main_option(self, name: str) -> str | None:
        return self.options.get(name)

    def get_section(
        self,
        name: str,
        default: dict[str, object],
    ) -> dict[str, object]:
        return {"sqlalchemy.url": "postgresql://localhost/avalan"}


class FakeTransaction:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeAlembicEnvironmentContext:
    def __init__(
        self,
        *,
        offline: bool,
        config: FakeAlembicEnvironmentConfig,
    ) -> None:
        self.config = config
        self.offline = offline
        self.configure_kwargs: dict[str, object] | None = None
        self.ran_migrations = False

    def is_offline_mode(self) -> bool:
        return self.offline

    def configure(self, **kwargs: object) -> None:
        self.configure_kwargs = kwargs

    def begin_transaction(self) -> FakeTransaction:
        return FakeTransaction()

    def run_migrations(self) -> None:
        self.ran_migrations = True


class FakeConnectionContext:
    def __init__(self, connection: "FakeSqlalchemyConnection") -> None:
        self.connection = connection

    def __enter__(self) -> "FakeSqlalchemyConnection":
        return self.connection

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        return False


class FakeSqlalchemyConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[object, object | None]] = []

    def execute(
        self,
        statement: object,
        parameters: object | None = None,
    ) -> None:
        self.executed.append((statement, parameters))


class FakeSqlalchemyConnectable:
    def __init__(self, connection: FakeSqlalchemyConnection) -> None:
        self.connection = connection

    def connect(self) -> FakeConnectionContext:
        return FakeConnectionContext(self.connection)


class FakeAlembicBind:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def exec_driver_sql(self, statement: str) -> None:
        self.statements.append(statement)


class FakeRevisionOp:
    def __init__(self) -> None:
        self.bind = FakeAlembicBind()

    def get_bind(self) -> FakeAlembicBind:
        return self.bind


class PgsqlMigrationSchemaTest(TestCase):
    def test_task_schema_has_expected_tables_and_constraints(self) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for table_name in (
            "task_definitions",
            "task_runs",
            "task_run_transitions",
            "task_attempts",
            "task_attempt_transitions",
            "task_artifacts",
            "task_artifact_bytes",
            "task_idempotency_keys",
            "task_queue_items",
            "task_events",
            "task_usage_records",
            "task_run_rollups",
        ):
            self.assertIn(f'"{table_name}"', schema)

        self.assertNotIn("task_schema_migrations", schema)
        self.assertIn('"uq_task_definitions_identity"', schema)
        self.assertIn('"uq_task_attempts_run_order"', schema)
        self.assertIn('"uq_task_attempts_one_active_per_run"', schema)
        self.assertIn('"uq_task_idempotency_keys_identity"', schema)
        self.assertIn('"uq_task_events_run_sequence"', schema)
        self.assertIn(
            "WHERE \"state\" NOT IN ('succeeded', 'failed', 'abandoned')",
            schema,
        )
        self.assertIn('"task_reject_terminal_run_state_change"', schema)
        self.assertIn('"tr_task_runs_terminal_state"', schema)

    def test_schema_covers_current_state_and_artifact_vocabularies(
        self,
    ) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for state in TaskRunState:
            self.assertIn(f"'{state.value}'", schema)
        for state in TaskAttemptState:
            self.assertIn(f"'{state.value}'", schema)
        for purpose in TaskArtifactPurpose:
            self.assertIn(f"'{purpose.value}'", schema)

    def test_schema_covers_idempotency_and_byte_storage_contracts(
        self,
    ) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for column_name in (
            "owner_scope_hash",
            "window_hash",
            "input_hash",
            "file_hash",
            "custom_hash",
            "ciphertext",
            "encryption_key_id",
            "retention_days",
        ):
            self.assertIn(f'"{column_name}"', schema)
        self.assertIn('"fk_task_idempotency_keys__task_runs"', schema)
        self.assertIn("'input_hash'", schema)
        self.assertIn("'input_and_files_hash'", schema)
        self.assertIn("'custom'", schema)
        self.assertIn('"ck_task_artifact_bytes_retention_positive"', schema)

    def test_state_and_claim_predicates_are_parameterized(self) -> None:
        state_sql, state_params = task_pgsql_state_predicate(
            "state",
            {TaskRunState.QUEUED, TaskRunState.CLAIMED},
            table_alias="r",
        )
        no_claim_sql, no_claim_params = task_pgsql_claim_token_predicate(
            "claim",
            None,
            table_alias="r",
        )
        claim_sql, claim_params = task_pgsql_claim_token_predicate(
            "claim",
            "claim-secret",
        )

        self.assertEqual(state_sql, '"r"."state" IN (%s, %s)')
        self.assertEqual(set(state_params), {"queued", "claimed"})
        self.assertEqual(no_claim_sql, '"r"."claim" IS NULL')
        self.assertEqual(no_claim_params, ())
        self.assertEqual(claim_sql, "\"claim\" ->> 'claim_token' = %s")
        self.assertEqual(claim_params, ("claim-secret",))
        self.assertNotIn("claim-secret", claim_sql)

    def test_state_and_claim_predicates_reject_unsafe_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            task_pgsql_state_predicate("state;drop", {TaskRunState.QUEUED})
        with self.assertRaises(AssertionError):
            task_pgsql_state_predicate("state", set())
        with self.assertRaises(AssertionError):
            task_pgsql_state_predicate(
                "state",
                cast(set[TaskRunState], {"queued"}),
            )
        with self.assertRaises(AssertionError):
            task_pgsql_claim_token_predicate("claim", "")
        with self.assertRaises(AssertionError):
            task_pgsql_claim_token_predicate(
                "claim",
                "token",
                table_alias="bad-alias",
            )


class PgsqlMigrationHelperTest(TestCase):
    def test_builds_alembic_config_with_schema_and_metadata(self) -> None:
        modules = FakeAlembicModules()
        settings = PgsqlTaskMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            schema="task_schema",
            module_finder=modules.module_finder,
            module_importer=modules.module_importer,
            attributes={"connection_name": "test"},
        )

        config = cast(
            FakeAlembicConfig,
            task_pgsql_alembic_config(settings),
        )

        self.assertEqual(
            config.options["script_location"],
            task_pgsql_script_location(),
        )
        self.assertEqual(
            config.options["version_table"],
            TASK_PGSQL_ALEMBIC_VERSION_TABLE,
        )
        self.assertEqual(config.options["task_schema"], "task_schema")
        self.assertEqual(
            config.options["version_table_schema"],
            "task_schema",
        )
        self.assertEqual(config.attributes["connection_name"], "test")

    def test_missing_dependencies_raise_stable_diagnostic(self) -> None:
        settings = PgsqlTaskMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=lambda module: None,
            module_importer=_unexpected_import,
        )

        with self.assertRaisesRegex(
            PgsqlTaskMigrationError,
            "dependency.task_pgsql_migrations_missing",
        ):
            task_pgsql_alembic_config(settings)

    def test_helpers_dispatch_to_alembic_commands(self) -> None:
        modules = FakeAlembicModules()
        settings = PgsqlTaskMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=modules.module_finder,
            module_importer=modules.module_importer,
        )

        task_pgsql_upgrade(settings)
        task_pgsql_current(settings, verbose=True)
        task_pgsql_check(settings)
        task_pgsql_stamp(settings, revision=TASK_PGSQL_HEAD_REVISION)

        self.assertEqual(
            [
                (name, args[1:] if len(args) > 1 else (), kwargs)
                for name, args, kwargs in modules.command.calls
            ],
            [
                ("upgrade", ("head",), {}),
                ("current", (), {"verbose": True}),
                ("current", (), {"check_heads": True}),
                ("stamp", (TASK_PGSQL_HEAD_REVISION,), {}),
            ],
        )

    def test_invalid_helper_settings_fail_fast(self) -> None:
        modules = FakeAlembicModules()

        for factory in (
            lambda: PgsqlTaskMigrationSettings(url=""),
            lambda: PgsqlTaskMigrationSettings(
                url="postgresql+psycopg://localhost/avalan",
                schema="bad-name",
            ),
            lambda: PgsqlTaskMigrationSettings(
                url="postgresql+psycopg://localhost/avalan",
                version_table="1bad",
            ),
        ):
            with self.assertRaises(AssertionError):
                factory()

        settings = PgsqlTaskMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=modules.module_finder,
            module_importer=modules.module_importer,
        )
        with self.assertRaises(AssertionError):
            task_pgsql_upgrade(settings, revision="head;drop")

    def test_upgrade_real_postgresql_when_configured(self) -> None:
        dsn = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
        if not dsn:
            self.skipTest("AVALAN_TASK_TEST_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("sqlalchemy")
        schema = f"avalan_task_test_{uuid4().hex}"

        settings = PgsqlTaskMigrationSettings(
            url=dsn,
            schema=schema,
        )
        task_pgsql_upgrade(settings)
        task_pgsql_check(settings)


class PgsqlMigrationEnvironmentTest(TestCase):
    def test_offline_environment_uses_default_version_table(self) -> None:
        context = FakeAlembicEnvironmentContext(
            offline=True,
            config=FakeAlembicEnvironmentConfig(),
        )

        self._import_env(context=context)

        self.assertTrue(context.ran_migrations)
        assert context.configure_kwargs is not None
        self.assertEqual(
            context.configure_kwargs["version_table"],
            TASK_PGSQL_ALEMBIC_VERSION_TABLE,
        )
        self.assertEqual(
            context.configure_kwargs["version_table_schema"], None
        )

    def test_online_environment_prepares_schema_and_lock(self) -> None:
        connection = FakeSqlalchemyConnection()
        context = FakeAlembicEnvironmentContext(
            offline=False,
            config=FakeAlembicEnvironmentConfig(
                options={
                    "task_advisory_lock_id": "42",
                    "version_table": "custom_version_table",
                },
                attributes={"task_schema": "task_schema"},
            ),
        )

        self._import_env(context=context, connection=connection)

        self.assertTrue(context.ran_migrations)
        assert context.configure_kwargs is not None
        self.assertTrue(context.configure_kwargs["include_schemas"])
        self.assertEqual(
            context.configure_kwargs["version_table"],
            "custom_version_table",
        )
        self.assertEqual(
            connection.executed,
            [
                ('CREATE SCHEMA IF NOT EXISTS "task_schema"', None),
                ('SET search_path TO "task_schema"', None),
                (
                    "SELECT pg_advisory_xact_lock(:lock_id)",
                    {"lock_id": 42},
                ),
            ],
        )

    def _import_env(
        self,
        *,
        context: FakeAlembicEnvironmentContext,
        connection: FakeSqlalchemyConnection | None = None,
    ) -> None:
        module_name = "avalan.task.stores.pgsql_migrations.env"
        old_alembic = modules.get("alembic")
        old_sqlalchemy = modules.get("sqlalchemy")
        old_env = modules.pop(module_name, None)
        fake_connection = connection or FakeSqlalchemyConnection()
        modules["alembic"] = SimpleNamespace(context=context)
        modules["sqlalchemy"] = SimpleNamespace(
            engine_from_config=lambda *args, **kwargs: (
                FakeSqlalchemyConnectable(
                    fake_connection,
                )
            ),
            pool=SimpleNamespace(NullPool=object),
            text=lambda value: value,
        )
        try:
            import_module(module_name)
        finally:
            modules.pop(module_name, None)
            if old_env is not None:
                modules[module_name] = old_env
            if old_alembic is None:
                modules.pop("alembic", None)
            else:
                modules["alembic"] = old_alembic
            if old_sqlalchemy is None:
                modules.pop("sqlalchemy", None)
            else:
                modules["sqlalchemy"] = old_sqlalchemy


class PgsqlMigrationRevisionTest(TestCase):
    def test_revision_upgrade_executes_schema_statements(self) -> None:
        revision_module = import_module(
            "avalan.task.stores.pgsql_migrations.versions."
            "v20260530_0001_task_schema"
        )
        fake_op = FakeRevisionOp()
        old_alembic = modules.get("alembic")
        modules["alembic"] = SimpleNamespace(op=fake_op)
        try:
            revision_module.upgrade()
        finally:
            if old_alembic is None:
                modules.pop("alembic", None)
            else:
                modules["alembic"] = old_alembic

        self.assertEqual(
            fake_op.bind.statements,
            list(task_pgsql_schema_statements()),
        )

    def test_revision_downgrade_is_forward_only(self) -> None:
        revision_module = import_module(
            "avalan.task.stores.pgsql_migrations.versions."
            "v20260530_0001_task_schema"
        )

        with self.assertRaises(NotImplementedError):
            revision_module.downgrade()


def _unexpected_import(module: str) -> object:
    raise AssertionError(f"unexpected module import: {module}")


if __name__ == "__main__":
    main()
