from importlib import import_module
from os import environ
from sys import modules
from types import SimpleNamespace
from typing import cast
from unittest import TestCase, main
from uuid import uuid4

from pytest import importorskip

from avalan.memory.permanent.pgsql_migrations import (
    MEMORY_PGSQL_ADVISORY_LOCK_ID,
    MEMORY_PGSQL_ALEMBIC_VERSION_TABLE,
    MEMORY_PGSQL_HEAD_REVISION,
    PgsqlMemoryMigrationError,
    PgsqlMemoryMigrationSettings,
    memory_pgsql_alembic_config,
    memory_pgsql_check,
    memory_pgsql_current,
    memory_pgsql_schema_statements,
    memory_pgsql_script_location,
    memory_pgsql_stamp,
    memory_pgsql_upgrade,
)
from avalan.task.stores import task_pgsql_schema_statements


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


class PgsqlMemoryMigrationSchemaTest(TestCase):
    def test_memory_schema_has_expected_tables_and_extensions(self) -> None:
        schema = "\n".join(memory_pgsql_schema_statements())

        for table_name in (
            "sessions",
            "messages",
            "message_partitions",
            "memories",
            "memory_partitions",
            "hyperedges",
            "hyperedges_memories",
            "entities",
            "hyperedge_entities",
        ):
            self.assertIn(f'"{table_name}"', schema)

        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", schema)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS ltree", schema)
        self.assertIn("CREATE EXTENSION IF NOT EXISTS pg_trgm", schema)
        self.assertIn('"message_author_type"', schema)
        self.assertIn('"memory_types"', schema)
        self.assertIn('"v_live_hyperedges"', schema)

    def test_pg_trgm_is_declared_before_trigram_index(self) -> None:
        schema = "\n".join(memory_pgsql_schema_statements())

        self.assertLess(
            schema.index("CREATE EXTENSION IF NOT EXISTS pg_trgm"),
            schema.index("gin_trgm_ops"),
        )

    def test_task_schema_stays_independent_from_memory_extensions(
        self,
    ) -> None:
        schema = "\n".join(task_pgsql_schema_statements())

        for memory_only_token in (
            "VECTOR(",
            "pg_trgm",
            "gin_trgm_ops",
            "message_author_type",
            "memory_types",
        ):
            self.assertNotIn(memory_only_token, schema)


class PgsqlMemoryMigrationHelperTest(TestCase):
    def test_builds_alembic_config_with_schema_and_metadata(self) -> None:
        fake_modules = FakeAlembicModules()
        settings = PgsqlMemoryMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            schema="memory_schema",
            module_finder=fake_modules.module_finder,
            module_importer=fake_modules.module_importer,
            attributes={"connection_name": "test"},
        )

        config = cast(
            FakeAlembicConfig,
            memory_pgsql_alembic_config(settings),
        )

        self.assertEqual(
            config.options["script_location"],
            memory_pgsql_script_location(),
        )
        self.assertEqual(
            config.options["version_table"],
            MEMORY_PGSQL_ALEMBIC_VERSION_TABLE,
        )
        self.assertEqual(config.options["memory_schema"], "memory_schema")
        self.assertEqual(
            config.options["version_table_schema"],
            "memory_schema",
        )
        self.assertEqual(config.attributes["connection_name"], "test")

    def test_missing_dependencies_raise_stable_diagnostic(self) -> None:
        settings = PgsqlMemoryMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=lambda module: None,
            module_importer=_unexpected_import,
        )

        with self.assertRaisesRegex(
            PgsqlMemoryMigrationError,
            "dependency.memory_pgsql_migrations_missing",
        ):
            memory_pgsql_alembic_config(settings)

    def test_helpers_dispatch_to_alembic_commands(self) -> None:
        fake_modules = FakeAlembicModules()
        settings = PgsqlMemoryMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=fake_modules.module_finder,
            module_importer=fake_modules.module_importer,
        )

        memory_pgsql_upgrade(settings)
        memory_pgsql_current(settings, verbose=True)
        memory_pgsql_check(settings)
        memory_pgsql_stamp(settings, revision=MEMORY_PGSQL_HEAD_REVISION)

        self.assertEqual(
            [
                (name, args[1:] if len(args) > 1 else (), kwargs)
                for name, args, kwargs in fake_modules.command.calls
            ],
            [
                ("upgrade", ("head",), {}),
                ("current", (), {"verbose": True}),
                ("current", (), {"check_heads": True}),
                ("stamp", (MEMORY_PGSQL_HEAD_REVISION,), {}),
            ],
        )

    def test_invalid_helper_settings_fail_fast(self) -> None:
        fake_modules = FakeAlembicModules()

        for factory in (
            lambda: PgsqlMemoryMigrationSettings(url=""),
            lambda: PgsqlMemoryMigrationSettings(
                url="postgresql+psycopg://localhost/avalan",
                schema="bad-name",
            ),
            lambda: PgsqlMemoryMigrationSettings(
                url="postgresql+psycopg://localhost/avalan",
                version_table="1bad",
            ),
        ):
            with self.assertRaises(AssertionError):
                factory()

        settings = PgsqlMemoryMigrationSettings(
            url="postgresql+psycopg://localhost/avalan",
            module_finder=fake_modules.module_finder,
            module_importer=fake_modules.module_importer,
        )
        with self.assertRaises(AssertionError):
            memory_pgsql_upgrade(settings, revision="head;drop")

    def test_upgrade_real_postgresql_when_configured(self) -> None:
        dsn = environ.get("AVALAN_MEMORY_TEST_POSTGRESQL_DSN")
        if not dsn:
            self.skipTest("AVALAN_MEMORY_TEST_POSTGRESQL_DSN is not set")
        importorskip("alembic")
        importorskip("sqlalchemy")
        schema = f"avalan_memory_test_{uuid4().hex}"

        settings = PgsqlMemoryMigrationSettings(url=dsn, schema=schema)

        memory_pgsql_upgrade(settings)
        memory_pgsql_check(settings)


class PgsqlMemoryMigrationEnvironmentTest(TestCase):
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
            MEMORY_PGSQL_ALEMBIC_VERSION_TABLE,
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
                    "memory_advisory_lock_id": "43",
                    "version_table": "custom_version_table",
                },
                attributes={"memory_schema": "memory_schema"},
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
                ('CREATE SCHEMA IF NOT EXISTS "memory_schema"', None),
                ('SET search_path TO "memory_schema", public', None),
                (
                    "SELECT pg_advisory_xact_lock(:lock_id)",
                    {"lock_id": 43},
                ),
            ],
        )

    def test_online_environment_uses_default_lock(self) -> None:
        connection = FakeSqlalchemyConnection()
        context = FakeAlembicEnvironmentContext(
            offline=False,
            config=FakeAlembicEnvironmentConfig(),
        )

        self._import_env(context=context, connection=connection)

        self.assertTrue(context.ran_migrations)
        self.assertEqual(
            connection.executed,
            [
                (
                    "SELECT pg_advisory_xact_lock(:lock_id)",
                    {"lock_id": MEMORY_PGSQL_ADVISORY_LOCK_ID},
                ),
            ],
        )

    def _import_env(
        self,
        *,
        context: FakeAlembicEnvironmentContext,
        connection: FakeSqlalchemyConnection | None = None,
    ) -> None:
        module_name = "avalan.memory.permanent.pgsql_migrations.env"
        old_alembic = modules.get("alembic")
        old_sqlalchemy = modules.get("sqlalchemy")
        old_env = modules.pop(module_name, None)
        fake_connection = connection or FakeSqlalchemyConnection()
        modules["alembic"] = SimpleNamespace(context=context)
        modules["sqlalchemy"] = SimpleNamespace(
            engine_from_config=lambda *args, **kwargs: (
                FakeSqlalchemyConnectable(fake_connection)
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


class PgsqlMemoryMigrationRevisionTest(TestCase):
    def test_revisions_execute_schema_statements(self) -> None:
        for module_path in (
            (
                "avalan.memory.permanent.pgsql_migrations.versions."
                "v20260530_0001_memory_schema"
            ),
            (
                "avalan.memory.permanent.pgsql_migrations.versions."
                "v20260530_0002_reasoning_graph_schema"
            ),
        ):
            with self.subTest(module_path=module_path):
                revision_module = import_module(module_path)
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
                    list(revision_module.MEMORY_SCHEMA_STATEMENTS),
                )

    def test_revisions_are_forward_only(self) -> None:
        for module_path in (
            (
                "avalan.memory.permanent.pgsql_migrations.versions."
                "v20260530_0001_memory_schema"
            ),
            (
                "avalan.memory.permanent.pgsql_migrations.versions."
                "v20260530_0002_reasoning_graph_schema"
            ),
        ):
            with self.subTest(module_path=module_path):
                revision_module = import_module(module_path)

                with self.assertRaises(NotImplementedError):
                    revision_module.downgrade()


def _unexpected_import(module: str) -> object:
    raise AssertionError(f"unexpected module import: {module}")


if __name__ == "__main__":
    main()
