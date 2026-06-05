from ....types import assert_non_empty_string as _assert_non_empty_string

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from re import fullmatch
from typing import Any, Protocol, cast

MEMORY_PGSQL_ALEMBIC_VERSION_TABLE = "avalan_memory_alembic_version"
MEMORY_PGSQL_HEAD_REVISION = "20260530_0002"
MEMORY_PGSQL_ADVISORY_LOCK_ID = 8_172_673_911_930_301_928
MEMORY_PGSQL_MIGRATIONS_MISSING_CODE = (
    "dependency.memory_pgsql_migrations_missing"
)
_MIGRATION_DEPENDENCY_MODULES = ("alembic", "sqlalchemy")
_MEMORY_PGSQL_REVISION_MODULES = (
    (
        "avalan.memory.permanent.pgsql_migrations.versions."
        "v20260530_0001_memory_schema"
    ),
    (
        "avalan.memory.permanent.pgsql_migrations.versions."
        "v20260530_0002_reasoning_graph_schema"
    ),
)

ModuleFinder = Callable[[str], object | None]
ModuleImporter = Callable[[str], object]


class PgsqlMemoryMigrationError(RuntimeError):
    pass


class _AlembicConfig(Protocol):
    attributes: dict[str, object]

    def set_main_option(self, name: str, value: str) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlMemoryMigrationSettings:
    url: str
    schema: str | None = None
    version_table: str = MEMORY_PGSQL_ALEMBIC_VERSION_TABLE
    advisory_lock_id: int = MEMORY_PGSQL_ADVISORY_LOCK_ID
    module_finder: ModuleFinder = find_spec
    module_importer: ModuleImporter = import_module
    attributes: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.url, "url")
        if self.schema is not None:
            _assert_pgsql_identifier(self.schema, "schema")
        _assert_pgsql_identifier(self.version_table, "version_table")
        assert isinstance(self.advisory_lock_id, int)
        assert not isinstance(self.advisory_lock_id, bool)
        assert callable(self.module_finder)
        assert callable(self.module_importer)
        assert isinstance(self.attributes, Mapping)


def memory_pgsql_script_location() -> str:
    return str(Path(__file__).parent)


def memory_pgsql_schema_statements() -> tuple[str, ...]:
    statements: list[str] = []
    for module_name in _MEMORY_PGSQL_REVISION_MODULES:
        revision = cast(Any, import_module(module_name))
        revision_statements = revision.MEMORY_SCHEMA_STATEMENTS
        assert isinstance(revision_statements, tuple)
        for statement in revision_statements:
            _assert_non_empty_string(statement, "statement")
            statements.append(statement)
    return tuple(statements)


def memory_pgsql_upgrade(
    settings: PgsqlMemoryMigrationSettings,
    *,
    revision: str = "head",
) -> None:
    _assert_revision(revision)
    _run_alembic_command(settings, "upgrade", revision)


def memory_pgsql_current(
    settings: PgsqlMemoryMigrationSettings,
    *,
    verbose: bool = False,
) -> None:
    assert isinstance(verbose, bool)
    _run_alembic_command(settings, "current", verbose=verbose)


def memory_pgsql_check(settings: PgsqlMemoryMigrationSettings) -> None:
    _run_alembic_command(settings, "current", check_heads=True)


def memory_pgsql_stamp(
    settings: PgsqlMemoryMigrationSettings,
    *,
    revision: str = "head",
) -> None:
    _assert_revision(revision)
    _run_alembic_command(settings, "stamp", revision)


def memory_pgsql_alembic_config(
    settings: PgsqlMemoryMigrationSettings,
) -> _AlembicConfig:
    _require_migration_dependencies(settings)
    config_module = cast(Any, settings.module_importer("alembic.config"))
    config = cast(_AlembicConfig, config_module.Config())
    config.set_main_option("script_location", memory_pgsql_script_location())
    config.set_main_option("sqlalchemy.url", settings.url)
    config.set_main_option("version_table", settings.version_table)
    config.set_main_option(
        "memory_advisory_lock_id",
        str(settings.advisory_lock_id),
    )
    if settings.schema is not None:
        config.set_main_option("memory_schema", settings.schema)
        config.set_main_option("version_table_schema", settings.schema)
    config.attributes.update(dict(settings.attributes))
    return config


def _require_migration_dependencies(
    settings: PgsqlMemoryMigrationSettings,
) -> None:
    for module in _MIGRATION_DEPENDENCY_MODULES:
        if settings.module_finder(module) is None:
            raise PgsqlMemoryMigrationError(
                f"{MEMORY_PGSQL_MIGRATIONS_MISSING_CODE}: "
                "PostgreSQL memory migrations require Alembic and "
                "SQLAlchemy."
            )


def _run_alembic_command(
    settings: PgsqlMemoryMigrationSettings,
    command_name: str,
    *args: object,
    **kwargs: object,
) -> None:
    config = memory_pgsql_alembic_config(settings)
    command_module = cast(Any, settings.module_importer("alembic.command"))
    command = getattr(command_module, command_name)
    command(config, *args, **kwargs)


def _assert_pgsql_identifier(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]{0,62}",
        value,
    ), f"{field_name} must be a PostgreSQL identifier"


def _assert_revision(value: str) -> None:
    _assert_non_empty_string(value, "revision")
    assert fullmatch(r"[A-Za-z0-9_.@+-]+", value)
