from ..feature_gate import ModuleFinder, TaskFeature, require_features
from ..store import TaskStoreError

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from re import fullmatch
from typing import Any, Protocol, cast

TASK_PGSQL_ALEMBIC_VERSION_TABLE = "avalan_task_alembic_version"
TASK_PGSQL_HEAD_REVISION = "20260530_0001"
TASK_PGSQL_ADVISORY_LOCK_ID = 8_172_673_911_930_301_927
_TASK_PGSQL_REVISION_MODULE = (
    "avalan.task.stores.pgsql_migrations.versions.v20260530_0001_task_schema"
)

ModuleImporter = Callable[[str], object]


class PgsqlTaskMigrationError(TaskStoreError):
    pass


class _AlembicConfig(Protocol):
    attributes: dict[str, object]

    def set_main_option(self, name: str, value: str) -> None: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class PgsqlTaskMigrationSettings:
    url: str
    schema: str | None = None
    version_table: str = TASK_PGSQL_ALEMBIC_VERSION_TABLE
    advisory_lock_id: int = TASK_PGSQL_ADVISORY_LOCK_ID
    enabled_features: tuple[TaskFeature, ...] = (
        TaskFeature.POSTGRESQL_MIGRATIONS,
    )
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
        assert isinstance(self.enabled_features, tuple)
        for feature in self.enabled_features:
            assert isinstance(feature, TaskFeature)
        assert callable(self.module_finder)
        assert callable(self.module_importer)
        assert isinstance(self.attributes, Mapping)


def task_pgsql_script_location() -> str:
    return str(Path(__file__).with_name("pgsql_migrations"))


def task_pgsql_schema_statements() -> tuple[str, ...]:
    revision = cast(Any, import_module(_TASK_PGSQL_REVISION_MODULE))
    statements = revision.TASK_SCHEMA_STATEMENTS
    assert isinstance(statements, tuple)
    for statement in statements:
        _assert_non_empty_string(statement, "statement")
    return statements


def task_pgsql_upgrade(
    settings: PgsqlTaskMigrationSettings,
    *,
    revision: str = "head",
) -> None:
    _assert_revision(revision)
    _run_alembic_command(settings, "upgrade", revision)


def task_pgsql_current(
    settings: PgsqlTaskMigrationSettings,
    *,
    verbose: bool = False,
) -> None:
    assert isinstance(verbose, bool)
    _run_alembic_command(settings, "current", verbose=verbose)


def task_pgsql_check(settings: PgsqlTaskMigrationSettings) -> None:
    _run_alembic_command(settings, "current", check_heads=True)


def task_pgsql_stamp(
    settings: PgsqlTaskMigrationSettings,
    *,
    revision: str = "head",
) -> None:
    _assert_revision(revision)
    _run_alembic_command(settings, "stamp", revision)


def task_pgsql_alembic_config(
    settings: PgsqlTaskMigrationSettings,
) -> _AlembicConfig:
    diagnostics = require_features(
        (TaskFeature.POSTGRESQL_MIGRATIONS,),
        enabled_features=settings.enabled_features,
        module_finder=settings.module_finder,
    )
    if diagnostics:
        diagnostic = diagnostics[0]
        raise PgsqlTaskMigrationError(
            f"{diagnostic.code}: {diagnostic.message}"
        )

    config_module = cast(Any, settings.module_importer("alembic.config"))
    config = cast(_AlembicConfig, config_module.Config())
    config.set_main_option("script_location", task_pgsql_script_location())
    config.set_main_option("sqlalchemy.url", settings.url)
    config.set_main_option("version_table", settings.version_table)
    config.set_main_option(
        "task_advisory_lock_id",
        str(settings.advisory_lock_id),
    )
    if settings.schema is not None:
        config.set_main_option("task_schema", settings.schema)
        config.set_main_option("version_table_schema", settings.schema)
    config.attributes.update(dict(settings.attributes))
    return config


def _run_alembic_command(
    settings: PgsqlTaskMigrationSettings,
    command_name: str,
    *args: object,
    **kwargs: object,
) -> None:
    config = task_pgsql_alembic_config(settings)
    command_module = cast(Any, settings.module_importer("alembic.command"))
    command = getattr(command_module, command_name)
    command(config, *args, **kwargs)


def _assert_non_empty_string(value: object, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def _assert_pgsql_identifier(value: str, field_name: str) -> None:
    _assert_non_empty_string(value, field_name)
    assert fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]{0,62}",
        value,
    ), f"{field_name} must be a PostgreSQL identifier"


def _assert_revision(value: str) -> None:
    _assert_non_empty_string(value, "revision")
    assert fullmatch(r"[A-Za-z0-9_.@+-]+", value)
