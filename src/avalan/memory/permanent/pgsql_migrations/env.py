from importlib import import_module
from typing import Any, cast

alembic = cast(Any, import_module("alembic"))
sqlalchemy = cast(Any, import_module("sqlalchemy"))
context = alembic.context
engine_from_config = sqlalchemy.engine_from_config
pool = sqlalchemy.pool
text = sqlalchemy.text

config = context.config
target_metadata = None


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=_version_table(),
        version_table_schema=_memory_schema(),
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        _prepare_connection(connection)
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table=_version_table(),
            version_table_schema=_memory_schema(),
            include_schemas=_memory_schema() is not None,
        )

        with context.begin_transaction():
            context.run_migrations()


def _prepare_connection(connection: Any) -> None:
    lock_id = int(
        config.get_main_option("memory_advisory_lock_id")
        or "8172673911930301928"
    )
    schema = _memory_schema()
    if schema is not None:
        connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
        connection.execute(text(f'SET search_path TO "{schema}", public'))
    connection.execute(
        text("SELECT pg_advisory_xact_lock(:lock_id)"),
        {"lock_id": lock_id},
    )


def _memory_schema() -> str | None:
    schema = config.attributes.get("memory_schema")
    if schema is None:
        schema = config.get_main_option("memory_schema")
    return str(schema) if schema else None


def _version_table() -> str:
    version_table = config.get_main_option("version_table")
    return version_table or "avalan_memory_alembic_version"


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
