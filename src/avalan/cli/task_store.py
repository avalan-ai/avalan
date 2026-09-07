"""Read the canonical shared task and trigger PostgreSQL configuration."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from os import environ
from re import fullmatch


class TaskStoreConfigurationError(ValueError):
    """Report a safe configuration field without revealing its value."""

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name
        super().__init__("task.store_configuration: " + field_name)


@dataclass(frozen=True, slots=True)
class TaskStoreConfiguration:
    dsn: str | None = field(default=None, repr=False)
    schema: str | None = None

    def __post_init__(self) -> None:
        if self.dsn is not None and (
            not isinstance(self.dsn, str) or not self.dsn.strip()
        ):
            raise TaskStoreConfigurationError("store_dsn")
        if self.schema is not None and (
            not isinstance(self.schema, str)
            or fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", self.schema) is None
        ):
            raise TaskStoreConfigurationError("store_schema")

    def require_dsn(self) -> str:
        if self.dsn is None:
            raise TaskStoreConfigurationError("store_dsn")
        return self.dsn


def task_store_configuration(args: Namespace) -> TaskStoreConfiguration:
    """Read explicit store options or their canonical environment names."""
    values: list[str | None] = []
    for name in ("dsn", "schema"):
        value = getattr(args, "store_" + name, None)
        if value is None:
            value = environ.get("AVALAN_TASK_STORE_" + name.upper())
        if value is not None and not isinstance(value, str):
            raise TaskStoreConfigurationError("store_" + name)
        values.append(value.strip() or None if value is not None else None)
    return TaskStoreConfiguration(dsn=values[0], schema=values[1])


def add_task_store_arguments(parser: ArgumentParser) -> None:
    """Declare the canonical connection options on a command parser."""
    parser.add_argument(
        "--store-dsn",
        default=None,
        help="PostgreSQL DSN; defaults to AVALAN_TASK_STORE_DSN.",
    )
    parser.add_argument(
        "--store-schema",
        default=None,
        help="PostgreSQL schema; defaults to AVALAN_TASK_STORE_SCHEMA.",
    )
