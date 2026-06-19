from ...types import assert_env_name, assert_non_empty_string

from dataclasses import dataclass, field
from os import getenv
from typing import Literal, final


def resolve_database_dsn(dsn: str) -> str:
    """Return the configured database DSN, resolving env references."""
    assert_non_empty_string(dsn, "dsn")
    if not dsn.startswith("env:"):
        return dsn

    env_name = dsn.removeprefix("env:")
    assert_env_name(env_name, "dsn environment variable")
    env_value = getenv(env_name)
    assert (
        env_value and env_value.strip()
    ), f"{env_name} environment variable must be set"
    return env_value


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class DatabaseToolSettings:
    """Configuration settings for database tools.

    This class is separated from the main database module to allow
    importing settings without requiring sqlglot or other heavy dependencies.
    """

    dsn: str
    delay_secs: float | None = None
    identifier_case: Literal["preserve", "lower", "upper"] = "preserve"
    read_only: bool = True
    allowed_commands: list[str] | None = field(
        default_factory=lambda: ["select"],
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "dsn", resolve_database_dsn(self.dsn))
        allowed_commands_value: object = self.allowed_commands
        if allowed_commands_value is None:
            return
        commands = (
            [allowed_commands_value]
            if isinstance(allowed_commands_value, str)
            else allowed_commands_value
        )
        assert isinstance(commands, list), "allowed_commands must be a list"
        assert commands, "allowed_commands must not be empty"
        for command in commands:
            assert_non_empty_string(command, "allowed_commands")
        object.__setattr__(self, "allowed_commands", commands)
