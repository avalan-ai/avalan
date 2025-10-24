from dataclasses import dataclass, field
from typing import Literal, final


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
