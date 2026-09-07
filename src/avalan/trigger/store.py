"""Define typed owner-bound trigger management and history contracts."""

from .definition import integer
from .records import (
    TriggerCoverageSpan,
    TriggerDefinition,
    TriggerEvent,
    TriggerOccurrence,
    TriggerSnapshot,
)
from .registration import TriggerRegistration

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Protocol, TypeVar

_Item = TypeVar("_Item")


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryCursor:
    """Identify an append-only history position without wall-clock ties."""

    offset: int

    def __post_init__(self) -> None:
        integer(self.offset, 0, 9223372036854775807, "cursor")


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerPage(Generic[_Item]):
    items: tuple[_Item, ...]
    next_cursor: HistoryCursor | None

    def __post_init__(self) -> None:
        assert isinstance(self.items, tuple)
        integer(len(self.items), 0, 200, "page")
        assert self.next_cursor is None or isinstance(
            self.next_cursor, HistoryCursor
        )


class TriggerStore(Protocol):
    """Bind persistence operations to a trusted owner at construction."""

    async def apply(
        self,
        registration: TriggerRegistration,
        *,
        expected_generation: int | None,
    ) -> TriggerSnapshot: ...

    async def inspect(self, name: str) -> TriggerSnapshot | None: ...

    async def list(
        self,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerSnapshot]: ...

    async def set_enabled(
        self,
        name: str,
        *,
        enabled: bool,
        expected_generation: int,
    ) -> TriggerSnapshot: ...

    async def discover(
        self,
        *,
        decision_time: datetime,
        limit: int = 100,
    ) -> tuple[TriggerSnapshot, ...]: ...

    async def events(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerEvent]: ...

    async def revision(
        self,
        name: str,
        revision: int,
    ) -> TriggerDefinition | None: ...

    async def coverage(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerCoverageSpan]: ...

    async def occurrences(
        self,
        name: str,
        *,
        cursor: HistoryCursor | None = None,
        limit: int = 50,
    ) -> TriggerPage[TriggerOccurrence]: ...
