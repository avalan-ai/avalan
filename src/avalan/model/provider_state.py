"""Define the private asynchronous provider-state response boundary."""

from ..conversation.settings import (
    EffectiveReasoningMetadata,
    ProviderUsage,
)

from dataclasses import dataclass
from typing import Protocol, final


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderStateFinalization:
    """Report content-safe metadata after private state is finalized."""

    reasoning: EffectiveReasoningMetadata
    usage: ProviderUsage
    item_count: int

    def __post_init__(self) -> None:
        if type(self.reasoning) is not EffectiveReasoningMetadata:
            raise TypeError("reasoning must be effective reasoning metadata")
        if type(self.usage) is not ProviderUsage:
            raise TypeError("usage must be provider usage")
        if type(self.item_count) is not int or self.item_count < 0:
            raise TypeError("item_count must be a non-negative integer")


class ProviderStateSink(Protocol):
    """Finalize and clean one response-owned private provider sidecar."""

    async def finalize(self) -> ProviderStateFinalization:
        """Finalize complete provider state without exposing its payload."""
        ...

    async def cleanup(self) -> None:
        """Release every resource owned by the private sidecar."""
        ...


class ProviderStateError(RuntimeError):
    """Report a content-safe private provider-state lifecycle failure."""

    def __init__(self) -> None:
        super().__init__("private provider-state lifecycle failed")
