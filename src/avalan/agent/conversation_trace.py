"""Expose runtime-only agent conversation execution boundaries."""

from ..entities import ToolCall, ToolCallOutcome

from dataclasses import dataclass
from typing import Protocol, final


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentProviderResponseTrace:
    """Record one classified provider response before any tool effect."""

    text: str
    calls: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        if type(self.text) is not str or type(self.calls) is not tuple:
            raise TypeError("provider trace must contain text and calls")
        if any(type(call) is not ToolCall for call in self.calls):
            raise TypeError("provider trace calls must be exact tool calls")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentToolOutputTrace:
    """Record one actual ToolManager result after its effect boundary."""

    call: ToolCall
    outcome: ToolCallOutcome | None

    def __post_init__(self) -> None:
        if type(self.call) is not ToolCall:
            raise TypeError("tool trace must contain an exact call")


class AgentConversationTraceSink(Protocol):
    """Receive invocation-local provider and tool completion events."""

    async def record_provider_response(
        self,
        trace: AgentProviderResponseTrace,
    ) -> None:
        """Record one provider response before advancing its tool loop."""
        ...

    async def record_tool_output(self, trace: AgentToolOutputTrace) -> None:
        """Record one completed ToolManager outcome."""
        ...
