"""Bind configured child orchestrators to one conversation invocation."""

from ..conversation.contract import ProviderLaneId
from ..conversation.errors import ConversationValidationError
from ..event.manager import EventManager
from ..tool.manager import ToolManager
from . import AgentOperation
from .engine import EngineAgent

from dataclasses import dataclass
from typing import Protocol, final
from uuid import UUID


class ConfiguredChildOrchestrator(Protocol):
    """Expose the configured child surfaces needed for one invocation."""

    @property
    def id(self) -> UUID:
        """Return the configured child agent identifier."""
        ...

    @property
    def operations(self) -> list[AgentOperation]:
        """Return configured child operations."""
        ...

    @property
    def tool(self) -> ToolManager:
        """Return the configured child tool manager."""
        ...

    @property
    def event_manager(self) -> EventManager:
        """Return the configured child event manager."""
        ...

    def conversation_engine_args(self) -> dict[str, object]:
        """Return isolated configured arguments for a child invocation."""
        ...

    def engine_agent_for_operation(self, operation_index: int) -> EngineAgent:
        """Return the loaded engine agent for one operation."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AgentConversationChildBinding:
    """Map one frozen child lane to a configured child operation."""

    lane_id: ProviderLaneId
    orchestrator: ConfiguredChildOrchestrator
    operation_index: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.lane_id) is not str
            or not self.lane_id.strip()
            or type(self.operation_index) is not int
            or self.operation_index < 0
            or not callable(
                getattr(self.orchestrator, "engine_agent_for_operation", None)
            )
            or not isinstance(getattr(self.orchestrator, "id", None), UUID)
            or not isinstance(
                getattr(self.orchestrator, "operations", None), list
            )
        ):
            raise ConversationValidationError()

    def resolve(self) -> tuple[EngineAgent, AgentOperation]:
        """Resolve the exact loaded child engine and immutable operation."""
        operations = self.orchestrator.operations
        if self.operation_index >= len(operations):
            raise ConversationValidationError()
        operation = operations[self.operation_index]
        engine_agent = self.orchestrator.engine_agent_for_operation(
            self.operation_index
        )
        if (
            type(operation) is not AgentOperation
            or not isinstance(engine_agent, EngineAgent)
            or engine_agent.id != self.orchestrator.id
        ):
            raise ConversationValidationError()
        return engine_agent, operation

    def __repr__(self) -> str:
        """Return only content-free child binding metadata."""
        return (
            "AgentConversationChildBinding("
            f"lane_id={self.lane_id!r}, operation_index="
            f"{self.operation_index})"
        )
