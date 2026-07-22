from ..agent import Specification
from ..entities import EngineUri, Input, Operation
from .capability import ModelCapabilityCatalog

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from .engine import Engine


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelCallContext:
    specification: Specification
    input: Input | None
    capability: ModelCapabilityCatalog | None = None
    engine_args: dict[str, Any] = field(default_factory=dict)
    parent: "ModelCallContext | None" = None
    root_parent: "ModelCallContext | None" = None
    agent_id: UUID | None = None
    participant_id: UUID | None = None
    session_id: UUID | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelCall:
    engine_uri: EngineUri
    model: "Engine"
    operation: Operation
    capability: ModelCapabilityCatalog | None = None
    context: ModelCallContext

    def __post_init__(self) -> None:
        context_capability = self.context.capability
        if self.capability is None:
            if context_capability is not None:
                object.__setattr__(self, "capability", context_capability)
            return
        if context_capability is None:
            object.__setattr__(
                self,
                "context",
                replace(self.context, capability=self.capability),
            )
            return
        assert (
            context_capability is self.capability
        ), "model call and context capabilities must be identical"
