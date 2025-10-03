from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..agent import Specification
from ..entities import EngineUri, Input, Operation
from ..tool.manager import ToolManager

if TYPE_CHECKING:
    from .engine import Engine


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelCallContext:
    specification: Specification
    input: Input | None
    engine_args: dict[str, Any] = field(default_factory=dict)
    parent: "ModelCallContext | None" = None
    root_parent: "ModelCallContext | None" = None


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelCall:
    engine_uri: EngineUri
    model: "Engine"
    operation: Operation
    tool: ToolManager | None = None
    context: ModelCallContext
