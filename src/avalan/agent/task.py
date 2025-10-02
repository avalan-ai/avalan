from . import Specification
from ..entities import EngineUri, Input, Operation
from ..tool.manager import ToolManager
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..model.engine import Engine


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelTaskContext:
    specification: Specification
    input: Input | None
    engine_args: dict[str, Any] = field(default_factory=dict)
    parent: "ModelTaskContext | None" = None
    root_parent: "ModelTaskContext | None" = None


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelTask:
    engine_uri: EngineUri
    model: "Engine"
    operation: Operation
    tool: ToolManager | None = None
    context: ModelTaskContext
