from dataclasses import dataclass
from types import FunctionType
from typing import Sequence


@dataclass(frozen=True, kw_only=True)
class ToolSet:
    """Collection of tools sharing an optional namespace."""

    tools: Sequence[FunctionType]
    namespace: str | None = None
