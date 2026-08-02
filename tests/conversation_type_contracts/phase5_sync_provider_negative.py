"""Reject a synchronous Phase 5 native provider tool effect."""

from collections.abc import Mapping

from avalan.conversation import NativeOpenAIFunctionTool
from avalan.types import JsonValue


def synchronous_tool(arguments: Mapping[str, JsonValue]) -> str:
    """Return a result without an awaitable boundary."""
    return str(arguments)


NativeOpenAIFunctionTool(
    name="sync_tool",
    parameters={"type": "object"},
    handler=synchronous_tool,
)
