"""Prove the Phase 5 native stateless adapter is strictly async and typed."""

from collections.abc import Mapping
from typing import assert_type

from avalan.conversation import (
    NativeOpenAIConversationLaneRuntime,
    NativeOpenAIFunctionTool,
    NativeOpenAIProviderDiagnostics,
    NativeOpenAIStatelessProvider,
    ProviderItem,
    ProviderResult,
    StatelessProviderPlan,
)
from avalan.types import JsonValue


async def phase5_tool(arguments: Mapping[str, JsonValue]) -> str:
    """Return one typed asynchronous tool result."""
    return str(arguments)


async def prove_phase5_native_provider(
    provider: NativeOpenAIStatelessProvider,
    plan: StatelessProviderPlan,
    item: ProviderItem,
) -> tuple[ProviderResult, ProviderResult, str]:
    """Return exact native dispatch, stream, and tool result types."""
    assert_type(provider.diagnostics, NativeOpenAIProviderDiagnostics)
    assert_type(
        NativeOpenAIConversationLaneRuntime(provider=provider),
        NativeOpenAIConversationLaneRuntime,
    )
    direct = assert_type(await provider.dispatch(plan), ProviderResult)
    stream = await provider.stream(plan)
    async for output_item in stream:
        assert_type(output_item, ProviderItem)
    streamed = assert_type(await stream.terminal(), ProviderResult)
    assert_type(await stream.aclose(), None)
    tool_result = assert_type(await provider.execute_tool(item), str)
    assert_type(await provider.aclose(), None)
    NativeOpenAIFunctionTool(
        name="typed_tool",
        parameters={"type": "object"},
        handler=phase5_tool,
    )
    return direct, streamed, tool_result
