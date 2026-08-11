"""Assert the public patch toolset remains capability-bound and async typed."""

from typing import assert_type

from avalan.entities import ToolCallContext
from avalan.patch.domain import (
    OperationType,
    PatchPending,
    PatchResult,
)
from avalan.patch.toolset import (
    PatchInvocationCapability,
    PatchSdkHost,
    PatchToolManagerBundle,
    PatchToolSet,
)
from avalan.tool import Tool


async def assert_patch_toolset_types(
    toolset: PatchToolSet,
    host: PatchSdkHost,
    pending: PatchPending,
) -> None:
    """Assert raw calls, pending settlement, and host lifecycle types."""
    capability = toolset.capability
    assert_type(capability, PatchInvocationCapability)
    assert_type(toolset.available_tools, tuple[Tool, ...])
    await toolset.invoke_json(
        OperationType.EDIT,
        {},
        ToolCallContext(patch_capability=capability),
    )
    assert_type(await host.await_terminal(pending), PatchResult)
    assert_type(
        await host.invoke_json(OperationType.EDIT, {}),
        PatchResult | PatchPending,
    )


def assert_patch_loader_bundle_types(bundle: PatchToolManagerBundle) -> None:
    """Assert loader results never widen trusted patch activation types."""
    assert_type(bundle.toolset, PatchToolSet | None)
