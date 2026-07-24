"""Prove public attached handlers and strict run-result narrowing."""

from avalan import (
    AgentRunCancelled,
    AgentRunCompleted,
    AgentRunFailed,
    AgentRunInputRequired,
    AgentRunResult,
    AttachedInputContext,
    AttachedInputDetached,
    AttachedInputHandler,
    AttachedInputOutcome,
)


async def detach(
    context: AttachedInputContext,
) -> AttachedInputOutcome:
    """Detach one public semantic request without blocking synchronously."""
    _ = context
    return AttachedInputDetached()


HANDLER: AttachedInputHandler = detach


def completed_text(result: AgentRunResult[str]) -> str | None:
    """Convert only the completed member of the strict public union."""
    if isinstance(result, AgentRunCompleted):
        return result.to_str()
    if isinstance(result, AgentRunInputRequired):
        return None
    if isinstance(result, AgentRunCancelled):
        return None
    assert isinstance(result, AgentRunFailed)
    return None
