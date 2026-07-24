"""Reject conversion of public non-completed results."""

from avalan import (
    AgentRunCancelled,
    AgentRunFailed,
    AgentRunInputRequired,
)


def convert_input_required(result: AgentRunInputRequired) -> str:
    """Attempt to convert an input-required result as completed output."""
    return result.to_str()


def convert_cancelled(result: AgentRunCancelled) -> object:
    """Attempt to convert a cancelled result as completed output."""
    return result.to_json()


def convert_failed(result: AgentRunFailed) -> str:
    """Attempt to convert a failed result as completed output."""
    return result.to_str()
