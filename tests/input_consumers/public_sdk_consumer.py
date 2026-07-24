"""Consume durable task input exclusively through the public Avalan SDK."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from avalan import (
    AgentHeadlessInputPolicy,
    AgentInteractionRuntime,
    AgentRunCompleted,
    AgentRunInputRequired,
    Input,
    InputControllerClient,
    InputInspection,
    InputResolutionAccepted,
    InputSubmission,
    Orchestrator,
    ResolutionIdempotencyKey,
    inspect_input,
    resolve_input,
    run_agent,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicDurableRun:
    """Return the complete public pause, resolution, and completion path."""

    pause: AgentRunInputRequired
    inspection: InputInspection
    resolution: InputResolutionAccepted
    completion: AgentRunCompleted[str]


async def complete_durable_run(
    initial_orchestrator: Orchestrator,
    input: Input,
    *,
    interaction_runtime: AgentInteractionRuntime,
    policy: AgentHeadlessInputPolicy,
    controller: InputControllerClient,
    submission: InputSubmission,
    idempotency_key: ResolutionIdempotencyKey,
    resume_continuation: Callable[
        [AgentRunInputRequired],
        Awaitable[AgentRunCompleted[str]],
    ],
) -> PublicDurableRun:
    """Pause, inspect, resolve, resume, and complete through public types."""
    pause = await run_agent(
        initial_orchestrator,
        input,
        interaction_runtime=interaction_runtime,
        headless_policy=policy,
    )
    if not isinstance(pause, AgentRunInputRequired):
        raise RuntimeError("initial durable segment did not request input")
    if pause.request_id is None or pause.continuation_id is None:
        raise RuntimeError("durable input correlation is unavailable")
    inspection = await inspect_input(
        controller,
        pause.request_id,
        pause.continuation_id,
    )
    resolution = await resolve_input(
        controller,
        pause.request_id,
        pause.continuation_id,
        submission,
        idempotency_key=idempotency_key,
    )
    completion = await resume_continuation(pause)
    if not isinstance(completion, AgentRunCompleted):
        raise RuntimeError("resumed durable segment did not complete")
    return PublicDurableRun(
        pause=pause,
        inspection=inspection,
        resolution=resolution,
        completion=completion,
    )
