"""Assemble durable agent suspension and cold-resume components."""

from ..agent.continuation import (
    AgentDurableContinuationStore,
    DurableAgentContinuationResumer,
)
from ..agent.durable_runtime import (
    PortableAgentContinuationStager,
    TrustedAgentContinuationRuntimeLoader,
)
from ..agent.execution import (
    DurableInteractionRuntime,
    UuidExecutionIdFactory,
)
from ..agent.loader import OrchestratorLoader
from ..interaction.continuation import ContinuationRuntimeResolver
from ..interaction.entities import PrincipalScope, RunId, TaskId
from ..interaction.policy import InteractionActor
from ..tool.context import ToolSettingsContext
from .context import TaskTargetContext
from .resume import (
    TaskContinuationRecordStore,
    TaskDurableResumeCoordinator,
)

from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import cast, final

TaskInteractionActorResolver = Callable[
    [TaskTargetContext],
    InteractionActor,
]


@final
class DurableAgentTaskHost:
    """Own production suspension and fresh-runtime resume wiring."""

    def __init__(
        self,
        *,
        orchestrator_loader: OrchestratorLoader,
        stack: AsyncExitStack,
        allowed_roots: Sequence[str | Path],
        continuation_store: object,
        tool_settings: ToolSettingsContext | None = None,
        actor_resolver: TaskInteractionActorResolver | None = None,
        disable_memory: bool = False,
        uri: str | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not callable(
            getattr(
                continuation_store,
                "get_task_continuation_record",
                None,
            )
        ):
            raise TypeError(
                "continuation_store must expose task continuation records"
            )
        resolved_actor = actor_resolver or _default_task_actor
        if not callable(resolved_actor):
            raise TypeError("actor_resolver must be callable")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        clock_source = clock if clock is not None else _utc_now

        def resolved_clock() -> datetime:
            return clock_source()

        stager = PortableAgentContinuationStager(clock=resolved_clock)
        runtime_loader = TrustedAgentContinuationRuntimeLoader(
            orchestrator_loader,
            stack=stack,
            allowed_roots=allowed_roots,
            stager=stager,
            tool_settings=tool_settings,
            disable_memory=disable_memory,
            uri=uri,
        )
        resolver = ContinuationRuntimeResolver(
            runtime_loader,
            clock=resolved_clock,
        )
        resumer = DurableAgentContinuationResumer(
            cast(AgentDurableContinuationStore, continuation_store),
            resolver,
            clock=resolved_clock,
        )
        self._stager = stager
        self._runtime_loader = runtime_loader
        self._actor_resolver = resolved_actor
        self._resume_coordinator = TaskDurableResumeCoordinator(
            cast(TaskContinuationRecordStore, continuation_store),
            resumer,
        )

    @property
    def resume_coordinator(self) -> TaskDurableResumeCoordinator:
        """Return the exact coordinator consumed by a task worker."""
        return self._resume_coordinator

    @property
    def continuation_runtime_loader(
        self,
    ) -> TrustedAgentContinuationRuntimeLoader:
        """Return the trusted cold-process runtime loader."""
        return self._runtime_loader

    def interaction_runtime(
        self,
        context: TaskTargetContext,
    ) -> DurableInteractionRuntime:
        """Return a durable runtime bound to one fresh task execution."""
        if not isinstance(context, TaskTargetContext):
            raise TypeError("context must be a task target context")
        actor = self._actor_resolver(context)
        if not isinstance(actor, InteractionActor):
            raise TypeError("actor_resolver returned an invalid actor")
        run_id = RunId(context.execution.run_id)
        return DurableInteractionRuntime(
            actor=actor,
            stager=self._stager,
            id_factory=UuidExecutionIdFactory(),
            run_id=run_id,
            task_id=TaskId(context.execution.run_id),
        )


def _default_task_actor(context: TaskTargetContext) -> InteractionActor:
    _ = context
    return InteractionActor(principal=PrincipalScope())


def _utc_now() -> datetime:
    return datetime.now(UTC)
