"""Assemble durable agent suspension and cold-resume components."""

from ..agent.continuation import (
    AgentConversationContinuationResolver,
    AgentConversationContinuationResult,
    AgentDurableContinuationStore,
    DurableAgentContinuationResumer,
    ResolvedAgentConversationContinuation,
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
from ..conversation.contract import (
    AuthorityScope,
    CheckpointId,
    CheckpointKind,
    PortableContinuationReference,
)
from ..conversation.state import CheckpointLifecycle, ConversationCheckpoint
from ..interaction.continuation import (
    ContinuationRuntimeResolver,
    PortableContinuation,
)
from ..interaction.entities import PrincipalScope, RunId, TaskId
from ..interaction.policy import (
    InteractionActor,
    InteractionPolicy,
    RuntimeInteractionClock,
)
from ..model.capability import (
    CorrelatedCapabilityResult,
    TaskInputCapabilityCall,
)
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
from typing import Protocol, cast, final

TaskInteractionActorResolver = Callable[
    [TaskTargetContext],
    InteractionActor,
]


class TaskAgentConversationStore(Protocol):
    """Load exact durable conversation suspension state."""

    async def load(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> ConversationCheckpoint: ...

    async def load_continuation_reference(
        self,
        checkpoint_id: CheckpointId,
        authority: AuthorityScope,
    ) -> PortableContinuationReference: ...


class TaskAgentConversationCoordinator(Protocol):
    """Resume one exact coordinated structured-input suspension."""

    async def resume_structured_input(
        self,
        checkpoint: ConversationCheckpoint,
        call: TaskInputCapabilityCall,
        result: CorrelatedCapabilityResult,
    ) -> AgentConversationContinuationResult: ...


@final
class TaskDurableAgentRuntime:
    """Resolve task continuations from configured production state."""

    def __init__(
        self,
        *,
        store: TaskAgentConversationStore,
        coordinator: TaskAgentConversationCoordinator,
        authority: AuthorityScope,
    ) -> None:
        if (
            not callable(getattr(store, "load", None))
            or not callable(
                getattr(store, "load_continuation_reference", None)
            )
            or not callable(
                getattr(coordinator, "resume_structured_input", None)
            )
            or type(authority) is not AuthorityScope
        ):
            raise TypeError("conversation runtime configuration is invalid")
        self._store = store
        self._coordinator = coordinator
        self._authority = authority

    def resolver(self) -> AgentConversationContinuationResolver:
        """Return the exact resolver consumed by fresh-worker admission."""
        return AgentConversationContinuationResolver(
            resolve_continuation=self.resolve_continuation,
        )

    async def resolve_continuation(
        self,
        continuation: PortableContinuation,
        continuation_digest: str,
    ) -> ResolvedAgentConversationContinuation:
        """Load and bind one exact suspension to its configured coordinator."""
        reference = continuation.conversation_checkpoint_reference
        if reference is None:
            raise TypeError("conversation checkpoint reference is unavailable")
        checkpoint_id = CheckpointId(reference.checkpoint_id)
        checkpoint = await self._store.load(checkpoint_id, self._authority)
        persisted = await self._store.load_continuation_reference(
            checkpoint_id,
            self._authority,
        )
        if (
            checkpoint.kind is not CheckpointKind.STRUCTURED_INPUT_SUSPENSION
            or checkpoint.lifecycle is not CheckpointLifecycle.COMMITTED
            or str(checkpoint.identity.execution_segment_id)
            != reference.execution_segment_id
            or persisted.continuation_id != continuation.continuation_id
            or int(persisted.state_revision) + 1
            != int(continuation.state_revision)
            or str(persisted.digest) != continuation_digest
            or persisted.definition != continuation.definition
            or persisted.revision_binding != continuation.revision_binding
        ):
            raise TypeError("conversation suspension state does not correlate")

        async def apply_result(
            call: TaskInputCapabilityCall,
            result: CorrelatedCapabilityResult,
        ) -> AgentConversationContinuationResult:
            applied = await self._coordinator.resume_structured_input(
                checkpoint,
                call,
                result,
            )
            if type(applied) is not AgentConversationContinuationResult:
                raise TypeError(
                    "conversation coordinator returned invalid state"
                )
            return applied

        return ResolvedAgentConversationContinuation(
            checkpoint=checkpoint,
            continuation_reference=persisted,
            apply_result=apply_result,
        )


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
        policy: InteractionPolicy | None = None,
        conversation_resolver: (
            AgentConversationContinuationResolver | None
        ) = None,
        conversation_runtime: TaskDurableAgentRuntime | None = None,
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
        if (
            conversation_resolver is not None
            and type(conversation_resolver)
            is not AgentConversationContinuationResolver
        ):
            raise TypeError(
                "conversation_resolver must be an agent conversation resolver"
            )
        if (
            conversation_runtime is not None
            and type(conversation_runtime) is not TaskDurableAgentRuntime
        ):
            raise TypeError(
                "conversation_runtime must be a task durable agent runtime"
            )
        if (
            conversation_resolver is not None
            and conversation_runtime is not None
        ):
            raise TypeError(
                "conversation resolver must have exactly one configured owner"
            )
        if conversation_runtime is not None:
            conversation_resolver = conversation_runtime.resolver()
        clock_source = clock if clock is not None else _utc_now
        resolved_policy = InteractionPolicy() if policy is None else policy
        if not isinstance(resolved_policy, InteractionPolicy):
            raise TypeError("policy must be an interaction policy")

        def resolved_clock() -> datetime:
            return clock_source()

        interaction_clock = RuntimeInteractionClock(resolved_clock)
        stager = PortableAgentContinuationStager(clock=resolved_clock)
        runtime_loader = TrustedAgentContinuationRuntimeLoader(
            orchestrator_loader,
            stack=stack,
            allowed_roots=allowed_roots,
            stager=stager,
            tool_settings=tool_settings,
            disable_memory=disable_memory,
            uri=uri,
            clock=interaction_clock,
            policy=resolved_policy,
        )
        resolver = ContinuationRuntimeResolver(
            runtime_loader,
            clock=resolved_clock,
        )
        resumer = DurableAgentContinuationResumer(
            cast(AgentDurableContinuationStore, continuation_store),
            resolver,
            conversation_resolver=conversation_resolver,
            clock=resolved_clock,
        )
        self._stager = stager
        self._runtime_loader = runtime_loader
        self._actor_resolver = resolved_actor
        self._clock = interaction_clock
        self._policy = resolved_policy
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
            clock=self._clock,
            policy=self._policy,
            id_factory=UuidExecutionIdFactory(),
            run_id=run_id,
            task_id=TaskId(context.execution.run_id),
        )


def _default_task_actor(context: TaskTargetContext) -> InteractionActor:
    _ = context
    return InteractionActor(principal=PrincipalScope())


def _utc_now() -> datetime:
    return datetime.now(UTC)
