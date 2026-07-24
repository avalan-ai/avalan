"""Reconstruct durable agent continuations from trusted definitions."""

from ..entities import Message, MessageRole
from ..event.manager import EventManager, Listener
from ..interaction.continuation import (
    ResolvedContinuationRuntime,
)
from ..interaction.entities import (
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    InputRequiredResult,
    create_input_request,
)
from ..interaction.error import (
    InputErrorCode,
    InputValidationError,
)
from ..interaction.policy import InteractionActor
from ..memory.permanent.codec import decode_message_data
from ..model.capability import ModelCapabilityCatalog
from ..tool.context import ToolSettingsContext
from ..types import JsonValue
from . import continuation_stager as continuation_stager_module
from .continuation import (
    AgentContinuationEventListenerRegistration,
    AgentContinuationResumeCommand,
)
from .continuation_stager import (
    PortableAgentContinuationStager as PortableAgentContinuationStager,
)
from .execution import (
    AgentExecution,
    DurableInteractionRuntime,
    ExecutionCorrelationError,
    UuidExecutionIdFactory,
)
from .execution import (
    DurableInteractionStagingContext as DurableInteractionStagingContext,
)
from .loader import OrchestratorLoader
from .orchestrator import Orchestrator
from .orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)

from asyncio import Lock, Task, create_task, shield
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import NoReturn, cast, final
from urllib.parse import unquote, urlsplit


@final
class _TrustedContinuationEventListenerRegistration:
    """Own one listener attached to a reconstructed event manager."""

    def __init__(
        self,
        event_manager: EventManager,
        listener: Listener,
    ) -> None:
        if not isinstance(event_manager, EventManager):
            raise TypeError("event_manager must be an event manager")
        if not callable(listener):
            raise TypeError("listener must be callable")
        self._event_manager = event_manager
        self._listener = listener
        self._closed = False
        try:
            self._event_manager.add_listener(self._listener)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Remove the reconstructed-runtime listener exactly once."""
        if self._closed:
            return
        self._closed = True
        self._event_manager.remove_listener(self._listener)


@final
class _TrustedContinuationRuntimeOwnership:
    """Serialize every close request for one reconstructed runtime."""

    def __init__(self, ownership: AsyncExitStack) -> None:
        if not isinstance(ownership, AsyncExitStack):
            raise TypeError("ownership must be an async exit stack")
        self._ownership = ownership
        self._lock = Lock()
        self._close_task: Task[None] | None = None
        self._event_listener_registrations: list[
            _TrustedContinuationEventListenerRegistration
        ] = []

    async def __aenter__(self) -> "_TrustedContinuationRuntimeOwnership":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> bool:
        del exc_type, exc, traceback
        await self.close()
        return False

    def register_event_listener(
        self,
        event_manager: EventManager,
        listener: Listener,
    ) -> AgentContinuationEventListenerRegistration:
        """Retain one listener registration through runtime shutdown."""
        if self._close_task is not None:
            raise RuntimeError("continuation runtime is closing")
        registration = _TrustedContinuationEventListenerRegistration(
            event_manager,
            listener,
        )
        self._event_listener_registrations.append(registration)
        return registration

    async def close(self) -> None:
        """Close the retained stack once across concurrent owners."""
        async with self._lock:
            if self._close_task is None:
                self._close_task = create_task(
                    self._close_owned_runtime(),
                    name="durable-agent-runtime-close",
                )
            task = self._close_task
        await shield(task)

    async def _close_owned_runtime(self) -> None:
        errors: list[BaseException] = []
        registrations = tuple(self._event_listener_registrations)
        self._event_listener_registrations.clear()
        for registration in reversed(registrations):
            try:
                registration.close()
            except BaseException as error:
                errors.append(error)
        try:
            await self._ownership.aclose()
        except BaseException as error:
            errors.append(error)
        if errors:
            raise BaseExceptionGroup(
                "durable agent runtime cleanup failed",
                errors,
            )


@final
class TrustedAgentContinuationExecutor:
    """Resume one reconstructed execution through a freshly loaded agent."""

    trusted_agent_continuation_executor = True

    def __init__(
        self,
        orchestrator: Orchestrator,
        *,
        stager: PortableAgentContinuationStager,
        ownership: _TrustedContinuationRuntimeOwnership,
    ) -> None:
        if not isinstance(orchestrator, Orchestrator):
            raise TypeError("orchestrator must be a concrete orchestrator")
        if type(stager) is not PortableAgentContinuationStager:
            raise TypeError("stager must be a portable continuation stager")
        if type(ownership) is not _TrustedContinuationRuntimeOwnership:
            raise TypeError("ownership must be a shared runtime owner")
        self._orchestrator = orchestrator
        self._stager = stager
        self._ownership = ownership

    def register_event_listener(
        self,
        listener: Listener,
    ) -> AgentContinuationEventListenerRegistration:
        """Register one task listener before resumed provider dispatch."""
        return self._ownership.register_event_listener(
            self._orchestrator.event_manager,
            listener,
        )

    async def resume_agent_continuation(
        self,
        command: AgentContinuationResumeCommand,
    ) -> OrchestratorResponse:
        """Append one correlated result and continue the exact model turn."""
        if type(command) is not AgentContinuationResumeCommand:
            raise TypeError("command must be an agent continuation command")
        if command.resolved_runtime.runtime is not self:
            raise ExecutionCorrelationError(
                "resume command targets a different continuation runtime"
            )
        continuation = command.continuation
        transcript = _decode_transcript(continuation.transcript)
        (
            active_fingerprint,
            fingerprint_counts,
            assistant_message,
        ) = _decode_execution_observation(continuation.observations)
        if sum(fingerprint_counts.values()) != continuation.interaction_count:
            _invalid(
                "continuation.observations",
                "interaction counts do not match the checkpoint",
            )
        active_count = fingerprint_counts.get(active_fingerprint, 0)
        if active_count < 1:
            _invalid(
                "continuation.observations",
                "active interaction is absent from its counts",
            )
        staged_request = create_input_request(
            request_id=command.request.request_id,
            continuation_id=command.request.continuation_id,
            origin=command.request.origin,
            mode=command.request.mode,
            reason=command.request.reason,
            questions=command.request.questions,
            created_at=command.request.created_at,
            continuation_ttl_seconds=(
                command.request.continuation_ttl_seconds
            ),
            advisory_wait_seconds=command.request.advisory_wait_seconds,
        )
        id_factory = UuidExecutionIdFactory()
        origin = continuation.origin
        runtime = DurableInteractionRuntime(
            actor=InteractionActor(principal=origin.principal),
            stager=self._stager,
            id_factory=id_factory,
            run_id=origin.run_id,
            task_id=origin.task_id,
            branch_id=origin.branch_id,
            parent_branch_id=origin.parent_branch_id,
        )
        execution = AgentExecution(
            origin=origin,
            id_factory=id_factory,
            initial_messages=transcript,
            synced_message_prefix=len(transcript),
            interaction_runtime=runtime,
        )
        for fingerprint, count in sorted(fingerprint_counts.items()):
            prior_count = count - int(fingerprint == active_fingerprint)
            for _ in range(prior_count):
                await execution.begin_interaction(
                    fingerprint,
                    command.task_input_call,
                    assistant_message,
                )
                await execution.abandon_interaction()
        await execution.begin_interaction(
            active_fingerprint,
            command.task_input_call,
            assistant_message,
        )
        required = InputRequiredResult(
            request_id=staged_request.request_id,
            continuation_id=staged_request.continuation_id,
            detached_resumption_available=True,
        )
        await execution.stage_durable_input_required(
            staged_request,
            required,
        )
        committed = await execution.record_interaction_result(
            command.request,
            command.model_result,
            (
                assistant_message,
                command.correlated_result.tool_result_message(
                    command.task_input_call
                ),
            ),
        )
        if not committed:
            raise ExecutionCorrelationError(
                "durable interaction result was already committed"
            )
        return await self._orchestrator.resume_agent_execution(
            execution,
            operation_index=continuation.operation_cursor,
            capability=cast(
                ModelCapabilityCatalog,
                command.resolved_runtime.capabilities,
            ),
            generation_settings=continuation.generation_settings,
            initial_tool_cycle_count=continuation.tool_loop_count,
        )

    async def close_continuation_runtime(self) -> None:
        """Close resources owned by this reconstructed admission."""
        await self._ownership.close()


@final
class TrustedAgentContinuationRuntimeLoader:
    """Load exact agent continuations only from approved file roots."""

    trusted_continuation_runtime_loader = True

    def __init__(
        self,
        loader: OrchestratorLoader,
        *,
        stack: AsyncExitStack,
        allowed_roots: Sequence[str | Path],
        stager: PortableAgentContinuationStager | None = None,
        tool_settings: ToolSettingsContext | None = None,
        disable_memory: bool = False,
        uri: str | None = None,
    ) -> None:
        if not isinstance(loader, OrchestratorLoader):
            raise TypeError("loader must be an orchestrator loader")
        if not isinstance(stack, AsyncExitStack):
            raise TypeError("stack must be an async exit stack")
        roots = tuple(
            Path(root).resolve(strict=True) for root in allowed_roots
        )
        if not roots or any(not root.is_dir() for root in roots):
            raise ValueError("allowed_roots must contain existing directories")
        if tool_settings is not None and not isinstance(
            tool_settings,
            ToolSettingsContext,
        ):
            raise TypeError("tool_settings must be trusted tool settings")
        if type(disable_memory) is not bool:
            raise TypeError("disable_memory must be a boolean")
        if uri is not None and not isinstance(uri, str):
            raise TypeError("uri must be a string or None")
        self._loader = loader
        self._stack = stack
        self._allowed_roots = roots
        self._stager = stager or PortableAgentContinuationStager()
        self._tool_settings = tool_settings
        self._disable_memory = disable_memory
        self._uri = uri

    async def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> ResolvedContinuationRuntime:
        """Load and revalidate exact live components from trusted config."""
        if type(definition) is not ExecutionDefinitionRef:
            raise TypeError("definition must be an execution definition")
        if type(revision_binding) is not ContinuationRevisionBinding:
            raise TypeError("revision_binding must be exact")
        path = self._trusted_definition_path(
            definition.agent_definition_locator
        )
        local_stack = AsyncExitStack()
        try:
            admission_loader = self._loader.clone_for_stack(local_stack)
            orchestrator = await admission_loader.from_file(
                str(path),
                agent_id=None,
                disable_memory=self._disable_memory,
                uri=self._uri,
                tool_settings=self._tool_settings,
            )
            if not isinstance(orchestrator, Orchestrator):
                raise InputValidationError(
                    InputErrorCode.UNAVAILABLE,
                    "continuation_runtime.orchestrator",
                    "trusted definition did not load a concrete orchestrator",
                )
            await local_stack.enter_async_context(orchestrator)
            (
                resolved_definition,
                resolved_binding,
                capability,
            ) = orchestrator.continuation_execution_contract(
                definition.operation_index
            )
            if resolved_definition != definition:
                raise InputValidationError(
                    InputErrorCode.SNAPSHOT_REVISION_DRIFT,
                    "continuation_runtime.definition",
                    "fresh execution definition has drifted",
                )
            if resolved_binding != revision_binding:
                raise InputValidationError(
                    InputErrorCode.SNAPSHOT_REVISION_DRIFT,
                    "continuation_runtime.revision_binding",
                    "fresh provider revision has drifted",
                )
            engine_agent = orchestrator.engine_agent_for_operation(
                definition.operation_index
            )
            adapter = engine_agent.engine.model
            if adapter is None:
                raise InputValidationError(
                    InputErrorCode.UNAVAILABLE,
                    "continuation_runtime.model",
                    "fresh provider adapter is unavailable",
                )
            retained_stack = local_stack.pop_all()
            ownership = _TrustedContinuationRuntimeOwnership(retained_stack)
            try:
                executor = TrustedAgentContinuationExecutor(
                    orchestrator,
                    stager=self._stager,
                    ownership=ownership,
                )
                await self._stack.enter_async_context(ownership)
            except BaseException:
                await ownership.close()
                raise
        except BaseException:
            await local_stack.aclose()
            raise
        return ResolvedContinuationRuntime(
            definition=resolved_definition,
            revision_binding=resolved_binding,
            runtime=executor,
            operation=orchestrator.operations[definition.operation_index],
            model=adapter,
            tools=orchestrator.tool,
            capabilities=capability,
            credentials_reloaded_from_trusted_config=True,
        )

    def _trusted_definition_path(self, locator: str) -> Path:
        parsed = urlsplit(locator)
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
        ):
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "continuation_runtime.definition",
                "execution definition locator is not an approved file URI",
            )
        path = Path(unquote(parsed.path)).resolve(strict=True)
        if path.as_uri() != locator or not path.is_file():
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "continuation_runtime.definition",
                "execution definition locator is not canonical",
            )
        if not any(path.is_relative_to(root) for root in self._allowed_roots):
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "continuation_runtime.definition",
                "execution definition is outside approved roots",
            )
        return path


def _decode_transcript(
    records: tuple[Mapping[str, JsonValue], ...],
) -> tuple[Message, ...]:
    messages: list[Message] = []
    for index, record in enumerate(records):
        path = f"continuation.transcript[{index}]"
        if set(record) != {"version", "role", "data"}:
            _invalid(path, "transcript record fields do not match")
        if record["version"] != continuation_stager_module._TRANSCRIPT_VERSION:
            _invalid(path, "transcript record version is unsupported")
        role = record["role"]
        data = record["data"]
        if not isinstance(role, str) or not isinstance(data, str):
            _invalid(path, "transcript role and data must be strings")
        try:
            message = decode_message_data(MessageRole(role), data)
        except (TypeError, ValueError) as error:
            raise InputValidationError(
                InputErrorCode.SNAPSHOT_INVALID,
                path,
                "transcript message is invalid",
            ) from error
        messages.append(message)
    if not messages:
        _invalid("continuation.transcript", "transcript must not be empty")
    return tuple(messages)


def _decode_execution_observation(
    observations: tuple[Mapping[str, JsonValue], ...],
) -> tuple[str, dict[str, int], Message]:
    if len(observations) != 1:
        _invalid(
            "continuation.observations",
            "exactly one execution observation is required",
        )
    observation = observations[0]
    expected = {
        "version",
        "kind",
        "active_interaction_fingerprint",
        "interaction_fingerprint_counts",
        "assistant_message",
    }
    if set(observation) != expected:
        _invalid(
            "continuation.observations[0]",
            "execution observation fields do not match",
        )
    if (
        observation["version"]
        != continuation_stager_module._OBSERVATION_VERSION
        or observation["kind"]
        != continuation_stager_module._EXECUTION_OBSERVATION_KIND
    ):
        _invalid(
            "continuation.observations[0]",
            "execution observation version is unsupported",
        )
    active = observation["active_interaction_fingerprint"]
    raw_counts = observation["interaction_fingerprint_counts"]
    raw_assistant = observation["assistant_message"]
    if not isinstance(active, str) or not isinstance(raw_counts, tuple):
        _invalid(
            "continuation.observations[0]",
            "execution observation values are invalid",
        )
    counts: dict[str, int] = {}
    for index, raw_count in enumerate(raw_counts):
        path = (
            "continuation.observations[0]."
            f"interaction_fingerprint_counts[{index}]"
        )
        if not isinstance(raw_count, Mapping) or set(raw_count) != {
            "fingerprint",
            "count",
        }:
            _invalid(path, "interaction count fields do not match")
        fingerprint = raw_count["fingerprint"]
        count = raw_count["count"]
        if (
            not isinstance(fingerprint, str)
            or type(count) is not int
            or count < 1
            or fingerprint in counts
        ):
            _invalid(path, "interaction count is invalid")
        counts[fingerprint] = count
    if not isinstance(raw_assistant, Mapping):
        _invalid(
            "continuation.observations[0].assistant_message",
            "assistant message is invalid",
        )
    assistant = _decode_transcript((raw_assistant,))[0]
    if assistant.role is not MessageRole.ASSISTANT:
        _invalid(
            "continuation.observations[0].assistant_message",
            "originating message must be an assistant message",
        )
    return active, counts, assistant


def _invalid(path: str, message: str) -> NoReturn:
    raise InputValidationError(
        InputErrorCode.SNAPSHOT_INVALID,
        path,
        message,
    )
