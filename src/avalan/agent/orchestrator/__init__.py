from ...cli import CommandAbortException
from ...entities import (
    EngineMessage as EngineMessage,
)
from ...entities import (
    Input,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentText,
    MessageRole,
    TransformerEngineSettings,
    merge_generation_settings_options,
)
from ...entities import Modality as Modality
from ...event import Event, EventType
from ...event.manager import EventManager
from ...interaction.entities import (
    AgentId,
    ExecutionDefinitionRef,
    PrincipalScope,
)
from ...memory.manager import MemoryManager
from ...model.call import ModelCallContext
from ...model.capability import (
    ModelCapabilityCatalog,
    ProviderCapabilitySupport,
)
from ...model.engine import Engine
from ...model.manager import ModelManager
from ...model.response.text import TextGenerationResponse
from ...tool.manager import ToolManager
from ...tool.shell.input_files import shell_input_file_manifest
from ...tool.shell.settings import ShellToolSettings
from ...tool_cycles import (
    DEFAULT_MAXIMUM_TOOL_CYCLES,
    MaximumToolCycles,
    validate_maximum_tool_cycles,
)
from .. import (
    AgentOperation,
    InputType,
    NoOperationAvailableException,
    Specification,
)
from .. import (
    EngineEnvironment as EngineEnvironment,
)
from ..engine import EngineAgent
from ..execution import (
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionIdFactory,
    create_agent_execution,
    snapshot_execution_messages,
)
from ..renderer import Renderer, TemplateEngineAgent
from .response.orchestrator_response import OrchestratorResponse

import asyncio
from asyncio import Lock, Task, create_task, gather, shield
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, replace
from enum import Enum
from hashlib import sha256
from inspect import isawaitable
from json import dumps
from logging import Logger
from re import compile as compile_regex
from time import perf_counter
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

_INPUT_TEMPLATE_REFERENCE_PATTERN = compile_regex(r"{{\s*input\b")
_MAXIMUM_TOOL_CYCLE_OPTION_KEYS = ("maximum_tool_cycles", "max_tool_cycles")
_BLOCK_REPEATED_TOOL_CALLS_OPTION_KEY = "block_repeated_tool_calls"
_TERMINAL_EXECUTION_STATUSES = frozenset(
    {
        AgentExecutionStatus.COMPLETED,
        AgentExecutionStatus.CANCELLED,
        AgentExecutionStatus.ERRORED,
    }
)
_PROVIDER_CLEANUP_TIMEOUT_SECONDS = 5.0


class MemorySynchronizableResponse(Protocol):
    """Synchronize the exact transcript owned by one response."""

    async def sync_messages(self) -> None:
        """Synchronize unacknowledged transcript entries."""
        ...


class _ProviderCleanupState(Enum):
    """Describe ownership while a raw provider result changes hands."""

    TRANSFERRING = "transferring"
    CLEANUP_REQUIRED = "cleanup_required"
    COMPLETE = "complete"


class _PendingProviderCleanup:
    """Retain one provider result until failed transfer cleanup converges."""

    def __init__(
        self,
        response: TextGenerationResponse,
        execution: AgentExecution,
        engine_agent: EngineAgent,
    ) -> None:
        self.response = response
        self.execution = execution
        self.engine_agent = engine_agent
        self.state = _ProviderCleanupState.TRANSFERRING
        self.cancelled: bool | None = None
        self._cancel_task: Task[Any] | None = None
        self._close_task: Task[Any] | None = None
        self._terminal_task: Task[Any] | None = None
        self._cancel_attempted = False
        self._attempt_task: Task[tuple[BaseException, ...]] | None = None
        self._sync_task: Task[None] | None = None

    @property
    def cleanup_required(self) -> bool:
        """Return whether failed-transfer cleanup may be attempted."""
        return self.state is _ProviderCleanupState.CLEANUP_REQUIRED

    @property
    def ownership_cleanup_complete(self) -> bool:
        """Return whether provider and execution ownership both settled."""
        return self.state is _ProviderCleanupState.COMPLETE

    def require_cleanup(self, *, cancelled: bool) -> None:
        """Atomically classify one failed transfer before cleanup awaits."""
        assert isinstance(cancelled, bool)
        if self.state is _ProviderCleanupState.COMPLETE:
            return
        if self.state is _ProviderCleanupState.CLEANUP_REQUIRED:
            return
        self.cancelled = cancelled
        self.state = _ProviderCleanupState.CLEANUP_REQUIRED

    def complete_transfer(self) -> None:
        """Release raw-result ownership after wrapper registration."""
        assert self.state is _ProviderCleanupState.TRANSFERRING
        self.state = _ProviderCleanupState.COMPLETE

    @staticmethod
    def _observe_task(task: Task[Any]) -> None:
        """Observe one retained operation task after eventual completion."""
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            return

    @staticmethod
    async def _poll_task(
        task: Task[Any],
        stage: str,
        *,
        timeout: float = _PROVIDER_CLEANUP_TIMEOUT_SECONDS,
    ) -> tuple[bool, tuple[BaseException, ...]]:
        """Poll one cleanup operation and cancel it at the stage deadline."""
        await asyncio.sleep(0)
        if not task.done():
            await asyncio.wait(
                {task},
                timeout=timeout,
            )
        if not task.done():
            task.cancel()
            await asyncio.sleep(0)
            timeout_failure = TimeoutError(
                f"provider {stage} cleanup exceeded "
                f"{_PROVIDER_CLEANUP_TIMEOUT_SECONDS:g} seconds"
            )
            if not task.done():
                return False, (timeout_failure,)
            try:
                task.result()
            except asyncio.CancelledError:
                return True, (timeout_failure,)
            except BaseException as error:
                return True, (timeout_failure, error)
            return True, (timeout_failure,)
        try:
            result = task.result()
        except BaseException as error:
            return True, (error,)
        if result is None:
            return True, ()
        assert isinstance(result, tuple)
        assert all(isinstance(item, BaseException) for item in result)
        return True, cast(tuple[BaseException, ...], result)

    def _new_operation_task(
        self,
        operation: Any,
    ) -> Task[Any]:
        """Create and retain one observed cleanup operation task."""
        task = create_task(operation)
        task.add_done_callback(self._observe_task)
        return task

    async def _run_attempt(self) -> tuple[BaseException, ...]:
        """Attempt cancel, close, and terminalization independently."""
        if not self.cleanup_required:
            return ()
        assert self.cancelled is not None
        failures: list[BaseException] = []
        deadline = (
            asyncio.get_running_loop().time()
            + _PROVIDER_CLEANUP_TIMEOUT_SECONDS
        )

        if self._cancel_task is None and not self._cancel_attempted:
            self._cancel_attempted = True
            self._cancel_task = self._new_operation_task(
                self.response.cancel()
            )
        if (
            self._terminal_task is None
            and self.execution.status not in _TERMINAL_EXECUTION_STATUSES
        ):
            self._terminal_task = self._new_operation_task(
                self.execution.settle_provider_exit(
                    cancelled=self.cancelled,
                )
            )

        await asyncio.sleep(0)
        if self._close_task is None and not self.response.cleanup_complete:
            self._close_task = self._new_operation_task(self.response.aclose())

        operations = tuple(
            (stage, task)
            for stage, task in (
                ("cancel", self._cancel_task),
                ("close", self._close_task),
                ("execution terminalization", self._terminal_task),
            )
            if task is not None
        )
        results = await gather(
            *(self._poll_task(task, stage) for stage, task in operations)
        )
        for (stage, task), (done, stage_failures) in zip(
            operations,
            results,
            strict=True,
        ):
            failures.extend(stage_failures)
            if not done:
                continue
            if stage == "cancel" and self._cancel_task is task:
                self._cancel_task = None
            elif stage == "close" and self._close_task is task:
                self._close_task = None
            elif (
                stage == "execution terminalization"
                and self._terminal_task is task
            ):
                self._terminal_task = None

        if self._close_task is None and not self.response.cleanup_complete:
            retry_task = self._new_operation_task(self.response.aclose())
            self._close_task = retry_task
            retry_done, retry_failures = await self._poll_task(
                retry_task,
                "close retry",
                timeout=max(
                    0.0,
                    deadline - asyncio.get_running_loop().time(),
                ),
            )
            failures.extend(retry_failures)
            if retry_done and self._close_task is retry_task:
                self._close_task = None

        if (
            self.response.cleanup_complete
            and self.execution.status in _TERMINAL_EXECUTION_STATUSES
            and self._cancel_task is None
            and self._close_task is None
            and self._terminal_task is None
        ):
            self.state = _ProviderCleanupState.COMPLETE
        unique_failures: list[BaseException] = []
        seen_failure_ids: set[int] = set()
        for failure in failures:
            if id(failure) in seen_failure_ids:
                continue
            seen_failure_ids.add(id(failure))
            unique_failures.append(failure)
        return tuple(unique_failures)

    async def converge(self) -> tuple[BaseException, ...]:
        """Join one bounded cleanup attempt and permit a later retry."""
        if not self.cleanup_required:
            return ()
        task = self._attempt_task
        if task is None:
            task = create_task(self._run_attempt())
            task.add_done_callback(self._observe_task)
            self._attempt_task = task
        while True:
            try:
                return await shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    return (asyncio.CancelledError(),)
            finally:
                if task.done() and self._attempt_task is task:
                    self._attempt_task = None

    async def sync_messages(self) -> None:
        """Retry cleanup and synchronize only after ownership converges."""
        task = self._sync_task
        if task is None:
            task = create_task(self._run_sync())
            task.add_done_callback(self._observe_task)
            self._sync_task = task
        try:
            await shield(task)
        finally:
            if task.done() and self._sync_task is task:
                self._sync_task = None

    async def _run_sync(self) -> None:
        """Run cleanup and memory synchronization as one shared attempt."""
        failures = list(await self.converge())
        if self.execution.status in _TERMINAL_EXECUTION_STATUSES:
            try:
                await self.engine_agent.sync_messages(self.execution)
            except BaseException as error:
                failures.append(error)
        if not self.ownership_cleanup_complete and not failures:
            failures.append(RuntimeError("provider cleanup did not converge"))
        if failures:
            primary_failure = failures[0]
            Orchestrator._attach_cleanup_failures(
                primary_failure,
                tuple(failures[1:]),
            )
            raise primary_failure


class Orchestrator:
    _INTERRUPTED_EXIT_EXCEPTIONS = (
        asyncio.CancelledError,
        KeyboardInterrupt,
        CommandAbortException,
    )
    _id: UUID
    _name: str | None
    _operations: list[AgentOperation]
    _renderer: Renderer
    _total_operations: int
    _logger: Logger
    _model_manager: ModelManager
    _memory: MemoryManager
    _tool: ToolManager
    _event_manager: EventManager
    _engine_agents: dict[str, EngineAgent]
    _engines_stack: ExitStack
    _engines: list[Engine]
    _model_ids: set[str] = set()
    _call_options: dict[str, Any] | None = None
    _last_engine_agent: EngineAgent | None = None
    _pending_responses: dict[int, OrchestratorResponse]
    _pending_responses_lock: Lock
    _pending_provider_cleanups: dict[int, _PendingProviderCleanup]
    _pending_response_syncs: dict[
        int,
        tuple[MemorySynchronizableResponse, Task[None]],
    ]
    _exiting: bool
    _exit_memory: bool = True
    _shell_input_file_settings: ShellToolSettings | None
    _user: str | None
    _user_template: str | None

    def __init__(
        self,
        logger: Logger,
        model_manager: ModelManager,
        memory: MemoryManager,
        tool: ToolManager,
        event_manager: EventManager,
        operations: AgentOperation | list[AgentOperation],
        *,
        call_options: dict[str, Any] | None = None,
        exit_memory: bool = True,
        id: UUID | None = None,
        name: str | None = None,
        renderer: Renderer | None = None,
        shell_input_file_settings: ShellToolSettings | None = None,
        user: str | None = None,
        user_template: str | None = None,
    ):
        assert not (user and user_template)
        assert shell_input_file_settings is None or isinstance(
            shell_input_file_settings, ShellToolSettings
        )
        self._logger = logger
        self._model_manager = model_manager
        self._memory = memory
        self._tool = tool
        self._event_manager = event_manager
        self._operations = (
            [operations]
            if isinstance(operations, AgentOperation)
            else operations
        )
        self._id = id or uuid4()
        self._exit_memory = exit_memory
        self._name = name
        self._renderer = renderer or Renderer()
        self._total_operations = len(self._operations)
        self._call_options = call_options
        self._shell_input_file_settings = shell_input_file_settings
        self._user = user
        self._user_template = user_template
        self._engines = []
        self._engine_agents = {}
        self._engines_stack = ExitStack()
        self._model_ids = set()
        self._pending_responses = {}
        self._pending_provider_cleanups = {}
        self._pending_responses_lock = Lock()
        self._pending_response_syncs = {}
        self._exiting = False

    @staticmethod
    def _pop_maximum_tool_cycles(
        engine_args: dict[str, Any],
    ) -> MaximumToolCycles:
        values: dict[str, Any] = {}
        for key in _MAXIMUM_TOOL_CYCLE_OPTION_KEYS:
            if key in engine_args:
                values[key] = engine_args.pop(key)
        assert (
            len(values) <= 1
        ), "Use only one of maximum_tool_cycles or max_tool_cycles"
        if not values:
            return DEFAULT_MAXIMUM_TOOL_CYCLES
        value = next(iter(values.values()))
        return validate_maximum_tool_cycles(value)

    @staticmethod
    def _pop_block_repeated_tool_calls(engine_args: dict[str, Any]) -> bool:
        if _BLOCK_REPEATED_TOOL_CALLS_OPTION_KEY not in engine_args:
            return False
        value = engine_args.pop(_BLOCK_REPEATED_TOOL_CALLS_OPTION_KEY)
        assert type(value) is bool, "block_repeated_tool_calls must be a bool"
        return value

    @property
    def engine_agent(self) -> EngineAgent | None:
        return self._last_engine_agent

    @property
    def engine(self) -> Engine | None:
        engine_agent = self.engine_agent
        return engine_agent.engine if engine_agent else None

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def input_token_count(self) -> int | None:
        engine_agent = self.engine_agent
        if not engine_agent:
            return None
        count = engine_agent.input_token_count
        if callable(count):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(count())
            if engine_agent.output:
                return engine_agent.output.input_token_count
            return None
        return cast(int | None, count)

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def model_ids(self) -> set[str]:
        return self._model_ids

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def operations(self) -> list[AgentOperation]:
        return self._operations

    @property
    def tool(self) -> ToolManager:
        return self._tool

    @property
    def event_manager(self) -> EventManager:
        return self._event_manager

    @property
    def renderer(self) -> Renderer:
        """Return the renderer used by the orchestrator."""
        return self._renderer

    async def __call__(
        self,
        input: Input,
        *,
        generation_options_override: Mapping[str, Any] | None = None,
        operation_index: int = 0,
        interaction_runtime: AttachedInteractionRuntime | None = None,
        execution_id_factory: ExecutionIdFactory | None = None,
        **kwargs: Any,
    ) -> OrchestratorResponse:
        tool_confirm = kwargs.pop("tool_confirm", None)
        if (
            type(operation_index) is not int
            or operation_index < 0
            or operation_index >= self._total_operations
        ):
            raise NoOperationAvailableException()

        # Load engine agent
        operation = self._operations[operation_index]
        environment_hash = dumps(asdict(operation.environment))
        engine_agents = self._engine_agents
        assert engine_agents and environment_hash in engine_agents
        engine_agent = engine_agents[environment_hash]
        history = await self._sync_terminal_responses_and_snapshot()

        await self._event_manager.trigger(
            Event(type=EventType.START, payload={"step": operation_index})
        )

        messages = self._input_messages(operation.specification, input)
        messages = await self._input_messages_with_shell_manifest(messages)
        current_messages = self._execution_messages(messages)
        execution_messages = (*history, *current_messages)
        messages = cast(Input, list(execution_messages))

        participant_id = getattr(self._memory, "participant_id", None)
        session_id = (
            self._memory.permanent_message.session_id
            if self._memory.permanent_message
            else None
        )

        # Execute operation
        engine_args = {**(self._call_options or {}), **kwargs}
        if generation_options_override:
            engine_args = merge_generation_settings_options(
                engine_args,
                generation_options_override,
            )
        maximum_tool_cycles = self._pop_maximum_tool_cycles(engine_args)
        block_repeated_tool_calls = self._pop_block_repeated_tool_calls(
            engine_args
        )
        start = perf_counter()
        await self._event_manager.trigger(
            Event(
                type=EventType.ENGINE_RUN_BEFORE,
                payload={
                    "input": messages,
                    "specification": operation.specification,
                },
                started=start,
            )
        )

        self._logger.info(
            "Orchestrator calling engine agent %s", str(engine_agent)
        )
        capability_seed = self._tool.export_model_capability_seed()
        tokenizer = engine_agent.engine.tokenizer
        eos_token = tokenizer.eos_token if tokenizer else None
        parser_seed = capability_seed.get("parser")
        if eos_token and isinstance(parser_seed, dict):
            parser_seed["eos_token"] = eos_token
        provider_support = getattr(
            engine_agent.engine,
            "provider_capability_support",
            ProviderCapabilitySupport(),
        )
        if type(provider_support) is not ProviderCapabilitySupport:
            raise TypeError(
                "engine provider_capability_support must be trusted support"
            )
        support = ProviderCapabilitySupport(
            structured_invocation=provider_support.structured_invocation,
            stable_call_ids=provider_support.stable_call_ids,
            correlated_results=provider_support.correlated_results,
            attached_resolution=interaction_runtime is not None,
        )
        capability = ModelCapabilityCatalog.create(
            capability_seed,
            support=support,
        )
        definition = self._execution_definition(
            operation,
            operation_index=operation_index,
            capability_seed=capability_seed,
            attached=interaction_runtime is not None,
        )
        execution = await create_agent_execution(
            definition=definition,
            agent_id=AgentId(str(self._id)),
            principal=(
                interaction_runtime.actor.principal
                if interaction_runtime is not None
                else PrincipalScope()
            ),
            initial_messages=execution_messages,
            synced_message_prefix=len(history),
            id_factory=(
                interaction_runtime.id_factory
                if interaction_runtime is not None
                else execution_id_factory
            ),
            interaction_runtime=interaction_runtime,
        )
        messages = cast(Input, list(execution.messages))
        context = ModelCallContext(
            specification=operation.specification,
            input=messages,
            capability=capability,
            engine_args=dict(engine_args),
            agent_id=self._id,
            participant_id=participant_id,
            session_id=session_id,
            execution=execution,
            execution_origin=execution.origin,
            interaction_broker=execution.interaction_broker,
        )
        try:
            result = cast(TextGenerationResponse, await engine_agent(context))
        except BaseException as error:
            await self._settle_engine_call_failure(
                engine_agent,
                execution,
                error,
            )
            raise
        provider_cleanup: _PendingProviderCleanup | None = None
        try:
            provider_cleanup = _PendingProviderCleanup(
                result,
                execution,
                engine_agent,
            )
            self._pending_provider_cleanups[id(result)] = provider_cleanup
            engine_agent.acknowledge_provider_handoff(result)
        except BaseException as error:
            await self._settle_provider_handoff_failure(
                engine_agent,
                execution,
                result,
                provider_cleanup,
                error,
            )
            raise
        assert provider_cleanup is not None
        response: OrchestratorResponse | None = None
        try:
            self._logger.info(
                "Engine agent %s responded to orchestrator",
                str(engine_agent),
            )

            end = perf_counter()
            await self._event_manager.trigger(
                Event(
                    type=EventType.ENGINE_RUN_AFTER,
                    payload={
                        "result": result,
                        "input": messages,
                        "specification": operation.specification,
                        "context": context,
                    },
                    started=start,
                    finished=end,
                    elapsed=end - start,
                )
            )

            last_prompt = execution.last_prompt
            response_input = cast(
                Input,
                (
                    list(last_prompt.input)
                    if last_prompt is not None
                    and isinstance(last_prompt.input, tuple)
                    else (
                        last_prompt.input
                        if last_prompt is not None
                        else messages
                    )
                ),
            )

            response = OrchestratorResponse(
                response_input,
                result,
                engine_agent,
                operation,
                engine_args,
                context,
                capability=context.capability,
                event_manager=self._event_manager,
                tool=self._tool,
                tool_confirm=tool_confirm,
                agent_id=self._id,
                participant_id=participant_id,
                session_id=session_id,
                block_repeated_tool_calls=block_repeated_tool_calls,
                maximum_tool_cycles=maximum_tool_cycles,
            )
            async with self._pending_responses_lock:
                if self._exiting:
                    raise RuntimeError("orchestrator is closing")
                self._pending_responses[id(response)] = response
                self._pending_provider_cleanups.pop(id(result), None)
                provider_cleanup.complete_transfer()
        except BaseException as error:
            if (
                response is not None
                and self._pending_responses.get(id(response)) is response
            ):
                self._pending_responses.pop(id(response))
            provider_cleanup.require_cleanup(
                cancelled=isinstance(
                    error,
                    self._INTERRUPTED_EXIT_EXCEPTIONS,
                )
            )
            await self._settle_unowned_provider_response(
                result,
                execution,
                error,
                cancelled=isinstance(
                    error,
                    self._INTERRUPTED_EXIT_EXCEPTIONS,
                ),
            )
            raise
        assert response is not None
        return response

    async def _settle_engine_call_failure(
        self,
        engine_agent: EngineAgent,
        execution: AgentExecution,
        primary_failure: BaseException,
    ) -> None:
        """Settle pre-handoff engine failures without masking the primary."""
        cancelled = isinstance(
            primary_failure,
            self._INTERRUPTED_EXIT_EXCEPTIONS,
        )

        async def drain_engine() -> tuple[BaseException, ...]:
            try:
                return await engine_agent.drain_pending_provider_cleanups(
                    execution,
                    abandon_unclaimed=True,
                )
            except BaseException as error:
                return (error,)

        async def settle_execution() -> tuple[BaseException, ...]:
            try:
                return await execution.settle_provider_exit(
                    cancelled=cancelled,
                )
            except BaseException as error:
                return (error,)

        async def run_cleanup() -> tuple[
            tuple[BaseException, ...],
            tuple[BaseException, ...],
        ]:
            results = await gather(drain_engine(), settle_execution())
            return results[0], results[1]

        cleanup_task = create_task(run_cleanup())
        results: tuple[
            tuple[BaseException, ...],
            tuple[BaseException, ...],
        ]
        while True:
            try:
                results = await shield(cleanup_task)
                break
            except asyncio.CancelledError:
                continue
        self._attach_cleanup_failures(
            primary_failure,
            tuple(failure for result in results for failure in result),
        )

    async def _settle_provider_handoff_failure(
        self,
        engine_agent: EngineAgent,
        execution: AgentExecution,
        response: TextGenerationResponse,
        owner: _PendingProviderCleanup | None,
        primary_failure: BaseException,
    ) -> None:
        """Settle a raw result whose local handoff installation failed."""
        cancelled = isinstance(
            primary_failure,
            self._INTERRUPTED_EXIT_EXCEPTIONS,
        )
        secondary_failures: list[BaseException] = []
        try:
            secondary_failures.extend(
                await execution.settle_provider_exit(cancelled=cancelled)
            )
        except BaseException as error:
            secondary_failures.append(error)
        local_owner_installed = (
            owner is not None
            and self._pending_provider_cleanups.get(id(response)) is owner
        )
        engine_owner_retired = False
        if local_owner_installed:
            try:
                engine_agent.acknowledge_provider_handoff(response)
                engine_owner_retired = True
            except BaseException as error:
                secondary_failures.append(error)
            assert owner is not None
            owner.require_cleanup(cancelled=cancelled)
            try:
                secondary_failures.extend(await owner.converge())
            except BaseException as error:
                secondary_failures.append(error)
            if owner.ownership_cleanup_complete:
                if self._pending_provider_cleanups.get(id(response)) is owner:
                    self._pending_provider_cleanups.pop(id(response))
        if not engine_owner_retired:
            try:
                secondary_failures.extend(
                    await engine_agent.drain_pending_provider_cleanups(
                        execution,
                        abandon_unclaimed=True,
                    )
                )
            except BaseException as error:
                secondary_failures.append(error)
        self._attach_cleanup_failures(
            primary_failure,
            tuple(secondary_failures),
        )

    @staticmethod
    def _attach_cleanup_failures(
        primary_failure: BaseException,
        cleanup_failures: tuple[BaseException, ...],
    ) -> None:
        """Attach cleanup failures without replacing the primary exit."""
        seen_failure_ids = {id(primary_failure)}
        for cleanup_failure in cleanup_failures:
            if id(cleanup_failure) in seen_failure_ids:
                continue
            seen_failure_ids.add(id(cleanup_failure))
            primary_failure.add_note(
                "post-provider cleanup failure: "
                f"{cleanup_failure.__class__.__name__}: "
                f"{cleanup_failure}"
            )

    @classmethod
    def _raise_cleanup_failures(
        cls,
        cleanup_failures: list[BaseException],
    ) -> None:
        """Raise the first cleanup failure with every later one attached."""
        if not cleanup_failures:
            return
        primary_failure = cleanup_failures[0]
        cls._attach_cleanup_failures(
            primary_failure,
            tuple(cleanup_failures[1:]),
        )
        raise primary_failure

    async def _settle_execution_provider_exit(
        self,
        execution: AgentExecution,
        primary_failure: BaseException,
        *,
        cancelled: bool,
    ) -> None:
        """Terminalize a pre-handoff execution without masking its exit."""
        cleanup_task = create_task(
            execution.settle_provider_exit(cancelled=cancelled)
        )
        while True:
            try:
                secondary_failures = await shield(cleanup_task)
                break
            except asyncio.CancelledError:
                if cleanup_task.cancelled():
                    secondary_failures = (asyncio.CancelledError(),)
                    break
        self._attach_cleanup_failures(primary_failure, secondary_failures)

    async def _settle_unowned_provider_response(
        self,
        response: TextGenerationResponse,
        execution: AgentExecution,
        primary_failure: BaseException,
        *,
        cancelled: bool,
    ) -> None:
        """Clean and terminalize a result before ownership transfer."""
        owner = self._pending_provider_cleanups.get(id(response))
        if owner is None:
            cleanup_task = create_task(
                self._cleanup_unowned_provider_response(
                    response,
                    execution,
                    cancelled=cancelled,
                )
            )
            while True:
                try:
                    secondary_failures = await shield(cleanup_task)
                    break
                except asyncio.CancelledError:
                    if cleanup_task.cancelled():
                        secondary_failures = (asyncio.CancelledError(),)
                        break
        else:
            owner.require_cleanup(cancelled=cancelled)
            try:
                secondary_failures = await owner.converge()
            except BaseException as cleanup_failure:
                secondary_failures = (cleanup_failure,)

        self._attach_cleanup_failures(primary_failure, secondary_failures)
        if owner is not None and owner.ownership_cleanup_complete:
            if self._pending_provider_cleanups.get(id(response)) is owner:
                self._pending_provider_cleanups.pop(id(response))

    @staticmethod
    async def _cleanup_unowned_provider_response(
        response: TextGenerationResponse,
        execution: AgentExecution,
        *,
        cancelled: bool,
    ) -> tuple[BaseException, ...]:
        """Finish unowned provider and execution cleanup uninterrupted."""
        secondary_failures: list[BaseException] = []
        try:
            await response.cancel()
        except BaseException as error:
            secondary_failures.append(error)
        try:
            await response.aclose()
        except BaseException as error:
            secondary_failures.append(error)
        try:
            secondary_failures.extend(
                await execution.settle_provider_exit(cancelled=cancelled)
            )
        except BaseException as error:
            secondary_failures.append(error)
        return tuple(secondary_failures)

    @staticmethod
    def _execution_messages(input: Input) -> tuple[Message, ...]:
        """Normalize one model input into the execution transcript."""
        if isinstance(input, Message):
            return (input,)
        if isinstance(input, str):
            return (Message(role=MessageRole.USER, content=input),)
        if all(isinstance(item, Message) for item in input):
            return tuple(cast(list[Message], input))
        return tuple(
            Message(role=MessageRole.USER, content=item)
            for item in cast(list[str], input)
        )

    def _recent_message_snapshot(self) -> tuple[Message, ...]:
        """Return one invocation-local copy of prior conversation history."""
        recent = self._memory.recent_messages
        if not recent:
            return ()
        return snapshot_execution_messages(
            tuple(item.message for item in tuple(recent))
        )

    @staticmethod
    def _response_is_terminal(response: OrchestratorResponse) -> bool:
        """Return whether one owned response reached a terminal state."""
        execution = response.execution
        return (
            execution is not None
            and execution.status in _TERMINAL_EXECUTION_STATUSES
        )

    async def _sync_owned_response(
        self,
        response: MemorySynchronizableResponse,
        *,
        owned_only: bool,
    ) -> None:
        """Synchronize one response and release terminal ownership safely."""
        response_id = id(response)
        async with self._pending_responses_lock:
            owned_response = self._pending_responses.get(response_id)
            is_owned = owned_response is response
            if owned_only and not is_owned:
                return
            terminal_before = (
                self._response_is_terminal(owned_response)
                if is_owned and owned_response is not None
                else False
            )

        await response.sync_messages()
        if not is_owned:
            return

        async with self._pending_responses_lock:
            owned_response = self._pending_responses.get(response_id)
            terminal_after = (
                self._response_is_terminal(owned_response)
                if owned_response is response
                else False
            )
        if not terminal_after:
            return
        if not terminal_before:
            await response.sync_messages()
        async with self._pending_responses_lock:
            owned_response = self._pending_responses.get(response_id)
            if (
                owned_response is response
                and self._response_is_terminal(owned_response)
                and owned_response.ownership_cleanup_complete
            ):
                self._pending_responses.pop(response_id)

    async def _response_sync_task(
        self,
        response: MemorySynchronizableResponse,
        *,
        owned_only: bool,
    ) -> Task[None]:
        """Return one shared in-flight synchronization for a response."""
        response_id = id(response)
        async with self._pending_responses_lock:
            pending = self._pending_response_syncs.get(response_id)
            if pending is not None and pending[0] is response:
                return pending[1]

            task: Task[None]

            async def sync_response() -> None:
                try:
                    await self._sync_owned_response(
                        response,
                        owned_only=owned_only,
                    )
                finally:
                    async with self._pending_responses_lock:
                        current = self._pending_response_syncs.get(response_id)
                        if current is not None and current[1] is task:
                            self._pending_response_syncs.pop(response_id)

            task = create_task(sync_response())
            self._pending_response_syncs[response_id] = (response, task)
            return task

    async def _sync_terminal_responses_and_snapshot(
        self,
    ) -> tuple[Message, ...]:
        """Commit prior terminal responses before one new root snapshot."""
        cleanup_failures: list[BaseException] = []
        try:
            await self._sync_pending_provider_cleanups()
        except BaseException as error:
            cleanup_failures.append(error)
        async with self._pending_responses_lock:
            terminal_responses = tuple(
                response
                for response in self._pending_responses.values()
                if self._response_is_terminal(response)
            )
        cleanup_failures.extend(
            await self._sync_response_collection(
                terminal_responses,
                owned_only=True,
            )
        )
        self._raise_cleanup_failures(cleanup_failures)
        return self._recent_message_snapshot()

    async def _sync_response_collection(
        self,
        responses: tuple[MemorySynchronizableResponse, ...],
        *,
        owned_only: bool,
    ) -> tuple[BaseException, ...]:
        """Synchronize responses in transcript order despite prior failures."""
        failures: list[BaseException] = []
        for response in responses:
            task = await self._response_sync_task(
                response,
                owned_only=owned_only,
            )
            try:
                await shield(task)
            except BaseException as error:
                failures.append(error)
        return tuple(failures)

    @staticmethod
    async def _close_response_collection(
        responses: tuple[OrchestratorResponse, ...],
    ) -> tuple[BaseException, ...]:
        """Close every owned response without head-of-line skips."""

        async def close_response(
            response: OrchestratorResponse,
        ) -> BaseException | None:
            try:
                await response.aclose()
            except BaseException as error:
                return error
            return None

        results = await gather(
            *(close_response(response) for response in responses)
        )
        return tuple(
            result for result in results if isinstance(result, BaseException)
        )

    async def _sync_pending_provider_cleanups(
        self,
        *,
        abandon_unclaimed: bool = False,
    ) -> None:
        """Drain cleanup-required provider owners concurrently."""
        async with self._pending_responses_lock:
            if abandon_unclaimed:
                for owner in self._pending_provider_cleanups.values():
                    if not owner.cleanup_required:
                        owner.require_cleanup(cancelled=True)
            owners = tuple(
                (response_id, owner)
                for response_id, owner in (
                    self._pending_provider_cleanups.items()
                )
                if owner.cleanup_required
            )
        engine_agents = tuple(dict.fromkeys(self._engine_agents.values()))

        async def drain_engine(
            engine_agent: EngineAgent,
        ) -> tuple[BaseException, ...]:
            try:
                return await engine_agent.drain_pending_provider_cleanups(
                    abandon_unclaimed=abandon_unclaimed,
                )
            except BaseException as error:
                return (error,)

        async def drain_owner(
            owner: _PendingProviderCleanup,
        ) -> tuple[BaseException, ...]:
            try:
                await owner.sync_messages()
            except BaseException as error:
                return (error,)
            return ()

        results = await gather(
            *(drain_engine(engine_agent) for engine_agent in engine_agents),
            *(drain_owner(owner) for _, owner in owners),
        )
        async with self._pending_responses_lock:
            for response_id, owner in owners:
                if not owner.ownership_cleanup_complete:
                    continue
                if self._pending_provider_cleanups.get(response_id) is owner:
                    self._pending_provider_cleanups.pop(response_id)
        failures = [failure for result in results for failure in result]
        self._raise_cleanup_failures(failures)

    def _execution_definition(
        self,
        operation: AgentOperation,
        *,
        operation_index: int,
        capability_seed: Mapping[str, object],
        attached: bool,
    ) -> ExecutionDefinitionRef:
        """Return the immutable trusted definition for one invocation."""
        operation_payload = dumps(
            asdict(operation),
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        operation_digest = sha256(operation_payload.encode()).hexdigest()
        agent_payload = dumps(
            [asdict(item) for item in self._operations],
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        agent_revision = sha256(agent_payload.encode()).hexdigest()
        capability_payload = dumps(
            capability_seed,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        tool_revision = sha256(capability_payload.encode()).hexdigest()
        capability_revision = sha256(
            f"{tool_revision}:{int(attached)}".encode()
        ).hexdigest()
        model_reference = sha256(
            repr(operation.environment).encode()
        ).hexdigest()
        return ExecutionDefinitionRef(
            agent_definition_locator=f"agent:{self._id}",
            agent_definition_revision=agent_revision,
            operation_id=f"operation:{operation_digest}",
            operation_index=operation_index,
            model_config_reference=f"model-config:{model_reference}",
            tool_revision=tool_revision,
            capability_revision=capability_revision,
        )

    async def __aenter__(self) -> "Orchestrator":
        first_agent: TemplateEngineAgent | None = None
        model_ids: list[str] = []
        for operation in self._operations:
            # Load engine with environment
            environment = operation.environment
            environment_hash = dumps(asdict(environment))
            engine_agents = self._engine_agents
            if environment_hash not in engine_agents:
                assert environment.engine_uri.model_id is not None
                model_ids.append(environment.engine_uri.model_id)
                engine = self._model_manager.load_engine(
                    environment.engine_uri,
                    cast(TransformerEngineSettings, environment.settings),
                    operation.modality,
                )
                if not engine:
                    raise NotImplementedError()

                self._engines_stack.enter_context(engine)
                self._engines.append(engine)
                agent = TemplateEngineAgent(
                    engine,
                    self._memory,
                    self._tool,
                    self._event_manager,
                    self._model_manager,
                    self._renderer,
                    environment.engine_uri,
                    name=self._name,
                    id=self._id,
                )
                engine_agents[environment_hash] = agent
                if not first_agent:
                    first_agent = agent

        self._last_engine_agent = first_agent
        self._model_ids = set(model_ids)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> bool | None:
        cleanup_failures: list[BaseException] = []
        result: bool | None = None
        async with self._pending_responses_lock:
            self._exiting = True
            for owner in self._pending_provider_cleanups.values():
                if not owner.cleanup_required:
                    owner.require_cleanup(cancelled=True)
            responses = tuple(self._pending_responses.values())
        try:
            try:
                await self._sync_pending_provider_cleanups(
                    abandon_unclaimed=True,
                )
            except BaseException as error:
                cleanup_failures.append(error)
            try:
                cleanup_failures.extend(
                    await self._close_response_collection(responses)
                )
            except BaseException as error:
                cleanup_failures.append(error)
            try:
                cleanup_failures.extend(
                    await self._sync_response_collection(
                        responses,
                        owned_only=True,
                    )
                )
            except BaseException as error:
                cleanup_failures.append(error)
            if self._exit_memory:
                try:
                    self._memory.__exit__(exc_type, exc_value, traceback)
                except BaseException as error:
                    cleanup_failures.append(error)
            try:
                result = self._engines_stack.__exit__(
                    exc_type, exc_value, traceback
                )
            except BaseException as error:
                cleanup_failures.append(error)
            for engine in self._engines:
                try:
                    wait_closed = getattr(engine, "wait_closed", None)
                    if wait_closed:
                        close_result = wait_closed()
                        if isawaitable(close_result):
                            await close_result
                except BaseException as error:
                    cleanup_failures.append(error)
        finally:
            try:
                event_manager_close = getattr(
                    self._event_manager,
                    "aclose",
                    None,
                )
                if callable(event_manager_close):
                    close_result = event_manager_close()
                    if isawaitable(close_result):
                        await close_result
            except BaseException as error:
                cleanup_failures.append(error)
            finally:
                self._engines.clear()

        if cleanup_failures:
            if exc_value is not None:
                self._attach_cleanup_failures(
                    exc_value,
                    tuple(cleanup_failures),
                )
                return False
            self._raise_cleanup_failures(cleanup_failures)
        return result

    async def sync_messages(
        self,
        response: MemorySynchronizableResponse | None = None,
    ) -> None:
        """Synchronize one explicit response or every pending response."""
        if response is not None:
            task = await self._response_sync_task(
                response,
                owned_only=False,
            )
            await shield(task)
            return
        cleanup_failures: list[BaseException] = []
        try:
            await self._sync_pending_provider_cleanups()
        except BaseException as error:
            cleanup_failures.append(error)
        async with self._pending_responses_lock:
            responses = tuple(self._pending_responses.values())
        cleanup_failures.extend(
            await self._sync_response_collection(
                responses,
                owned_only=True,
            )
        )
        self._raise_cleanup_failures(cleanup_failures)

    def _input_messages(
        self, specification: Specification, input: Input
    ) -> Input:
        input_type = specification.input_type
        assert (
            input_type != InputType.TEXT
            or isinstance(input, str)
            or isinstance(input, Message)
            or isinstance(input, list)
        )

        if input_type == InputType.TEXT and isinstance(input, str):
            input = Message(role=MessageRole.USER, content=input)

        if self._user_template:
            input = self._render_user_template_input(specification, input)
        elif self._user:
            input = self._prefix_user_input(specification, input)

        return input

    async def _input_messages_with_shell_manifest(
        self,
        input: Input,
    ) -> Input:
        if self._shell_input_file_settings is None:
            return input

        target = self._last_user_file_message(input)
        if target is None:
            return input

        manifest = await shell_input_file_manifest(
            input,
            self._shell_input_file_settings,
        )
        if manifest is None:
            return input

        index, message = target
        replacement = replace(
            message,
            content=self._append_message_text_content(message, manifest),
        )
        if index is None:
            return replacement

        assert isinstance(input, list)
        messages = cast(list[Message], list(input))
        messages[index] = replacement
        return messages

    def _prefix_user_input(
        self, specification: Specification, input: Input
    ) -> Input:
        message = self._last_input_message(input)
        if message is not None:
            content = self._message_text_content(message)
            if content is None:
                return input

            render_vars = self._input_render_vars(specification, content)
            rendered_user = self._renderer.from_string(
                self._user or "", template_vars=render_vars
            )
            rendered_text = self._rendered_text(rendered_user)
            message_content = (
                rendered_text
                if self._user_references_input(self._user or "")
                else self._prefix_text(rendered_text, content)
            )
            return self._replace_last_message_input(input, message_content)

        if isinstance(input, list) and input and isinstance(input[-1], str):
            render_vars = self._input_render_vars(specification, input[-1])
            rendered_user = self._renderer.from_string(
                self._user or "", template_vars=render_vars
            )
            rendered_text = self._rendered_text(rendered_user)
            input[-1] = (
                rendered_text
                if self._user_references_input(self._user or "")
                else self._prefix_text(rendered_text, input[-1])
            )

        return input

    def _render_user_template_input(
        self, specification: Specification, input: Input
    ) -> Input:
        message = self._last_input_message(input)
        if message is None:
            return input

        content = self._message_text_content(message)
        if content is None:
            return input

        render_vars = self._input_render_vars(specification, content)
        rendered = self._renderer(self._user_template or "", **render_vars)
        return self._replace_last_message_input(input, rendered)

    @staticmethod
    def _last_input_message(input: Input) -> Message | None:
        if isinstance(input, Message):
            return input
        if (
            isinstance(input, list)
            and input
            and isinstance(input[-1], Message)
        ):
            return input[-1]
        return None

    @staticmethod
    def _last_user_file_message(
        input: Input,
    ) -> tuple[int | None, Message] | None:
        if isinstance(input, Message):
            if Orchestrator._message_has_file_content(input):
                return None, input
            return None

        if not isinstance(input, list):
            return None

        for index in range(len(input) - 1, -1, -1):
            message = input[index]
            if not isinstance(message, Message):
                continue
            if Orchestrator._message_has_file_content(message):
                return index, message
        return None

    @staticmethod
    def _message_has_file_content(message: Message) -> bool:
        if message.role != MessageRole.USER:
            return False
        if isinstance(message.content, MessageContentFile):
            return True
        if isinstance(message.content, list):
            return any(
                isinstance(content, MessageContentFile)
                for content in message.content
            )
        return False

    @staticmethod
    def _message_text_content(message: Message) -> str | None:
        if isinstance(message.content, MessageContentText):
            return message.content.text
        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            for content in message.content:
                if isinstance(content, MessageContentText):
                    return content.text
        return None

    @staticmethod
    def _prefix_text(prefix: str, content: str) -> str:
        prefix = prefix.strip()
        return f"{prefix}\n\n{content}" if prefix else content

    @staticmethod
    def _rendered_text(value: str | bytes) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    @staticmethod
    def _user_references_input(user: str) -> bool:
        return bool(_INPUT_TEMPLATE_REFERENCE_PATTERN.search(user))

    @staticmethod
    def _replace_last_message_input(
        input: Input,
        content: str,
    ) -> Input:
        message = Orchestrator._last_input_message(input)
        if message is None:
            return input

        replacement = replace(
            message,
            content=Orchestrator._replace_message_text_content(
                message, content
            ),
        )
        if isinstance(input, list):
            assert input and isinstance(input[-1], Message)
            input[-1] = replacement
            return input
        return replacement

    @staticmethod
    def _replace_message_text_content(
        message: Message, content: str
    ) -> str | MessageContent | list[MessageContent]:
        if isinstance(message.content, list):
            replacement: list[MessageContent] = []
            replaced = False
            for item in message.content:
                if not replaced and isinstance(item, MessageContentText):
                    replacement.append(
                        MessageContentText(type="text", text=content)
                    )
                    replaced = True
                else:
                    replacement.append(item)
            if replaced:
                return replacement
        return content

    @staticmethod
    def _append_message_text_content(
        message: Message,
        content: str,
    ) -> MessageContent | list[MessageContent]:
        text = MessageContentText(type="text", text=content)
        if isinstance(message.content, list):
            return [*message.content, text]
        if isinstance(message.content, MessageContentFile):
            return [message.content, text]
        return text

    @staticmethod
    def _input_render_vars(
        specification: Specification,
        input_content: str,
    ) -> dict[str, Any]:
        render_vars = (
            specification.template_vars.copy()
            if specification.template_vars
            else {}
        )
        if specification.settings and specification.settings.template_vars:
            render_vars.update(specification.settings.template_vars)
        render_vars.update({"input": input_content})
        return render_vars
