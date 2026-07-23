from ..cli import CommandAbortException
from ..entities import (
    ChatSettings,
    EngineMessage,
    EngineMessageIdempotencyKey,
    EngineUri,
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    Modality,
    Operation,
    OperationParameters,
    OperationTextParameters,
    ReasoningEffort,
    ReasoningSettings,
    ReasoningSummaryMode,
    ReasoningTag,
)
from ..event import Event, EventType
from ..event.manager import EventManager
from ..memory.manager import MemoryManager
from ..model.call import ModelCall, ModelCallContext
from ..model.capability import ModelCapabilityCatalog
from ..model.engine import Engine
from ..model.manager import ModelManager
from ..model.response.text import TextGenerationResponse
from ..tool.manager import ToolManager
from .execution import (
    AgentExecution,
    AgentExecutionStatus,
    ExecutionMemoryEntry,
    ModelPromptRecord,
)

from abc import ABC, abstractmethod
from asyncio import (
    CancelledError,
    Lock,
    Task,
    create_task,
    gather,
    shield,
    wait_for,
)
from contextvars import ContextVar
from dataclasses import Field, fields, replace
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

_TERMINAL_EXECUTION_STATUSES = frozenset(
    {
        AgentExecutionStatus.COMPLETED,
        AgentExecutionStatus.CANCELLED,
        AgentExecutionStatus.ERRORED,
    }
)


class _EngineProviderCleanup:
    """Retain and retry one provider response until cleanup converges."""

    def __init__(
        self,
        response: TextGenerationResponse,
        execution: AgentExecution | None,
        *,
        timeout_seconds: float,
    ) -> None:
        assert isinstance(response, TextGenerationResponse)
        assert isinstance(timeout_seconds, float)
        assert timeout_seconds > 0
        self.response = response
        self.execution = execution
        self.timeout_seconds = timeout_seconds
        self.cleanup_required = False
        self.cancelled = False
        self._attempt_lock = Lock()
        self._attempt_task: Task[tuple[BaseException, ...]] | None = None
        self._cancel_task: Task[None] | None = None
        self._close_task: Task[None] | None = None
        self._close_satisfied = False
        self._settlement_task: Task[tuple[BaseException, ...]] | None = None

    @property
    def cleanup_complete(self) -> bool:
        """Return whether no provider cleanup ownership remains."""
        execution_terminal = self.execution is None or (
            self.execution.status in _TERMINAL_EXECUTION_STATUSES
        )
        return (
            self.cleanup_required
            and self.response.cleanup_complete
            and self._close_satisfied
            and execution_terminal
            and self._attempt_task is None
            and self._cancel_task is None
            and self._close_task is None
            and self._settlement_task is None
        )

    def require_cleanup(self, *, cancelled: bool) -> None:
        """Mark an unclaimed response for cancellation or failure cleanup."""
        assert isinstance(cancelled, bool)
        if not self.cleanup_required:
            self.cancelled = cancelled
        self.cleanup_required = True

    async def converge(self) -> tuple[BaseException, ...]:
        """Join one coalesced, time-bounded cleanup attempt."""
        if not self.cleanup_required:
            return ()
        async with self._attempt_lock:
            task = self._attempt_task
            if task is None:
                task = create_task(self._run_attempt())
                self._attempt_task = task
                task.add_done_callback(self._observe_task_completion)

        try:
            return await shield(task)
        finally:
            if task.done():
                async with self._attempt_lock:
                    if self._attempt_task is task:
                        self._attempt_task = None

    async def _run_attempt(self) -> tuple[BaseException, ...]:
        """Run cancel, close, and settlement without serially hanging."""
        settlement_waiter = create_task(self._await_settlement())
        cancel_failures = await self._await_cancel()
        close_failures, settlement_failures = await gather(
            self._await_close(),
            settlement_waiter,
        )
        return (
            *cancel_failures,
            *close_failures,
            *settlement_failures,
        )

    async def _await_cancel(self) -> tuple[BaseException, ...]:
        """Join or start the bounded provider cancellation operation."""
        task = self._cancel_task
        if task is None:
            if self.response.cleanup_complete:
                return ()
            task = create_task(self.response.cancel())
            self._cancel_task = task
            task.add_done_callback(self._observe_task_completion)
        try:
            await wait_for(shield(task), timeout=self.timeout_seconds)
        except TimeoutError as error:
            if task.done():
                if self._cancel_task is task:
                    self._cancel_task = None
                try:
                    task.result()
                except BaseException as task_failure:
                    return (error, task_failure)
                return (error,)
            return (self._timeout_failure("cancel"),)
        except BaseException as error:
            if self._cancel_task is task:
                self._cancel_task = None
            return (error,)
        if self._cancel_task is task:
            self._cancel_task = None
        if self.response.cleanup_complete:
            self._close_satisfied = True
        return ()

    async def _await_close(self) -> tuple[BaseException, ...]:
        """Join or start the bounded provider close operation."""
        task = self._close_task
        if task is None:
            if self._close_satisfied:
                return ()
            task = create_task(self.response.aclose())
            self._close_task = task
            task.add_done_callback(self._observe_task_completion)
        try:
            await wait_for(shield(task), timeout=self.timeout_seconds)
        except TimeoutError as error:
            if task.done():
                if self._close_task is task:
                    self._close_task = None
                self._close_satisfied = self.response.cleanup_complete
                try:
                    task.result()
                except BaseException as task_failure:
                    return (error, task_failure)
                return (error,)
            return (self._timeout_failure("close"),)
        except BaseException as error:
            if self._close_task is task:
                self._close_task = None
            self._close_satisfied = self.response.cleanup_complete
            return (error,)
        if self._close_task is task:
            self._close_task = None
        self._close_satisfied = self.response.cleanup_complete
        if not self._close_satisfied:
            return (RuntimeError("provider close completed without closure"),)
        return ()

    async def _await_settlement(self) -> tuple[BaseException, ...]:
        """Join or start the bounded execution settlement operation."""
        execution = self.execution
        if execution is None:
            return ()
        task = self._settlement_task
        if task is None:
            if execution.status in _TERMINAL_EXECUTION_STATUSES:
                return ()
            task = create_task(
                execution.settle_provider_exit(cancelled=self.cancelled)
            )
            self._settlement_task = task
            task.add_done_callback(self._observe_task_completion)
        try:
            failures = await wait_for(
                shield(task),
                timeout=self.timeout_seconds,
            )
        except TimeoutError as error:
            if task.done():
                if self._settlement_task is task:
                    self._settlement_task = None
                try:
                    failures = task.result()
                except BaseException as task_failure:
                    return (error, task_failure)
                return (error, *failures)
            return (self._timeout_failure("execution settlement"),)
        except BaseException as error:
            if self._settlement_task is task:
                self._settlement_task = None
            return (error,)
        if self._settlement_task is task:
            self._settlement_task = None
        return failures

    def _timeout_failure(self, operation: str) -> TimeoutError:
        """Return one diagnostic while retaining the unfinished task."""
        return TimeoutError(
            f"provider {operation} exceeded the "
            f"{self.timeout_seconds:g}s cleanup budget"
        )

    @staticmethod
    def _observe_task_completion(task: Task[Any]) -> None:
        """Observe a retained task failure while preserving its result."""
        try:
            task.exception()
        except BaseException:
            return


class InputTokenCountModel(Protocol):
    def input_token_count(
        self,
        input: Input,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        *,
        instructions: str | None = None,
    ) -> int: ...


class EngineAgent(ABC):
    _PROVIDER_CLEANUP_TIMEOUT_SECONDS = 0.25
    _INTERRUPTED_EXIT_EXCEPTIONS = (
        CancelledError,
        KeyboardInterrupt,
        CommandAbortException,
    )
    _GENERATION_FIELDS: dict[str, Field[Any]] = {
        field.name: field for field in fields(GenerationSettings)
    }
    _id: UUID
    _name: str | None
    _model: Engine
    _memory: MemoryManager
    _tool: ToolManager
    _event_manager: EventManager
    _model_manager: ModelManager
    _engine_uri: EngineUri
    _legacy_output_context: ContextVar[TextGenerationResponse | None]
    _legacy_prompt_context: ContextVar[
        tuple[
            Input,
            str | None,
            str | None,
            str | None,
        ]
        | None
    ]
    _execution_context: ContextVar[AgentExecution | None]

    @abstractmethod
    def _prepare_call(self, context: ModelCallContext) -> Any:
        raise NotImplementedError()

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def engine(self) -> Engine:
        return self._model

    @property
    def engine_uri(self) -> EngineUri:
        return self._engine_uri

    @property
    def _last_output(self) -> TextGenerationResponse | None:
        """Return task-local legacy output state."""
        return self._legacy_output_context.get()

    @_last_output.setter
    def _last_output(self, value: TextGenerationResponse | None) -> None:
        """Set task-local legacy output state."""
        self._legacy_output_context.set(value)

    @property
    def _last_prompt(
        self,
    ) -> tuple[Input, str | None, str | None, str | None] | None:
        """Return task-local legacy prompt state."""
        return self._legacy_prompt_context.get()

    @_last_prompt.setter
    def _last_prompt(
        self,
        value: tuple[Input, str | None, str | None, str | None] | None,
    ) -> None:
        """Set task-local legacy prompt state."""
        self._legacy_prompt_context.set(value)

    @property
    def output(self) -> TextGenerationResponse | None:
        execution = self._execution_context.get()
        if execution is not None and isinstance(
            execution.last_response,
            TextGenerationResponse,
        ):
            return execution.last_response
        return self._last_output

    @property
    def last_prompt(
        self,
    ) -> tuple[Input, str | None, str | None, str | None] | None:
        execution = self._execution_context.get()
        if execution is not None and execution.last_prompt is not None:
            prompt = execution.last_prompt
            prompt_input = cast(
                Input,
                (
                    list(prompt.input)
                    if isinstance(prompt.input, tuple)
                    else prompt.input
                ),
            )
            return (
                prompt_input,
                prompt.instructions,
                prompt.system_prompt,
                prompt.developer_prompt,
            )
        return self._last_prompt

    async def input_token_count(self) -> int | None:
        prompt = self.last_prompt
        if not prompt:
            return None
        await self._event_manager.trigger(
            Event(
                type=EventType.INPUT_TOKEN_COUNT_BEFORE,
                payload={
                    "model_type": self._model.model_type,
                    "model_id": self._model.model_id,
                },
            )
        )
        count_model = cast(InputTokenCountModel, self._model)
        count = count_model.input_token_count(
            prompt[0],
            instructions=prompt[1],
            system_prompt=prompt[2],
            developer_prompt=prompt[3],
        )
        await self._event_manager.trigger(
            Event(
                type=EventType.INPUT_TOKEN_COUNT_AFTER,
                payload={
                    "model_type": self._model.model_type,
                    "model_id": self._model.model_id,
                    "count": count,
                },
            )
        )
        return count

    def __init__(
        self,
        model: Engine,
        memory: MemoryManager,
        tool: ToolManager,
        event_manager: EventManager,
        model_manager: ModelManager,
        engine_uri: EngineUri,
        *args: object,
        name: str | None = None,
        id: UUID | None = None,
    ) -> None:
        self._id = id or uuid4()
        self._name = name
        self._model = model
        self._memory = memory
        self._tool = tool
        self._event_manager = event_manager
        self._model_manager = model_manager
        self._engine_uri = engine_uri
        self._execution_context = ContextVar(
            f"engine-agent-execution-{self._id}",
            default=None,
        )
        self._legacy_output_context = ContextVar(
            f"engine-agent-output-{self._id}",
            default=None,
        )
        self._legacy_prompt_context = ContextVar(
            f"engine-agent-prompt-{self._id}",
            default=None,
        )
        self._pending_provider_cleanups: dict[
            int,
            _EngineProviderCleanup,
        ] = {}

    async def __call__(
        self,
        context: ModelCallContext,
    ) -> TextGenerationResponse | str:
        context = self._model_call_context_with_capability(context)
        token = self._execution_context.set(context.execution)
        try:
            return await self._call_with_execution(context)
        finally:
            self._execution_context.reset(token)

    async def _call_with_execution(
        self,
        context: ModelCallContext,
    ) -> TextGenerationResponse | str:
        """Execute one model call while its execution context is bound."""
        if context.parent and context.root_parent is None:
            root_parent_context = context.parent.root_parent or context.parent
            context = replace(context, root_parent=root_parent_context)

        await self._event_manager.trigger(
            Event(
                type=EventType.ENGINE_AGENT_CALL_BEFORE,
                payload={
                    "model_type": self._model.model_type,
                    "model_id": self._model.model_id,
                    "context": context,
                },
            )
        )

        await self._event_manager.trigger(
            Event(
                type=EventType.CALL_PREPARE_BEFORE,
                payload={
                    "model_type": self._model.model_type,
                    "model_id": self._model.model_id,
                    "specification": context.specification,
                    "input": context.input,
                    "context": context,
                },
            )
        )
        run_args = self._prepare_call(context)
        await self._event_manager.trigger(
            Event(
                type=EventType.CALL_PREPARE_AFTER,
                payload={
                    "model_type": self._model.model_type,
                    "model_id": self._model.model_id,
                    "specification": context.specification,
                    "input": context.input,
                    "context": context,
                },
            )
        )
        assert context.input is not None
        output = await self._run(context, context.input, **run_args)
        try:
            if isinstance(output, TextGenerationResponse):
                self._retain_provider_cleanup(output, context.execution)
            await self._event_manager.trigger(
                Event(
                    type=EventType.ENGINE_AGENT_CALL_AFTER,
                    payload={
                        "model_type": self._model.model_type,
                        "model_id": self._model.model_id,
                        "context": context,
                        "result": output,
                    },
                )
            )
        except BaseException as error:
            await self._settle_failed_output(
                output,
                context.execution,
                error,
                cancelled=isinstance(
                    error,
                    self._INTERRUPTED_EXIT_EXCEPTIONS,
                ),
            )
            raise
        return output

    async def _run(
        self,
        context: ModelCallContext,
        input: Input,
        instructions: str | None = None,
        system_prompt: str | None = None,
        developer_prompt: str | None = None,
        *args: object,
        settings: GenerationSettings | None = None,
        skip_special_tokens: bool = True,
        **kwargs: Any,
    ) -> TextGenerationResponse:
        context = self._model_call_context_with_capability(context)
        input_value = input
        generation_fields = self._GENERATION_FIELDS
        uri_defaults = {
            k: v
            for k, v in self._engine_uri.params.items()
            if k in generation_fields
        }
        if settings:
            settings_dict = {
                name: getattr(settings, name) for name in generation_fields
            }
            field_defaults = {
                name: field.default
                for name, field in generation_fields.items()
            }
            for key, value in uri_defaults.items():
                if settings_dict.get(key) == field_defaults[key]:
                    settings_dict[key] = value
        else:
            settings_dict = {**uri_defaults}
            settings_dict.setdefault("temperature", None)
            settings_dict.setdefault("do_sample", False)
        settings_dict.update(kwargs)
        settings_dict = self._normalize_generation_settings(settings_dict)
        settings = GenerationSettings(**settings_dict)
        assert settings

        # Prepare memory
        assert (
            not self._memory.has_recent_message
            or self._memory.recent_message is not None
        ) and (
            not self._memory.has_permanent_message
            or self._memory.permanent_message is not None
        )

        if isinstance(input_value, Message):
            input_value = [input_value]
        if (
            isinstance(input_value, list)
            and input_value
            and isinstance(input_value[0], str)
        ):
            input_messages: list[Message] = []
            for item in input_value:
                assert isinstance(item, str)
                input_messages.append(
                    Message(
                        role=MessageRole.USER,
                        content=item,
                    )
                )
            input_value = input_messages

        # Transform input (by adding memory, if necessary)
        should_use_memory = (
            context.execution is None
            and not context.parent
            and (
                self._memory.has_permanent_message
                or self._memory.has_recent_message
            )
        )
        if should_use_memory and isinstance(input_value, list):
            # Handle last message if not already consumed

            previous_message: Message | None = None
            previous_output = self._last_output
            if previous_output and isinstance(
                previous_output, TextGenerationResponse
            ):
                previous_message = Message(
                    role=MessageRole.ASSISTANT,
                    content=await previous_output.to_str(),
                )

                # Append messages

                if previous_message:
                    await self.sync_message(previous_message)

            for current_message in input_value:
                assert isinstance(current_message, Message)
                await self.sync_message(current_message)

            # Make recent memory the new model input
            assert self._memory.recent_messages is not None
            input_value = [rm.message for rm in self._memory.recent_messages]

        developer_prompt = self._developer_prompt_with_tool_bootstrap(
            developer_prompt
        )

        # Have model generate output from input
        if context.execution is not None:
            await context.execution.record_prompt(
                ModelPromptRecord(
                    input=input_value,
                    instructions=instructions,
                    system_prompt=system_prompt,
                    developer_prompt=developer_prompt,
                )
            )
        else:
            self._last_prompt = (
                input_value,
                instructions,
                system_prompt,
                developer_prompt,
            )

        operation = Operation(
            generation_settings=settings,
            input=input_value,
            modality=Modality.TEXT_GENERATION,
            parameters=OperationParameters(
                text=OperationTextParameters(
                    instructions=instructions,
                    system_prompt=system_prompt,
                    developer_prompt=developer_prompt,
                    skip_special_tokens=skip_special_tokens,
                )
            ),
            requires_input=True,
        )

        capability = context.capability
        assert capability is not None

        await self._event_manager.trigger(
            Event(
                type=EventType.MODEL_EXECUTE_BEFORE,
                payload={
                    "model_type": self._model.model_type,
                    "model_id": self._model.model_id,
                    "input": input_value,
                    "instructions": instructions,
                    "system_prompt": system_prompt,
                    "developer_prompt": developer_prompt,
                    "settings": settings,
                    "context": context,
                },
            )
        )
        model_task = ModelCall(
            engine_uri=self._engine_uri,
            model=self._model,
            operation=operation,
            capability=capability,
            context=context,
        )
        output = cast(
            TextGenerationResponse, await self._model_manager(model_task)
        )
        try:
            if isinstance(output, TextGenerationResponse):
                self._retain_provider_cleanup(output, context.execution)
            await self._event_manager.trigger(
                Event(
                    type=EventType.MODEL_EXECUTE_AFTER,
                    payload={
                        "model_type": self._model.model_type,
                        "model_id": self._model.model_id,
                        "input": input_value,
                        "instructions": instructions,
                        "system_prompt": system_prompt,
                        "developer_prompt": developer_prompt,
                        "settings": settings,
                        "context": context,
                    },
                )
            )

            # Update memory
            if context.execution is None and self._memory.has_recent_message:
                self._last_output = output
        except BaseException as error:
            await self._settle_failed_output(
                output,
                context.execution,
                error,
                cancelled=isinstance(
                    error,
                    self._INTERRUPTED_EXIT_EXCEPTIONS,
                ),
            )
            raise

        return output

    async def _settle_failed_output(
        self,
        output: object,
        execution: AgentExecution | None,
        primary_failure: BaseException,
        *,
        cancelled: bool,
    ) -> None:
        """Settle an engine exit whether or not a provider was returned."""
        if isinstance(output, TextGenerationResponse):
            try:
                await self._settle_unhanded_provider_response(
                    output,
                    execution,
                    primary_failure,
                    cancelled=cancelled,
                )
            except BaseException as cleanup_failure:
                provider_failures = [cleanup_failure]
                try:
                    provider_failures.extend(
                        await self._recover_failed_provider_cleanup(
                            output,
                            execution,
                            cancelled=cancelled,
                        )
                    )
                except BaseException as recovery_failure:
                    provider_failures.append(recovery_failure)
                self._attach_cleanup_failures(
                    primary_failure,
                    tuple(provider_failures),
                )
            return
        if execution is None:
            return
        try:
            secondary_failures = await execution.settle_provider_exit(
                cancelled=cancelled
            )
        except BaseException as error:
            secondary_failures = (error,)
        self._attach_cleanup_failures(primary_failure, secondary_failures)

    async def _recover_failed_provider_cleanup(
        self,
        response: TextGenerationResponse,
        execution: AgentExecution | None,
        *,
        cancelled: bool,
    ) -> tuple[BaseException, ...]:
        """Retry all cleanup stages after an unexpected setup failure."""
        failures: list[BaseException] = []
        response_id = id(response)
        owner = self._pending_provider_cleanups.get(response_id)
        if owner is not None and (
            owner.response is not response or owner.execution is not execution
        ):
            return (RuntimeError("provider cleanup ownership collision"),)
        if owner is None:
            try:
                owner = _EngineProviderCleanup(
                    response,
                    execution,
                    timeout_seconds=self._PROVIDER_CLEANUP_TIMEOUT_SECONDS,
                )
                self._pending_provider_cleanups[response_id] = owner
            except BaseException as error:
                failures.append(error)
                owner = None

        if owner is None:
            return tuple(failures)
        try:
            owner.require_cleanup(cancelled=cancelled)
            failures.extend(await owner.converge())
        except BaseException as error:
            failures.append(error)
        if owner.cleanup_complete:
            self._release_provider_cleanup(owner)
        return tuple(failures)

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

    async def _settle_unhanded_provider_response(
        self,
        response: TextGenerationResponse,
        execution: AgentExecution | None,
        primary_failure: BaseException,
        *,
        cancelled: bool,
    ) -> None:
        """Clean a provider response before ownership can be handed off."""
        owner = self._retain_provider_cleanup(response, execution)
        owner.require_cleanup(cancelled=cancelled)
        cleanup_task = create_task(owner.converge())
        while True:
            try:
                secondary_failures = await shield(cleanup_task)
                break
            except CancelledError:
                if cleanup_task.cancelled():
                    secondary_failures = (CancelledError(),)
                    break
            except BaseException as cleanup_failure:
                secondary_failures = (cleanup_failure,)
                break

        if owner.cleanup_complete:
            self._release_provider_cleanup(owner)
        self._attach_cleanup_failures(primary_failure, secondary_failures)

    def acknowledge_provider_handoff(
        self,
        response: TextGenerationResponse,
    ) -> None:
        """Release a provider only after its next owner registers it."""
        owner = self._pending_provider_cleanups.get(id(response))
        if (
            owner is not None
            and owner.response is response
            and not owner.cleanup_required
        ):
            self._pending_provider_cleanups.pop(id(response))

    async def drain_pending_provider_cleanups(
        self,
        execution: AgentExecution | None = None,
        *,
        abandon_unclaimed: bool = False,
    ) -> tuple[BaseException, ...]:
        """Bound and join retained cleanup without head-of-line blocking."""
        assert execution is None or isinstance(execution, AgentExecution)
        assert isinstance(abandon_unclaimed, bool)
        owners = tuple(self._pending_provider_cleanups.values())
        selected: list[_EngineProviderCleanup] = []
        for owner in owners:
            if execution is not None and owner.execution is not execution:
                continue
            if abandon_unclaimed and not owner.cleanup_required:
                owner.require_cleanup(cancelled=True)
            if owner.cleanup_required:
                selected.append(owner)
        if not selected:
            return ()

        results = await gather(
            *(self._drain_provider_cleanup(owner) for owner in selected),
            return_exceptions=True,
        )
        failures: list[BaseException] = []
        for result in results:
            if isinstance(result, BaseException):
                failures.append(result)
            else:
                failures.extend(result)
        return tuple(failures)

    async def _drain_provider_cleanup(
        self,
        owner: _EngineProviderCleanup,
    ) -> tuple[BaseException, ...]:
        """Drain and release one owner independently of every peer."""
        failures = await owner.converge()
        if owner.cleanup_complete:
            self._release_provider_cleanup(owner)
        return failures

    def _retain_provider_cleanup(
        self,
        response: TextGenerationResponse,
        execution: AgentExecution | None,
    ) -> _EngineProviderCleanup:
        """Return the unique retained owner for one raw provider response."""
        response_id = id(response)
        owner = self._pending_provider_cleanups.get(response_id)
        if owner is not None:
            if (
                owner.response is not response
                or owner.execution is not execution
            ):
                raise RuntimeError("provider cleanup ownership collision")
            return owner
        owner = _EngineProviderCleanup(
            response,
            execution,
            timeout_seconds=self._PROVIDER_CLEANUP_TIMEOUT_SECONDS,
        )
        self._pending_provider_cleanups[response_id] = owner

        def acknowledge_consumed() -> None:
            self.acknowledge_provider_handoff(response)

        response.add_done_callback(acknowledge_consumed)
        return owner

    def _release_provider_cleanup(
        self,
        owner: _EngineProviderCleanup,
    ) -> None:
        """Release exactly one converged provider cleanup owner."""
        response_id = id(owner.response)
        if self._pending_provider_cleanups.get(response_id) is owner:
            self._pending_provider_cleanups.pop(response_id)

    def _model_call_context_with_capability(
        self,
        context: ModelCallContext,
    ) -> ModelCallContext:
        """Return a per-call context bound to one capability catalog."""
        if context.capability is not None:
            return context
        capability = ModelCapabilityCatalog.create(
            self._tool.export_model_capability_seed()
        )
        return replace(context, capability=capability)

    def _developer_prompt_with_tool_bootstrap(
        self,
        developer_prompt: str | None,
    ) -> str | None:
        bootstrap_prompt = self._tool_bootstrap_prompt()
        if not bootstrap_prompt:
            return developer_prompt
        if developer_prompt:
            return f"{developer_prompt}\n\n{bootstrap_prompt}"
        return bootstrap_prompt

    def _tool_bootstrap_prompt(self) -> str | None:
        bootstrap_prompt = getattr(self._tool, "bootstrap_prompt", None)
        if not callable(bootstrap_prompt):
            return None
        prompt = bootstrap_prompt()
        if isinstance(prompt, str) and prompt:
            return prompt
        return None

    @staticmethod
    def _normalize_generation_settings(
        settings_dict: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(settings_dict)
        chat_settings = normalized.get("chat_settings")
        if isinstance(chat_settings, dict):
            normalized["chat_settings"] = ChatSettings(**chat_settings)

        reasoning = normalized.get("reasoning")
        if isinstance(reasoning, dict):
            reasoning_config = dict(reasoning)
            effort = reasoning_config.get("effort")
            if isinstance(effort, str):
                reasoning_config["effort"] = ReasoningEffort(effort)
            summary = reasoning_config.get("summary")
            if type(summary) is str:
                reasoning_config["summary"] = ReasoningSummaryMode(summary)
            tag = reasoning_config.get("tag")
            if isinstance(tag, str):
                reasoning_config["tag"] = ReasoningTag(tag)
            normalized["reasoning"] = ReasoningSettings(**reasoning_config)

        return normalized

    async def sync_messages(
        self,
        execution: AgentExecution | None = None,
    ) -> None:
        execution = execution or self._execution_context.get()
        if execution is not None:
            if not (
                self._memory.has_permanent_message
                or self._memory.has_recent_message
            ):
                return
            await execution.sync_memory(self)
            return
        if self._last_output and (
            self._memory.has_permanent_message
            or self._memory.has_recent_message
        ):
            previous_message = Message(
                role=MessageRole.ASSISTANT,
                content=await self._last_output.to_str(
                    raise_terminal_exception=False
                ),
            )
            await self.sync_message(previous_message)

    async def append_execution_memory_entry(
        self,
        entry: ExecutionMemoryEntry,
    ) -> None:
        """Persist one keyed execution-ledger memory entry."""
        if not isinstance(entry, ExecutionMemoryEntry):
            raise TypeError("entry must be an execution-memory entry")
        await self.sync_message(
            entry.message,
            idempotency_key=entry.idempotency_key,
        )

    async def sync_message(
        self,
        message: Message,
        *,
        idempotency_key: EngineMessageIdempotencyKey | None = None,
    ) -> None:
        await self._event_manager.trigger(
            Event(
                type=EventType.MEMORY_APPEND_BEFORE,
                payload={
                    "model_type": self._model.model_type,
                    "model_id": self._model.model_id,
                    "message": message,
                    "participant_id": getattr(
                        self._memory, "participant_id", None
                    ),
                    "session_id": (
                        getattr(self._memory, "permanent_message", None)
                        and getattr(
                            self._memory.permanent_message,
                            "session_id",
                            None,
                        )
                    ),
                },
            )
        )
        assert self._model.model_id is not None
        await self._memory.append_message(
            EngineMessage(
                agent_id=self._id,
                model_id=self._model.model_id,
                message=message,
                idempotency_key=idempotency_key,
            )
        )
        await self._event_manager.trigger(
            Event(
                type=EventType.MEMORY_APPEND_AFTER,
                payload={
                    "model_type": self._model.model_type,
                    "model_id": self._model.model_id,
                    "message": message,
                    "participant_id": getattr(
                        self._memory, "participant_id", None
                    ),
                    "session_id": (
                        getattr(self._memory, "permanent_message", None)
                        and getattr(
                            self._memory.permanent_message,
                            "session_id",
                            None,
                        )
                    ),
                },
            )
        )
