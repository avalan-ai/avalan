from ....cli import CommandAbortException
from ....entities import (
    Input,
    Message,
    MessageRole,
    MessageToolCall,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallOutcome,
    ToolCallResult,
    ToolCallToken,
)
from ....event import Event, EventType
from ....event.manager import EventManager
from ....model.call import ModelCallContext
from ....model.response.parsers.tool import ToolCallResponseParser
from ....model.response.text import TextGenerationResponse
from ....tool.manager import ToolManager
from ....utils import tool_call_diagnostic_payload
from ... import AgentOperation
from ...engine import EngineAgent

from base64 import b64encode
from dataclasses import asdict, is_dataclass
from inspect import iscoroutine
from json import dumps, loads
from queue import Queue
from time import perf_counter
from typing import Any, AsyncIterator, Awaitable, Callable, cast
from uuid import UUID, uuid4


class OrchestratorResponse(AsyncIterator[Token | TokenDetail | Event]):
    """Async iterator handling tool execution during streaming."""

    _MAXIMUM_TOOL_CYCLES = 8
    _MAXIMUM_CONSECUTIVE_NON_EXECUTED_CYCLES = 2

    _response: TextGenerationResponse
    _response_iterator: AsyncIterator[Token | TokenDetail | str] | None
    _engine_agent: EngineAgent
    _operation: AgentOperation
    _engine_args: dict[str, Any]
    _event_manager: EventManager | None
    _tool_manager: ToolManager | None
    _calls: Queue[ToolCall]
    _tool_call_events: Queue[Event]
    _tool_process_events: Queue[Event]
    _tool_result_events: Queue[Event]
    _input: Input
    _context: ModelCallContext
    _tool_context: ToolCallContext | None
    _call_history: list[ToolCall]
    _attempted_call_signatures: set[str]
    _tool_cycle_signatures: set[str]
    _tool_cycle_count: int
    _consecutive_non_executed_cycles: int
    _agent_id: UUID | None
    _participant_id: UUID | None
    _session_id: UUID | None
    _parser_queue: Queue[Token | TokenDetail | Event] | None
    _tool_parser: ToolCallResponseParser | None
    _cancellation_checker: Callable[[], Awaitable[None]] | None

    def __init__(
        self,
        input: Input,
        response: TextGenerationResponse,
        engine_agent: EngineAgent,
        operation: AgentOperation,
        engine_args: dict[str, Any],
        context: ModelCallContext,
        event_manager: EventManager | None = None,
        tool: ToolManager | None = None,
        *,
        agent_id: UUID | None = None,
        participant_id: UUID | None = None,
        session_id: UUID | None = None,
        tool_confirm: Callable[[ToolCall], str | None] | None = None,
        enable_tool_parsing: bool = True,
    ) -> None:
        assert input and response and engine_agent and operation
        self._input = input
        self._response = response
        self._engine_agent = engine_agent
        self._operation = operation
        self._engine_args = engine_args
        self._event_manager = event_manager
        self._tool_manager = None if tool and tool.is_empty else tool
        self._context = context
        self._finished = False
        self._step = 0
        self._tool_context = None
        self._call_history = []
        self._attempted_call_signatures = set()
        self._tool_cycle_signatures = set()
        self._tool_cycle_count = 0
        self._consecutive_non_executed_cycles = 0
        self._agent_id = agent_id
        self._participant_id = participant_id
        self._session_id = session_id
        self._tool_confirm = tool_confirm
        self._tool_confirm_all = False
        self._parser_queue = Queue()
        self._cancellation_checker = None
        self._model_responses = [response]
        self._tool_parser = (
            ToolCallResponseParser(self._tool_manager, self._event_manager)
            if enable_tool_parsing and self._tool_manager
            else None
        )

    @property
    def input_token_count(self) -> int:
        return self._response.input_token_count

    @property
    def output_token_count(self) -> int:
        return self._response.output_token_count

    @property
    def usage(self) -> object | None:
        usage_values = tuple(
            usage
            for response in self._model_responses
            if (usage := response.usage) is not None
        )
        if not usage_values:
            return None
        if len(usage_values) == 1:
            return usage_values[0]
        return usage_values

    @property
    def usage_responses(self) -> tuple[TextGenerationResponse, ...]:
        return tuple(self._model_responses)

    @property
    def can_think(self) -> bool:
        return self._response.can_think

    def set_cancellation_checker(
        self,
        checker: Callable[[], Awaitable[None]] | None,
    ) -> None:
        self._cancellation_checker = checker

    @property
    def is_thinking(self) -> bool:
        return self._response.is_thinking

    def set_thinking(self, thinking: bool) -> None:
        self._response.set_thinking(thinking)

    async def to_str(self) -> str:
        output = await self._react(self._response)
        return output

    async def to_json(self) -> str:
        output = await self._react(self._response)
        return TextGenerationResponse.extract_json(output)

    async def to(self, entity_class: type) -> Any:
        json = await self.to_json()
        return entity_class(**loads(json))

    def __aiter__(self) -> "OrchestratorResponse":
        if self._event_manager:
            self._response.add_done_callback(self._on_consumed)
        self._response_iterator = aiter(self._response)
        self._calls = Queue()
        self._parser_queue = Queue()
        self._tool_context = ToolCallContext(
            input=self._input,
            agent_id=self._agent_id,
            participant_id=self._participant_id,
            session_id=self._session_id,
            calls=list(self._call_history),
            cancellation_checker=self._cancellation_checker,
        )
        self._tool_call_events = Queue()
        self._tool_process_events = Queue()
        self._tool_result_events = Queue()
        self._step = 0
        return self

    async def __anext__(self) -> Token | TokenDetail | Event:
        assert self._response_iterator

        if self._parser_queue and not self._parser_queue.empty():
            return self._parser_queue.get()

        if not self._tool_process_events.empty():
            event = self._tool_process_events.get()
            assert event.type == EventType.TOOL_PROCESS
            self._tool_call_events.put(event)
            return event

        if not self._tool_call_events.empty():
            event = self._tool_call_events.get()
            assert event.type == EventType.TOOL_PROCESS
            assert self._event_manager
            await self._event_manager.trigger(event)

            calls = cast(list[ToolCall], event.payload or [])
            if calls:
                for call in calls:
                    assert isinstance(call, ToolCall)
                    self._calls.put(call)

        if not self._calls.empty():
            call = self._calls.get()

            if self._tool_confirm and not self._tool_confirm_all:
                action = self._tool_confirm(call)
                if iscoroutine(action):
                    action = await action
                if action == "a":
                    self._tool_confirm_all = True
                elif action != "y":
                    raise CommandAbortException()

            start = perf_counter()
            execute_event = Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": call},
                started=start,
            )
            if self._event_manager:
                await self._event_manager.trigger(execute_event)

            context = ToolCallContext(
                input=self._tool_context.input if self._tool_context else None,
                agent_id=self._agent_id,
                participant_id=self._participant_id,
                session_id=self._session_id,
                calls=list(self._call_history),
                cancellation_checker=self._cancellation_checker,
            )

            result = await self._execute_tool_call(
                call,
                context,
                confirm=False,
            )

            self._record_tool_outcome(result)
            self._tool_context = context

            end = perf_counter()
            result_event = Event(
                type=EventType.TOOL_RESULT,
                payload={"call": call, "result": result},
                started=start,
                finished=end,
                elapsed=end - start,
            )
            if self._event_manager:
                await self._trigger_tool_diagnostic_event(
                    call=call,
                    result=result,
                    started=start,
                    finished=end,
                    elapsed=end - start,
                )
                await self._event_manager.trigger(result_event)

            self._tool_result_events.put(result_event)

            return result_event

        # Wait until all results are collected
        if (
            self._tool_call_events.empty()
            and self._calls.empty()
            and not self._tool_result_events.empty()
        ):
            result_events: list[Event] = []
            while not self._tool_result_events.empty():
                result_event = self._tool_result_events.get()
                result_events.append(result_event)

            tool_messages = []
            outcomes = []
            for e in result_events:
                assert e.payload is not None and "result" in e.payload
                tool_result = e.payload["result"]
                event_call = e.payload.get("call")
                if not isinstance(
                    tool_result,
                    (ToolCallResult, ToolCallError, ToolCallDiagnostic),
                ):
                    continue
                outcomes.append(tool_result)
                tool_messages.extend(
                    self._tool_observation_messages(
                        tool_result,
                        call=(
                            event_call
                            if isinstance(event_call, ToolCall)
                            else None
                        ),
                        json_output=True,
                    )
                )

            if not self._should_continue_tool_cycle(
                tool_messages,
                outcomes,
            ):
                if self._event_manager and not self._finished:
                    self._finished = True
                    await self._event_manager.trigger(
                        Event(type=EventType.END)
                    )
                raise StopAsyncIteration

            assert self._input and (
                (
                    isinstance(self._input, list)
                    and isinstance(self._input[0], Message)
                )
                or isinstance(self._input, Message)
            )

            messages = (
                list(cast(list[Message], self._input))
                if isinstance(self._input, list)
                else [self._input]
            )

            messages.extend(tool_messages)
            self._input = cast(Input, messages)
            self._tool_context = ToolCallContext(
                input=self._input,
                agent_id=self._agent_id,
                participant_id=self._participant_id,
                session_id=self._session_id,
                calls=list(self._call_history),
                cancellation_checker=self._cancellation_checker,
            )

            event_tool_model_run = Event(
                type=EventType.TOOL_MODEL_RUN,
                payload={
                    "model_id": self._engine_agent.engine.model_id,
                    "messages": messages,
                    "engine_args": self._engine_args,
                },
            )
            assert self._event_manager
            await self._event_manager.trigger(event_tool_model_run)

            model_context = self._make_child_context(messages)
            inner_response = await self._engine_agent(model_context)
            assert inner_response
            assert isinstance(inner_response, TextGenerationResponse)

            self._model_responses.append(inner_response)
            self._response = inner_response
            self.__aiter__()

            event_tool_model_response = Event(
                type=EventType.TOOL_MODEL_RESPONSE,
                payload={
                    "response": inner_response,
                    "model_id": self._engine_agent.engine.model_id,
                    "messages": messages,
                    "engine_args": self._engine_args,
                },
            )
            assert self._event_manager
            await self._event_manager.trigger(event_tool_model_response)

            return event_tool_model_response

        try:
            token = await self._response_iterator.__anext__()
            if isinstance(token, ToolCallToken) and token.call:
                event = Event(
                    type=EventType.TOOL_PROCESS,
                    payload=cast(dict[str, Any], [token.call]),
                    started=perf_counter(),
                )
                self._tool_process_events.put(event)
        except StopAsyncIteration:
            if self._tool_parser:
                parser_items: list[Token | TokenDetail | Event] = []
                parser_events: list[Event] = []
                for item in await self._tool_parser.flush():
                    if isinstance(item, Event):
                        if item.type == EventType.TOOL_PROCESS:
                            self._tool_process_events.put(item)
                        elif item.type == EventType.TOOL_DIAGNOSTIC:
                            parser_events.append(item)
                        else:
                            self._tool_process_events.put(item)
                    else:
                        parser_items.append(item)
                assert self._parser_queue
                for item in parser_items:
                    self._parser_queue.put(item)
                for event in parser_events:
                    self._parser_queue.put(event)
                if self._parser_queue and not self._parser_queue.empty():
                    return self._parser_queue.get()
                if not self._tool_process_events.empty():
                    event = self._tool_process_events.get()
                    assert event.type == EventType.TOOL_PROCESS
                    self._tool_call_events.put(event)
                    return event
            if self._event_manager and not self._finished:
                self._finished = True
                await self._event_manager.trigger(Event(type=EventType.END))

            raise

        return await self._emit(token)

    async def _react(
        self, response: TextGenerationResponse, output: str | None = None
    ) -> str:
        if self._event_manager:
            response.add_done_callback(self._on_consumed)

        structured_calls: list[ToolCall] = []
        if output is None:
            text, structured_calls = await self._response_text_and_calls(
                response
            )
        else:
            text = output

        if self._tool_context is None:
            self._tool_context = ToolCallContext(
                input=self._input,
                agent_id=self._agent_id,
                participant_id=self._participant_id,
                session_id=self._session_id,
                calls=list(self._call_history),
                cancellation_checker=self._cancellation_checker,
            )

        if not self._tool_manager:
            self._response = response
            return text

        current_response = response
        previous_text = text
        delta = text
        while True:
            if self._event_manager:
                await self._event_manager.trigger(
                    Event(type=EventType.TOOL_DETECT)
                )

            calls = (
                structured_calls
                if structured_calls
                else (
                    self._tool_manager.get_calls(delta)
                    if self._tool_manager
                    else None
                )
            )
            if not calls:
                break

            results: list[ToolCallOutcome] = []
            for call in calls:
                if self._event_manager:
                    start = perf_counter()
                    execute_event = Event(
                        type=EventType.TOOL_EXECUTE,
                        payload={"call": call},
                        started=start,
                    )
                    await self._event_manager.trigger(execute_event)

                context = ToolCallContext(
                    input=self._tool_context.input,
                    agent_id=self._agent_id,
                    participant_id=self._participant_id,
                    session_id=self._session_id,
                    calls=list(self._call_history),
                    cancellation_checker=self._cancellation_checker,
                )

                result = await self._execute_tool_call(
                    call,
                    context,
                    confirm=True,
                )
                self._record_tool_outcome(result)
                self._tool_context = context
                if result is not None:
                    results.append(result)

                if self._event_manager:
                    end = perf_counter()
                    result_event = Event(
                        type=EventType.TOOL_RESULT,
                        payload={"result": result},
                        started=start,
                        finished=end,
                        elapsed=end - start,
                    )
                    await self._trigger_tool_diagnostic_event(
                        call=call,
                        result=result,
                        started=start,
                        finished=end,
                        elapsed=end - start,
                    )
                    await self._event_manager.trigger(result_event)

            next_response = await self._react_process(delta, results)
            if next_response is None:
                break
            current_response = next_response
            new_text, structured_calls = await self._response_text_and_calls(
                current_response
            )
            delta = new_text.replace(previous_text, "")
            previous_text = new_text

        self._response = current_response
        return delta

    async def _response_text_and_calls(
        self,
        response: TextGenerationResponse,
    ) -> tuple[str, list[ToolCall]]:
        if not response.is_async_generator:
            return await response.to_str(), []

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        async for item in response:
            if isinstance(item, Event):
                continue
            await self._emit_token_generated_event(item)
            self._step += 1
            if isinstance(item, ToolCallToken):
                if item.call is not None:
                    calls.append(item.call)
                continue
            text_parts.append(
                item.token if hasattr(item, "token") else str(item)
            )
        return "".join(text_parts), calls

    async def _emit_token_generated_event(
        self,
        item: Token | TokenDetail | str,
    ) -> None:
        if not self._should_emit_token_generated_event():
            return
        token_str = item.token if hasattr(item, "token") else str(item)
        token_id = getattr(item, "id", None)
        tokenizer = (
            self._engine_agent.engine.tokenizer
            if self._engine_agent.engine
            else None
        )
        if token_id is None and tokenizer and self._should_enrich_token_ids():
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            token_id = ids[0] if ids else None

        assert self._event_manager
        await self._event_manager.trigger(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload={
                    "token_id": token_id,
                    "model_id": self._engine_agent.engine.model_id,
                    "token": token_str,
                    "token_type": type(item).__qualname__,
                    "step": self._step,
                },
            )
        )

    def _should_emit_token_generated_event(self) -> bool:
        if not self._event_manager:
            return False
        should_emit = getattr(self._event_manager, "should_emit", None)
        if not callable(should_emit):
            return True
        result = should_emit(EventType.TOKEN_GENERATED)
        return result if isinstance(result, bool) else False

    def _should_enrich_token_ids(self) -> bool:
        if not self._event_manager:
            return False
        value = getattr(self._event_manager, "enrich_token_ids", False)
        return value if isinstance(value, bool) else False

    async def _execute_tool_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
        *,
        confirm: bool,
    ) -> ToolCallOutcome | None:
        if self._tool_manager is None:
            return None
        repeated_diagnostic = self._repeated_call_diagnostic(call)
        if repeated_diagnostic is not None:
            return repeated_diagnostic

        self._attempted_call_signatures.add(self._call_signature(call))
        if type(self._tool_manager) is ToolManager:
            confirmation = self._tool_confirm if confirm else None
            return await self._tool_manager.execute_call(
                call,
                context,
                confirm=confirmation,
            )
        return await self._tool_manager(call, context)

    async def _react_process(
        self, output: str, results: list[ToolCallOutcome]
    ) -> TextGenerationResponse | None:
        tool_messages: list[Message] = []
        for result in results:
            tool_messages.extend(
                self._tool_observation_messages(
                    result,
                    json_output=False,
                )
            )

        if not self._should_continue_tool_cycle(tool_messages, results):
            return None

        assert self._input and (
            (
                isinstance(self._input, list)
                and isinstance(self._input[0], Message)
            )
            or isinstance(self._input, Message)
        )

        messages = (
            list(cast(list[Message], self._input))
            if isinstance(self._input, list)
            else [self._input]
        )
        messages.extend(tool_messages)

        self._input = cast(Input, messages)
        self._tool_context = ToolCallContext(
            input=self._input,
            agent_id=self._agent_id,
            participant_id=self._participant_id,
            session_id=self._session_id,
            calls=list(self._call_history),
            cancellation_checker=self._cancellation_checker,
        )

        event_tool_model_run = Event(
            type=EventType.TOOL_MODEL_RUN,
            payload={
                "model_id": self._engine_agent.engine.model_id,
                "messages": messages,
                "engine_args": self._engine_args,
            },
        )
        assert self._event_manager
        await self._event_manager.trigger(event_tool_model_run)

        context = self._make_child_context(messages)
        response = await self._engine_agent(context)
        assert response
        assert isinstance(response, TextGenerationResponse)
        self._model_responses.append(response)
        return response

    def _should_continue_tool_cycle(
        self,
        tool_messages: list[Message],
        outcomes: list[ToolCallOutcome],
    ) -> bool:
        if not tool_messages:
            return False

        cycle_signature = self._tool_cycle_signature(tool_messages)
        if cycle_signature in self._tool_cycle_signatures:
            return False

        if self._tool_cycle_count >= self._MAXIMUM_TOOL_CYCLES:
            return False

        non_executed = bool(outcomes) and all(
            isinstance(outcome, ToolCallDiagnostic) for outcome in outcomes
        )
        if non_executed:
            self._consecutive_non_executed_cycles += 1
        else:
            self._consecutive_non_executed_cycles = 0

        if (
            self._consecutive_non_executed_cycles
            > self._MAXIMUM_CONSECUTIVE_NON_EXECUTED_CYCLES
        ):
            return False

        self._tool_cycle_signatures.add(cycle_signature)
        self._tool_cycle_count += 1
        return True

    def _record_tool_outcome(self, result: ToolCallOutcome | None) -> None:
        if isinstance(result, (ToolCallResult, ToolCallError)):
            self._call_history.append(result.call)

    async def _trigger_tool_diagnostic_event(
        self,
        *,
        call: ToolCall,
        result: ToolCallOutcome | None,
        started: float,
        finished: float,
        elapsed: float,
    ) -> None:
        if not isinstance(result, ToolCallDiagnostic):
            return
        assert self._event_manager
        await self._event_manager.trigger(
            Event(
                type=EventType.TOOL_DIAGNOSTIC,
                payload={
                    "call": call,
                    "diagnostic": result,
                    "diagnostics": [result],
                    "result": result,
                },
                started=started,
                finished=finished,
                elapsed=elapsed,
            )
        )

    def _repeated_call_diagnostic(
        self, call: ToolCall
    ) -> ToolCallDiagnostic | None:
        if self._call_signature(call) not in self._attempted_call_signatures:
            return None
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=call.name,
            code=ToolCallDiagnosticCode.REPEATED_CALL,
            stage=ToolCallDiagnosticStage.GUARD,
            message="Tool call repeats a previous attempt.",
        )

    @staticmethod
    def _call_signature(call: ToolCall) -> str:
        return dumps(
            {
                "arguments": call.arguments,
                "name": call.name,
            },
            default=str,
            sort_keys=True,
        )

    @classmethod
    def _tool_cycle_signature(cls, messages: list[Message]) -> str:
        payload = [
            cls._tool_cycle_message_payload(message) for message in messages
        ]
        return dumps(
            payload,
            default=str,
            sort_keys=True,
        )

    @classmethod
    def _tool_cycle_message_payload(cls, message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "arguments": message.arguments,
            "content": message.content,
            "name": message.name,
            "role": message.role.value,
            "thinking": message.thinking,
            "tool_calls": (
                [asdict(tool_call) for tool_call in message.tool_calls]
                if message.tool_calls
                else None
            ),
        }
        if message.tool_call_result is not None:
            result = message.tool_call_result
            payload["tool_call_result"] = {
                "arguments": result.arguments,
                "call": cls._call_signature(result.call),
                "name": result.name,
                "result": cls._json_content(result.result),
            }
        if message.tool_call_error is not None:
            error = message.tool_call_error
            payload["tool_call_error"] = {
                "arguments": error.arguments,
                "call": cls._call_signature(error.call),
                "message": error.message,
                "name": error.name,
            }
        if message.tool_call_diagnostic is not None:
            diagnostic = message.tool_call_diagnostic
            payload["tool_call_diagnostic"] = {
                "call_id": diagnostic.call_id,
                "canonical_name": diagnostic.canonical_name,
                "code": diagnostic.code.value,
                "details": diagnostic.details,
                "message": diagnostic.message,
                "requested_name": diagnostic.requested_name,
                "retryable": diagnostic.retryable,
                "stage": diagnostic.stage.value,
                "status": diagnostic.status.value,
            }
        return payload

    @classmethod
    def _tool_observation_messages(
        cls,
        outcome: ToolCallOutcome,
        *,
        call: ToolCall | None = None,
        json_output: bool,
    ) -> list[Message]:
        if isinstance(outcome, ToolCallDiagnostic):
            return cls._diagnostic_messages(
                outcome,
                call=call,
                json_output=json_output,
            )

        return [
            Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[
                    MessageToolCall(
                        id=str(outcome.call.id),
                        name=outcome.name,
                        arguments=cast(Any, outcome.arguments),
                    )
                ],
            ),
            Message(
                role=MessageRole.TOOL,
                name=outcome.name,
                arguments=outcome.arguments,
                content=cls._outcome_content(
                    outcome,
                    json_output=json_output,
                ),
                tool_call_result=(
                    outcome if isinstance(outcome, ToolCallResult) else None
                ),
                tool_call_error=(
                    outcome if isinstance(outcome, ToolCallError) else None
                ),
            ),
        ]

    @classmethod
    def _diagnostic_messages(
        cls,
        diagnostic: ToolCallDiagnostic,
        *,
        call: ToolCall | None,
        json_output: bool,
    ) -> list[Message]:
        call_id = diagnostic.call_id or (call.id if call else None)
        name = (
            diagnostic.canonical_name
            or diagnostic.requested_name
            or (call.name if call else "tool")
        )
        arguments = call.arguments if call else None
        content = cls._outcome_content(
            diagnostic,
            json_output=json_output,
        )
        if call_id is None:
            return [
                Message(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    tool_call_diagnostic=diagnostic,
                )
            ]

        return [
            Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[
                    MessageToolCall(
                        id=str(call_id),
                        name=name,
                        arguments=cast(Any, arguments),
                    )
                ],
            ),
            Message(
                role=MessageRole.TOOL,
                name=name,
                arguments=arguments,
                content=content,
                tool_call_diagnostic=diagnostic,
            ),
        ]

    @classmethod
    def _outcome_content(
        cls,
        outcome: ToolCallOutcome,
        *,
        json_output: bool,
    ) -> str:
        if isinstance(outcome, ToolCallDiagnostic):
            return cls._json_content(tool_call_diagnostic_payload(outcome))
        if isinstance(outcome, ToolCallError):
            return (
                cls._json_content(outcome.message)
                if json_output
                else outcome.message
            )

        result = outcome.result
        if not json_output and isinstance(result, str):
            return result
        if not json_output and result is None:
            return ""
        return cls._json_content(result)

    @staticmethod
    def _json_content(value: Any) -> str:
        return dumps(
            asdict(cast(Any, value)) if is_dataclass(value) else value,
            default=lambda o: (
                b64encode(o).decode()
                if isinstance(o, (bytes, bytearray, memoryview))
                else str(o)
            ),
        )

    async def _emit(
        self, item: Token | TokenDetail | Event | str
    ) -> Token | TokenDetail | Event:
        if self._event_manager and not isinstance(item, Event):
            await self._emit_token_generated_event(item)

        self._step += 1

        if isinstance(item, Event):
            if item.type == EventType.TOOL_PROCESS:
                self._tool_process_events.put(item)
                return await self.__anext__()
            return item

        if isinstance(item, str) and self._tool_parser:
            items = await self._tool_parser.push(item)
            if not items:
                return await self.__anext__()
        else:
            items = [item]

        for it in items:
            if isinstance(it, Event):
                if it.type == EventType.TOOL_PROCESS:
                    self._tool_process_events.put(it)
                else:
                    assert self._parser_queue
                    self._parser_queue.put(it)
            else:
                assert self._parser_queue
                self._parser_queue.put(it)

        assert self._parser_queue
        return self._parser_queue.get()

    async def _on_consumed(self) -> None:
        assert self._event_manager
        await self._event_manager.trigger(Event(type=EventType.STREAM_END))

    def _make_child_context(self, messages: Input) -> ModelCallContext:
        parent_context = self._context
        root_parent = (
            parent_context.root_parent or parent_context
            if parent_context
            else None
        )
        context = ModelCallContext(
            specification=self._operation.specification,
            input=messages,
            engine_args=dict(self._engine_args),
            parent=parent_context,
            root_parent=root_parent,
            agent_id=(
                parent_context.agent_id if parent_context else self._agent_id
            ),
            participant_id=(
                parent_context.participant_id
                if parent_context
                else self._participant_id
            ),
            session_id=(
                parent_context.session_id
                if parent_context
                else self._session_id
            ),
        )
        self._context = context
        return context
