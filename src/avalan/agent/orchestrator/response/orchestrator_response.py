from ... import Operation
from ...engine import EngineAgent
from ....entities import (
    Input,
    Message,
    MessageRole,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
)
from ....event import Event, EventType
from ....event.manager import EventManager
from ....model import TextGenerationResponse
from ....tool.manager import ToolManager
from ....cli import CommandAbortException
from queue import Queue
from inspect import iscoroutine
from io import StringIO
from time import perf_counter
from typing import Any, AsyncIterator, Callable
from uuid import UUID


class OrchestratorResponse(AsyncIterator[Token | TokenDetail | Event]):
    """Async iterator handling tool execution during streaming."""

    _response: TextGenerationResponse
    _response_iterator: AsyncIterator[Token | TokenDetail | Event] | None
    _engine_agent: EngineAgent
    _operation: Operation
    _engine_args: dict
    _event_manager: EventManager | None
    _tool_manager: ToolManager | None
    _buffer: StringIO
    _calls: Queue[ToolCall]
    _tool_call_events: Queue[Event]
    _tool_process_events: Queue[Event]
    _tool_result_events: Queue[Event]
    _input: Input
    _tool_context: ToolCallContext | None
    _call_history: list[ToolCall]
    _agent_id: UUID | None
    _participant_id: UUID | None
    _session_id: UUID | None

    def __init__(
        self,
        input: Input,
        response: TextGenerationResponse,
        engine_agent: EngineAgent,
        operation: Operation,
        engine_args: dict,
        event_manager: EventManager | None = None,
        tool: ToolManager | None = None,
        *,
        agent_id: UUID | None = None,
        participant_id: UUID | None = None,
        session_id: UUID | None = None,
        tool_confirm: Callable[[ToolCall], str | None] | None = None,
    ) -> None:
        assert input and response and engine_agent and operation
        self._input = input
        self._response = response
        self._engine_agent = engine_agent
        self._operation = operation
        self._engine_args = engine_args
        self._event_manager = event_manager
        self._tool_manager = None if tool and tool.is_empty else tool
        self._finished = False
        self._step = 0
        self._tool_context = None
        self._call_history = []
        self._agent_id = agent_id
        self._participant_id = participant_id
        self._session_id = session_id
        self._tool_confirm = tool_confirm
        self._tool_confirm_all = False

    @property
    def input_token_count(self) -> int:
        return self._response.input_token_count

    async def to_str(self) -> str:
        output = await self._react(self._response)
        return output

    async def to_json(self) -> str:
        await self._react(self._response)
        return await self._response.to_json()

    async def to(self, entity_class: type) -> Any:
        await self._react(self._response)
        return await self._response.to(entity_class)

    def __aiter__(self) -> "OrchestratorResponse":
        if self._event_manager:
            self._response.add_done_callback(self._on_consumed)
        self._response_iterator = self._response.__aiter__()
        self._buffer = StringIO()
        self._calls = Queue()
        self._tool_context = ToolCallContext(
            input=self._input,
            agent_id=self._agent_id,
            participant_id=self._participant_id,
            session_id=self._session_id,
            calls=list(self._call_history),
        )
        self._tool_call_events = Queue()
        self._tool_process_events = Queue()
        self._tool_result_events = Queue()
        self._step = 0
        return self

    async def __anext__(self) -> Token | TokenDetail | Event:
        assert self._response_iterator

        if not self._tool_process_events.empty():
            event = self._tool_process_events.get()
            assert event.type == EventType.TOOL_PROCESS
            self._tool_call_events.put(event)
            return event

        if not self._tool_call_events.empty():
            event = self._tool_call_events.get()
            assert event.type == EventType.TOOL_PROCESS
            await self._event_manager.trigger(event)

            calls: list[ToolCall] = event.payload or []
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
                input=self._tool_context.input,
                agent_id=self._agent_id,
                participant_id=self._participant_id,
                session_id=self._session_id,
                calls=list(self._call_history),
            )

            result = (
                await self._tool_manager(call, context)
                if self._tool_manager
                else None
            )

            self._call_history.append(call)
            self._tool_context = context

            end = perf_counter()
            result_event = Event(
                type=EventType.TOOL_RESULT,
                payload={"result": result},
                started=start,
                finished=end,
                ellapsed=end - start,
            )
            if self._event_manager:
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

            tool_messages = [
                Message(
                    role=MessageRole.TOOL,
                    name=e.payload["result"].name,
                    arguments=e.payload["result"].arguments,
                    content=e.payload["result"].result,
                )
                for e in result_events
            ]

            assert self._input and (
                (
                    isinstance(self._input, list)
                    and isinstance(self._input[0], Message)
                )
                or isinstance(self._input, Message)
            )

            messages = list(
                self._input if isinstance(self._input, list) else [self._input]
            )
            messages.extend(tool_messages)

            event_tool_model_run = Event(
                type=EventType.TOOL_MODEL_RUN,
                payload={
                    "model_id": self._engine_agent.engine.model_id,
                    "messages": messages,
                    "engine_args": self._engine_args,
                },
            )
            await self._event_manager.trigger(event_tool_model_run)

            inner_response = await self._engine_agent(
                self._operation.specification,
                messages,
                **self._engine_args,
            )
            assert inner_response

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
            await self._event_manager.trigger(event_tool_model_response)
            return event_tool_model_response

        try:
            token = await self._response_iterator.__anext__()
        except StopAsyncIteration:
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

        text = output or await response.to_str()

        if self._tool_context is None:
            self._tool_context = ToolCallContext(
                input=self._input,
                agent_id=self._agent_id,
                participant_id=self._participant_id,
                session_id=self._session_id,
                calls=list(self._call_history),
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
                self._tool_manager.get_calls(delta)
                if self._tool_manager
                else None
            )
            if not calls:
                break

            results: list[ToolCallResult] = []
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
                )

                result = (
                    await self._tool_manager(call, context)
                    if self._tool_manager
                    else None
                )
                self._call_history.append(call)
                self._tool_context = context
                results.append(result)

                if self._event_manager:
                    end = perf_counter()
                    result_event = Event(
                        type=EventType.TOOL_RESULT,
                        payload={"result": result},
                        started=start,
                        finished=end,
                        ellapsed=end - start,
                    )
                    await self._event_manager.trigger(result_event)

            current_response = await self._react_process(delta, results)
            new_text = await current_response.to_str()
            delta = new_text.replace(previous_text, "")
            previous_text = new_text

        self._response = current_response
        return delta

    async def _react_process(
        self, output: str, results: list[ToolCallResult]
    ) -> TextGenerationResponse:
        tool_messages = [
            Message(
                role=MessageRole.TOOL,
                name=result.name,
                arguments=result.arguments,
                content=result.result,
            )
            for result in results
        ]

        assert self._input and (
            (
                isinstance(self._input, list)
                and isinstance(self._input[0], Message)
            )
            or isinstance(self._input, Message)
        )

        messages = list(
            self._input if isinstance(self._input, list) else [self._input]
        )
        messages.extend(tool_messages)

        self._input = messages
        self._tool_context = ToolCallContext(
            input=self._input,
            agent_id=self._agent_id,
            participant_id=self._participant_id,
            session_id=self._session_id,
            calls=list(self._call_history),
        )

        event_tool_model_run = Event(
            type=EventType.TOOL_MODEL_RUN,
            payload={
                "model_id": self._engine_agent.engine.model_id,
                "messages": messages,
                "engine_args": self._engine_args,
            },
        )
        await self._event_manager.trigger(event_tool_model_run)

        response = await self._engine_agent(
            self._operation.specification,
            messages,
            **self._engine_args,
        )
        assert response
        return response

    async def _emit(
        self, token: Token | TokenDetail | str
    ) -> Token | TokenDetail | Event:
        token_str = token.token if hasattr(token, "token") else token

        if self._event_manager:
            token_id = getattr(token, "id", None)
            tokenizer = (
                self._engine_agent.engine.tokenizer
                if self._engine_agent.engine
                else None
            )
            if token_id is None and tokenizer:
                ids = tokenizer.encode(token_str, add_special_tokens=False)
                token_id = ids[0] if ids else None

            await self._event_manager.trigger(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "token_id": token_id,
                        "model_id": self._engine_agent.engine.model_id,
                        "token": token_str,
                        "step": self._step,
                    },
                )
            )

        self._step += 1

        if not self._tool_manager:
            return token

        buffer_value = self._buffer.getvalue()
        should_check = self._tool_manager.is_potential_tool_call(
            buffer_value, token_str
        )
        self._buffer.write(token_str)

        if not should_check:
            return token

        if self._event_manager:
            await self._event_manager.trigger(
                Event(type=EventType.TOOL_DETECT)
            )

        calls = self._tool_manager.get_calls(self._buffer.getvalue())
        if not calls:
            return token

        self._tool_process_events.put(
            Event(
                type=EventType.TOOL_PROCESS,
                payload=calls,
                started=perf_counter(),
            )
        )

        return token

    async def _on_consumed(self) -> None:
        assert self._event_manager
        await self._event_manager.trigger(Event(type=EventType.STREAM_END))
