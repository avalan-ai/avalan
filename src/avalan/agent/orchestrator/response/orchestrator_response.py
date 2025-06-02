from ... import Operation
from ...engine import EngineAgent
from ....entities import (
    Input,
    Message,
    MessageRole,
    Token,
    TokenDetail,
    ToolCall,
)
from ....event import Event, EventType
from ....event.manager import EventManager
from ....model import TextGenerationResponse
from ....tool.manager import ToolManager
from queue import Queue
from io import StringIO
from typing import Any, AsyncIterator, Union


class OrchestratorResponse(AsyncIterator[Union[Token, TokenDetail, Event]]):
    """Async iterator handling tool execution during streaming."""

    _response: TextGenerationResponse
    _response_iterator: AsyncIterator[Union[Token, TokenDetail, Event]] | None
    _engine_agent: EngineAgent
    _operation: Operation
    _engine_args: dict
    _event_manager: EventManager | None
    _tool: ToolManager | None
    _buffer: StringIO
    _calls: Queue[ToolCall]
    _tool_call_events: Queue[Event]
    _tool_process_events: Queue[Event]
    _tool_result_events: Queue[Event]
    _input: Input

    def __init__(
        self,
        input: Input,
        response: TextGenerationResponse,
        engine_agent: EngineAgent,
        operation: Operation,
        engine_args: dict,
        event_manager: EventManager | None = None,
        tool: ToolManager | None = None,
    ) -> None:
        assert input and response and engine_agent and operation
        self._input = input
        self._response = response
        self._engine_agent = engine_agent
        self._operation = operation
        self._engine_args = engine_args
        self._event_manager = event_manager
        self._tool = None if tool and tool.is_empty else tool
        self._finished = False
        self._step = 0
        if self._event_manager:

            async def _on_consumed() -> None:
                await self._event_manager.trigger(
                    Event(type=EventType.STREAM_END)
                )

            self._response.add_done_callback(_on_consumed)

    @property
    def input_token_count(self) -> int:
        return self._response.input_token_count

    async def to_str(self) -> str:
        return await self._response.to_str()

    async def to_json(self) -> str:
        return await self._response.to_json()

    async def to(self, entity_class: type) -> Any:
        return await self._response.to(entity_class)

    def __aiter__(self) -> "OrchestratorResponse":
        self._response_iterator = self._response.__aiter__()
        self._buffer = StringIO()
        self._calls = Queue()
        self._tool_call_events = Queue()
        self._tool_process_events = Queue()
        self._tool_result_events = Queue()
        self._step = 0
        return self

    async def __anext__(self) -> Union[Token, TokenDetail, Event]:
        assert self._response_iterator

        total_process_events = self._tool_process_events.qsize()
        if total_process_events:
            event = self._tool_process_events.get()
            assert event.type == EventType.TOOL_PROCESS
            self._tool_call_events.put(event)
            return event

        total_call_events = self._tool_call_events.qsize()
        if total_call_events:
            event = self._tool_call_events.get()
            assert event.type == EventType.TOOL_PROCESS
            await self._event_manager.trigger(event)

            calls: list[ToolCall] = event.payload or []
            if calls:
                for call in calls:
                    assert isinstance(call, ToolCall)
                    self._calls.put(call)

        total_calls = self._calls.qsize()
        if total_calls:
            call = self._calls.get()

            execute_event = Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": call},
            )
            if self._event_manager:
                await self._event_manager.trigger(execute_event)

            result = await self._tool(call) if self._tool else None

            result_event = Event(
                type=EventType.TOOL_RESULT,
                payload={"result": result},
            )
            if self._event_manager:
                await self._event_manager.trigger(result_event)

            self._tool_result_events.put(result_event)

            return result_event

        # Wait untill all results are collected
        total_results = self._tool_result_events.qsize()
        if total_results and not total_call_events and not total_calls:
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

            messages = (
                self._input if isinstance(self._input, list) else [self._input]
            )
            messages.extend(tool_messages)

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
                payload={"response": inner_response},
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

    async def _emit(
        self, token: Union[Token, TokenDetail, str]
    ) -> Union[Token, TokenDetail, Event]:
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

        if not self._tool:
            return token

        self._buffer.write(token_str)

        if self._event_manager:
            await self._event_manager.trigger(Event(type=EventType.TOOL_DETECT))

        calls = (
            self._tool.get_calls(self._buffer.getvalue())
            if self._tool
            else None
        )
        if not calls:
            return token

        self._tool_process_events.put(
            Event(type=EventType.TOOL_PROCESS, payload=calls)
        )

        return token
