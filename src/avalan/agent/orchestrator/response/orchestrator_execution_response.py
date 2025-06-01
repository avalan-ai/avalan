from ... import Operation
from ...engine import EngineAgent
from ....entities import Input, Message, MessageRole, Token, TokenDetail
from ....event import Event, EventType
from ....event.manager import EventManager
from ....model import TextGenerationResponse
from ....tool.manager import ToolManager
from io import StringIO
from typing import Any, AsyncIterator, Union


class OrchestratorExecutionResponse(
    AsyncIterator[Union[Token, TokenDetail, Event]]
):
    """Async iterator handling tool execution during streaming."""

    _response: TextGenerationResponse
    _engine_agent: EngineAgent
    _operation: Operation
    _engine_args: dict
    _event_manager: EventManager | None
    _tool: ToolManager | None
    _buffer: StringIO
    _tool_call_events: list[Event]
    _current_tool_event: int
    _current_tool_event_call: int
    _tool_result_events: list[Event]
    _current_tool_result: int
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
        self._input = input
        self._response = response
        self._engine_agent = engine_agent
        self._operation = operation
        self._engine_args = engine_args
        self._event_manager = event_manager
        self._tool = tool
        self._buffer = StringIO()
        self._tool_call_events = []
        self._current_tool_event = -1
        self._current_tool_event_call = 0
        self._tool_result_events = []
        self._current_tool_result = 0

    def __aiter__(self) -> "OrchestratorExecutionResponse":
        self._response.__aiter__()
        return self

    async def __anext__(self) -> Union[Token, TokenDetail, Event]:
        if self._tool_call_events:
            if self._current_tool_event == -1:
                self._current_tool_event = 0
                event = self._tool_call_events[self._current_tool_event]
                assert event.type == EventType.TOOL_PROCESS
                return event

            event = self._tool_call_events[self._current_tool_event]
            calls: list[Any] = event.payload or []
            call = calls[self._current_tool_event_call]

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
            self._tool_result_events.append(result_event)

            if self._current_tool_event_call + 1 < len(calls):
                self._current_tool_event_call += 1
            else:
                self._current_tool_event += 1
                self._current_tool_event_call = 0
                if self._current_tool_event >= len(self._tool_call_events):
                    self._current_tool_event = -1
                    self._tool_call_events = []

            return result_event

        if self._tool_result_events and self._current_tool_result < len(
            self._tool_result_events
        ):
            event = self._tool_result_events[self._current_tool_result]
            self._current_tool_result += 1

            if self._current_tool_result == len(self._tool_result_events):
                tool_messages = [
                    Message(
                        role=MessageRole.TOOL,
                        name=e.payload["result"].name,
                        arguments=e.payload["result"].arguments,
                        content=e.payload["result"].result,
                    )
                    for e in self._tool_result_events
                ]

                assert self._input and (
                    (
                        isinstance(self._input, list)
                        and isinstance(self._input[0], Message)
                    )
                    or isinstance(self._input, Message)
                )

                messages = (
                    self._input
                    if isinstance(self._input, list)
                    else [self._input]
                )
                messages.extend(tool_messages)

                self._response = await self._engine_agent(
                    self._operation.specification,
                    messages,
                    **self._engine_args,
                )
                self._current_tool_result = 0
                self._tool_result_events = []

            return event

        token = await self._response.__anext__()
        return await self._emit(token)

    async def _emit(
        self, token: Union[Token, TokenDetail, str]
    ) -> Union[Token, TokenDetail, Event]:
        if not self._tool:
            return token

        self._buffer.write(token.token if hasattr(token, "token") else token)

        if self._event_manager:
            await self._event_manager.trigger(Event(type=EventType.TOOL_DETECT))

        calls = (
            self._tool.get_calls(self._buffer.getvalue())
            if self._tool
            else None
        )
        if not calls:
            return token

        self._tool_call_events.append(
            Event(type=EventType.TOOL_PROCESS, payload=calls)
        )

        return token
