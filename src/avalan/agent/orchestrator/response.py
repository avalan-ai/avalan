from .. import Operation
from ..engine import EngineAgent
from ...event import Event, EventType
from ...event.manager import EventManager
from ...model import TextGenerationResponse
from ...model.entities import Input, Message, MessageRole
from ...tool.manager import ToolManager
from io import StringIO
from typing import Any, AsyncIterator, Union


class ObservableTextGenerationResponse(TextGenerationResponse):
    """Wrap a TextGenerationResponse to emit token and end events."""

    def __init__(
        self,
        response: TextGenerationResponse,
        event_manager: EventManager,
        model_id: str,
        tokenizer: Any | None,
    ) -> None:
        self._response = response
        self._event_manager = event_manager
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._step = 0
        self._response.add_done_callback(self._on_consumed)

    async def _on_consumed(self) -> None:
        await self._event_manager.trigger(Event(type=EventType.STREAM_END))

    @property
    def input_token_count(self) -> int:
        return self._response.input_token_count

    def __aiter__(self) -> "ObservableTextGenerationResponse":
        self._response.__aiter__()
        return self

    async def __anext__(self) -> Any:
        token = await self._response.__anext__()
        token_str = token.token if hasattr(token, "token") else token
        token_id = getattr(token, "id", None)
        if token_id is None and self._tokenizer:
            ids = self._tokenizer.encode(token_str, add_special_tokens=False)
            token_id = ids[0] if ids else None

        await self._event_manager.trigger(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload={
                    "token_id": token_id,
                    "model_id": self._model_id,
                    "token": token_str,
                    "step": self._step,
                },
            )
        )
        self._step += 1
        return token

    async def to_str(self) -> str:
        return await self._response.to_str()

    async def to_json(self) -> str:
        return await self._response.to_json()

    async def to(self, entity_class: type) -> Any:
        return await self._response.to(entity_class)


class OrchestratorResponse(AsyncIterator[Union[TextGenerationResponse, Event]]):
    """Async iterator yielding TextGenerationResponses handling tool calls."""

    _responses_with_events: list[Union[TextGenerationResponse, Event]]

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
        self._engine_agent = engine_agent
        self._operation = operation
        self._engine_args = engine_args
        self._event_manager = event_manager
        self._tool = tool
        self._responses_with_events = [self._response(response)]
        self._index = 0
        self._buffer = StringIO()
        self._finished = False

    def __aiter__(self) -> AsyncIterator[Union[TextGenerationResponse, Event]]:
        return self

    async def __anext__(self) -> Union[TextGenerationResponse, Event]:
        if self._index >= len(self._responses_with_events):
            if not self._finished and self._event_manager:
                self._finished = True
                await self._event_manager.trigger(Event(type=EventType.END))
            raise StopAsyncIteration
        resp = self._responses_with_events[self._index]
        self._index += 1
        return resp

    def _response(
        self, response: TextGenerationResponse
    ) -> ObservableTextGenerationResponse:
        assert self._engine_agent.engine
        return ToolAwareResponse(
            response,
            self._event_manager,
            self._engine_agent.engine.model_id,
            self._engine_agent.engine.tokenizer,
            self._on_token,
        )

    async def _on_token(self, token: str) -> None:
        if not self._tool or not self._event_manager:
            return
        self._buffer.write(token)
        text = self._buffer.getvalue()
        if not self._tool.has_tool_call(text):
            return

        await self._event_manager.trigger(
            Event(type=EventType.TOOL_PROCESS, payload={"output": text})
        )
        tool_calls, tool_results = self._tool(text)

        if tool_calls:
            for call in tool_calls:
                event = Event(
                    type=EventType.TOOL_EXECUTE, payload={"call": call}
                )
                self._responses_with_events.append(event)
                await self._event_manager.trigger(event)

        if tool_results:
            for res in tool_results:
                event = Event(
                    type=EventType.TOOL_RESULT, payload={"result": res}
                )
                self._responses_with_events.append(event)
                await self._event_manager.trigger(event)

        tool_messages = (
            [
                Message(
                    role=MessageRole.TOOL,
                    name=r.name,
                    arguments=r.arguments,
                    content=r.result,
                )
                for r in tool_results
            ]
            if tool_results
            else None
        )

        if tool_messages:
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

            result = await self._engine_agent(
                self._operation.specification,
                messages,
                **self._engine_args,
            )
            self._responses_with_events.append(self._response(result))

        self._buffer = StringIO()


class ToolAwareResponse(ObservableTextGenerationResponse):
    """ObservableTextGenerationResponse that notifies on each token."""

    def __init__(
        self,
        response: TextGenerationResponse,
        event_manager: EventManager,
        model_id: str,
        tokenizer: Any | None,
        on_token,
    ) -> None:
        super().__init__(response, event_manager, model_id, tokenizer)
        self._on_token = on_token

    async def __anext__(self) -> Any:
        token = await super().__anext__()
        token_str = token.token if hasattr(token, "token") else token
        await self._on_token(token_str)
        return token
