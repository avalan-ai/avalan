from ..agent import (
    EngineEnvironment,
    EngineType,
    InputType,
    NoOperationAvailableException,
    Operation,
)
from ..agent.engine import EngineAgent
from ..agent.renderer import Renderer, TemplateEngineAgent
from ..event import Event, EventType
from ..event.manager import EventManager
from ..memory.manager import MemoryManager
from ..model import TextGenerationResponse
from ..model.engine import Engine
from ..model.entities import (
    EngineMessage,
    Input,
    Message,
    MessageRole,
)
from ..model.manager import ModelManager
from ..tool.manager import ToolManager
from contextlib import ExitStack
from dataclasses import asdict
from io import StringIO
from json import dumps
from logging import Logger
from typing import Any, Optional, Union, Type
from uuid import UUID, uuid4


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


class OrchestratorResponse:
    """Async iterator yielding TextGenerationResponses handling tool calls."""

    _responses: list[ObservableTextGenerationResponse]

    def __init__(
        self,
        response: TextGenerationResponse,
        engine_agent: EngineAgent,
        operation: Operation,
        engine_args: dict,
        event_manager: EventManager | None = None,
        tool: ToolManager | None = None,
    ) -> None:
        self._engine_agent = engine_agent
        self._operation = operation
        self._engine_args = engine_args
        self._event_manager = event_manager
        self._tool = tool
        self._responses = [self._wrap_response(response)]
        self._index = 0
        self._buffer = StringIO()
        self._finished = False

    def _wrap_response(
        self, response: TextGenerationResponse
    ) -> ObservableTextGenerationResponse:
        assert self._engine_agent.engine
        return _ToolAwareResponse(
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
        if "</tool_call>" not in text:
            return

        await self._event_manager.trigger(
            Event(type=EventType.TOOL_PROCESS, payload={"output": text})
        )
        tool_calls, tool_results = self._tool(text)

        if tool_calls:
            for call in tool_calls:
                await self._event_manager.trigger(
                    Event(type=EventType.TOOL_EXECUTE, payload={"call": call})
                )

        if tool_results:
            for res in tool_results:
                await self._event_manager.trigger(
                    Event(type=EventType.TOOL_RESULT, payload={"result": res})
                )

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
            result = await self._engine_agent(
                self._operation.specification,
                tool_messages,
                **self._engine_args,
            )
            self._responses.append(self._wrap_response(result))

        self._buffer = StringIO()

    def __aiter__(self) -> "OrchestratorResponse":
        return self

    async def __anext__(self) -> ObservableTextGenerationResponse:
        if self._index >= len(self._responses):
            if not self._finished and self._event_manager:
                self._finished = True
                await self._event_manager.trigger(Event(type=EventType.END))
            raise StopAsyncIteration
        resp = self._responses[self._index]
        self._index += 1
        return resp


class _ToolAwareResponse(ObservableTextGenerationResponse):
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


class Orchestrator:
    _id: UUID
    _name: Optional[str]
    _operations: list[Operation]
    _renderer: Renderer
    _total_operations: int
    _logger: Logger
    _model_manager: ModelManager
    _memory: MemoryManager
    _tool: ToolManager
    _event_manager: EventManager
    _engine_agents: dict[EngineEnvironment, EngineAgent] = {}
    _engines_stack: ExitStack = ExitStack()
    _operation_step: Optional[int] = None
    _model_ids: set[str] = set()
    _call_options: Optional[dict] = None
    _last_engine_agent: Optional[EngineAgent] = None
    _exit_memory: bool = True

    def __init__(
        self,
        logger: Logger,
        model_manager: ModelManager,
        memory: MemoryManager,
        tool: ToolManager,
        event_manager: EventManager,
        operations: Union[Operation, list[Operation]],
        *,
        call_options: Optional[dict] = None,
        exit_memory: bool = True,
        id: Optional[UUID] = None,
        name: Optional[str] = None,
        renderer: Optional[Renderer] = None,
    ):
        self._logger = logger
        self._model_manager = model_manager
        self._memory = memory
        self._tool = tool
        self._event_manager = event_manager
        self._operations = (
            [operations] if isinstance(operations, Operation) else operations
        )
        self._id = id or uuid4()
        self._exit_memory = exit_memory
        self._name = name
        self._renderer = renderer or Renderer()
        self._total_operations = len(self._operations)
        self._call_options = call_options

    @property
    def engine_agent(self) -> Optional[EngineAgent]:
        return self._last_engine_agent

    @property
    def engine(self) -> Optional[Engine]:
        return (
            self._last_engine_agent.engine if self._last_engine_agent else None
        )

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def input_token_count(self) -> Optional[int]:
        return (
            self._last_engine_agent.input_token_count
            if self._last_engine_agent
            else None
        )

    @property
    def is_finished(self) -> bool:
        return (
            self._operation_step is not None
            and self._operation_step == self._total_operations - 1
        )

    @property
    def memory(self) -> MemoryManager:
        return self._memory

    @property
    def model_ids(self) -> set[str]:
        return self._model_ids

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def operations(self) -> list[Operation]:
        return self._operations

    @property
    def tool(self) -> ToolManager:
        return self._tool

    @property
    def event_manager(self) -> EventManager:
        return self._event_manager

    async def __call__(
        self, input: Input, **kwargs
    ) -> Union[TextGenerationResponse, str]:
        if self.is_finished:
            self._operation_step = 0

        # Pick next operation step
        operation_step = (
            self._operation_step + 1
            if self._operation_step
            and self._operation_step < self._total_operations
            else 0
            if not self._operation_step
            else None
        )
        self._operation_step = operation_step
        if self._operation_step is None:
            raise NoOperationAvailableException()

        # Load engine agent
        operation = self._operations[self._operation_step]
        environment_hash = dumps(asdict(operation.environment))
        assert self._engine_agents and environment_hash in self._engine_agents
        engine_agent = self._engine_agents[environment_hash]

        # Adapt tool manager
        if (
            engine_agent.engine.tokenizer
            and engine_agent.engine.tokenizer.eos_token
        ):
            self._tool.set_eos_token(engine_agent.engine.tokenizer.eos_token)

        await self._event_manager.trigger(
            Event(type=EventType.START, payload={"step": self._operation_step})
        )

        # Validate input
        input_type = operation.specification.input_type
        assert (
            input_type != InputType.TEXT
            or isinstance(input, str)
            or isinstance(input, Message)
            or isinstance(input, list)
        )

        if input_type == InputType.TEXT and isinstance(input, str):
            input = Message(role=MessageRole.USER, content=input)

        # Execute operation
        engine_args = {**(self._call_options or {}), **kwargs}
        result = await engine_agent(
            operation.specification, input, **engine_args
        )

        self._last_engine_agent = engine_agent

        if isinstance(result, TextGenerationResponse):
            result = ObservableTextGenerationResponse(
                result,
                self._event_manager,
                engine_agent.engine.model_id,
                engine_agent.engine.tokenizer,
            )

        if self._tool.is_empty:
            await self._event_manager.trigger(Event(type=EventType.END))
            return result

        return OrchestratorResponse(
            result,
            engine_agent,
            operation,
            engine_args,
            event_manager=self._event_manager,
            tool=self._tool,
        )

    async def __aenter__(self):
        first_agent: Optional[TemplateEngineAgent] = None
        model_ids: list[str] = []
        for operation in self._operations:
            # Load engine with environment
            environment = operation.environment
            environment_hash = dumps(asdict(environment))
            if environment_hash not in self._engine_agents:
                model_ids.append(environment.engine_uri.model_id)
                engine = (
                    self._model_manager.load_engine(
                        environment.engine_uri, environment.settings
                    )
                    if environment.type == EngineType.TEXT_GENERATION
                    else None
                )
                if not engine:
                    raise NotImplementedError()

                self._engines_stack.enter_context(engine)
                agent = TemplateEngineAgent(
                    engine,
                    self._memory,
                    self._tool,
                    self._event_manager,
                    self._renderer,
                    name=self._name,
                    id=self._id,
                )
                self._engine_agents[environment_hash] = agent
                if not first_agent:
                    first_agent = agent

        self._last_engine_agent = first_agent
        self._model_ids = set(model_ids)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ):
        if (
            self._last_engine_agent
            and self._last_engine_agent.output
            and (
                self._memory.has_permanent_message
                or self._memory.has_recent_message
            )
        ):
            previous_message = Message(
                role=MessageRole.ASSISTANT,
                content=await self._last_engine_agent.output.to_str(),
            )
            await self._memory.append_message(
                EngineMessage(
                    agent_id=self._id,
                    model_id=self._last_engine_agent.engine.model_id,
                    message=previous_message,
                )
            )

        if self._exit_memory:
            self._memory.__exit__(exc_type, exc_value, traceback)

        return self._engines_stack.__exit__(exc_type, exc_value, traceback)
