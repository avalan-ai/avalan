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
)
from ...entities import Modality as Modality
from ...event import Event, EventType
from ...event.manager import EventManager
from ...memory.manager import MemoryManager
from ...model.call import ModelCallContext
from ...model.engine import Engine
from ...model.manager import ModelManager
from ...model.response.text import TextGenerationResponse
from ...tool.manager import ToolManager
from ...tool.shell.input_files import shell_input_file_manifest
from ...tool.shell.settings import ShellToolSettings
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
from ..renderer import Renderer, TemplateEngineAgent
from .response.orchestrator_response import (
    DEFAULT_MAXIMUM_TOOL_CYCLES,
    OrchestratorResponse,
)

import asyncio
from contextlib import ExitStack
from dataclasses import asdict, replace
from inspect import isawaitable
from json import dumps
from logging import Logger
from re import compile as compile_regex
from time import perf_counter
from typing import Any, cast
from uuid import UUID, uuid4

_INPUT_TEMPLATE_REFERENCE_PATTERN = compile_regex(r"{{\s*input\b")
_MAXIMUM_TOOL_CYCLE_OPTION_KEYS = ("maximum_tool_cycles", "max_tool_cycles")


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
    _engine_agents: dict[str, EngineAgent] = {}
    _engines_stack: ExitStack = ExitStack()
    _engines: list[Engine]
    _operation_step: int | None = None
    _model_ids: set[str] = set()
    _call_options: dict[str, Any] | None = None
    _last_engine_agent: EngineAgent | None = None
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
        self._model_ids = set()

    @staticmethod
    def _pop_maximum_tool_cycles(engine_args: dict[str, Any]) -> int:
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
        assert (
            type(value) is int and value > 0
        ), "maximum_tool_cycles must be a positive integer"
        return value

    @property
    def engine_agent(self) -> EngineAgent | None:
        return self._last_engine_agent

    @property
    def engine(self) -> Engine | None:
        return (
            self._last_engine_agent.engine if self._last_engine_agent else None
        )

    @property
    def id(self) -> UUID:
        return self._id

    @property
    def input_token_count(self) -> int | None:
        if not self._last_engine_agent:
            return None
        count = self._last_engine_agent.input_token_count
        if callable(count):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(count())
            if self._last_engine_agent.output:
                return self._last_engine_agent.output.input_token_count
            return None
        return cast(int | None, count)

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
        self, input: Input, **kwargs: Any
    ) -> OrchestratorResponse:
        tool_confirm = kwargs.pop("tool_confirm", None)
        if self.is_finished:
            self._operation_step = 0

        # Pick next operation step
        operation_step = (
            self._operation_step + 1
            if self._operation_step
            and self._operation_step < self._total_operations
            else 0 if not self._operation_step else None
        )
        self._operation_step = operation_step
        if self._operation_step is None:
            raise NoOperationAvailableException()

        # Load engine agent
        operation = self._operations[self._operation_step]
        environment_hash = dumps(asdict(operation.environment))
        engine_agents = self._engine_agents
        if (
            not engine_agents or environment_hash not in engine_agents
        ) and Orchestrator._engine_agents:
            engine_agents = Orchestrator._engine_agents
        assert engine_agents and environment_hash in engine_agents
        engine_agent = engine_agents[environment_hash]

        # Adapt tool manager
        if (
            engine_agent.engine.tokenizer
            and engine_agent.engine.tokenizer.eos_token
        ):
            self._tool.set_eos_token(engine_agent.engine.tokenizer.eos_token)

        await self._event_manager.trigger(
            Event(type=EventType.START, payload={"step": self._operation_step})
        )

        messages = self._input_messages(operation.specification, input)
        messages = await self._input_messages_with_shell_manifest(messages)

        participant_id = getattr(self._memory, "participant_id", None)
        session_id = (
            self._memory.permanent_message.session_id
            if self._memory.permanent_message
            else None
        )

        # Execute operation
        engine_args = {**(self._call_options or {}), **kwargs}
        maximum_tool_cycles = self._pop_maximum_tool_cycles(engine_args)
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
        context = ModelCallContext(
            specification=operation.specification,
            input=messages,
            engine_args=dict(engine_args),
            agent_id=self._id,
            participant_id=participant_id,
            session_id=session_id,
        )
        result = cast(TextGenerationResponse, await engine_agent(context))
        self._logger.info(
            "Engine agent %s responded to orchestrator", str(engine_agent)
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

        self._last_engine_agent = engine_agent
        last_prompt = engine_agent.last_prompt
        response_input = (
            last_prompt[0] if isinstance(last_prompt, tuple) else messages
        )

        return OrchestratorResponse(
            response_input,
            result,
            engine_agent,
            operation,
            engine_args,
            context,
            event_manager=self._event_manager,
            tool=self._tool,
            tool_confirm=tool_confirm,
            agent_id=self._id,
            participant_id=participant_id,
            session_id=session_id,
            maximum_tool_cycles=maximum_tool_cycles,
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
        interrupted_exit = exc_type is not None and issubclass(
            exc_type, self._INTERRUPTED_EXIT_EXCEPTIONS
        )

        try:
            if not interrupted_exit:
                await self.sync_messages()

            if self._exit_memory:
                self._memory.__exit__(exc_type, exc_value, traceback)

            result = self._engines_stack.__exit__(
                exc_type, exc_value, traceback
            )
            if not interrupted_exit:
                for engine in self._engines:
                    wait_closed = getattr(engine, "wait_closed", None)
                    if wait_closed:
                        close_result = wait_closed()
                        if isawaitable(close_result):
                            await close_result
            return result
        finally:
            event_manager_close = getattr(self._event_manager, "aclose", None)
            if callable(event_manager_close):
                close_result = event_manager_close()
                if isawaitable(close_result):
                    await close_result
            self._engines.clear()

    async def sync_messages(self) -> None:
        if self._last_engine_agent:
            await self._last_engine_agent.sync_messages()

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
