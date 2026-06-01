from ...agent.loader import OrchestratorLoader
from ...agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from ...entities import (
    Input,
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentText,
    MessageFile,
    MessageRole,
)
from ..context import TaskInputFile, TaskTargetContext
from ..definition import (
    TaskDefinition,
    TaskInputType,
    TaskOutputType,
    TaskTargetType,
)
from ..target import TaskTargetRunner, TaskValidationContext
from ..validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
)

from base64 import b64encode
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from json import dumps, loads
from pathlib import Path
from tomllib import TOMLDecodeError, load
from typing import Protocol, cast
from uuid import UUID


class AgentOrchestrator(Protocol):
    async def __aenter__(self) -> "AgentOrchestrator": ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool | None: ...

    async def __call__(self, input: Input) -> object: ...


class AgentOrchestratorLoader(Protocol):
    async def from_file(
        self,
        path: str,
        *,
        agent_id: UUID | None,
        disable_memory: bool = False,
        uri: str | None = None,
        tool_settings: object | None = None,
    ) -> AgentOrchestrator: ...


class AgentTaskTargetRunner(TaskTargetRunner):
    def __init__(
        self,
        loader: AgentOrchestratorLoader,
        *,
        agent_id: UUID | None = None,
        disable_memory: bool = False,
        ref_base: str | Path | None = None,
        uri: str | None = None,
    ) -> None:
        self._loader = loader
        self._agent_id = agent_id
        self._disable_memory = disable_memory
        self._ref_base = Path(ref_base) if ref_base is not None else None
        self._uri = uri

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(context, TaskValidationContext)
        if definition.execution.type != TaskTargetType.AGENT:
            return (_agent_target_issue(path="execution.type"),)

        try:
            OrchestratorLoader.validate_agent_file(
                str(self._agent_path(definition))
            )
        except (
            AssertionError,
            FileNotFoundError,
            OSError,
            PermissionError,
            TOMLDecodeError,
            TypeError,
            ValueError,
        ):
            return (_agent_target_issue(path="execution.ref"),)
        if definition.input.type in {
            TaskInputType.FILE,
            TaskInputType.FILE_ARRAY,
        }:
            return _validate_agent_file_input(
                definition,
                _agent_file_capability(self._agent_uri(definition)),
            )
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        assert isinstance(context, TaskTargetContext)
        assert context.definition.execution.type == TaskTargetType.AGENT
        orchestrator = await self._loader.from_file(
            str(self._agent_path(context.definition)),
            agent_id=self._agent_id,
            disable_memory=self._disable_memory,
            uri=self._uri,
        )
        async with orchestrator:
            async with _agent_event_listener(orchestrator, context):
                await context.check_cancelled()
                response = await orchestrator(
                    await _agent_input(
                        context,
                        _agent_file_capability(
                            self._agent_uri(
                                context.definition,
                            )
                        ),
                    )
                )
                _attach_cancellation_checker(response, context.check_cancelled)
                try:
                    output = await _agent_output(context.definition, response)
                finally:
                    await context.observe_usage(response)
                await context.check_cancelled()
                return output

    def _agent_path(self, definition: TaskDefinition) -> Path:
        ref = Path(definition.execution.ref)
        if self._ref_base is not None and not ref.is_absolute():
            return self._ref_base / ref
        return ref

    def _agent_uri(self, definition: TaskDefinition) -> str | None:
        if self._uri is not None:
            return self._uri
        try:
            with self._agent_path(definition).open("rb") as file:
                data = load(file)
        except (OSError, TOMLDecodeError):
            return None
        engine = data.get("engine")
        if not isinstance(engine, Mapping):
            return None
        uri = engine.get("uri")
        return uri if isinstance(uri, str) else None


def _attach_cancellation_checker(
    response: object,
    checker: Callable[[], Awaitable[None]],
) -> None:
    set_checker = getattr(response, "set_cancellation_checker", None)
    if callable(set_checker):
        set_checker(checker)


@asynccontextmanager
async def _agent_event_listener(
    orchestrator: AgentOrchestrator,
    context: TaskTargetContext,
) -> AsyncIterator[None]:
    if context.event_listener is None:
        yield
        return

    event_manager = getattr(orchestrator, "event_manager", None)
    add_listener = getattr(event_manager, "add_listener", None)
    remove_listener = getattr(event_manager, "remove_listener", None)
    if not callable(add_listener):
        yield
        return

    add_listener(context.event_listener)
    try:
        yield
    finally:
        if callable(remove_listener):
            remove_listener(context.event_listener)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AgentFileCapability:
    native_files: bool = False
    text_documents: bool = False
    urls: bool = False
    inline_bytes: bool = False
    object_store_uris: bool = False


async def _agent_input(
    context: TaskTargetContext,
    capability: _AgentFileCapability,
) -> Input:
    value = _agent_input_value(context.input_value)
    if not context.files:
        return value
    file_blocks: list[MessageContentFile] = []
    for index, file in enumerate(context.files):
        file_blocks.append(
            MessageContentFile(
                type="file",
                file=await _agent_message_file(
                    file,
                    context=context,
                    capability=capability,
                    path=f"input.files[{index}]",
                ),
            )
        )
    if isinstance(value, Message):
        return _message_with_files(value, tuple(file_blocks))
    if isinstance(value, list) and all(
        isinstance(item, Message) for item in value
    ):
        return [
            *cast(list[Message], value),
            Message(role=MessageRole.USER, content=list(file_blocks)),
        ]
    content: list[MessageContent] = []
    text = _file_prompt_text(context.definition, cast(str | list[str], value))
    if text is not None:
        content.append(MessageContentText(type="text", text=text))
    content.extend(file_blocks)
    return Message(role=MessageRole.USER, content=content)


def _agent_input_value(value: object) -> Input:
    if isinstance(value, str | Message):
        return value
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return cast(list[str], value)
        if all(isinstance(item, Message) for item in value):
            return cast(list[Message], value)
    if _is_json_value(value):
        return dumps(value, sort_keys=True, separators=(",", ":"))
    return str(type(value).__name__)


async def _agent_message_file(
    file: TaskInputFile,
    *,
    context: TaskTargetContext,
    capability: _AgentFileCapability,
    path: str,
) -> MessageFile:
    provider_file_id = _metadata_string(file.metadata, "provider_file_id")
    if provider_file_id is not None and capability.native_files:
        return _message_file(file, file_id=provider_file_id)
    provider_url = _metadata_string(file.metadata, "provider_file_url")
    if provider_url is not None and capability.urls:
        return _message_file(file, file_url=provider_url)
    provider_uri = _metadata_string(file.metadata, "provider_uri")
    if provider_uri is not None and capability.object_store_uris:
        return _message_file(file, file_url=provider_uri)
    if (
        file.artifact_ref is not None
        and context.artifact_store is not None
        and _can_inline_artifact(file.media_type, capability)
    ):
        reader = await context.artifact_store.open(file.artifact_ref)
        try:
            data = b64encode(reader.read()).decode("ascii")
        finally:
            reader.close()
        return _message_file(file, file_data=data)
    raise TaskValidationError(
        (
            TaskValidationIssue(
                code="input.invalid_file",
                path=path,
                message="Task file cannot be sent to the agent target.",
                hint=(
                    "Use a provider-supported file reference, configure an "
                    "artifact backend, or declare a compatible conversion."
                ),
                category=TaskValidationCategory.UNSUPPORTED,
            ),
        )
    )


def _message_file(
    file: object,
    *,
    file_data: str | None = None,
    file_id: str | None = None,
    file_url: str | None = None,
) -> MessageFile:
    media_type = getattr(file, "media_type")
    payload: MessageFile = {}
    if file_id is not None:
        payload["file_id"] = file_id
    if file_url is not None:
        payload["file_url"] = file_url
    if file_data is not None:
        payload["file_data"] = file_data
    if isinstance(media_type, str):
        payload["mime_type"] = media_type
    return payload


def _message_with_files(
    message: Message,
    file_blocks: tuple[MessageContentFile, ...],
) -> Message:
    content = _content_blocks(message.content)
    content.extend(file_blocks)
    return Message(
        role=message.role,
        thinking=message.thinking,
        content=content,
        name=message.name,
        arguments=message.arguments,
        tool_calls=message.tool_calls,
        tool_call_result=message.tool_call_result,
        tool_call_error=message.tool_call_error,
    )


def _content_blocks(
    content: str | MessageContent | list[MessageContent] | None,
) -> list[MessageContent]:
    if content is None:
        return []
    if isinstance(content, str):
        return [MessageContentText(type="text", text=content)]
    if isinstance(content, list):
        return list(content)
    return [content]


def _file_prompt_text(
    definition: TaskDefinition,
    value: str | list[str],
) -> str | None:
    if definition.input.type in {TaskInputType.FILE, TaskInputType.FILE_ARRAY}:
        return None
    if isinstance(value, str):
        return value
    return "\n".join(value)


def _can_inline_artifact(
    media_type: str | None,
    capability: _AgentFileCapability,
) -> bool:
    if not capability.inline_bytes:
        return False
    return capability.native_files or (
        capability.text_documents and _is_text_media_type(media_type)
    )


def _metadata_string(
    metadata: Mapping[str, object],
    key: str,
) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _validate_agent_file_input(
    definition: TaskDefinition,
    capability: _AgentFileCapability,
) -> tuple[TaskValidationIssue, ...]:
    if not _has_file_capability(capability):
        return (_agent_file_issue(path="input.type"),)
    if (
        capability.native_files
        or definition.input.file_conversions
        or (
            bool(definition.input.mime_types)
            and all(
                _is_text_media_type(value)
                for value in definition.input.mime_types
            )
        )
    ):
        return ()
    return (_agent_file_issue(path="input.file_conversions"),)


def _agent_file_issue(*, path: str) -> TaskValidationIssue:
    return TaskValidationIssue(
        code="input.invalid_file",
        path=path,
        message="Agent target does not support the declared file input.",
        hint=(
            "Use a provider with compatible file support or declare a "
            "conversion path."
        ),
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _has_file_capability(capability: _AgentFileCapability) -> bool:
    return any(
        (
            capability.native_files,
            capability.text_documents,
            capability.urls,
            capability.inline_bytes,
            capability.object_store_uris,
        )
    )


def _agent_file_capability(uri: str | None) -> _AgentFileCapability:
    vendor = _vendor_from_uri(uri)
    if vendor in {"openai", "anthropic"}:
        return _AgentFileCapability(
            native_files=True,
            text_documents=True,
            urls=True,
            inline_bytes=True,
        )
    if vendor == "google":
        return _AgentFileCapability(
            native_files=True,
            text_documents=True,
            urls=True,
            inline_bytes=True,
            object_store_uris=True,
        )
    if vendor == "bedrock":
        return _AgentFileCapability(
            text_documents=True,
            inline_bytes=True,
            object_store_uris=True,
        )
    return _AgentFileCapability()


def _vendor_from_uri(uri: str | None) -> str | None:
    if not isinstance(uri, str) or not uri.startswith("ai://"):
        return None
    value = uri.removeprefix("ai://")
    if "@" in value:
        value = value.rsplit("@", maxsplit=1)[1]
    vendor = value.split("/", maxsplit=1)[0].strip().lower()
    return vendor or None


def _is_text_media_type(media_type: str | None) -> bool:
    if media_type is None:
        return False
    return media_type.startswith("text/") or media_type in {
        "application/json",
        "application/markdown",
        "application/xml",
    }


async def _agent_output(
    definition: TaskDefinition,
    response: object,
) -> object:
    output_type = definition.output.type
    if output_type in {
        TaskOutputType.JSON,
        TaskOutputType.OBJECT,
        TaskOutputType.ARRAY,
    }:
        if isinstance(response, OrchestratorResponse):
            return loads(await response.to_json())
        to_json = getattr(response, "to_json", None)
        if to_json is not None:
            return loads(await to_json())
        return loads(await _response_text(response))
    return await _response_text(response)


async def _response_text(response: object) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, OrchestratorResponse):
        return await response.to_str()
    to_str = getattr(response, "to_str", None)
    if to_str is not None:
        return str(await to_str())
    return str(type(response).__name__)


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, str | bool | int | float):
        return not isinstance(value, float) or value == value
    if isinstance(value, list | tuple):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _agent_target_issue(*, path: str) -> TaskValidationIssue:
    return TaskValidationIssue(
        code="execution.unknown_target",
        path=path,
        message="Agent execution target could not be loaded.",
        hint=(
            "Use a readable agent definition with valid agent and engine"
            " sections."
        ),
        category=TaskValidationCategory.UNSUPPORTED,
    )
