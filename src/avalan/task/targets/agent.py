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
from ...model.file_delivery import (
    FileDeliveryDecision,
    FileDeliveryMode,
    FileDeliveryProfile,
    FileDeliveryRequest,
    plan_file_delivery,
    resolve_file_delivery_profile,
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
from json import dumps, loads
from pathlib import Path
from tomllib import TOMLDecodeError, load
from typing import Protocol, cast
from uuid import UUID

FileDeliveryProfileResolver = Callable[[str | None], FileDeliveryProfile]


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
        file_delivery_resolver: FileDeliveryProfileResolver = (
            resolve_file_delivery_profile
        ),
        ref_base: str | Path | None = None,
        uri: str | None = None,
    ) -> None:
        self._loader = loader
        self._agent_id = agent_id
        self._disable_memory = disable_memory
        assert callable(file_delivery_resolver)
        self._file_delivery_resolver = file_delivery_resolver
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
                self._agent_file_profile(definition),
            )
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        assert isinstance(context, TaskTargetContext)
        assert context.definition.execution.type == TaskTargetType.AGENT
        await context.check_cancelled()
        agent_input = await _agent_input(
            context,
            self._agent_file_profile(context.definition),
        )
        orchestrator = await self._loader.from_file(
            str(self._agent_path(context.definition)),
            agent_id=self._agent_id,
            disable_memory=self._disable_memory,
            uri=self._uri,
        )
        async with orchestrator:
            async with _agent_event_listener(orchestrator, context):
                await context.check_cancelled()
                response = await orchestrator(agent_input)
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

    def _agent_file_profile(
        self,
        definition: TaskDefinition,
    ) -> FileDeliveryProfile:
        return self._file_delivery_resolver(self._agent_uri(definition))


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


async def _agent_input(
    context: TaskTargetContext,
    profile: FileDeliveryProfile,
) -> Input:
    value = _agent_input_value(context.input_value)
    if not context.files:
        return value
    if not profile.accepts_file_count(len(context.files)):
        raise _agent_file_error(path="input.files")
    file_content: list[MessageContent] = []
    for index, file in enumerate(context.files):
        file_content.append(
            await _agent_file_content(
                file,
                context=context,
                profile=profile,
                path=f"input.files[{index}]",
            )
        )
    if isinstance(value, Message):
        return _message_with_content(value, tuple(file_content))
    if isinstance(value, list) and all(
        isinstance(item, Message) for item in value
    ):
        return [
            *cast(list[Message], value),
            Message(role=MessageRole.USER, content=list(file_content)),
        ]
    content: list[MessageContent] = []
    text = _file_prompt_text(context.definition, cast(str | list[str], value))
    if text is not None:
        content.append(MessageContentText(type="text", text=text))
    content.extend(file_content)
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


async def _agent_file_content(
    file: TaskInputFile,
    *,
    context: TaskTargetContext,
    profile: FileDeliveryProfile,
    path: str,
) -> MessageContent:
    if (
        file.provider_reference is not None
        and file.provider_reference.is_expired()
    ):
        raise _agent_file_error(path=path)
    metadata: Mapping[str, object] = file.metadata
    if file.provider_reference is not None:
        metadata = {
            **file.metadata,
            "provider_reference": file.provider_reference.execution_metadata(),
        }
    decision = plan_file_delivery(
        profile,
        FileDeliveryRequest(
            mime_type=(
                file.media_type
                or (
                    file.provider_reference.mime_type
                    if file.provider_reference is not None
                    else None
                )
            ),
            size_bytes=file.size_bytes,
            has_artifact=(
                file.artifact_ref is not None
                and context.artifact_store is not None
            ),
            metadata=metadata,
        ),
    )
    match decision.mode:
        case FileDeliveryMode.PROVIDER_FILE_ID:
            return MessageContentFile(
                type="file",
                file=_message_file(
                    file, file_id=_decision_reference(decision)
                ),
            )
        case FileDeliveryMode.HOSTED_URL | FileDeliveryMode.OBJECT_STORE_URI:
            return MessageContentFile(
                type="file",
                file=_message_file(
                    file,
                    file_url=_decision_reference(decision),
                ),
            )
        case FileDeliveryMode.INLINE_BYTES:
            data = await _artifact_bytes(file, context=context, path=path)
            return MessageContentFile(
                type="file",
                file=_message_file(
                    file,
                    file_data=b64encode(data).decode("ascii"),
                ),
            )
        case FileDeliveryMode.INLINE_TEXT:
            data = await _artifact_bytes(file, context=context, path=path)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _agent_file_error(path=path) from exc
            return MessageContentText(type="text", text=text)
        case _:
            raise _agent_file_error(path=path, decision=decision)


async def _artifact_bytes(
    file: TaskInputFile,
    *,
    context: TaskTargetContext,
    path: str,
) -> bytes:
    if file.artifact_ref is None or context.artifact_store is None:
        raise _agent_file_error(path=path)
    reader = await context.artifact_store.open(file.artifact_ref)
    try:
        return reader.read()
    finally:
        reader.close()


def _decision_reference(decision: FileDeliveryDecision) -> str:
    if decision.reference is None:
        raise _agent_file_error(path="input.files")
    return decision.reference


def _agent_file_error(
    *,
    path: str,
    decision: FileDeliveryDecision | None = None,
) -> TaskValidationError:
    hint = (
        decision.diagnostic.hint
        if decision is not None and decision.diagnostic is not None
        else (
            "Use a provider-supported file reference, configure an "
            "artifact backend, or declare a compatible conversion."
        )
    )
    return TaskValidationError(
        (
            TaskValidationIssue(
                code="input.invalid_file",
                path=path,
                message="Task file cannot be sent to the agent target.",
                hint=hint,
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


def _message_with_content(
    message: Message,
    file_content: tuple[MessageContent, ...],
) -> Message:
    content = _content_blocks(message.content)
    content.extend(file_content)
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


def _validate_agent_file_input(
    definition: TaskDefinition,
    profile: FileDeliveryProfile,
) -> tuple[TaskValidationIssue, ...]:
    if not profile.supports_file_delivery:
        return (_agent_file_issue(path="input.type"),)
    mime_types_accepted = bool(definition.input.mime_types) and all(
        profile.accepts_mime_type(value)
        for value in definition.input.mime_types
    )
    if (
        definition.input.mime_types
        and not mime_types_accepted
        and not definition.input.file_conversions
    ):
        return (_agent_file_issue(path="input.file_conversions"),)
    if (
        profile.has_native_file_delivery
        or definition.input.file_conversions
        or mime_types_accepted
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
