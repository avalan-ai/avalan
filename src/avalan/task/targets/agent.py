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
    LocalFileDeliveryProfile,
    resolve_file_delivery_profile,
)
from ..artifact import ArtifactStoreError
from ..context import TaskInputFile, TaskTargetContext
from ..definition import (
    TaskDefinition,
    TaskInputType,
    TaskOutputType,
    TaskTargetType,
)
from ..delivery import TaskFileDeliveryPlan, plan_task_file_delivery
from ..error import (
    TaskOutputParseError,
    TaskProviderStructuredOutputError,
)
from ..schema import (
    TaskSchemaResolutionError,
    canonical_schema_json,
)
from ..target import TaskTargetRunner, TaskValidationContext
from ..text_strategy import (
    TextStrategyKind,
    TextStrategyPlan,
    plan_text_strategy,
)
from ..validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
)

from base64 import b64encode
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from json import JSONDecodeError, dumps, loads
from mimetypes import guess_extension
from pathlib import Path
from sys import maxsize
from tomllib import TOMLDecodeError, load
from typing import Protocol, cast
from uuid import UUID

from jinja2 import (
    Environment as TemplateEnvironment,
)
from jinja2 import (
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
)

FileDeliveryProfileResolver = Callable[[str | None], FileDeliveryProfile]
TokenCounter = Callable[[str], int]


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


@dataclass(frozen=True, slots=True, kw_only=True)
class _AgentPrompt:
    user: str | None = None
    user_template: str | None = None
    templates_path: Path | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _AgentFileBlock:
    content: MessageContent
    metadata: Mapping[str, object]


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
        token_counter: TokenCounter | None = None,
        uri: str | None = None,
    ) -> None:
        self._loader = loader
        self._agent_id = agent_id
        self._disable_memory = disable_memory
        assert callable(file_delivery_resolver)
        self._file_delivery_resolver = file_delivery_resolver
        self._ref_base = Path(ref_base) if ref_base is not None else None
        if token_counter is not None:
            assert callable(token_counter)
        self._token_counter = token_counter or _estimated_token_count
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
            config = OrchestratorLoader.validate_agent_file(
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
        issues: list[TaskValidationIssue] = []
        if definition.input.type in {
            TaskInputType.FILE,
            TaskInputType.FILE_ARRAY,
        }:
            issues.extend(
                _validate_agent_file_input(
                    definition,
                    self._agent_file_profile(definition),
                )
            )
        issues.extend(_validate_agent_output_schema(definition, config))
        return tuple(issues)

    async def run(self, context: TaskTargetContext) -> object:
        assert isinstance(context, TaskTargetContext)
        assert context.definition.execution.type == TaskTargetType.AGENT
        await context.check_cancelled()
        profile = self._agent_file_profile(context.definition)
        prompt = self._agent_prompt(context.definition)
        agent_input = await _agent_input(context, profile, prompt=prompt)
        agent_input, text_plan = _plan_local_text_delivery(
            agent_input,
            context=context,
            profile=profile,
            token_counter=self._token_counter,
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
                if (
                    text_plan is not None
                    and text_plan.kind == TextStrategyKind.MAP_REDUCE
                ):
                    response = await _run_map_reduce(
                        orchestrator,
                        context=context,
                        plan=text_plan,
                        token_counter=self._token_counter,
                        token_limit=(
                            context.definition.limits.total_tokens or maxsize
                        ),
                    )
                else:
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
        engine = self._agent_engine_config(definition)
        if engine is None:
            return None
        uri = engine.get("uri")
        return uri if isinstance(uri, str) else None

    def _agent_engine_config(
        self,
        definition: TaskDefinition,
    ) -> Mapping[str, object] | None:
        config = self._agent_config(definition)
        if config is None:
            return None
        engine = config.get("engine")
        return engine if isinstance(engine, Mapping) else None

    def _agent_prompt(self, definition: TaskDefinition) -> _AgentPrompt:
        config = self._agent_config(definition)
        if config is None:
            return _AgentPrompt()
        agent = config.get("agent")
        if not isinstance(agent, Mapping):
            return _AgentPrompt()
        user = agent.get("user")
        user_template = agent.get("user_template")
        return _AgentPrompt(
            user=user if isinstance(user, str) else None,
            user_template=(
                user_template if isinstance(user_template, str) else None
            ),
            templates_path=self._agent_path(definition).parent,
        )

    def _agent_config(
        self,
        definition: TaskDefinition,
    ) -> Mapping[str, object] | None:
        try:
            with self._agent_path(definition).open("rb") as file:
                return load(file)
        except (OSError, TOMLDecodeError):
            return None

    def _agent_local_file_delivery_profile(
        self,
        definition: TaskDefinition,
    ) -> LocalFileDeliveryProfile:
        engine = self._agent_engine_config(definition)
        if engine is None:
            return LocalFileDeliveryProfile.TEXT
        value = engine.get("file_delivery_profile")
        if not isinstance(value, str):
            return LocalFileDeliveryProfile.TEXT
        try:
            return LocalFileDeliveryProfile(value)
        except ValueError:
            return LocalFileDeliveryProfile.TEXT

    def _agent_file_profile(
        self,
        definition: TaskDefinition,
    ) -> FileDeliveryProfile:
        uri = self._agent_uri(definition)
        local_profile = self._agent_local_file_delivery_profile(definition)
        if self._file_delivery_resolver is resolve_file_delivery_profile:
            return resolve_file_delivery_profile(
                uri,
                local_profile=local_profile,
            )
        return self._file_delivery_resolver(uri)


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
    *,
    prompt: _AgentPrompt | None = None,
) -> Input:
    value = _agent_input_value(context.input_value)
    if not context.files:
        return value
    if not profile.accepts_file_count(len(context.files)):
        raise _agent_file_error(path="input.files")
    file_blocks: list[_AgentFileBlock] = []
    for index, file in enumerate(context.files):
        file_blocks.append(
            await _agent_file_content(
                file,
                context=context,
                profile=profile,
                index=index,
                path=f"input.files[{index}]",
            )
        )
    file_content = tuple(block.content for block in file_blocks)
    file_metadata = tuple(block.metadata for block in file_blocks)
    if isinstance(value, Message):
        return _message_with_content(
            context.definition,
            value,
            file_content,
            prompt=prompt,
            files=file_metadata,
        )
    if isinstance(value, list) and all(
        isinstance(item, Message) for item in value
    ):
        return _messages_with_content(
            context.definition,
            cast(list[Message], value),
            file_content,
            prompt=prompt,
            files=file_metadata,
        )
    content: list[MessageContent] = []
    text = _file_prompt_text(
        context.definition,
        cast(str | list[str], value),
        prompt=prompt,
        files=file_metadata,
    )
    if text is not None:
        content.append(MessageContentText(type="text", text=text))
    content.extend(file_content)
    return Message(role=MessageRole.USER, content=content)


def _validate_agent_output_schema(
    definition: TaskDefinition,
    config: Mapping[str, object],
) -> tuple[TaskValidationIssue, ...]:
    task_schema = definition.output.schema
    agent_schema = _agent_response_format_schema(config)
    if task_schema is None or agent_schema is None:
        return ()
    try:
        task_schema_json = canonical_schema_json(task_schema)
        agent_schema_json = canonical_schema_json(agent_schema)
    except TaskSchemaResolutionError:
        return (
            TaskValidationIssue(
                code="output.invalid_schema",
                path="output.schema",
                message="Task output schema is invalid.",
                hint="Use a JSON-compatible task output schema.",
                category=TaskValidationCategory.VALUE,
            ),
        )
    if task_schema_json == agent_schema_json:
        return ()
    return (
        TaskValidationIssue(
            code="output.invalid_schema",
            path="output.schema",
            message=(
                "Task output schema does not match the agent response "
                "format schema."
            ),
            hint="Use the same schema for task output and agent response.",
            category=TaskValidationCategory.VALUE,
        ),
    )


def _agent_response_format_schema(
    config: Mapping[str, object],
) -> Mapping[str, object] | None:
    run_config = config.get("run")
    if not isinstance(run_config, Mapping):
        return None
    response_format = run_config.get("response_format")
    if not isinstance(response_format, Mapping):
        return None
    if response_format.get("type") != "json_schema":
        return None
    schema = response_format.get("schema")
    if isinstance(schema, Mapping):
        return cast(Mapping[str, object], schema)
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, Mapping):
        return None
    nested_schema = json_schema.get("schema")
    if isinstance(nested_schema, Mapping):
        return cast(Mapping[str, object], nested_schema)
    return None


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
    index: int,
    path: str,
) -> _AgentFileBlock:
    assert isinstance(index, int)
    assert not isinstance(index, bool)
    assert index >= 0
    if (
        file.provider_reference is not None
        and file.provider_reference.is_expired()
    ):
        raise _agent_file_error(path=path)
    plan = await plan_task_file_delivery(
        context.definition,
        file,
        profile=profile,
        artifact_store=context.artifact_store,
    )
    decision = plan.decision
    metadata = _safe_file_template_metadata(file, index=index, plan=plan)
    match decision.mode:
        case FileDeliveryMode.PROVIDER_FILE_ID:
            return _AgentFileBlock(
                content=MessageContentFile(
                    type="file",
                    file=_message_file(
                        file, file_id=_decision_reference(decision)
                    ),
                ),
                metadata=metadata,
            )
        case FileDeliveryMode.HOSTED_URL | FileDeliveryMode.OBJECT_STORE_URI:
            return _AgentFileBlock(
                content=MessageContentFile(
                    type="file",
                    file=_message_file(
                        file,
                        file_url=_decision_reference(decision),
                    ),
                ),
                metadata=metadata,
            )
        case FileDeliveryMode.INLINE_BYTES:
            data = await _artifact_bytes(
                file,
                context=context,
                path=path,
                max_bytes=_artifact_read_max_bytes(
                    context.definition,
                    decision=decision,
                    profile=profile,
                ),
            )
            return _AgentFileBlock(
                content=MessageContentFile(
                    type="file",
                    file=_message_file(
                        file,
                        file_data=b64encode(data).decode("ascii"),
                    ),
                ),
                metadata=metadata,
            )
        case FileDeliveryMode.INLINE_TEXT:
            data = await _artifact_bytes(
                file,
                context=context,
                path=path,
                max_bytes=_artifact_read_max_bytes(
                    context.definition,
                    decision=decision,
                    profile=profile,
                ),
            )
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _agent_file_error(path=path) from exc
            return _AgentFileBlock(
                content=MessageContentText(type="text", text=text),
                metadata=metadata,
            )
        case (
            FileDeliveryMode.CONVERTED_ARTIFACT
            | FileDeliveryMode.RETRIEVAL_CONTEXT
            | FileDeliveryMode.MAP_REDUCE_CONTEXT
        ):
            data = await _artifact_bytes(file, context=context, path=path)
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise _agent_file_error(path=path, decision=decision) from exc
            return _AgentFileBlock(
                content=MessageContentText(type="text", text=text),
                metadata=metadata,
            )
        case _:
            raise _agent_file_error(path=path, decision=decision)


async def _artifact_bytes(
    file: TaskInputFile,
    *,
    context: TaskTargetContext,
    path: str,
    max_bytes: int | None = None,
) -> bytes:
    if file.artifact_ref is None or context.artifact_store is None:
        raise _agent_file_error(path=path)
    try:
        reader = await context.artifact_store.open_stream(
            file.artifact_ref,
            max_bytes=max_bytes,
        )
    except ArtifactStoreError as exc:
        raise _agent_file_error(path=path) from exc
    try:
        return reader.read()
    except ArtifactStoreError as exc:
        raise _agent_file_error(path=path) from exc
    finally:
        reader.close()


def _artifact_read_max_bytes(
    definition: TaskDefinition,
    *,
    decision: FileDeliveryDecision,
    profile: FileDeliveryProfile,
) -> int | None:
    limits = [
        definition.limits.file_bytes,
        definition.artifact.max_bytes,
    ]
    match decision.mode:
        case FileDeliveryMode.INLINE_BYTES:
            if (
                profile.inline_byte_limit is not None
                and profile.inline_byte_limit.max_bytes is not None
            ):
                limits.append(
                    _max_raw_bytes_for_base64(
                        profile.inline_byte_limit.max_bytes
                    )
                )
        case FileDeliveryMode.INLINE_TEXT:
            if (
                profile.inline_text_limit is not None
                and profile.inline_text_limit.max_bytes is not None
            ):
                limits.append(profile.inline_text_limit.max_bytes)
        case _:
            pass
    bounded = tuple(limit for limit in limits if limit is not None)
    return min(bounded) if bounded else None


def _max_raw_bytes_for_base64(encoded_max_bytes: int) -> int:
    assert isinstance(encoded_max_bytes, int)
    assert not isinstance(encoded_max_bytes, bool)
    assert encoded_max_bytes > 0
    return (encoded_max_bytes // 4) * 3


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
        payload["filename"] = _message_file_name(media_type)
    if isinstance(media_type, str):
        payload["mime_type"] = media_type
    return payload


def _message_file_name(media_type: object) -> str:
    extension = (
        guess_extension(media_type) if isinstance(media_type, str) else None
    )
    return f"task-file{extension or '.bin'}"


def _message_with_content(
    definition: TaskDefinition,
    message: Message,
    file_content: tuple[MessageContent, ...],
    *,
    prompt: _AgentPrompt | None = None,
    files: tuple[Mapping[str, object], ...] = (),
) -> Message:
    content = _content_blocks(message.content)
    text = _message_prompt_text(message)
    prompt_text = (
        _file_prompt_text(
            definition,
            text or "",
            prompt=prompt,
            files=files,
        )
        if text is not None or _has_agent_prompt(prompt)
        else None
    )
    content = _message_content_with_prompt(
        definition,
        content,
        prompt_text=prompt_text,
    )
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


def _messages_with_content(
    definition: TaskDefinition,
    messages: list[Message],
    file_content: tuple[MessageContent, ...],
    *,
    prompt: _AgentPrompt | None = None,
    files: tuple[Mapping[str, object], ...] = (),
) -> list[Message]:
    assert messages
    last = messages[-1]
    if (
        not _has_agent_prompt(prompt)
        and _message_prompt_text(last) is None
        and not _content_blocks(last.content)
    ):
        return [
            *messages,
            Message(role=MessageRole.USER, content=list(file_content)),
        ]
    return [
        *messages[:-1],
        _message_with_content(
            definition,
            last,
            file_content,
            prompt=prompt,
            files=files,
        ),
    ]


def _has_agent_prompt(prompt: _AgentPrompt | None) -> bool:
    return prompt is not None and (
        prompt.user is not None or prompt.user_template is not None
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


def _message_prompt_text(message: Message) -> str | None:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, MessageContentText):
        return content.text
    if isinstance(content, list):
        texts = tuple(
            block.text
            for block in content
            if isinstance(block, MessageContentText)
        )
        return "\n".join(texts) if texts else None
    return None


def _message_content_with_prompt(
    definition: TaskDefinition,
    content: list[MessageContent],
    *,
    prompt_text: str | None,
) -> list[MessageContent]:
    if prompt_text is None:
        return [
            block
            for block in content
            if not _is_file_input(definition)
            or not isinstance(block, MessageContentText)
        ]

    if _is_file_input(definition):
        return [
            MessageContentText(type="text", text=prompt_text),
            *(
                block
                for block in content
                if not isinstance(block, MessageContentText)
            ),
        ]

    replacement = MessageContentText(type="text", text=prompt_text)
    replaced = False
    updated: list[MessageContent] = []
    for block in content:
        if isinstance(block, MessageContentText) and not replaced:
            updated.append(replacement)
            replaced = True
        else:
            updated.append(block)
    if not replaced:
        updated.insert(0, replacement)
    return updated


def _file_prompt_text(
    definition: TaskDefinition,
    value: str | list[str],
    *,
    prompt: _AgentPrompt | None = None,
    files: tuple[Mapping[str, object], ...] = (),
) -> str | None:
    input_text = None if _is_file_input(definition) else _input_text(value)
    if prompt is None or (
        prompt.user is None and prompt.user_template is None
    ):
        return input_text
    render_vars = {
        "files": [dict(file) for file in files],
        "input": input_text or "",
    }
    rendered = _render_agent_prompt(prompt, render_vars)
    if input_text and prompt.user and not _user_references_input(prompt.user):
        return _prefix_text(rendered, input_text)
    if input_text and prompt.user_template:
        return _prefix_text(rendered, input_text)
    return rendered or input_text


def _is_file_input(definition: TaskDefinition) -> bool:
    return definition.input.type in {
        TaskInputType.FILE,
        TaskInputType.FILE_ARRAY,
    }


def _input_text(value: str | list[str]) -> str:
    if isinstance(value, str):
        return value
    return "\n".join(value)


def _render_agent_prompt(
    prompt: _AgentPrompt,
    render_vars: Mapping[str, object],
) -> str:
    try:
        if prompt.user_template is not None:
            environment = TemplateEnvironment(
                loader=FileSystemLoader(str(prompt.templates_path or Path())),
                trim_blocks=True,
                lstrip_blocks=True,
                undefined=StrictUndefined,
            )
            output = environment.get_template(prompt.user_template).render(
                **render_vars
            )
        else:
            environment = TemplateEnvironment(undefined=StrictUndefined)
            output = environment.from_string(prompt.user or "").render(
                **render_vars
            )
    except TemplateError as exc:
        raise _agent_prompt_error() from exc
    return "\n".join(line.strip() for line in output.splitlines())


def _agent_prompt_error() -> TaskValidationError:
    return TaskValidationError(
        (
            TaskValidationIssue(
                code="input.invalid_prompt",
                path="execution.ref",
                message="Agent prompt could not be rendered.",
                hint=(
                    "Check the agent prompt template variables and template "
                    "reference."
                ),
                category=TaskValidationCategory.VALUE,
            ),
        )
    )


def _safe_file_template_metadata(
    file: TaskInputFile,
    *,
    index: int,
    plan: TaskFileDeliveryPlan,
) -> Mapping[str, object]:
    value: dict[str, object] = {"index": index}
    if file.media_type is not None:
        value["mime_type"] = file.media_type
    if plan.size_bucket is not None:
        value["size_bucket"] = plan.size_bucket
    if file.provider_reference is not None:
        if file.provider_reference.mime_type is not None:
            value.setdefault("mime_type", file.provider_reference.mime_type)
        if file.provider_reference.size_bucket is not None:
            value.setdefault(
                "size_bucket", file.provider_reference.size_bucket
            )
        if file.provider_reference.identity_hmac is not None:
            value["identity_hmac"] = file.provider_reference.identity_hmac
    role = _safe_file_role(file)
    if role is not None:
        value["role"] = role
    return value


def _safe_file_role(file: TaskInputFile) -> str | None:
    role = file.metadata.get("role")
    return role if isinstance(role, str) and role.strip() else None


def _prefix_text(prefix: str, content: str) -> str:
    prefix = prefix.strip()
    return f"{prefix}\n\n{content}" if prefix else content


def _user_references_input(user: str) -> bool:
    return "{{input" in user or "{{ input" in user


def _plan_local_text_delivery(
    agent_input: Input,
    *,
    context: TaskTargetContext,
    profile: FileDeliveryProfile,
    token_counter: TokenCounter,
) -> tuple[Input, TextStrategyPlan | None]:
    if not context.files or not profile.requires_conversion_for_file_blocks:
        return agent_input, None
    prompt_texts, document_texts = _local_text_prompt_and_documents(
        agent_input,
        file_count=len(context.files),
    )
    plan = plan_text_strategy(
        prompt_texts=prompt_texts,
        document_texts=document_texts,
        token_limit=context.definition.limits.total_tokens or maxsize,
        token_counter=token_counter,
    )
    if plan.kind == TextStrategyKind.INLINE:
        return agent_input, plan
    if plan.kind == TextStrategyKind.RETRIEVAL:
        return _text_strategy_input(plan.texts), plan
    if plan.kind == TextStrategyKind.MAP_REDUCE:
        return agent_input, plan
    raise TaskValidationError(plan.issues)


def _local_text_prompt_and_documents(
    value: Input,
    *,
    file_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    assert isinstance(file_count, int)
    assert not isinstance(file_count, bool)
    assert file_count > 0
    texts = _input_text_blocks(value)
    if len(texts) <= file_count:
        return (), texts
    return texts[:-file_count], texts[-file_count:]


def _text_strategy_input(texts: tuple[str, ...]) -> Message:
    return Message(
        role=MessageRole.USER,
        content=[
            MessageContentText(type="text", text=text)
            for text in texts
            if text
        ],
    )


async def _run_map_reduce(
    orchestrator: AgentOrchestrator,
    *,
    context: TaskTargetContext,
    plan: TextStrategyPlan,
    token_counter: TokenCounter,
    token_limit: int,
) -> object:
    assert callable(token_counter)
    assert isinstance(token_limit, int)
    assert not isinstance(token_limit, bool)
    assert token_limit > 0
    summaries: list[str] = []
    for chunk in plan.chunks:
        await context.check_cancelled()
        response = await orchestrator(
            _text_strategy_input((*plan.prompt_texts, chunk.text))
        )
        _attach_cancellation_checker(response, context.check_cancelled)
        try:
            summaries.append(await _response_text(response))
        finally:
            await context.observe_usage(response)
        await context.check_cancelled()
    await context.check_cancelled()
    reduce_texts = (*plan.prompt_texts, *summaries)
    if _text_token_total(reduce_texts, token_counter) > token_limit:
        raise TaskValidationError(
            (
                TaskValidationIssue(
                    code="limits.invalid_value",
                    path="limits.total_tokens",
                    message="Task input exceeds the configured token limit.",
                    hint=(
                        "Reduce the input text, raise the token limit, or "
                        "use a smaller file input."
                    ),
                    category=TaskValidationCategory.VALUE,
                ),
            )
        )
    return await orchestrator(_text_strategy_input(reduce_texts))


def _input_text_blocks(value: Input) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Message):
        return _message_text_blocks(value)
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, Message):
                texts.extend(_message_text_blocks(item))
        return tuple(texts)
    return ()


def _message_text_blocks(message: Message) -> tuple[str, ...]:
    content = message.content
    if content is None:
        return ()
    if isinstance(content, str):
        return (content,)
    if isinstance(content, MessageContentText):
        return (content.text,)
    if isinstance(content, list):
        return tuple(
            block.text
            for block in content
            if isinstance(block, MessageContentText)
        )
    return ()


def _estimated_token_count(text: str) -> int:
    assert isinstance(text, str)
    return len(text.split())


def _text_token_total(
    texts: tuple[str, ...],
    token_counter: TokenCounter,
) -> int:
    total = 0
    for text in texts:
        count = token_counter(text)
        assert isinstance(count, int)
        assert not isinstance(count, bool)
        assert count >= 0
        total += count
    return total


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
            return _parse_agent_json_payload(
                await _response_json_payload(response)
            )
        to_json = getattr(response, "to_json", None)
        if to_json is not None:
            return _parse_agent_json_payload(
                await _response_json_payload(response)
            )
        return _parse_agent_json_payload(await _response_text(response))
    return await _response_text(response)


async def _response_json_payload(response: object) -> object:
    to_json = getattr(response, "to_json")
    try:
        return await to_json()
    except Exception as exc:
        raise TaskProviderStructuredOutputError() from exc


def _parse_agent_json_payload(payload: object) -> object:
    if not isinstance(payload, str | bytes | bytearray):
        raise TaskOutputParseError()
    try:
        return loads(payload)
    except (JSONDecodeError, TypeError) as exc:
        raise TaskOutputParseError() from exc


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
