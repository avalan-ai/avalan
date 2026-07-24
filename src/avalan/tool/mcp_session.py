"""Run initialized MCP calls with bounded reverse form elicitation."""

from ..agent.execution import AttachedInteractionRuntime
from ..entities import ToolCallContext
from ..interaction.broker import (
    InteractionBrokerRequest,
    InteractionRequestResult,
)
from ..interaction.entities import (
    AnsweredResolution,
    CancelledResolution,
    Choice,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    DeclinedResolution,
    ExpiredResolution,
    FreeFormOther,
    InputAnswer,
    InputQuestion,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    QuestionId,
    RequirementMode,
    SelectedChoice,
    SelectionValidationConstraints,
    SelectionValue,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    SupersededResolution,
    TextAnswer,
    TextQuestion,
    TextValidationConstraints,
    TimedOutResolution,
    UnavailableResolution,
)
from ..interaction.error import InputContractError

from asyncio import CancelledError, timeout
from collections.abc import Awaitable, Callable, Mapping
from datetime import timedelta
from importlib.metadata import version as metadata_version
from json import dumps
from re import IGNORECASE
from re import compile as compile_pattern
from typing import Any, TypedDict, cast
from urllib.parse import urlsplit

from httpx import AsyncClient
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.session import (
    ElicitationFnT,
    LoggingFnT,
    MessageHandlerFnT,
)
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.context import RequestContext
from mcp.shared.session import ProgressFnT

MCP_PROTOCOL_VERSION = "2025-11-25"
MCP_RELATED_TASK_METADATA_KEY = "io.modelcontextprotocol/related-task"
MCP_REQUIRED_OTHER_METADATA_KEY = "ai.avalan/required-other"

_CALL_TIMEOUT_SECONDS = 300.0
_MAX_CALL_TIMEOUT_SECONDS = 3_600.0
_MAX_SCHEMA_CHARACTERS = 32_768
_MAX_SCHEMA_BYTES = 131_072
_MCP_CONFLICT = -32_009
_MCP_EXPIRED = -32_010
_MCP_UNAVAILABLE = -32_001
_FORM_OTHER_PROPERTY_PREFIX = "__avalan_other__"
_MULTILINE_PATTERNS = frozenset(
    {
        "^[^\\r]*$",
        "^(?:[^\\r]|\\r\\n)*$",
    }
)
_FORM_ROOT_KEYS = frozenset(
    {
        "$schema",
        "additionalProperties",
        "description",
        "properties",
        "required",
        "title",
        "type",
    }
)
_FORM_COMMON_PROPERTY_KEYS = frozenset(
    {
        "default",
        "description",
        "title",
        "type",
    }
)
_SECRET_PATTERN = compile_pattern(
    (
        r"(?:\bpassword\b|\bpasscode\b|\bsecret\b|\bcredential(?:s)?\b|"
        r"\bapi[\s._-]*key\b|\baccess[\s._-]*token\b|"
        r"\brefresh[\s._-]*token\b|\bprivate[\s._-]*key\b|"
        r"\b(?:bearer|session|identity)[\s._-]*token\b|"
        r"\bpayment\b|\b(?:credit|debit)[\s._-]*card\b|"
        r"\bcard[\s._-]*(?:details?|number|security[\s._-]*code)\b|"
        r"\b(?:cvv|cvc|iban)\b|"
        r"\bbank[\s._-]*(?:account|credential)\b|"
        r"\brouting[\s._-]*number\b|\b(?:mfa|2fa|otp|totp|hotp|pin)\b|"
        r"\bone[\s._-]*time[\s._-]*(?:code|password)\b|"
        r"\b(?:recovery|security|verification)[\s._-]*code\b|"
        r"\bauth(?:entication|orization|orisation)?\b|"
        r"\bauthentication[\s._-]*challenge\b|\boauth\b)"
    ),
    IGNORECASE,
)

McpFormContent = dict[str, str | int | float | bool | list[str] | None]


class _QuestionCommon(TypedDict):
    question_id: QuestionId
    prompt: str
    header: str | None
    required: bool


class _ElicitationRouter:
    """Bind reverse input to the immutable origin of one tool call."""

    def __init__(
        self,
        *,
        uri: str,
        tool_name: str,
        context: ToolCallContext,
        related_task_id: str | None,
        ttl_seconds: int,
    ) -> None:
        self._uri = uri
        self._tool_name = tool_name
        self._context = context
        self._origin = context.execution_origin
        self._broker = context.interaction_broker
        self._related_task_id = related_task_id
        self._ttl_seconds = ttl_seconds

    @property
    def form_capable(self) -> bool:
        """Return whether this exact call has a live authorized route."""
        execution = self._context.execution
        origin = self._origin
        runtime = (
            execution.interaction_runtime if execution is not None else None
        )
        return (
            execution is not None
            and origin is not None
            and execution.origin == origin
            and runtime is not None
            and runtime.actor.principal == origin.principal
            and self._broker is not None
            and (
                self._context.agent_id is None
                or str(self._context.agent_id) == str(origin.agent_id)
            )
        )

    async def __call__(
        self,
        _request_context: RequestContext[ClientSession, Any, Any],
        params: mcp_types.ElicitRequestParams,
    ) -> mcp_types.ElicitResult | mcp_types.ErrorData:
        if isinstance(params, mcp_types.ElicitRequestURLParams):
            return _invalid("URL elicitation is not available")
        if not isinstance(params, mcp_types.ElicitRequestFormParams):
            return _invalid("Unsupported MCP elicitation mode")
        related = _related_task(params)
        if isinstance(related, mcp_types.ErrorData):
            return related
        required_other = _required_other(params)
        if isinstance(required_other, mcp_types.ErrorData):
            return required_other
        if not self.form_capable or related != self._related_task_id:
            return _invalid("MCP elicitation origin is unavailable")
        if params.task is not None:
            return _invalid("Task-augmented elicitation is not negotiated")
        if _SECRET_PATTERN.search(params.message):
            return _invalid(
                "Sensitive or authentication input requires a separate flow"
            )
        try:
            questions = _questions(
                params.requestedSchema,
                required_other=required_other,
            )
        except (InputContractError, TypeError, ValueError):
            return _invalid("MCP form schema is invalid")
        try:
            return _broker_result(
                await self._request(params.message, questions),
                related_task_id=related,
            )
        except CancelledError:
            raise
        except InputContractError as error:
            return _invalid(error.safe_message)
        except (RuntimeError, TypeError, ValueError):
            return mcp_types.ErrorData(
                code=mcp_types.INTERNAL_ERROR,
                message="Avalan could not complete downstream elicitation",
            )

    async def _request(
        self,
        message: str,
        questions: tuple[InputQuestion, ...],
    ) -> InteractionRequestResult:
        execution = self._context.execution
        origin = self._origin
        broker = self._broker
        assert (
            execution is not None and origin is not None and broker is not None
        )
        if execution.origin != origin:
            raise RuntimeError("MCP call origin is no longer active")
        runtime = execution.interaction_runtime
        assert runtime is not None
        return await broker.request(
            InteractionBrokerRequest(
                actor=runtime.actor,
                origin=origin,
                mode=RequirementMode.REQUIRED,
                reason=message,
                questions=questions,
                context_label=_context_label(
                    self._uri,
                    self._tool_name,
                    origin.task_id,
                    origin.agent_id,
                    origin.branch_id,
                ),
                handler=(
                    runtime.handler
                    if isinstance(runtime, AttachedInteractionRuntime)
                    else None
                ),
                continuation_ttl_seconds=self._ttl_seconds,
            )
        )


async def call_initialized_mcp_tool(
    *,
    uri: str,
    name: str,
    arguments: dict[str, object],
    context: ToolCallContext,
    client_params: Mapping[str, object],
    call_params: Mapping[str, object],
    progress_callback: Callable[
        [float, float | None, str | None],
        Awaitable[None],
    ],
    logging_callback: LoggingFnT,
    message_handler: MessageHandlerFnT,
) -> dict[str, object]:
    """Run one initialized bidirectional MCP tool call."""
    call_timeout = _call_timeout(call_params)
    related_task_id = _expected_related_task(call_params)
    router = _ElicitationRouter(
        uri=uri,
        tool_name=name,
        context=context,
        related_task_id=related_task_id,
        ttl_seconds=max(60, int(call_timeout)),
    )
    if context.cancellation_checker is not None:
        await context.cancellation_checker()
    try:
        async with timeout(call_timeout):
            client_factory = cast(Callable[..., AsyncClient], AsyncClient)
            async with client_factory(
                **_http_options(client_params)
            ) as client:
                async with streamable_http_client(
                    uri,
                    http_client=client,
                ) as streams:
                    read_stream, write_stream, _session_id = streams
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=call_timeout),
                        elicitation_callback=cast(ElicitationFnT, router),
                        logging_callback=logging_callback,
                        message_handler=message_handler,
                        client_info=mcp_types.Implementation(
                            name="avalan",
                            version=metadata_version("avalan"),
                        ),
                    ) as session:
                        initialized = await _initialize(
                            session,
                            form_capable=router.form_capable,
                        )
                        if initialized.capabilities.tools is None:
                            raise RuntimeError(
                                "MCP server did not negotiate tools"
                            )
                        result = await session.call_tool(
                            name,
                            arguments,
                            read_timeout_seconds=timedelta(
                                seconds=call_timeout
                            ),
                            progress_callback=cast(
                                ProgressFnT,
                                progress_callback,
                            ),
                            meta=_call_meta(call_params, related_task_id),
                        )
                        if context.cancellation_checker is not None:
                            await context.cancellation_checker()
                        return cast(
                            dict[str, object],
                            result.model_dump(
                                by_alias=True,
                                mode="json",
                                exclude_defaults=True,
                                exclude_none=True,
                            ),
                        )
    except TimeoutError as error:
        raise RuntimeError("MCP tool call timed out") from error


def _client_capabilities(
    form_capable: bool,
) -> mcp_types.ClientCapabilities:
    return mcp_types.ClientCapabilities(
        elicitation=(
            mcp_types.ElicitationCapability(
                form=mcp_types.FormElicitationCapability()
            )
            if form_capable
            else None
        )
    )


async def _initialize(
    session: ClientSession,
    *,
    form_capable: bool,
) -> mcp_types.InitializeResult:
    result = await session.send_request(
        mcp_types.ClientRequest(
            mcp_types.InitializeRequest(
                params=mcp_types.InitializeRequestParams(
                    protocolVersion=MCP_PROTOCOL_VERSION,
                    capabilities=_client_capabilities(form_capable),
                    clientInfo=mcp_types.Implementation(
                        name="avalan",
                        version=metadata_version("avalan"),
                    ),
                )
            )
        ),
        mcp_types.InitializeResult,
    )
    if result.protocolVersion != MCP_PROTOCOL_VERSION:
        raise RuntimeError("MCP server did not negotiate protocol 2025-11-25")
    session._server_capabilities = result.capabilities
    await session.send_notification(
        mcp_types.ClientNotification(mcp_types.InitializedNotification())
    )
    return result


def _call_timeout(params: Mapping[str, object]) -> float:
    value = params.get("timeout", _CALL_TIMEOUT_SECONDS)
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not 0 < value <= _MAX_CALL_TIMEOUT_SECONDS
    ):
        raise ValueError("MCP call timeout is outside its permitted range")
    return float(value)


def _expected_related_task(params: Mapping[str, object]) -> str | None:
    value = params.get("related_task_id")
    if value is None:
        return None
    if not _valid_task_id(value):
        raise ValueError("MCP related task identity is invalid")
    return cast(str, value)


def _call_meta(
    params: Mapping[str, object],
    related_task_id: str | None,
) -> dict[str, object]:
    progress = params.get("progress_token", params.get("request_id"))
    meta: dict[str, object] = {}
    if isinstance(progress, str | int) and not isinstance(progress, bool):
        meta["progressToken"] = progress
    if related_task_id:
        meta.update(_related_task_meta(related_task_id))
    return meta


def _related_task_meta(related_task_id: str) -> dict[str, object]:
    return {
        MCP_RELATED_TASK_METADATA_KEY: {
            "taskId": related_task_id,
        }
    }


def _http_options(params: Mapping[str, object]) -> dict[str, object]:
    options = dict(params)
    raw_headers = options.pop("headers", None)
    headers = (
        dict(cast(Mapping[str, str], raw_headers))
        if isinstance(raw_headers, Mapping)
        else {}
    )
    headers.setdefault("Accept", "application/json, text/event-stream")
    headers.setdefault("Content-Type", "application/json")
    options.setdefault("timeout", None)
    options["headers"] = headers
    return options


def _related_task(
    params: mcp_types.ElicitRequestFormParams,
) -> str | None | mcp_types.ErrorData:
    raw = (
        params.meta.model_dump(mode="python").get(
            MCP_RELATED_TASK_METADATA_KEY
        )
        if params.meta is not None
        else None
    )
    if raw is None:
        return None
    if (
        not isinstance(raw, dict)
        or set(raw) != {"taskId"}
        or not _valid_task_id(raw.get("taskId"))
    ):
        return _invalid("MCP related-task metadata is malformed")
    return cast(str, raw["taskId"])


def _valid_task_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 128
        and len(value.encode("utf-8")) <= 512
        and not any(character.isspace() for character in value)
    )


def _required_other(
    params: mcp_types.ElicitRequestFormParams,
) -> frozenset[str] | mcp_types.ErrorData:
    raw = (
        params.meta.model_dump(mode="python").get(
            MCP_REQUIRED_OTHER_METADATA_KEY
        )
        if params.meta is not None
        else None
    )
    if raw is None:
        return frozenset()
    if (
        not isinstance(raw, list)
        or not 1 <= len(raw) <= 3
        or any(not isinstance(item, str) for item in raw)
        or len(set(raw)) != len(raw)
    ):
        return _invalid("Avalan required-Other metadata is malformed")
    return frozenset(cast(list[str], raw))


def _questions(
    schema: Mapping[str, object],
    *,
    required_other: frozenset[str] = frozenset(),
) -> tuple[InputQuestion, ...]:
    encoded = dumps(schema, ensure_ascii=False, separators=(",", ":"))
    if (
        len(encoded) > _MAX_SCHEMA_CHARACTERS
        or len(encoded.encode("utf-8")) > _MAX_SCHEMA_BYTES
        or _SECRET_PATTERN.search(encoded)
    ):
        raise ValueError("MCP form schema is unsafe or oversized")
    properties = schema.get("properties")
    required = schema.get("required", [])
    if (
        set(schema) - _FORM_ROOT_KEYS
        or schema.get("type") != "object"
        or (
            "additionalProperties" in schema
            and schema["additionalProperties"] is not False
        )
        or not isinstance(properties, dict)
        or not isinstance(required, list)
        or any(not isinstance(item, str) for item in required)
        or len(set(required)) != len(required)
        or any(item not in properties for item in required)
        or any(
            not isinstance(name, str) or not isinstance(value, dict)
            for name, value in properties.items()
        )
    ):
        raise ValueError("MCP form schema must be one flat object")
    typed_properties = cast(dict[str, dict[str, object]], properties)
    other_questions = _other_questions(typed_properties, required)
    if not required_other.issubset(other_questions):
        raise ValueError("Avalan required-Other metadata is invalid")
    questions = tuple(
        (name, value)
        for name, value in typed_properties.items()
        if not name.startswith(_FORM_OTHER_PROPERTY_PREFIX)
    )
    if not 1 <= len(questions) <= 3:
        raise ValueError("MCP form schema must contain one to three questions")
    return tuple(
        _question(
            name,
            value,
            name in required or name in required_other,
            allow_other=name in other_questions,
        )
        for name, value in questions
    )


def _other_questions(
    properties: Mapping[str, Mapping[str, object]],
    required: list[object],
) -> frozenset[str]:
    questions: set[str] = set()
    for name, schema in properties.items():
        if not name.startswith(_FORM_OTHER_PROPERTY_PREFIX):
            continue
        question_id = name.removeprefix(_FORM_OTHER_PROPERTY_PREFIX)
        description = schema.get("description")
        if (
            not question_id
            or question_id not in properties
            or name in required
            or set(schema)
            != {
                "description",
                "maxLength",
                "minLength",
                "title",
                "type",
            }
            or schema.get("type") != "string"
            or schema.get("title") != "Other"
            or not isinstance(description, str)
            or not description.startswith("Free-form alternative for ")
            or schema.get("minLength") != 1
            or schema.get("maxLength") != 4_096
        ):
            raise ValueError("MCP free-form alternative is malformed")
        question = properties[question_id]
        if question.get("type") == "string":
            if "enum" not in question and "oneOf" not in question:
                raise ValueError("MCP free-form alternative is unsupported")
        elif question.get("type") != "array":
            raise ValueError("MCP free-form alternative is unsupported")
        questions.add(question_id)
    return frozenset(questions)


def _question(
    name: str,
    raw: dict[str, object],
    required: bool,
    *,
    allow_other: bool = False,
) -> InputQuestion:
    question_id = QuestionId(name)
    title = _optional_text(raw.get("title"), 40)
    description = _optional_text(raw.get("description"), 500)
    common = _QuestionCommon(
        question_id=question_id,
        prompt=description or title or name,
        header=title if description else None,
        required=required,
    )
    match raw.get("type"):
        case "boolean":
            _require_property_keys(raw)
            default = raw.get("default")
            if default is not None and not isinstance(default, bool):
                raise ValueError("MCP boolean default is invalid")
            return ConfirmationQuestion(
                **common,
                default_value=default,
            )
        case "string" if "enum" in raw or "oneOf" in raw:
            _require_property_keys(raw, "enum", "oneOf")
            choices = _choices(raw, "oneOf")
            default = raw.get("default")
            if default is not None and default not in {
                str(choice.value) for choice in choices
            }:
                raise ValueError("MCP selection default is invalid")
            return SingleSelectionQuestion(
                **common,
                choices=choices,
                allow_other=allow_other,
                default_value=(
                    ChoiceValue(default) if default is not None else None
                ),
            )
        case "string":
            _require_property_keys(raw, "minLength", "maxLength", "pattern")
            pattern = raw.get("pattern")
            multiline = pattern in _MULTILINE_PATTERNS
            if pattern is not None and not multiline:
                raise ValueError("MCP text pattern is unsupported")
            minimum, maximum = _bounds(
                raw.get("minLength", 0),
                raw.get("maxLength", 65_536 if multiline else 4_096),
                65_536 if multiline else 4_096,
            )
            default = raw.get("default")
            if default is not None and not isinstance(default, str):
                raise ValueError("MCP text default is invalid")
            question_type = (
                MultilineTextQuestion if multiline else TextQuestion
            )
            return question_type(
                **common,
                default_value=default,
                constraints=TextValidationConstraints(
                    minimum_length=minimum,
                    maximum_length=maximum,
                ),
            )
        case "array":
            _require_property_keys(
                raw,
                "items",
                "maxItems",
                "minItems",
                "uniqueItems",
            )
            items = raw.get("items")
            if (
                not isinstance(items, dict)
                or set(items) - {"anyOf", "enum", "type"}
                or items.get("type") != "string"
                or ("uniqueItems" in raw and raw["uniqueItems"] is not True)
            ):
                raise ValueError("MCP selection array is invalid")
            choices = _choices(items, "anyOf")
            minimum, maximum = _bounds(
                raw.get("minItems", int(required)),
                raw.get("maxItems", len(choices)),
                min(20, len(choices) + int(allow_other)),
            )
            default = raw.get("default")
            values = {str(choice.value) for choice in choices}
            if default is not None and (
                not isinstance(default, list)
                or any(
                    not isinstance(item, str) or item not in values
                    for item in default
                )
                or len(set(default)) != len(default)
            ):
                raise ValueError("MCP selection default is invalid")
            return MultipleSelectionQuestion(
                **common,
                choices=choices,
                allow_other=allow_other,
                default_value=(
                    tuple(ChoiceValue(cast(str, item)) for item in default)
                    if isinstance(default, list)
                    else None
                ),
                constraints=SelectionValidationConstraints(
                    minimum=minimum,
                    maximum=maximum,
                ),
            )
        case _:
            raise ValueError("MCP form property type is unsupported")


def _require_property_keys(
    schema: Mapping[str, object],
    *keys: str,
) -> None:
    if set(schema) - (_FORM_COMMON_PROPERTY_KEYS | set(keys)):
        raise ValueError("MCP form property is unsupported")


def _choices(
    schema: Mapping[str, object],
    titled_key: str,
) -> tuple[Choice, ...]:
    raw_enum = schema.get("enum")
    raw_titled = schema.get(titled_key)
    if (raw_enum is None) == (raw_titled is None):
        raise ValueError("MCP selection choice form is invalid")
    pairs: list[tuple[str, str]] = []
    if raw_enum is not None:
        if not isinstance(raw_enum, list) or any(
            not isinstance(item, str) for item in raw_enum
        ):
            raise ValueError("MCP selection enum is invalid")
        pairs.extend((item, item) for item in raw_enum)
    else:
        if not isinstance(raw_titled, list):
            raise ValueError("MCP titled selection is invalid")
        for item in raw_titled:
            if (
                not isinstance(item, dict)
                or set(item) != {"const", "title"}
                or not isinstance(item.get("const"), str)
                or not isinstance(item.get("title"), str)
            ):
                raise ValueError("MCP titled selection is invalid")
            pairs.append((item["const"], item["title"]))
    if (
        not 1 <= len(pairs) <= 20
        or any(
            not isinstance(value, str)
            or not value
            or not isinstance(label, str)
            or not label
            for value, label in pairs
        )
        or len({value for value, _label in pairs}) != len(pairs)
    ):
        raise ValueError("MCP selection choices are invalid")
    return tuple(
        Choice(
            value=ChoiceValue(value),
            label=("Other option" if label == "Other" else label),
        )
        for value, label in pairs
    )


def _bounds(minimum: object, maximum: object, limit: int) -> tuple[int, int]:
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or not 0 <= minimum <= maximum <= limit
    ):
        raise ValueError("MCP form constraints are invalid")
    return minimum, maximum


def _optional_text(value: object, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError("MCP form presentation text is invalid")
    return value


def _broker_result(
    result: InteractionRequestResult,
    *,
    related_task_id: str | None = None,
) -> mcp_types.ElicitResult | mcp_types.ErrorData:
    if result.delivery is None:
        return _invalid("Avalan rejected downstream elicitation")
    resolution = result.delivery.record.request.resolution
    meta = (
        _related_task_meta(related_task_id)
        if related_task_id is not None
        else None
    )
    if isinstance(resolution, AnsweredResolution):
        return mcp_types.ElicitResult(
            action="accept",
            content=_answer_content(resolution.answers),
            _meta=meta,
        )
    if isinstance(resolution, DeclinedResolution):
        return mcp_types.ElicitResult(
            action="decline",
            _meta=meta,
        )
    if isinstance(resolution, CancelledResolution | TimedOutResolution):
        return mcp_types.ElicitResult(
            action="cancel",
            _meta=meta,
        )
    if isinstance(resolution, ExpiredResolution):
        return mcp_types.ErrorData(
            code=_MCP_EXPIRED,
            message="Avalan input request expired",
            data={"code": "avalan.input.expired"},
        )
    if isinstance(resolution, UnavailableResolution):
        return mcp_types.ErrorData(
            code=_MCP_UNAVAILABLE,
            message="Avalan input is unavailable",
            data={"code": "avalan.input.unavailable"},
        )
    if isinstance(resolution, SupersededResolution):
        return mcp_types.ErrorData(
            code=_MCP_CONFLICT,
            message="Avalan input request was superseded",
            data={"code": "avalan.input.already_resolved"},
        )
    return mcp_types.ErrorData(
        code=mcp_types.INTERNAL_ERROR,
        message="Avalan elicitation did not reach a terminal state",
    )


def _answer_content(answers: tuple[InputAnswer, ...]) -> McpFormContent:
    content: McpFormContent = {}
    for answer in answers:
        key = str(answer.question_id)
        if isinstance(
            answer,
            ConfirmationAnswer | TextAnswer | MultilineTextAnswer,
        ):
            content[key] = answer.value
        elif isinstance(answer, SingleSelectionAnswer):
            if isinstance(answer.value, FreeFormOther):
                content[_other_property_name(key)] = answer.value.text
            else:
                content[key] = _selection(answer.value)
        elif isinstance(answer, MultipleSelectionAnswer):
            selected: list[str] = []
            for value in answer.values:
                if isinstance(value, FreeFormOther):
                    content[_other_property_name(key)] = value.text
                else:
                    selected.append(_selection(value))
            content[key] = selected
        else:
            raise TypeError("Unsupported Avalan input answer")
    return content


def _other_property_name(question_id: str) -> str:
    return f"{_FORM_OTHER_PROPERTY_PREFIX}{question_id}"


def _selection(value: SelectionValue) -> str:
    if not isinstance(value, SelectedChoice):
        raise TypeError("Unsupported Avalan selected choice")
    return str(value.value)


def _context_label(
    uri: str,
    tool: str,
    task: object | None,
    agent: object,
    branch: object,
) -> str:
    parts = [
        f"MCP {urlsplit(uri).hostname or 'local'}",
        tool,
        *(
            f"{name} {str(value)[-8:]}"
            for name, value in (
                ("task", task),
                ("agent", agent),
                ("branch", branch),
            )
            if value is not None
        ),
    ]
    return " · ".join(" ".join(part.split()) for part in parts)[:80]


def _invalid(message: str) -> mcp_types.ErrorData:
    return mcp_types.ErrorData(
        code=mcp_types.INVALID_PARAMS,
        message=message,
    )
