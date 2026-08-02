"""Adapt exact native OpenAI Responses profiles to stateless replay."""

from ...types import JsonValue
from ..binding import (
    ConversationCapability,
    ConversationCapabilityProfile,
    ProviderFamily,
    ProviderLaneBinding,
    ProviderTransport,
    normalize_endpoint,
)
from ..contract import (
    ChildLaneRetentionPolicy,
    ConversationModelCallId,
    FailureBoundary,
)
from ..errors import (
    ConversationAmbiguousDispatchError,
    ConversationBindingDriftError,
    ConversationCapabilityError,
    ConversationCommitError,
    ConversationError,
    ConversationErrorCode,
    ConversationProviderResponseError,
    ConversationValidationError,
)
from ..items import (
    PROVIDER_ITEM_NORMALIZATION_VERSION,
    PROVIDER_ITEM_SEMANTICS,
    ProviderItem,
    ProviderItemCaller,
    ProviderItemKind,
    ProviderItemPhase,
    provider_item_byte_count,
)
from ..protocols import (
    ConversationProviderStream,
    ProviderPlan,
    ProviderResult,
    StatelessProviderPlan,
)
from ..settings import (
    DisabledCompaction,
    EffectiveReasoningContext,
    EffectiveReasoningMetadata,
    ProviderUsage,
    ReasoningContext,
)
from ..value import (
    OpaqueProviderState,
    ProviderCallId,
    ProviderItemId,
    ProviderItemIndex,
    ProviderItemOrder,
    canonical_json_bytes,
    freeze_json_value,
    thaw_json_value,
    validate_identifier,
)

from asyncio import CancelledError, Task, create_task, wait
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from inspect import iscoroutinefunction
from json import JSONDecodeError, loads
from typing import cast, final
from urllib.parse import urlsplit

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
)
from openai import (
    __version__ as openai_version,
)
from openai.resources.responses.responses import AsyncResponses
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseIncludable,
    ResponseInputItemParam,
    ResponseStreamEvent,
)
from openai.types.shared_params import Reasoning

_ADAPTER_TYPE = (
    "avalan.conversation.providers.openai.NativeOpenAIStatelessProvider"
)
_SDK_REVISION = "openai-python-2.42.0"
_OPENAI_API_REVISION = "openapi-2.3.0"
_AZURE_API_REVISIONS = frozenset(
    {
        "azure-openai-v1",
        "azure-openai-v1-preview",
    }
)
_ENCRYPTED_CONTENT_INCLUDE: ResponseIncludable = "reasoning.encrypted_content"
_MAX_OPAQUE_BYTES = 1_048_576


async def _owned_close_outcome(
    task: Task[None],
) -> tuple[CancelledError | None, BaseException | None, bool]:
    """Settle one close task without transferring caller cancellation."""
    cancellation: CancelledError | None = None
    while not task.done():
        try:
            await wait({task})
        except CancelledError as exc:
            cancellation = cancellation or exc
    if task.cancelled():
        return cancellation, None, True
    return cancellation, task.exception(), False


class NativeOpenAIEncryptedContentPolicy(StrEnum):
    """Select the exact conformed encrypted-reasoning return behavior."""

    DEFAULT_RETURN = "default_return"
    EXPLICIT_INCLUDE = "explicit_include"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NativeOpenAIStatelessProfile:
    """Bind exact request behavior to one native Responses profile."""

    profile_id: str
    binding: ProviderLaneBinding
    encrypted_content: NativeOpenAIEncryptedContentPolicy
    scripted_tcp_test: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.profile_id, "profile_id")
        if (
            type(self.binding) is not ProviderLaneBinding
            or not isinstance(
                self.encrypted_content,
                NativeOpenAIEncryptedContentPolicy,
            )
            or type(self.scripted_tcp_test) is not bool
        ):
            raise ConversationValidationError()
        binding = self.binding
        if (
            binding.adapter_type != _ADAPTER_TYPE
            or binding.provider_family
            not in {ProviderFamily.OPENAI, ProviderFamily.AZURE_OPENAI}
            or binding.sdk_revision != _SDK_REVISION
            or binding.continuation_codec_version
            != PROVIDER_ITEM_NORMALIZATION_VERSION
        ):
            raise ConversationCapabilityError()
        if binding.provider_family is ProviderFamily.OPENAI:
            if (
                binding.provider_api_revision != _OPENAI_API_REVISION
                or self.encrypted_content
                is not NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ):
                raise ConversationCapabilityError()
        elif (
            binding.provider_api_revision not in _AZURE_API_REVISIONS
            or self.encrypted_content
            is not NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
        ):
            raise ConversationCapabilityError()


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NativeOpenAIFunctionTool:
    """Bind one strict function schema to an asynchronous tool effect."""

    name: str
    parameters: Mapping[str, JsonValue]
    handler: Callable[[Mapping[str, JsonValue]], Awaitable[str]]
    description: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.name, "tool_name")
        if self.description is not None:
            validate_identifier(
                self.description,
                "tool_description",
                max_length=4_096,
            )
        if not callable(self.handler) or not iscoroutinefunction(self.handler):
            raise ConversationValidationError()
        frozen = freeze_json_value(self.parameters)
        if not isinstance(frozen, Mapping):
            raise ConversationValidationError()
        if frozen.get("type") != "object":
            raise ConversationValidationError()
        object.__setattr__(self, "parameters", frozen)

    @property
    def schema(self) -> FunctionToolParam:
        """Return one fresh strict SDK function-tool schema."""
        schema: FunctionToolParam = {
            "type": "function",
            "name": self.name,
            "parameters": cast(
                dict[str, object],
                thaw_json_value(self.parameters),
            ),
            "strict": True,
        }
        if self.description is not None:
            schema["description"] = self.description
        return schema

    async def execute(self, arguments: str) -> str:
        """Validate arguments and execute the bound tool asynchronously."""
        try:
            decoded = loads(
                arguments,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (JSONDecodeError, RecursionError, ValueError):
            raise _provider_failure(boundary="failure_before_output") from None
        frozen = freeze_json_value(decoded)
        if not isinstance(frozen, Mapping):
            raise _provider_failure(boundary="failure_before_output")
        try:
            result = await self.handler(frozen)
        except CancelledError:
            raise
        except BaseException:
            raise _provider_failure(boundary="tool_effect") from None
        if type(result) is not str or len(result.encode("utf-8")) > 1_048_576:
            raise ConversationValidationError()
        return result


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NativeOpenAIProviderDiagnostics:
    """Report content-free request, response, and cleanup accounting."""

    request_count: int
    response_count: int
    stream_close_count: int
    request_item_count: int
    response_item_count: int
    request_byte_count: int
    response_byte_count: int
    request_item_kinds: tuple[str, ...]
    response_item_kinds: tuple[str, ...]
    profile_id: str
    effective_context: EffectiveReasoningContext | None
    failure_boundary: str | None


@final
@dataclass(slots=True)
class _NativeOpenAIDiagnosticsState:
    request_count: int = 0
    response_count: int = 0
    stream_close_count: int = 0
    request_item_count: int = 0
    response_item_count: int = 0
    request_byte_count: int = 0
    response_byte_count: int = 0
    request_item_kinds: tuple[str, ...] = ()
    response_item_kinds: tuple[str, ...] = ()
    effective_context: EffectiveReasoningContext | None = None
    failure_boundary: str | None = None


@final
class NativeOpenAIStatelessProvider:
    """Dispatch exact native OpenAI or Azure stateless Responses calls."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        profile: NativeOpenAIStatelessProfile,
        capability_profile: ConversationCapabilityProfile,
        tools: tuple[NativeOpenAIFunctionTool, ...] = (),
    ) -> None:
        if (
            type(client) is not AsyncOpenAI
            or type(profile) is not NativeOpenAIStatelessProfile
            or type(capability_profile) is not ConversationCapabilityProfile
            or type(tools) is not tuple
            or any(
                type(tool) is not NativeOpenAIFunctionTool for tool in tools
            )
        ):
            raise ConversationValidationError()
        profile.binding.assert_compatible(profile.binding)
        capability_profile.assert_binding(profile.binding)
        if len({tool.name for tool in tools}) != len(tools):
            raise ConversationValidationError()
        _validate_sdk_client(client, profile, capability_profile)
        self._client = client
        self._profile = profile
        self._capability_profile = capability_profile
        self._tools = {tool.name: tool for tool in tools}
        self._diagnostics = _NativeOpenAIDiagnosticsState()
        self._closed = False
        self._close_task: Task[None] | None = None

    @property
    def binding(self) -> ProviderLaneBinding:
        """Return the immutable exact provider-lane binding."""
        return self._profile.binding

    @property
    def capability_profile(self) -> ConversationCapabilityProfile:
        """Return the immutable exact capability evidence profile."""
        return self._capability_profile

    @property
    def diagnostics(self) -> NativeOpenAIProviderDiagnostics:
        """Return a content-free immutable accounting snapshot."""
        state = self._diagnostics
        return NativeOpenAIProviderDiagnostics(
            request_count=state.request_count,
            response_count=state.response_count,
            stream_close_count=state.stream_close_count,
            request_item_count=state.request_item_count,
            response_item_count=state.response_item_count,
            request_byte_count=state.request_byte_count,
            response_byte_count=state.response_byte_count,
            request_item_kinds=state.request_item_kinds,
            response_item_kinds=state.response_item_kinds,
            profile_id=self._profile.profile_id,
            effective_context=state.effective_context,
            failure_boundary=state.failure_boundary,
        )

    async def dispatch(self, plan: ProviderPlan) -> ProviderResult:
        """Dispatch one exact non-streaming stateless request."""
        stateless = self._validate_plan(plan, ProviderTransport.NON_STREAMING)
        input_items = _request_input_items(stateless)
        try:
            response = await self._create(input_items, stateless, stream=False)
            if type(response) is not Response:
                raise _provider_failure(boundary="failure_before_output")
            result = _provider_result(response, stateless)
        except ConversationError as exc:
            self._diagnostics.failure_boundary = exc.boundary.value
            raise
        self._record_response(result)
        return result

    async def stream(self, plan: ProviderPlan) -> ConversationProviderStream:
        """Open one exact streaming stateless request."""
        stateless = self._validate_plan(plan, ProviderTransport.STREAMING)
        input_items = _request_input_items(stateless)
        try:
            response = await self._create(input_items, stateless, stream=True)
            if type(response) is not AsyncStream:
                raise _provider_failure(boundary="failure_before_output")
        except ConversationError as exc:
            self._diagnostics.failure_boundary = exc.boundary.value
            raise
        return _NativeOpenAIProviderStream(
            source=response,
            plan=stateless,
            owner=self,
        )

    async def execute_tool(self, item: ProviderItem) -> str:
        """Execute one exact configured function call asynchronously."""
        if (
            type(item) is not ProviderItem
            or item.kind is not ProviderItemKind.FUNCTION_CALL
        ):
            raise ConversationValidationError()
        name = item.canonical_input["name"]
        arguments = item.canonical_input["arguments"]
        if type(name) is not str or type(arguments) is not str:
            raise ConversationValidationError()
        tool = self._tools.get(name)
        if tool is None:
            raise ConversationCapabilityError()
        return await tool.execute(arguments)

    async def aclose(self) -> None:
        """Settle the owned SDK client close exactly once."""
        if self._closed:
            return
        task = self._close_task
        if task is None:
            task = create_task(self._client.close())
            self._close_task = task
        cancellation, error, task_cancelled = await _owned_close_outcome(task)
        if task_cancelled:
            self._close_task = None
            raise cancellation or CancelledError()
        if error is not None:
            self._close_task = None
            failure = ConversationCommitError()
            if cancellation is not None:
                raise cancellation from failure
            raise failure from None
        self._closed = True
        if cancellation is not None:
            raise cancellation

    def _validate_plan(
        self,
        plan: ProviderPlan,
        transport: ProviderTransport,
    ) -> StatelessProviderPlan:
        if self._closed or type(plan) is not StatelessProviderPlan:
            raise ConversationCapabilityError()
        binding = self.binding
        plan.binding.assert_compatible(binding)
        self._capability_profile.assert_binding(binding)
        if binding.transport is not transport:
            raise ConversationBindingDriftError()
        self._capability_profile.require(
            ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY
        )
        if transport is ProviderTransport.STREAMING:
            self._capability_profile.require(
                ConversationCapability.STREAMING_ITEM_FIDELITY
            )
        if plan.reasoning.requested is ReasoningContext.CURRENT_TURN:
            self._capability_profile.require(
                ConversationCapability.REASONING_CONTEXT_CURRENT_TURN
            )
        elif plan.reasoning.requested is ReasoningContext.ALL_TURNS:
            self._capability_profile.require(
                ConversationCapability.REASONING_CONTEXT_ALL_TURNS
            )
        if type(plan.compaction) is not DisabledCompaction:
            raise ConversationCapabilityError()
        _validate_sdk_client(
            self._client,
            self._profile,
            self._capability_profile,
        )
        if plan.ledger.lane_id != binding.lane_id:
            raise ConversationBindingDriftError()
        return plan

    async def _create(
        self,
        input_items: list[ResponseInputItemParam],
        plan: StatelessProviderPlan,
        *,
        stream: bool,
    ) -> Response | AsyncStream[ResponseStreamEvent]:
        include = (
            [_ENCRYPTED_CONTENT_INCLUDE]
            if self._profile.encrypted_content
            is NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
            else None
        )
        reasoning: Reasoning | None = None
        if plan.reasoning.requested is ReasoningContext.CURRENT_TURN:
            reasoning = {"context": "current_turn"}
        elif plan.reasoning.requested is ReasoningContext.ALL_TURNS:
            reasoning = {"context": "all_turns"}
        tools = [tool.schema for tool in self._tools.values()]
        self._record_request(input_items)
        try:
            return await _create_exact_response(
                self._client,
                model=self.binding.model_or_deployment,
                input_items=input_items,
                include=include,
                reasoning=reasoning,
                tools=tools,
                stream=stream,
            )
        except CancelledError:
            raise
        except APIResponseValidationError:
            raise _provider_failure(boundary="failure_before_output") from None
        except APIStatusError:
            raise _provider_failure(boundary="provider_rejection") from None
        except (APIConnectionError, APITimeoutError):
            raise ConversationAmbiguousDispatchError() from None
        except Exception:
            raise ConversationAmbiguousDispatchError() from None

    def _record_request(
        self,
        items: list[ResponseInputItemParam],
    ) -> None:
        payload = freeze_json_value(items)
        state = self._diagnostics
        state.request_count += 1
        state.request_item_count += len(items)
        state.request_byte_count += len(canonical_json_bytes(payload))
        state.request_item_kinds += tuple(
            str(item.get("type", "unknown")) for item in items
        )

    def _record_response(self, result: ProviderResult) -> None:
        state = self._diagnostics
        state.response_count += 1
        state.response_item_count += len(result.items)
        state.response_byte_count += sum(
            provider_item_byte_count(item) for item in result.items
        )
        state.response_item_kinds += tuple(
            item.kind.value for item in result.items
        )
        state.effective_context = result.reasoning.effective
        state.failure_boundary = None

    def _record_stream_close(self) -> None:
        self._diagnostics.stream_close_count += 1


@final
@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class NativeOpenAIConversationLaneRuntime:
    """Bind one closed native provider to coordinator lane authority."""

    provider: NativeOpenAIStatelessProvider
    retention_policy: ChildLaneRetentionPolicy = (
        ChildLaneRetentionPolicy.RETAIN
    )
    max_output_items: int = 1_024
    max_output_bytes: int = 8_388_608
    max_output_segments: int = 1_024

    def __post_init__(self) -> None:
        if (
            type(self.provider) is not NativeOpenAIStatelessProvider
            or type(self.retention_policy) is not ChildLaneRetentionPolicy
            or type(self.max_output_items) is not int
            or self.max_output_items <= 0
            or type(self.max_output_bytes) is not int
            or self.max_output_bytes <= 0
            or type(self.max_output_segments) is not int
            or self.max_output_segments <= 0
        ):
            raise ConversationValidationError()

    @property
    def binding(self) -> ProviderLaneBinding:
        """Return the provider's exact immutable binding."""
        return self.provider.binding

    @property
    def capability_profile(self) -> ConversationCapabilityProfile:
        """Return the provider's exact immutable capability profile."""
        return self.provider.capability_profile


@final
class _NativeOpenAIProviderStream(AsyncIterator[ProviderItem]):
    def __init__(
        self,
        *,
        source: AsyncStream[ResponseStreamEvent],
        plan: StatelessProviderPlan,
        owner: NativeOpenAIStatelessProvider,
    ) -> None:
        self._source = source
        self._iterator = source.__aiter__()
        self._plan = plan
        self._owner = owner
        self._done_items: dict[int, Mapping[str, JsonValue]] = {}
        self._sequence_numbers: set[int] = set()
        self._terminal_result: ProviderResult | None = None
        self._closed = False
        self._close_task: Task[None] | None = None
        self._saw_output = False

    def __aiter__(self) -> AsyncIterator[ProviderItem]:
        return self

    async def __anext__(self) -> ProviderItem:
        while True:
            try:
                event = await self._iterator.__anext__()
            except StopAsyncIteration:
                raise
            except CancelledError:
                raise
            except BaseException:
                error = (
                    _stream_failure(True)
                    if self._saw_output
                    else ConversationAmbiguousDispatchError()
                )
                self._owner._diagnostics.failure_boundary = (
                    error.boundary.value
                )
                raise error from None
            try:
                payload = _sdk_mapping(event)
                self._accept_sequence(payload)
                event_type = payload.get("type")
                if event_type == "response.output_item.done":
                    if self._terminal_result is not None:
                        raise _stream_failure(self._saw_output)
                    index = payload.get("output_index")
                    raw_item = payload.get("item")
                    if (
                        type(index) is not int
                        or index < 0
                        or index != len(self._done_items)
                        or index in self._done_items
                        or not isinstance(raw_item, Mapping)
                    ):
                        raise _stream_failure(self._saw_output)
                    item_mapping = raw_item
                    self._done_items[index] = item_mapping
                    item = _provider_item(
                        item_mapping,
                        plan=self._plan,
                        provider_index=index,
                    )
                    self._saw_output = True
                    return item
                if event_type == "response.completed":
                    if self._terminal_result is not None:
                        raise _stream_failure(self._saw_output)
                    response = payload.get("response")
                    if not isinstance(response, Mapping):
                        raise _stream_failure(self._saw_output)
                    response_mapping = response
                    _validate_stream_terminal(
                        response_mapping,
                        self._done_items,
                    )
                    result = _provider_result_mapping(
                        response_mapping,
                        self._plan,
                    )
                    self._terminal_result = result
                elif event_type in {
                    "response.failed",
                    "response.incomplete",
                    "error",
                }:
                    raise _stream_failure(self._saw_output)
            except ConversationError as exc:
                if self._saw_output and isinstance(
                    exc, ConversationProviderResponseError
                ):
                    error = _stream_failure(True)
                    self._owner._diagnostics.failure_boundary = (
                        error.boundary.value
                    )
                    raise error from None
                self._owner._diagnostics.failure_boundary = exc.boundary.value
                raise

    async def terminal(self) -> ProviderResult:
        """Return validated terminal metadata after complete iteration."""
        result = self._terminal_result
        if result is None:
            error = _stream_failure(self._saw_output)
            self._owner._diagnostics.failure_boundary = error.boundary.value
            raise error
        self._owner._record_response(result)
        return result

    async def aclose(self) -> None:
        """Settle the exact SDK stream close idempotently."""
        if self._closed:
            return
        task = self._close_task
        if task is None:
            task = create_task(self._source.close())
            self._close_task = task
        cancellation, error, task_cancelled = await _owned_close_outcome(task)
        if task_cancelled:
            self._close_task = None
            raise cancellation or CancelledError()
        if error is not None:
            self._close_task = None
            failure = _provider_failure(boundary="malformed_stream_item")
            if cancellation is not None:
                raise cancellation from failure
            raise failure from None
        self._closed = True
        self._owner._record_stream_close()
        if cancellation is not None:
            raise cancellation

    def _accept_sequence(self, payload: Mapping[str, JsonValue]) -> None:
        sequence = payload.get("sequence_number")
        if sequence is None:
            return
        if (
            type(sequence) is not int
            or sequence < 0
            or sequence in self._sequence_numbers
        ):
            raise _stream_failure(self._saw_output)
        self._sequence_numbers.add(sequence)


async def _create_exact_response(
    client: AsyncOpenAI,
    *,
    model: str,
    input_items: list[ResponseInputItemParam],
    include: list[ResponseIncludable] | None,
    reasoning: Reasoning | None,
    tools: list[FunctionToolParam],
    stream: bool,
) -> Response | AsyncStream[ResponseStreamEvent]:
    if stream:
        if include is not None and reasoning is not None:
            return await client.responses.create(
                model=model,
                input=input_items,
                include=include,
                reasoning=reasoning,
                tools=tools,
                store=False,
                stream=True,
            )
        if include is not None:
            return await client.responses.create(
                model=model,
                input=input_items,
                include=include,
                tools=tools,
                store=False,
                stream=True,
            )
        if reasoning is not None:
            return await client.responses.create(
                model=model,
                input=input_items,
                reasoning=reasoning,
                tools=tools,
                store=False,
                stream=True,
            )
        return await client.responses.create(
            model=model,
            input=input_items,
            tools=tools,
            store=False,
            stream=True,
        )
    if include is not None and reasoning is not None:
        return await client.responses.create(
            model=model,
            input=input_items,
            include=include,
            reasoning=reasoning,
            tools=tools,
            store=False,
            stream=False,
        )
    if include is not None:
        return await client.responses.create(
            model=model,
            input=input_items,
            include=include,
            tools=tools,
            store=False,
            stream=False,
        )
    if reasoning is not None:
        return await client.responses.create(
            model=model,
            input=input_items,
            reasoning=reasoning,
            tools=tools,
            store=False,
            stream=False,
        )
    return await client.responses.create(
        model=model,
        input=input_items,
        tools=tools,
        store=False,
        stream=False,
    )


def _request_input_items(
    plan: StatelessProviderPlan,
) -> list[ResponseInputItemParam]:
    items: list[ResponseInputItemParam] = []
    for item in plan.ledger.items:
        value = thaw_json_value(item.canonical_input)
        if type(value) is not dict:
            raise ConversationValidationError()
        payload = cast(dict[str, object], value)
        if item.opaque_state is not None:
            try:
                opaque = item.opaque_state._codec_bytes().decode("utf-8")
            except UnicodeDecodeError:
                raise ConversationValidationError() from None
            _validate_opaque_text(opaque)
            payload["encrypted_content"] = opaque
        items.append(cast(ResponseInputItemParam, payload))
    new_input = plan.new_input
    if new_input is not None:
        if set(new_input) != {"text"}:
            raise ConversationValidationError()
        text = new_input.get("text")
        if type(text) is not str or not text:
            raise ConversationValidationError()
        items.append(
            cast(
                ResponseInputItemParam,
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            )
        )
    if not items:
        raise ConversationValidationError()
    return items


def _provider_result(
    response: Response,
    plan: StatelessProviderPlan,
) -> ProviderResult:
    return _provider_result_mapping(_sdk_mapping(response), plan)


def _provider_result_mapping(
    payload: Mapping[str, JsonValue],
    plan: StatelessProviderPlan,
) -> ProviderResult:
    if (
        payload.get("object") != "response"
        or payload.get("status") != "completed"
    ):
        raise _provider_failure(boundary="failure_before_output")
    raw_output = payload.get("output")
    if type(raw_output) is not tuple:
        raise _provider_failure(boundary="failure_before_output")
    items = tuple(
        _provider_item(raw, plan=plan, provider_index=index)
        for index, raw in enumerate(raw_output)
        if isinstance(raw, Mapping)
    )
    if len(items) != len(raw_output):
        raise _provider_failure(boundary="failure_before_output")
    reasoning = _effective_reasoning(payload, plan)
    usage = _provider_usage(payload)
    return ProviderResult(items=items, reasoning=reasoning, usage=usage)


def _provider_item(
    raw: Mapping[str, JsonValue],
    *,
    plan: StatelessProviderPlan,
    provider_index: int,
) -> ProviderItem:
    raw_type = raw.get("type")
    try:
        if type(raw_type) is not str:
            raise ValueError()
        kind = ProviderItemKind(raw_type)
    except ValueError:
        raise _provider_failure(boundary="failure_before_output") from None
    if kind in {
        ProviderItemKind.FUNCTION_CALL_OUTPUT,
        ProviderItemKind.COMPUTER_CALL_OUTPUT,
        ProviderItemKind.TOOL_SEARCH_OUTPUT,
        ProviderItemKind.LOCAL_SHELL_CALL_OUTPUT,
        ProviderItemKind.SHELL_CALL_OUTPUT,
        ProviderItemKind.APPLY_PATCH_CALL_OUTPUT,
        ProviderItemKind.MCP_APPROVAL_RESPONSE,
        ProviderItemKind.CUSTOM_TOOL_CALL_OUTPUT,
    }:
        raise _provider_failure(boundary="failure_before_output")
    phase = _provider_phase(kind, raw)
    rules = tuple(
        rule
        for rule in PROVIDER_ITEM_SEMANTICS[kind]
        if phase in rule.phases and ProviderItemCaller.PROVIDER in rule.callers
    )
    if len(rules) != 1:
        raise _provider_failure(boundary="failure_before_output")
    rule = rules[0]
    canonical = {
        key: value for key, value in raw.items() if key in rule.allowed_fields
    }
    status = canonical.get("status")
    if status is not None and status != "completed":
        raise _provider_failure(boundary="failure_before_output")
    raw_id = raw.get("id")
    model_call_id = _model_call_id(plan)
    item_id = (
        ProviderItemId(raw_id)
        if type(raw_id) is str and raw_id
        else ProviderItemId(f"provider-item-{model_call_id}-{provider_index}")
    )
    call_id_value = raw.get("call_id")
    call_id = (
        ProviderCallId(call_id_value)
        if type(call_id_value) is str and call_id_value
        else None
    )
    opaque = None
    if rule.opaque_required:
        encrypted = raw.get("encrypted_content")
        if type(encrypted) is not str:
            raise _provider_failure(boundary="failure_before_output")
        _validate_opaque_text(encrypted)
        opaque = OpaqueProviderState(_value=encrypted.encode("utf-8"))
    try:
        return ProviderItem(
            item_id=item_id,
            lane_id=plan.binding.lane_id,
            model_call_id=model_call_id,
            kind=kind,
            order=ProviderItemOrder(len(plan.ledger.items) + provider_index),
            provider_index=ProviderItemIndex(provider_index),
            phase=phase,
            caller=ProviderItemCaller.PROVIDER,
            canonical_input=canonical,
            normalization_version=PROVIDER_ITEM_NORMALIZATION_VERSION,
            call_id=call_id,
            opaque_state=opaque,
        )
    except ConversationValidationError:
        raise _provider_failure(boundary="failure_before_output") from None


def _provider_phase(
    kind: ProviderItemKind,
    raw: Mapping[str, JsonValue],
) -> ProviderItemPhase:
    if kind is ProviderItemKind.COMPACTION:
        return ProviderItemPhase.COMPACTION
    if kind is ProviderItemKind.MESSAGE:
        phase = raw.get("phase")
        if phase is None or phase == "final_answer":
            return ProviderItemPhase.FINAL
        if phase == "commentary":
            return ProviderItemPhase.ASSISTANT
        raise _provider_failure(boundary="failure_before_output")
    return ProviderItemPhase.ASSISTANT


def _model_call_id(plan: StatelessProviderPlan) -> ConversationModelCallId:
    prior = {item.model_call_id for item in plan.ledger.items}
    return ConversationModelCallId(f"native-model-call-{len(prior) + 1}")


def _effective_reasoning(
    payload: Mapping[str, JsonValue],
    plan: StatelessProviderPlan,
) -> EffectiveReasoningMetadata:
    raw_reasoning = payload.get("reasoning")
    if not isinstance(raw_reasoning, Mapping):
        raise _provider_failure(boundary="failure_before_output")
    raw_context = raw_reasoning.get("context")
    try:
        if type(raw_context) is not str:
            raise ValueError()
        effective = EffectiveReasoningContext(raw_context)
    except ValueError:
        raise _provider_failure(boundary="failure_before_output") from None
    return EffectiveReasoningMetadata(
        requested=plan.reasoning.requested,
        effective=effective,
    )


def _provider_usage(payload: Mapping[str, JsonValue]) -> ProviderUsage:
    raw_usage = payload.get("usage")
    if not isinstance(raw_usage, Mapping):
        raise _provider_failure(boundary="failure_before_output")
    input_tokens = raw_usage.get("input_tokens")
    output_tokens = raw_usage.get("output_tokens")
    if (
        type(input_tokens) is not int
        or input_tokens < 0
        or type(output_tokens) is not int
        or output_tokens < 0
    ):
        raise _provider_failure(boundary="failure_before_output")
    return ProviderUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _validate_stream_terminal(
    response: Mapping[str, JsonValue],
    done_items: Mapping[int, Mapping[str, JsonValue]],
) -> None:
    raw_output = response.get("output")
    if type(raw_output) is not tuple or set(done_items) != set(
        range(len(raw_output))
    ):
        raise _stream_failure(bool(done_items))
    for index, terminal in enumerate(raw_output):
        if not isinstance(terminal, Mapping):
            raise _stream_failure(bool(done_items))
        if canonical_json_bytes(terminal) != canonical_json_bytes(
            done_items[index]
        ):
            raise _stream_failure(True)


def _sdk_mapping(value: object) -> Mapping[str, JsonValue]:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        raise _provider_failure(boundary="failure_before_output")
    dumped = model_dump(mode="json", exclude_none=True)
    frozen = freeze_json_value(dumped)
    if not isinstance(frozen, Mapping):
        raise _provider_failure(boundary="failure_before_output")
    return frozen


def _validate_sdk_client(
    client: AsyncOpenAI,
    profile: NativeOpenAIStatelessProfile,
    capability_profile: ConversationCapabilityProfile,
) -> None:
    binding = profile.binding
    capability_profile.assert_binding(binding)
    if (
        openai_version != "2.42.0"
        or client.max_retries != 0
        or type(client.responses) is not AsyncResponses
    ):
        raise ConversationCapabilityError()
    base_url = normalize_endpoint(str(client.base_url))
    if base_url != binding.normalized_endpoint:
        raise ConversationBindingDriftError()
    default_query = dict(client.default_query)
    parsed = urlsplit(base_url)
    if profile.scripted_tcp_test:
        if (
            not capability_profile.test_only
            or parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
        ):
            raise ConversationCapabilityError()
    elif binding.provider_family is ProviderFamily.OPENAI:
        if base_url != "https://api.openai.com/v1" or default_query:
            raise ConversationCapabilityError()
    else:
        hostname = parsed.hostname
        if (
            parsed.scheme != "https"
            or hostname is None
            or not hostname.endswith(
                (
                    ".openai.azure.com",
                    ".cognitiveservices.azure.com",
                )
            )
            or parsed.path != "/openai/v1"
            or binding.azure_resource_identity != hostname
        ):
            raise ConversationCapabilityError()
    expected_query = (
        {"api-version": "preview"}
        if binding.provider_api_revision == "azure-openai-v1-preview"
        else {}
    )
    if default_query != expected_query:
        raise ConversationBindingDriftError()


def _validate_opaque_text(value: str) -> None:
    if (
        not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > _MAX_OPAQUE_BYTES
    ):
        raise _provider_failure(boundary="failure_before_output")


def _provider_failure(*, boundary: str) -> ConversationError:
    if boundary == FailureBoundary.FAILURE_BEFORE_OUTPUT.value:
        return ConversationProviderResponseError()
    return ConversationError(
        ConversationErrorCode.VALIDATION_FAILED,
        boundary=FailureBoundary(boundary),
    )


def _stream_failure(saw_output: bool) -> ConversationError:
    return _provider_failure(
        boundary=(
            "malformed_stream_item" if saw_output else "failure_before_output"
        )
    )


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(value)
