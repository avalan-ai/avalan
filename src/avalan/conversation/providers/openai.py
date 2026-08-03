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
    UpstreamResponseId,
)
from ..errors import (
    ConversationAmbiguousDispatchError,
    ConversationBindingDriftError,
    ConversationCapabilityError,
    ConversationCommitError,
    ConversationError,
    ConversationErrorCode,
    ConversationLimitError,
    ConversationProviderResponseError,
    ConversationValidationError,
)
from ..items import (
    PROVIDER_ITEM_NORMALIZATION_VERSION,
    PROVIDER_ITEM_SEMANTICS,
    ProviderItem,
    ProviderItemCaller,
    ProviderItemKind,
    ProviderItemLedger,
    ProviderItemPhase,
    provider_item_byte_count,
    provider_replay_items,
)
from ..protocols import (
    ConversationProviderStream,
    FirstStoredProviderPlan,
    ProviderPlan,
    ProviderResult,
    StandaloneCompactProviderPlan,
    StatelessProviderPlan,
    StoredProviderPlan,
)
from ..settings import (
    DisabledCompaction,
    EffectiveReasoningContext,
    EffectiveReasoningMetadata,
    InlineCompaction,
    ProviderUsage,
    ReasoningContext,
)
from ..value import (
    IntegrityDigest,
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
from hashlib import sha256
from inspect import iscoroutinefunction, signature
from json import JSONDecodeError, loads
from typing import Protocol, TypeAlias, cast, final
from urllib.parse import urlsplit

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    omit,
)
from openai import (
    __version__ as openai_version,
)
from openai.resources.responses.responses import AsyncResponses
from openai.types.responses import (
    CompactedResponse,
    FunctionToolParam,
    Response,
    ResponseIncludable,
    ResponseInputItemParam,
    ResponseStreamEvent,
)
from openai.types.responses.response_create_params import ContextManagement
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

NativeOpenAIResponsePlan: TypeAlias = (
    StatelessProviderPlan
    | StandaloneCompactProviderPlan
    | FirstStoredProviderPlan
    | StoredProviderPlan
)


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
class NativeOpenAICompactionLimits:
    """Bound exact inline and standalone compaction allocation."""

    min_compact_threshold: int
    max_compact_threshold: int
    max_input_items: int = 10_000
    max_input_bytes: int = 8_388_608
    max_input_depth: int = 32
    max_output_items: int = 10_000
    max_output_bytes: int = 8_388_608
    max_output_depth: int = 32

    def __post_init__(self) -> None:
        values = (
            self.min_compact_threshold,
            self.max_compact_threshold,
            self.max_input_items,
            self.max_input_bytes,
            self.max_input_depth,
            self.max_output_items,
            self.max_output_bytes,
            self.max_output_depth,
        )
        if (
            any(type(value) is not int or value <= 0 for value in values)
            or self.min_compact_threshold > self.max_compact_threshold
            or self.max_input_depth > 32
            or self.max_output_depth > 32
        ):
            raise ConversationValidationError()


def native_openai_compaction_policy_digest(
    limits: NativeOpenAICompactionLimits | None,
) -> IntegrityDigest | None:
    """Return the exact binding digest for one compaction limit policy."""
    if limits is None:
        return None
    if type(limits) is not NativeOpenAICompactionLimits:
        raise ConversationValidationError()
    payload = freeze_json_value(
        {
            "max_compact_threshold": limits.max_compact_threshold,
            "max_input_bytes": limits.max_input_bytes,
            "max_input_depth": limits.max_input_depth,
            "max_input_items": limits.max_input_items,
            "max_output_bytes": limits.max_output_bytes,
            "max_output_depth": limits.max_output_depth,
            "max_output_items": limits.max_output_items,
            "min_compact_threshold": limits.min_compact_threshold,
        }
    )
    return IntegrityDigest(sha256(canonical_json_bytes(payload)).hexdigest())


def _json_value_depth(value: object) -> int:
    """Return maximum container-edge depth without recursive traversal."""
    maximum = 0
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        maximum = max(maximum, depth)
        if isinstance(current, Mapping):
            pending.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list | tuple):
            pending.extend((item, depth + 1) for item in current)
    return maximum


def _validate_compaction_output_item(
    item: ProviderItem,
    limits: NativeOpenAICompactionLimits,
    *,
    item_count: int,
    byte_count: int,
) -> int:
    """Validate one normalized compaction item before aggregation."""
    next_bytes = byte_count + provider_item_byte_count(item)
    if (
        item_count > limits.max_output_items
        or next_bytes > limits.max_output_bytes
        or _json_value_depth(item.canonical_input) > limits.max_output_depth
    ):
        raise ConversationLimitError()
    return next_bytes


def _bounded_provider_items(
    raw_output: tuple[JsonValue, ...],
    *,
    plan: NativeOpenAIResponsePlan,
    limits: NativeOpenAICompactionLimits | None,
) -> tuple[ProviderItem, ...]:
    """Normalize provider items with incremental compaction bounds."""
    if limits is not None and len(raw_output) > limits.max_output_items:
        raise ConversationLimitError()
    items: list[ProviderItem] = []
    byte_count = 0
    for index, raw in enumerate(raw_output):
        if not isinstance(raw, Mapping):
            raise _provider_failure(boundary="failure_before_output")
        if (
            limits is not None
            and _json_value_depth(raw) > limits.max_output_depth
        ):
            raise ConversationLimitError()
        item = _provider_item(raw, plan=plan, provider_index=index)
        if limits is not None:
            byte_count = _validate_compaction_output_item(
                item,
                limits,
                item_count=index + 1,
                byte_count=byte_count,
            )
        items.append(item)
    return tuple(items)


def _append_bounded_input_item(
    items: list[ResponseInputItemParam],
    payload: dict[str, object],
    limits: NativeOpenAICompactionLimits | None,
    byte_count: int,
) -> int:
    """Append one wire input only after exact incremental bounds pass."""
    if limits is None:
        items.append(cast(ResponseInputItemParam, payload))
        return byte_count
    if _json_value_depth(payload) > limits.max_input_depth:
        raise ConversationLimitError()
    frozen = freeze_json_value(payload)
    next_bytes = byte_count + len(canonical_json_bytes(frozen))
    if (
        len(items) + 1 > limits.max_input_items
        or next_bytes > limits.max_input_bytes
    ):
        raise ConversationLimitError()
    items.append(cast(ResponseInputItemParam, payload))
    return next_bytes


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NativeOpenAIStatelessProfile:
    """Bind exact request behavior to one native Responses profile."""

    profile_id: str
    binding: ProviderLaneBinding
    encrypted_content: NativeOpenAIEncryptedContentPolicy
    compaction_limits: NativeOpenAICompactionLimits | None = None
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
            or (
                self.compaction_limits is not None
                and type(self.compaction_limits)
                is not NativeOpenAICompactionLimits
            )
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
        if binding.compaction_policy_digest != (
            native_openai_compaction_policy_digest(self.compaction_limits)
        ):
            raise ConversationBindingDriftError()


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
    inline_compaction_request_count: int
    standalone_compaction_request_count: int
    compaction_boundary_count: int
    compaction_failure_count: int
    last_compact_threshold: int | None


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
    inline_compaction_request_count: int = 0
    standalone_compaction_request_count: int = 0
    compaction_boundary_count: int = 0
    compaction_failure_count: int = 0
    last_compact_threshold: int | None = None


class _NativeOpenAIProviderOwner(Protocol):
    """Record one native stream without constraining provider mode."""

    _diagnostics: _NativeOpenAIDiagnosticsState

    def _record_response(self, result: ProviderResult) -> None:
        """Record one complete provider response."""
        ...

    def _record_stream_close(self) -> None:
        """Record one settled SDK stream close."""
        ...


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
            inline_compaction_request_count=(
                state.inline_compaction_request_count
            ),
            standalone_compaction_request_count=(
                state.standalone_compaction_request_count
            ),
            compaction_boundary_count=state.compaction_boundary_count,
            compaction_failure_count=state.compaction_failure_count,
            last_compact_threshold=state.last_compact_threshold,
        )

    async def dispatch(self, plan: ProviderPlan) -> ProviderResult:
        """Dispatch one exact non-streaming stateless request."""
        inline = (
            type(plan) is StatelessProviderPlan
            and type(plan.compaction) is InlineCompaction
        )
        try:
            stateless = self._validate_plan(
                plan,
                ProviderTransport.NON_STREAMING,
            )
            limits = self._profile.compaction_limits if inline else None
            input_items = _request_input_items(stateless, limits=limits)
            response = await self._create(input_items, stateless, stream=False)
            if type(response) is not Response:
                raise _provider_failure(boundary="failure_before_output")
            result = _provider_result(response, stateless, limits=limits)
        except CancelledError:
            if inline:
                self._diagnostics.compaction_failure_count += 1
            raise
        except ConversationError as exc:
            self._diagnostics.failure_boundary = exc.boundary.value
            if inline:
                self._diagnostics.compaction_failure_count += 1
            raise
        self._record_response(result)
        return result

    async def compact(
        self,
        plan: StandaloneCompactProviderPlan,
    ) -> ProviderResult:
        """Dispatch one exact standalone compact operation."""
        try:
            compact_plan = self._validate_compact_plan(plan)
            limits = self._profile.compaction_limits
            assert limits is not None
            input_items = _request_input_items(compact_plan, limits=limits)
            self._record_request(input_items)
            self._diagnostics.standalone_compaction_request_count += 1
            response = await self._client.responses.compact(
                model=self.binding.model_or_deployment,
                input=input_items,
            )
            if type(response) is not CompactedResponse:
                raise _provider_failure(boundary="failure_before_output")
            result = _compact_provider_result(
                response,
                compact_plan,
                limits=limits,
            )
        except CancelledError:
            self._diagnostics.compaction_failure_count += 1
            raise
        except ConversationError as exc:
            self._diagnostics.failure_boundary = exc.boundary.value
            self._diagnostics.compaction_failure_count += 1
            raise
        except APIResponseValidationError:
            self._diagnostics.compaction_failure_count += 1
            error = _provider_failure(boundary="failure_before_output")
            self._diagnostics.failure_boundary = error.boundary.value
            raise error from None
        except APIStatusError:
            self._diagnostics.compaction_failure_count += 1
            error = _provider_failure(boundary="provider_rejection")
            self._diagnostics.failure_boundary = error.boundary.value
            raise error from None
        except (APIConnectionError, APITimeoutError):
            self._diagnostics.compaction_failure_count += 1
            error = ConversationAmbiguousDispatchError()
            self._diagnostics.failure_boundary = error.boundary.value
            raise error from None
        except Exception:
            self._diagnostics.compaction_failure_count += 1
            error = ConversationAmbiguousDispatchError()
            self._diagnostics.failure_boundary = error.boundary.value
            raise error from None
        self._record_response(result)
        return result

    async def stream(self, plan: ProviderPlan) -> ConversationProviderStream:
        """Open one exact streaming stateless request."""
        inline = (
            type(plan) is StatelessProviderPlan
            and type(plan.compaction) is InlineCompaction
        )
        try:
            stateless = self._validate_plan(
                plan,
                ProviderTransport.STREAMING,
            )
            limits = self._profile.compaction_limits if inline else None
            input_items = _request_input_items(stateless, limits=limits)
            response = await self._create(input_items, stateless, stream=True)
            if type(response) is not AsyncStream:
                raise _provider_failure(boundary="failure_before_output")
        except CancelledError:
            if inline:
                self._diagnostics.compaction_failure_count += 1
            raise
        except ConversationError as exc:
            self._diagnostics.failure_boundary = exc.boundary.value
            if inline:
                self._diagnostics.compaction_failure_count += 1
            raise
        return _NativeOpenAIProviderStream(
            source=response,
            plan=stateless,
            owner=self,
            compaction_limits=limits,
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
        if type(plan.compaction) is InlineCompaction:
            self._validate_inline_compaction(plan.compaction)
        elif type(plan.compaction) is not DisabledCompaction:
            raise ConversationCapabilityError()
        _validate_sdk_client(
            self._client,
            self._profile,
            self._capability_profile,
        )
        if plan.ledger.lane_id != binding.lane_id:
            raise ConversationBindingDriftError()
        return plan

    def validate_compaction_request(
        self,
        plan: ProviderPlan,
        transport: ProviderTransport,
    ) -> None:
        """Validate compact input without dispatching provider work."""
        if type(plan) is StandaloneCompactProviderPlan:
            if transport is not ProviderTransport.NON_STREAMING:
                raise ConversationCapabilityError()
            compact = self._validate_compact_plan(plan)
            limits = self._profile.compaction_limits
            assert limits is not None
            _request_input_items(compact, limits=limits)
            return
        stateless = self._validate_plan(plan, transport)
        if type(stateless.compaction) is not InlineCompaction:
            raise ConversationValidationError()
        limits = self._profile.compaction_limits
        assert limits is not None
        _request_input_items(stateless, limits=limits)

    def _validate_inline_compaction(
        self,
        compaction: InlineCompaction,
    ) -> None:
        """Validate one exact inline threshold against proven limits."""
        self._capability_profile.require(
            ConversationCapability.INLINE_COMPACTION
        )
        limits = self._profile.compaction_limits
        if (
            limits is None
            or not limits.min_compact_threshold
            <= compaction.compact_threshold
            <= limits.max_compact_threshold
        ):
            raise ConversationCapabilityError()

    def _validate_compact_plan(
        self,
        plan: StandaloneCompactProviderPlan,
    ) -> StandaloneCompactProviderPlan:
        """Validate standalone authority, capability, and input limits."""
        if self._closed or type(plan) is not StandaloneCompactProviderPlan:
            raise ConversationCapabilityError()
        binding = self.binding
        plan.binding.assert_compatible(binding)
        self._capability_profile.assert_binding(binding)
        if binding.transport is not ProviderTransport.NON_STREAMING:
            raise ConversationBindingDriftError()
        self._capability_profile.require(
            ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY
        )
        self._capability_profile.require(
            ConversationCapability.STANDALONE_COMPACTION
        )
        limits = self._profile.compaction_limits
        if limits is None:
            raise ConversationCapabilityError()
        _validate_sdk_client(
            self._client,
            self._profile,
            self._capability_profile,
        )
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
        context_management: list[ContextManagement] | None = None
        if type(plan.compaction) is InlineCompaction:
            context_management = [
                {
                    "type": "compaction",
                    "compact_threshold": plan.compaction.compact_threshold,
                }
            ]
            self._diagnostics.inline_compaction_request_count += 1
            self._diagnostics.last_compact_threshold = (
                plan.compaction.compact_threshold
            )
        self._record_request(input_items)
        try:
            return await _create_exact_response(
                self._client,
                model=self.binding.model_or_deployment,
                input_items=input_items,
                include=include,
                reasoning=reasoning,
                context_management=context_management,
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
        state.compaction_boundary_count += sum(
            item.kind is ProviderItemKind.COMPACTION for item in result.items
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
        plan: NativeOpenAIResponsePlan,
        owner: _NativeOpenAIProviderOwner,
        compaction_limits: NativeOpenAICompactionLimits | None = None,
    ) -> None:
        self._source = source
        self._iterator = source.__aiter__()
        self._plan = plan
        self._owner = owner
        self._compaction_limits = compaction_limits
        self._done_items: dict[int, Mapping[str, JsonValue]] = {}
        self._sequence_numbers: set[int] = set()
        self._terminal_result: ProviderResult | None = None
        self._closed = False
        self._close_task: Task[None] | None = None
        self._saw_output = False
        self._compaction_output_bytes = 0
        self._compaction_failure_recorded = False

    def _record_compaction_failure(self) -> None:
        """Count one failed inline stream without retaining content."""
        if (
            self._compaction_limits is not None
            and not self._compaction_failure_recorded
        ):
            self._owner._diagnostics.compaction_failure_count += 1
            self._compaction_failure_recorded = True

    def __aiter__(self) -> AsyncIterator[ProviderItem]:
        return self

    async def __anext__(self) -> ProviderItem:
        while True:
            try:
                event = await self._iterator.__anext__()
            except StopAsyncIteration:
                raise
            except CancelledError:
                self._record_compaction_failure()
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
                self._record_compaction_failure()
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
                    limits = self._compaction_limits
                    if limits is not None and (
                        index + 1 > limits.max_output_items
                        or _json_value_depth(item_mapping)
                        > limits.max_output_depth
                    ):
                        raise ConversationLimitError()
                    item = _provider_item(
                        item_mapping,
                        plan=self._plan,
                        provider_index=index,
                    )
                    if limits is not None:
                        self._compaction_output_bytes = (
                            _validate_compaction_output_item(
                                item,
                                limits,
                                item_count=index + 1,
                                byte_count=self._compaction_output_bytes,
                            )
                        )
                    self._done_items[index] = item_mapping
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
                        limits=self._compaction_limits,
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
                    self._record_compaction_failure()
                    raise error from None
                self._owner._diagnostics.failure_boundary = exc.boundary.value
                self._record_compaction_failure()
                raise

    async def terminal(self) -> ProviderResult:
        """Return validated terminal metadata after complete iteration."""
        result = self._terminal_result
        if result is None:
            error = _stream_failure(self._saw_output)
            self._owner._diagnostics.failure_boundary = error.boundary.value
            self._record_compaction_failure()
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
            self._record_compaction_failure()
            raise cancellation or CancelledError()
        if error is not None:
            self._close_task = None
            failure = _provider_failure(boundary="malformed_stream_item")
            if cancellation is not None:
                self._record_compaction_failure()
                raise cancellation from failure
            self._record_compaction_failure()
            raise failure from None
        self._closed = True
        self._owner._record_stream_close()
        if cancellation is not None:
            self._record_compaction_failure()
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
    context_management: list[ContextManagement] | None,
    tools: list[FunctionToolParam],
    stream: bool,
) -> Response | AsyncStream[ResponseStreamEvent]:
    if stream:
        return await client.responses.create(
            model=model,
            input=input_items,
            context_management=(
                context_management if context_management is not None else omit
            ),
            include=include if include is not None else omit,
            reasoning=reasoning if reasoning is not None else omit,
            tools=tools,
            store=False,
            stream=True,
        )
    return await client.responses.create(
        model=model,
        input=input_items,
        context_management=(
            context_management if context_management is not None else omit
        ),
        include=include if include is not None else omit,
        reasoning=reasoning if reasoning is not None else omit,
        tools=tools,
        store=False,
        stream=False,
    )


def _request_input_items(
    plan: StatelessProviderPlan | StandaloneCompactProviderPlan,
    *,
    limits: NativeOpenAICompactionLimits | None = None,
) -> list[ResponseInputItemParam]:
    items: list[ResponseInputItemParam] = []
    byte_count = 0
    for item in provider_replay_items(plan.ledger):
        value = thaw_json_value(item.canonical_input)
        if type(value) is not dict:
            raise ConversationValidationError()
        payload = cast(dict[str, object], value)
        if item.kind is ProviderItemKind.COMPACTION:
            payload.pop("created_by", None)
        if item.opaque_state is not None:
            try:
                opaque = item.opaque_state._codec_bytes().decode("utf-8")
            except UnicodeDecodeError:
                raise ConversationValidationError() from None
            _validate_opaque_text(opaque)
            payload["encrypted_content"] = opaque
        byte_count = _append_bounded_input_item(
            items,
            payload,
            limits,
            byte_count,
        )
    new_input = plan.new_input if type(plan) is StatelessProviderPlan else None
    if new_input is not None:
        if set(new_input) != {"text"}:
            raise ConversationValidationError()
        text = new_input.get("text")
        if type(text) is not str or not text:
            raise ConversationValidationError()
        _append_bounded_input_item(
            items,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
            limits,
            byte_count,
        )
    if not items:
        raise ConversationValidationError()
    return items


def _provider_result(
    response: Response,
    plan: StatelessProviderPlan,
    *,
    limits: NativeOpenAICompactionLimits | None = None,
) -> ProviderResult:
    return _provider_result_mapping(
        _sdk_mapping(response),
        plan,
        limits=limits,
    )


def _compact_provider_result(
    response: CompactedResponse,
    plan: StandaloneCompactProviderPlan,
    *,
    limits: NativeOpenAICompactionLimits,
) -> ProviderResult:
    """Return one exact provider-private canonical compact context."""
    payload = _sdk_mapping(response)
    if (
        set(payload) != {"created_at", "id", "object", "output", "usage"}
        or payload.get("object") != "response.compaction"
        or type(payload.get("created_at")) is not int
        or cast(int, payload["created_at"]) < 0
    ):
        raise _provider_failure(boundary="failure_before_output")
    response_id = payload.get("id")
    if type(response_id) is not str:
        raise _provider_failure(boundary="failure_before_output")
    try:
        validate_identifier(response_id, "compact_response_id")
    except ConversationValidationError:
        raise _provider_failure(boundary="failure_before_output") from None
    raw_output = payload.get("output")
    if type(raw_output) is not tuple:
        raise _provider_failure(boundary="failure_before_output")
    items = _bounded_provider_items(
        raw_output,
        plan=plan,
        limits=limits,
    )
    if (
        len(items) != len(raw_output)
        or not items
        or items[-1].kind is not ProviderItemKind.COMPACTION
        or sum(item.kind is ProviderItemKind.COMPACTION for item in items) != 1
        or any(
            item.kind is not ProviderItemKind.MESSAGE
            or item.phase is not ProviderItemPhase.INPUT
            or item.caller is not ProviderItemCaller.CALLER
            for item in items[:-1]
        )
    ):
        raise _provider_failure(boundary="failure_before_output")
    try:
        ProviderItemLedger(
            lane_id=plan.binding.lane_id,
            normalization_version=PROVIDER_ITEM_NORMALIZATION_VERSION,
            items=items,
        )
    except ConversationValidationError:
        raise _provider_failure(boundary="failure_before_output") from None
    return ProviderResult(
        items=items,
        reasoning=plan.reasoning,
        usage=_provider_usage(payload),
    )


def _provider_result_mapping(
    payload: Mapping[str, JsonValue],
    plan: NativeOpenAIResponsePlan,
    *,
    limits: NativeOpenAICompactionLimits | None = None,
) -> ProviderResult:
    if (
        payload.get("object") != "response"
        or payload.get("status") != "completed"
    ):
        raise _provider_failure(boundary="failure_before_output")
    raw_output = payload.get("output")
    if type(raw_output) is not tuple:
        raise _provider_failure(boundary="failure_before_output")
    items = _bounded_provider_items(
        raw_output,
        plan=plan,
        limits=limits,
    )
    if len(items) != len(raw_output):
        raise _provider_failure(boundary="failure_before_output")
    reasoning = _effective_reasoning(payload, plan)
    usage = _provider_usage(payload)
    upstream_response_id = None
    if isinstance(plan, FirstStoredProviderPlan | StoredProviderPlan):
        raw_response_id = payload.get("id")
        expected_parent = (
            plan.upstream_response_id
            if type(plan) is StoredProviderPlan
            else None
        )
        if (
            type(raw_response_id) is not str
            or not raw_response_id
            or payload.get("store") is not True
            or payload.get("previous_response_id") != expected_parent
        ):
            raise _provider_failure(boundary="failure_before_output")
        validate_identifier(raw_response_id, "upstream_response_id")
        upstream_response_id = UpstreamResponseId(raw_response_id)
    return ProviderResult(
        items=items,
        reasoning=reasoning,
        usage=usage,
        upstream_response_id=upstream_response_id,
    )


def _provider_item(
    raw: Mapping[str, JsonValue],
    *,
    plan: NativeOpenAIResponsePlan,
    provider_index: int,
) -> ProviderItem:
    raw_type = raw.get("type")
    try:
        if type(raw_type) is not str:
            raise ValueError()
        kind = ProviderItemKind(raw_type)
    except ValueError:
        raise _provider_failure(boundary="failure_before_output") from None
    if kind is ProviderItemKind.COMPACTION and not set(raw) <= {
        "created_by",
        "encrypted_content",
        "id",
        "type",
    }:
        raise _provider_failure(boundary="failure_before_output")
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
    compact_input = (
        type(plan) is StandaloneCompactProviderPlan
        and kind is ProviderItemKind.MESSAGE
        and raw.get("role") != "assistant"
    )
    phase = (
        ProviderItemPhase.INPUT
        if compact_input
        else _provider_phase(kind, raw)
    )
    caller = (
        ProviderItemCaller.CALLER
        if compact_input
        else ProviderItemCaller.PROVIDER
    )
    rules = tuple(
        rule
        for rule in PROVIDER_ITEM_SEMANTICS[kind]
        if phase in rule.phases and caller in rule.callers
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
    order_offset = (
        len(plan.ledger.items)
        if type(plan) is StatelessProviderPlan
        else (
            plan.item_order_offset
            if isinstance(plan, FirstStoredProviderPlan | StoredProviderPlan)
            else 0
        )
    )
    try:
        return ProviderItem(
            item_id=item_id,
            lane_id=plan.binding.lane_id,
            model_call_id=model_call_id,
            kind=kind,
            order=ProviderItemOrder(order_offset + provider_index),
            provider_index=ProviderItemIndex(provider_index),
            phase=phase,
            caller=caller,
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


def _model_call_id(
    plan: NativeOpenAIResponsePlan,
) -> ConversationModelCallId:
    if type(plan) is StandaloneCompactProviderPlan:
        return ConversationModelCallId("native-compact-call-1")
    index = (
        len({item.model_call_id for item in plan.ledger.items}) + 1
        if type(plan) is StatelessProviderPlan
        else plan.model_call_index
    )
    return ConversationModelCallId(f"native-model-call-{index}")


def _effective_reasoning(
    payload: Mapping[str, JsonValue],
    plan: NativeOpenAIResponsePlan,
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
    try:
        parameters = signature(model_dump).parameters
    except (TypeError, ValueError):
        raise _provider_failure(boundary="failure_before_output") from None
    dumped = (
        model_dump(mode="json", exclude_none=True, warnings=False)
        if "warnings" in parameters
        else model_dump(mode="json", exclude_none=True)
    )
    frozen = freeze_json_value(dumped)
    if not isinstance(frozen, Mapping):
        raise _provider_failure(boundary="failure_before_output")
    return frozen


def _validate_sdk_client(
    client: AsyncOpenAI,
    profile: NativeOpenAIStatelessProfile,
    capability_profile: ConversationCapabilityProfile,
) -> None:
    _validate_sdk_client_binding(
        client,
        binding=profile.binding,
        scripted_tcp_test=profile.scripted_tcp_test,
        capability_profile=capability_profile,
    )


def _validate_sdk_client_binding(
    client: AsyncOpenAI,
    *,
    binding: ProviderLaneBinding,
    scripted_tcp_test: bool,
    capability_profile: ConversationCapabilityProfile,
) -> None:
    """Validate one exact native SDK client and endpoint binding."""
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
    if scripted_tcp_test:
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
