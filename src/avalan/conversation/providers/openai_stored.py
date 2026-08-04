"""Adapt exact native OpenAI Responses profiles to stored chaining."""

from ..activation import AZURE_OPENAI_API_REVISIONS, AsyncActivationRegistry
from ..binding import (
    ConversationCapability,
    ConversationCapabilityProfile,
    ProviderFamily,
    ProviderLaneBinding,
    ProviderTransport,
)
from ..contract import (
    ChildLaneRetentionPolicy,
    RequestIdempotencyKey,
    UpstreamResponseId,
)
from ..errors import (
    ConversationAmbiguousDispatchError,
    ConversationBindingDriftError,
    ConversationCapabilityError,
    ConversationCommitError,
    ConversationError,
    ConversationProviderResponseError,
    ConversationValidationError,
)
from ..execution import ProviderToolExecution, ToolExecutionPhase
from ..items import (
    PROVIDER_ITEM_NORMALIZATION_VERSION,
    ProviderItem,
    ProviderItemKind,
)
from ..lifecycle import (
    RetrievedUpstreamResponse,
    UpstreamAvailability,
    UpstreamDeleteDisposition,
    UpstreamDeleteResult,
    UpstreamRetentionMetadata,
)
from ..protocols import (
    ConversationProviderStream,
    FirstStoredProviderPlan,
    ProviderPlan,
    ProviderResult,
    StoredProviderPlan,
)
from ..settings import (
    CompactionOperation,
    ConversationMode,
    DisabledCompaction,
    EffectiveReasoningContext,
    InlineCompaction,
    ReasoningContext,
)
from ..value import (
    IntegrityDigest,
    ProviderItemId,
    canonical_json_bytes,
    freeze_json_value,
    thaw_json_value,
    validate_identifier,
)
from .openai import (
    _ENCRYPTED_CONTENT_INCLUDE,
    _OPENAI_API_REVISION,
    _SDK_REVISION,
    NativeOpenAICompactionLimits,
    NativeOpenAIEncryptedContentPolicy,
    NativeOpenAIFunctionTool,
    NativeOpenAIProviderDiagnostics,
    _append_bounded_input_item,
    _configured_function_tool,
    _configured_tool_execution_metadata,
    _NativeOpenAIDiagnosticsState,
    _NativeOpenAIProviderStream,
    _NativeOpenAITestAuthority,
    _owned_close_outcome,
    _provider_failure,
    _provider_result_mapping,
    _sdk_mapping,
    _validate_sdk_client_binding,
    native_openai_compaction_policy_digest,
)

from asyncio import CancelledError, Task, create_task
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, TypeAlias, final

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    omit,
)
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseIncludable,
    ResponseInputItemParam,
    ResponseStreamEvent,
)
from openai.types.responses.response_create_params import ContextManagement
from openai.types.shared_params import Reasoning

_STORED_ADAPTER_TYPE = (
    "avalan.conversation.providers.openai_stored.NativeOpenAIStoredProvider"
)

NativeOpenAIStoredPlan: TypeAlias = (
    FirstStoredProviderPlan | StoredProviderPlan
)


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NativeOpenAIStoredExecution:
    """Freeze execution fields reapplied to every stored SDK request."""

    instructions: str
    max_output_tokens: int
    max_tool_calls: int
    parallel_tool_calls: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    truncation: Literal["auto", "disabled"] = "disabled"
    safety_identifier: str = "avalan-conversation"

    def __post_init__(self) -> None:
        validate_identifier(
            self.instructions,
            "instructions",
            max_length=1_048_576,
        )
        if (
            type(self.max_output_tokens) is not int
            or self.max_output_tokens <= 0
            or type(self.max_tool_calls) is not int
            or self.max_tool_calls <= 0
            or type(self.parallel_tool_calls) is not bool
            or type(self.temperature) is not float
            or not 0.0 <= self.temperature <= 2.0
            or type(self.top_p) is not float
            or not 0.0 < self.top_p <= 1.0
            or self.truncation not in {"auto", "disabled"}
        ):
            raise ConversationValidationError()
        validate_identifier(
            self.safety_identifier,
            "safety_identifier",
            max_length=64,
        )


def native_openai_stored_execution_digest(
    *,
    binding: ProviderLaneBinding,
    execution: NativeOpenAIStoredExecution,
    encrypted_content: NativeOpenAIEncryptedContentPolicy,
    tools: tuple[NativeOpenAIFunctionTool, ...] = (),
    compaction_limits: NativeOpenAICompactionLimits | None = None,
) -> IntegrityDigest:
    """Return canonical bytes binding every static stored request field."""
    if (
        type(binding) is not ProviderLaneBinding
        or type(execution) is not NativeOpenAIStoredExecution
        or not isinstance(
            encrypted_content,
            NativeOpenAIEncryptedContentPolicy,
        )
        or type(tools) is not tuple
        or any(type(tool) is not NativeOpenAIFunctionTool for tool in tools)
        or (
            compaction_limits is not None
            and type(compaction_limits) is not NativeOpenAICompactionLimits
        )
    ):
        raise ConversationValidationError()
    include = (
        (_ENCRYPTED_CONTENT_INCLUDE,)
        if encrypted_content
        is NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
        else ()
    )
    payload: dict[str, object] = {
        "include": include,
        "instructions": execution.instructions,
        "max_output_tokens": execution.max_output_tokens,
        "max_tool_calls": execution.max_tool_calls,
        "model": binding.model_or_deployment,
        "parallel_tool_calls": execution.parallel_tool_calls,
        "reasoning_context_policy": {
            "all_turns": "all_turns",
            "auto": None,
            "current_turn": "current_turn",
        },
        "safety_identifier": execution.safety_identifier,
        "store": True,
        "stream": binding.transport is ProviderTransport.STREAMING,
        "temperature": execution.temperature,
        "tool_choice": "auto",
        "tools": tuple(tool.schema for tool in tools),
        "top_p": execution.top_p,
        "truncation": execution.truncation,
    }
    if compaction_limits is not None:
        payload["compaction_limits"] = {
            "max_compact_threshold": compaction_limits.max_compact_threshold,
            "max_input_bytes": compaction_limits.max_input_bytes,
            "max_input_depth": compaction_limits.max_input_depth,
            "max_input_items": compaction_limits.max_input_items,
            "max_output_bytes": compaction_limits.max_output_bytes,
            "max_output_depth": compaction_limits.max_output_depth,
            "max_output_items": compaction_limits.max_output_items,
            "min_compact_threshold": compaction_limits.min_compact_threshold,
        }
    canonical_payload = freeze_json_value(payload)
    return IntegrityDigest(
        sha256(canonical_json_bytes(canonical_payload)).hexdigest()
    )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class NativeOpenAIStoredProfile:
    """Bind exact request behavior to one stored Responses profile."""

    profile_id: str
    binding: ProviderLaneBinding
    execution: NativeOpenAIStoredExecution
    encrypted_content: NativeOpenAIEncryptedContentPolicy
    compaction_limits: NativeOpenAICompactionLimits | None = None
    scripted_tcp_test: bool = False

    def __post_init__(self) -> None:
        validate_identifier(self.profile_id, "profile_id")
        if (
            type(self.binding) is not ProviderLaneBinding
            or type(self.execution) is not NativeOpenAIStoredExecution
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
            binding.adapter_type != _STORED_ADAPTER_TYPE
            or binding.provider_family
            not in {ProviderFamily.OPENAI, ProviderFamily.AZURE_OPENAI}
            or binding.sdk_revision != _SDK_REVISION
            or binding.continuation_codec_version
            != PROVIDER_ITEM_NORMALIZATION_VERSION
        ):
            raise ConversationCapabilityError()
        if binding.execution_definition_digest is None:
            raise ConversationBindingDriftError()
        if binding.provider_family is ProviderFamily.OPENAI:
            if (
                binding.provider_api_revision != _OPENAI_API_REVISION
                or self.encrypted_content
                is not NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ):
                raise ConversationCapabilityError()
        elif (
            binding.provider_api_revision not in AZURE_OPENAI_API_REVISIONS
            or self.encrypted_content
            is not NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
        ):
            raise ConversationCapabilityError()
        if binding.compaction_policy_digest != (
            native_openai_compaction_policy_digest(self.compaction_limits)
        ):
            raise ConversationBindingDriftError()


@final
class NativeOpenAIStoredProvider:
    """Dispatch exact native provider-stored Responses calls."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        profile: NativeOpenAIStoredProfile,
        capability_profile: ConversationCapabilityProfile,
        tools: tuple[NativeOpenAIFunctionTool, ...] = (),
        activation_registry: AsyncActivationRegistry | None = None,
        test_authority: _NativeOpenAITestAuthority | None = None,
    ) -> None:
        if (
            type(client) is not AsyncOpenAI
            or type(profile) is not NativeOpenAIStoredProfile
            or type(capability_profile) is not ConversationCapabilityProfile
            or type(tools) is not tuple
            or (
                activation_registry is not None
                and type(activation_registry) is not AsyncActivationRegistry
            )
            or (
                test_authority is not None
                and type(test_authority) is not _NativeOpenAITestAuthority
            )
            or (activation_registry is not None and test_authority is not None)
            or any(
                type(tool) is not NativeOpenAIFunctionTool for tool in tools
            )
        ):
            raise ConversationValidationError()
        profile.binding.assert_compatible(profile.binding)
        capability_profile.assert_binding(profile.binding)
        if len({tool.name for tool in tools}) != len(tools):
            raise ConversationValidationError()
        expected_execution_digest = native_openai_stored_execution_digest(
            binding=profile.binding,
            execution=profile.execution,
            encrypted_content=profile.encrypted_content,
            tools=tools,
            compaction_limits=profile.compaction_limits,
        )
        if (
            profile.binding.execution_definition_digest
            != expected_execution_digest
        ):
            raise ConversationBindingDriftError()
        _validate_sdk_client_binding(
            client,
            binding=profile.binding,
            scripted_tcp_test=profile.scripted_tcp_test,
            capability_profile=capability_profile,
        )
        if test_authority is not None:
            test_authority.assert_bound(
                client=client,
                binding=profile.binding,
                scripted_tcp_test=profile.scripted_tcp_test,
                capability_profile=capability_profile,
            )
        self._client = client
        self._profile = profile
        self._capability_profile = capability_profile
        self._tools = {tool.name: tool for tool in tools}
        self._activation_registry = activation_registry
        self._test_authority = test_authority
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
        """Dispatch one exact non-streaming stored request."""
        inline = (
            isinstance(plan, FirstStoredProviderPlan | StoredProviderPlan)
            and type(plan.compaction) is InlineCompaction
        )
        try:
            stored = self._validate_plan(
                plan,
                ProviderTransport.NON_STREAMING,
            )
            await self._authorize_dispatch(
                stored,
                operation=(
                    CompactionOperation.INLINE
                    if inline
                    else CompactionOperation.NONE
                ),
            )
            limits = self._profile.compaction_limits if inline else None
            input_items = _stored_request_input_items(stored, limits=limits)
            response = await self._create(input_items, stored, stream=False)
            if not isinstance(response, Response):
                raise _provider_failure(boundary="failure_before_output")
            result = _provider_result_mapping(
                _sdk_mapping(response),
                stored,
                limits=limits,
            )
        except CancelledError:
            if inline:
                self._diagnostics.compaction_failure_count += 1
            raise
        except ConversationError as error:
            self._diagnostics.failure_boundary = error.boundary.value
            if inline:
                self._diagnostics.compaction_failure_count += 1
            raise
        self._record_response(result)
        return result

    async def stream(self, plan: ProviderPlan) -> ConversationProviderStream:
        """Open one exact streaming stored request."""
        inline = (
            isinstance(plan, FirstStoredProviderPlan | StoredProviderPlan)
            and type(plan.compaction) is InlineCompaction
        )
        try:
            stored = self._validate_plan(plan, ProviderTransport.STREAMING)
            await self._authorize_dispatch(
                stored,
                operation=(
                    CompactionOperation.INLINE
                    if inline
                    else CompactionOperation.NONE
                ),
            )
            limits = self._profile.compaction_limits if inline else None
            input_items = _stored_request_input_items(stored, limits=limits)
            response = await self._create(input_items, stored, stream=True)
            if type(response) is not AsyncStream:
                raise _provider_failure(boundary="failure_before_output")
        except CancelledError:
            if inline:
                self._diagnostics.compaction_failure_count += 1
            raise
        except ConversationError as error:
            self._diagnostics.failure_boundary = error.boundary.value
            if inline:
                self._diagnostics.compaction_failure_count += 1
            raise
        return _NativeOpenAIProviderStream(
            source=response,
            plan=stored,
            owner=self,
            compaction_limits=limits,
        )

    async def execute_tool(self, item: ProviderItem) -> str:
        """Execute one exact configured function call asynchronously."""
        tool, arguments = _configured_function_tool(
            self._tools,
            self.binding,
            item,
        )
        return await tool.execute(arguments)

    def tool_execution_metadata(
        self,
        item: ProviderItem,
        *,
        request_idempotency_key: RequestIdempotencyKey,
        phase: ToolExecutionPhase,
        output_id: ProviderItemId | None = None,
    ) -> ProviderToolExecution:
        """Return exact durable metadata for one configured tool call."""
        return _configured_tool_execution_metadata(
            self._tools,
            self.binding,
            item,
            request_idempotency_key=request_idempotency_key,
            phase=phase,
            output_id=output_id,
        )

    async def retrieve(
        self,
        upstream_response_id: UpstreamResponseId,
    ) -> RetrievedUpstreamResponse:
        """Retrieve one private stored response without exposing its body."""
        await self._validate_lifecycle(
            upstream_response_id,
            ConversationCapability.STORED_RESPONSE_RETRIEVAL,
        )
        try:
            response = await self._client.responses.retrieve(
                str(upstream_response_id)
            )
        except CancelledError:
            raise
        except APIStatusError as error:
            if error.status_code == 404:
                return RetrievedUpstreamResponse(
                    upstream_response_id=upstream_response_id,
                    availability=UpstreamAvailability.UNKNOWN_UNAVAILABLE,
                    retention=UpstreamRetentionMetadata.unknown(),
                    binding_digest=self.binding.integrity_digest,
                    execution_definition_digest=(
                        self.binding.execution_definition_digest
                    ),
                )
            raise ConversationProviderResponseError() from None
        except (APIConnectionError, APITimeoutError):
            raise ConversationAmbiguousDispatchError() from None
        except Exception:
            raise ConversationProviderResponseError() from None
        if not isinstance(response, Response):
            raise ConversationProviderResponseError()
        payload = _sdk_mapping(response)
        effective_reasoning_context = self._validate_retrieved_execution(
            payload,
            upstream_response_id,
        )
        return RetrievedUpstreamResponse(
            upstream_response_id=upstream_response_id,
            availability=UpstreamAvailability.AVAILABLE,
            retention=UpstreamRetentionMetadata.unknown(),
            binding_digest=self.binding.integrity_digest,
            execution_definition_digest=(
                self.binding.execution_definition_digest
            ),
            effective_reasoning_context=effective_reasoning_context,
        )

    def _validate_retrieved_execution(
        self,
        payload: Mapping[str, object],
        upstream_response_id: UpstreamResponseId,
    ) -> EffectiveReasoningContext:
        """Validate every returned field in the durable execution profile."""
        execution = self._profile.execution
        expected = {
            "id": str(upstream_response_id),
            "instructions": execution.instructions,
            "max_output_tokens": execution.max_output_tokens,
            "max_tool_calls": execution.max_tool_calls,
            "model": self.binding.model_or_deployment,
            "object": "response",
            "parallel_tool_calls": execution.parallel_tool_calls,
            "safety_identifier": execution.safety_identifier,
            "status": "completed",
            "store": True,
            "temperature": execution.temperature,
            "tool_choice": "auto",
            "top_p": execution.top_p,
            "truncation": execution.truncation,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ConversationProviderResponseError()
        expected_tools = freeze_json_value(
            tuple(tool.schema for tool in self._tools.values())
        )
        if payload.get("tools") != expected_tools:
            raise ConversationProviderResponseError()
        expected_include = (
            (_ENCRYPTED_CONTENT_INCLUDE,)
            if self._profile.encrypted_content
            is NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
            else ()
        )
        if "include" in payload and payload.get("include") != expected_include:
            raise ConversationProviderResponseError()
        expected_stream = self.binding.transport is ProviderTransport.STREAMING
        if (
            "stream" in payload
            and payload.get("stream") is not expected_stream
        ):
            raise ConversationProviderResponseError()
        reasoning = payload.get("reasoning")
        if not isinstance(reasoning, Mapping):
            raise ConversationProviderResponseError()
        context = reasoning.get("context")
        try:
            if type(context) is not str:
                raise ValueError()
            return EffectiveReasoningContext(context)
        except ValueError:
            raise ConversationProviderResponseError() from None

    async def delete(
        self,
        upstream_response_id: UpstreamResponseId,
    ) -> UpstreamDeleteResult:
        """Delete one private stored response idempotently."""
        await self._validate_lifecycle(
            upstream_response_id,
            ConversationCapability.STORED_RESPONSE_DELETION,
        )
        try:
            await self._client.responses.delete(str(upstream_response_id))
        except CancelledError:
            raise
        except APIStatusError as error:
            if error.status_code == 404:
                return UpstreamDeleteResult(
                    disposition=UpstreamDeleteDisposition.ALREADY_ABSENT
                )
            raise ConversationProviderResponseError() from None
        except (APIConnectionError, APITimeoutError):
            raise ConversationAmbiguousDispatchError() from None
        except Exception:
            raise ConversationProviderResponseError() from None
        return UpstreamDeleteResult(
            disposition=UpstreamDeleteDisposition.DELETED
        )

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
    ) -> NativeOpenAIStoredPlan:
        if self._closed or not isinstance(
            plan,
            FirstStoredProviderPlan | StoredProviderPlan,
        ):
            raise ConversationCapabilityError()
        binding = self.binding
        plan.binding.assert_compatible(binding)
        self._capability_profile.assert_binding(binding)
        if binding.transport is not transport:
            raise ConversationBindingDriftError()
        self._capability_profile.require(
            ConversationCapability.STORED_RESPONSES_CHAINING
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
            self._capability_profile.require(
                ConversationCapability.INLINE_COMPACTION
            )
            limits = self._profile.compaction_limits
            if (
                limits is None
                or not limits.min_compact_threshold
                <= plan.compaction.compact_threshold
                <= limits.max_compact_threshold
            ):
                raise ConversationCapabilityError()
        elif type(plan.compaction) is not DisabledCompaction:
            raise ConversationCapabilityError()
        _validate_sdk_client_binding(
            self._client,
            binding=binding,
            scripted_tcp_test=self._profile.scripted_tcp_test,
            capability_profile=self._capability_profile,
        )
        return plan

    def validate_compaction_request(
        self,
        plan: ProviderPlan,
        transport: ProviderTransport,
    ) -> None:
        """Validate exact inline input without dispatching provider work."""
        stored = self._validate_plan(plan, transport)
        if type(stored.compaction) is not InlineCompaction:
            raise ConversationValidationError()
        limits = self._profile.compaction_limits
        assert limits is not None
        _stored_request_input_items(stored, limits=limits)

    async def _validate_lifecycle(
        self,
        upstream_response_id: UpstreamResponseId,
        capability: ConversationCapability,
    ) -> None:
        if self._closed:
            raise ConversationCapabilityError()
        validate_identifier(upstream_response_id, "upstream_response_id")
        self._capability_profile.assert_binding(self.binding)
        self._capability_profile.require(capability)
        _validate_sdk_client_binding(
            self._client,
            binding=self.binding,
            scripted_tcp_test=self._profile.scripted_tcp_test,
            capability_profile=self._capability_profile,
        )
        test_authority = self._test_authority
        if test_authority is not None:
            test_authority.assert_bound(
                client=self._client,
                binding=self.binding,
                scripted_tcp_test=self._profile.scripted_tcp_test,
                capability_profile=self._capability_profile,
            )
            return
        registry = self._activation_registry
        if registry is None:
            raise ConversationCapabilityError()
        await registry.resolve_lifecycle(
            self.binding,
            capability=capability,
        )

    async def _authorize_dispatch(
        self,
        plan: NativeOpenAIStoredPlan,
        *,
        operation: CompactionOperation,
    ) -> None:
        """Authorize one exact production row before any SDK effect."""
        test_authority = self._test_authority
        if test_authority is not None:
            test_authority.assert_bound(
                client=self._client,
                binding=self.binding,
                scripted_tcp_test=self._profile.scripted_tcp_test,
                capability_profile=self._capability_profile,
            )
            return
        registry = self._activation_registry
        if registry is None:
            raise ConversationCapabilityError()
        await registry.resolve(
            self.binding,
            mode=ConversationMode.STORED,
            reasoning_context=plan.reasoning.requested,
            compaction_operation=operation,
        )

    async def _create(
        self,
        input_items: list[ResponseInputItemParam],
        plan: NativeOpenAIStoredPlan,
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
        previous_response_id = (
            str(plan.upstream_response_id)
            if type(plan) is StoredProviderPlan
            else None
        )
        self._record_request(input_items)
        try:
            return await _create_exact_stored_response(
                self._client,
                model=self.binding.model_or_deployment,
                input_items=input_items,
                include=include,
                reasoning=reasoning,
                context_management=context_management,
                tools=tools,
                previous_response_id=previous_response_id,
                execution=self._profile.execution,
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

    def _record_request(self, items: list[ResponseInputItemParam]) -> None:
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
            len(canonical_json_bytes(item.canonical_input))
            for item in result.items
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
class NativeOpenAIStoredLaneRuntime:
    """Bind one closed stored provider to coordinator lane authority."""

    provider: NativeOpenAIStoredProvider
    retention_policy: ChildLaneRetentionPolicy = (
        ChildLaneRetentionPolicy.RETAIN
    )
    max_output_items: int = 1_024
    max_output_bytes: int = 8_388_608
    max_output_segments: int = 1_024

    def __post_init__(self) -> None:
        if (
            type(self.provider) is not NativeOpenAIStoredProvider
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


async def _create_exact_stored_response(
    client: AsyncOpenAI,
    *,
    model: str,
    input_items: list[ResponseInputItemParam],
    include: list[ResponseIncludable] | None,
    reasoning: Reasoning | None,
    context_management: list[ContextManagement] | None,
    tools: list[FunctionToolParam],
    previous_response_id: str | None,
    execution: NativeOpenAIStoredExecution,
    stream: bool,
) -> Response | AsyncStream[ResponseStreamEvent]:
    """Create one stored response with every frozen execution field."""
    previous = (
        previous_response_id if previous_response_id is not None else omit
    )
    if stream:
        return await client.responses.create(
            model=model,
            input=input_items,
            context_management=(
                context_management if context_management is not None else omit
            ),
            include=include if include is not None else omit,
            instructions=execution.instructions,
            max_output_tokens=execution.max_output_tokens,
            max_tool_calls=execution.max_tool_calls,
            parallel_tool_calls=execution.parallel_tool_calls,
            previous_response_id=previous,
            reasoning=reasoning if reasoning is not None else omit,
            store=True,
            stream=True,
            temperature=execution.temperature,
            tool_choice="auto",
            tools=tools,
            top_p=execution.top_p,
            truncation=execution.truncation,
            safety_identifier=execution.safety_identifier,
        )
    return await client.responses.create(
        model=model,
        input=input_items,
        context_management=(
            context_management if context_management is not None else omit
        ),
        include=include if include is not None else omit,
        instructions=execution.instructions,
        max_output_tokens=execution.max_output_tokens,
        max_tool_calls=execution.max_tool_calls,
        parallel_tool_calls=execution.parallel_tool_calls,
        previous_response_id=previous,
        reasoning=reasoning if reasoning is not None else omit,
        store=True,
        stream=False,
        temperature=execution.temperature,
        tool_choice="auto",
        tools=tools,
        top_p=execution.top_p,
        truncation=execution.truncation,
        safety_identifier=execution.safety_identifier,
    )


def _stored_request_input_items(
    plan: NativeOpenAIStoredPlan,
    *,
    limits: NativeOpenAICompactionLimits | None = None,
) -> list[ResponseInputItemParam]:
    new_input = plan.new_input
    if new_input is None:
        raise ConversationValidationError()
    if set(new_input) == {"text"}:
        text = new_input.get("text")
        if type(text) is not str or not text:
            raise ConversationValidationError()
        text_items: list[ResponseInputItemParam] = []
        _append_bounded_input_item(
            text_items,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
            limits,
            0,
        )
        return text_items
    if set(new_input) != {"items"}:
        raise ConversationValidationError()
    raw_items = new_input.get("items")
    if type(raw_items) is not tuple or not raw_items:
        raise ConversationValidationError()
    items: list[ResponseInputItemParam] = []
    byte_count = 0
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise ConversationValidationError()
        value = thaw_json_value(raw_item)
        if (
            type(value) is not dict
            or set(value) != {"type", "call_id", "output"}
            or value.get("type") != "function_call_output"
            or type(value.get("call_id")) is not str
            or type(value.get("output")) is not str
        ):
            raise ConversationValidationError()
        byte_count = _append_bounded_input_item(
            items,
            value,
            limits,
            byte_count,
        )
    return items
