"""Exercise closed native Responses validation and privacy boundaries."""

from asyncio import CancelledError, Event, create_task
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
)
from dataclasses import replace
from typing import cast

import httpx
import pytest
from native_openai_provider_test import (
    _binding,
    _capabilities,
    _direct_client,
    _function_call,
    _message,
    _plan,
    _profile,
    _provider,
    _reasoning,
    _response,
)
from openai import APIResponseValidationError, AsyncOpenAI, AsyncStream
from openai.types.responses import ResponseStreamEvent
from phase2_fixtures import authority, request, root_identity

import avalan
import avalan.conversation as conversation
from avalan.conversation import coordinator as coordinator_module
from avalan.conversation import items as items_module
from avalan.conversation import protocols as protocols_module
from avalan.conversation import sdk as sdk_module
from avalan.conversation.providers import openai as provider_module
from avalan.model.nlp.text.vendor import openai as legacy_openai_module
from avalan.model.stream import StreamRetentionPolicy
from avalan.types import JsonValue

pytestmark = pytest.mark.anyio


class _SdkEvent:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def model_dump(
        self,
        *,
        mode: str,
        exclude_none: bool,
    ) -> object:
        assert mode == "json"
        assert exclude_none
        return self._payload


class _ScriptedSdkStream(AsyncIterator[object]):
    def __init__(
        self,
        steps: tuple[object | BaseException, ...],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self._steps = list(steps)
        self._close_error = close_error
        self.close_count = 0

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        if not self._steps:
            raise StopAsyncIteration
        step = self._steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    async def close(self) -> None:
        self.close_count += 1
        error = self._close_error
        self._close_error = None
        if error is not None:
            raise error


class _CloseSequenceStream:
    def __init__(self, steps: tuple[BaseException | None, ...]) -> None:
        self._steps = list(steps)
        self.close_count = 0

    async def aclose(self) -> None:
        self.close_count += 1
        step = self._steps.pop(0) if self._steps else None
        if step is not None:
            raise step


class _DirectProviderStream(AsyncIterator[conversation.ProviderItem]):
    def __init__(
        self,
        items: tuple[conversation.ProviderItem, ...],
        result: conversation.ProviderResult,
    ) -> None:
        self._items = list(items)
        self._result = result
        self.close_count = 0

    def __aiter__(self) -> AsyncIterator[conversation.ProviderItem]:
        return self

    async def __anext__(self) -> conversation.ProviderItem:
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    async def terminal(self) -> conversation.ProviderResult:
        return self._result

    async def aclose(self) -> None:
        self.close_count += 1


class _RecordingSink:
    def __init__(self) -> None:
        self.staged_items: list[conversation.ProviderItem] = []
        self.finalize_calls = 0
        self.cleanup_calls = 0

    async def stage(self, item: conversation.ProviderItem) -> None:
        self.staged_items.append(item)

    async def finalize(
        self,
        outputs: tuple[conversation.ProviderLaneOutputCandidate, ...],
    ) -> None:
        del outputs
        self.finalize_calls += 1

    async def cleanup(self) -> None:
        self.cleanup_calls += 1


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic provider validation on asyncio only."""
    return "asyncio"


async def _unused_handler(request: httpx.Request) -> httpx.Response:
    await request.aread()
    return httpx.Response(
        200,
        json=_response(
            "unused-response",
            [
                _reasoning("unused-reasoning", "unused-private"),
                _message("unused-message", "unused"),
            ],
        ),
    )


def _frozen_mapping(value: object) -> Mapping[str, JsonValue]:
    frozen = conversation.freeze_json_value(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _provider_result(
    payload: dict[str, object],
    binding: conversation.ProviderLaneBinding | None = None,
) -> conversation.ProviderResult:
    selected = binding or _binding(lane_id="lane-result-validation")
    return provider_module._provider_result_mapping(
        _frozen_mapping(payload),
        _plan(selected),
    )


def _capability_profile(
    binding: conversation.ProviderLaneBinding,
    *,
    unsupported: conversation.ConversationCapability | None = None,
    test_only: bool = True,
) -> conversation.ConversationCapabilityProfile:
    original = _capabilities(binding)
    return conversation.ConversationCapabilityProfile(
        profile_id=original.profile_id,
        schema_version=original.schema_version,
        revision=original.revision,
        binding_alias=original.binding_alias,
        capabilities=tuple(
            conversation.CapabilityEvidence(
                capability=item.capability,
                state=(
                    conversation.CapabilityEvidenceState.INCAPABLE
                    if item.capability is unsupported
                    else (
                        conversation.CapabilityEvidenceState.CONFORMANT
                        if not test_only
                        and item.state
                        is conversation.CapabilityEvidenceState.TEST_ONLY
                        else item.state
                    )
                ),
                evidence_ids=(
                    () if item.capability is unsupported else item.evidence_ids
                ),
            )
            for item in original.capabilities
        ),
        test_only=test_only,
    )


def _raw_provider(
    binding: conversation.ProviderLaneBinding,
    capability_profile: conversation.ConversationCapabilityProfile,
    *,
    profile: conversation.NativeOpenAIStatelessProfile | None = None,
    default_query: Mapping[str, object] | None = None,
    handler: Callable[
        [httpx.Request], Coroutine[None, None, httpx.Response]
    ] = _unused_handler,
) -> conversation.NativeOpenAIStatelessProvider:
    client = AsyncOpenAI(
        api_key="validation-key",
        base_url=binding.normalized_endpoint,
        default_query=default_query,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    return conversation.NativeOpenAIStatelessProvider(
        client=client,
        profile=profile or _profile(binding),
        capability_profile=capability_profile,
    )


def _scripted_stream(
    owner: conversation.NativeOpenAIStatelessProvider,
    steps: tuple[object | BaseException, ...],
    *,
    close_error: BaseException | None = None,
) -> tuple[
    provider_module._NativeOpenAIProviderStream,
    _ScriptedSdkStream,
]:
    source = _ScriptedSdkStream(steps, close_error=close_error)
    stream = provider_module._NativeOpenAIProviderStream(
        source=cast(
            AsyncStream[ResponseStreamEvent],
            source,
        ),
        plan=_plan(owner.binding),
        owner=owner,
    )
    return stream, source


def _done_event(
    index: int,
    item: dict[str, object],
    *,
    sequence: object = 0,
) -> _SdkEvent:
    return _SdkEvent(
        {
            "type": "response.output_item.done",
            "sequence_number": sequence,
            "output_index": index,
            "item": item,
        }
    )


def _terminal_event(
    output: list[object],
    *,
    sequence: object = 1,
) -> _SdkEvent:
    return _SdkEvent(
        {
            "type": "response.completed",
            "sequence_number": sequence,
            "response": _response(
                "stream-validation-response",
                cast(list[dict[str, object]], output),
            ),
        }
    )


async def test_profiles_tools_and_runtime_fail_closed() -> None:
    """Reject malformed profiles, tools, and runtime values eagerly."""
    binding = _binding(lane_id="lane-constructor-validation")
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIStatelessProfile(
            profile_id="profile-invalid-policy",
            binding=binding,
            encrypted_content=cast(
                conversation.NativeOpenAIEncryptedContentPolicy,
                object(),
            ),
        )
    with pytest.raises(conversation.ConversationCapabilityError):
        _profile(replace(binding, adapter_type="wrong.Adapter"))
    with pytest.raises(conversation.ConversationCapabilityError):
        conversation.NativeOpenAIStatelessProfile(
            profile_id="profile-openai-include",
            binding=binding,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
            ),
        )
    azure = _binding(azure=True, lane_id="lane-azure-policy")
    with pytest.raises(conversation.ConversationCapabilityError):
        conversation.NativeOpenAIStatelessProfile(
            profile_id="profile-azure-default",
            binding=azure,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        )

    async def valid(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments
        return "valid"

    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIFunctionTool(
            name="invalid-handler",
            parameters={"type": "object"},
            handler=cast(
                Callable[[Mapping[str, JsonValue]], Awaitable[str]],
                lambda arguments: str(arguments),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIFunctionTool(
            name="invalid-parameters",
            parameters=cast(Mapping[str, JsonValue], []),
            handler=valid,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIFunctionTool(
            name="invalid-schema",
            parameters={"type": "array"},
            handler=valid,
        )

    provider = _provider(binding, _unused_handler)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIConversationLaneRuntime(
            provider=cast(
                conversation.NativeOpenAIStatelessProvider,
                object(),
            )
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIConversationLaneRuntime(
            provider=provider,
            max_output_items=0,
        )
    for limits in (
        {"max_output_bytes": 0},
        {"max_output_bytes": cast(int, True)},
        {"max_output_segments": 0},
        {"max_output_segments": cast(int, True)},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.NativeOpenAIConversationLaneRuntime(
                provider=provider,
                **limits,
            )
    await provider.aclose()


def test_new_input_freezing_and_legacy_facade_is_stateless(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject bad freezes and keep the compatibility facade stateless."""
    record_property("conversation_acceptance_evidence", "contract")
    monkeypatch.setattr(
        protocols_module,
        "freeze_json_value",
        lambda value: () if value else value,
    )
    with pytest.raises(conversation.ConversationValidationError):
        _plan(_binding(lane_id="lane-invalid-frozen-input"))

    facade = legacy_openai_module._OpenAIDirectReplayCompatibilityFacade()
    assert not hasattr(facade, "__dict__")
    for state_name in (
        "policy",
        "_items",
        "_ledger",
        "_replay_item_count",
        "_reasoning_item_count",
        "_attempt_checkpoint",
        "_release_count",
    ):
        assert not hasattr(facade, state_name)
    with pytest.raises(AttributeError):
        setattr(facade, "policy", StreamRetentionPolicy())
    owner = facade.create_execution_state(StreamRetentionPolicy())
    assert type(owner) is (
        legacy_openai_module._OpenAIDirectReplayExecutionState
    )
    assert owner.owns_conversation_state is False
    assert legacy_openai_module._OpenAIReplayOwner is type(owner)


async def test_native_coordinator_runtime_validation_is_closed() -> None:
    """Reject malformed and unsupported exact native lane runtimes."""
    binding = _binding(lane_id="lane-runtime-validation")
    provider = _provider(binding, _unused_handler)
    runtime = conversation.NativeOpenAIConversationLaneRuntime(
        provider=provider
    )
    assert coordinator_module._validate_native_lane_runtime(runtime) is runtime

    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_native_lane_runtime(object())
    missing = object.__new__(conversation.NativeOpenAIConversationLaneRuntime)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_native_lane_runtime(missing)
    object.__setattr__(runtime, "max_output_items", 0)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_native_lane_runtime(runtime)
    object.__setattr__(runtime, "max_output_items", 1_024)
    object.__setattr__(runtime, "max_output_bytes", 0)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_native_lane_runtime(runtime)
    object.__setattr__(runtime, "max_output_bytes", 8_388_608)
    object.__setattr__(runtime, "max_output_segments", 0)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_native_lane_runtime(runtime)
    object.__setattr__(runtime, "max_output_segments", 1_024)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_any_lane_runtime(object())

    synthetic = replace(
        binding,
        provider_family=conversation.ProviderFamily.SYNTHETIC,
    )
    object.__setattr__(provider._profile, "binding", synthetic)
    object.__setattr__(
        provider,
        "_capability_profile",
        replace(
            provider.capability_profile,
            binding_alias=synthetic.safe_alias,
        ),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        coordinator_module._validate_native_lane_runtime(runtime)
    await provider.aclose()


async def test_native_coordinator_diagnostics_and_close_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deduplicate closes and preserve cancellation-safe failures."""
    provider = _provider(
        _binding(lane_id="lane-native-close-deduplicated"),
        _unused_handler,
    )
    _, coordinator, _ = _direct_client(
        provider,
        namespace="native-close-deduplicated",
    )
    assert (
        coordinator.native_provider_diagnostics(provider.binding.lane_id)
        == provider.diagnostics
    )
    with pytest.raises(conversation.ConversationValidationError):
        coordinator.native_provider_diagnostics(
            conversation.ProviderLaneId("missing-native-lane")
        )

    close_count = 0
    original_close = provider.aclose

    async def counted_close() -> None:
        nonlocal close_count
        close_count += 1
        await original_close()

    runtime = next(iter(coordinator._lanes.values()))
    coordinator._lanes[
        conversation.ProviderLaneId("duplicate-native-runtime")
    ] = runtime
    with monkeypatch.context() as patch:
        patch.setattr(provider, "aclose", counted_close)
        await coordinator._close_native_providers()
    assert close_count == 1
    coordinator._lanes.pop(
        conversation.ProviderLaneId("duplicate-native-runtime")
    )
    await coordinator.close()

    cancelled_provider = _provider(
        _binding(lane_id="lane-native-close-cancelled"),
        _unused_handler,
    )
    _, cancelled_coordinator, _ = _direct_client(
        cancelled_provider,
        namespace="native-close-cancelled",
    )

    async def cancelled_close() -> None:
        raise CancelledError()

    with monkeypatch.context() as patch:
        patch.setattr(cancelled_provider, "aclose", cancelled_close)
        with pytest.raises(CancelledError):
            await cancelled_coordinator._close_native_providers()
    await cancelled_coordinator.close()

    failed_provider = _provider(
        _binding(lane_id="lane-native-close-failed"),
        _unused_handler,
    )
    _, failed_coordinator, _ = _direct_client(
        failed_provider,
        namespace="native-close-failed",
    )

    async def failed_close() -> None:
        raise RuntimeError("private-close-failure")

    with monkeypatch.context() as patch:
        patch.setattr(failed_provider, "aclose", failed_close)
        with pytest.raises(conversation.ConversationCommitError) as failure:
            await failed_coordinator._close_native_providers()
        assert failure.value.__cause__ is None
        assert "private-close-failure" not in repr(failure.value)
    await failed_coordinator.close()


@pytest.mark.parametrize("cleanup", ("cancelled", "failed", "settled"))
async def test_native_coordinator_close_settles_provider_cancellation(
    cleanup: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry one cancelled provider close before preserving cancellation."""
    provider = _provider(
        _binding(lane_id=f"lane-native-close-{cleanup}"),
        _unused_handler,
    )
    _, coordinator, _ = _direct_client(
        provider,
        namespace=f"native-close-{cleanup}",
    )
    close_count = 0

    async def sequenced_close() -> None:
        nonlocal close_count
        close_count += 1
        if close_count == 1 or cleanup == "cancelled":
            raise CancelledError()
        if cleanup == "failed":
            raise RuntimeError("private-provider-cleanup")

    with monkeypatch.context() as patch:
        patch.setattr(provider, "aclose", sequenced_close)
        with pytest.raises(CancelledError) as cancellation:
            await coordinator.close()
    assert close_count == 2
    if cleanup == "settled":
        assert cancellation.value.__cause__ is None
        assert coordinator.diagnostics.closed
        await provider.aclose()
    else:
        assert isinstance(
            cancellation.value.__cause__,
            conversation.ConversationCommitError,
        )
        assert "private-provider-cleanup" not in repr(cancellation.value)
        await coordinator.close()


@pytest.mark.parametrize("store_failure", ("action", "probe", "none"))
async def test_native_coordinator_close_composes_provider_store_failures(
    store_failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve provider-close precedence over each store-close failure."""
    provider = _provider(
        _binding(lane_id=f"lane-native-store-close-{store_failure}"),
        _unused_handler,
    )
    _, coordinator, store = _direct_client(
        provider,
        namespace=f"native-store-close-{store_failure}",
    )

    async def provider_failure() -> None:
        raise RuntimeError("private-provider-close")

    async def action_failure() -> conversation.StoreCloseResolution:
        raise RuntimeError("private-store-action")

    async def probe_failure() -> conversation.StoreCloseResolution:
        raise RuntimeError("private-store-probe")

    with monkeypatch.context() as patch:
        patch.setattr(provider, "aclose", provider_failure)
        if store_failure == "action":
            patch.setattr(store, "close", action_failure)
        elif store_failure == "probe":
            patch.setattr(store, "inspect_close", probe_failure)
        with pytest.raises(conversation.ConversationCommitError) as failure:
            await coordinator.close()
    assert "private-provider-close" not in repr(failure.value)
    if store_failure == "none":
        assert failure.value.__cause__ is None
    else:
        assert isinstance(failure.value.__cause__, RuntimeError)
    await coordinator.close()


async def test_native_coordinator_plan_boundaries_are_closed() -> None:
    """Reject semantic, transport, mode, and tool-cycle boundary drift."""
    scope = authority()
    binding = _binding(lane_id="lane-native-plan-boundaries")
    provider = _provider(binding, _unused_handler)
    _, coordinator, _ = _direct_client(
        provider,
        namespace="native-plan-boundaries",
    )
    runtime = next(iter(coordinator._lanes.values()))
    run_request = request(
        scope=scope,
        identity=root_identity("native-plan-boundaries"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key="native-plan-boundaries",
        response_suffix="native-plan-boundaries",
    )
    malformed_semantics = replace(
        run_request.semantics,
        semantic_input="not-a-mapping",
    )
    with pytest.raises(conversation.ConversationValidationError):
        coordinator._plan_lanes(
            replace(run_request, semantics=malformed_semantics),
            None,
            streaming=False,
        )
    with pytest.raises(conversation.ConversationBindingDriftError):
        coordinator._require_capabilities(
            run_request.lanes[0],
            runtime,
            streaming=True,
            standalone_compaction=False,
        )
    stored = conversation.StoredProviderPlan(
        binding=binding,
        upstream_response_id=conversation.UpstreamResponseId(
            "native-stored-plan"
        ),
        reasoning=_plan(binding).reasoning,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await coordinator._dispatch_complete_lane(
            runtime,
            stored,
            streaming=False,
            progress=coordinator_module._DispatchProgress(),
            sink=None,
        )
    await coordinator.close()


async def test_native_output_item_limit_precedes_tool_effect_and_commit(
    record_property: Callable[[str, object], None],
) -> None:
    """Reserve a generated tool output before crossing its item limit."""
    record_property("conversation_acceptance_evidence", "negative")
    scope = authority()
    tool_effects = 0
    dispatches = 0

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        assert arguments == {"value": 1}
        return "bounded"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )

    async def response_handler(request_value: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request_value.aread()
        return httpx.Response(
            200,
            json=_response(
                "bounded-tool-response",
                [
                    _reasoning("bounded-reasoning", "bounded-private"),
                    _function_call("bounded-call", "bounded-call-id"),
                ],
            ),
        )

    bounded_binding = _binding(lane_id="lane-native-cycle-boundary")
    bounded_provider = _provider(
        bounded_binding,
        response_handler,
        tools=(tool,),
    )
    _, bounded_coordinator, bounded_store = _direct_client(
        bounded_provider,
        namespace="native-cycle-boundary",
    )
    bounded_runtime = next(iter(bounded_coordinator._lanes.values()))
    object.__setattr__(bounded_runtime, "max_output_items", 2)
    bounded_request = request(
        scope=scope,
        identity=root_identity("native-cycle-boundary"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(bounded_binding.lane_id),),
        key="native-cycle-boundary",
        response_suffix="native-cycle-boundary",
    )
    with pytest.raises(conversation.ConversationLimitError):
        await bounded_coordinator.execute(bounded_request)
    assert dispatches == 1
    assert tool_effects == 0
    with pytest.raises(conversation.ConversationAuthorizationError):
        await bounded_store.load(
            bounded_request.identity.checkpoint_id,
            scope,
        )
    await bounded_coordinator.close()


async def test_native_provider_items_reject_before_tool_effect() -> None:
    """Reject provider items that exceed the bound before tool execution."""
    scope = authority()
    tool_effects = 0

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        return str(arguments)

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )

    async def response_handler(request_value: httpx.Request) -> httpx.Response:
        await request_value.aread()
        return httpx.Response(
            200,
            json=_response(
                "provider-item-overflow-response",
                [
                    _reasoning(
                        "provider-item-overflow-reasoning",
                        "provider-item-overflow-private",
                    ),
                    _function_call(
                        "provider-item-overflow-call",
                        "provider-item-overflow-call-id",
                    ),
                ],
            ),
        )

    binding = _binding(lane_id="lane-native-provider-item-overflow")
    provider = _provider(binding, response_handler, tools=(tool,))
    _, coordinator, store = _direct_client(
        provider,
        namespace="native-provider-item-overflow",
    )
    runtime = next(iter(coordinator._lanes.values()))
    object.__setattr__(runtime, "max_output_items", 1)
    run_request = request(
        scope=scope,
        identity=root_identity("native-provider-item-overflow"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key="native-provider-item-overflow",
        response_suffix="native-provider-item-overflow",
    )

    with pytest.raises(conversation.ConversationLimitError):
        await coordinator.execute(run_request)
    assert tool_effects == 0
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(run_request.identity.checkpoint_id, scope)
    await coordinator.close()


async def test_native_output_byte_limit_precedes_tool_effect_and_commit(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject oversized generated tool bytes before staging or commit."""
    record_property("conversation_acceptance_evidence", "negative")
    scope = authority()
    tool_effects = 0
    dispatches = 0

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        assert arguments == {"value": 1}
        return "x" * 10_240

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )

    payload = _response(
        "bounded-byte-response",
        [
            _reasoning("bounded-byte-reasoning", "private-bytes"),
            _function_call("bounded-byte-call", "byte-call-id"),
        ],
    )
    binding = _binding(
        streaming=True,
        lane_id="lane-native-byte-boundary",
    )
    result = _provider_result(payload, binding)
    stream = _DirectProviderStream(result.items, result)

    async def response_handler(request_value: httpx.Request) -> httpx.Response:
        await request_value.aread()
        return httpx.Response(200, json=payload)

    provider = _provider(binding, response_handler, tools=(tool,))

    async def open_stream(
        plan: conversation.ProviderPlan,
    ) -> conversation.ConversationProviderStream:
        nonlocal dispatches
        assert type(plan) is conversation.StatelessProviderPlan
        dispatches += 1
        return stream

    monkeypatch.setattr(provider, "stream", open_stream)
    _, coordinator, store = _direct_client(
        provider,
        namespace="native-byte-boundary",
    )
    runtime = next(iter(coordinator._lanes.values()))
    provider_bytes = sum(
        items_module.provider_item_byte_count(item) for item in result.items
    )
    object.__setattr__(runtime, "max_output_bytes", provider_bytes + 1_024)
    run_request = request(
        scope=scope,
        identity=root_identity("native-byte-boundary"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key="native-byte-boundary",
        response_suffix="native-byte-boundary",
    )
    sink = _RecordingSink()
    with pytest.raises(conversation.ConversationLimitError):
        await coordinator.stream_with_sink(run_request, sink)
    assert dispatches == 1
    assert tool_effects == 1
    assert sink.staged_items == []
    assert sink.finalize_calls == 0
    assert sink.cleanup_calls == 1
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(run_request.identity.checkpoint_id, scope)
    await coordinator.close()


async def test_native_output_limits_accumulate_across_segments() -> None:
    """Carry generated-item reservations across every tool segment."""
    scope = authority()
    dispatches = 0
    tool_effects = 0
    responses = (
        _response(
            "multi-segment-response-one",
            [_function_call("multi-segment-call-one", "segment-call-one")],
        ),
        _response(
            "multi-segment-response-two",
            [_function_call("multi-segment-call-two", "segment-call-two")],
        ),
        _response(
            "multi-segment-response-three",
            [_message("multi-segment-message", "must not dispatch")],
        ),
    )

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        assert arguments == {"value": 1}
        return f"segment-output-{tool_effects}"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )

    async def response_handler(request_value: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request_value.aread()
        return httpx.Response(200, json=responses[dispatches - 1])

    binding = _binding(lane_id="lane-native-multi-segment-limit")
    provider = _provider(binding, response_handler, tools=(tool,))
    _, coordinator, store = _direct_client(
        provider,
        namespace="native-multi-segment-limit",
    )
    runtime = next(iter(coordinator._lanes.values()))
    object.__setattr__(runtime, "max_output_items", 3)
    run_request = request(
        scope=scope,
        identity=root_identity("native-multi-segment-limit"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key="native-multi-segment-limit",
        response_suffix="native-multi-segment-limit",
    )

    with pytest.raises(conversation.ConversationLimitError):
        await coordinator.execute(run_request)
    assert dispatches == 2
    assert tool_effects == 1
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(run_request.identity.checkpoint_id, scope)
    await coordinator.close()


async def test_native_generated_output_limits_accept_exact_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept exact aggregate item and byte bounds through a tool cycle."""
    scope = authority()
    dispatches = 0
    tool_effects = 0
    tool_output = "exact-boundary-output"
    call_id = "exact-boundary-call-id"
    responses = (
        _response(
            "exact-boundary-response-one",
            [_function_call("exact-boundary-call", call_id)],
        ),
        _response(
            "exact-boundary-response-two",
            [_message("exact-boundary-message", "exact boundary")],
        ),
    )

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        assert arguments == {"value": 1}
        return tool_output

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )

    async def response_handler(request_value: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request_value.aread()
        return httpx.Response(200, json=responses[dispatches - 1])

    binding = _binding(
        streaming=True,
        lane_id="lane-native-exact-output-limit",
    )
    expected_provider_items = tuple(
        item
        for payload in responses
        for item in _provider_result(payload, binding).items
    )
    generated_bytes = len(
        conversation.canonical_json_bytes(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_output,
            }
        )
    )
    exact_bytes = generated_bytes + sum(
        items_module.provider_item_byte_count(item)
        for item in expected_provider_items
    )
    provider = _provider(binding, response_handler, tools=(tool,))

    async def open_stream(
        plan: conversation.ProviderPlan,
    ) -> conversation.ConversationProviderStream:
        nonlocal dispatches
        assert type(plan) is conversation.StatelessProviderPlan
        payload = responses[dispatches]
        dispatches += 1
        result = provider_module._provider_result_mapping(
            _frozen_mapping(payload),
            plan,
        )
        return _DirectProviderStream(result.items, result)

    monkeypatch.setattr(provider, "stream", open_stream)
    _, coordinator, store = _direct_client(
        provider,
        namespace="native-exact-output-limit",
    )
    runtime = next(iter(coordinator._lanes.values()))
    object.__setattr__(runtime, "max_output_items", 3)
    object.__setattr__(runtime, "max_output_bytes", exact_bytes)
    run_request = request(
        scope=scope,
        identity=root_identity("native-exact-output-limit"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key="native-exact-output-limit",
        response_suffix="native-exact-output-limit",
    )

    sink = _RecordingSink()
    receipt = await coordinator.stream_with_sink(run_request, sink)
    completed = receipt.output_candidates[0].completed_items
    assert dispatches == 2
    assert tool_effects == 1
    assert len(completed) == 3
    assert sink.staged_items == list(completed)
    assert sink.finalize_calls == 1
    assert sink.cleanup_calls == 1
    assert (
        sum(items_module.provider_item_byte_count(item) for item in completed)
        == exact_bytes
    )
    assert (
        await store.load(
            run_request.identity.checkpoint_id,
            scope,
        )
        == receipt.checkpoint
    )
    await coordinator.close()


async def test_native_output_segment_limit_stops_before_redispatch() -> None:
    """Stop a tool cycle at its segment bound before another dispatch."""
    scope = authority()
    dispatches = 0
    tool_effects = 0

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        assert arguments == {"value": 1}
        return "segment-result"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )

    async def response_handler(request_value: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request_value.aread()
        return httpx.Response(
            200,
            json=_response(
                "bounded-segment-response",
                [
                    _reasoning(
                        "bounded-segment-reasoning",
                        "bounded-segment-private",
                    ),
                    _function_call(
                        "bounded-segment-call",
                        "bounded-segment-call-id",
                    ),
                ],
            ),
        )

    binding = _binding(lane_id="lane-native-segment-boundary")
    provider = _provider(binding, response_handler, tools=(tool,))
    _, coordinator, store = _direct_client(
        provider,
        namespace="native-segment-boundary",
    )
    runtime = next(iter(coordinator._lanes.values()))
    object.__setattr__(runtime, "max_output_segments", 1)
    run_request = request(
        scope=scope,
        identity=root_identity("native-segment-boundary"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key="native-segment-boundary",
        response_suffix="native-segment-boundary",
    )
    with pytest.raises(conversation.ConversationLimitError):
        await coordinator.execute(run_request)
    assert dispatches == 1
    assert tool_effects == 1
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(run_request.identity.checkpoint_id, scope)
    await coordinator.close()


async def test_native_stream_staging_rejects_item_and_terminal_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed streamed items and terminal parity before commit."""
    scope = authority()
    for mutation in ("item", "terminal"):
        binding = _binding(
            streaming=True,
            lane_id=f"lane-native-stream-{mutation}",
        )
        provider = _provider(binding, _unused_handler)
        result = _provider_result(
            _response(
                f"stream-{mutation}-response",
                [
                    _reasoning(
                        f"stream-{mutation}-reasoning",
                        "stream-private",
                    ),
                    _message(f"stream-{mutation}-message", "visible"),
                ],
            ),
            binding,
        )
        if mutation == "item":
            items = (
                replace(
                    result.items[0],
                    lane_id=conversation.ProviderLaneId("forged-lane"),
                ),
            )
            terminal = result
        else:
            items = result.items
            terminal = replace(result, items=result.items[:-1])
        stream = _DirectProviderStream(items, terminal)

        async def open_stream(
            plan: conversation.ProviderPlan,
        ) -> conversation.ConversationProviderStream:
            assert type(plan) is conversation.StatelessProviderPlan
            return cast(conversation.ConversationProviderStream, stream)

        with monkeypatch.context() as patch:
            patch.setattr(provider, "stream", open_stream)
            _, coordinator, store = _direct_client(
                provider,
                namespace=f"native-stream-{mutation}",
            )
            run_request = request(
                scope=scope,
                identity=root_identity(f"native-stream-{mutation}"),
                advance=conversation.FirstTurnAdvance(),
                lane_ids=(str(binding.lane_id),),
                key=f"native-stream-{mutation}",
                response_suffix=f"native-stream-{mutation}",
            )
            with pytest.raises(conversation.ConversationProviderResponseError):
                await coordinator.stream(run_request)
            assert stream.close_count == (2 if mutation == "item" else 1)
            with pytest.raises(conversation.ConversationAuthorizationError):
                await store.load(run_request.identity.checkpoint_id, scope)
        await coordinator.close()


def test_native_segment_rejects_stored_provider_identity() -> None:
    """Reject an upstream response identity on a stateless segment."""
    binding = _binding(lane_id="lane-native-stored-segment")
    plan = _plan(binding)
    result = _provider_result(
        _response(
            "stored-segment-response",
            [_message("stored-segment-message", "visible")],
        ),
        binding,
    )
    malformed = replace(
        result,
        upstream_response_id=conversation.UpstreamResponseId(
            "forged-upstream-response"
        ),
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        conversation.RunScopedConversationCoordinator._validate_native_provider_segment(
            plan,
            (),
            malformed,
        )


def test_public_projection_and_item_byte_guards_cover_each_part() -> None:
    """Reject malformed item bytes and inspect every visible text part."""
    with pytest.raises(conversation.ConversationValidationError):
        items_module.provider_item_byte_count(
            cast(conversation.ProviderItem, object())
        )

    item = object.__new__(conversation.ProviderItem)
    object.__setattr__(item, "kind", conversation.ProviderItemKind.MESSAGE)
    object.__setattr__(
        item,
        "phase",
        conversation.ProviderItemPhase.ASSISTANT,
    )
    object.__setattr__(
        item,
        "caller",
        conversation.ProviderItemCaller.PROVIDER,
    )
    object.__setattr__(
        item,
        "canonical_input",
        {
            "content": (
                {"type": "input_text", "text": "private-input"},
                {"type": "output_text", "text": "visible-output"},
            )
        },
    )
    assert sdk_module.public_provider_item_projection((item,)) == (
        conversation.VisibleTranscriptEntry(
            role=conversation.VisibleTranscriptRole.ASSISTANT,
            content="visible-output",
        ),
    )

    object.__setattr__(
        item,
        "canonical_input",
        {"content": ({"type": "output_text", "text": 1},)},
    )
    with pytest.raises(conversation.ConversationValidationError):
        sdk_module.public_provider_item_projection((item,))


@pytest.mark.parametrize(
    "malformation",
    (
        "duplicate_item_id",
        "duplicate_call_id",
        "unpermitted_open_call",
    ),
)
async def test_complete_provider_segment_is_validated_before_tool_effect(
    malformation: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject duplicate and unowned call state before any tool effect."""
    record_property("conversation_acceptance_evidence", "security")
    scope = authority()
    tool_effects = 0

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        return str(arguments)

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )

    async def response_handler(request_value: httpx.Request) -> httpx.Response:
        await request_value.aread()
        if malformation == "duplicate_item_id":
            output = [
                _reasoning("duplicate-item", "private-duplicate"),
                _function_call("duplicate-item", "unique-call-id"),
            ]
        elif malformation == "duplicate_call_id":
            output = [
                _reasoning("duplicate-call-reasoning", "private-duplicate"),
                _function_call("duplicate-call-one", "duplicate-call-id"),
                _function_call("duplicate-call-two", "duplicate-call-id"),
            ]
        else:
            output = [
                _reasoning("open-call-reasoning", "private-open"),
                {
                    "arguments": "{}",
                    "call_id": "unpermitted-call-id",
                    "id": "unpermitted-call",
                    "status": "completed",
                    "type": "tool_search_call",
                },
            ]
        return httpx.Response(
            200,
            json=_response(f"malformed-{malformation}", output),
        )

    binding = _binding(lane_id=f"lane-{malformation}")
    provider = _provider(binding, response_handler, tools=(tool,))
    _, coordinator, store = _direct_client(
        provider,
        namespace=f"segment-{malformation}",
    )
    run_request = request(
        scope=scope,
        identity=root_identity(f"segment-{malformation}"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key=f"segment-{malformation}",
        response_suffix=f"segment-{malformation}",
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        await coordinator.execute(run_request)
    assert tool_effects == 0
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(run_request.identity.checkpoint_id, scope)
    await coordinator.close()


@pytest.mark.parametrize(
    "mutation",
    ("lane_identity", "model_call_identity", "order", "provider_index"),
)
async def test_segment_identity_order_and_indexes_precede_effects(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject forged segment identity and positions before tool execution."""
    scope = authority()
    tool_effects = 0

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        return str(arguments)

    binding = _binding(lane_id=f"lane-segment-{mutation}")
    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )
    provider = _provider(binding, _unused_handler, tools=(tool,))
    valid = _provider_result(
        _response(
            f"segment-{mutation}",
            [
                _reasoning(f"reasoning-{mutation}", "private-segment"),
                _function_call(f"call-{mutation}", f"call-id-{mutation}"),
            ],
        ),
        binding,
    )
    items = list(valid.items)
    selected = items[1]
    if mutation == "lane_identity":
        selected = replace(
            selected,
            lane_id=conversation.ProviderLaneId("forged-lane"),
        )
    elif mutation == "model_call_identity":
        selected = replace(
            selected,
            model_call_id=conversation.ConversationModelCallId(
                "forged-model-call"
            ),
        )
    elif mutation == "order":
        selected = replace(selected, order=conversation.ProviderItemOrder(0))
    else:
        selected = replace(
            selected,
            provider_index=conversation.ProviderItemIndex(0),
        )
    items[1] = selected
    malformed = conversation.ProviderResult(
        items=tuple(items),
        reasoning=valid.reasoning,
        usage=valid.usage,
    )

    async def dispatch(
        plan: conversation.ProviderPlan,
    ) -> conversation.ProviderResult:
        assert type(plan) is conversation.StatelessProviderPlan
        return malformed

    monkeypatch.setattr(provider, "dispatch", dispatch)
    _, coordinator, store = _direct_client(
        provider,
        namespace=f"segment-{mutation}",
    )
    run_request = request(
        scope=scope,
        identity=root_identity(f"forged-segment-{mutation}"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key=f"forged-segment-{mutation}",
        response_suffix=f"forged-segment-{mutation}",
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        await coordinator.execute(run_request)
    assert tool_effects == 0
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(run_request.identity.checkpoint_id, scope)
    await coordinator.close()


async def test_tool_arguments_effects_and_results_are_bounded() -> None:
    """Reject malformed arguments and conceal every tool-side failure."""
    calls: list[Mapping[str, JsonValue]] = []

    async def valid(arguments: Mapping[str, JsonValue]) -> str:
        calls.append(arguments)
        return "valid"

    tool = conversation.NativeOpenAIFunctionTool(
        name="validation-tool",
        parameters={"type": "object"},
        handler=valid,
    )
    for arguments in ("{", '{"x":1,"x":2}', '{"x":NaN}', "[]"):
        with pytest.raises(
            conversation.ConversationProviderResponseError
        ) as exc:
            await tool.execute(arguments)
        assert exc.value.__cause__ is None
    assert await tool.execute('{"x":1}') == "valid"
    assert len(calls) == 1

    async def fails(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments
        raise RuntimeError("tool-private-sentinel")

    failed_tool = replace(tool, handler=fails)
    with pytest.raises(conversation.ConversationError) as effect:
        await failed_tool.execute('{"x":1}')
    assert effect.value.boundary is conversation.FailureBoundary.TOOL_EFFECT
    assert effect.value.__cause__ is None
    assert "tool-private-sentinel" not in repr(effect.value)

    async def cancels(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments
        raise CancelledError()

    with pytest.raises(CancelledError):
        await replace(tool, handler=cancels).execute('{"x":1}')

    async def wrong_type(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments
        return cast(str, 1)

    with pytest.raises(conversation.ConversationValidationError):
        await replace(tool, handler=wrong_type).execute('{"x":1}')

    async def too_large(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments
        return "x" * 1_048_577

    with pytest.raises(conversation.ConversationValidationError):
        await replace(tool, handler=too_large).execute('{"x":1}')


async def test_provider_constructor_and_plan_validation_are_closed() -> None:
    """Reject wrong owners, duplicate tools, and every direct plan drift."""
    binding = _binding(lane_id="lane-plan-validation")
    capability_profile = _capabilities(binding)
    profile = _profile(binding)
    client = AsyncOpenAI(
        api_key="validation-key",
        base_url=binding.normalized_endpoint,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_unused_handler)
        ),
        max_retries=0,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIStatelessProvider(
            client=cast(AsyncOpenAI, object()),
            profile=profile,
            capability_profile=capability_profile,
        )

    async def valid(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments
        return "valid"

    duplicate = conversation.NativeOpenAIFunctionTool(
        name="duplicate",
        parameters={"type": "object"},
        handler=valid,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIStatelessProvider(
            client=client,
            profile=profile,
            capability_profile=capability_profile,
            tools=(duplicate, duplicate),
        )

    provider = conversation.NativeOpenAIStatelessProvider(
        client=client,
        profile=profile,
        capability_profile=capability_profile,
    )
    plan = _plan(binding)
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.dispatch(cast(conversation.ProviderPlan, object()))
    drifted_bindings = (
        replace(binding, normalized_endpoint="https://drift.example/v1"),
        replace(binding, model_or_deployment="drift-model"),
        replace(
            binding,
            tool_schema_revision=conversation.ToolSchemaRevision(
                "tools-drift"
            ),
        ),
        replace(
            binding,
            execution_definition_revision=(
                conversation.ExecutionDefinitionRevision("execution-drift")
            ),
        ),
        replace(
            binding,
            capability_profile_revision=(
                conversation.CapabilityProfileRevision("capability-drift")
            ),
        ),
        replace(
            binding,
            continuation_codec_version=conversation.ConversationCodecVersion(
                2
            ),
        ),
        replace(
            binding,
            sdk_revision=conversation.ProviderSdkRevision(
                "openai-python-drift"
            ),
        ),
        replace(
            binding,
            provider_api_revision=conversation.ProviderApiRevision(
                "openapi-drift"
            ),
        ),
        replace(binding, transport=conversation.ProviderTransport.STREAMING),
    )
    for drifted in drifted_bindings:
        with pytest.raises(conversation.ConversationBindingDriftError):
            await provider.dispatch(replace(plan, binding=drifted))

    with pytest.raises(conversation.ConversationBindingDriftError):
        await provider.stream(plan)
    drifted_ledger = plan.ledger
    object.__setattr__(
        drifted_ledger,
        "lane_id",
        conversation.ProviderLaneId("lane-forged-ledger"),
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        await provider.dispatch(plan)

    stored = conversation.StoredProviderPlan(
        binding=binding,
        upstream_response_id=conversation.UpstreamResponseId("stored-only"),
        reasoning=plan.reasoning,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.dispatch(stored)
    await provider.aclose()
    await provider.aclose()
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.dispatch(plan)


@pytest.mark.parametrize(
    ("unsupported", "streaming", "reasoning"),
    [
        (
            conversation.ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY,
            False,
            avalan.ReasoningContext.AUTO,
        ),
        (
            conversation.ConversationCapability.STREAMING_ITEM_FIDELITY,
            True,
            avalan.ReasoningContext.AUTO,
        ),
        (
            conversation.ConversationCapability.REASONING_CONTEXT_CURRENT_TURN,
            False,
            avalan.ReasoningContext.CURRENT_TURN,
        ),
        (
            conversation.ConversationCapability.REASONING_CONTEXT_ALL_TURNS,
            False,
            avalan.ReasoningContext.ALL_TURNS,
        ),
    ],
)
async def test_capabilities_fail_before_dispatch(
    unsupported: conversation.ConversationCapability,
    streaming: bool,
    reasoning: avalan.ReasoningContext,
    record_property: Callable[[str, object], None],
) -> None:
    """Require each independently conformed capability at dispatch."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        return await _unused_handler(request)

    binding = _binding(
        streaming=streaming,
        lane_id=f"lane-unsupported-{unsupported.value}",
    )
    provider = _raw_provider(
        binding,
        _capability_profile(binding, unsupported=unsupported),
        handler=handler,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        if streaming:
            await provider.stream(_plan(binding, reasoning=reasoning))
        else:
            await provider.dispatch(_plan(binding, reasoning=reasoning))
    assert dispatches == 0
    await provider.aclose()


async def test_compaction_and_forged_input_cannot_override_wire() -> None:
    """Reject unsupported compaction and every forged request-root mapping."""
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        return await _unused_handler(request)

    binding = _binding(lane_id="lane-forged-input")
    provider = _provider(binding, handler)
    plan = _plan(binding)
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.dispatch(
            replace(
                plan,
                compaction=conversation.InlineCompaction(
                    compact_threshold=1_000
                ),
            )
        )
    forged_inputs: tuple[Mapping[str, JsonValue] | None, ...] = (
        None,
        {"text": "safe", "store": True},
        {"text": ""},
        {"text": 1},
    )
    for forged in forged_inputs:
        with pytest.raises(conversation.ConversationValidationError):
            await provider.dispatch(replace(plan, new_input=forged))
    assert dispatches == 0
    await provider.aclose()


async def test_exact_sdk_endpoint_and_query_classification() -> None:
    """Classify only proven production hosts or explicit loopback tests."""
    production = _binding(lane_id="lane-production-classifier")

    async def build_rejected(
        binding: conversation.ProviderLaneBinding,
        profile: conversation.NativeOpenAIStatelessProfile,
        capability_profile: conversation.ConversationCapabilityProfile,
        *,
        default_query: Mapping[str, object] | None = None,
    ) -> None:
        client = AsyncOpenAI(
            api_key="classifier-key",
            base_url=binding.normalized_endpoint,
            default_query=default_query,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(_unused_handler)
            ),
            max_retries=0,
        )
        try:
            with pytest.raises(conversation.ConversationError):
                conversation.NativeOpenAIStatelessProvider(
                    client=client,
                    profile=profile,
                    capability_profile=capability_profile,
                )
        finally:
            await client.close()

    await build_rejected(
        production,
        replace(_profile(production), scripted_tcp_test=True),
        _capability_profile(production, test_only=False),
    )
    await build_rejected(
        production,
        _profile(production),
        _capabilities(production),
        default_query={"stateful-extension": "forbidden"},
    )

    loopback = _binding(
        endpoint="http://127.0.0.1:18080/v1",
        lane_id="lane-loopback-classifier",
    )
    provider = _raw_provider(
        loopback,
        _capabilities(loopback),
        profile=_profile(loopback, scripted_tcp_test=True),
    )
    await provider.aclose()

    nonlocal_test = _binding(
        endpoint="https://example.test/v1",
        lane_id="lane-nonlocal-test",
    )
    await build_rejected(
        nonlocal_test,
        _profile(nonlocal_test, scripted_tcp_test=True),
        _capabilities(nonlocal_test),
    )

    azure = _binding(azure=True, lane_id="lane-azure-classifier")
    wrong_host = replace(
        azure,
        normalized_endpoint="https://azure.example/openai/v1",
        azure_resource_identity="azure.example",
    )
    await build_rejected(
        wrong_host,
        _profile(wrong_host),
        _capabilities(wrong_host),
    )
    wrong_path = replace(
        azure,
        normalized_endpoint="https://resource.openai.azure.com/v1",
    )
    await build_rejected(
        wrong_path,
        _profile(wrong_path),
        _capabilities(wrong_path),
    )
    wrong_resource = replace(
        azure,
        azure_resource_identity="other.openai.azure.com",
    )
    await build_rejected(
        wrong_resource,
        _profile(wrong_resource),
        _capabilities(wrong_resource),
    )

    preview = replace(
        azure,
        provider_api_revision=conversation.ProviderApiRevision(
            "azure-openai-v1-preview"
        ),
    )
    await build_rejected(
        preview,
        _profile(preview),
        _capabilities(preview),
    )
    preview_provider = _raw_provider(
        preview,
        _capabilities(preview),
        default_query={"api-version": "preview"},
    )
    await preview_provider.aclose()


async def test_sdk_failures_are_typed_redacted_and_counted(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Map SDK rejection, ambiguity, cancellation, and invalid unions."""
    record_property("conversation_acceptance_evidence", "negative")
    status_dispatches = 0

    async def rejected(request: httpx.Request) -> httpx.Response:
        nonlocal status_dispatches
        status_dispatches += 1
        await request.aread()
        return httpx.Response(
            400,
            json={"error": {"message": "status-private", "type": "bad"}},
        )

    binding = _binding(lane_id="lane-status-failure")
    rejected_provider = _provider(binding, rejected)
    with pytest.raises(conversation.ConversationError) as status:
        await rejected_provider.dispatch(_plan(binding))
    assert status_dispatches == 1
    assert status.value.boundary is (
        conversation.FailureBoundary.PROVIDER_REJECTION
    )
    assert status.value.__cause__ is None
    assert "status-private" not in repr(status.value)
    assert rejected_provider.diagnostics.failure_boundary == (
        conversation.FailureBoundary.PROVIDER_REJECTION.value
    )
    await rejected_provider.aclose()

    async def invalid_json(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            content=b"not-json-private",
            headers={"content-type": "application/json"},
        )

    invalid_provider = _provider(
        _binding(lane_id="lane-invalid-sdk-response"),
        invalid_json,
    )
    with pytest.raises(conversation.ConversationError) as invalid:
        await invalid_provider.dispatch(_plan(invalid_provider.binding))
    assert invalid.value.__cause__ is None
    assert "not-json-private" not in repr(invalid.value)
    await invalid_provider.aclose()

    async def validation_error(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(200, request=request)
        raise APIResponseValidationError(
            response=response,
            body={"private": "validation-private"},
            message="validation-private",
        )

    validation_provider = _provider(
        _binding(lane_id="lane-sdk-validation-error"),
        _unused_handler,
    )
    monkeypatch.setattr(
        provider_module,
        "_create_exact_response",
        validation_error,
    )
    with pytest.raises(conversation.ConversationProviderResponseError) as sdk:
        await validation_provider.dispatch(_plan(validation_provider.binding))
    assert sdk.value.__cause__ is None
    assert "validation-private" not in repr(sdk.value)
    await validation_provider.aclose()

    async def broken_create(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        raise RuntimeError("generic-sdk-private")

    generic_provider = _provider(
        _binding(lane_id="lane-generic-sdk-failure"),
        _unused_handler,
    )
    monkeypatch.setattr(
        provider_module,
        "_create_exact_response",
        broken_create,
    )
    with pytest.raises(
        conversation.ConversationAmbiguousDispatchError
    ) as generic:
        await generic_provider.dispatch(_plan(generic_provider.binding))
    assert generic.value.__cause__ is None
    assert "generic-sdk-private" not in repr(generic.value)

    async def cancelled_create(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        raise CancelledError()

    monkeypatch.setattr(
        provider_module,
        "_create_exact_response",
        cancelled_create,
    )
    with pytest.raises(CancelledError):
        await generic_provider.dispatch(_plan(generic_provider.binding))

    async def wrong_union(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        return object()

    monkeypatch.setattr(
        provider_module,
        "_create_exact_response",
        wrong_union,
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        await generic_provider.dispatch(_plan(generic_provider.binding))
    streaming_binding = _binding(
        streaming=True,
        lane_id="lane-wrong-stream-union",
    )
    streaming_provider = _provider(streaming_binding, _unused_handler)
    with pytest.raises(conversation.ConversationProviderResponseError):
        await streaming_provider.stream(_plan(streaming_binding))
    await generic_provider.aclose()
    await streaming_provider.aclose()


async def test_execute_tool_and_replay_opaque_validation() -> None:
    """Reject wrong tools and malformed checkpoint-only opaque payloads."""
    binding = _binding(lane_id="lane-tool-dispatch-validation")
    provider = _provider(binding, _unused_handler)
    result = _provider_result(
        _response(
            "tool-validation-response",
            [
                _reasoning("tool-validation-reasoning", "private-reasoning"),
                _function_call("tool-validation-call", "tool-validation-id"),
            ],
        ),
        binding,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await provider.execute_tool(result.items[0])
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.execute_tool(result.items[1])
    corrupted_call = result.items[1]
    object.__setattr__(
        corrupted_call,
        "canonical_input",
        {"name": 1, "arguments": 2},
    )
    with pytest.raises(conversation.ConversationValidationError):
        await provider.execute_tool(corrupted_call)

    reasoning = result.items[0]
    for opaque in (
        b"\xff",
        b" private ",
        b"x" * 1_048_577,
    ):
        malformed = replace(
            reasoning,
            opaque_state=conversation.OpaqueProviderState(_value=opaque),
        )
        ledger = conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=(
                conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
            ),
            items=(malformed,),
        )
        with pytest.raises(conversation.ConversationError) as failure:
            await provider.dispatch(replace(_plan(binding), ledger=ledger))
        assert failure.value.__cause__ is None
    valid_ledger = conversation.ProviderItemLedger(
        lane_id=binding.lane_id,
        normalization_version=(
            conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
        ),
        items=(reasoning,),
    )
    object.__setattr__(reasoning, "canonical_input", ())
    with pytest.raises(conversation.ConversationValidationError):
        await provider.dispatch(replace(_plan(binding), ledger=valid_ledger))
    await provider.aclose()


def test_sdk_mapping_and_synthetic_optional_item_id_are_strict() -> None:
    """Reject non-SDK payloads and synthesize only allowed optional IDs."""
    with pytest.raises(conversation.ConversationProviderResponseError):
        provider_module._sdk_mapping(object())

    class _BadDump:
        def model_dump(
            self,
            *,
            mode: str,
            exclude_none: bool,
        ) -> object:
            assert mode == "json"
            assert exclude_none
            return []

    with pytest.raises(conversation.ConversationProviderResponseError):
        provider_module._sdk_mapping(_BadDump())

    payload = _response(
        "optional-id-response",
        [_function_call("discarded-id", "optional-id-call")],
    )
    cast(list[dict[str, object]], payload["output"])[0].pop("id")
    result = _provider_result(payload)
    assert str(result.items[0].item_id).startswith("provider-item-native-")


async def test_stream_transport_failures_use_exact_output_boundary(
    record_property: Callable[[str, object], None],
) -> None:
    """Distinguish ambiguous stream open from malformed post-output state."""
    record_property("conversation_acceptance_evidence", "negative")
    binding = _binding(streaming=True, lane_id="lane-stream-transport")
    provider = _provider(binding, _unused_handler)

    before, _ = _scripted_stream(
        provider,
        (RuntimeError("stream-private-before"),),
    )
    with pytest.raises(conversation.ConversationAmbiguousDispatchError) as pre:
        await before.__anext__()
    assert pre.value.__cause__ is None
    assert "stream-private-before" not in repr(pre.value)
    assert provider.diagnostics.failure_boundary == (
        conversation.FailureBoundary.AMBIGUOUS_POSSIBLE_DISPATCH.value
    )
    await before.aclose()

    cancelled, _ = _scripted_stream(provider, (CancelledError(),))
    with pytest.raises(CancelledError):
        await cancelled.__anext__()
    await cancelled.aclose()

    done = _done_event(
        0,
        _reasoning("stream-transport-item", "stream-transport-private"),
    )
    after, _ = _scripted_stream(
        provider,
        (done, RuntimeError("stream-private-after")),
    )
    await after.__anext__()
    with pytest.raises(conversation.ConversationError) as post:
        await after.__anext__()
    assert post.value.boundary is (
        conversation.FailureBoundary.MALFORMED_STREAM_ITEM
    )
    assert post.value.__cause__ is None
    assert "stream-private-after" not in repr(post.value)
    await after.aclose()
    await provider.aclose()


async def test_stream_event_order_sequence_and_terminal_are_strict(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject duplicate, partial, failed, and post-terminal stream events."""
    record_property("conversation_acceptance_evidence", "security")
    binding = _binding(streaming=True, lane_id="lane-stream-events")
    provider = _provider(binding, _unused_handler)
    reasoning = _reasoning("stream-event-item", "stream-event-private")

    malformed_steps: tuple[tuple[object, ...], ...] = (
        (_done_event(-1, reasoning),),
        (_done_event(1, reasoning),),
        (
            _SdkEvent(
                {
                    "type": "response.output_item.done",
                    "sequence_number": 0,
                    "output_index": 0,
                    "item": "wrong",
                }
            ),
        ),
        (_SdkEvent({"type": "error", "sequence_number": 0}),),
        (_SdkEvent({"type": "error", "sequence_number": "bad"}),),
        (
            _SdkEvent({"type": "ignored"}),
            _SdkEvent({"type": "response.failed"}),
        ),
        (
            _SdkEvent({"type": "ignored", "sequence_number": 0}),
            _SdkEvent({"type": "response.failed", "sequence_number": 0}),
        ),
        (_SdkEvent({"type": "response.completed", "sequence_number": 0}),),
        (
            _terminal_event([], sequence=0),
            _terminal_event([], sequence=1),
        ),
        (
            _terminal_event([], sequence=0),
            _done_event(0, reasoning, sequence=1),
        ),
    )
    for steps in malformed_steps:
        stream, _ = _scripted_stream(provider, steps)
        with pytest.raises(conversation.ConversationError):
            await stream.__anext__()
        await stream.aclose()

    duplicate, _ = _scripted_stream(
        provider,
        (
            _done_event(0, reasoning, sequence=0),
            _done_event(0, reasoning, sequence=1),
        ),
    )
    await duplicate.__anext__()
    with pytest.raises(conversation.ConversationError) as duplicate_error:
        await duplicate.__anext__()
    assert duplicate_error.value.boundary is (
        conversation.FailureBoundary.MALFORMED_STREAM_ITEM
    )
    await duplicate.aclose()

    unknown_after_output = {**reasoning, "type": "unknown-output-kind"}
    converted, _ = _scripted_stream(
        provider,
        (
            _done_event(0, reasoning, sequence=0),
            _done_event(1, unknown_after_output, sequence=1),
        ),
    )
    await converted.__anext__()
    with pytest.raises(conversation.ConversationError) as converted_error:
        await converted.__anext__()
    assert converted_error.value.boundary is (
        conversation.FailureBoundary.MALFORMED_STREAM_ITEM
    )
    await converted.aclose()
    await provider.aclose()


async def test_stream_terminal_and_close_fail_closed() -> None:
    """Require terminal parity and close each SDK stream exactly once."""
    binding = _binding(streaming=True, lane_id="lane-stream-terminal")
    provider = _provider(binding, _unused_handler)
    reasoning = _reasoning("stream-terminal-item", "terminal-private")

    missing, missing_source = _scripted_stream(provider, ())
    with pytest.raises(conversation.ConversationProviderResponseError):
        await missing.terminal()
    await missing.aclose()
    await missing.aclose()
    assert missing_source.close_count == 1

    partial, _ = _scripted_stream(
        provider,
        (_done_event(0, reasoning),),
    )
    assert await partial.__anext__()
    with pytest.raises(conversation.ConversationError) as partial_error:
        await partial.terminal()
    assert partial_error.value.boundary is (
        conversation.FailureBoundary.MALFORMED_STREAM_ITEM
    )
    await partial.aclose()

    failed_close, failed_source = _scripted_stream(
        provider,
        (),
        close_error=RuntimeError("close-private"),
    )
    with pytest.raises(conversation.ConversationError) as close_error:
        await failed_close.aclose()
    assert close_error.value.__cause__ is None
    assert "close-private" not in repr(close_error.value)
    assert failed_source.close_count == 1
    await failed_close.aclose()
    assert failed_source.close_count == 2

    cancelled_close, cancelled_source = _scripted_stream(
        provider,
        (),
        close_error=CancelledError(),
    )
    with pytest.raises(CancelledError):
        await cancelled_close.aclose()
    assert cancelled_source.close_count == 1
    await cancelled_close.aclose()
    assert cancelled_source.close_count == 2
    assert provider.diagnostics.stream_close_count == 4
    await provider.aclose()


async def test_stream_close_owns_cancellation_to_terminal_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish one SDK stream close before preserving caller cancellation."""
    provider = _provider(
        _binding(streaming=True, lane_id="lane-stream-close-cancel"),
        _unused_handler,
    )
    stream, source = _scripted_stream(provider, ())
    entered = Event()
    release = Event()

    async def blocking_close() -> None:
        source.close_count += 1
        entered.set()
        await release.wait()

    monkeypatch.setattr(source, "close", blocking_close)
    close_task = create_task(stream.aclose())
    await entered.wait()
    close_task.cancel()
    release.set()
    with pytest.raises(CancelledError):
        await close_task
    assert source.close_count == 1
    assert provider.diagnostics.stream_close_count == 1
    assert stream._closed
    await stream.aclose()
    assert source.close_count == 1
    await provider.aclose()


async def test_provider_close_owns_cancellation_to_terminal_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish SDK client close once before preserving cancellation."""
    provider = _provider(
        _binding(lane_id="lane-provider-close-cancel"),
        _unused_handler,
    )
    entered = Event()
    release = Event()
    close_count = 0

    async def blocking_close() -> None:
        nonlocal close_count
        close_count += 1
        entered.set()
        await release.wait()

    monkeypatch.setattr(provider._client, "close", blocking_close)
    close_task = create_task(provider.aclose())
    await entered.wait()
    close_task.cancel()
    release.set()
    with pytest.raises(CancelledError):
        await close_task
    assert close_count == 1
    assert provider._closed
    await provider.aclose()
    assert close_count == 1


async def test_provider_close_retries_cancelled_and_failed_close_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry SDK close tasks that cancel themselves or fail."""
    cancelled_provider = _provider(
        _binding(lane_id="lane-provider-close-self-cancel"),
        _unused_handler,
    )
    close_count = 0

    async def cancel_then_close() -> None:
        nonlocal close_count
        close_count += 1
        if close_count == 1:
            raise CancelledError()

    with monkeypatch.context() as patch:
        patch.setattr(cancelled_provider._client, "close", cancel_then_close)
        with pytest.raises(CancelledError):
            await cancelled_provider.aclose()
        assert cancelled_provider._close_task is None
        assert not cancelled_provider._closed
        await cancelled_provider.aclose()
    assert close_count == 2
    assert cancelled_provider._closed

    failed_provider = _provider(
        _binding(lane_id="lane-provider-close-failed-task"),
        _unused_handler,
    )

    async def failed_close() -> None:
        raise RuntimeError("private-sdk-close")

    with monkeypatch.context() as patch:
        patch.setattr(failed_provider._client, "close", failed_close)
        with pytest.raises(conversation.ConversationCommitError) as failure:
            await failed_provider.aclose()
        assert failure.value.__cause__ is None
        assert "private-sdk-close" not in repr(failure.value)
        assert failed_provider._close_task is None
        assert not failed_provider._closed
    await failed_provider.aclose()


async def test_provider_and_stream_close_preserve_cancelled_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve caller cancellation after owned close tasks fail."""
    provider = _provider(
        _binding(lane_id="lane-provider-close-cancelled-failure"),
        _unused_handler,
    )
    provider_entered = Event()
    provider_release = Event()

    async def blocked_provider_failure() -> None:
        provider_entered.set()
        await provider_release.wait()
        raise RuntimeError("private-provider-close-failure")

    with monkeypatch.context() as patch:
        patch.setattr(provider._client, "close", blocked_provider_failure)
        provider_close = create_task(provider.aclose())
        await provider_entered.wait()
        provider_close.cancel()
        provider_release.set()
        with pytest.raises(CancelledError) as provider_cancelled:
            await provider_close
    assert isinstance(
        provider_cancelled.value.__cause__,
        conversation.ConversationCommitError,
    )
    assert "private-provider-close-failure" not in repr(
        provider_cancelled.value
    )
    await provider.aclose()

    stream_owner = _provider(
        _binding(
            streaming=True,
            lane_id="lane-stream-close-cancelled-failure",
        ),
        _unused_handler,
    )
    stream, source = _scripted_stream(stream_owner, ())
    stream_entered = Event()
    stream_release = Event()

    async def blocked_stream_failure() -> None:
        source.close_count += 1
        stream_entered.set()
        await stream_release.wait()
        raise RuntimeError("private-stream-close-failure")

    with monkeypatch.context() as patch:
        patch.setattr(source, "close", blocked_stream_failure)
        stream_close = create_task(stream.aclose())
        await stream_entered.wait()
        stream_close.cancel()
        stream_release.set()
        with pytest.raises(CancelledError) as stream_cancelled:
            await stream_close
    assert isinstance(
        stream_cancelled.value.__cause__,
        conversation.ConversationError,
    )
    assert stream_cancelled.value.__cause__.boundary is (
        conversation.FailureBoundary.MALFORMED_STREAM_ITEM
    )
    assert "private-stream-close-failure" not in repr(stream_cancelled.value)
    await stream.aclose()
    await stream_owner.aclose()


async def test_native_close_hook_cancellation_closes_without_commit(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Close after pre-close cancellation without committing state."""
    record_property("conversation_acceptance_evidence", "negative")
    scope = authority()
    binding = _binding(streaming=True, lane_id="lane-hook-close-cancel")
    provider = _provider(binding, _unused_handler)
    output = [
        _reasoning("hook-close-reasoning", "hook-close-private"),
        _message("hook-close-message", "hook close"),
    ]
    stream, source = _scripted_stream(
        provider,
        (
            _done_event(0, output[0], sequence=0),
            _done_event(1, output[1], sequence=1),
            _SdkEvent(
                {
                    "response": _response("hook-close-response", output),
                    "sequence_number": 2,
                    "type": "response.completed",
                }
            ),
        ),
    )

    async def open_stream(
        plan: conversation.ProviderPlan,
    ) -> conversation.ConversationProviderStream:
        assert type(plan) is conversation.StatelessProviderPlan
        return stream

    monkeypatch.setattr(provider, "stream", open_stream)
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_stream_close",
                exception=CancelledError(),
            ),
        )
    )
    _, coordinator, store = _direct_client(
        provider,
        namespace="hook-close-cancel",
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(controller),
    )
    run_request = request(
        scope=scope,
        identity=root_identity("hook-close-cancel"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        key="hook-close-cancel",
        response_suffix="hook-close-cancel",
    )
    with pytest.raises(CancelledError):
        await coordinator.stream(run_request)
    assert source.close_count == 1
    assert provider.diagnostics.stream_close_count == 1
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(run_request.identity.checkpoint_id, scope)
    await coordinator.close()


@pytest.mark.parametrize(
    ("primary", "close_steps", "expected"),
    (
        (None, (RuntimeError("cleanup-only"),), "cleanup"),
        (
            None,
            (CancelledError(), RuntimeError("cleanup-after-cancel")),
            "cancelled",
        ),
        (
            RuntimeError("primary-and-cleanup"),
            (RuntimeError("cleanup-with-primary"),),
            "primary_cleanup",
        ),
        (RuntimeError("primary-only"), (None,), "primary"),
        (
            RuntimeError("primary-cancelled"),
            (CancelledError(), RuntimeError("cleanup-cancelled")),
            "cancelled_primary_cleanup",
        ),
    ),
)
async def test_native_stream_close_settles_every_failure_combination(
    primary: BaseException | None,
    close_steps: tuple[BaseException | None, ...],
    expected: str,
) -> None:
    """Compose hook, cancellation, and provider-stream close failures."""
    provider = _provider(
        _binding(
            streaming=True,
            lane_id=f"lane-native-close-matrix-{expected}",
        ),
        _unused_handler,
    )
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_stream_close",
                exception=primary,
            ),
        )
        if primary is not None
        else ()
    )
    _, coordinator, _ = _direct_client(
        provider,
        namespace=f"native-close-matrix-{expected}",
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(controller),
    )
    stream = _CloseSequenceStream(close_steps)
    if expected.startswith("cancelled"):
        with pytest.raises(CancelledError) as failure:
            await coordinator._close_native_stream(
                cast(conversation.ConversationProviderStream, stream)
            )
    else:
        with pytest.raises(RuntimeError) as failure:
            await coordinator._close_native_stream(
                cast(conversation.ConversationProviderStream, stream)
            )
    if expected == "cleanup":
        assert str(failure.value) == "cleanup-only"
    elif expected == "cancelled":
        assert isinstance(failure.value.__cause__, RuntimeError)
        assert str(failure.value.__cause__) == "cleanup-after-cancel"
    elif expected == "primary_cleanup":
        assert str(failure.value) == "primary-and-cleanup"
        assert isinstance(failure.value.__cause__, RuntimeError)
        assert str(failure.value.__cause__) == "cleanup-with-primary"
    elif expected == "primary":
        assert str(failure.value) == "primary-only"
        assert failure.value.__cause__ is None
    else:
        assert isinstance(failure.value.__cause__, RuntimeError)
        assert str(failure.value.__cause__) == "primary-cancelled"
        assert isinstance(failure.value.__cause__.__cause__, RuntimeError)
        assert str(failure.value.__cause__.__cause__) == "cleanup-cancelled"
    assert stream.close_count == len(close_steps)
    await provider.aclose()


def test_stream_terminal_payload_parity_is_exact() -> None:
    """Reject missing indexes, scalar members, and terminal item drift."""
    reasoning = _frozen_mapping(
        _reasoning("terminal-parity-item", "terminal-parity-private")
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        provider_module._validate_stream_terminal(
            _frozen_mapping({"output": "wrong"}),
            {},
        )
    with pytest.raises(conversation.ConversationProviderResponseError):
        provider_module._validate_stream_terminal(
            _frozen_mapping({"output": [reasoning]}),
            {},
        )
    with pytest.raises(conversation.ConversationError):
        provider_module._validate_stream_terminal(
            _frozen_mapping({"output": [1]}),
            {0: reasoning},
        )
    with pytest.raises(conversation.ConversationError) as mismatch:
        provider_module._validate_stream_terminal(
            _frozen_mapping(
                {
                    "output": [
                        _reasoning(
                            "terminal-parity-item",
                            "terminal-parity-different",
                        )
                    ]
                }
            ),
            {0: reasoning},
        )
    assert mismatch.value.boundary is (
        conversation.FailureBoundary.MALFORMED_STREAM_ITEM
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "object",
        "status",
        "output-shape",
        "output-member",
        "raw-type",
        "unknown-kind",
        "no-provider-rule",
        "provider-output-kind",
        "incomplete-status",
        "missing-opaque",
        "malformed-opaque",
        "huge-opaque",
        "missing-fields",
        "bad-phase",
        "bad-reasoning",
        "bad-context",
        "numeric-context",
        "missing-usage",
        "negative-usage",
    ],
)
def test_malformed_provider_results_fail_content_free(mutation: str) -> None:
    """Reject every malformed required response component before commit."""
    payload = _response(
        "malformed-private-response-id",
        [
            _reasoning("malformed-reasoning", "malformed-private-opaque"),
            _message("malformed-message", "safe"),
        ],
    )
    if mutation == "object":
        payload["object"] = "wrong"
    elif mutation == "status":
        payload["status"] = "incomplete"
    elif mutation == "output-shape":
        payload["output"] = "wrong"
    elif mutation == "output-member":
        payload["output"] = [1]
    elif mutation == "raw-type":
        cast(list[dict[str, object]], payload["output"])[0]["type"] = 1
    elif mutation == "unknown-kind":
        cast(list[dict[str, object]], payload["output"])[0]["type"] = "new"
    elif mutation == "no-provider-rule":
        payload["output"] = [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [],
            }
        ]
    elif mutation == "provider-output-kind":
        payload["output"] = [
            {
                "type": "function_call_output",
                "call_id": "call-one",
                "output": "wrong-owner",
            }
        ]
    elif mutation == "incomplete-status":
        cast(list[dict[str, object]], payload["output"])[0][
            "status"
        ] = "in_progress"
    elif mutation == "missing-opaque":
        cast(list[dict[str, object]], payload["output"])[0].pop(
            "encrypted_content"
        )
    elif mutation == "malformed-opaque":
        cast(list[dict[str, object]], payload["output"])[0][
            "encrypted_content"
        ] = " malformed-private-opaque "
    elif mutation == "huge-opaque":
        cast(list[dict[str, object]], payload["output"])[0][
            "encrypted_content"
        ] = ("x" * 1_048_577)
    elif mutation == "missing-fields":
        cast(list[dict[str, object]], payload["output"])[1].pop("content")
    elif mutation == "bad-phase":
        cast(list[dict[str, object]], payload["output"])[1]["phase"] = "bad"
    elif mutation == "bad-reasoning":
        payload["reasoning"] = "wrong"
    elif mutation == "bad-context":
        payload["reasoning"] = {"context": "unsupported"}
    elif mutation == "numeric-context":
        payload["reasoning"] = {"context": 1}
    elif mutation == "missing-usage":
        payload["usage"] = None
    else:
        cast(dict[str, object], payload["usage"])["input_tokens"] = -1
    with pytest.raises(conversation.ConversationError) as exc:
        _provider_result(payload)
    assert exc.value.__cause__ is None
    assert "malformed-private" not in repr(exc.value)


def test_supported_mixed_items_keep_order_and_private_state() -> None:
    """Normalize mixed assistant, compaction, and final items exactly."""
    output: list[dict[str, object]] = [
        {
            "id": "file-search-one",
            "type": "file_search_call",
            "status": "completed",
            "queries": ["safe query"],
        },
        {
            "id": "web-search-one",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "query": "safe query"},
        },
        {
            "id": "compaction-one",
            "type": "compaction",
            "encrypted_content": "compaction-private",
        },
        {
            **_message("commentary-one", "working"),
            "phase": "commentary",
        },
        _message("final-one", "finished"),
    ]
    result = _provider_result(_response("mixed-response", output))
    assert [item.kind.value for item in result.items] == [
        "file_search_call",
        "web_search_call",
        "compaction",
        "message",
        "message",
    ]
    assert [item.phase.value for item in result.items] == [
        "assistant",
        "assistant",
        "compaction",
        "assistant",
        "final",
    ]
    assert result.items[2].opaque_state is not None
    assert "compaction-private" not in repr(result.items)
    assert "file-search-one" not in repr(result.items)


def test_duplicate_and_orphaned_items_fail_before_checkpoint() -> None:
    """Reject duplicate provider IDs and unresolved call correlations."""
    binding = _binding(lane_id="lane-ledger-validation")
    duplicate = _provider_result(
        _response(
            "duplicate-response",
            [
                _reasoning("duplicate-id", "duplicate-private"),
                _message("duplicate-id", "safe"),
            ],
        ),
        binding,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=(
                conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
            ),
            items=duplicate.items,
        )
    orphaned = _provider_result(
        _response(
            "orphan-response",
            [_function_call("orphan-call", "orphan-call-id")],
        ),
        binding,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=(
                conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
            ),
            items=orphaned.items,
        )
