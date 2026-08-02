"""Exercise closed Phase 6 provider and lifecycle validation boundaries."""

from asyncio import CancelledError, Event, create_task
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from typing import cast

import httpx
import pytest
from native_openai_stored_provider_test import (
    _binding,
    _capabilities,
    _direct_client,
    _execution,
    _function_call,
    _message,
    _profile,
    _provider,
    _response,
)
from openai import APIConnectionError, APIResponseValidationError, AsyncOpenAI
from phase2_fixtures import authority, request, root_identity
from store_conformance_test import _stored_atomic_commit

import avalan
import avalan.conversation as conversation
from avalan.conversation import coordinator as coordinator_module
from avalan.conversation import protocols as protocols_module
from avalan.conversation import sdk as sdk_module
from avalan.conversation import state as state_module
from avalan.conversation.providers import openai_stored as provider_module
from avalan.types import JsonValue

pytestmark = pytest.mark.anyio

_NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    """Run Phase 6 validation under asyncio only."""
    return "asyncio"


async def _unused_handler(request: httpx.Request) -> httpx.Response:
    await request.aread()
    raise AssertionError("provider dispatch was not expected")


def _plan(
    binding: conversation.ProviderLaneBinding,
    *,
    reasoning: avalan.ReasoningContext = avalan.ReasoningContext.AUTO,
) -> conversation.FirstStoredProviderPlan:
    return conversation.FirstStoredProviderPlan(
        binding=binding,
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=reasoning,
            effective=None,
        ),
        new_input={"text": "phase 6 input"},
    )


def test_stored_input_freeze_rejects_non_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a frozen provider input that loses its object shape."""
    monkeypatch.setattr(
        protocols_module,
        "freeze_json_value",
        lambda value: (value,),
    )
    with pytest.raises(conversation.ConversationValidationError):
        _plan(_binding(lane_id="lane-freeze-validation"))


def test_ambiguity_reconciliation_values_reject_malformed_types() -> None:
    """Reject forged ambiguity decisions before durable store access."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AmbiguousDispatchReconciliationRequest(
            authority=cast(conversation.AuthorityScope, object()),
            operation=conversation.ConversationOperation.CREATE,
            idempotency_key=conversation.RequestIdempotencyKey(
                "phase6-invalid-reconciliation"
            ),
            resolution=(conversation.AmbiguousDispatchResolution.RETAIN_FENCE),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AmbiguousDispatchReconciliationResult(
            disposition=cast(
                conversation.AmbiguousDispatchReconciliationDisposition,
                object(),
            )
        )


def test_stored_and_stateless_parent_shapes_never_mix(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject every parent handle placed on the opposite mode axis."""
    record_property(
        "conversation_acceptance_evidence", "pre_dispatch_rejection"
    )
    stateless_handle = avalan.StatelessConversationHandle(
        conversation_id=conversation.ConversationId("stateless-conversation"),
        checkpoint_id=conversation.CheckpointId("stateless-checkpoint"),
        branch_id=conversation.ConversationBranchId("stateless-branch"),
    )
    stored_handle = avalan.StoredConversationHandle(
        conversation_id=conversation.ConversationId("stored-conversation"),
        checkpoint_id=conversation.CheckpointId("stored-checkpoint"),
        branch_id=conversation.ConversationBranchId("stored-branch"),
        public_response_id=conversation.PublicResponseId("stored-response"),
    )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StoredParent(
            handle=cast(avalan.StoredConversationHandle, stateless_handle)
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StatelessParent(
            handle=cast(avalan.StatelessConversationHandle, stored_handle)
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StoredConversationSettings(
            provider_storage_disclosed=True,
            parent=cast(
                avalan.StoredParent,
                avalan.StatelessParent(handle=stateless_handle),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        avalan.StatelessConversationSettings(
            parent=cast(
                avalan.StatelessParent,
                avalan.StoredParent(handle=stored_handle),
            )
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("max_output_tokens", 0),
        ("max_tool_calls", 0),
        ("parallel_tool_calls", 1),
        ("temperature", 3.0),
        ("top_p", 0.0),
        ("truncation", "forged"),
    ),
)
def test_stored_execution_rejects_unfrozen_shapes(
    field: str,
    value: object,
) -> None:
    """Reject every malformed frozen execution-definition field."""
    values: dict[str, object] = {
        "instructions": "phase 6",
        "max_output_tokens": 10,
        "max_tool_calls": 2,
        "parallel_tool_calls": False,
        "temperature": 0.5,
        "top_p": 0.9,
        "truncation": "disabled",
    }
    values[field] = value
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIStoredExecution(**values)


def test_unknown_upstream_retention_keeps_independent_local_bounds() -> None:
    """Apply every known local bound without inventing upstream lifetime."""
    storage = conversation.StoragePolicy(
        local=conversation.LocalResponseStorage.DURABLE,
        upstream=conversation.ProviderLaneStorage.STORED,
        provider_storage_disclosed=True,
    )
    cases = (
        (None, None, None),
        (60, None, 60),
        (None, 30, 30),
        (60, 30, 30),
        (30, 60, 30),
    )
    for local, envelope, expected in cases:
        limits = conversation.RetentionLimits(
            storage=storage,
            upstream_lifetime_status=(
                conversation.UpstreamLifetimeStatus.UNKNOWN
            ),
            local_ttl_seconds=local,
            envelope_ttl_seconds=envelope,
        )
        assert limits.effective_ttl_seconds == expected


def test_stored_profile_rejects_unproven_provider_forms() -> None:
    """Reject generic, revision-drifted, and policy-mismatched profiles."""
    openai_binding = _binding(lane_id="lane-profile-validation")
    with pytest.raises(conversation.ConversationBindingDriftError):
        conversation.NativeOpenAIStoredProfile(
            profile_id="missing-execution-digest",
            binding=replace(
                openai_binding,
                execution_definition_digest=None,
            ),
            execution=_execution(),
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        )

    with pytest.raises(conversation.ConversationValidationError):
        conversation.native_openai_stored_execution_digest(
            binding=cast(conversation.ProviderLaneBinding, object()),
            execution=_execution(),
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.native_openai_stored_execution_digest(
            binding=openai_binding,
            execution=cast(
                conversation.NativeOpenAIStoredExecution,
                object(),
            ),
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.native_openai_stored_execution_digest(
            binding=openai_binding,
            execution=_execution(),
            encrypted_content=cast(
                conversation.NativeOpenAIEncryptedContentPolicy,
                object(),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.native_openai_stored_execution_digest(
            binding=openai_binding,
            execution=_execution(),
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
            tools=cast(
                tuple[conversation.NativeOpenAIFunctionTool, ...],
                (object(),),
            ),
        )
    invalid_profiles = (
        {
            "binding": replace(
                openai_binding,
                provider_family=conversation.ProviderFamily.OPENAI_COMPATIBLE,
            ),
            "encrypted_content": (
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        },
        {
            "binding": replace(
                openai_binding,
                sdk_revision=conversation.ProviderSdkRevision("unknown-sdk"),
            ),
            "encrypted_content": (
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        },
        {
            "binding": replace(
                openai_binding,
                provider_api_revision=conversation.ProviderApiRevision(
                    "unknown-api"
                ),
            ),
            "encrypted_content": (
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        },
        {
            "binding": openai_binding,
            "encrypted_content": (
                conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
            ),
        },
    )
    for index, values in enumerate(invalid_profiles):
        with pytest.raises(conversation.ConversationCapabilityError):
            conversation.NativeOpenAIStoredProfile(
                profile_id=f"invalid-openai-{index}",
                execution=_execution(),
                **values,
            )
    azure = _binding(
        family=conversation.ProviderFamily.AZURE_OPENAI,
        endpoint="https://resource.openai.azure.com/openai/v1",
        lane_id="lane-azure-profile-validation",
    )
    for revision, policy in (
        (
            "unsupported-azure-form",
            conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE,
        ),
        (
            "azure-openai-v1",
            conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN,
        ),
    ):
        with pytest.raises(conversation.ConversationCapabilityError):
            conversation.NativeOpenAIStoredProfile(
                profile_id=f"invalid-azure-{revision}",
                binding=replace(
                    azure,
                    provider_api_revision=conversation.ProviderApiRevision(
                        revision
                    ),
                ),
                execution=_execution(),
                encrypted_content=policy,
            )

    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIStoredProfile(
            profile_id="invalid-shape",
            binding=openai_binding,
            execution=cast(
                conversation.NativeOpenAIStoredExecution,
                object(),
            ),
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        )


def test_public_and_upstream_aliases_fail_closed_in_runtime_and_sdk() -> None:
    """Reject one private provider ID reused as the public response ID."""
    commit = _stored_atomic_commit("phase6-runtime-alias")
    upstream_response_id = commit.output_candidates[0].upstream_response_id
    assert upstream_response_id is not None
    alias = conversation.PublicResponseId(str(upstream_response_id))
    assert (
        type(commit.candidate) is conversation.OutwardTurnCheckpointCandidate
    )
    object.__setattr__(commit.candidate, "public_response_id", alias)
    with pytest.raises(conversation.ConversationValidationError):
        replace(commit, public_response_id=alias)

    result = object.__new__(conversation.ConversationResult)
    object.__setattr__(result, "public_response_id", alias)
    receipt = object.__new__(conversation.AtomicCommitReceipt)
    object.__setattr__(receipt, "checkpoint", commit.candidate.checkpoint)
    object.__setattr__(receipt, "result", result)
    object.__setattr__(receipt, "output_candidates", commit.output_candidates)
    with pytest.raises(conversation.ConversationValidationError):
        sdk_module._direct_result(receipt)


async def test_checkpoint_alias_fails_codec_atomic_store_and_sdk() -> None:
    """Reject a private response ID forged as a checkpoint identifier."""
    commit = _stored_atomic_commit("phase6-checkpoint-alias")
    durable_codec = conversation.DurableConversationCodec()
    output = commit.output_candidates[0]
    assert (
        durable_codec.decode_output(durable_codec.encode_output(output))
        == output
    )
    checkpoint = commit.candidate.checkpoint
    lane = checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
    object.__setattr__(
        lane,
        "upstream_response_id",
        conversation.UpstreamResponseId(
            str(checkpoint.identity.checkpoint_id)
        ),
    )

    with pytest.raises(conversation.ConversationCodecError):
        conversation.ConversationCheckpointCodec().encode(checkpoint)
    with pytest.raises(conversation.ConversationValidationError):
        replace(commit)
    store = conversation.InMemoryConversationStore()
    with pytest.raises(conversation.ConversationValidationError):
        await store.commit_atomic(commit)

    result = conversation.InMemoryConversationStore._build_result(
        commit,
        checkpoint,
    )
    assert result is not None
    receipt = object.__new__(conversation.AtomicCommitReceipt)
    object.__setattr__(receipt, "checkpoint", checkpoint)
    object.__setattr__(receipt, "result", result)
    object.__setattr__(receipt, "output_candidates", commit.output_candidates)
    with pytest.raises(conversation.ConversationValidationError):
        sdk_module._direct_result(receipt)
    with pytest.raises(conversation.ConversationValidationError):
        state_module.validate_upstream_identifier_separation(
            cast(conversation.ConversationCheckpoint, object())
        )


async def test_stored_provider_constructor_and_runtime_are_closed() -> None:
    """Reject malformed providers, duplicate tools, and runtime limits."""
    binding = _binding(lane_id="lane-provider-constructor")
    client = AsyncOpenAI(
        api_key="phase6-key",
        base_url=binding.normalized_endpoint,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_unused_handler)
        ),
        max_retries=0,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIStoredProvider(
            client=client,
            profile=cast(conversation.NativeOpenAIStoredProfile, object()),
            capability_profile=_capabilities(binding),
        )

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        return str(arguments)

    tool = conversation.NativeOpenAIFunctionTool(
        name="duplicate",
        parameters={"type": "object"},
        handler=tool_handler,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.NativeOpenAIStoredProvider(
            client=client,
            profile=_profile(binding),
            capability_profile=_capabilities(binding),
            tools=(tool, tool),
        )

    provider = _provider(binding, _unused_handler)
    for values in (
        {"provider": cast(conversation.NativeOpenAIStoredProvider, object())},
        {"provider": provider, "max_output_items": 0},
        {"provider": provider, "max_output_bytes": 0},
        {"provider": provider, "max_output_segments": 0},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.NativeOpenAIStoredLaneRuntime(**values)

    runtime = conversation.NativeOpenAIStoredLaneRuntime(provider=provider)
    assert (
        coordinator_module._validate_stored_native_lane_runtime(runtime)
        is runtime
    )
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_stored_native_lane_runtime(object())
    missing = object.__new__(conversation.NativeOpenAIStoredLaneRuntime)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_stored_native_lane_runtime(missing)
    object.__setattr__(runtime, "max_output_items", 0)
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_stored_native_lane_runtime(runtime)
    await provider.aclose()
    await client.close()


async def test_stored_coordinator_runtime_and_plan_boundaries_are_closed() -> (
    None
):
    """Close stored runtime selection, diagnostics, and plan construction."""
    binding = _binding(lane_id="lane-stored-coordinator-validation")
    provider = _provider(binding, _unused_handler)
    client, coordinator, _ = _direct_client(
        provider,
        namespace="stored-coordinator-validation",
    )
    runtime = next(iter(coordinator._lanes.values()))
    assert type(runtime) is conversation.NativeOpenAIStoredLaneRuntime
    assert coordinator_module._validate_any_native_lane_runtime(runtime) is (
        runtime
    )
    with pytest.raises(conversation.ConversationValidationError):
        coordinator_module._validate_any_native_lane_runtime(object())
    assert (
        coordinator.native_provider_diagnostics(binding.lane_id)
        == provider.diagnostics
    )

    run = request(
        scope=authority(),
        identity=root_identity("stored-coordinator-validation"),
        advance=conversation.FirstTurnAdvance(),
        lane_ids=(str(binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        key="stored-coordinator-validation",
        response_suffix="stored-coordinator-validation",
    )
    malformed = replace(run.semantics, semantic_input="not-a-mapping")
    with pytest.raises(conversation.ConversationValidationError):
        coordinator._plan_lanes(
            replace(run, semantics=malformed),
            None,
            streaming=False,
        )
    with pytest.raises(conversation.ConversationCapabilityError):
        await coordinator._dispatch_complete_lane(
            runtime,
            cast(conversation.ProviderPlan, object()),
            streaming=False,
            progress=coordinator_module._DispatchProgress(),
            sink=None,
        )

    original_binding = provider._profile.binding
    original_capabilities = provider._capability_profile
    synthetic = replace(
        original_binding,
        provider_family=conversation.ProviderFamily.SYNTHETIC,
    )
    object.__setattr__(provider._profile, "binding", synthetic)
    object.__setattr__(
        provider,
        "_capability_profile",
        replace(
            original_capabilities,
            binding_alias=synthetic.safe_alias,
        ),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        coordinator_module._validate_stored_native_lane_runtime(runtime)
    object.__setattr__(provider._profile, "binding", original_binding)
    object.__setattr__(provider, "_capability_profile", original_capabilities)
    await coordinator.close()
    assert client._runtime.coordinator is coordinator


async def test_direct_stored_runtime_and_conversion_shapes_are_closed() -> (
    None
):
    """Require paired exact lifecycle wiring and typed conversion inputs."""
    binding = _binding(lane_id="lane-direct-runtime-validation")
    provider = _provider(binding, _unused_handler)
    client, coordinator, _ = _direct_client(
        provider,
        namespace="direct-runtime-validation",
        lifecycle=True,
    )
    runtime = client._runtime
    assert runtime.provider_resolver is not None
    assert runtime.lifecycle_reconciler is not None
    for values in (
        {"provider_resolver": None},
        {
            "provider_resolver": cast(
                conversation.StoredProviderResolver,
                object(),
            )
        },
        {
            "lifecycle_reconciler": cast(
                conversation.ProviderLifecycleReconciler,
                object(),
            )
        },
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(runtime, **values)
    resolver_only = replace(runtime, lifecycle_reconciler=None)
    assert resolver_only.provider_resolver is runtime.provider_resolver
    with pytest.raises(conversation.ConversationValidationError):
        await client.convert(
            "invalid conversion",
            cast(conversation.ConversationModeConversion, object()),
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
    with pytest.raises(conversation.ConversationValidationError):
        sdk_module._direct_result_from_resource(
            cast(conversation.ConversationResult, object()),
            cast(conversation.ConversationCheckpoint, object()),
        )
    await coordinator.close()


async def test_stored_plan_and_capabilities_fail_before_dispatch() -> None:
    """Reject mode, transport, capability, binding, and forged input drift."""
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        return await _unused_handler(request)

    binding = _binding(lane_id="lane-plan-validation")
    provider = _provider(binding, handler)
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.dispatch(cast(conversation.ProviderPlan, object()))
    with pytest.raises(conversation.ConversationBindingDriftError):
        await provider.stream(_plan(binding))
    with pytest.raises(conversation.ConversationBindingDriftError):
        await provider.dispatch(
            replace(
                _plan(binding),
                binding=_binding(lane_id="lane-plan-drift"),
            )
        )
    for forged in (
        None,
        {"text": "safe", "store": False},
        {"text": ""},
        {"items": ()},
        {"items": ("not-a-mapping",)},
        {"items": ({"type": "message"},)},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            await provider.dispatch(replace(_plan(binding), new_input=forged))
    assert dispatches == 0
    await provider.aclose()

    for capability, reasoning in (
        (
            conversation.ConversationCapability.STORED_RESPONSES_CHAINING,
            avalan.ReasoningContext.AUTO,
        ),
        (
            conversation.ConversationCapability.REASONING_CONTEXT_CURRENT_TURN,
            avalan.ReasoningContext.CURRENT_TURN,
        ),
        (
            conversation.ConversationCapability.REASONING_CONTEXT_ALL_TURNS,
            avalan.ReasoningContext.ALL_TURNS,
        ),
    ):
        incapable = _provider(
            _binding(lane_id=f"lane-incapable-{capability.value}"),
            handler,
            capabilities=_capabilities(
                _binding(lane_id=f"lane-incapable-{capability.value}"),
                exclude=capability,
            ),
        )
        with pytest.raises(conversation.ConversationCapabilityError):
            await incapable.dispatch(
                _plan(incapable.binding, reasoning=reasoning)
            )
        await incapable.aclose()

    stream_binding = _binding(
        streaming=True,
        lane_id="lane-incapable-stream-fidelity",
    )
    incapable_stream = _provider(
        stream_binding,
        handler,
        capabilities=_capabilities(
            stream_binding,
            exclude=(
                conversation.ConversationCapability.STREAMING_ITEM_FIDELITY
            ),
        ),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await incapable_stream.stream(_plan(stream_binding))
    await incapable_stream.aclose()
    assert dispatches == 0


@pytest.mark.parametrize(
    "store,previous_response_id",
    ((False, None), (True, "non-immediate-parent")),
)
async def test_stored_response_envelope_requires_exact_parent_and_store(
    store: bool,
    previous_response_id: str | None,
) -> None:
    """Reject provider responses that do not attest the exact stored chain."""
    binding = _binding(lane_id=f"lane-envelope-{store}")
    response = _response(
        "private-envelope-response",
        [_message("envelope-message", "visible")],
        previous_response_id=previous_response_id,
    )
    response["store"] = store

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(200, json=response)

    provider = _provider(binding, handler)
    with pytest.raises(conversation.ConversationProviderResponseError):
        await provider.dispatch(_plan(binding))
    await provider.aclose()


async def test_stored_sdk_failure_classes_are_typed_and_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map status, validation, ambiguity, cancellation, and union failures."""

    async def rejected(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            400,
            json={"error": {"message": "private-status", "type": "bad"}},
        )

    rejected_provider = _provider(
        _binding(lane_id="lane-stored-status"), rejected
    )
    with pytest.raises(conversation.ConversationError) as status:
        await rejected_provider.dispatch(_plan(rejected_provider.binding))
    assert (
        status.value.boundary
        is conversation.FailureBoundary.PROVIDER_REJECTION
    )
    assert "private-status" not in repr(status.value)
    await rejected_provider.aclose()

    async def validation_error(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        request = httpx.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx.Response(200, request=request)
        raise APIResponseValidationError(
            response=response,
            body={"private": "private-validation"},
            message="private-validation",
        )

    provider = _provider(
        _binding(lane_id="lane-stored-sdk-failures"), _unused_handler
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            provider_module,
            "_create_exact_stored_response",
            validation_error,
        )
        with pytest.raises(conversation.ConversationProviderResponseError):
            await provider.dispatch(_plan(provider.binding))

    async def generic_error(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        raise RuntimeError("private-generic")

    with monkeypatch.context() as patch:
        patch.setattr(
            provider_module,
            "_create_exact_stored_response",
            generic_error,
        )
        with pytest.raises(conversation.ConversationAmbiguousDispatchError):
            await provider.dispatch(_plan(provider.binding))

    async def connection_error(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        raise APIConnectionError(
            request=httpx.Request(
                "POST", "https://api.openai.com/v1/responses"
            )
        )

    with monkeypatch.context() as patch:
        patch.setattr(
            provider_module,
            "_create_exact_stored_response",
            connection_error,
        )
        with pytest.raises(conversation.ConversationAmbiguousDispatchError):
            await provider.dispatch(_plan(provider.binding))

    async def cancel(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        raise CancelledError()

    with monkeypatch.context() as patch:
        patch.setattr(
            provider_module,
            "_create_exact_stored_response",
            cancel,
        )
        with pytest.raises(CancelledError):
            await provider.dispatch(_plan(provider.binding))

    async def wrong_union(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        return object()

    with monkeypatch.context() as patch:
        patch.setattr(
            provider_module,
            "_create_exact_stored_response",
            wrong_union,
        )
        with pytest.raises(conversation.ConversationProviderResponseError):
            await provider.dispatch(_plan(provider.binding))
    await provider.aclose()

    stream_binding = _binding(
        streaming=True,
        lane_id="lane-stored-wrong-stream-union",
    )
    stream_provider = _provider(stream_binding, _unused_handler)
    with monkeypatch.context() as patch:
        patch.setattr(
            provider_module,
            "_create_exact_stored_response",
            wrong_union,
        )
        with pytest.raises(conversation.ConversationProviderResponseError):
            await stream_provider.stream(_plan(stream_binding))
    await stream_provider.aclose()


@pytest.mark.parametrize(
    "reasoning",
    (
        avalan.ReasoningContext.CURRENT_TURN,
        avalan.ReasoningContext.ALL_TURNS,
    ),
)
@pytest.mark.parametrize("azure", (False, True), ids=("openai", "azure"))
@pytest.mark.parametrize("streaming", (False, True), ids=("sync", "stream"))
async def test_stored_reasoning_context_is_sent_explicitly(
    reasoning: avalan.ReasoningContext,
    azure: bool,
    streaming: bool,
    record_property: Callable[[str, object], None],
) -> None:
    """Map each exact profile's proven reasoning scope to its request."""
    record_property("conversation_acceptance_evidence", "contract")
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], loads(await request.aread()))
        requests.append(payload)
        response = _response(
            "private-reasoning",
            [_message("reasoning-message", "visible")],
        )
        response["model"] = "deployment-stored" if azure else "gpt-5"
        response["reasoning"] = {"context": reasoning.value}
        if not streaming:
            return httpx.Response(200, json=response)
        output = cast(list[dict[str, object]], response["output"])
        events = (
            {
                "type": "response.output_item.done",
                "sequence_number": 0,
                "output_index": 0,
                "item": output[0],
            },
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": response,
            },
        )
        body = (
            "".join(f"data: {dumps(event)}\n\n" for event in events)
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    family = (
        conversation.ProviderFamily.AZURE_OPENAI
        if azure
        else conversation.ProviderFamily.OPENAI
    )
    binding = _binding(
        streaming=streaming,
        family=family,
        endpoint=(
            "https://resource.openai.azure.com/openai/v1"
            if azure
            else "https://api.openai.com/v1"
        ),
        lane_id=(
            f"lane-reasoning-{'azure' if azure else 'openai'}-"
            f"{'stream' if streaming else 'sync'}-{reasoning.value}"
        ),
    )
    provider = _provider(binding, handler)
    plan = _plan(binding, reasoning=reasoning)
    if streaming:
        stream = await provider.stream(plan)
        assert [item async for item in stream]
        await stream.terminal()
        await stream.aclose()
    else:
        await provider.dispatch(plan)
    assert requests[0]["reasoning"] == {"context": reasoning.value}
    assert (requests[0].get("include") == ["reasoning.encrypted_content"]) is (
        azure
    )
    assert requests[0]["stream"] is streaming
    await provider.aclose()


async def test_stored_reasoning_summary_is_independent_from_opaque_state(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep display summary selection separate from retained opaque bytes."""
    record_property("conversation_acceptance_evidence", "contract")
    response = _response(
        "private-reasoning-summary",
        [
            {
                "id": "reasoning-summary-item",
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "displayable summary",
                    }
                ],
                "encrypted_content": "private-reasoning-bytes",
            },
            _message("reasoning-summary-message", "visible"),
        ],
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(200, json=response)

    binding = _binding(lane_id="lane-reasoning-summary")
    provider = _provider(binding, handler)
    result = await provider.dispatch(
        _plan(binding, reasoning=avalan.ReasoningContext.ALL_TURNS)
    )
    reasoning = next(
        item
        for item in result.items
        if item.kind is conversation.ProviderItemKind.REASONING
    )
    assert reasoning.opaque_state is not None
    original_digest = reasoning.opaque_state.digest
    changed_input = dict(reasoning.canonical_input)
    changed_input["summary"] = (
        {"type": "summary_text", "text": "different display summary"},
    )
    changed = replace(reasoning, canonical_input=changed_input)
    assert changed.opaque_state is reasoning.opaque_state
    assert changed.opaque_state.digest == original_digest
    assert result.upstream_response_id == "private-reasoning-summary"
    assert "private-reasoning-bytes" not in repr(changed)
    await provider.aclose()


async def test_stored_tool_execution_is_exact() -> None:
    """Reject wrong caller, lane, shape, and unknown stored tools."""
    binding = _binding(lane_id="lane-stored-tool-validation")

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments == {"value": 1}
        return "ok"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )
    binding = replace(
        binding,
        execution_definition_digest=(
            conversation.native_openai_stored_execution_digest(
                binding=binding,
                execution=_execution(),
                encrypted_content=(
                    conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
                ),
                tools=(tool,),
            )
        ),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            json=_response(
                "private-tool-validation",
                [_function_call("tool-validation", "call-validation")],
            ),
        )

    provider = _provider(binding, handler, tools=(tool,))
    result = await provider.dispatch(_plan(binding))
    assert "private-tool-validation" not in repr(result)
    call = result.items[0]
    assert await provider.execute_tool(call) == "ok"
    with pytest.raises(conversation.ConversationValidationError):
        await provider.execute_tool(cast(conversation.ProviderItem, object()))
    mutations = (
        ("caller", conversation.ProviderItemCaller.TOOL),
        ("lane_id", conversation.ProviderLaneId("wrong-lane")),
        ("kind", conversation.ProviderItemKind.MESSAGE),
        ("canonical_input", {"name": 1, "arguments": 2}),
    )
    for field, malformed in mutations:
        original = getattr(call, field)
        object.__setattr__(call, field, malformed)
        with pytest.raises(conversation.ConversationValidationError):
            await provider.execute_tool(call)
        object.__setattr__(call, field, original)
    unknown_input = dict(call.canonical_input)
    unknown_input["name"] = "missing"
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.execute_tool(
            replace(call, canonical_input=unknown_input)
        )
    await provider.aclose()


@pytest.mark.parametrize(
    "boundary",
    ("segments", "provider_items", "reserved_tool_item"),
)
async def test_stored_coordinator_limits_precede_commit_and_tool_effects(
    boundary: str,
) -> None:
    """Enforce each stored segment and item bound before commit."""
    dispatches = 0
    tool_effects = 0

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

    async def handler(request_value: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        await request_value.aread()
        dispatches += 1
        output = (
            [
                _message("bounded-message-one", "one"),
                _message("bounded-message-two", "two"),
            ]
            if boundary == "provider_items"
            else [_function_call("bounded-call", "bounded-call-id")]
        )
        return httpx.Response(
            200,
            json=_response(f"private-bounded-{dispatches}", output),
        )

    binding = _binding(
        lane_id=f"lane-stored-bound-{boundary}",
        tools=(tool,),
    )
    provider = _provider(binding, handler, tools=(tool,))
    client, coordinator, store = _direct_client(
        provider,
        namespace=f"stored-bound-{boundary}",
    )
    runtime = next(iter(coordinator._lanes.values()))
    assert type(runtime) is conversation.NativeOpenAIStoredLaneRuntime
    if boundary == "segments":
        object.__setattr__(runtime, "max_output_segments", 1)
    else:
        object.__setattr__(runtime, "max_output_items", 1)

    with pytest.raises(conversation.ConversationLimitError):
        await client.create(
            "bounded stored request",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
    assert dispatches == 1
    assert tool_effects == (1 if boundary == "segments" else 0)
    page = await store.list_checkpoints(authority(), cursor=None, limit=10)
    assert len(page.checkpoints) == 1
    assert str(page.checkpoints[0].identity.checkpoint_id).startswith(
        "quarantine-"
    )
    assert (
        len(await store.claim_provider_lifecycle(authority(), limit=10)) == 1
    )
    await coordinator.close()


async def test_stored_segment_identity_and_sequence_are_closed() -> None:
    """Reject missing IDs, forged positions, and duplicate provider items."""
    binding = _binding(lane_id="lane-stored-segment-validation")
    plan = _plan(binding)

    async def handler(request_value: httpx.Request) -> httpx.Response:
        await request_value.aread()
        return httpx.Response(
            200,
            json=_response(
                "private-segment-validation",
                [
                    _message("segment-message-one", "one"),
                    _message("segment-message-two", "two"),
                ],
            ),
        )

    provider = _provider(binding, handler)
    result = await provider.dispatch(plan)
    validator = (
        conversation.RunScopedConversationCoordinator._validate_native_stored_provider_segment
    )
    validator(plan, (), result)

    with pytest.raises(conversation.ConversationProviderResponseError):
        validator(plan, (), replace(result, upstream_response_id=None))

    forged_position = replace(
        result.items[1],
        model_call_id=conversation.ConversationModelCallId(
            "forged-model-call"
        ),
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        validator(
            plan,
            (),
            replace(
                result,
                items=(result.items[0], forged_position),
            ),
        )

    duplicate_input = dict(result.items[1].canonical_input)
    duplicate_input["id"] = result.items[0].item_id
    duplicate = replace(
        result.items[1],
        item_id=result.items[0].item_id,
        canonical_input=duplicate_input,
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        validator(
            plan,
            (),
            replace(result, items=(result.items[0], duplicate)),
        )
    await provider.aclose()


@pytest.mark.parametrize(
    "boundary",
    (
        "provider_pending",
        "tombstone_deleted_race",
        "tombstone_active_race",
        "delete_deleted_race",
        "delete_tombstoned_race",
    ),
)
async def test_direct_stored_deletion_races_are_constant_disclosure(
    boundary: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve local deletion races without exposing private state."""

    async def handler(request_value: httpx.Request) -> httpx.Response:
        if request_value.method == "POST":
            await request_value.aread()
            return httpx.Response(
                200,
                json=_response(
                    f"private-delete-{boundary}",
                    [_message(f"delete-message-{boundary}", "retained")],
                ),
            )
        assert request_value.method == "DELETE"
        if boundary == "provider_pending":
            return httpx.Response(
                500,
                json={"error": {"message": "temporary outage"}},
            )
        return httpx.Response(204)

    binding = _binding(lane_id=f"lane-delete-{boundary}")
    provider = _provider(binding, handler)
    client, coordinator, store = _direct_client(
        provider,
        namespace=f"delete-{boundary}",
        lifecycle=True,
    )
    created = await client.create(
        "delete race",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(created.handle) is avalan.StoredConversationHandle
    public_id = created.handle.public_response_id
    assert public_id is not None

    original_tombstone = store.tombstone
    original_delete = store.delete
    if boundary == "tombstone_deleted_race":

        async def tombstone_deleted(
            response_id: conversation.PublicResponseId,
            scope: conversation.AuthorityScope,
            at: datetime,
        ) -> conversation.ConversationCheckpoint:
            await original_tombstone(response_id, scope, at)
            work = await store.claim_provider_lifecycle(scope, limit=1)
            assert len(work) == 1
            await store.acknowledge_provider_lifecycle(
                work[0],
                succeeded=True,
            )
            await original_delete(response_id, scope, at)
            raise conversation.ConversationConflictError()

        monkeypatch.setattr(store, "tombstone", tombstone_deleted)
    elif boundary == "tombstone_active_race":

        async def tombstone_active(
            response_id: conversation.PublicResponseId,
            scope: conversation.AuthorityScope,
            at: datetime,
        ) -> conversation.ConversationCheckpoint:
            assert response_id == public_id
            assert scope == authority()
            assert at.tzinfo is not None
            raise conversation.ConversationConflictError()

        monkeypatch.setattr(store, "tombstone", tombstone_active)
    elif boundary == "delete_deleted_race":

        async def delete_deleted(
            response_id: conversation.PublicResponseId,
            scope: conversation.AuthorityScope,
            at: datetime,
        ) -> None:
            await original_delete(response_id, scope, at)
            raise conversation.ConversationAuthorizationError()

        monkeypatch.setattr(store, "delete", delete_deleted)
    elif boundary == "delete_tombstoned_race":

        async def delete_tombstoned(
            response_id: conversation.PublicResponseId,
            scope: conversation.AuthorityScope,
            at: datetime,
        ) -> None:
            assert response_id == public_id
            assert scope == authority()
            assert at.tzinfo is not None
            raise conversation.ConversationAuthorizationError()

        monkeypatch.setattr(store, "delete", delete_tombstoned)

    if boundary in {"tombstone_active_race", "delete_tombstoned_race"}:
        expected = (
            conversation.ConversationConflictError
            if boundary == "tombstone_active_race"
            else conversation.ConversationAuthorizationError
        )
        with pytest.raises(expected):
            await client.delete(public_id)
    else:
        deletion = await client.delete(public_id)
        assert deletion.local_tombstoned
        assert deletion.upstream_pending is (boundary == "provider_pending")
    await coordinator.close()


async def test_direct_sdk_reconciliation_and_retrieval_aliases_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed fence decisions and private/public ID aliasing."""

    async def handler(request_value: httpx.Request) -> httpx.Response:
        await request_value.aread()
        return httpx.Response(
            200,
            json=_response(
                "private-sdk-boundary",
                [_message("sdk-boundary-message", "retained")],
            ),
        )

    provider = _provider(
        _binding(lane_id="lane-sdk-boundary"),
        handler,
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace="sdk-boundary",
        lifecycle=True,
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.reconcile_ambiguous_dispatch(
            cast(conversation.ConversationOperation, object()),
            conversation.RequestIdempotencyKey("sdk-invalid-operation"),
            conversation.AmbiguousDispatchResolution.RETAIN_FENCE,
        )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await client.reconcile_ambiguous_dispatch(
            conversation.ConversationOperation.CREATE,
            conversation.RequestIdempotencyKey("sdk-unknown-fence"),
            conversation.AmbiguousDispatchResolution.RETAIN_FENCE,
        )

    created = await client.create(
        "stored alias",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(created.handle) is avalan.StoredConversationHandle
    public_response_id = created.handle.public_response_id
    assert public_response_id is not None
    load = store.load

    async def aliased_load(
        checkpoint_id: conversation.CheckpointId,
        scope: conversation.AuthorityScope,
    ) -> conversation.ConversationCheckpoint:
        checkpoint = await load(checkpoint_id, scope)
        lane = checkpoint.content.lanes[0]
        assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
        object.__setattr__(
            lane,
            "upstream_response_id",
            conversation.UpstreamResponseId(str(public_response_id)),
        )
        return checkpoint

    monkeypatch.setattr(store, "load", aliased_load)
    with pytest.raises(conversation.ConversationValidationError):
        await client.retrieve(public_response_id)
    await coordinator.close()


@pytest.mark.parametrize(
    "drift",
    ("missing_local_execution_digest", "retrieved_execution_binding"),
)
async def test_direct_sdk_revalidates_durable_execution_binding(
    drift: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject missing local bindings and drifted retrieved metadata."""

    async def handler(request_value: httpx.Request) -> httpx.Response:
        await request_value.aread()
        return httpx.Response(
            200,
            json=_response(
                "private-sdk-execution-binding",
                [_message("sdk-execution-binding-message", "retained")],
            ),
        )

    provider = _provider(
        _binding(lane_id=f"lane-sdk-execution-{drift}"),
        handler,
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace=f"sdk-execution-{drift}",
        lifecycle=True,
    )
    created = await client.create(
        "bind retrieved execution",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(created.handle) is avalan.StoredConversationHandle
    public_response_id = created.handle.public_response_id
    assert public_response_id is not None
    resolver = client._runtime.provider_resolver
    assert resolver is not None

    adapter_binding = provider.binding
    if drift == "missing_local_execution_digest":
        load = store.load

        async def load_without_execution_digest(
            checkpoint_id: conversation.CheckpointId,
            scope: conversation.AuthorityScope,
        ) -> conversation.ConversationCheckpoint:
            checkpoint = await load(checkpoint_id, scope)
            lane = checkpoint.content.lanes[0]
            assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
            missing_digest_binding = replace(
                lane.binding,
                execution_definition_digest=None,
            )
            object.__setattr__(lane, "binding", missing_digest_binding)
            return checkpoint

        monkeypatch.setattr(store, "load", load_without_execution_digest)
        adapter_binding = replace(
            provider.binding,
            execution_definition_digest=None,
        )

    adapter = _LifecycleAdapter(adapter_binding)

    async def resolve(
        binding_digest: conversation.IntegrityDigest,
    ) -> conversation.StoredResponseLifecycleAdapter:
        assert binding_digest == adapter.binding.integrity_digest
        return adapter

    monkeypatch.setattr(resolver, "resolve", resolve)
    with pytest.raises(conversation.ConversationBindingDriftError):
        await client.retrieve(public_response_id)
    await coordinator.close()


@pytest.mark.parametrize("operation", ("retrieve", "delete"))
@pytest.mark.parametrize("failure", ("status", "connection", "malformed"))
async def test_stored_lifecycle_failures_are_typed(
    operation: str,
    failure: str,
) -> None:
    """Fail closed for each proven lifecycle transport failure class."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if failure == "connection":
            raise httpx.ConnectError("private-connect", request=request)
        if failure == "status":
            return httpx.Response(
                500,
                json={"error": {"message": "private", "type": "bad"}},
            )
        if request.method == "GET":
            return httpx.Response(200, json={"id": "wrong", "object": "list"})
        raise RuntimeError("private-delete-malformed")

    provider = _provider(
        _binding(lane_id=f"lane-lifecycle-{operation}-{failure}"),
        handler,
    )
    method = getattr(provider, operation)
    expected = (
        conversation.ConversationAmbiguousDispatchError
        if failure == "connection"
        or failure == "malformed"
        and operation == "delete"
        else conversation.ConversationProviderResponseError
    )
    with pytest.raises(expected):
        await method(conversation.UpstreamResponseId("private-target"))
    await provider.aclose()


@pytest.mark.parametrize("operation", ("retrieve", "delete"))
@pytest.mark.parametrize("failure", ("cancelled", "generic"))
async def test_stored_lifecycle_sdk_exceptions_are_concealed(
    operation: str,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conceal direct SDK lifecycle exceptions without losing cancellation."""
    provider = _provider(
        _binding(lane_id=f"lane-sdk-{operation}-{failure}"),
        _unused_handler,
    )

    async def fail(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        if failure == "cancelled":
            raise CancelledError()
        raise RuntimeError("private-lifecycle-sdk")

    with monkeypatch.context() as patch:
        patch.setattr(getattr(provider._client, "responses"), operation, fail)
        expected = (
            CancelledError
            if failure == "cancelled"
            else conversation.ConversationProviderResponseError
        )
        with pytest.raises(expected) as raised:
            await getattr(provider, operation)(
                conversation.UpstreamResponseId("private-target")
            )
        assert "private-lifecycle-sdk" not in repr(raised.value)
    await provider.aclose()


async def test_stored_provider_close_settles_all_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settle repeated, cancelled, failed, and caller-cancelled closes."""
    settled = _provider(
        _binding(lane_id="lane-close-settled"), _unused_handler
    )
    await settled.aclose()
    await settled.aclose()

    cancelled = _provider(
        _binding(lane_id="lane-close-task-cancelled"), _unused_handler
    )

    async def cancelled_close() -> None:
        raise CancelledError()

    with monkeypatch.context() as patch:
        patch.setattr(cancelled._client, "close", cancelled_close)
        with pytest.raises(CancelledError):
            await cancelled.aclose()
    await cancelled.aclose()

    failed = _provider(_binding(lane_id="lane-close-failed"), _unused_handler)

    async def failed_close() -> None:
        raise RuntimeError("private-close")

    with monkeypatch.context() as patch:
        patch.setattr(failed._client, "close", failed_close)
        with pytest.raises(conversation.ConversationCommitError) as error:
            await failed.aclose()
        assert "private-close" not in repr(error.value)
    await failed.aclose()

    for outcome in ("success", "failure"):
        provider = _provider(
            _binding(lane_id=f"lane-close-outer-cancel-{outcome}"),
            _unused_handler,
        )
        started = Event()
        release = Event()

        async def delayed_close() -> None:
            started.set()
            await release.wait()
            if outcome == "failure":
                raise RuntimeError("private-delayed-close")

        with monkeypatch.context() as patch:
            patch.setattr(provider._client, "close", delayed_close)
            task = create_task(provider.aclose())
            await started.wait()
            task.cancel()
            release.set()
            with pytest.raises(CancelledError) as cancellation:
                await task
            if outcome == "failure":
                assert isinstance(
                    cancellation.value.__cause__,
                    conversation.ConversationCommitError,
                )
        await provider.aclose()


async def test_lifecycle_capability_and_closed_provider_fail_before_wire() -> (
    None
):
    """Require proven lifecycle operations before any wire I/O."""
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        return await _unused_handler(request)

    for capability, operation in (
        (
            conversation.ConversationCapability.STORED_RESPONSE_RETRIEVAL,
            "retrieve",
        ),
        (
            conversation.ConversationCapability.STORED_RESPONSE_DELETION,
            "delete",
        ),
    ):
        binding = _binding(lane_id=f"lane-lifecycle-incapable-{operation}")
        provider = _provider(
            binding,
            handler,
            capabilities=_capabilities(binding, exclude=capability),
        )
        with pytest.raises(conversation.ConversationCapabilityError):
            await getattr(provider, operation)(
                conversation.UpstreamResponseId("private-target")
            )
        await provider.aclose()
    assert dispatches == 0

    closed = _provider(_binding(lane_id="lane-lifecycle-closed"), handler)
    await closed.aclose()
    with pytest.raises(conversation.ConversationCapabilityError):
        await closed.retrieve(
            conversation.UpstreamResponseId("private-target")
        )


def test_lifecycle_value_validation_is_closed() -> None:
    """Reject malformed lifecycle value objects."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.UpstreamRetentionMetadata(
            status=cast(conversation.UpstreamLifetimeStatus, "bad")
        )
    for values in (
        {
            "status": conversation.UpstreamLifetimeStatus.KNOWN,
            "expires_at": datetime(2026, 8, 2),
        },
        {
            "status": conversation.UpstreamLifetimeStatus.KNOWN,
            "ttl_seconds": 0,
        },
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.UpstreamRetentionMetadata(**values)
    known = conversation.UpstreamRetentionMetadata(
        status=conversation.UpstreamLifetimeStatus.KNOWN,
        ttl_seconds=60,
    )
    assert known.ttl_seconds == 60

    with pytest.raises(conversation.ConversationValidationError):
        conversation.RetrievedUpstreamResponse(
            upstream_response_id=conversation.UpstreamResponseId("private"),
            availability=cast(conversation.UpstreamAvailability, "bad"),
            retention=conversation.UpstreamRetentionMetadata.unknown(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.RetrievedUpstreamResponse(
            upstream_response_id=conversation.UpstreamResponseId("private"),
            availability=conversation.UpstreamAvailability.AVAILABLE,
            retention=cast(
                conversation.UpstreamRetentionMetadata,
                object(),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.RetrievedUpstreamResponse(
            upstream_response_id=conversation.UpstreamResponseId("private"),
            availability=conversation.UpstreamAvailability.AVAILABLE,
            retention=conversation.UpstreamRetentionMetadata.unknown(),
            effective_reasoning_context=cast(
                conversation.EffectiveReasoningContext,
                object(),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.UpstreamDeleteResult(
            disposition=cast(conversation.UpstreamDeleteDisposition, "bad")
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderQuarantineReceipt(
            checkpoint_id=conversation.CheckpointId("checkpoint"),
            target_count=0,
        )


def test_quarantine_request_requires_private_root_and_aware_time() -> None:
    """Accept only stored private root candidates as quarantine work."""
    lane_binding = _binding(lane_id="lane-quarantine-request")
    lane = conversation.StoredProviderLaneSnapshot(
        binding=lane_binding,
        upstream_response_id=conversation.UpstreamResponseId("private-child"),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=conversation.EffectiveReasoningContext.CURRENT_TURN,
        ),
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
    )

    def candidate(
        checkpoint_id: str,
    ) -> conversation.ExecutionSegmentCheckpointCandidate:
        identity = replace(
            root_identity(f"quarantine-{checkpoint_id}"),
            checkpoint_id=conversation.CheckpointId(checkpoint_id),
        )
        run = request(
            scope=authority(),
            identity=identity,
            advance=conversation.FirstTurnAdvance(),
            lane_ids=(str(lane_binding.lane_id),),
            modes=(conversation.ConversationMode.STORED,),
            stored_retention=True,
            boundary=conversation.ConversationCommitBoundary.INTERNAL_SEGMENT,
            key=f"key-{checkpoint_id}",
            response_suffix=checkpoint_id,
        )
        built = conversation.build_checkpoint_candidate(
            run,
            parent=None,
            completed_lanes=(lane,),
            created_at=_NOW,
        )
        assert type(built) is conversation.ExecutionSegmentCheckpointCandidate
        return built

    valid = candidate("quarantine-valid")
    request_value = conversation.ProviderQuarantineRequest(
        candidate=valid,
        created_at=_NOW,
    )
    assert request_value.candidate is valid
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderQuarantineRequest(
            candidate=valid,
            created_at=_NOW,
            additional_candidates=cast(
                tuple[conversation.ExecutionSegmentCheckpointCandidate, ...],
                [valid],
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderQuarantineRequest(
            candidate=valid,
            created_at=_NOW,
            additional_candidates=(valid,),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderQuarantineRequest(
            candidate=cast(
                conversation.ExecutionSegmentCheckpointCandidate,
                object(),
            ),
            created_at=_NOW,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderQuarantineRequest(
            candidate=candidate("ordinary-private-boundary"),
            created_at=_NOW,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderQuarantineRequest(
            candidate=valid,
            created_at=datetime(2026, 8, 2),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"origin": "bad"},
        {"attempts": -1},
        {
            "state": conversation.ProviderLifecycleWorkState.CLAIMED,
            "lease_owner": None,
            "lease_expires_at": None,
        },
        {
            "state": conversation.ProviderLifecycleWorkState.PENDING,
            "lease_owner": "owner",
            "lease_expires_at": _NOW,
        },
        {
            "state": conversation.ProviderLifecycleWorkState.CLAIMED,
            "lease_owner": "owner",
            "lease_expires_at": datetime(2026, 8, 2),
        },
    ),
)
def test_lifecycle_work_record_rejects_invalid_lease_shapes(
    changes: Mapping[str, object],
) -> None:
    """Bind claimed work to one exact valid owner lease."""
    values: dict[str, object] = {
        "work_id": "work",
        "checkpoint_id": conversation.CheckpointId("checkpoint"),
        "lane_id": conversation.ProviderLaneId("lane"),
        "binding_digest": conversation.IntegrityDigest("digest"),
        "upstream_response_id": conversation.UpstreamResponseId("private"),
        "origin": conversation.ProviderLifecycleOrigin.LOCAL_TOMBSTONE,
        "state": conversation.ProviderLifecycleWorkState.PENDING,
        "attempts": 0,
    }
    values.update(changes)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLifecycleWorkRecord(**values)


class _LifecycleAdapter:
    def __init__(self, binding: conversation.ProviderLaneBinding) -> None:
        self.binding = binding

    async def retrieve(
        self,
        upstream_response_id: conversation.UpstreamResponseId,
    ) -> conversation.RetrievedUpstreamResponse:
        return conversation.RetrievedUpstreamResponse(
            upstream_response_id=upstream_response_id,
            availability=conversation.UpstreamAvailability.AVAILABLE,
            retention=conversation.UpstreamRetentionMetadata.unknown(),
        )

    async def delete(
        self,
        upstream_response_id: conversation.UpstreamResponseId,
    ) -> conversation.UpstreamDeleteResult:
        assert upstream_response_id
        return conversation.UpstreamDeleteResult(
            disposition=conversation.UpstreamDeleteDisposition.ALREADY_ABSENT
        )


class _ResolverRuntime:
    def __init__(self, binding: conversation.ProviderLaneBinding) -> None:
        self.binding = binding


async def test_resolver_windows_and_reconciler_inputs_are_closed() -> None:
    """Reject malformed, duplicate, absent, early, and expired resolvers."""
    binding = _binding(lane_id="lane-resolver-validation")
    adapter = _LifecycleAdapter(binding)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.StoredProviderResolverEntry(
            adapter=adapter,
            revision="resolver-runtime-mismatch",
            valid_from=_NOW - timedelta(seconds=1),
            continuation_runtime=_ResolverRuntime(
                _binding(lane_id="lane-resolver-runtime-mismatch")
            ),
        )
    for values in (
        {"valid_from": datetime(2026, 8, 2)},
        {"valid_from": _NOW, "valid_until": _NOW},
        {"valid_from": _NOW, "valid_until": datetime(2026, 8, 3)},
        {"valid_from": _NOW, "adapter": object()},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.StoredProviderResolverEntry(
                adapter=cast(
                    conversation.StoredResponseLifecycleAdapter,
                    values.get("adapter", adapter),
                ),
                revision="resolver-entry",
                valid_from=cast(datetime, values["valid_from"]),
                valid_until=cast(datetime | None, values.get("valid_until")),
            )

    entry = conversation.StoredProviderResolverEntry(
        adapter=adapter,
        revision="resolver-valid",
        valid_from=_NOW - timedelta(seconds=1),
        valid_until=_NOW + timedelta(seconds=1),
    )

    async def now() -> datetime:
        return _NOW

    for entries in (
        (),
        (cast(conversation.StoredProviderResolverEntry, object()),),
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.StoredProviderResolver(entries, clock=now)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.StoredProviderResolver((entry, entry), clock=now)

    resolver = conversation.StoredProviderResolver((entry,), clock=now)
    with pytest.raises(conversation.ConversationValidationError):
        await resolver.resolve(conversation.IntegrityDigest("missing"))
    with pytest.raises(conversation.ConversationValidationError):
        await resolver.resolve_continuation_runtime(binding.integrity_digest)

    async def invalid_clock() -> datetime:
        return datetime(2026, 8, 2)

    invalid = conversation.StoredProviderResolver(
        (entry,), clock=invalid_clock
    )
    with pytest.raises(conversation.ConversationValidationError):
        await invalid.resolve(binding.integrity_digest)

    store = conversation.InMemoryConversationStore()
    reconciler = conversation.ProviderLifecycleReconciler(
        store=store,
        resolver=resolver,
        authority=authority(),
    )
    reconciler.assert_runtime(
        store=store,
        resolver=resolver,
        authority=authority(),
    )
    with pytest.raises(conversation.ConversationValidationError):
        reconciler.assert_runtime(
            store=conversation.InMemoryConversationStore(),
            resolver=resolver,
            authority=authority(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        await reconciler.run_once(limit=0)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLifecycleReconciler(
            store=cast(conversation.ProviderLifecycleStore, object()),
            resolver=resolver,
            authority=authority(),
        )
