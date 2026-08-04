"""Verify exact native OpenAI provider-stored conversation chaining."""

from asyncio import CancelledError
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI
from openai.types.responses import Response
from openai.types.responses.parsed_response import ParsedResponse
from phase2_fixtures import authority, child_identity, request

import avalan
import avalan.conversation as conversation
from avalan.conversation import coordinator as coordinator_module
from avalan.conversation.providers.openai import (
    _native_openai_test_authority,
)
from avalan.types import JsonValue

pytestmark = pytest.mark.anyio

_ADAPTER = (
    "avalan.conversation.providers.openai_stored.NativeOpenAIStoredProvider"
)
_NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)


@pytest.fixture
def anyio_backend() -> str:
    """Run stored provider tests on asyncio only."""
    return "asyncio"


def _binding(
    *,
    streaming: bool = False,
    endpoint: str = "https://api.openai.com/v1",
    family: conversation.ProviderFamily = conversation.ProviderFamily.OPENAI,
    lane_id: str = "lane-stored-native",
    tools: tuple[conversation.NativeOpenAIFunctionTool, ...] = (),
) -> conversation.ProviderLaneBinding:
    binding = conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=_ADAPTER,
        provider_family=family,
        normalized_endpoint=endpoint,
        azure_resource_identity=(
            "resource.openai.azure.com"
            if family is conversation.ProviderFamily.AZURE_OPENAI
            else None
        ),
        model_or_deployment=(
            "deployment-stored"
            if family is conversation.ProviderFamily.AZURE_OPENAI
            else "gpt-5"
        ),
        provider_api_revision=conversation.ProviderApiRevision(
            "azure-openai-v1"
            if family is conversation.ProviderFamily.AZURE_OPENAI
            else "openapi-2.3.0"
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=conversation.ModelConfigurationRevision(
            "model-config-phase6"
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-phase6")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-phase6"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-phase6")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=(
            conversation.ProviderTransport.STREAMING
            if streaming
            else conversation.ProviderTransport.NON_STREAMING
        ),
        agent_id=conversation.ConversationAgentId("agent-phase2"),
    )
    encrypted_content = (
        conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
        if family is conversation.ProviderFamily.AZURE_OPENAI
        else conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
    )
    return replace(
        binding,
        execution_definition_digest=(
            conversation.native_openai_stored_execution_digest(
                binding=binding,
                execution=_execution(),
                encrypted_content=encrypted_content,
                tools=tools,
            )
        ),
    )


def _capabilities(
    binding: conversation.ProviderLaneBinding,
    *,
    exclude: conversation.ConversationCapability | None = None,
) -> conversation.ConversationCapabilityProfile:
    supported = {
        conversation.ConversationCapability.STORED_RESPONSES_CHAINING,
        conversation.ConversationCapability.STORED_RESPONSE_RETRIEVAL,
        conversation.ConversationCapability.STORED_RESPONSE_DELETION,
        conversation.ConversationCapability.REASONING_CONTEXT_CURRENT_TURN,
        conversation.ConversationCapability.REASONING_CONTEXT_ALL_TURNS,
    }
    if binding.transport is conversation.ProviderTransport.STREAMING:
        supported.add(
            conversation.ConversationCapability.STREAMING_ITEM_FIDELITY
        )
    if exclude is not None:
        supported.discard(exclude)
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId(
            f"stored-profile-{binding.lane_id}"
        ),
        schema_version=1,
        revision=binding.capability_profile_revision,
        binding_alias=binding.safe_alias,
        capabilities=tuple(
            conversation.CapabilityEvidence(
                capability=capability,
                state=(
                    conversation.CapabilityEvidenceState.TEST_ONLY
                    if capability in supported
                    else conversation.CapabilityEvidenceState.INCAPABLE
                ),
                evidence_ids=(
                    (f"phase6-{capability.value}",)
                    if capability in supported
                    else ()
                ),
            )
            for capability in conversation.ConversationCapability
        ),
        test_only=True,
    )


def _execution() -> conversation.NativeOpenAIStoredExecution:
    return conversation.NativeOpenAIStoredExecution(
        instructions="Answer using the bound Phase 6 execution definition.",
        max_output_tokens=512,
        max_tool_calls=4,
        parallel_tool_calls=False,
        temperature=0.2,
        top_p=0.8,
        truncation="disabled",
    )


def _profile(
    binding: conversation.ProviderLaneBinding,
    *,
    scripted_tcp_test: bool = False,
) -> conversation.NativeOpenAIStoredProfile:
    return conversation.NativeOpenAIStoredProfile(
        profile_id=f"stored-{binding.lane_id}",
        binding=binding,
        execution=_execution(),
        encrypted_content=(
            conversation.NativeOpenAIEncryptedContentPolicy.EXPLICIT_INCLUDE
            if binding.provider_family
            is conversation.ProviderFamily.AZURE_OPENAI
            else conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
        ),
        scripted_tcp_test=scripted_tcp_test,
    )


def _provider(
    binding: conversation.ProviderLaneBinding,
    handler: Callable[
        [httpx.Request],
        Coroutine[None, None, httpx.Response],
    ],
    *,
    tools: tuple[conversation.NativeOpenAIFunctionTool, ...] = (),
    capabilities: conversation.ConversationCapabilityProfile | None = None,
) -> conversation.NativeOpenAIStoredProvider:
    client = AsyncOpenAI(
        api_key="stored-test-key",
        base_url=binding.normalized_endpoint,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    profile = _profile(binding)
    selected_capabilities = capabilities or _capabilities(binding)
    return conversation.NativeOpenAIStoredProvider(
        client=client,
        profile=profile,
        capability_profile=selected_capabilities,
        tools=tools,
        test_authority=_native_openai_test_authority(
            client=client,
            binding=binding,
            scripted_tcp_test=profile.scripted_tcp_test,
            capability_profile=selected_capabilities,
        ),
    )


def _retention() -> conversation.RetentionLimits:
    return conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.PROCESS_LOCAL,
            upstream=conversation.ProviderLaneStorage.STORED,
            provider_storage_disclosed=True,
        ),
        upstream_lifetime_status=conversation.UpstreamLifetimeStatus.UNKNOWN,
        local_ttl_seconds=3_600,
    )


async def _resolver_clock() -> datetime:
    return _NOW


def _direct_client(
    provider: conversation.NativeOpenAIStoredProvider,
    *,
    namespace: str,
    lifecycle: bool = False,
    boundary_hook: conversation.CoordinatorBoundaryHook | None = None,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.RunScopedConversationCoordinator,
    conversation.InMemoryConversationStore,
]:
    store = conversation.InMemoryConversationStore(
        clock=conversation.DeterministicFakeClock(_NOW)
    )
    scope = authority()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(conversation.NativeOpenAIStoredLaneRuntime(provider=provider),),
        boundary_hook=boundary_hook,
    )
    resolver = None
    reconciler = None
    if lifecycle:
        resolver = conversation.StoredProviderResolver(
            (
                conversation.StoredProviderResolverEntry(
                    adapter=provider,
                    revision="resolver-phase6",
                    valid_from=_NOW - timedelta(minutes=1),
                    valid_until=_NOW + timedelta(days=1),
                ),
            ),
            clock=_resolver_clock,
        )
        reconciler = conversation.ProviderLifecycleReconciler(
            store=store,
            resolver=resolver,
            authority=scope,
        )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=store,
        authority=scope,
        lane=provider.binding,
        retention=_retention(),
        id_namespace=namespace,
        provider_resolver=resolver,
        lifecycle_reconciler=reconciler,
    )
    return avalan.DirectConversationClient(runtime), coordinator, store


def _message(identifier: str, text: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_call(identifier: str, call_id: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": "lookup",
        "arguments": '{"value":1}',
    }


def _response(
    identifier: str,
    output: list[dict[str, object]],
    *,
    previous_response_id: str | None = None,
) -> dict[str, object]:
    return {
        "id": identifier,
        "object": "response",
        "created_at": 1.0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": _execution().instructions,
        "max_output_tokens": _execution().max_output_tokens,
        "max_tool_calls": _execution().max_tool_calls,
        "model": "gpt-5",
        "output": output,
        "parallel_tool_calls": False,
        "previous_response_id": previous_response_id,
        "reasoning": {"context": "current_turn"},
        "safety_identifier": _execution().safety_identifier,
        "store": True,
        "temperature": _execution().temperature,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": _execution().top_p,
        "truncation": _execution().truncation,
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 10,
        },
    }


async def test_stored_two_turn_chaining_reapplies_frozen_execution() -> None:
    """Chain the immediate private ID and reapply every frozen field."""
    requests: list[dict[str, object]] = []
    responses = (
        _response("private-response-one", [_message("message-one", "first")]),
        _response(
            "private-response-two",
            [_message("message-two", "second")],
            previous_response_id="private-response-one",
        ),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = loads(await request.aread())
        assert isinstance(payload, dict)
        requests.append(payload)
        return httpx.Response(200, json=responses[len(requests) - 1])

    provider = _provider(_binding(), handler)
    client, coordinator, store = _direct_client(
        provider,
        namespace="stored-two-turn",
    )
    first = await client.create(
        "first input",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(first.handle) is avalan.StoredConversationHandle
    second = await client.continue_conversation(
        "second input",
        avalan.StoredConversationSettings(
            provider_storage_disclosed=True,
            parent=avalan.StoredParent(handle=first.handle),
        ),
    )

    assert first.output == "first"
    assert second.output == "second"
    assert len(requests) == 2
    assert "previous_response_id" not in requests[0]
    assert requests[1]["previous_response_id"] == "private-response-one"
    for payload in requests:
        assert payload["store"] is True
        assert payload["instructions"] == _execution().instructions
        assert payload["max_output_tokens"] == 512
        assert payload["max_tool_calls"] == 4
        assert payload["parallel_tool_calls"] is False
        assert payload["temperature"] == 0.2
        assert payload["top_p"] == 0.8
        assert payload["truncation"] == "disabled"
        assert payload["tool_choice"] == "auto"
    checkpoint = await store.load(
        second.handle.checkpoint_id,
        authority(),
    )
    lane = checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
    assert lane.upstream_response_id == "private-response-two"
    assert checkpoint.retention.upstream_lifetime_status is (
        conversation.UpstreamLifetimeStatus.UNKNOWN
    )
    assert "private-response" not in repr(second)
    assert "private-response" not in repr(lane)
    plan = conversation.StoredProviderPlan(
        binding=provider.binding,
        upstream_response_id=conversation.UpstreamResponseId(
            "private-response-one"
        ),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=None,
        ),
        new_input={"text": "private-input"},
    )
    assert "private-response-one" not in repr(plan)
    assert "private-input" not in repr(plan)
    await coordinator.close()


async def test_stored_explicit_reset_starts_without_opaque_ancestry(
    record_property: Callable[[str, object], None],
) -> None:
    """Start a disclosed stored root without reusing the discarded parent."""
    record_property("conversation_acceptance_evidence", "runtime")
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], loads(await request.aread()))
        requests.append(payload)
        suffix = len(requests)
        return httpx.Response(
            200,
            json=_response(
                f"private-reset-{suffix}",
                [_message(f"reset-message-{suffix}", f"reset-{suffix}")],
            ),
        )

    provider = _provider(
        _binding(lane_id="lane-stored-reset"),
        handler,
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace="stored-reset",
    )
    first = await client.create(
        "before reset",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(first.handle) is avalan.StoredConversationHandle
    reset = await client.reset(
        "after reset",
        avalan.ConversationResetIntent(
            parent=avalan.StoredParent(handle=first.handle),
            target_mode=avalan.ConversationMode.STORED,
            provider_storage_disclosed=True,
        ),
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(reset.handle) is avalan.StoredConversationHandle
    assert reset.handle.conversation_id != first.handle.conversation_id
    assert all("previous_response_id" not in payload for payload in requests)
    assert requests[1]["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "after reset"}],
        }
    ]
    checkpoint = await store.load(reset.handle.checkpoint_id, authority())
    assert checkpoint.identity.parent_checkpoint_id is None
    assert checkpoint.content.visible_transcript.entries == (
        conversation.VisibleTranscriptEntry(
            role=conversation.VisibleTranscriptRole.USER,
            content="after reset",
        ),
    )
    await coordinator.close()


async def test_stored_tool_cycle_uses_only_immediate_id_and_tool_output() -> (
    None
):
    """Continue an internal tool cycle from exactly the immediate response."""
    requests: list[dict[str, object]] = []
    responses = (
        _response(
            "private-tool-one",
            [_function_call("function-one", "call-one")],
        ),
        _response(
            "private-tool-two",
            [_message("message-tool-two", "tool complete")],
            previous_response_id="private-tool-one",
        ),
    )

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments == {"value": 1}
        return "lookup result"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=tool_handler,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = loads(await request.aread())
        assert isinstance(payload, dict)
        requests.append(payload)
        return httpx.Response(200, json=responses[len(requests) - 1])

    provider = _provider(
        _binding(lane_id="lane-stored-tool", tools=(tool,)),
        handler,
        tools=(tool,),
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace="stored-tool",
    )
    result = await client.create(
        "use the tool",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )

    assert result.output == "tool complete"
    assert len(requests) == 2
    assert requests[1]["previous_response_id"] == "private-tool-one"
    assert requests[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-one",
            "output": "lookup result",
        }
    ]
    checkpoint = await store.load(result.handle.checkpoint_id, authority())
    lane = checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
    assert lane.upstream_response_id == "private-tool-two"
    segments = checkpoint.content.execution_segments
    assert tuple(segment.upstream_response_id for segment in segments) == (
        "private-tool-one",
        "private-tool-one",
        "private-tool-two",
    )
    assert tuple(segment.phase for segment in segments) == (
        conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
        conversation.ProviderExecutionSegmentPhase.TOOL_OUTPUT,
        conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
    )
    assert provider.diagnostics.request_count == 2
    assert provider.diagnostics.response_count == 2
    await coordinator.close()


async def test_stored_stream_commits_only_terminal_private_id() -> None:
    """Commit streamed stored state only after its terminal SDK response."""
    payloads: list[dict[str, object]] = []
    response = _response(
        "private-stream-response",
        [_message("stream-message", "streamed")],
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = loads(await request.aread())
        assert isinstance(payload, dict)
        payloads.append(payload)
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

    provider = _provider(
        _binding(streaming=True, lane_id="lane-stored-stream"),
        handler,
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace="stored-stream",
        lifecycle=True,
    )
    stream = await client.create(
        "stream this",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
        stream=True,
    )
    events = [event async for event in stream]

    assert [type(event) for event in events] == [
        avalan.DirectConversationOutputDelta,
        avalan.DirectConversationStreamTerminal,
    ]
    terminal = stream.terminal.result
    checkpoint = await store.load(
        terminal.handle.checkpoint_id,
        authority(),
    )
    lane = checkpoint.content.lanes[0]
    assert isinstance(lane, conversation.StoredProviderLaneSnapshot)
    assert lane.upstream_response_id == "private-stream-response"
    assert payloads[0]["store"] is True
    assert payloads[0]["stream"] is True
    assert provider.diagnostics.stream_close_count == 1
    await coordinator.close()


@pytest.mark.parametrize(
    "cancel_close", (False, True), ids=("fault", "cancel")
)
async def test_stored_stream_terminal_close_failure_is_quarantined(
    cancel_close: bool,
) -> None:
    """Quarantine a validated terminal child before close settlement."""
    response = _response(
        f"private-stream-close-{'cancel' if cancel_close else 'fault'}",
        [_message("stream-close-message", "never published")],
    )
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
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

    class CloseHook:
        def __init__(self) -> None:
            self.failed = False

        async def reach(
            self,
            boundary: conversation.CoordinatorAwaitBoundary,
        ) -> None:
            if (
                not self.failed
                and boundary
                is conversation.CoordinatorAwaitBoundary.PROVIDER_STREAM_CLOSE
            ):
                self.failed = True
                if cancel_close:
                    raise CancelledError()
                raise conversation.ConversationCommitError()

    provider = _provider(
        _binding(
            streaming=True,
            lane_id=(
                f"lane-stream-close-{'cancel' if cancel_close else 'fault'}"
            ),
        ),
        handler,
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace=f"stream-close-{'cancel' if cancel_close else 'fault'}",
        boundary_hook=CloseHook(),
    )
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True
    )
    key = conversation.RequestIdempotencyKey(
        f"stream-close-{'cancel' if cancel_close else 'fault'}-key"
    )
    stream = await client.create(
        "complete then fail close",
        settings,
        stream=True,
        idempotency_key=key,
    )
    expected = (
        avalan.DirectConversationCancelledError
        if cancel_close
        else conversation.ConversationCommitError
    )
    with pytest.raises(expected):
        _ = [event async for event in stream]

    page = await store.list_checkpoints(authority(), cursor=None, limit=10)
    assert len(page.checkpoints) == 1
    assert str(page.checkpoints[0].identity.checkpoint_id).startswith(
        "quarantine-"
    )
    work = await store.claim_provider_lifecycle(authority(), limit=10)
    assert len(work) == 1
    assert work[0].upstream_response_id == response["id"]
    retry = await client.create(
        "complete then fail close",
        settings,
        stream=True,
        idempotency_key=key,
    )
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        _ = [event async for event in retry]
    assert dispatches == 1
    assert await store.claim_provider_lifecycle(authority(), limit=10) == ()
    await coordinator.close()


async def test_unvalidated_stored_stream_is_not_completed_or_quarantined() -> (
    None
):
    """Keep partial output outside completed stored-response ownership."""
    response = _response(
        "private-unvalidated-stream",
        [_message("unvalidated-stream-message", "partial")],
    )
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        output = cast(list[dict[str, object]], response["output"])
        event = {
            "type": "response.output_item.done",
            "sequence_number": 0,
            "output_index": 0,
            "item": output[0],
        }
        body = f"data: {dumps(event)}\n\ndata: [DONE]\n\n"
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    provider = _provider(
        _binding(
            streaming=True,
            lane_id="lane-unvalidated-stored-stream",
        ),
        handler,
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace="unvalidated-stored-stream",
    )
    key = conversation.RequestIdempotencyKey("unvalidated-stored-stream-key")
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True
    )
    stream = await client.create(
        "never validate this terminal",
        settings,
        stream=True,
        idempotency_key=key,
    )
    iterator = stream.__aiter__()
    with pytest.raises(conversation.ConversationError) as captured:
        await anext(iterator)
    assert captured.value.boundary is (
        conversation.FailureBoundary.MALFORMED_STREAM_ITEM
    )
    with pytest.raises(avalan.ConversationHandleUnavailableError):
        _ = stream.terminal
    page = await store.list_checkpoints(authority(), cursor=None, limit=10)
    assert page.checkpoints == ()
    assert await store.claim_provider_lifecycle(authority(), limit=10) == ()
    assert provider.diagnostics.response_count == 0
    retry = await client.create(
        "never validate this terminal",
        settings,
        stream=True,
        idempotency_key=key,
    )
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await anext(retry.__aiter__())
    assert dispatches == 1
    await coordinator.close()


async def test_completed_stored_response_collection_is_exactly_once() -> None:
    """Ignore incomplete targets and deduplicate one validated terminal."""
    binding = _binding(lane_id="lane-completed-response-collection")
    result = conversation.ProviderResult(
        items=(),
        reasoning=conversation.EffectiveReasoningMetadata(
            requested=conversation.ReasoningContext.AUTO,
            effective=None,
        ),
    )
    completed: list[coordinator_module._CompletedStoredProviderResponse] = []
    coordinator_module._remember_completed_stored_response(
        completed,
        binding,
        result,
    )
    assert completed == []

    terminal = replace(
        result,
        upstream_response_id=conversation.UpstreamResponseId(
            "private-completed-response"
        ),
    )
    coordinator_module._remember_completed_stored_response(
        None,
        binding,
        terminal,
    )
    coordinator_module._remember_completed_stored_response(
        completed,
        binding,
        terminal,
    )
    coordinator_module._remember_completed_stored_response(
        completed,
        binding,
        terminal,
    )
    assert len(completed) == 1


async def test_stored_retrieve_and_local_first_delete_are_proven() -> None:
    """Prove retrieval and idempotent provider deletion behind local IDs."""
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            await request.aread()
            return httpx.Response(
                200,
                json=_response(
                    "private-lifecycle-response",
                    [_message("lifecycle-message", "retained")],
                ),
            )
        if request.method == "GET":
            return httpx.Response(
                200,
                json=_response(
                    "private-lifecycle-response",
                    [_message("lifecycle-message", "retained")],
                ),
            )
        assert request.method == "DELETE"
        return httpx.Response(204)

    provider = _provider(
        _binding(lane_id="lane-stored-lifecycle"),
        handler,
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace="stored-lifecycle",
        lifecycle=True,
    )
    created = await client.create(
        "retain this",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert isinstance(created.handle, avalan.StoredConversationHandle)
    public_id = created.handle.public_response_id
    assert public_id is not None

    retrieved = await client.retrieve(public_id)
    deletion = await client.delete(public_id)
    repeated_deletion = await client.delete(public_id)

    assert retrieved.output == "retained"
    assert deletion.local_tombstoned
    assert not deletion.upstream_pending
    assert repeated_deletion == deletion
    assert methods == ["POST", "GET", "DELETE"]
    assert "private-lifecycle-response" not in repr(retrieved)
    assert "private-lifecycle-response" not in repr(deletion)
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(public_id, authority())
    await coordinator.close()


async def test_lifecycle_404_is_typed_unknown_and_idempotent_absence() -> None:
    """Represent unknown retrieval absence and repeated deletion explicitly."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": "not found",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "not_found",
                }
            },
        )

    provider = _provider(_binding(lane_id="lane-stored-absent"), handler)
    upstream_id = conversation.UpstreamResponseId("private-absent-response")
    retrieved = await provider.retrieve(upstream_id)
    deleted = await provider.delete(upstream_id)

    assert retrieved.availability is (
        conversation.UpstreamAvailability.UNKNOWN_UNAVAILABLE
    )
    assert (
        retrieved.retention == conversation.UpstreamRetentionMetadata.unknown()
    )
    assert deleted.disposition is (
        conversation.UpstreamDeleteDisposition.ALREADY_ABSENT
    )
    assert "private-absent-response" not in repr(retrieved)
    await provider.aclose()


@pytest.mark.parametrize("azure", (False, True), ids=("openai", "azure"))
@pytest.mark.parametrize("streaming", (False, True), ids=("sync", "stream"))
async def test_exact_stored_lifecycle_profile_matrix(
    azure: bool,
    streaming: bool,
) -> None:
    """Exercise retrieval and deletion for every exact stored profile."""
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            response = _response(
                "private-matrix-lifecycle",
                [_message("matrix-lifecycle-message", "retained")],
            )
            response["model"] = "deployment-stored" if azure else "gpt-5"
            return httpx.Response(200, json=response)
        assert request.method == "DELETE"
        return httpx.Response(204)

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
            f"lane-lifecycle-{'azure' if azure else 'openai'}-"
            f"{'stream' if streaming else 'sync'}"
        ),
    )
    provider = _provider(binding, handler)
    upstream_id = conversation.UpstreamResponseId("private-matrix-lifecycle")

    retrieved = await provider.retrieve(upstream_id)
    deleted = await provider.delete(upstream_id)

    assert retrieved.availability is (
        conversation.UpstreamAvailability.AVAILABLE
    )
    assert (
        deleted.disposition is conversation.UpstreamDeleteDisposition.DELETED
    )
    assert methods == ["GET", "DELETE"]
    await provider.aclose()


async def test_resolver_retains_exact_adapter_inside_rotation_window() -> None:
    """Resolve one exact binding through its explicit retained window."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(204)

    provider = _provider(_binding(lane_id="lane-stored-resolver"), handler)

    async def before_window() -> datetime:
        return _NOW - timedelta(days=2)

    async def after_window() -> datetime:
        return _NOW + timedelta(days=2)

    entry = conversation.StoredProviderResolverEntry(
        adapter=provider,
        revision="resolver-window",
        valid_from=_NOW - timedelta(days=1),
        valid_until=_NOW + timedelta(days=1),
    )
    resolver = conversation.StoredProviderResolver(
        (entry,),
        clock=_resolver_clock,
    )
    assert (
        await resolver.resolve(provider.binding.integrity_digest) is provider
    )
    for clock in (before_window, after_window):
        outside = conversation.StoredProviderResolver((entry,), clock=clock)
        with pytest.raises(conversation.ConversationValidationError):
            await outside.resolve(provider.binding.integrity_digest)
    await provider.aclose()


async def test_unproven_lifecycle_and_conversion_fail_closed(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject unproven lifecycle and continuity-preserving conversion."""
    record_property("conversation_acceptance_evidence", "security")
    unknown_retention = conversation.RetentionLimits(
        storage=conversation.StoragePolicy(
            local=conversation.LocalResponseStorage.DURABLE,
            upstream=conversation.ProviderLaneStorage.STORED,
            provider_storage_disclosed=True,
        ),
        upstream_lifetime_status=(conversation.UpstreamLifetimeStatus.UNKNOWN),
        local_ttl_seconds=60,
        envelope_ttl_seconds=30,
    )
    assert unknown_retention.effective_ttl_seconds == 30
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        await request.aread()
        return httpx.Response(
            200,
            json=_response(
                "private-unproven",
                [_message("unproven-message", "retained")],
            ),
        )

    provider = _provider(_binding(lane_id="lane-stored-unproven"), handler)
    client, coordinator, store = _direct_client(
        provider,
        namespace="stored-unproven",
    )
    created = await client.create(
        "unproven lifecycle",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(created.handle) is avalan.StoredConversationHandle
    assert created.handle.public_response_id is not None
    with pytest.raises(conversation.ConversationCapabilityError):
        await client.retrieve(created.handle.public_response_id)
    deleted = await client.delete(created.handle.public_response_id)
    assert deleted.upstream_pending
    restarted = avalan.DirectConversationClient(client._runtime)
    repeated = await restarted.delete(created.handle.public_response_id)
    assert repeated == deleted
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(created.handle.public_response_id, authority())

    parent = avalan.StoredParent(handle=created.handle)
    transition = avalan.ConversationModeConversion(
        authorization=avalan.ModeTransitionAuthority(
            authority=authority(),
            binding=provider.binding,
            checkpoint_id=created.handle.checkpoint_id,
            parent=parent,
            source_mode=avalan.ConversationMode.STORED,
            target_mode=avalan.ConversationMode.STATELESS,
            operation=avalan.ConversationModeChangeOperation.CONVERT,
        )
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await client.convert(
            "convert explicitly",
            transition,
            avalan.StatelessConversationSettings(),
        )
    assert methods == ["POST"]
    await coordinator.close()


async def test_direct_retrieve_conceals_unknown_upstream_absence() -> None:
    """Keep a locally retained result unavailable when upstream is unknown."""
    methods: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            await request.aread()
            return httpx.Response(
                200,
                json=_response(
                    "private-now-unavailable",
                    [_message("unavailable-message", "retained")],
                ),
            )
        return httpx.Response(
            404,
            json={"error": {"message": "missing", "type": "not_found"}},
        )

    provider = _provider(_binding(lane_id="lane-stored-unavailable"), handler)
    client, coordinator, _ = _direct_client(
        provider,
        namespace="stored-unavailable",
        lifecycle=True,
    )
    created = await client.create(
        "become unavailable",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(created.handle) is avalan.StoredConversationHandle
    assert created.handle.public_response_id is not None
    with pytest.raises(conversation.ConversationAuthorizationError):
        await client.retrieve(created.handle.public_response_id)
    assert methods == ["POST", "GET"]
    await coordinator.close()


async def test_known_provider_rejection_releases_idempotency_fence(
    record_property: Callable[[str, object], None],
) -> None:
    """Retry one exact key after a definitive provider HTTP rejection."""
    record_property("conversation_acceptance_evidence", "security")
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        if dispatches == 1:
            return httpx.Response(
                400,
                json={"error": {"message": "rejected", "type": "bad"}},
            )
        return httpx.Response(
            200,
            json=_response(
                "private-after-rejection",
                [_message("after-rejection", "accepted")],
            ),
        )

    provider = _provider(
        _binding(lane_id="lane-known-rejection"),
        handler,
    )
    client, coordinator, _ = _direct_client(
        provider,
        namespace="known-rejection",
    )
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True
    )
    key = conversation.RequestIdempotencyKey("known-rejection-key")
    with pytest.raises(conversation.ConversationError) as failure:
        await client.create("same request", settings, idempotency_key=key)
    assert (
        failure.value.boundary
        is conversation.FailureBoundary.PROVIDER_REJECTION
    )
    result = await client.create("same request", settings, idempotency_key=key)
    assert result.output == "accepted"
    assert dispatches == 2
    await coordinator.close()


async def test_ambiguous_dispatch_requires_explicit_async_resolution() -> None:
    """Keep an exact key fenced until an explicit durable decision."""
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        if dispatches == 1:
            raise httpx.ConnectError("ambiguous", request=request)
        return httpx.Response(
            200,
            json=_response(
                "private-after-reconciliation",
                [_message("after-reconciliation", "reconciled")],
            ),
        )

    provider = _provider(
        _binding(lane_id="lane-ambiguous-reconciliation"),
        handler,
    )
    client, coordinator, _ = _direct_client(
        provider,
        namespace="ambiguous-reconciliation",
    )
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True
    )
    key = conversation.RequestIdempotencyKey("ambiguous-reconciliation-key")
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await client.create("same request", settings, idempotency_key=key)
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await client.create("same request", settings, idempotency_key=key)
    assert dispatches == 1
    retained = await client.reconcile_ambiguous_dispatch(
        conversation.ConversationOperation.CREATE,
        key,
        conversation.AmbiguousDispatchResolution.RETAIN_FENCE,
    )
    assert retained.disposition is (
        conversation.AmbiguousDispatchReconciliationDisposition.FENCE_RETAINED
    )
    with pytest.raises(conversation.ConversationAmbiguousDispatchError):
        await client.create("same request", settings, idempotency_key=key)
    resolved = await client.reconcile_ambiguous_dispatch(
        conversation.ConversationOperation.CREATE,
        key,
        conversation.AmbiguousDispatchResolution.CONFIRMED_NO_DISPATCH,
    )
    assert resolved.disposition is (
        conversation.AmbiguousDispatchReconciliationDisposition.RESOLVED_NO_DISPATCH
    )
    result = await client.create("same request", settings, idempotency_key=key)
    assert result.output == "reconciled"
    assert dispatches == 2
    await coordinator.close()


async def test_execution_definition_bytes_fail_before_dispatch(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject instruction and tool-schema drift under unchanged labels."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        return httpx.Response(500)

    async def original_tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        return str(arguments)

    original_tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
        handler=original_tool_handler,
    )
    binding = _binding(
        lane_id="lane-execution-byte-drift",
        tools=(original_tool,),
    )
    client = AsyncOpenAI(
        api_key="phase6-key",
        base_url=binding.normalized_endpoint,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    drifted_profile = replace(
        _profile(binding),
        execution=replace(_execution(), instructions="drifted instructions"),
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        conversation.NativeOpenAIStoredProvider(
            client=client,
            profile=drifted_profile,
            capability_profile=_capabilities(binding),
            tools=(original_tool,),
        )
    drifted_tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
        },
        handler=original_tool_handler,
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        conversation.NativeOpenAIStoredProvider(
            client=client,
            profile=_profile(binding),
            capability_profile=_capabilities(binding),
            tools=(drifted_tool,),
        )
    assert dispatches == 0
    await client.close()


async def test_public_and_upstream_id_alias_is_rejected_end_to_end() -> None:
    """Reject a provider response that aliases the reserved public ID."""
    alias = "direct-id-alias-1-create-response"

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            json=_response(alias, [_message("alias-message", "hidden")]),
        )

    provider = _provider(_binding(lane_id="lane-id-alias"), handler)
    client, coordinator, store = _direct_client(
        provider,
        namespace="id-alias",
    )
    with pytest.raises(conversation.ConversationValidationError):
        await client.create(
            "alias",
            avalan.StoredConversationSettings(provider_storage_disclosed=True),
        )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(
            conversation.PublicResponseId(alias),
            authority(),
        )
    work = await store.claim_provider_lifecycle(authority(), limit=1)
    assert len(work) == 1
    assert work[0].upstream_response_id == alias
    await coordinator.close()


async def test_generated_checkpoint_alias_is_rejected_for_every_topology() -> (
    None
):
    """Reject generated checkpoint aliases for create, continue, and branch."""
    settings = avalan.StoredConversationSettings(
        provider_storage_disclosed=True
    )
    for operation in ("create", "continue", "branch"):
        namespace = f"generated-checkpoint-{operation}"
        responses = [
            _response(
                f"private-{operation}-parent",
                [_message(f"{operation}-parent-message", "parent")],
            )
        ]
        if operation == "create":
            responses[0] = _response(
                f"direct-{namespace}-1-create-checkpoint",
                [_message("create-collision-message", "hidden")],
            )
        else:
            responses.append(
                _response(
                    f"direct-{namespace}-2-{operation}-checkpoint",
                    [_message(f"{operation}-collision-message", "hidden")],
                    previous_response_id=f"private-{operation}-parent",
                )
            )
        request_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal request_count
            await request.aread()
            response = responses[request_count]
            request_count += 1
            return httpx.Response(200, json=response)

        provider = _provider(
            _binding(lane_id=f"lane-generated-checkpoint-{operation}"),
            handler,
        )
        client, coordinator, store = _direct_client(
            provider,
            namespace=namespace,
        )
        parent: avalan.DirectConversationResult | None = None
        if operation != "create":
            parent = await client.create("parent", settings)
            assert type(parent.handle) is avalan.StoredConversationHandle
            parent_checkpoint = await store.load(
                parent.handle.checkpoint_id,
                authority(),
            )
        with pytest.raises(conversation.ConversationValidationError):
            if operation == "create":
                await client.create("collide create", settings)
            elif operation == "continue":
                assert parent is not None
                await client.continue_conversation(
                    "collide continue",
                    avalan.StoredConversationSettings(
                        provider_storage_disclosed=True,
                        parent=avalan.StoredParent(handle=parent.handle),
                    ),
                )
            else:
                assert parent is not None
                stored_parent = avalan.StoredParent(handle=parent.handle)
                await client.branch(
                    "collide branch",
                    avalan.StoredConversationSettings(
                        provider_storage_disclosed=True,
                        parent=stored_parent,
                        branch=avalan.ConversationBranchIntent(
                            parent=stored_parent,
                            branch_id=conversation.ConversationBranchId(
                                f"{namespace}-child-branch"
                            ),
                        ),
                    ),
                )
        failed_sequence = 1 if operation == "create" else 2
        public_response_id = conversation.PublicResponseId(
            f"direct-{namespace}-{failed_sequence}-{operation}-response"
        )
        with pytest.raises(conversation.ConversationAuthorizationError):
            await store.retrieve(public_response_id, authority())
        work = await store.claim_provider_lifecycle(authority(), limit=10)
        assert len(work) == 1
        assert (
            work[0].upstream_response_id
            == f"direct-{namespace}-{failed_sequence}-{operation}-checkpoint"
        )
        if parent is not None:
            assert (
                await store.load(parent.handle.checkpoint_id, authority())
                == parent_checkpoint
            )
        await coordinator.close()


@pytest.mark.parametrize(
    "field,value",
    (("model", "wrong-model"), ("store", False), ("status", "failed")),
)
async def test_retrieve_validates_exact_stored_profile(
    field: str,
    value: object,
) -> None:
    """Reject lifecycle retrieval metadata drift before availability proof."""

    async def handler(request: httpx.Request) -> httpx.Response:
        response = _response(
            "private-retrieve-profile",
            [_message("retrieve-profile", "retained")],
        )
        response[field] = value
        return httpx.Response(200, json=response)

    provider = _provider(
        _binding(lane_id=f"lane-retrieve-{field}"),
        handler,
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        await provider.retrieve(
            conversation.UpstreamResponseId("private-retrieve-profile")
        )
    await provider.aclose()


async def test_retrieve_requires_a_typed_response_or_subclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept SDK subclasses and reject callable model-dump ducks."""

    async def unused_handler(request: httpx.Request) -> httpx.Response:
        del request
        raise AssertionError("HTTP transport must not be reached")

    provider = _provider(
        _binding(lane_id="lane-retrieve-response-type"),
        unused_handler,
    )
    upstream_response_id = conversation.UpstreamResponseId(
        "private-retrieve-response-type"
    )
    parsed = ParsedResponse[object].model_validate(
        _response(
            str(upstream_response_id),
            [_message("retrieve-response-type", "retained")],
        )
    )
    assert isinstance(parsed, Response)

    async def retrieve_parsed(response_id: str) -> ParsedResponse[object]:
        assert response_id == str(upstream_response_id)
        return parsed

    monkeypatch.setattr(
        provider._client.responses,
        "retrieve",
        retrieve_parsed,
    )
    retrieved = await provider.retrieve(upstream_response_id)
    assert (
        retrieved.availability is conversation.UpstreamAvailability.AVAILABLE
    )

    class ModelDumpDuck:
        def __init__(self) -> None:
            self.called = False

        def model_dump(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            self.called = True
            return _response(str(upstream_response_id), [])

    duck = ModelDumpDuck()

    async def retrieve_duck(response_id: str) -> Any:
        assert response_id == str(upstream_response_id)
        return duck

    monkeypatch.setattr(
        provider._client.responses,
        "retrieve",
        retrieve_duck,
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        await provider.retrieve(upstream_response_id)
    assert not duck.called
    await provider.aclose()


@pytest.mark.parametrize(
    "drift",
    (
        "valid",
        "instructions",
        "temperature",
        "tools",
        "include",
        "stream",
        "reasoning_missing",
        "reasoning_context_type",
        "reasoning_context_value",
        "combined",
    ),
)
async def test_retrieve_validates_full_execution_definition(
    drift: str,
) -> None:
    """Bind retrieval to every provider-returned execution field."""

    async def tool_handler(arguments: Mapping[str, JsonValue]) -> str:
        assert arguments == {"value": 1}
        return "ok"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        description="Look up one value.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=tool_handler,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        response = _response(
            "private-retrieve-execution",
            [_message("retrieve-execution", "retained")],
        )
        response["tools"] = [tool.schema]
        if drift in {"instructions", "combined"}:
            response["instructions"] = "drifted instructions"
        if drift in {"temperature", "combined"}:
            response["temperature"] = 0.3
        if drift in {"tools", "combined"}:
            drifted_tool = dict(tool.schema)
            drifted_tool["name"] = "drifted_lookup"
            response["tools"] = [drifted_tool]
        if drift == "include":
            response["include"] = ["reasoning.encrypted_content"]
        if drift == "stream":
            response["stream"] = True
        if drift == "reasoning_missing":
            response["reasoning"] = None
        if drift == "reasoning_context_type":
            response["reasoning"] = {}
        if drift == "reasoning_context_value":
            response["reasoning"] = {"context": "future_turn"}
        if drift == "combined":
            response["max_tool_calls"] = 5
            response["reasoning"] = {"context": "all_turns"}
            response["safety_identifier"] = "drifted-safety"
        return httpx.Response(200, json=response)

    provider = _provider(
        _binding(
            lane_id=f"lane-retrieve-execution-{drift}",
            tools=(tool,),
        ),
        handler,
        tools=(tool,),
    )
    upstream_response_id = conversation.UpstreamResponseId(
        "private-retrieve-execution"
    )
    if drift == "valid":
        retrieved = await provider.retrieve(upstream_response_id)
        assert retrieved.binding_digest == provider.binding.integrity_digest
        assert retrieved.execution_definition_digest == (
            provider.binding.execution_definition_digest
        )
        assert retrieved.effective_reasoning_context is (
            conversation.EffectiveReasoningContext.CURRENT_TURN
        )
    else:
        with pytest.raises(conversation.ConversationProviderResponseError):
            await provider.retrieve(upstream_response_id)
    await provider.aclose()


async def test_retired_runtime_continues_old_parent_with_old_provider() -> (
    None
):
    """Use retained exact credentials for old parents and current for roots."""
    old_dispatches = 0
    new_dispatches = 0

    async def old_handler(request: httpx.Request) -> httpx.Response:
        nonlocal old_dispatches
        old_dispatches += 1
        payload = cast(dict[str, object], loads(await request.aread()))
        return httpx.Response(
            200,
            json=_response(
                f"private-old-{old_dispatches}",
                [_message(f"old-message-{old_dispatches}", "old")],
                previous_response_id=cast(
                    str | None,
                    payload.get("previous_response_id"),
                ),
            ),
        )

    async def new_handler(request: httpx.Request) -> httpx.Response:
        nonlocal new_dispatches
        new_dispatches += 1
        await request.aread()
        return httpx.Response(
            200,
            json=_response(
                f"private-new-{new_dispatches}",
                [_message(f"new-message-{new_dispatches}", "new")],
            ),
        )

    old_binding = _binding(lane_id="lane-retired-runtime")
    old_provider = _provider(old_binding, old_handler)
    old_client, old_coordinator, store = _direct_client(
        old_provider,
        namespace="retired-old",
    )
    first = await old_client.create(
        "old root",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert type(first.handle) is avalan.StoredConversationHandle
    new_binding = replace(
        _binding(lane_id="lane-retired-runtime"),
        model_configuration_revision=conversation.ModelConfigurationRevision(
            "model-config-phase6-current"
        ),
    )
    new_provider = _provider(new_binding, new_handler)
    new_runtime = conversation.NativeOpenAIStoredLaneRuntime(
        provider=new_provider
    )
    new_coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            authority()
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(new_runtime,),
    )
    old_runtime = old_coordinator._lanes[old_binding.lane_id]
    resolver = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=old_provider,
                revision="retired-runtime-old",
                valid_from=_NOW - timedelta(minutes=1),
                valid_until=_NOW + timedelta(minutes=5),
                continuation_runtime=old_runtime,
            ),
        ),
        clock=_resolver_clock,
    )
    current_client = avalan.DirectConversationClient(
        avalan.DirectConversationRuntime(
            coordinator=new_coordinator,
            store=store,
            authority=authority(),
            lane=new_binding,
            retention=_retention(),
            id_namespace="retired-current",
            provider_resolver=resolver,
        )
    )
    continued = await current_client.continue_conversation(
        "continue old",
        avalan.StoredConversationSettings(
            provider_storage_disclosed=True,
            parent=avalan.StoredParent(handle=first.handle),
        ),
    )
    assert continued.output == "old"
    assert old_dispatches == 2
    assert new_dispatches == 0
    root = await current_client.create(
        "new root",
        avalan.StoredConversationSettings(provider_storage_disclosed=True),
    )
    assert root.output == "new"
    assert new_dispatches == 1

    async def expired_clock() -> datetime:
        return _NOW + timedelta(hours=1)

    expired = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=old_provider,
                revision="retired-runtime-expired",
                valid_from=_NOW - timedelta(minutes=1),
                valid_until=_NOW + timedelta(minutes=5),
                continuation_runtime=old_runtime,
            ),
        ),
        clock=expired_clock,
    )
    expired_client = avalan.DirectConversationClient(
        replace(current_client._runtime, provider_resolver=expired)
    )
    with pytest.raises(conversation.ConversationValidationError):
        await expired_client.continue_conversation(
            "expired",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=avalan.StoredParent(handle=first.handle),
            ),
        )
    assert old_dispatches == 2
    assert new_dispatches == 1
    no_resolver_client = avalan.DirectConversationClient(
        replace(current_client._runtime, provider_resolver=None)
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        await no_resolver_client.continue_conversation(
            "missing retired runtime",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=avalan.StoredParent(handle=first.handle),
            ),
        )

    class MutableRuntime:
        def __init__(self, binding: conversation.ProviderLaneBinding) -> None:
            self.binding = binding

    mutable_runtime = MutableRuntime(old_binding)
    mismatched_runtime_resolver = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=old_provider,
                revision="retired-runtime-mutated-sdk",
                valid_from=_NOW - timedelta(minutes=1),
                valid_until=_NOW + timedelta(minutes=5),
                continuation_runtime=mutable_runtime,
            ),
        ),
        clock=_resolver_clock,
    )
    mutable_runtime.binding = new_binding
    mismatched_runtime_client = avalan.DirectConversationClient(
        replace(
            current_client._runtime,
            provider_resolver=mismatched_runtime_resolver,
        )
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        await mismatched_runtime_client.continue_conversation(
            "mismatched retired runtime",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=avalan.StoredParent(handle=first.handle),
            ),
        )

    retired_runtime = conversation.NativeOpenAIStoredLaneRuntime(
        provider=old_provider
    )
    coordinator_runtime_resolver = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=old_provider,
                revision="retired-runtime-mutated-coordinator",
                valid_from=_NOW - timedelta(minutes=1),
                valid_until=_NOW + timedelta(minutes=5),
                continuation_runtime=retired_runtime,
            ),
        ),
        clock=_resolver_clock,
    )
    object.__setattr__(retired_runtime, "provider", new_provider)
    parent_checkpoint = await store.load(
        first.handle.checkpoint_id,
        authority(),
    )
    mismatched_runtime_request = request(
        scope=authority(),
        identity=child_identity(
            parent_checkpoint,
            "retired-runtime-coordinator-mismatch",
        ),
        advance=conversation.OrdinaryChildAdvance(
            parent_checkpoint_id=parent_checkpoint.identity.checkpoint_id
        ),
        lane_ids=(str(old_binding.lane_id),),
        modes=(conversation.ConversationMode.STORED,),
        stored_retention=True,
        response_suffix="retired-runtime-coordinator-mismatch",
        key="retired-runtime-coordinator-mismatch",
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        await new_coordinator.execute(
            mismatched_runtime_request,
            stored_provider_resolver=coordinator_runtime_resolver,
        )
    assert old_dispatches == 2
    assert new_dispatches == 1
    current_only = conversation.StoredProviderResolver(
        (
            conversation.StoredProviderResolverEntry(
                adapter=new_provider,
                revision="retired-runtime-wrong",
                valid_from=_NOW - timedelta(minutes=1),
                valid_until=_NOW + timedelta(minutes=5),
                continuation_runtime=new_runtime,
            ),
        ),
        clock=_resolver_clock,
    )
    wrong_client = avalan.DirectConversationClient(
        replace(current_client._runtime, provider_resolver=current_only)
    )
    with pytest.raises(conversation.ConversationValidationError):
        await wrong_client.continue_conversation(
            "wrong runtime",
            avalan.StoredConversationSettings(
                provider_storage_disclosed=True,
                parent=avalan.StoredParent(handle=first.handle),
            ),
        )
    assert old_dispatches == 2
    assert new_dispatches == 1
    await old_provider.aclose()
    await new_provider.aclose()
    await store.close()
