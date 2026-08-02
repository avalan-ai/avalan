"""Verify exact native OpenAI and Azure stateless conversation replay."""

from asyncio import CancelledError, Event, create_task, gather
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from json import dumps, loads
from typing import Any, cast

import httpx
import pytest
from openai import AsyncOpenAI
from phase2_fixtures import authority, retention

import avalan
import avalan.conversation as conversation
from avalan.types import JsonValue

pytestmark = pytest.mark.anyio

_ADAPTER = "avalan.conversation.providers.openai.NativeOpenAIStatelessProvider"


@pytest.fixture
def anyio_backend() -> str:
    """Run deterministic provider tests on asyncio only."""
    return "asyncio"


def _binding(
    *,
    streaming: bool = False,
    azure: bool = False,
    endpoint: str | None = None,
    lane_id: str = "lane-native",
) -> conversation.ProviderLaneBinding:
    selected_endpoint = endpoint or (
        "https://resource.openai.azure.com/openai/v1"
        if azure
        else "https://api.openai.com/v1"
    )
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=_ADAPTER,
        provider_family=(
            conversation.ProviderFamily.AZURE_OPENAI
            if azure
            else conversation.ProviderFamily.OPENAI
        ),
        normalized_endpoint=selected_endpoint,
        azure_resource_identity=(
            "resource.openai.azure.com" if azure else None
        ),
        model_or_deployment="deployment-native" if azure else "gpt-5",
        provider_api_revision=conversation.ProviderApiRevision(
            "azure-openai-v1" if azure else "openapi-2.3.0"
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("model-config-phase5")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-phase5")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-phase5"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-phase5")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=(
            conversation.ProviderTransport.STREAMING
            if streaming
            else conversation.ProviderTransport.NON_STREAMING
        ),
        agent_id=conversation.ConversationAgentId("agent-phase2"),
    )


def _capabilities(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ConversationCapabilityProfile:
    supported = {
        conversation.ConversationCapability.STATELESS_ENCRYPTED_REASONING_REPLAY,
        conversation.ConversationCapability.REASONING_CONTEXT_CURRENT_TURN,
        conversation.ConversationCapability.REASONING_CONTEXT_ALL_TURNS,
    }
    if binding.transport is conversation.ProviderTransport.STREAMING:
        supported.add(
            conversation.ConversationCapability.STREAMING_ITEM_FIDELITY
        )
    return conversation.ConversationCapabilityProfile(
        profile_id=conversation.CapabilityProfileId(
            f"profile-{binding.lane_id}"
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
                    (f"phase5-{capability.value}",)
                    if capability in supported
                    else ()
                ),
            )
            for capability in conversation.ConversationCapability
        ),
        test_only=True,
    )


def _response(
    response_id: str,
    output: list[dict[str, object]],
    *,
    context: str = "current_turn",
    model: str = "gpt-5",
) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1.0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"context": context},
        "store": False,
        "temperature": None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": {
            "input_tokens": 4,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 6,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 10,
        },
    }


def _reasoning(identifier: str, opaque: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "reasoning",
        "status": "completed",
        "encrypted_content": opaque,
        "summary": [],
    }


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


def _profile(
    binding: conversation.ProviderLaneBinding,
    *,
    scripted_tcp_test: bool = False,
) -> conversation.NativeOpenAIStatelessProfile:
    return conversation.NativeOpenAIStatelessProfile(
        profile_id=f"native-{binding.provider_family.value}",
        binding=binding,
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
) -> conversation.NativeOpenAIStatelessProvider:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="test-key",
        base_url=binding.normalized_endpoint,
        default_query=(
            {"api-version": "preview"}
            if binding.provider_api_revision == "azure-openai-v1-preview"
            else None
        ),
        http_client=http_client,
        max_retries=0,
    )
    return conversation.NativeOpenAIStatelessProvider(
        client=client,
        profile=_profile(binding),
        capability_profile=_capabilities(binding),
        tools=tools,
    )


def _direct_client(
    provider: conversation.NativeOpenAIStatelessProvider,
    *,
    store: conversation.InMemoryConversationStore | None = None,
    namespace: str = "native",
    boundary_hook: conversation.CoordinatorBoundaryHook | None = None,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.RunScopedConversationCoordinator,
    conversation.InMemoryConversationStore,
]:
    selected_store = store or conversation.InMemoryConversationStore()
    scope = authority()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=selected_store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(
            datetime(2026, 8, 2, 12, tzinfo=UTC)
        ),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.NativeOpenAIConversationLaneRuntime(
                provider=provider
            ),
        ),
        boundary_hook=boundary_hook,
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=selected_store,
        authority=scope,
        lane=provider.binding,
        retention=retention(),
        id_namespace=namespace,
    )
    return (
        avalan.DirectConversationClient(runtime),
        coordinator,
        selected_store,
    )


def _plan(
    binding: conversation.ProviderLaneBinding,
    *,
    reasoning: avalan.ReasoningContext = avalan.ReasoningContext.AUTO,
) -> conversation.StatelessProviderPlan:
    return conversation.StatelessProviderPlan(
        binding=binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=(
                conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
            ),
            items=(),
        ),
        reasoning=avalan.EffectiveReasoningMetadata(
            requested=reasoning,
            effective=None,
        ),
        new_input={"text": "matrix input"},
    )


async def test_native_openai_two_turn_replay_is_exact_and_private(
    record_property: Callable[[str, object], None],
) -> None:
    """Replay complete ordered private items and append only new input."""
    record_property("conversation_acceptance_evidence", "wire")
    requests: list[dict[str, object]] = []
    responses = [
        _response(
            "upstream-response-one",
            [
                _reasoning("reasoning-one", "opaque-private-one"),
                _message("message-one", "first"),
            ],
        ),
        _response(
            "upstream-response-two",
            [
                _reasoning("reasoning-two", "opaque-private-two"),
                _message("message-two", "second"),
            ],
            context="all_turns",
        ),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(cast(dict[str, object], loads(await request.aread())))
        return httpx.Response(200, json=responses[len(requests) - 1])

    provider = _provider(_binding(), handler)
    client, coordinator, store = _direct_client(provider)
    first = await client.create(
        "first input",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    second = await client.continue_conversation(
        "second input",
        avalan.StatelessConversationSettings(
            parent=avalan.StatelessParent(handle=first.handle)
        ),
    )

    assert first.output == "first"
    assert second.output == "second"
    assert second.reasoning.effective is (
        avalan.EffectiveReasoningContext.ALL_TURNS
    )
    assert requests[0] == {
        "input": [
            {
                "content": [{"text": "first input", "type": "input_text"}],
                "role": "user",
                "type": "message",
            }
        ],
        "model": "gpt-5",
        "store": False,
        "stream": False,
        "tools": [],
    }
    assert requests[1]["input"] == [
        {
            "encrypted_content": "opaque-private-one",
            "id": "reasoning-one",
            "status": "completed",
            "summary": [],
            "type": "reasoning",
        },
        {
            "content": [
                {"annotations": [], "text": "first", "type": "output_text"}
            ],
            "id": "message-one",
            "role": "assistant",
            "status": "completed",
            "type": "message",
        },
        {
            "content": [{"text": "second input", "type": "input_text"}],
            "role": "user",
            "type": "message",
        },
    ]
    assert requests[1]["store"] is False
    assert "previous_response_id" not in requests[1]
    assert "include" not in requests[1]
    assert "reasoning" not in requests[1]
    outward = repr((first, second, provider.diagnostics))
    assert "opaque-private" not in outward
    assert "upstream-response" not in outward
    assert type(second.handle) is avalan.StatelessConversationHandle
    checkpoint = await store.load(second.handle.checkpoint_id, authority())
    checkpoint_text = repr(checkpoint)
    assert "opaque-private" not in checkpoint_text
    assert "reasoning-one" not in checkpoint_text
    assert "message-one" not in checkpoint_text
    assert provider.diagnostics.request_item_count == 4
    await coordinator.close()


@pytest.mark.parametrize("azure", [False, True], ids=["openai", "azure"])
@pytest.mark.parametrize("streaming", [False, True], ids=["sync", "stream"])
@pytest.mark.parametrize(
    "reasoning",
    [
        avalan.ReasoningContext.AUTO,
        avalan.ReasoningContext.CURRENT_TURN,
        avalan.ReasoningContext.ALL_TURNS,
    ],
    ids=["auto", "current", "all"],
)
async def test_exact_profile_request_matrix(
    azure: bool,
    streaming: bool,
    reasoning: avalan.ReasoningContext,
    record_property: Callable[[str, object], None],
) -> None:
    """Exercise every typed include, context, and transport call shape."""
    record_property("conversation_acceptance_evidence", "matrix")
    requests: list[dict[str, object]] = []
    output = [
        _reasoning("matrix-reasoning", "matrix-private"),
        _message("matrix-message", "matrix result"),
    ]
    context = (
        "all_turns"
        if reasoning is avalan.ReasoningContext.ALL_TURNS
        else "current_turn"
    )
    response = _response(
        "matrix-response",
        output,
        context=context,
        model="deployment-native" if azure else "gpt-5",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(cast(dict[str, object], loads(await request.aread())))
        if not streaming:
            return httpx.Response(200, json=response)
        events = [
            {
                "type": "response.output_item.done",
                "sequence_number": index,
                "output_index": index,
                "item": item,
            }
            for index, item in enumerate(output)
        ]
        events.append(
            {
                "type": "response.completed",
                "sequence_number": len(events),
                "response": response,
            }
        )
        sse = "".join(f"data: {dumps(event)}\n\n" for event in events)
        return httpx.Response(
            200,
            text=sse + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    binding = _binding(
        azure=azure,
        streaming=streaming,
        lane_id=(
            f"lane-matrix-{'azure' if azure else 'openai'}-"
            f"{'stream' if streaming else 'sync'}-{reasoning.value}"
        ),
    )
    provider = _provider(binding, handler)
    plan = _plan(binding, reasoning=reasoning)
    if streaming:
        stream = await provider.stream(plan)
        assert [item async for item in stream]
        result = await stream.terminal()
        await stream.aclose()
    else:
        result = await provider.dispatch(plan)

    request = requests[0]
    assert request["store"] is False
    assert request["stream"] is streaming
    assert (request.get("include") == ["reasoning.encrypted_content"]) is azure
    if reasoning is avalan.ReasoningContext.AUTO:
        assert "reasoning" not in request
    else:
        assert request["reasoning"] == {
            "context": (
                "current_turn"
                if reasoning is avalan.ReasoningContext.CURRENT_TURN
                else "all_turns"
            )
        }
    assert result.reasoning.effective is (
        avalan.EffectiveReasoningContext.ALL_TURNS
        if context == "all_turns"
        else avalan.EffectiveReasoningContext.CURRENT_TURN
    )
    await provider.aclose()


async def test_native_azure_emits_exact_include_and_context() -> None:
    """Keep Azure profile differences exact and explicit on the wire."""
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(cast(dict[str, object], loads(await request.aread())))
        return httpx.Response(
            200,
            json=_response(
                "azure-response",
                [
                    _reasoning("azure-reasoning", "azure-opaque"),
                    _message("azure-message", "azure"),
                ],
                context="current_turn",
                model="deployment-native",
            ),
        )

    binding = _binding(azure=True, lane_id="lane-azure")
    provider = _provider(binding, handler)
    client, coordinator, _ = _direct_client(provider, namespace="azure")
    result = await client.create(
        "azure input",
        avalan.StatelessConversationSettings(
            reasoning_context=avalan.ReasoningContext.CURRENT_TURN
        ),
    )

    assert result.output == "azure"
    assert requests[0]["model"] == "deployment-native"
    assert requests[0]["include"] == ["reasoning.encrypted_content"]
    assert requests[0]["reasoning"] == {"context": "current_turn"}
    assert requests[0]["store"] is False
    assert "previous_response_id" not in requests[0]
    await coordinator.close()


async def test_native_function_cycles_use_the_coordinator_ledger() -> None:
    """Execute tools asynchronously and replay each call/output once."""
    requests: list[dict[str, object]] = []
    executed: list[Mapping[str, JsonValue]] = []

    async def lookup(
        arguments: Mapping[str, JsonValue],
    ) -> str:
        executed.append(arguments)
        return '{"value":2}'

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        description="Look up one value.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ("value",),
            "additionalProperties": False,
        },
        handler=lookup,
    )
    responses = [
        _response(
            "tool-response-one",
            [
                _reasoning("tool-reasoning-one", "tool-opaque-one"),
                _function_call("function-one", "call-one"),
            ],
        ),
        _response(
            "tool-response-two",
            [
                _reasoning("tool-reasoning-two", "tool-opaque-two"),
                _message("tool-message", "tool-finished"),
            ],
        ),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(cast(dict[str, object], loads(await request.aread())))
        return httpx.Response(200, json=responses[len(requests) - 1])

    provider = _provider(_binding(lane_id="lane-tool"), handler, tools=(tool,))
    client, coordinator, _ = _direct_client(provider, namespace="tool")
    result = await client.create(
        "use tool",
        avalan.StatelessConversationSettings(),
    )

    assert result.output == "tool-finished"
    assert len(executed) == 1
    assert dict(executed[0]) == {"value": 1}
    second_input = cast(list[dict[str, Any]], requests[1]["input"])
    assert [item["type"] for item in second_input] == [
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    assert second_input[1]["call_id"] == "call-one"
    assert second_input[2] == {
        "call_id": "call-one",
        "output": '{"value":2}',
        "type": "function_call_output",
    }
    assert second_input[0]["encrypted_content"] == "tool-opaque-one"
    await coordinator.close()


async def test_fresh_runtime_continues_from_checkpoint_only() -> None:
    """Continue with a fresh SDK client and the committed parent ledger."""
    first_requests: list[dict[str, object]] = []
    second_requests: list[dict[str, object]] = []

    async def first_handler(request: httpx.Request) -> httpx.Response:
        first_requests.append(
            cast(dict[str, object], loads(await request.aread()))
        )
        return httpx.Response(
            200,
            json=_response(
                "restart-response-one",
                [
                    _reasoning("restart-reasoning", "restart-private"),
                    _message("restart-message", "before restart"),
                ],
            ),
        )

    binding = _binding(lane_id="lane-restart")
    first_provider = _provider(binding, first_handler)
    first_client, _, store = _direct_client(
        first_provider,
        namespace="restart-one",
    )
    first = await first_client.create(
        "before restart",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    await first_provider.aclose()

    async def second_handler(request: httpx.Request) -> httpx.Response:
        second_requests.append(
            cast(dict[str, object], loads(await request.aread()))
        )
        return httpx.Response(
            200,
            json=_response(
                "restart-response-two",
                [
                    _reasoning("restart-reasoning-two", "restart-private-two"),
                    _message("restart-message-two", "after restart"),
                ],
            ),
        )

    second_provider = _provider(binding, second_handler)
    second_client, second_coordinator, _ = _direct_client(
        second_provider,
        store=store,
        namespace="restart-two",
    )
    second = await second_client.continue_conversation(
        "after restart",
        avalan.StatelessConversationSettings(
            parent=avalan.StatelessParent(handle=first.handle)
        ),
    )

    assert first_requests[0]["input"] != second_requests[0]["input"]
    replay = cast(list[dict[str, object]], second_requests[0]["input"])
    assert [item["type"] for item in replay] == [
        "reasoning",
        "message",
        "message",
    ]
    assert replay[0]["encrypted_content"] == "restart-private"
    assert second.output == "after restart"
    await second_coordinator.close()


async def test_parallel_branches_do_not_share_provider_state(
    record_property: Callable[[str, object], None],
) -> None:
    """Create isolated children without mutating their immutable parent."""
    record_property("conversation_acceptance_evidence", "runtime")
    requests: list[dict[str, object]] = []
    both_children_in_flight = Event()
    release_children = Event()
    children_in_flight = 0
    maximum_children_in_flight = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal children_in_flight, maximum_children_in_flight
        body = cast(dict[str, object], loads(await request.aread()))
        requests.append(body)
        request_number = len(requests) - 1
        encoded = dumps(body, sort_keys=True)
        if request_number == 0:
            suffix = "root"
            response_context = "current_turn"
        else:
            suffix = (
                "left"
                if "child left" in encoded
                else "right" if "child right" in encoded else "invalid"
            )
            response_context = "all_turns"
            children_in_flight += 1
            maximum_children_in_flight = max(
                maximum_children_in_flight,
                children_in_flight,
            )
            if children_in_flight == 2:
                both_children_in_flight.set()
            await both_children_in_flight.wait()
            await release_children.wait()
            children_in_flight -= 1
        return httpx.Response(
            200,
            json=_response(
                f"branch-response-{suffix}",
                [
                    _reasoning(
                        f"branch-reasoning-{suffix}",
                        f"branch-private-{suffix}",
                    ),
                    _message(f"branch-message-{suffix}", f"answer {suffix}"),
                ],
                context=response_context,
            ),
        )

    provider = _provider(_binding(lane_id="lane-branch"), handler)
    client, coordinator, store = _direct_client(provider, namespace="branch")
    parent_result = await client.create(
        "parent root",
        avalan.StatelessConversationSettings(),
    )
    assert type(parent_result.handle) is avalan.StatelessConversationHandle
    parent = avalan.StatelessParent(handle=parent_result.handle)
    before = conversation.ConversationCheckpointCodec().encode(
        await store.load(parent.handle.checkpoint_id, authority())
    )

    async def branch(
        branch_id: str,
    ) -> avalan.DirectConversationResult:
        intent = avalan.ConversationBranchIntent(
            parent=parent,
            branch_id=conversation.ConversationBranchId(branch_id),
        )
        return await client.branch(
            f"child {branch_id}",
            avalan.StatelessConversationSettings(
                parent=parent,
                branch=intent,
                reasoning_context=avalan.ReasoningContext.ALL_TURNS,
            ),
        )

    left_task = create_task(branch("left"))
    right_task = create_task(branch("right"))
    await both_children_in_flight.wait()
    assert maximum_children_in_flight == 2
    release_children.set()
    left, right = await gather(left_task, right_task)
    after = conversation.ConversationCheckpointCodec().encode(
        await store.load(parent.handle.checkpoint_id, authority())
    )

    assert {left.output, right.output} == {"answer left", "answer right"}
    assert (
        left.reasoning.effective is avalan.EffectiveReasoningContext.ALL_TURNS
    )
    assert (
        right.reasoning.effective is avalan.EffectiveReasoningContext.ALL_TURNS
    )
    child_requests = requests[1:]
    assert len(child_requests) == 2
    parent_provider_items = [
        _reasoning("branch-reasoning-root", "branch-private-root"),
        _message("branch-message-root", "answer root"),
    ]
    for branch_id in ("left", "right"):
        matching = [
            request
            for request in child_requests
            if f"child {branch_id}" in dumps(request, sort_keys=True)
        ]
        assert len(matching) == 1
        assert matching[0] == {
            "input": [
                *parent_provider_items,
                {
                    "content": [
                        {
                            "text": f"child {branch_id}",
                            "type": "input_text",
                        }
                    ],
                    "role": "user",
                    "type": "message",
                },
            ],
            "model": "gpt-5",
            "reasoning": {"context": "all_turns"},
            "store": False,
            "stream": False,
            "tools": [],
        }
    assert before == after
    await coordinator.close()


async def test_malformed_pre_output_response_retries_at_most_once() -> None:
    """Retry one malformed response and then commit the valid one."""
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        await request.aread()
        reasoning = _reasoning("retry-reasoning", "retry-private")
        if requests == 1:
            reasoning.pop("encrypted_content")
        return httpx.Response(
            200,
            json=_response(
                f"retry-response-{requests}",
                [reasoning, _message("retry-message", "retried")],
            ),
        )

    provider = _provider(_binding(lane_id="lane-retry"), handler)
    client, coordinator, _ = _direct_client(provider, namespace="retry")
    result = await client.create(
        "retry input",
        avalan.StatelessConversationSettings(),
    )

    assert result.output == "retried"
    assert requests == 2
    await coordinator.close()


async def test_no_retry_after_tool_effect_or_ambiguous_transport() -> None:
    """Fence failures after tool execution and ambiguous dispatch."""
    tool_dispatches = 0
    tool_effects = 0

    async def lookup(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal tool_effects
        tool_effects += 1
        assert dict(arguments) == {"value": 1}
        return "done"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ("value",),
            "additionalProperties": False,
        },
        handler=lookup,
    )

    async def tool_handler(request: httpx.Request) -> httpx.Response:
        nonlocal tool_dispatches
        tool_dispatches += 1
        await request.aread()
        if tool_dispatches == 1:
            output = [
                _reasoning("effect-reasoning", "effect-private"),
                _function_call("effect-call", "effect-call-id"),
            ]
        else:
            broken = _reasoning("effect-broken", "effect-private-two")
            broken.pop("encrypted_content")
            output = [broken]
        return httpx.Response(
            200,
            json=_response(f"effect-response-{tool_dispatches}", output),
        )

    provider = _provider(
        _binding(lane_id="lane-effect"),
        tool_handler,
        tools=(tool,),
    )
    client, _, _ = _direct_client(provider, namespace="effect")
    with pytest.raises(conversation.ConversationError):
        await client.create(
            "effect input",
            avalan.StatelessConversationSettings(),
        )
    assert tool_dispatches == 2
    assert tool_effects == 1
    await provider.aclose()

    ambiguous_dispatches = 0

    async def ambiguous_handler(request: httpx.Request) -> httpx.Response:
        nonlocal ambiguous_dispatches
        ambiguous_dispatches += 1
        raise httpx.ReadError("ambiguous-private", request=request)

    ambiguous_provider = _provider(
        _binding(lane_id="lane-ambiguous"),
        ambiguous_handler,
    )
    ambiguous_client, _, _ = _direct_client(
        ambiguous_provider,
        namespace="ambiguous",
    )
    with pytest.raises(conversation.ConversationAmbiguousDispatchError) as exc:
        await ambiguous_client.create(
            "ambiguous input",
            avalan.StatelessConversationSettings(),
        )
    assert ambiguous_dispatches == 1
    assert exc.value.__cause__ is None
    assert "ambiguous-private" not in repr(exc.value)
    await ambiguous_provider.aclose()


async def test_native_stream_matches_non_stream_and_closes() -> None:
    """Commit complete fragmented stream items and close the SDK stream."""
    output = [
        _reasoning("stream-reasoning", "stream-private"),
        _message("stream-message", "streamed"),
    ]
    response = _response("stream-response", output)
    events = [
        {
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {
                "id": "stream-reasoning",
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            },
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 0,
            "item": output[0],
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 2,
            "output_index": 1,
            "item": output[1],
        },
        {
            "type": "response.completed",
            "sequence_number": 3,
            "response": response,
        },
    ]
    sse = "".join(f"data: {dumps(event)}\n\n" for event in events)
    sse += "data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], loads(await request.aread()))
        assert body["stream"] is True
        assert body["store"] is False
        return httpx.Response(
            200,
            text=sse,
            headers={"content-type": "text/event-stream"},
        )

    provider = _provider(_binding(streaming=True), handler)
    client, coordinator, _ = _direct_client(provider, namespace="stream")
    stream = await client.create(
        "stream input",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    events_out = [event async for event in stream]

    terminal = events_out[-1]
    assert type(terminal) is avalan.DirectConversationStreamTerminal
    assert terminal.result.output == "streamed"
    assert provider.diagnostics.response_item_kinds == (
        "reasoning",
        "message",
    )
    assert provider.diagnostics.stream_close_count == 1
    assert "stream-private" not in repr(events_out)
    await coordinator.close()


async def test_stream_cancellation_closes_without_committing_child() -> None:
    """Close a cancelled SDK stream and keep its parent reusable."""
    binding = _binding(streaming=True, lane_id="lane-cancel")

    def response(text: str, suffix: str) -> tuple[dict[str, object], str]:
        output = [
            _reasoning(
                f"cancel-reasoning-{suffix}",
                f"cancel-private-{suffix}",
            ),
            _message(f"cancel-message-{suffix}", text),
        ]
        terminal = _response(f"cancel-response-{suffix}", output)
        events = [
            {
                "type": "response.output_item.done",
                "sequence_number": index,
                "output_index": index,
                "item": item,
            }
            for index, item in enumerate(output)
        ]
        events.append(
            {
                "type": "response.completed",
                "sequence_number": len(events),
                "response": terminal,
            }
        )
        body = "".join(f"data: {dumps(event)}\n\n" for event in events)
        return terminal, body + "data: [DONE]\n\n"

    async def parent_handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        _, body = response("parent", "parent")
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    store = conversation.InMemoryConversationStore()
    parent_provider = _provider(binding, parent_handler)
    parent_client, _, _ = _direct_client(
        parent_provider,
        store=store,
        namespace="cancel-parent",
    )
    parent_stream = await parent_client.create(
        "parent input",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    parent_events = [event async for event in parent_stream]
    parent_terminal = parent_events[-1]
    assert type(parent_terminal) is avalan.DirectConversationStreamTerminal
    parent_handle = parent_terminal.result.handle
    assert type(parent_handle) is avalan.StatelessConversationHandle
    parent_checkpoint = await store.load(
        parent_handle.checkpoint_id,
        authority(),
    )
    parent_bytes = conversation.ConversationCheckpointCodec().encode(
        parent_checkpoint
    )
    await parent_provider.aclose()

    async def child_handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        _, body = response("cancelled child", "child")
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    cancelled_provider = _provider(binding, child_handler)
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="coordinator:provider_stream_item",
                exception=CancelledError(),
            ),
        )
    )
    cancelled_client, _, _ = _direct_client(
        cancelled_provider,
        store=store,
        namespace="cancel-child",
        boundary_hook=conversation.FakeCoordinatorBoundaryHook(controller),
    )
    cancelled_stream = await cancelled_client.continue_conversation(
        "cancel child",
        avalan.StatelessConversationSettings(
            parent=avalan.StatelessParent(handle=parent_handle)
        ),
        stream=True,
    )
    with pytest.raises(avalan.DirectConversationCancelledError):
        _ = [event async for event in cancelled_stream]
    assert cancelled_provider.diagnostics.stream_close_count == 1
    assert (
        conversation.ConversationCheckpointCodec().encode(
            await store.load(parent_handle.checkpoint_id, authority())
        )
        == parent_bytes
    )
    page = await store.list_checkpoints(authority(), cursor=None, limit=10)
    assert tuple(item.identity.checkpoint_id for item in page.checkpoints) == (
        parent_handle.checkpoint_id,
    )
    await cancelled_provider.aclose()

    async def retry_handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        _, body = response("reused parent", "retry")
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    retry_provider = _provider(binding, retry_handler)
    retry_client, _, _ = _direct_client(
        retry_provider,
        store=store,
        namespace="cancel-retry",
    )
    retry_stream = await retry_client.continue_conversation(
        "reuse parent",
        avalan.StatelessConversationSettings(
            parent=avalan.StatelessParent(handle=parent_handle)
        ),
        stream=True,
    )
    retry_events = [event async for event in retry_stream]
    retry_terminal = retry_events[-1]
    assert type(retry_terminal) is avalan.DirectConversationStreamTerminal
    assert retry_terminal.result.output == "reused parent"
    await retry_provider.aclose()


@pytest.mark.parametrize(
    "mutation",
    [
        "generic",
        "sdk-retry",
        "endpoint",
        "api-revision",
    ],
)
async def test_unproven_or_drifted_profiles_fail_without_dispatch(
    mutation: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject every unsupported identity cell before network dispatch."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        return httpx.Response(500)

    binding = _binding()
    if mutation == "generic":
        binding = replace(
            binding,
            provider_family=conversation.ProviderFamily.OPENAI_COMPATIBLE,
        )
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncOpenAI(
        api_key="test-key",
        base_url=(
            "https://wrong.example/v1"
            if mutation == "endpoint"
            else binding.normalized_endpoint
        ),
        http_client=http_client,
        max_retries=1 if mutation == "sdk-retry" else 0,
    )
    selected_binding = binding
    if mutation == "api-revision":
        selected_binding = replace(
            binding,
            provider_api_revision=conversation.ProviderApiRevision("drifted"),
        )
    with pytest.raises(conversation.ConversationError):
        conversation.NativeOpenAIStatelessProvider(
            client=client,
            profile=_profile(selected_binding),
            capability_profile=_capabilities(selected_binding),
        )
    assert dispatches == 0
    await client.close()
