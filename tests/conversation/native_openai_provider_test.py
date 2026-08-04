"""Verify exact native OpenAI and Azure stateless conversation replay."""

from asyncio import CancelledError, Event, create_task, gather, sleep
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
from avalan.agent import InputType, Specification
from avalan.agent.continuation_stager import PortableAgentContinuationStager
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecutionStatus,
    DurableInteractionRuntime,
    ExecutionInputRequiredError,
    create_agent_execution,
)
from avalan.conversation.providers.openai import (
    _native_openai_test_authority,
    _replay_item_to_input_item,
)
from avalan.entities import GenerationSettings
from avalan.interaction.codec import (
    decode_continuation_snapshot,
    encode_continuation_snapshot,
)
from avalan.interaction.entities import (
    AgentId,
    CapabilityRevision,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ModelConfigRevision,
    ModelId,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
)
from avalan.interaction.policy import InteractionActor
from avalan.model.call import ModelCallContext
from avalan.model.capability import (
    ContinuationSnapshotCodecRegistry,
    ModelCapabilityCatalog,
    ProviderCapabilitySupport,
)
from avalan.types import JsonValue

pytestmark = pytest.mark.anyio

_ADAPTER = "avalan.conversation.providers.openai.NativeOpenAIStatelessProvider"


class _StagingEngineAgent(EngineAgent):
    """Expose only the coordinated suspension bridge under test."""

    def _prepare_call(self, context: ModelCallContext) -> dict[str, object]:
        del context
        return {}


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


def _rich_message(identifier: str, text: str) -> dict[str, object]:
    """Return one typed OpenAI output with meaningful replay metadata."""
    return {
        "id": identifier,
        "type": "message",
        "status": "completed",
        "phase": "final_answer",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [
                    {
                        "type": "url_citation",
                        "start_index": 0,
                        "end_index": len(text),
                        "title": "Replay source",
                        "url": "https://example.com/replay",
                    }
                ],
                "logprobs": [
                    {
                        "token": text,
                        "logprob": -0.25,
                        "bytes": list(text.encode("utf-8")),
                        "top_logprobs": [],
                    }
                ],
            }
        ],
    }


def _azure_null_message(identifier: str, text: str) -> dict[str, object]:
    """Return the proven Azure null-metadata output shape."""
    return {
        "id": identifier,
        "type": "message",
        "status": "completed",
        "phase": None,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": None,
                "logprobs": None,
            }
        ],
    }


def _function_call(
    identifier: str,
    call_id: str,
    *,
    name: str = "lookup",
    arguments: str = '{"value":1}',
) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "function_call",
        "status": "completed",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
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
    profile = _profile(binding)
    capabilities = _capabilities(binding)
    return conversation.NativeOpenAIStatelessProvider(
        client=client,
        profile=profile,
        capability_profile=capabilities,
        tools=tools,
        test_authority=_native_openai_test_authority(
            client=client,
            binding=binding,
            scripted_tcp_test=profile.scripted_tcp_test,
            capability_profile=capabilities,
        ),
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


@pytest.mark.parametrize("azure", [False, True], ids=["openai", "azure"])
async def test_native_openai_two_turn_replay_is_exact_and_private(
    azure: bool,
    record_property: Callable[[str, object], None],
) -> None:
    """Replay complete ordered private items and append only new input."""
    record_property("conversation_acceptance_evidence", "wire")
    requests: list[dict[str, object]] = []
    model = "deployment-native" if azure else "gpt-5"
    first_message = (
        _azure_null_message("message-one", "first")
        if azure
        else _rich_message("message-one", "first")
    )
    responses = [
        _response(
            "upstream-response-one",
            [
                _reasoning("reasoning-one", "opaque-private-one"),
                first_message,
            ],
            model=model,
        ),
        _response(
            "upstream-response-two",
            [
                _reasoning("reasoning-two", "opaque-private-two"),
                _message("message-two", "second"),
            ],
            context="all_turns",
            model=model,
        ),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(cast(dict[str, object], loads(await request.aread())))
        return httpx.Response(200, json=responses[len(requests) - 1])

    binding = _binding(
        azure=azure,
        lane_id=f"lane-two-turn-{'azure' if azure else 'openai'}",
    )
    provider = _provider(binding, handler)
    client, coordinator, store = _direct_client(provider)
    first = await client.create(
        "first input",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    parent_before_replay = await store.load(
        first.handle.checkpoint_id,
        authority(),
    )
    parent_lane_before_replay = parent_before_replay.content.lanes[0]
    assert isinstance(
        parent_lane_before_replay,
        conversation.StatelessProviderLaneSnapshot,
    )
    canonical_message_before_replay = conversation.thaw_json_value(
        parent_lane_before_replay.ledger.items[1].canonical_input
    )
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
    first_request = {
        "input": [
            {
                "content": [{"text": "first input", "type": "input_text"}],
                "role": "user",
                "type": "message",
            }
        ],
        "model": model,
        "store": False,
        "stream": False,
        "tools": [],
    }
    if azure:
        first_request["include"] = ["reasoning.encrypted_content"]
    assert requests[0] == first_request
    replayed_message = cast(
        dict[str, object],
        loads(dumps(first_message)),
    )
    replayed_message.pop("status")
    if azure:
        replayed_message.pop("phase")
        content = cast(list[dict[str, object]], replayed_message["content"])
        content[0].pop("annotations")
        content[0].pop("logprobs")
    assert requests[1]["input"] == [
        {
            "encrypted_content": "opaque-private-one",
            "id": "reasoning-one",
            "summary": [],
            "type": "reasoning",
        },
        replayed_message,
        {
            "content": [{"text": "second input", "type": "input_text"}],
            "role": "user",
            "type": "message",
        },
    ]
    assert requests[1]["store"] is False
    assert "previous_response_id" not in requests[1]
    assert (
        requests[1].get("include") == ["reasoning.encrypted_content"]
    ) is azure
    assert "reasoning" not in requests[1]
    replay = cast(list[dict[str, object]], requests[1]["input"])
    assert all("status" not in item for item in replay)
    parent_checkpoint = await store.load(
        first.handle.checkpoint_id,
        authority(),
    )
    parent_lane = parent_checkpoint.content.lanes[0]
    assert isinstance(parent_lane, conversation.StatelessProviderLaneSnapshot)
    assert tuple(
        item.canonical_input.get("status") for item in parent_lane.ledger.items
    ) == ("completed", "completed")
    assert (
        conversation.thaw_json_value(
            parent_lane.ledger.items[1].canonical_input
        )
        == canonical_message_before_replay
    )
    if azure:
        assert canonical_message_before_replay == {
            "content": [{"text": "first", "type": "output_text"}],
            "id": "message-one",
            "role": "assistant",
            "status": "completed",
            "type": "message",
        }
    else:
        assert canonical_message_before_replay == first_message
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


def test_replay_item_normalization_is_exact_and_closed() -> None:
    """Strip only response metadata and reject malformed replay items."""
    azure_message = {
        "content": [
            {
                "annotations": None,
                "logprobs": None,
                "text": "normalize answer",
                "type": "output_text",
            }
        ],
        "id": "normalize-message",
        "phase": None,
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }
    source = (
        (
            _reasoning("normalize-reasoning", "normalize-private"),
            conversation.ProviderFamily.OPENAI,
        ),
        (
            _function_call("normalize-function", "normalize-call"),
            conversation.ProviderFamily.OPENAI,
        ),
        (azure_message, conversation.ProviderFamily.AZURE_OPENAI),
        (
            {
                "id": "normalize-compaction",
                "type": "compaction",
                "encrypted_content": "normalize-compact-private",
                "created_by": "provider-compact",
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                "id": "normalize-tool-output",
                "type": "function_call_output",
                "call_id": "normalize-call",
                "output": "normalize output",
                "status": "completed",
            },
            conversation.ProviderFamily.OPENAI,
        ),
    )

    normalized = tuple(
        cast(
            dict[str, object],
            _replay_item_to_input_item(
                item,
                provider_family=provider_family,
            ),
        )
        for item, provider_family in source
    )

    assert tuple(item["type"] for item in normalized) == (
        "reasoning",
        "function_call",
        "message",
        "compaction",
        "function_call_output",
    )
    assert all("status" not in item for item in normalized)
    assert "created_by" not in normalized[3]
    assert normalized[0]["id"] == "normalize-reasoning"
    assert normalized[0]["encrypted_content"] == "normalize-private"
    assert normalized[1]["id"] == "normalize-function"
    assert normalized[1]["call_id"] == "normalize-call"
    assert normalized[2] == {
        "content": [{"text": "normalize answer", "type": "output_text"}],
        "id": "normalize-message",
        "role": "assistant",
        "type": "message",
    }
    assert azure_message["phase"] is None
    assert cast(list[object], azure_message["content"])[0] == {
        "annotations": None,
        "logprobs": None,
        "text": "normalize answer",
        "type": "output_text",
    }
    assert normalized[3]["encrypted_content"] == "normalize-compact-private"
    assert source[3][0]["created_by"] == "provider-compact"
    assert normalized[4]["call_id"] == "normalize-call"

    rich_message = _rich_message("rich-message", "rich answer")
    rich_before = dumps(rich_message, sort_keys=True)
    rich_normalized = _replay_item_to_input_item(
        rich_message,
        provider_family=conversation.ProviderFamily.OPENAI,
    )
    assert rich_normalized == {
        key: value for key, value in rich_message.items() if key != "status"
    }
    assert dumps(rich_message, sort_keys=True) == rich_before
    with pytest.raises(conversation.ConversationValidationError):
        _replay_item_to_input_item(
            rich_message,
            provider_family=conversation.ProviderFamily.AZURE_OPENAI,
        )
    assert dumps(rich_message, sort_keys=True) == rich_before

    malformed = (
        (
            {"type": "reasoning", "id": "missing-opaque", "summary": []},
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                "type": "reasoning",
                "id": "untyped-opaque",
                "summary": [],
                "encrypted_content": 1,
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                **_reasoning("incomplete-reasoning", "private"),
                "status": "incomplete",
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                **_function_call("unknown-field", "unknown-call"),
                "unknown": 1,
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                "id": "missing-call-id",
                "type": "function_call",
                "name": "lookup",
                "arguments": "{}",
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                "id": "empty-compaction",
                "type": "compaction",
                "encrypted_content": "",
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                "id": "invalid-created-compaction",
                "type": "compaction",
                "encrypted_content": "private",
                "created_by": 1,
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {**azure_message, "phase": "final_answer"},
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [
                    {
                        "annotations": [],
                        "text": "answer",
                        "type": "output_text",
                    }
                ],
            },
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [
                    {
                        "annotations": [{"type": "file_citation"}],
                        "text": "answer",
                        "type": "output_text",
                    }
                ],
            },
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [
                    {
                        "logprobs": [{"token": "answer"}],
                        "text": "answer",
                        "type": "output_text",
                    }
                ],
            },
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [
                    {
                        "annotations": [{"type": "file_citation"}],
                        "text": "answer",
                        "type": "output_text",
                    }
                ],
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [
                    {
                        "logprobs": [{"token": "answer"}],
                        "text": "answer",
                        "type": "output_text",
                    }
                ],
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [
                    {
                        "text": "answer",
                        "type": "output_text",
                        "unknown": 1,
                    }
                ],
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [{"annotations": None, "type": "output_text"}],
            },
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [{"text": "answer", "type": "input_text"}],
            },
            conversation.ProviderFamily.OPENAI,
        ),
        (
            {**azure_message, "content": []},
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {**azure_message, "content": ["not-an-output-part"]},
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {
                **azure_message,
                "content": [{"refusal": 1, "type": "refusal"}],
            },
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {**azure_message, "role": "user"},
            conversation.ProviderFamily.AZURE_OPENAI,
        ),
        (
            {"type": "not-a-response-item"},
            conversation.ProviderFamily.OPENAI,
        ),
        ({"type": 1}, conversation.ProviderFamily.OPENAI),
    )
    for item, provider_family in malformed:
        with pytest.raises(conversation.ConversationValidationError):
            _replay_item_to_input_item(
                item,
                provider_family=provider_family,
            )

    with pytest.raises(conversation.ConversationValidationError):
        _replay_item_to_input_item(
            azure_message,
            provider_family=conversation.ProviderFamily.OPENAI_COMPATIBLE,
        )

    refusal = {
        **azure_message,
        "content": [{"refusal": "cannot comply", "type": "refusal"}],
    }
    assert _replay_item_to_input_item(
        refusal,
        provider_family=conversation.ProviderFamily.AZURE_OPENAI,
    ) == {
        "content": [{"refusal": "cannot comply", "type": "refusal"}],
        "id": "normalize-message",
        "role": "assistant",
        "type": "message",
    }
    canonical_without_null_metadata = conversation.ProviderItem(
        item_id=conversation.ProviderItemId("canonical-azure-message"),
        lane_id=conversation.ProviderLaneId("lane-canonical-azure"),
        model_call_id=conversation.ConversationModelCallId(
            "call-canonical-azure"
        ),
        kind=conversation.ProviderItemKind.MESSAGE,
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.FINAL,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input=cast(
            dict[str, JsonValue],
            {
                "content": [
                    {"text": "normalize answer", "type": "output_text"}
                ],
                "id": "canonical-azure-message",
                "role": "assistant",
                "status": "completed",
                "type": "message",
            },
        ),
        normalization_version=conversation.ConversationCodecVersion(1),
    )
    canonical_payload = cast(
        dict[str, object],
        conversation.thaw_json_value(
            canonical_without_null_metadata.canonical_input
        ),
    )
    assert canonical_payload["content"] == [
        {"text": "normalize answer", "type": "output_text"}
    ]


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
    tool_entered = Event()
    tool_heartbeat_complete = Event()
    heartbeat_ticks = 0

    async def tool_heartbeat() -> None:
        nonlocal heartbeat_ticks
        await tool_entered.wait()
        for _ in range(3):
            await sleep(0)
            heartbeat_ticks += 1
        tool_heartbeat_complete.set()

    async def lookup(
        arguments: Mapping[str, JsonValue],
    ) -> str:
        page = await store.list_checkpoints(authority(), cursor=None, limit=10)
        durable = tuple(
            checkpoint.content.execution_segments[-1]
            for checkpoint in page.checkpoints
            if checkpoint.kind
            is conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
            and checkpoint.content.execution_segments
        )
        assert len(durable) == 1
        assert durable[0].phase is (
            conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
        )
        assert durable[0].tools[0].phase is (
            conversation.ToolExecutionPhase.REQUESTED
        )
        tool_entered.set()
        await tool_heartbeat_complete.wait()
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
        if len(requests) == 2:
            page = await store.list_checkpoints(
                authority(),
                cursor=None,
                limit=10,
            )
            segments = tuple(
                checkpoint.content.execution_segments[-1]
                for checkpoint in page.checkpoints
                if checkpoint.kind
                is conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
                and checkpoint.content.execution_segments
            )
            requested = next(
                segment
                for segment in segments
                if segment.tools
                and segment.phase
                is conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
            )
            persisted = next(
                segment
                for segment in segments
                if segment.phase
                is conversation.ProviderExecutionSegmentPhase.TOOL_OUTPUT
            )
            assert (
                conversation.durable_tool_recovery_action(
                    (
                        requested,
                        persisted,
                    )
                )
                is conversation.DurableToolRecoveryAction.RESUME_PROVIDER
            )
        return httpx.Response(200, json=responses[len(requests) - 1])

    provider = _provider(_binding(lane_id="lane-tool"), handler, tools=(tool,))
    client, coordinator, store = _direct_client(provider, namespace="tool")
    heartbeat_task = create_task(tool_heartbeat())
    result = await client.create(
        "use tool",
        avalan.StatelessConversationSettings(),
    )
    await heartbeat_task

    assert result.output == "tool-finished"
    assert heartbeat_ticks == 3
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
    checkpoint = await store.load(result.handle.checkpoint_id, authority())
    segments = checkpoint.content.execution_segments
    assert tuple(segment.phase for segment in segments) == (
        conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
        conversation.ProviderExecutionSegmentPhase.TOOL_OUTPUT,
        conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
    )
    assert segments[0].tools[0].effect_policy is (
        conversation.ToolEffectPolicy.FENCED_UNPROTECTED
    )
    assert segments[0].tools[0].arguments == {"value": 1}
    assert segments[1].tools[0].output_id is not None
    assert not segments[2].tools
    await coordinator.close()


async def test_internal_completion_precedes_outward_commit_crash() -> None:
    """Recover deterministically after internal completion before commit."""
    requests = 0
    effects = 0
    store = conversation.InMemoryConversationStore()
    recovery_actions: list[conversation.DurableToolRecoveryAction] = []

    class CommitCrashHook:
        async def reach(
            self,
            boundary: conversation.CoordinatorAwaitBoundary,
        ) -> None:
            if boundary is not conversation.CoordinatorAwaitBoundary.COMMIT:
                return
            page = await store.list_checkpoints(
                authority(),
                cursor=None,
                limit=10,
            )
            segments = tuple(
                checkpoint.content.execution_segments[-1]
                for checkpoint in page.checkpoints
                if checkpoint.kind
                is conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
                and checkpoint.content.execution_segments
            )
            ordered = tuple(
                sorted(
                    segments,
                    key=lambda segment: (
                        segment.segment_index,
                        (
                            0
                            if segment.phase
                            is (
                                conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
                            )
                            else 1
                        ),
                    ),
                )
            )
            action = conversation.durable_tool_recovery_action(ordered)
            recovery_actions.append(action)
            assert action is (
                conversation.DurableToolRecoveryAction.COMMIT_OUTWARD
            )
            assert all(
                checkpoint.kind
                is conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
                for checkpoint in page.checkpoints
            )
            raise conversation.ConversationCommitError()

    async def lookup(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal effects
        assert arguments == {"value": 1}
        effects += 1
        return "durable-output"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=lookup,
        effect_policy=conversation.ToolEffectPolicy.IDEMPOTENT,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        await request.aread()
        if requests == 1:
            return httpx.Response(
                200,
                json=_response(
                    "internal-complete-request",
                    [
                        _reasoning(
                            "internal-complete-reasoning-1",
                            "internal-complete-private-1",
                        ),
                        _function_call(
                            "internal-complete-call",
                            "internal-complete-call-id",
                        ),
                    ],
                ),
            )
        return httpx.Response(
            200,
            json=_response(
                "internal-complete-terminal",
                [
                    _reasoning(
                        "internal-complete-reasoning-2",
                        "internal-complete-private-2",
                    ),
                    _message(
                        "internal-complete-message",
                        "not outwardly committed",
                    ),
                ],
            ),
        )

    provider = _provider(
        _binding(lane_id="lane-internal-complete-crash"),
        handler,
        tools=(tool,),
    )
    client, coordinator, _ = _direct_client(
        provider,
        store=store,
        namespace="internal-complete-crash",
        boundary_hook=CommitCrashHook(),
    )

    with pytest.raises(conversation.ConversationCommitError):
        await client.create(
            "complete internally then crash",
            avalan.StatelessConversationSettings(),
        )

    page = await store.list_checkpoints(authority(), cursor=None, limit=10)
    segments = tuple(
        checkpoint.content.execution_segments[-1]
        for checkpoint in page.checkpoints
        if checkpoint.content.execution_segments
    )
    ordered = tuple(
        sorted(
            segments,
            key=lambda segment: (
                segment.segment_index,
                (
                    0
                    if segment.phase
                    is (
                        conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
                    )
                    else 1
                ),
            ),
        )
    )
    assert recovery_actions == [
        conversation.DurableToolRecoveryAction.COMMIT_OUTWARD
    ]
    assert conversation.durable_tool_recovery_action(ordered) is (
        conversation.DurableToolRecoveryAction.COMMIT_OUTWARD
    )
    assert effects == 1
    assert requests == 2
    assert len(page.checkpoints) == 3
    assert tuple(segment.phase for segment in ordered) == (
        conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
        conversation.ProviderExecutionSegmentPhase.TOOL_OUTPUT,
        conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE,
    )
    assert all(
        checkpoint.content.execution_segments
        == ordered[
            : ordered.index(checkpoint.content.execution_segments[-1]) + 1
        ]
        for checkpoint in page.checkpoints
    )
    assert all(
        checkpoint.kind
        is conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
        for checkpoint in page.checkpoints
    )
    assert store.diagnostics.public_responses == 0
    assert store.diagnostics.outbox_records == 0
    assert store.diagnostics.output_records == 0
    await coordinator.close()


async def test_agent_turn_propagates_typed_structured_input_suspension() -> (
    None
):
    """Suspend only after the complete requesting segment is durable."""
    input_arguments = {
        "mode": "required",
        "reason": "Need one bounded decision.",
        "questions": [
            {
                "question_id": "continue",
                "kind": "confirmation",
                "prompt": "Continue?",
                "required": True,
                "choices": [],
                "allow_other": False,
            }
        ],
    }
    scope = authority()
    conversation_id = conversation.ConversationId(
        "conversation-agent-structured-input"
    )
    model_slot = conversation.AgentModelSlot("primary")
    topology_path = conversation.parent_agent_topology_path(
        scope.agent_id,
        model_slot,
    )
    binding_seed = _binding(lane_id="lane-agent-structured-input-seed")
    lane_id = conversation.derive_agent_provider_lane_id(
        conversation_id=conversation_id,
        owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
        topology_path=topology_path,
        model_slot=model_slot,
        binding=binding_seed,
    )
    binding = replace(binding_seed, lane_id=lane_id)
    store = conversation.InMemoryConversationStore()

    async def request_input(arguments: Mapping[str, JsonValue]) -> str:
        page = await store.list_checkpoints(scope, cursor=None, limit=10)
        assert len(page.checkpoints) == 1
        durable = page.checkpoints[0]
        assert durable.kind is (
            conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
        )
        assert durable.lifecycle is conversation.CheckpointLifecycle.COMMITTED
        assert durable.content.lanes[0].lane_id == lane_id
        assert (
            durable.content.visible_transcript.entries[0].content
            == "need structured input"
        )
        segment = durable.content.execution_segments[0]
        assert segment.phase is (
            conversation.ProviderExecutionSegmentPhase.PROVIDER_RESPONSE
        )
        assert segment.tools[0].phase is (
            conversation.ToolExecutionPhase.REQUESTED
        )
        return await conversation.request_agent_structured_input(arguments)

    tool = conversation.NativeOpenAIFunctionTool(
        name="request_user_input",
        parameters={"type": "object"},
        handler=request_input,
        effect_policy=conversation.ToolEffectPolicy.PURE,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json=_response(
                "structured-input-response",
                [
                    _function_call(
                        "structured-input-call",
                        "call-input",
                        name="request_user_input",
                        arguments=dumps(
                            input_arguments,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    )
                ],
            ),
        )

    provider = _provider(binding, handler, tools=(tool,))
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
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
    )
    topology = conversation.AgentLaneTopology(
        conversation_id=conversation_id,
        lanes=(
            conversation.AgentProviderLane(
                owner_kind=conversation.ProviderLaneOwnerKind.PARENT_AGENT,
                agent_id=scope.agent_id,
                topology_path=topology_path,
                model_slot=model_slot,
                binding=binding,
                retention_policy=(
                    conversation.ChildLaneRetentionPolicy.RETAIN
                ),
            ),
        ),
    )
    turn = conversation.AgentConversationTurn(
        coordinator=coordinator,
        authority=scope,
        topology=topology,
        lanes=(
            conversation.AgentConversationLane(
                lane_id=lane_id,
                mode=conversation.ConversationMode.STATELESS,
            ),
        ),
        logical_turn_id=conversation.LogicalTurnId(
            "agent-structured-input-turn"
        ),
        execution_segment_id=conversation.ExecutionSegmentId(
            "agent-structured-input-outward-segment"
        ),
        checkpoint_id=conversation.CheckpointId(
            "agent-structured-input-outward-checkpoint"
        ),
        branch_id=conversation.ConversationBranchId(
            "agent-structured-input-branch"
        ),
        provisional_response_id=conversation.ProvisionalResponseId(
            "agent-structured-input-provisional"
        ),
        public_response_id=conversation.PublicResponseId(
            "agent-structured-input-response"
        ),
        idempotency_key=conversation.RequestIdempotencyKey(
            "agent-structured-input-key"
        ),
        retention=retention(),
    )

    with pytest.raises(
        conversation.AgentConversationSuspensionBoundary
    ) as raised:
        await turn.execute("need structured input")

    boundary = raised.value
    checkpoint = boundary.checkpoint
    assert boundary.request.arguments == conversation.freeze_json_value(
        input_arguments
    )
    assert boundary.tool.arguments == boundary.request.arguments
    assert checkpoint.kind is (
        conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION
    )
    assert checkpoint.lifecycle is conversation.CheckpointLifecycle.STAGED
    assert checkpoint.identity.parent_checkpoint_id is not None
    assert checkpoint.content.lanes[0].lifecycle is (
        conversation.ProviderLaneLifecycle.SUSPENDED
    )
    assert checkpoint.content.lane_topology == topology.checkpoint_topology()
    assert store.diagnostics.checkpoints == 1
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(
            conversation.CheckpointId(
                "agent-structured-input-outward-checkpoint"
            ),
            scope,
        )

    revision_binding = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("openai"),
        model_id=ModelId("coordinated-model"),
        provider_config_revision=ProviderConfigRevision("provider-r1"),
        model_config_revision=ModelConfigRevision("model-r1"),
        capability_revision=CapabilityRevision("capability-r1"),
    )
    registry = ContinuationSnapshotCodecRegistry(
        "coordinated-staging-registry"
    )
    registry.register(
        codec_id="unused-provider-codec",
        revision_binding=revision_binding,
        snapshot_kind="unused-provider-snapshot",
        export_snapshot=encode_continuation_snapshot,
        restore_snapshot=lambda value, expected: decode_continuation_snapshot(
            value,
            expected_binding=expected,
        ),
    )
    capability = ModelCapabilityCatalog.create(
        support=ProviderCapabilitySupport(
            provider_family=ProviderFamilyName("openai"),
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
            continuation_snapshot_codec_registry=registry,
            continuation_snapshot_codec=registry.reference(
                "unused-provider-codec"
            ),
        ),
        revision_binding=revision_binding,
    )
    principal = PrincipalScope()
    runtime = DurableInteractionRuntime(
        actor=InteractionActor(principal=principal),
        stager=PortableAgentContinuationStager(
            clock=lambda: datetime(2026, 8, 2, 12, 1, tzinfo=UTC)
        ),
    )
    execution = await create_agent_execution(
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://coordinated-staging",
            agent_definition_revision="agent-r1",
            operation_id="coordinated-operation",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capability-r1",
        ),
        agent_id=AgentId(str(scope.agent_id)),
        principal=principal,
        initial_messages=(),
        interaction_runtime=runtime,
    )
    context = ModelCallContext(
        specification=Specification(
            role=None,
            goal=None,
            input_type=InputType.TEXT,
        ),
        input="need structured input",
        capability=capability,
        execution=execution,
        conversation_turn=turn,
        conversation_input="need structured input",
    )
    engine_agent = object.__new__(_StagingEngineAgent)
    with pytest.raises(ExecutionInputRequiredError) as staged:
        await engine_agent._stage_conversation_input_required(
            context,
            GenerationSettings(),
            boundary,
        )

    error = staged.value
    assert execution.status is AgentExecutionStatus.INPUT_REQUIRED
    assert error.durable is not None
    assert error.durable.continuation.provider_snapshot is None
    reference = error.durable.continuation.conversation_checkpoint_reference
    assert reference is not None
    assert error.checkpoint_id == reference.checkpoint_id
    assert error.conversation_unit is not None
    conversation_unit = cast(
        conversation.ConversationUnitOfWork,
        error.conversation_unit,
    )
    committed = await conversation_unit.commit()
    assert committed.identity.checkpoint_id == (
        boundary.checkpoint.identity.checkpoint_id
    )
    restarted = await store.load(
        boundary.checkpoint.identity.checkpoint_id,
        scope,
    )
    assert restarted.kind is (
        conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION
    )
    assert restarted.lifecycle is conversation.CheckpointLifecycle.COMMITTED
    assert reference.checkpoint_id == str(restarted.identity.checkpoint_id)
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
        {
            "encrypted_content": "branch-private-root",
            "id": "branch-reasoning-root",
            "summary": [],
            "type": "reasoning",
        },
        {
            "content": [
                {
                    "annotations": [],
                    "text": "answer root",
                    "type": "output_text",
                }
            ],
            "id": "branch-message-root",
            "role": "assistant",
            "type": "message",
        },
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


async def test_branches_reuse_one_multi_tool_parent_ledger_once() -> None:
    """Branch after two tools without retaining a second replay transcript."""
    requests: list[dict[str, object]] = []
    tool_effects: list[Mapping[str, JsonValue]] = []

    async def lookup(arguments: Mapping[str, JsonValue]) -> str:
        tool_effects.append(arguments)
        return f"tool-output-{len(tool_effects)}"

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=lookup,
        effect_policy=conversation.ToolEffectPolicy.IDEMPOTENT,
    )
    root_responses = (
        _response(
            "multi-tool-response-1",
            [
                _reasoning("multi-tool-reasoning-1", "multi-private-1"),
                _function_call("multi-tool-call-1", "multi-call-1"),
            ],
        ),
        _response(
            "multi-tool-response-2",
            [
                _reasoning("multi-tool-reasoning-2", "multi-private-2"),
                _function_call("multi-tool-call-2", "multi-call-2"),
            ],
        ),
        _response(
            "multi-tool-response-3",
            [
                _reasoning("multi-tool-reasoning-3", "multi-private-3"),
                _message("multi-tool-message", "multi-tool parent"),
            ],
        ),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        body = cast(dict[str, object], loads(await request.aread()))
        requests.append(body)
        if len(requests) <= len(root_responses):
            return httpx.Response(200, json=root_responses[len(requests) - 1])
        suffix = "left" if "branch left" in dumps(body) else "right"
        return httpx.Response(
            200,
            json=_response(
                f"multi-tool-branch-{suffix}",
                [_message(f"multi-tool-message-{suffix}", suffix)],
            ),
        )

    provider = _provider(
        _binding(lane_id="lane-multi-tool-branch"),
        handler,
        tools=(tool,),
    )
    client, coordinator, store = _direct_client(
        provider,
        namespace="multi-tool-branch",
    )
    root = await client.create(
        "multi-tool root",
        avalan.StatelessConversationSettings(),
    )
    assert type(root.handle) is avalan.StatelessConversationHandle
    parent = avalan.StatelessParent(handle=root.handle)
    parent_bytes = conversation.ConversationCheckpointCodec().encode(
        await store.load(parent.handle.checkpoint_id, authority())
    )

    async def branch(branch_id: str) -> avalan.DirectConversationResult:
        intent = avalan.ConversationBranchIntent(
            parent=parent,
            branch_id=conversation.ConversationBranchId(branch_id),
        )
        return await client.branch(
            f"branch {branch_id}",
            avalan.StatelessConversationSettings(
                parent=parent,
                branch=intent,
            ),
        )

    left, right = await gather(branch("left"), branch("right"))

    assert {left.output, right.output} == {"left", "right"}
    assert len(tool_effects) == 2
    assert len(requests) == 5
    for request in requests[3:]:
        replay = cast(list[dict[str, object]], request["input"])
        assert [item["type"] for item in replay] == [
            "reasoning",
            "function_call",
            "function_call_output",
            "reasoning",
            "function_call",
            "function_call_output",
            "reasoning",
            "message",
            "message",
        ]
        assert [
            item.get("call_id")
            for item in replay
            if item["type"] in {"function_call", "function_call_output"}
        ] == ["multi-call-1", "multi-call-1", "multi-call-2", "multi-call-2"]
    assert parent_bytes == conversation.ConversationCheckpointCodec().encode(
        await store.load(parent.handle.checkpoint_id, authority())
    )
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
