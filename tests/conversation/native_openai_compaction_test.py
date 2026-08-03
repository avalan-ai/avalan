"""Verify exact native inline and standalone compaction wire behavior."""

from asyncio import CancelledError
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from json import dumps, loads
from typing import cast

import httpx
import pytest
from openai import APIResponseValidationError, AsyncOpenAI
from openai.types.responses import CompactedResponse
from phase2_fixtures import authority, retention

import avalan
import avalan.conversation as conversation
from avalan.conversation import codec as codec_module
from avalan.conversation import coordinator as coordinator_module
from avalan.conversation.providers import openai as provider_module
from avalan.types import JsonValue

pytestmark = pytest.mark.anyio

_STATELESS_ADAPTER = (
    "avalan.conversation.providers.openai.NativeOpenAIStatelessProvider"
)
_STORED_ADAPTER = (
    "avalan.conversation.providers.openai_stored.NativeOpenAIStoredProvider"
)


@pytest.fixture
def anyio_backend() -> str:
    """Run exact native compact tests on asyncio only."""
    return "asyncio"


def _ignore_acceptance_evidence(name: str, value: object) -> None:
    """Validate and discard delegated acceptance evidence."""
    assert name == "conversation_acceptance_evidence"
    assert isinstance(value, str)


def _binding(
    lane_id: str,
    *,
    stored: bool = False,
    streaming: bool = False,
) -> conversation.ProviderLaneBinding:
    return conversation.ProviderLaneBinding(
        lane_id=conversation.ProviderLaneId(lane_id),
        adapter_type=_STORED_ADAPTER if stored else _STATELESS_ADAPTER,
        provider_family=conversation.ProviderFamily.OPENAI,
        normalized_endpoint="https://api.openai.com/v1",
        model_or_deployment="gpt-5",
        provider_api_revision=conversation.ProviderApiRevision(
            "openapi-2.3.0"
        ),
        sdk_revision=conversation.ProviderSdkRevision("openai-python-2.42.0"),
        model_configuration_revision=(
            conversation.ModelConfigurationRevision("model-config-compact")
        ),
        capability_profile_revision=(
            conversation.CapabilityProfileRevision("capability-compact")
        ),
        tool_schema_revision=conversation.ToolSchemaRevision("tools-compact"),
        execution_definition_revision=(
            conversation.ExecutionDefinitionRevision("execution-compact")
        ),
        continuation_codec_version=conversation.ConversationCodecVersion(1),
        transport=(
            conversation.ProviderTransport.STREAMING
            if streaming
            else conversation.ProviderTransport.NON_STREAMING
        ),
        agent_id=conversation.ConversationAgentId("agent-compact"),
        execution_definition_digest=None,
        compaction_policy_digest=(
            conversation.native_openai_compaction_policy_digest(_limits())
        ),
    )


def _capabilities(
    binding: conversation.ProviderLaneBinding,
    *,
    stored: bool = False,
    compaction: bool = True,
) -> conversation.ConversationCapabilityProfile:
    capability = conversation.ConversationCapability
    supported = {
        capability.REASONING_CONTEXT_CURRENT_TURN,
        capability.REASONING_CONTEXT_ALL_TURNS,
        (
            capability.STORED_RESPONSES_CHAINING
            if stored
            else capability.STATELESS_ENCRYPTED_REASONING_REPLAY
        ),
    }
    if compaction:
        supported.add(conversation.ConversationCapability.INLINE_COMPACTION)
        if not stored:
            supported.add(
                conversation.ConversationCapability.STANDALONE_COMPACTION
            )
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
                    (f"compact-{capability.value}",)
                    if capability in supported
                    else ()
                ),
            )
            for capability in conversation.ConversationCapability
        ),
        test_only=True,
    )


def _limits() -> conversation.NativeOpenAICompactionLimits:
    return conversation.NativeOpenAICompactionLimits(
        min_compact_threshold=64,
        max_compact_threshold=4_096,
        max_input_items=32,
        max_input_bytes=65_536,
        max_output_items=32,
        max_output_bytes=65_536,
    )


def _binding_with_limits(
    lane_id: str,
    limits: conversation.NativeOpenAICompactionLimits,
    *,
    stored: bool = False,
    streaming: bool = False,
) -> conversation.ProviderLaneBinding:
    """Return one exact native binding for the selected limit policy."""
    return replace(
        _binding(lane_id, stored=stored, streaming=streaming),
        compaction_policy_digest=(
            conversation.native_openai_compaction_policy_digest(limits)
        ),
    )


@pytest.mark.parametrize(
    "field",
    (
        "min_compact_threshold",
        "max_compact_threshold",
        "max_input_items",
        "max_input_bytes",
        "max_input_depth",
        "max_output_items",
        "max_output_bytes",
        "max_output_depth",
    ),
)
def test_compaction_limit_policy_is_codec_bound_and_drift_closed(
    field: str,
    record_property: Callable[[str, object], None],
) -> None:
    """Bind every exact limit through durable lane identity."""
    record_property("conversation_acceptance_evidence", "contract")
    limits = _limits()
    binding = _binding(f"lane-policy-{field}")
    decoded = codec_module._decode_binding(
        codec_module._encode_binding(binding)
    )
    assert decoded == binding
    assert decoded.compaction_policy_digest == (
        conversation.native_openai_compaction_policy_digest(limits)
    )

    value = getattr(limits, field)
    drifted_limits = replace(limits, **{field: value - 1})
    drifted_digest = conversation.native_openai_compaction_policy_digest(
        drifted_limits
    )
    assert drifted_digest != binding.compaction_policy_digest
    with pytest.raises(conversation.ConversationBindingDriftError):
        conversation.NativeOpenAIStatelessProfile(
            profile_id=f"drift-{field}",
            binding=decoded,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
            compaction_limits=drifted_limits,
        )
    with pytest.raises(conversation.ConversationBindingDriftError):
        decoded.assert_compatible(
            replace(decoded, compaction_policy_digest=drifted_digest)
        )


def _usage() -> dict[str, object]:
    return {
        "input_tokens": 7,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 3,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 10,
    }


def _response(
    response_id: str,
    output: list[dict[str, object]],
    *,
    stored: bool = False,
    previous_response_id: str | None = None,
    reasoning_context: str = "current_turn",
) -> dict[str, object]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": "gpt-5",
        "output": output,
        "parallel_tool_calls": False,
        "previous_response_id": previous_response_id,
        "reasoning": {"context": reasoning_context},
        "store": stored,
        "temperature": None,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "truncation": "disabled",
        "usage": _usage(),
    }


def _message(identifier: str, text: str) -> dict[str, object]:
    return {
        "id": identifier,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _compact_item(identifier: str, opaque: str) -> dict[str, object]:
    return {
        "created_by": "provider-compact",
        "encrypted_content": opaque,
        "id": identifier,
        "type": "compaction",
    }


def _input_message(text: str) -> dict[str, object]:
    return {
        "content": [{"text": text, "type": "input_text"}],
        "role": "user",
        "type": "message",
    }


def _reasoning() -> conversation.EffectiveReasoningMetadata:
    return conversation.EffectiveReasoningMetadata(
        requested=conversation.ReasoningContext.CURRENT_TURN,
        effective=conversation.EffectiveReasoningContext.CURRENT_TURN,
    )


def _reasoning_context(
    requested: conversation.ReasoningContext,
) -> conversation.EffectiveReasoningMetadata:
    effective = {
        conversation.ReasoningContext.AUTO: None,
        conversation.ReasoningContext.CURRENT_TURN: (
            conversation.EffectiveReasoningContext.CURRENT_TURN
        ),
        conversation.ReasoningContext.ALL_TURNS: (
            conversation.EffectiveReasoningContext.ALL_TURNS
        ),
    }[requested]
    return conversation.EffectiveReasoningMetadata(
        requested=requested,
        effective=effective,
    )


def _prefix_item(
    lane_id: conversation.ProviderLaneId,
) -> conversation.ProviderItem:
    return conversation.ProviderItem(
        item_id=conversation.ProviderItemId("prefix-message"),
        lane_id=lane_id,
        model_call_id=conversation.ConversationModelCallId("prefix-call"),
        kind=conversation.ProviderItemKind.MESSAGE,
        order=conversation.ProviderItemOrder(0),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.FINAL,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input=cast(
            dict[str, JsonValue],
            _message("prefix-message", "prefix"),
        ),
        normalization_version=conversation.ConversationCodecVersion(1),
    )


def _client(
    binding: conversation.ProviderLaneBinding,
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key="compact-key",
        base_url=binding.normalized_endpoint,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )


def _stateless_provider(
    binding: conversation.ProviderLaneBinding,
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
    *,
    compaction: bool = True,
    limits: conversation.NativeOpenAICompactionLimits | None = None,
) -> conversation.NativeOpenAIStatelessProvider:
    return conversation.NativeOpenAIStatelessProvider(
        client=_client(binding, handler),
        profile=conversation.NativeOpenAIStatelessProfile(
            profile_id=f"compact-{binding.lane_id}",
            binding=binding,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
            compaction_limits=limits or _limits(),
        ),
        capability_profile=_capabilities(
            binding,
            compaction=compaction,
        ),
    )


def _stored_provider(
    binding: conversation.ProviderLaneBinding,
    handler: Callable[[httpx.Request], Coroutine[None, None, httpx.Response]],
    *,
    limits: conversation.NativeOpenAICompactionLimits,
) -> conversation.NativeOpenAIStoredProvider:
    """Return one exact stored provider for a limit-policy test."""
    execution = conversation.NativeOpenAIStoredExecution(
        instructions="Use the frozen stored execution definition.",
        max_output_tokens=256,
        max_tool_calls=4,
    )
    bound = replace(
        binding,
        execution_definition_digest=(
            conversation.native_openai_stored_execution_digest(
                binding=binding,
                execution=execution,
                encrypted_content=(
                    conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
                ),
                compaction_limits=limits,
            )
        ),
    )
    return conversation.NativeOpenAIStoredProvider(
        client=_client(bound, handler),
        profile=conversation.NativeOpenAIStoredProfile(
            profile_id=f"compact-{bound.lane_id}",
            binding=bound,
            execution=execution,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
            compaction_limits=limits,
        ),
        capability_profile=_capabilities(bound, stored=True),
    )


async def test_compaction_preflight_and_policy_validation_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject malformed policies, bounded output, and invalid preflights."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.native_openai_compaction_policy_digest(
            cast(conversation.NativeOpenAICompactionLimits, object())
        )

    limits = _limits()
    stateless_binding = _binding_with_limits(
        "lane-compact-preflight-stateless",
        limits,
    )

    async def unused(request: httpx.Request) -> httpx.Response:
        await request.aread()
        raise AssertionError("preflight must not dispatch")

    stateless = _stateless_provider(
        stateless_binding,
        unused,
        limits=limits,
    )
    stateless_plan = conversation.StatelessProviderPlan(
        binding=stateless_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=stateless_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(_prefix_item(stateless_binding.lane_id),),
        ),
        reasoning=_reasoning(),
    )
    compact_plan = conversation.StandaloneCompactProviderPlan(
        binding=stateless_binding,
        ledger=stateless_plan.ledger,
        reasoning=stateless_plan.reasoning,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        stateless.validate_compaction_request(
            compact_plan,
            conversation.ProviderTransport.STREAMING,
        )
    with pytest.raises(conversation.ConversationValidationError):
        stateless.validate_compaction_request(
            stateless_plan,
            conversation.ProviderTransport.NON_STREAMING,
        )

    depth_limits = replace(limits, max_output_depth=1)
    with pytest.raises(conversation.ConversationLimitError):
        provider_module._bounded_provider_items(
            (cast(JsonValue, _message("depth-output", "nested")),),
            plan=stateless_plan,
            limits=depth_limits,
        )

    payload = cast(
        Mapping[str, JsonValue],
        conversation.freeze_json_value(
            _response(
                "cardinality-response",
                [_message("cardinality-output", "visible")],
            )
        ),
    )

    def truncated_items(
        raw_output: tuple[JsonValue, ...],
        *,
        plan: provider_module.NativeOpenAIResponsePlan,
        limits: conversation.NativeOpenAICompactionLimits | None,
    ) -> tuple[conversation.ProviderItem, ...]:
        assert raw_output and plan is stateless_plan and limits is None
        return ()

    monkeypatch.setattr(
        provider_module,
        "_bounded_provider_items",
        truncated_items,
    )
    with pytest.raises(conversation.ConversationProviderResponseError):
        provider_module._provider_result_mapping(payload, stateless_plan)

    stored = _stored_provider(
        _binding_with_limits(
            "lane-compact-preflight-stored",
            limits,
            stored=True,
        ),
        unused,
        limits=limits,
    )
    with pytest.raises(conversation.ConversationBindingDriftError):
        conversation.NativeOpenAIStoredProfile(
            profile_id="compact-stored-policy-drift",
            binding=replace(
                stored.binding,
                compaction_policy_digest=conversation.IntegrityDigest(
                    "0" * 64
                ),
            ),
            execution=stored._profile.execution,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
            compaction_limits=limits,
        )
    stored_plan = conversation.StoredProviderPlan(
        binding=stored.binding,
        upstream_response_id=conversation.UpstreamResponseId(
            "stored-preflight-parent"
        ),
        reasoning=_reasoning(),
        new_input={"text": "stored preflight"},
    )
    with pytest.raises(conversation.ConversationValidationError):
        stored.validate_compaction_request(
            stored_plan,
            conversation.ProviderTransport.NON_STREAMING,
        )
    await stateless.aclose()
    await stored.aclose()


@pytest.mark.parametrize(
    ("adapter", "operation", "failure"),
    (
        ("stateless", "stream", "cancellation"),
        ("stateless", "stream", "response"),
        ("stored", "dispatch", "cancellation"),
        ("stored", "stream", "cancellation"),
        ("stored", "stream", "response"),
    ),
)
async def test_inline_preoutput_failures_are_counted(
    adapter: str,
    operation: str,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count inline cancellation and typed pre-output failures exactly once."""
    streaming = operation == "stream"
    limits = _limits()
    binding = _binding_with_limits(
        f"lane-{adapter}-{operation}-{failure}",
        limits,
        stored=adapter == "stored",
        streaming=streaming,
    )

    async def unused(request: httpx.Request) -> httpx.Response:
        await request.aread()
        raise AssertionError("mocked SDK call must not reach HTTP")

    provider = (
        _stored_provider(binding, unused, limits=limits)
        if adapter == "stored"
        else _stateless_provider(binding, unused, limits=limits)
    )

    async def failed_create(*args: object, **kwargs: object) -> object:
        assert args or kwargs
        if failure == "cancellation":
            raise CancelledError()
        return object()

    monkeypatch.setattr(provider._client.responses, "create", failed_create)
    if adapter == "stored":
        plan: conversation.ProviderPlan = conversation.StoredProviderPlan(
            binding=provider.binding,
            upstream_response_id=conversation.UpstreamResponseId(
                "stored-inline-failure-parent"
            ),
            reasoning=_reasoning(),
            compaction=conversation.InlineCompaction(compact_threshold=128),
            new_input={"text": "stored inline failure"},
        )
    else:
        plan = conversation.StatelessProviderPlan(
            binding=provider.binding,
            ledger=conversation.ProviderItemLedger(
                lane_id=provider.binding.lane_id,
                normalization_version=(
                    conversation.ConversationCodecVersion(1)
                ),
                items=(_prefix_item(provider.binding.lane_id),),
            ),
            reasoning=_reasoning(),
            compaction=conversation.InlineCompaction(compact_threshold=128),
        )
    expected = (
        CancelledError
        if failure == "cancellation"
        else conversation.ConversationProviderResponseError
    )
    with pytest.raises(expected):
        if operation == "dispatch":
            await provider.dispatch(plan)
        else:
            await provider.stream(plan)
    assert provider.diagnostics.compaction_failure_count == 1
    await provider.aclose()


async def test_native_inline_latest_replay_and_standalone_wire(
    record_property: Callable[[str, object], None],
) -> None:
    """Send typed inline config and canonical standalone compact context."""
    record_property("conversation_acceptance_evidence", "wire")
    requests: list[tuple[str, dict[str, object]]] = []
    inline_output = [
        _compact_item("compact-inline", "inline-private"),
        _message("inline-message", "after compact"),
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], loads(await request.aread()))
        requests.append((request.url.path, payload))
        if request.url.path.endswith("/responses/compact"):
            return httpx.Response(
                200,
                json={
                    "id": "compact-response",
                    "created_at": 2,
                    "object": "response.compaction",
                    "output": [
                        _input_message("retained user input"),
                        _compact_item(
                            "standalone-compact",
                            "standalone-private",
                        ),
                    ],
                    "usage": _usage(),
                },
            )
        output = (
            inline_output
            if len(
                [path for path, _ in requests if path.endswith("/responses")]
            )
            == 1
            else [_message("second-message", "second")]
        )
        return httpx.Response(
            200,
            json=_response(f"response-{len(requests)}", output),
        )

    binding = _binding("lane-compact-native")
    provider = _stateless_provider(binding, handler)
    prefix = _prefix_item(binding.lane_id)
    first_plan = conversation.StatelessProviderPlan(
        binding=binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(prefix,),
        ),
        reasoning=_reasoning(),
        compaction=conversation.InlineCompaction(compact_threshold=128),
        new_input={"text": "inline input"},
    )
    inline = await provider.dispatch(first_plan)
    assert [item.kind for item in inline.items] == [
        conversation.ProviderItemKind.COMPACTION,
        conversation.ProviderItemKind.MESSAGE,
    ]
    continued_ledger = conversation.ProviderItemLedger(
        lane_id=binding.lane_id,
        normalization_version=conversation.ConversationCodecVersion(1),
        items=(prefix, *inline.items),
    )
    await provider.dispatch(
        conversation.StatelessProviderPlan(
            binding=binding,
            ledger=continued_ledger,
            reasoning=_reasoning(),
            new_input={"text": "next input"},
        )
    )
    standalone = await provider.compact(
        conversation.StandaloneCompactProviderPlan(
            binding=binding,
            ledger=continued_ledger,
            reasoning=_reasoning(),
        )
    )

    inline_request = requests[0][1]
    assert inline_request["context_management"] == [
        {"type": "compaction", "compact_threshold": 128}
    ]
    for _, payload in requests[1:]:
        compact_input = cast(list[dict[str, object]], payload["input"])
        assert [item["id"] for item in compact_input[:2]] == [
            "compact-inline",
            "inline-message",
        ]
        assert "prefix-message" not in dumps(compact_input)
        assert compact_input[0]["encrypted_content"] == "inline-private"
        assert "created_by" not in compact_input[0]
    assert requests[2][0] == "/v1/responses/compact"
    assert standalone.items[0].caller is conversation.ProviderItemCaller.CALLER
    assert standalone.items[0].canonical_input == {
        "content": ({"text": "retained user input", "type": "input_text"},),
        "role": "user",
        "type": "message",
    }
    assert standalone.items[1].opaque_state is not None
    assert standalone.items[1].opaque_state.digest
    assert "standalone-private" not in repr(standalone)
    diagnostics = provider.diagnostics
    assert diagnostics.inline_compaction_request_count == 1
    assert diagnostics.standalone_compaction_request_count == 1
    assert diagnostics.compaction_boundary_count == 2
    assert diagnostics.last_compact_threshold == 128
    await provider.aclose()


async def test_coordinator_routes_only_stateless_native_compact() -> None:
    """Route native compact once and reject a stored-native runtime."""

    async def compact_handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            json={
                "created_at": 2,
                "id": "coordinator-compact-response",
                "object": "response.compaction",
                "output": [
                    _compact_item(
                        "coordinator-compact-boundary",
                        "coordinator-private-state",
                    )
                ],
                "usage": _usage(),
            },
        )

    binding = _binding("lane-coordinator-compact")
    provider = _stateless_provider(binding, compact_handler)
    runtime = conversation.NativeOpenAIConversationLaneRuntime(
        provider=provider
    )
    coordinator = conversation.RunScopedConversationCoordinator(
        store=conversation.InMemoryConversationStore(),
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            authority()
        ),
        clock=conversation.DeterministicFakeClock(
            datetime(2026, 8, 2, tzinfo=UTC)
        ),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(runtime,),
    )
    plan = conversation.StandaloneCompactProviderPlan(
        binding=binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(_prefix_item(binding.lane_id),),
        ),
        reasoning=_reasoning(),
    )
    result = await coordinator._dispatch_with_retry(
        runtime,
        plan,
        coordinator_module._AttemptStaging(
            lane_id=binding.lane_id,
            items=[],
        ),
        streaming=False,
        progress=coordinator_module._DispatchProgress(),
        sink=None,
    )
    assert result.items[-1].kind is conversation.ProviderItemKind.COMPACTION

    execution = conversation.NativeOpenAIStoredExecution(
        instructions="Use the frozen stored execution definition.",
        max_output_tokens=256,
        max_tool_calls=4,
    )
    limits = _limits()
    stored_base = _binding("lane-coordinator-stored", stored=True)
    stored_binding = replace(
        stored_base,
        execution_definition_digest=(
            conversation.native_openai_stored_execution_digest(
                binding=stored_base,
                execution=execution,
                encrypted_content=(
                    conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
                ),
                compaction_limits=limits,
            )
        ),
    )
    stored_provider = conversation.NativeOpenAIStoredProvider(
        client=_client(stored_binding, _unused),
        profile=conversation.NativeOpenAIStoredProfile(
            profile_id="coordinator-stored-compact",
            binding=stored_binding,
            execution=execution,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
            compaction_limits=limits,
        ),
        capability_profile=_capabilities(stored_binding, stored=True),
    )
    stored_runtime = conversation.NativeOpenAIStoredLaneRuntime(
        provider=stored_provider
    )
    stored_plan = conversation.StandaloneCompactProviderPlan(
        binding=stored_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=stored_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(_prefix_item(stored_binding.lane_id),),
        ),
        reasoning=_reasoning(),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await coordinator._dispatch_with_retry(
            stored_runtime,
            stored_plan,
            coordinator_module._AttemptStaging(
                lane_id=stored_binding.lane_id,
                items=[],
            ),
            streaming=False,
            progress=coordinator_module._DispatchProgress(),
            sink=None,
        )
    await stored_provider.aclose()
    await coordinator.close()


async def test_inline_no_boundary_preserves_complete_replay_context() -> None:
    """Keep the full canonical ledger when inline output has no boundary."""
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], loads(await request.aread()))
        requests.append(payload)
        output = (
            [_message("no-boundary-message", "first")]
            if len(requests) == 1
            else [_message("no-boundary-next", "second")]
        )
        return httpx.Response(
            200,
            json=_response(f"no-boundary-{len(requests)}", output),
        )

    binding = _binding("lane-compact-no-boundary")
    provider = _stateless_provider(binding, handler)
    prefix = _prefix_item(binding.lane_id)
    first = await provider.dispatch(
        conversation.StatelessProviderPlan(
            binding=binding,
            ledger=conversation.ProviderItemLedger(
                lane_id=binding.lane_id,
                normalization_version=conversation.ConversationCodecVersion(1),
                items=(prefix,),
            ),
            reasoning=_reasoning(),
            compaction=conversation.InlineCompaction(compact_threshold=128),
            new_input={"text": "first input"},
        )
    )
    await provider.dispatch(
        conversation.StatelessProviderPlan(
            binding=binding,
            ledger=conversation.ProviderItemLedger(
                lane_id=binding.lane_id,
                normalization_version=conversation.ConversationCodecVersion(1),
                items=(prefix, *first.items),
            ),
            reasoning=_reasoning(),
            new_input={"text": "second input"},
        )
    )

    assert requests[0]["context_management"] == [
        {"type": "compaction", "compact_threshold": 128}
    ]
    replay = cast(list[dict[str, object]], requests[1]["input"])
    assert [item.get("id") for item in replay[:2]] == [
        "prefix-message",
        "no-boundary-message",
    ]
    assert "context_management" not in requests[1]
    assert provider.diagnostics.inline_compaction_request_count == 1
    assert provider.diagnostics.compaction_boundary_count == 0
    await provider.aclose()


async def test_unproven_or_out_of_range_compaction_rejects_before_wire() -> (
    None
):
    """Fail closed before dispatch for incapable or unsafe profiles."""
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        return httpx.Response(500)

    binding = _binding("lane-compact-incapable")
    provider = _stateless_provider(binding, handler, compaction=False)
    ledger = conversation.ProviderItemLedger(
        lane_id=binding.lane_id,
        normalization_version=conversation.ConversationCodecVersion(1),
        items=(_prefix_item(binding.lane_id),),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.dispatch(
            conversation.StatelessProviderPlan(
                binding=binding,
                ledger=ledger,
                reasoning=_reasoning(),
                compaction=conversation.InlineCompaction(
                    compact_threshold=128
                ),
            )
        )
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.compact(
            conversation.StandaloneCompactProviderPlan(
                binding=binding,
                ledger=ledger,
                reasoning=_reasoning(),
            )
        )
    capable = _stateless_provider(
        _binding("lane-compact-range"),
        handler,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await capable.dispatch(
            conversation.StatelessProviderPlan(
                binding=capable.binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=capable.binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(capable.binding.lane_id),),
                ),
                reasoning=_reasoning(),
                compaction=conversation.InlineCompaction(compact_threshold=63),
            )
        )
    forged_plan = conversation.StatelessProviderPlan(
        binding=capable.binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=capable.binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(_prefix_item(capable.binding.lane_id),),
        ),
        reasoning=_reasoning(),
    )
    object.__setattr__(forged_plan, "compaction", object())
    with pytest.raises(conversation.ConversationCapabilityError):
        await capable.dispatch(forged_plan)

    streaming_binding = _binding("lane-compact-streaming", streaming=True)
    streaming_provider = _stateless_provider(streaming_binding, handler)
    with pytest.raises(conversation.ConversationBindingDriftError):
        await streaming_provider.compact(
            conversation.StandaloneCompactProviderPlan(
                binding=streaming_binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=streaming_binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(streaming_binding.lane_id),),
                ),
                reasoning=_reasoning(),
            )
        )
    await streaming_provider.aclose()

    no_limits_binding = replace(
        _binding("lane-compact-no-limits"),
        compaction_policy_digest=None,
    )
    no_limits_provider = conversation.NativeOpenAIStatelessProvider(
        client=_client(no_limits_binding, handler),
        profile=conversation.NativeOpenAIStatelessProfile(
            profile_id="compact-no-limits",
            binding=no_limits_binding,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
        ),
        capability_profile=_capabilities(no_limits_binding),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await no_limits_provider.compact(
            conversation.StandaloneCompactProviderPlan(
                binding=no_limits_binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=no_limits_binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(no_limits_binding.lane_id),),
                ),
                reasoning=_reasoning(),
            )
        )
    await no_limits_provider.aclose()

    closed_binding = _binding("lane-compact-closed")
    closed_provider = _stateless_provider(closed_binding, handler)
    await closed_provider.aclose()
    with pytest.raises(conversation.ConversationCapabilityError):
        await closed_provider.compact(
            conversation.StandaloneCompactProviderPlan(
                binding=closed_binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=closed_binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(closed_binding.lane_id),),
                ),
                reasoning=_reasoning(),
            )
        )
    assert dispatches == 0
    await provider.aclose()
    await capable.aclose()


async def _verify_streamed_compaction_done_items() -> (
    tuple[conversation.ProviderItem, ...]
):
    """Ignore fragments and return the same complete canonical boundary."""
    binding = _binding("lane-compact-stream", streaming=True)
    output = [
        _compact_item("stream-compact", "stream-private"),
        _message("stream-message", "stream answer"),
    ]
    response = _response("stream-response", output)
    events = [
        {
            "type": "response.output_item.added",
            "sequence_number": 0,
            "output_index": 0,
            "item": {
                "id": "stream-compact",
                "type": "compaction",
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
    body = "".join(f"data: {dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], loads(await request.aread()))
        assert payload["context_management"] == [
            {"type": "compaction", "compact_threshold": 256}
        ]
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    provider = _stateless_provider(binding, handler)
    stream = await provider.stream(
        conversation.StatelessProviderPlan(
            binding=binding,
            ledger=conversation.ProviderItemLedger(
                lane_id=binding.lane_id,
                normalization_version=conversation.ConversationCodecVersion(1),
                items=(_prefix_item(binding.lane_id),),
            ),
            reasoning=_reasoning(),
            compaction=conversation.InlineCompaction(compact_threshold=256),
        )
    )
    streamed = tuple([item async for item in stream])
    terminal = await stream.terminal()
    assert streamed == terminal.items
    assert streamed[0].kind is conversation.ProviderItemKind.COMPACTION
    await stream.aclose()
    await provider.aclose()
    return streamed


async def test_streamed_compaction_commits_complete_done_items(
    record_property: Callable[[str, object], None],
) -> None:
    """Ignore fragments and return the same complete canonical boundary."""
    record_property("conversation_acceptance_evidence", "wire")
    await _verify_streamed_compaction_done_items()


async def test_native_streamed_compaction_boundary_remains_exact(
    record_property: Callable[[str, object], None],
) -> None:
    """Preserve exact native SSE compaction done-item fidelity."""
    record_property("conversation_acceptance_evidence", "wire")
    assert tuple(
        item.kind for item in await _verify_streamed_compaction_done_items()
    ) == (
        conversation.ProviderItemKind.COMPACTION,
        conversation.ProviderItemKind.MESSAGE,
    )


async def _verify_malformed_stream(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject an incomplete streamed compact item before terminal state."""
    record_property("conversation_acceptance_evidence", "negative")
    binding = _binding("lane-compact-stream-malformed", streaming=True)
    events = (
        {
            "type": "response.output_item.done",
            "sequence_number": 0,
            "output_index": 0,
            "item": {
                "created_by": "private-stream-creator",
                "id": "malformed-stream-compact",
                "type": "compaction",
            },
        },
    )
    body = "".join(f"data: {dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], loads(await request.aread()))
        assert payload["context_management"] == [
            {"type": "compaction", "compact_threshold": 256}
        ]
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    provider = _stateless_provider(binding, handler)
    stream = await provider.stream(
        conversation.StatelessProviderPlan(
            binding=binding,
            ledger=conversation.ProviderItemLedger(
                lane_id=binding.lane_id,
                normalization_version=conversation.ConversationCodecVersion(1),
                items=(_prefix_item(binding.lane_id),),
            ),
            reasoning=_reasoning(),
            compaction=conversation.InlineCompaction(compact_threshold=256),
        )
    )
    with pytest.raises(
        conversation.ConversationProviderResponseError
    ) as error:
        await stream.__aiter__().__anext__()
    assert "private-stream-creator" not in repr(error.value)
    assert provider.diagnostics.failure_boundary == "failure_before_output"
    assert provider.diagnostics.compaction_failure_count == 1
    await stream.aclose()
    await provider.aclose()


async def test_stored_inline_compaction_only_sends_new_input_and_parent() -> (
    None
):
    """Leave stored history to the provider while sending typed policy."""
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = cast(dict[str, object], loads(await request.aread()))
        requests.append(payload)
        return httpx.Response(
            200,
            json=_response(
                "stored-inline-response",
                [_compact_item("stored-inline-compact", "stored-private")],
                stored=True,
                previous_response_id="stored-parent",
            ),
        )

    execution = conversation.NativeOpenAIStoredExecution(
        instructions="Use the frozen stored execution definition.",
        max_output_tokens=256,
        max_tool_calls=4,
    )
    limits = _limits()
    base_binding = _binding("lane-compact-stored", stored=True)
    binding = replace(
        base_binding,
        execution_definition_digest=(
            conversation.native_openai_stored_execution_digest(
                binding=base_binding,
                execution=execution,
                encrypted_content=(
                    conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
                ),
                compaction_limits=limits,
            )
        ),
    )
    provider = conversation.NativeOpenAIStoredProvider(
        client=_client(binding, handler),
        profile=conversation.NativeOpenAIStoredProfile(
            profile_id="compact-stored-inline",
            binding=binding,
            execution=execution,
            encrypted_content=(
                conversation.NativeOpenAIEncryptedContentPolicy.DEFAULT_RETURN
            ),
            compaction_limits=limits,
        ),
        capability_profile=_capabilities(
            binding,
            stored=True,
        ),
    )
    result = await provider.dispatch(
        conversation.StoredProviderPlan(
            binding=binding,
            upstream_response_id=conversation.UpstreamResponseId(
                "stored-parent"
            ),
            reasoning=_reasoning(),
            compaction=conversation.InlineCompaction(compact_threshold=512),
            new_input={"text": "stored new input"},
        )
    )
    assert len(requests) == 1
    assert requests[0]["input"] == [
        {
            "content": [{"text": "stored new input", "type": "input_text"}],
            "role": "user",
            "type": "message",
        }
    ]
    assert requests[0]["previous_response_id"] == "stored-parent"
    assert requests[0]["context_management"] == [
        {"type": "compaction", "compact_threshold": 512}
    ]
    assert "encrypted_content" not in dumps(requests[0])
    assert result.items[0].kind is conversation.ProviderItemKind.COMPACTION
    assert provider.diagnostics.inline_compaction_request_count == 1
    assert provider.diagnostics.compaction_boundary_count == 1
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.dispatch(
            conversation.StoredProviderPlan(
                binding=binding,
                upstream_response_id=conversation.UpstreamResponseId(
                    "stored-parent"
                ),
                reasoning=_reasoning(),
                compaction=conversation.InlineCompaction(compact_threshold=63),
                new_input={"text": "rejected stored input"},
            )
        )
    forged_plan = conversation.StoredProviderPlan(
        binding=binding,
        upstream_response_id=conversation.UpstreamResponseId("stored-parent"),
        reasoning=_reasoning(),
        new_input={"text": "forged stored input"},
    )
    object.__setattr__(forged_plan, "compaction", object())
    with pytest.raises(conversation.ConversationCapabilityError):
        await provider.dispatch(forged_plan)
    assert len(requests) == 1
    await provider.aclose()


@pytest.mark.parametrize(
    ("requested", "expected_wire", "reported"),
    (
        (conversation.ReasoningContext.AUTO, None, "current_turn"),
        (
            conversation.ReasoningContext.CURRENT_TURN,
            {"context": "current_turn"},
            "current_turn",
        ),
        (
            conversation.ReasoningContext.ALL_TURNS,
            {"context": "all_turns"},
            "all_turns",
        ),
    ),
    ids=("auto", "current", "all"),
)
async def test_inline_compaction_preserves_each_reasoning_context(
    requested: conversation.ReasoningContext,
    expected_wire: dict[str, str] | None,
    reported: str,
) -> None:
    """Keep reasoning selection independent from inline compaction."""
    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(cast(dict[str, object], loads(await request.aread())))
        return httpx.Response(
            200,
            json=_response(
                f"reasoning-{requested.value}",
                [_message(f"message-{requested.value}", "answer")],
                reasoning_context=reported,
            ),
        )

    binding = _binding(f"lane-compact-reasoning-{requested.value}")
    provider = _stateless_provider(binding, handler)
    result = await provider.dispatch(
        conversation.StatelessProviderPlan(
            binding=binding,
            ledger=conversation.ProviderItemLedger(
                lane_id=binding.lane_id,
                normalization_version=conversation.ConversationCodecVersion(1),
                items=(_prefix_item(binding.lane_id),),
            ),
            reasoning=_reasoning_context(requested),
            compaction=conversation.InlineCompaction(compact_threshold=128),
        )
    )
    if expected_wire is None:
        assert "reasoning" not in payloads[0]
    else:
        assert payloads[0]["reasoning"] == expected_wire
    assert payloads[0]["context_management"] == [
        {"type": "compaction", "compact_threshold": 128}
    ]
    assert result.reasoning.requested is requested
    assert result.reasoning.effective == reported
    await provider.aclose()


_MALFORMED_COMPACT_OUTPUTS: tuple[list[dict[str, object]], ...] = (
    [],
    [
        {
            "created_by": "provider-compact",
            "id": "missing",
            "type": "compaction",
        }
    ],
    [
        {
            **_compact_item("unknown-version", "private-version"),
            "version": "v2",
        }
    ],
    [
        _compact_item("duplicate-one", "private-one"),
        _compact_item("duplicate-two", "private-two"),
    ],
    [
        _message("assistant-text", "must not be public"),
        _compact_item("after-assistant", "private-assistant"),
    ],
    [
        _compact_item("not-final", "private-not-final"),
        _input_message("after compact"),
    ],
    [
        {**_input_message("one"), "id": "duplicate-input"},
        {**_input_message("two"), "id": "duplicate-input"},
        _compact_item("after-duplicates", "private-duplicates"),
    ],
)


@pytest.mark.parametrize(
    "output",
    _MALFORMED_COMPACT_OUTPUTS,
    ids=(
        "missing-boundary",
        "missing-payload",
        "unknown-version",
        "duplicate-boundary",
        "assistant-text",
        "boundary-not-final",
        "duplicate-item-id",
    ),
)
async def test_malformed_standalone_context_fails_content_free(
    output: list[dict[str, object]],
) -> None:
    """Reject malformed canonical context without exposing opaque bytes."""

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            json={
                "created_at": 3,
                "id": "malformed-compact-response",
                "object": "response.compaction",
                "output": output,
                "usage": _usage(),
            },
        )

    binding = _binding("lane-compact-malformed")
    provider = _stateless_provider(binding, handler)
    with pytest.raises(
        conversation.ConversationProviderResponseError
    ) as error:
        await provider.compact(
            conversation.StandaloneCompactProviderPlan(
                binding=binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(binding.lane_id),),
                ),
                reasoning=_reasoning(),
            )
        )
    assert "private" not in repr(error.value)
    assert provider.diagnostics.compaction_failure_count == 1
    assert provider.diagnostics.failure_boundary == "failure_before_output"
    await provider.aclose()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("created_at", -1),
        ("id", 7),
        ("id", ""),
        ("output", "not-an-output-sequence"),
    ),
    ids=("created-at", "id-type", "id-value", "output-type"),
)
async def test_malformed_compact_envelope_is_rejected_before_publication(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject SDK objects whose compact envelope violates exact fields."""
    binding = _binding(f"lane-compact-envelope-{field}-{type(value).__name__}")
    provider = _stateless_provider(binding, lambda request: _unused(request))
    response = CompactedResponse.model_validate(
        {
            "created_at": 3,
            "id": "compact-envelope-response",
            "object": "response.compaction",
            "output": [_compact_item("compact-envelope", "private-envelope")],
            "usage": _usage(),
        }
    ).model_copy(update={field: value})

    async def malformed(*args: object, **kwargs: object) -> CompactedResponse:
        assert args or kwargs
        return response

    monkeypatch.setattr(provider._client.responses, "compact", malformed)
    with pytest.raises(conversation.ConversationProviderResponseError):
        await provider.compact(
            conversation.StandaloneCompactProviderPlan(
                binding=binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(binding.lane_id),),
                ),
                reasoning=_reasoning(),
            )
        )
    assert provider.diagnostics.compaction_failure_count == 1
    await provider.aclose()


async def _unused(request: httpx.Request) -> httpx.Response:
    """Fail if a mocked SDK boundary unexpectedly reaches HTTP."""
    await request.aread()
    raise AssertionError("unexpected compact HTTP dispatch")


async def test_compact_sdk_mapping_rejects_signatureless_dump() -> None:
    """Reject SDK-like objects whose dump signature cannot be inspected."""

    class SignaturelessDump:
        model_dump = staticmethod(str.maketrans)

    with pytest.raises(conversation.ConversationProviderResponseError):
        provider_module._sdk_mapping(SignaturelessDump())


async def test_compact_input_and_output_limits_are_enforced() -> None:
    """Bound item and byte allocation on both sides of compact dispatch."""
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        return httpx.Response(
            200,
            json={
                "created_at": 4,
                "id": f"limited-compact-{dispatches}",
                "object": "response.compaction",
                "output": [
                    _input_message("retained"),
                    _compact_item("limited-boundary", "limited-private"),
                ],
                "usage": _usage(),
            },
        )

    binding = _binding("lane-compact-limits")
    first = _prefix_item(binding.lane_id)
    second = conversation.ProviderItem(
        item_id=conversation.ProviderItemId("second-message"),
        lane_id=binding.lane_id,
        model_call_id=conversation.ConversationModelCallId("second-call"),
        kind=conversation.ProviderItemKind.MESSAGE,
        order=conversation.ProviderItemOrder(1),
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.FINAL,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input=cast(
            dict[str, JsonValue],
            _message("second-message", "second"),
        ),
        normalization_version=conversation.ConversationCodecVersion(1),
    )
    ledger = conversation.ProviderItemLedger(
        lane_id=binding.lane_id,
        normalization_version=conversation.ConversationCodecVersion(1),
        items=(first, second),
    )
    item_limits = conversation.NativeOpenAICompactionLimits(
        min_compact_threshold=64,
        max_compact_threshold=4_096,
        max_input_items=1,
        max_input_bytes=65_536,
        max_output_items=32,
        max_output_bytes=65_536,
    )
    item_binding = replace(
        binding,
        compaction_policy_digest=(
            conversation.native_openai_compaction_policy_digest(item_limits)
        ),
    )
    item_limited = _stateless_provider(
        item_binding,
        handler,
        limits=item_limits,
    )
    plan = conversation.StandaloneCompactProviderPlan(
        binding=item_binding,
        ledger=ledger,
        reasoning=_reasoning(),
    )
    with pytest.raises(conversation.ConversationLimitError):
        await item_limited.compact(plan)
    assert dispatches == 0
    assert item_limited.diagnostics.compaction_failure_count == 1
    await item_limited.aclose()

    byte_limits = replace(item_limits, max_input_items=32, max_input_bytes=1)
    byte_binding = replace(
        binding,
        compaction_policy_digest=(
            conversation.native_openai_compaction_policy_digest(byte_limits)
        ),
    )
    byte_limited = _stateless_provider(
        byte_binding,
        handler,
        limits=byte_limits,
    )
    with pytest.raises(conversation.ConversationLimitError):
        await byte_limited.compact(replace(plan, binding=byte_binding))
    assert dispatches == 0
    assert byte_limited.diagnostics.compaction_failure_count == 1
    await byte_limited.aclose()

    output_item_limits = replace(
        item_limits,
        max_input_items=32,
        max_output_items=1,
    )
    output_item_binding = replace(
        binding,
        compaction_policy_digest=(
            conversation.native_openai_compaction_policy_digest(
                output_item_limits
            )
        ),
    )
    output_item_limited = _stateless_provider(
        output_item_binding,
        handler,
        limits=output_item_limits,
    )
    with pytest.raises(conversation.ConversationLimitError):
        await output_item_limited.compact(
            replace(plan, binding=output_item_binding)
        )
    assert dispatches == 1
    await output_item_limited.aclose()

    output_byte_limits = replace(
        item_limits,
        max_input_items=32,
        max_output_bytes=1,
    )
    output_byte_binding = replace(
        binding,
        compaction_policy_digest=(
            conversation.native_openai_compaction_policy_digest(
                output_byte_limits
            )
        ),
    )
    output_byte_limited = _stateless_provider(
        output_byte_binding,
        handler,
        limits=output_byte_limits,
    )
    with pytest.raises(conversation.ConversationLimitError):
        await output_byte_limited.compact(
            replace(plan, binding=output_byte_binding)
        )
    assert dispatches == 2
    await output_byte_limited.aclose()


async def _verify_stateless_inline_input_limits_are_exact_and_predispatch(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject item, byte, and depth overflow before stateless dispatch."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        return httpx.Response(
            200,
            json=_response(
                f"inline-input-response-{dispatches}",
                [_compact_item("inline-input-boundary", "inline-private")],
            ),
        )

    prefix_payload = _message("prefix-message", "prefix")
    new_payload = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "inline input"}],
    }
    exact_bytes = sum(
        len(conversation.canonical_json_bytes(cast(JsonValue, payload)))
        for payload in (prefix_payload, new_payload)
    )
    exact_depth = max(
        provider_module._json_value_depth(payload)
        for payload in (prefix_payload, new_payload)
    )
    exact_limits = replace(
        _limits(),
        max_input_items=2,
        max_input_bytes=exact_bytes,
        max_input_depth=exact_depth,
    )
    exact_binding = _binding_with_limits(
        "lane-inline-input-exact",
        exact_limits,
    )
    exact_provider = _stateless_provider(
        exact_binding,
        handler,
        limits=exact_limits,
    )
    exact_plan = conversation.StatelessProviderPlan(
        binding=exact_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=exact_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=(_prefix_item(exact_binding.lane_id),),
        ),
        reasoning=_reasoning(),
        compaction=conversation.InlineCompaction(compact_threshold=128),
        new_input={"text": "inline input"},
    )
    result = await exact_provider.dispatch(exact_plan)
    assert result.items[-1].kind is conversation.ProviderItemKind.COMPACTION
    assert dispatches == 1
    await exact_provider.aclose()

    overflow_limits = (
        replace(exact_limits, max_input_items=1),
        replace(exact_limits, max_input_bytes=exact_bytes - 1),
        replace(exact_limits, max_input_depth=exact_depth - 1),
    )
    for index, limits in enumerate(overflow_limits):
        binding = _binding_with_limits(
            f"lane-inline-input-overflow-{index}",
            limits,
        )
        provider = _stateless_provider(binding, handler, limits=limits)
        plan = replace(
            exact_plan,
            binding=binding,
            ledger=conversation.ProviderItemLedger(
                lane_id=binding.lane_id,
                normalization_version=(
                    conversation.ConversationCodecVersion(1)
                ),
                items=(_prefix_item(binding.lane_id),),
            ),
        )
        with pytest.raises(conversation.ConversationLimitError):
            await provider.dispatch(plan)
        assert dispatches == 1
        assert provider.diagnostics.compaction_failure_count == 1
        await provider.aclose()


async def _verify_stored_inline_input_limits_are_exact_and_predispatch(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject exact stored wire input overflow before provider dispatch."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        return httpx.Response(
            200,
            json=_response(
                f"stored-limit-response-{dispatches}",
                [_compact_item("stored-limit-boundary", "stored-private")],
                stored=True,
                previous_response_id="stored-limit-parent",
            ),
        )

    raw_items = (
        {
            "type": "function_call_output",
            "call_id": "stored-call-one",
            "output": "one",
        },
        {
            "type": "function_call_output",
            "call_id": "stored-call-two",
            "output": "two",
        },
    )
    exact_bytes = sum(
        len(conversation.canonical_json_bytes(cast(JsonValue, item)))
        for item in raw_items
    )
    exact_depth = max(
        provider_module._json_value_depth(item) for item in raw_items
    )
    exact_limits = replace(
        _limits(),
        max_input_items=2,
        max_input_bytes=exact_bytes,
        max_input_depth=exact_depth,
    )
    exact_binding = _binding_with_limits(
        "lane-stored-input-exact",
        exact_limits,
        stored=True,
    )
    exact_provider = _stored_provider(
        exact_binding,
        handler,
        limits=exact_limits,
    )
    exact_plan = conversation.StoredProviderPlan(
        binding=exact_provider.binding,
        upstream_response_id=conversation.UpstreamResponseId(
            "stored-limit-parent"
        ),
        reasoning=_reasoning(),
        compaction=conversation.InlineCompaction(compact_threshold=128),
        new_input={"items": raw_items},
    )
    result = await exact_provider.dispatch(exact_plan)
    assert result.items[-1].kind is conversation.ProviderItemKind.COMPACTION
    assert dispatches == 1
    await exact_provider.aclose()

    overflow_limits = (
        replace(exact_limits, max_input_items=1),
        replace(exact_limits, max_input_bytes=exact_bytes - 1),
    )
    for index, limits in enumerate(overflow_limits):
        binding = _binding_with_limits(
            f"lane-stored-input-overflow-{index}",
            limits,
            stored=True,
        )
        provider = _stored_provider(binding, handler, limits=limits)
        with pytest.raises(conversation.ConversationLimitError):
            await provider.dispatch(
                replace(exact_plan, binding=provider.binding)
            )
        assert dispatches == 1
        assert provider.diagnostics.compaction_failure_count == 1
        await provider.aclose()

    text_payload = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "nested"}],
    }
    text_depth = provider_module._json_value_depth(text_payload)
    depth_limits = replace(
        _limits(),
        max_input_depth=text_depth - 1,
    )
    depth_binding = _binding_with_limits(
        "lane-stored-input-depth-overflow",
        depth_limits,
        stored=True,
    )
    depth_provider = _stored_provider(
        depth_binding,
        handler,
        limits=depth_limits,
    )
    with pytest.raises(conversation.ConversationLimitError):
        await depth_provider.dispatch(
            conversation.StoredProviderPlan(
                binding=depth_provider.binding,
                upstream_response_id=conversation.UpstreamResponseId(
                    "stored-limit-parent"
                ),
                reasoning=_reasoning(),
                compaction=conversation.InlineCompaction(
                    compact_threshold=128
                ),
                new_input={"text": "nested"},
            )
        )
    assert dispatches == 1
    assert depth_provider.diagnostics.compaction_failure_count == 1
    await depth_provider.aclose()


async def _verify_nonstream_and_standalone_item_overflow_stop_normalization(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Stop normalization immediately at inline and standalone item bounds."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )
    raw_output = [
        _compact_item("bounded-one", "bounded-private-one"),
        _message("bounded-two", "bounded two"),
    ]
    dispatches = 0
    normalized = 0
    original = provider_module._provider_item

    def counting_provider_item(
        raw: Mapping[str, JsonValue],
        *,
        plan: provider_module.NativeOpenAIResponsePlan,
        provider_index: int,
    ) -> conversation.ProviderItem:
        nonlocal normalized
        normalized += 1
        return original(raw, plan=plan, provider_index=provider_index)

    monkeypatch.setattr(
        provider_module,
        "_provider_item",
        counting_provider_item,
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        if request.url.path.endswith("/responses/compact"):
            return httpx.Response(
                200,
                json={
                    "created_at": 4,
                    "id": "bounded-standalone-response",
                    "object": "response.compaction",
                    "output": [
                        _input_message("retained"),
                        _compact_item(
                            "bounded-standalone",
                            "bounded-standalone-private",
                        ),
                    ],
                    "usage": _usage(),
                },
            )
        return httpx.Response(
            200,
            json=_response("bounded-inline-response", raw_output),
        )

    limits = replace(_limits(), max_output_items=1)
    binding = _binding_with_limits("lane-bounded-inline", limits)
    provider = _stateless_provider(binding, handler, limits=limits)
    with pytest.raises(conversation.ConversationLimitError):
        await provider.dispatch(
            conversation.StatelessProviderPlan(
                binding=binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(binding.lane_id),),
                ),
                reasoning=_reasoning(),
                compaction=conversation.InlineCompaction(
                    compact_threshold=128
                ),
            )
        )
    assert normalized == 0
    await provider.aclose()

    standalone_binding = _binding_with_limits(
        "lane-bounded-standalone",
        limits,
    )
    standalone = _stateless_provider(
        standalone_binding,
        handler,
        limits=limits,
    )
    with pytest.raises(conversation.ConversationLimitError):
        await standalone.compact(
            conversation.StandaloneCompactProviderPlan(
                binding=standalone_binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=standalone_binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(standalone_binding.lane_id),),
                ),
                reasoning=_reasoning(),
            )
        )
    assert dispatches == 2
    assert normalized == 0
    await standalone.aclose()


async def _verify_streamed_inline_limit_rejects_incrementally(
    record_property: Callable[[str, object], None],
) -> None:
    """Yield the bounded prefix then reject the first excess done item."""
    record_property("conversation_acceptance_evidence", "negative")
    output = [
        _compact_item("stream-bounded-one", "stream-bounded-private"),
        _message("stream-bounded-two", "stream bounded two"),
    ]
    response = _response("stream-bounded-response", output)
    events = [
        {
            "type": "response.output_item.done",
            "sequence_number": 0,
            "output_index": 0,
            "item": output[0],
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 1,
            "output_index": 1,
            "item": output[1],
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": response,
        },
    ]
    body = "".join(f"data: {dumps(event)}\n\n" for event in events)
    body += "data: [DONE]\n\n"

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    limits = replace(_limits(), max_output_items=1)
    binding = _binding_with_limits(
        "lane-stream-bounded",
        limits,
        streaming=True,
    )
    provider = _stateless_provider(binding, handler, limits=limits)
    stream = await provider.stream(
        conversation.StatelessProviderPlan(
            binding=binding,
            ledger=conversation.ProviderItemLedger(
                lane_id=binding.lane_id,
                normalization_version=conversation.ConversationCodecVersion(1),
                items=(_prefix_item(binding.lane_id),),
            ),
            reasoning=_reasoning(),
            compaction=conversation.InlineCompaction(compact_threshold=128),
        )
    )
    iterator = stream.__aiter__()
    first = await iterator.__anext__()
    assert first.item_id == "stream-bounded-one"
    with pytest.raises(conversation.ConversationLimitError):
        await iterator.__anext__()
    with pytest.raises(conversation.ConversationError) as terminal_error:
        await stream.terminal()
    assert terminal_error.value.boundary.value == "malformed_stream_item"
    assert provider.diagnostics.compaction_failure_count == 1
    await stream.aclose()
    await provider.aclose()


async def _verify_failed_inline_limit_preserves_and_reuses_parent(
    record_property: Callable[[str, object], None],
) -> bool:
    """Leave the durable parent unchanged and reusable after limit failure."""
    record_property("conversation_acceptance_evidence", "negative")
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        await request.aread()
        return httpx.Response(
            200,
            json=_response(
                f"parent-reuse-response-{dispatches}",
                [
                    _message(
                        f"parent-reuse-message-{dispatches}",
                        f"parent reuse {dispatches}",
                    )
                ],
            ),
        )

    limits = replace(_limits(), max_input_items=1)
    scope = authority()
    binding = replace(
        _binding_with_limits("lane-parent-reuse", limits),
        agent_id=scope.agent_id,
    )
    provider = _stateless_provider(binding, handler, limits=limits)
    store = conversation.InMemoryConversationStore()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(
            datetime(2026, 8, 2, tzinfo=UTC)
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
    client = avalan.DirectConversationClient(
        avalan.DirectConversationRuntime(
            coordinator=coordinator,
            store=store,
            authority=scope,
            lane=binding,
            retention=retention(),
            id_namespace="inline-limit-parent-reuse",
        )
    )
    root = await client.create(
        "root",
        avalan.StatelessConversationSettings(),
    )
    assert type(root.handle) is avalan.StatelessConversationHandle
    original = await store.load(root.handle.checkpoint_id, scope)
    parent = avalan.StatelessParent(handle=root.handle)
    with pytest.raises(conversation.ConversationLimitError):
        await client.continue_conversation(
            "must not dispatch",
            avalan.StatelessConversationSettings(
                parent=parent,
                compaction=avalan.InlineCompaction(compact_threshold=128),
            ),
        )
    assert dispatches == 1
    assert store.diagnostics.checkpoints == 1
    assert await store.load(root.handle.checkpoint_id, scope) == original

    continued = await client.continue_conversation(
        "reused parent",
        avalan.StatelessConversationSettings(parent=parent),
    )
    assert type(continued.handle) is avalan.StatelessConversationHandle
    assert dispatches == 2
    assert await store.load(root.handle.checkpoint_id, scope) == original
    await coordinator.close()
    return continued.handle.checkpoint_id != root.handle.checkpoint_id


@pytest.mark.parametrize("failure", ("rejection", "cancellation"))
async def _verify_inline_failures(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Count typed inline transport failures without retaining payloads."""
    record_property("conversation_acceptance_evidence", "negative")

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        return httpx.Response(500, text="private-inline-rejection")

    binding = _binding("lane-inline-owner-" + failure)
    provider = _stateless_provider(binding, handler)
    if failure == "cancellation":

        async def cancel_create(*args: object, **kwargs: object) -> object:
            assert args or kwargs
            raise CancelledError()

        monkeypatch.setattr(
            provider._client.responses,
            "create",
            cancel_create,
        )
    expected = (
        CancelledError
        if failure == "cancellation"
        else conversation.ConversationError
    )
    with pytest.raises(expected) as error:
        await provider.dispatch(
            conversation.StatelessProviderPlan(
                binding=binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(binding.lane_id),),
                ),
                reasoning=_reasoning(),
                compaction=conversation.InlineCompaction(
                    compact_threshold=128
                ),
            )
        )
    assert "private-inline-rejection" not in repr(error.value)
    diagnostics = provider.diagnostics
    assert diagnostics.compaction_failure_count == 1
    if failure == "rejection":
        assert diagnostics.failure_boundary == "provider_rejection"
    await provider.aclose()


@pytest.mark.parametrize(
    "failure",
    (
        "rejection",
        "ambiguity",
        "cancellation",
        "validation",
        "wrong-type",
        "generic",
    ),
)
async def _verify_standalone_failures(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Classify standalone failures without retaining provider details."""
    record_property("conversation_acceptance_evidence", "negative")

    async def handler(request: httpx.Request) -> httpx.Response:
        await request.aread()
        if failure == "ambiguity":
            raise httpx.ConnectError(
                "private-compact-connection",
                request=request,
            )
        return httpx.Response(500, text="private-compact-rejection")

    binding = _binding(f"lane-compact-{failure}")
    provider = _stateless_provider(binding, handler)
    if failure in {"cancellation", "validation", "wrong-type", "generic"}:

        async def failed_sdk(*args: object, **kwargs: object) -> object:
            assert args or kwargs
            if failure == "cancellation":
                raise CancelledError()
            if failure == "validation":
                request = httpx.Request(
                    "POST", "https://api.openai.com/v1/responses/compact"
                )
                raise APIResponseValidationError(
                    response=httpx.Response(200, request=request),
                    body={"private": "compact-validation"},
                    message="private compact validation",
                )
            if failure == "generic":
                raise RuntimeError("private compact generic failure")
            return object()

        monkeypatch.setattr(provider._client.responses, "compact", failed_sdk)
    expected = {
        "rejection": conversation.ConversationError,
        "ambiguity": conversation.ConversationAmbiguousDispatchError,
        "cancellation": CancelledError,
        "validation": conversation.ConversationProviderResponseError,
        "wrong-type": conversation.ConversationProviderResponseError,
        "generic": conversation.ConversationAmbiguousDispatchError,
    }[failure]
    with pytest.raises(expected) as error:
        await provider.compact(
            conversation.StandaloneCompactProviderPlan(
                binding=binding,
                ledger=conversation.ProviderItemLedger(
                    lane_id=binding.lane_id,
                    normalization_version=(
                        conversation.ConversationCodecVersion(1)
                    ),
                    items=(_prefix_item(binding.lane_id),),
                ),
                reasoning=_reasoning(),
            )
        )
    assert "private" not in repr(error.value)
    diagnostics = provider.diagnostics
    assert diagnostics.compaction_failure_count == 1
    if failure == "rejection":
        assert diagnostics.failure_boundary == "provider_rejection"
    elif failure in {"ambiguity", "generic"}:
        assert diagnostics.failure_boundary == "ambiguous_possible_dispatch"
    elif failure in {"validation", "wrong-type"}:
        assert diagnostics.failure_boundary == "failure_before_output"
    await provider.aclose()


async def test_malformed_streamed_compaction_has_no_partial_result(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject an incomplete streamed compact item before terminal state."""
    await _verify_malformed_stream(record_property)


async def test_stateless_inline_input_limits_are_exact_and_predispatch(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject item, byte, and depth overflow before stateless dispatch."""
    await _verify_stateless_inline_input_limits_are_exact_and_predispatch(
        record_property
    )


async def test_stored_inline_input_limits_are_exact_and_predispatch(
    record_property: Callable[[str, object], None],
) -> None:
    """Reject exact stored wire input overflow before provider dispatch."""
    await _verify_stored_inline_input_limits_are_exact_and_predispatch(
        record_property
    )


async def test_nonstream_and_standalone_item_overflow_stop_normalization(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Stop normalization immediately at inline and standalone bounds."""
    await _verify_nonstream_and_standalone_item_overflow_stop_normalization(
        monkeypatch,
        record_property,
    )


async def test_streamed_inline_limit_rejects_incrementally(
    record_property: Callable[[str, object], None],
) -> None:
    """Yield the bounded prefix then reject the first excess done item."""
    await _verify_streamed_inline_limit_rejects_incrementally(record_property)


async def test_failed_inline_limit_preserves_and_reuses_parent(
    record_property: Callable[[str, object], None],
) -> None:
    """Leave the durable parent unchanged and reusable after limit failure."""
    await _verify_failed_inline_limit_preserves_and_reuses_parent(
        record_property
    )


@pytest.mark.parametrize("failure", ("rejection", "cancellation"))
async def test_inline_owner_counts_rejection_and_cancellation(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Count typed inline transport failures without retaining payloads."""
    await _verify_inline_failures(
        failure,
        monkeypatch,
        record_property,
    )


@pytest.mark.parametrize(
    "failure",
    (
        "rejection",
        "ambiguity",
        "cancellation",
        "validation",
        "wrong-type",
        "generic",
    ),
)
async def test_standalone_transport_failures_are_typed_and_counted(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Classify standalone failures without retaining provider details."""
    await _verify_standalone_failures(
        failure,
        monkeypatch,
        record_property,
    )


async def test_compaction_incremental_limits_and_parent_reuse_are_exact(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Exercise every incremental limit lane and preserve its parent."""
    record_property(
        "conversation_acceptance_evidence",
        "pre_dispatch_rejection",
    )

    await _verify_stateless_inline_input_limits_are_exact_and_predispatch(
        _ignore_acceptance_evidence
    )
    await _verify_stored_inline_input_limits_are_exact_and_predispatch(
        _ignore_acceptance_evidence
    )
    await _verify_nonstream_and_standalone_item_overflow_stop_normalization(
        monkeypatch,
        _ignore_acceptance_evidence,
    )
    await _verify_streamed_inline_limit_rejects_incrementally(
        _ignore_acceptance_evidence
    )
    assert await _verify_failed_inline_limit_preserves_and_reuses_parent(
        _ignore_acceptance_evidence
    )
