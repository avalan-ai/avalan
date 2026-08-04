"""Exercise the deterministic activation and effect matrix end to end."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from json import loads
from typing import cast

import httpx
import pytest
from activation_test import (
    NOW,
    _binding,
    _manifest,
    _production_profile,
    _production_stateless_manifest,
    _registry,
)
from native_openai_provider_test import (
    _binding as _native_binding,
)
from native_openai_provider_test import _message as _native_message
from native_openai_provider_test import _plan as _native_plan
from native_openai_provider_test import _profile as _native_profile
from native_openai_provider_test import _response as _native_response
from openai import AsyncOpenAI

import avalan.conversation as conversation

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run the deterministic matrix on asyncio only."""
    return "asyncio"


@dataclass(slots=True)
class _EffectCounts:
    """Count only externally meaningful matrix effects."""

    dispatch: int = 0
    tool_effect: int = 0
    checkpoint: int = 0
    publication: int = 0
    deletion: int = 0

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        """Return the exact ordered effect-count contract."""
        return (
            self.dispatch,
            self.tool_effect,
            self.checkpoint,
            self.publication,
            self.deletion,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _DispatchRow:
    """Select one exact activation-manifest dispatch row."""

    binding: conversation.ProviderLaneBinding
    mode: conversation.ConversationMode
    reasoning: conversation.ReasoningContext
    compaction: conversation.CompactionOperation


@dataclass(frozen=True, slots=True, kw_only=True)
class _MatrixRequest:
    """Describe one deterministic transaction across provider effects."""

    rows: tuple[_DispatchRow, ...]
    tool_effects: int = 0
    commit: bool = True
    publish: bool = True
    authorized: bool = True
    durable_storage: bool = True
    fail_after_dispatch: int | None = None


class _DeterministicMatrixRuntime:
    """Apply exact activation before every deterministic local effect."""

    def __init__(
        self,
        registry: conversation.AsyncActivationRegistry,
    ) -> None:
        self._registry = registry
        self.counts = _EffectCounts()

    async def execute(self, request: _MatrixRequest) -> None:
        """Execute one atomic matrix transaction or fail before commit."""
        if not request.authorized:
            raise conversation.ConversationAuthorizationError()
        if any(
            row.binding.provider_family
            not in {
                conversation.ProviderFamily.OPENAI,
                conversation.ProviderFamily.AZURE_OPENAI,
            }
            for row in request.rows
        ):
            raise conversation.ConversationCapabilityError()
        if not request.durable_storage and any(
            row.mode is conversation.ConversationMode.STORED
            for row in request.rows
        ):
            raise conversation.ConversationCapabilityError()
        for index, row in enumerate(request.rows, start=1):
            await self._registry.resolve(
                row.binding,
                mode=row.mode,
                reasoning_context=row.reasoning,
                compaction_operation=row.compaction,
            )
            self.counts.dispatch += 1
            if request.fail_after_dispatch == index:
                raise conversation.ConversationProviderResponseError()
        self.counts.tool_effect += request.tool_effects
        if request.commit:
            self.counts.checkpoint += 1
        if request.publish:
            if not request.commit:
                raise conversation.ConversationValidationError()
            self.counts.publication += 1

    async def delete_historical(
        self,
        binding: conversation.ProviderLaneBinding,
    ) -> None:
        """Delete old stored state through explicit compatibility evidence."""
        await self._registry.resolve_lifecycle(
            binding,
            capability=conversation.ConversationCapability.STORED_RESPONSE_DELETION,
        )
        self.counts.deletion += 1


def _supported_manifest(
    binding: conversation.ProviderLaneBinding,
) -> conversation.ActivationManifest:
    """Keep unsupported stored standalone compaction explicitly inactive."""
    source = _manifest(binding=binding)
    return replace(
        source,
        evidence=tuple(
            replace(
                row,
                active=not (
                    row.mode is conversation.ConversationMode.STORED
                    and row.compaction_operation
                    is conversation.CompactionOperation.STANDALONE
                ),
            )
            for row in source.evidence
        ),
    )


async def _applied_runtime(
    binding: conversation.ProviderLaneBinding,
    *,
    clock: Callable[[], Awaitable[datetime]] | None = None,
) -> tuple[
    conversation.ActivationManifest,
    conversation.ActivationSnapshot,
    _DeterministicMatrixRuntime,
]:
    """Return one atomically applied deterministic matrix runtime."""
    manifest = _supported_manifest(binding)
    registry = _registry(
        manifest,
        clock=(clock if clock is not None else _matrix_clock),
    )
    loaded = await registry.load(manifest)
    applied = await registry.apply(
        manifest.integrity_digest,
        expected_generation=loaded.generation,
    )
    return manifest, applied, _DeterministicMatrixRuntime(registry)


async def _matrix_clock() -> datetime:
    """Return the fixed aware timestamp through the strict clock boundary."""
    return NOW


async def _exercise_native_adapter(
    family: conversation.ProviderFamily,
) -> int:
    """Dispatch once through one exact production activation guard."""
    assert family in {
        conversation.ProviderFamily.OPENAI,
        conversation.ProviderFamily.AZURE_OPENAI,
    }
    dispatches = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        payload = cast(dict[str, object], loads((await request.aread())))
        azure = family is conversation.ProviderFamily.AZURE_OPENAI
        assert request.method == "POST"
        assert request.url.host == (
            "resource.openai.azure.com" if azure else "api.openai.com"
        )
        assert request.url.path == (
            "/openai/v1/responses" if azure else "/v1/responses"
        )
        assert payload["model"] == ("deployment-native" if azure else "gpt-5")
        assert payload["store"] is False
        assert payload["stream"] is False
        assert payload["reasoning"] == {"context": "current_turn"}
        assert (
            payload.get("include") == ["reasoning.encrypted_content"]
        ) is azure
        return httpx.Response(
            200,
            request=request,
            json=_native_response(
                f"response-matrix-{family.value}",
                [
                    _native_message(
                        f"message-matrix-{family.value}",
                        family.value,
                    )
                ],
            ),
        )

    azure = family is conversation.ProviderFamily.AZURE_OPENAI
    binding = _native_binding(
        azure=azure,
        lane_id=f"lane-matrix-{family.value}",
    )
    manifest = _production_stateless_manifest(binding)
    registry = _registry(manifest)
    client = AsyncOpenAI(
        api_key="matrix-activation-key",
        base_url=binding.normalized_endpoint,
        default_query=None,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        max_retries=0,
    )
    provider = conversation.NativeOpenAIStatelessProvider(
        client=client,
        profile=_native_profile(binding),
        capability_profile=_production_profile(binding),
        activation_registry=registry,
    )
    plan = _native_plan(
        binding,
        reasoning=conversation.ReasoningContext.CURRENT_TURN,
    )
    try:
        with pytest.raises(conversation.ConversationCapabilityError):
            await provider.dispatch(plan)
        assert dispatches == 0

        loaded = await registry.load(manifest)
        applied = await registry.apply(
            manifest.integrity_digest,
            expected_generation=loaded.generation,
        )
        result = await provider.dispatch(plan)
        assert tuple(item.kind for item in result.items) == (
            conversation.ProviderItemKind.MESSAGE,
        )
        assert dispatches == 1

        await registry.revoke(
            manifest.integrity_digest,
            expected_generation=applied.generation,
        )
        with pytest.raises(conversation.ConversationCapabilityError):
            await provider.dispatch(plan)
        assert dispatches == 1
    finally:
        await provider.aclose()
    return dispatches


async def test_required_matrix_cross_product(
    record_property: Callable[[str, object], None],
) -> None:
    """Exercise the exact deterministic activation-registry matrix."""
    record_property("conversation_acceptance_evidence", "matrix")
    aggregate = _EffectCounts()
    assert tuple(conversation.ProviderTransport) == (
        conversation.ProviderTransport.NON_STREAMING,
        conversation.ProviderTransport.STREAMING,
    )

    native_families = (
        conversation.ProviderFamily.OPENAI,
        conversation.ProviderFamily.AZURE_OPENAI,
    )
    for family in native_families:
        for transport in conversation.ProviderTransport:
            binding = _binding(family=family, transport=transport)
            manifest, _, runtime = await _applied_runtime(binding)
            for mode in (
                conversation.ConversationMode.STATELESS,
                conversation.ConversationMode.STORED,
            ):
                for reasoning in (
                    conversation.ReasoningContext.CURRENT_TURN,
                    conversation.ReasoningContext.ALL_TURNS,
                ):
                    for compaction in conversation.CompactionOperation:
                        request = _MatrixRequest(
                            rows=(
                                _DispatchRow(
                                    binding=binding,
                                    mode=mode,
                                    reasoning=reasoning,
                                    compaction=compaction,
                                ),
                            )
                        )
                        if (
                            mode is conversation.ConversationMode.STORED
                            and compaction
                            is conversation.CompactionOperation.STANDALONE
                        ):
                            with pytest.raises(
                                conversation.ConversationCapabilityError
                            ):
                                await runtime.execute(request)
                            continue
                        await runtime.execute(request)
            assert runtime.counts.as_tuple() == (10, 0, 10, 10, 0)
            aggregate.dispatch += runtime.counts.dispatch
            aggregate.checkpoint += runtime.counts.checkpoint
            aggregate.publication += runtime.counts.publication
            assert manifest.binding == binding

    adapter_dispatches = 0
    for family in native_families:
        adapter_dispatches += await _exercise_native_adapter(family)
    assert adapter_dispatches == 2

    binding = _binding()
    manifest, applied, runtime = await _applied_runtime(binding)
    stateless = _DispatchRow(
        binding=binding,
        mode=conversation.ConversationMode.STATELESS,
        reasoning=conversation.ReasoningContext.CURRENT_TURN,
        compaction=conversation.CompactionOperation.NONE,
    )
    compact = replace(
        stateless,
        compaction=conversation.CompactionOperation.STANDALONE,
    )
    await runtime.execute(_MatrixRequest(rows=(stateless,), tool_effects=1))
    await runtime.execute(_MatrixRequest(rows=(compact,)))
    await runtime.execute(
        _MatrixRequest(rows=(stateless, stateless), tool_effects=2)
    )
    assert runtime.counts.as_tuple() == (4, 3, 3, 3, 0)
    authority_digest = conversation.AuthorityDigest("a" * 64)
    first_key = conversation.ConversationDataKey(
        key_id="matrix-key-1",
        revision=1,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"1" * 32,
    )
    resolver = conversation.InMemoryConversationKeyResolver(
        {authority_digest: (first_key,)}
    )
    second_key = conversation.ConversationDataKey(
        key_id="matrix-key-2",
        revision=2,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"2" * 32,
    )
    grace_key = replace(
        first_key,
        status=conversation.ConversationKeyStatus.GRACE,
    )
    await resolver.replace_keys(authority_digest, (grace_key, second_key))
    assert await resolver.current_write_key(authority_digest) == second_key
    assert (
        await resolver.read_key(
            authority_digest,
            key_id=grace_key.key_id,
            revision=grace_key.revision,
        )
        == grace_key
    )

    restarted_registry = _registry(manifest)
    restarted_loaded = await restarted_registry.load(manifest)
    await restarted_registry.apply(
        manifest.integrity_digest,
        expected_generation=restarted_loaded.generation,
    )
    restarted = _DeterministicMatrixRuntime(restarted_registry)
    await restarted.execute(_MatrixRequest(rows=(stateless,)))
    assert restarted.counts.as_tuple() == (1, 0, 1, 1, 0)

    failure = _DeterministicMatrixRuntime(restarted_registry)
    with pytest.raises(conversation.ConversationProviderResponseError):
        await failure.execute(
            _MatrixRequest(rows=(stateless,), fail_after_dispatch=1)
        )
    assert failure.counts.as_tuple() == (1, 0, 0, 0, 0)

    denied = _DeterministicMatrixRuntime(restarted_registry)
    with pytest.raises(conversation.ConversationAuthorizationError):
        await denied.execute(
            _MatrixRequest(rows=(stateless,), authorized=False)
        )
    assert denied.counts.as_tuple() == (0, 0, 0, 0, 0)

    no_storage = _DeterministicMatrixRuntime(restarted_registry)
    await no_storage.execute(
        _MatrixRequest(rows=(stateless,), durable_storage=False)
    )
    stored = replace(stateless, mode=conversation.ConversationMode.STORED)
    with pytest.raises(conversation.ConversationCapabilityError):
        await no_storage.execute(
            _MatrixRequest(rows=(stored,), durable_storage=False)
        )
    assert no_storage.counts.as_tuple() == (1, 0, 1, 1, 0)

    compatible = replace(
        stateless,
        binding=_binding(
            family=conversation.ProviderFamily.OPENAI_COMPATIBLE,
            endpoint="https://compatible.example/v1",
        ),
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await restarted.execute(_MatrixRequest(rows=(compatible,)))
    assert restarted.counts.as_tuple() == (1, 0, 1, 1, 0)
    with pytest.raises(conversation.ConversationCapabilityError):
        _native_profile(compatible.binding)

    registry = restarted_registry
    revoked = await registry.revoke(
        manifest.integrity_digest,
        expected_generation=(await registry.snapshot()).generation,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await restarted.execute(_MatrixRequest(rows=(stateless,)))
    await restarted.delete_historical(binding)
    assert restarted.counts.as_tuple() == (1, 0, 1, 1, 1)
    dormant = conversation.ActivationSnapshot(
        registry_id=revoked.registry_id,
        generation=0,
        active_manifest=None,
        loaded_manifest_digests=(),
        activated_manifest_digests=(),
        revoked_manifest_digests=(),
    )
    await registry.rollback(
        dormant,
        expected_generation=revoked.generation,
    )
    with pytest.raises(conversation.ConversationCapabilityError):
        await restarted.execute(_MatrixRequest(rows=(stateless,)))
    await restarted.delete_historical(binding)
    assert restarted.counts.deletion == 2

    async def expired_clock() -> datetime:
        return NOW + timedelta(days=2)

    stale_registry = _registry(manifest, clock=expired_clock)
    stale_runtime = _DeterministicMatrixRuntime(stale_registry)
    with pytest.raises(conversation.ConversationValidationError):
        await stale_registry.load(manifest)
    with pytest.raises(conversation.ConversationCapabilityError):
        await stale_runtime.execute(_MatrixRequest(rows=(stateless,)))
    assert stale_runtime.counts.as_tuple() == (0, 0, 0, 0, 0)

    aggregate.dispatch += adapter_dispatches + 4 + 1 + 1 + 1
    aggregate.tool_effect += 3
    aggregate.checkpoint += 3 + 1 + 1
    aggregate.publication += 3 + 1 + 1
    aggregate.deletion += 2
    assert aggregate.as_tuple() == (49, 3, 45, 45, 2)
    assert applied.active_manifest == manifest
