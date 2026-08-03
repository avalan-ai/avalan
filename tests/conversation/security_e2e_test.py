"""Verify bounded conversation hardening without content disclosure."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    create_task,
    gather,
    get_running_loop,
    sleep,
    to_thread,
)
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from json import dumps
from os import environ
from pickle import dumps as pickle_dumps
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from phase2_fixtures import (
    binding as phase2_binding,
)
from phase2_fixtures import (
    coordinator as phase2_coordinator,
)
from phase2_fixtures import (
    empty_stateless_plan as phase2_empty_stateless_plan,
)

import avalan.conversation as conversation
from avalan.conversation import security
from avalan.conversation.binding import (
    CapabilityEvidence,
    CapabilityEvidenceState,
    ConversationCapability,
    ConversationCapabilityProfile,
)
from avalan.conversation.contract import (
    AuthorityEndpointId,
    AuthorityPrincipalId,
    AuthorityScope,
    AuthoritySource,
    AuthorityTenantId,
    ConfigurationSource,
    ConversationAgentId,
    ConversationOperation,
    LocalResponseStorage,
    ProviderLaneStorage,
    ResponseOperation,
    RetentionLimits,
    StoragePolicy,
    UpstreamLifetimeStatus,
)
from avalan.conversation.errors import (
    ConversationAuthorizationError,
    ConversationErrorCode,
    ConversationKeyCompromisedError,
    ConversationKeyMissingError,
    ConversationKeyPolicyError,
    ConversationKeyRetiredError,
    ConversationLimitError,
    ConversationMigrationRequiredError,
    ConversationValidationError,
)
from avalan.conversation.observability import authority_digest
from avalan.conversation.runtime import CoordinatorAwaitBoundary
from avalan.conversation.settings import (
    CompactionOperation,
    ConversationMode,
    ReasoningContext,
)
from avalan.conversation.value import (
    AuthorityDigest,
    CapabilityProfileId,
    CapabilityProfileRevision,
    IntegrityDigest,
    SafeAlias,
)
from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)

pytestmark = pytest.mark.anyio

_NOW = datetime(2026, 8, 3, 12, tzinfo=UTC)
_DIGEST = IntegrityDigest(sha256(b"safe").hexdigest())
_PGSQL_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")


@pytest.fixture
def anyio_backend() -> str:
    """Run hardening effects on asyncio only."""
    return "asyncio"


class _Clock:
    def __init__(self, value: datetime = _NOW) -> None:
        self.value = value

    async def now(self) -> datetime:
        return self.value


@dataclass(frozen=True, slots=True, kw_only=True)
class _Operation:
    kind: security.ConversationMaintenanceKind
    runner: Callable[[int], Awaitable[int]]

    def __post_init__(self) -> None:
        if not isinstance(
            self.kind, security.ConversationMaintenanceKind
        ) or not callable(self.runner):
            raise ConversationValidationError()

    async def run(self, *, limit: int) -> int:
        if type(limit) is not int or limit <= 0:
            raise ConversationValidationError()
        count = await self.runner(limit)
        if type(count) is not int or count < 0 or count > limit:
            raise ConversationValidationError()
        return count


def _authority(
    *,
    tenant: str = "tenant-1",
    principal: str = "principal-1",
    agent: str = "agent-1",
    endpoint: str = "endpoint-1",
) -> AuthorityScope:
    return AuthorityScope(
        source=AuthoritySource.AUTHENTICATED_SERVER_CONTEXT,
        tenant_id=AuthorityTenantId(tenant),
        principal_id=AuthorityPrincipalId(principal),
        agent_id=ConversationAgentId(agent),
        endpoint_id=AuthorityEndpointId(endpoint),
        network_exposed=True,
    )


def _retention(
    *,
    local: LocalResponseStorage = LocalResponseStorage.DURABLE,
    upstream: ProviderLaneStorage = ProviderLaneStorage.STORED,
    local_ttl: int | None = 600,
    envelope_ttl: int | None = 300,
) -> RetentionLimits:
    return RetentionLimits(
        storage=StoragePolicy(
            local=local,
            upstream=upstream,
            provider_storage_disclosed=(
                upstream is ProviderLaneStorage.STORED
            ),
        ),
        upstream_lifetime_status=(
            UpstreamLifetimeStatus.UNKNOWN
            if upstream is ProviderLaneStorage.STORED
            else UpstreamLifetimeStatus.NOT_APPLICABLE
        ),
        local_ttl_seconds=local_ttl,
        envelope_ttl_seconds=envelope_ttl,
    )


async def _force_pgsql_schema_version(
    dsn: str,
    schema: str,
    version: int,
) -> None:
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(
            dsn=dsn,
            schema=schema,
            application_name="avalan-phase11-future-schema",
        )
    )
    async with database:
        async with database.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        'UPDATE "conversation_store_metadata" '
                        'SET "schema_version" = %s, '
                        '"updated_at" = CURRENT_TIMESTAMP '
                        'WHERE "singleton_id" = 1',
                        (version,),
                    )


async def _drop_pgsql_schema(dsn: str, schema: str) -> None:
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


def _policy() -> security.ConversationHardeningPolicy:
    return security.ConversationHardeningPolicy(
        default_mode=ConversationMode.STORED,
        allowed_modes=frozenset(ConversationMode),
        allowed_reasoning_contexts=frozenset(ReasoningContext),
        compaction=security.ConversationCompactionPolicy(
            allowed_operations=frozenset(CompactionOperation),
            minimum_inline_threshold=10,
            maximum_inline_threshold=1_000,
        ),
        backend=security.ConversationCheckpointBackend.POSTGRESQL,
        retention=_retention(),
        resources=security.ConversationResourcePolicy(),
        checkpoint_keys=security.ConversationKeyRotationPolicy(),
        envelope_keys=security.ConversationKeyRotationPolicy(),
        capability_profiles=(SafeAlias("profile-1"), SafeAlias("profile-2")),
        telemetry=security.ConversationTelemetryPolicy(),
    )


def _key(
    *,
    key_id: str,
    revision: int,
    status: security.ConversationOperationalKeyStatus,
    material: bytes,
    purposes: frozenset[security.ConversationKeyPurpose] = frozenset(
        security.ConversationKeyPurpose
    ),
    read_until: datetime | None = None,
) -> security.ConversationOperationalKey:
    return security.ConversationOperationalKey(
        key_id=key_id,
        revision=revision,
        status=status,
        purposes=purposes,
        key_bytes=material,
        activated_at=_NOW,
        read_until=read_until,
    )


def _ring(
    clock: _Clock,
    *,
    compromised_deletion_access: bool = False,
) -> tuple[security.AsyncConversationKeyRing, AuthorityDigest]:
    scope = AuthorityDigest(authority_digest(_authority()))
    active = _key(
        key_id="key-2",
        revision=2,
        status=security.ConversationOperationalKeyStatus.ACTIVE,
        material=b"a" * 32,
    )
    retiring = _key(
        key_id="key-1",
        revision=1,
        status=security.ConversationOperationalKeyStatus.RETIRING,
        material=b"r" * 32,
        read_until=_NOW + timedelta(minutes=10),
    )
    return (
        security.AsyncConversationKeyRing(
            {scope: (active, retiring)},
            clock=clock,
            policy=security.ConversationKeyRotationPolicy(
                compromised_deletion_access=compromised_deletion_access,
            ),
        ),
        scope,
    )


def _profile(
    state: CapabilityEvidenceState,
    *,
    test_only: bool,
) -> ConversationCapabilityProfile:
    capabilities = tuple(
        CapabilityEvidence(
            capability=capability,
            state=state,
            evidence_ids=(
                (f"evidence-{index}",)
                if state
                in {
                    CapabilityEvidenceState.CONFORMANT,
                    CapabilityEvidenceState.TEST_ONLY,
                }
                else ()
            ),
        )
        for index, capability in enumerate(ConversationCapability)
    )
    return ConversationCapabilityProfile(
        profile_id=CapabilityProfileId("profile-1"),
        schema_version=1,
        revision=CapabilityProfileRevision("revision-1"),
        binding_alias=SafeAlias("binding-1"),
        capabilities=capabilities,
        test_only=test_only,
    )


def _exercise_configuration() -> None:
    policy = _policy()
    narrower = security.ConversationResourcePolicy(
        max_items=100,
        max_checkpoint_bytes=1_000,
        max_conversation_bytes=2_000,
        max_depth=8,
        max_branches=4,
        max_envelope_chars=2_000,
        max_stream_items=100,
        max_compact_items=100,
        max_global_concurrency=8,
        max_authority_concurrency=4,
        max_conversation_concurrency=1,
        max_queue_size=16,
        queue_timeout_seconds=1,
        provider_timeout_seconds=2,
        store_timeout_seconds=2,
        key_timeout_seconds=1,
    )
    layers = (
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.SERVED_AGENT,
            allowed_modes=frozenset(
                {
                    ConversationMode.OFF,
                    ConversationMode.STATELESS,
                }
            ),
            allowed_reasoning_contexts=frozenset(
                {
                    ReasoningContext.AUTO,
                    ReasoningContext.CURRENT_TURN,
                }
            ),
            capability_profiles=(SafeAlias("profile-1"),),
        ),
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.MODEL_PROVIDER,
            reasoning_context=ReasoningContext.CURRENT_TURN,
        ),
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            mode=ConversationMode.STATELESS,
            compaction_operation=CompactionOperation.INLINE,
            inline_threshold=100,
            retention=_retention(
                upstream=ProviderLaneStorage.STATELESS,
                local_ttl=300,
                envelope_ttl=120,
            ),
            resources=narrower,
            telemetry_enabled=False,
        ),
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.PROVIDER_DEFAULT,
            mode=ConversationMode.OFF,
        ),
    )
    effective = security.resolve_conversation_policy(policy, layers)
    assert effective.mode is ConversationMode.STATELESS
    assert effective.reasoning_context is ReasoningContext.CURRENT_TURN
    assert effective.inline_threshold == 100
    assert effective.resources == narrower
    assert effective.telemetry_enabled is False
    metadata = effective.diagnostic_metadata()
    assert metadata["mode"] == "stateless"
    assert metadata["effective_ttl_seconds"] == 120
    assert metadata["local_storage"] == "durable"
    assert metadata["upstream_storage"] == "stateless"
    assert metadata["local_ttl_seconds"] == 300
    assert metadata["envelope_ttl_seconds"] == 120
    sources = cast(
        tuple[Mapping[str, object], ...],
        metadata["sources"],
    )
    assert {value["field"]: value["source"] for value in sources} == {
        "allowed_modes": "served_agent",
        "allowed_reasoning_contexts": "served_agent",
        "backend": "server_policy",
        "capability_profiles": "served_agent",
        "compaction": "request",
        "inline_threshold": "request",
        "mode": "request",
        "reasoning_context": "model_provider",
        "resources": "request",
        "retention": "request",
        "telemetry_enabled": "request",
    }
    assert narrower.is_narrower_than(policy.resources)
    assert not policy.resources.is_narrower_than(narrower)
    with pytest.raises(ConversationValidationError):
        narrower.is_narrower_than(
            cast(security.ConversationResourcePolicy, object())
        )
    broadening = (
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            allowed_modes=frozenset(ConversationMode),
        ),
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            allowed_reasoning_contexts=frozenset(ReasoningContext),
        ),
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            retention=_retention(local_ttl=900, envelope_ttl=300),
        ),
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            resources=replace(narrower, max_items=20_000),
        ),
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            capability_profiles=(SafeAlias("profile-3"),),
        ),
        security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            telemetry_enabled=True,
        ),
    )
    ceilings = (
        replace(
            policy,
            default_mode=ConversationMode.OFF,
            allowed_modes=frozenset({ConversationMode.OFF}),
            retention=_retention(
                local=LocalResponseStorage.NONE,
                upstream=ProviderLaneStorage.OFF,
            ),
        ),
        replace(
            policy,
            allowed_reasoning_contexts=frozenset({ReasoningContext.AUTO}),
        ),
        replace(policy, retention=_retention(local_ttl=300, envelope_ttl=120)),
        replace(policy, resources=narrower),
        policy,
        replace(
            policy,
            telemetry=security.ConversationTelemetryPolicy(
                enabled=False,
                events=False,
                metrics=False,
                traces=False,
                correlation_digests=False,
            ),
        ),
    )
    for ceiling, layer in zip(ceilings, broadening, strict=True):
        with pytest.raises(ConversationValidationError):
            security.resolve_conversation_policy(ceiling, (layer,))
    with pytest.raises(ConversationValidationError):
        security.resolve_conversation_policy(
            policy,
            (layers[0], replace(layers[0], mode=ConversationMode.OFF)),
        )
    with pytest.raises(ConversationValidationError):
        security.resolve_conversation_policy(
            policy,
            (
                security.ConversationConfigurationLayer(
                    source=ConfigurationSource.REQUEST,
                    mode=ConversationMode.STORED,
                    allowed_modes=frozenset({ConversationMode.OFF}),
                ),
            ),
        )
    with pytest.raises(ConversationValidationError):
        security.resolve_conversation_policy(
            policy,
            (
                security.ConversationConfigurationLayer(
                    source=ConfigurationSource.REQUEST,
                    compaction_operation=CompactionOperation.INLINE,
                    inline_threshold=1,
                ),
            ),
        )


async def _exercise_keys() -> security.AsyncConversationKeyRing:
    clock = _Clock()
    ring, scope = _ring(clock)
    checkpoint = ring.checkpoint_resolver()
    envelope = ring.envelope_resolver()
    current = await checkpoint.current_write_key(scope)
    old = await checkpoint.read_key(scope, key_id="key-1", revision=1)
    sealed = await envelope.active_key(scope)
    opened = await envelope.read_key(scope, key_id="key-1", revision=1)
    assert (current.status.value, old.status.value) == ("current", "grace")
    assert (sealed.status.value, opened.status.value) == (
        "active",
        "retiring",
    )
    assert "aaaa" not in repr(
        await ring.resolve_active(
            scope, security.ConversationKeyPurpose.CHECKPOINT
        )
    )
    health = await ring.health(scope)
    assert health.active_checkpoint_keys == 1
    assert health.active_envelope_keys == 1
    assert health.retiring_keys == 1
    assert (await ring.health(AuthorityDigest("f" * 64))).highest_revision == 0
    with pytest.raises(ConversationKeyPolicyError):
        security.AsyncConversationKeyRing({}, clock=clock)
    with pytest.raises(ConversationKeyPolicyError):
        await ring.replace_keys(
            scope, (cast(security.ConversationOperationalKey, object()),)
        )
    with pytest.raises(ConversationValidationError):
        await ring.resolve_active(
            scope, cast(security.ConversationKeyPurpose, "bad")
        )
    with pytest.raises(ConversationValidationError):
        await ring.resolve_read(
            scope,
            purpose=security.ConversationKeyPurpose.CHECKPOINT,
            key_id="key-1",
            revision=0,
        )
    clock.value = _NOW + timedelta(minutes=11)
    with pytest.raises(ConversationKeyRetiredError):
        await checkpoint.read_key(scope, key_id="key-1", revision=1)
    compromised = _key(
        key_id="bad-key",
        revision=1,
        status=security.ConversationOperationalKeyStatus.COMPROMISED,
        material=b"c" * 32,
    )
    active = _key(
        key_id="key-3",
        revision=3,
        status=security.ConversationOperationalKeyStatus.ACTIVE,
        material=b"n" * 32,
    )
    await ring.replace_keys(scope, (active, compromised))
    with pytest.raises(ConversationKeyCompromisedError):
        await ring.resolve_read(
            scope,
            purpose=security.ConversationKeyPurpose.CHECKPOINT,
            key_id="bad-key",
            revision=1,
        )
    with pytest.raises(ConversationKeyCompromisedError):
        await checkpoint.deletion_key(scope, key_id="bad-key", revision=1)
    incident_ring, incident_scope = _ring(
        _Clock(), compromised_deletion_access=True
    )
    await incident_ring.replace_keys(incident_scope, (active, compromised))
    deletion_key = await incident_ring.checkpoint_resolver().deletion_key(
        incident_scope,
        key_id="bad-key",
        revision=1,
    )
    assert deletion_key.status.value == "grace"
    return incident_ring


async def _exercise_observability() -> None:
    correlation_key = security.ConversationCorrelationKey(
        key_id="telemetry-key",
        key_bytes=b"t" * 32,
    )
    digest = security.conversation_correlation_digest(
        "private-public-id",
        namespace="response",
        key=correlation_key,
    )
    assert "private-public-id" not in digest
    assert "tttt" not in repr(correlation_key)
    telemetry = security.BoundedConversationTelemetry(max_events=2)
    for kind in security.ConversationEventKind:
        event = security.SafeConversationEvent(
            kind=kind,
            correlation_digest=digest,
            parent_digest=_DIGEST,
            mode=ConversationMode.STATELESS,
            reasoning_context=ReasoningContext.CURRENT_TURN,
            compaction=CompactionOperation.INLINE,
            item_count=2,
            byte_count=128,
            revision=2,
            restarted=kind is security.ConversationEventKind.RESTART,
            failure_boundary="provider" if "failure" in kind.value else None,
        )
        assert "private-public-id" not in repr(event.to_mapping())
        await telemetry.emit(event)
    assert len(await telemetry.snapshot()) == 2
    await telemetry.clear()
    assert await telemetry.snapshot() == ()
    assert (
        security.project_conversation_exception(
            ConversationValidationError()
        ).error_code
        == "conversation_validation_failed"
    )
    assert security.project_conversation_exception(
        RuntimeError("secret-sentinel")
    ) == security.SafeConversationException(
        error_code="conversation_internal_failure",
        failure_boundary="internal",
    )


def _exercise_authority_and_deduplication() -> None:
    owner = security.ConversationSecurityContext(
        authority=_authority(), deployment_id="deployment-1"
    )
    exact = security.ConversationSecurityContext(
        authority=_authority(), deployment_id="deployment-1"
    )
    for operation in (*ConversationOperation, *ResponseOperation):
        security.authorize_conversation_target(owner, exact, operation)
    callers = (
        security.ConversationSecurityContext(
            authority=_authority(tenant="tenant-2"),
            deployment_id="deployment-1",
        ),
        security.ConversationSecurityContext(
            authority=_authority(principal="principal-2"),
            deployment_id="deployment-1",
        ),
        security.ConversationSecurityContext(
            authority=_authority(agent="agent-2"),
            deployment_id="deployment-1",
        ),
        security.ConversationSecurityContext(
            authority=_authority(endpoint="endpoint-2"),
            deployment_id="deployment-1",
        ),
        security.ConversationSecurityContext(
            authority=_authority(), deployment_id="deployment-2"
        ),
    )
    for caller in callers:
        with pytest.raises(ConversationAuthorizationError) as captured:
            security.authorize_conversation_target(
                owner, caller, ConversationOperation.CONTINUE
            )
        assert "tenant-2" not in str(captured.value)
    disabled = security.ConversationPayloadDeduplicationPolicy(
        disposition=security.ConversationDeduplicationDisposition.DISABLED,
        minimum_payload_bytes=1_024,
        tenant_isolation=False,
        authenticated_metadata=False,
        durable_refcounts=False,
        deletion_safe=False,
        rotation_safe=False,
    )
    enabled = security.ConversationPayloadDeduplicationPolicy(
        disposition=(
            security.ConversationDeduplicationDisposition.AUTHENTICATED_TENANT_SCOPED
        ),
        minimum_payload_bytes=1_024,
        tenant_isolation=True,
        authenticated_metadata=True,
        durable_refcounts=True,
        deletion_safe=True,
        rotation_safe=True,
    )
    scope = AuthorityDigest(authority_digest(owner.authority))
    payload_digest = IntegrityDigest("a" * 64)
    metadata_digest = IntegrityDigest("b" * 64)
    assert (
        disabled.address(
            authority=scope,
            authenticated_payload_digest=payload_digest,
            authenticated_metadata_digest=metadata_digest,
            payload_bytes=2_048,
        )
        is None
    )
    address = enabled.address(
        authority=scope,
        authenticated_payload_digest=payload_digest,
        authenticated_metadata_digest=metadata_digest,
        payload_bytes=2_048,
    )
    assert address is not None
    assert (
        enabled.address(
            authority=scope,
            authenticated_payload_digest=payload_digest,
            authenticated_metadata_digest=metadata_digest,
            payload_bytes=1,
        )
        is None
    )
    other = enabled.address(
        authority=AuthorityDigest("c" * 64),
        authenticated_payload_digest=payload_digest,
        authenticated_metadata_digest=metadata_digest,
        payload_bytes=2_048,
    )
    assert other != address
    with pytest.raises(ConversationValidationError):
        security.ConversationPayloadDeduplicationPolicy(
            disposition=(
                security.ConversationDeduplicationDisposition.AUTHENTICATED_TENANT_SCOPED
            ),
            minimum_payload_bytes=1,
            tenant_isolation=False,
            authenticated_metadata=True,
            durable_refcounts=True,
            deletion_safe=True,
            rotation_safe=True,
        )


async def _exercise_admission_and_effects() -> None:
    resources = security.ConversationResourcePolicy(
        max_global_concurrency=1,
        max_authority_concurrency=1,
        max_conversation_concurrency=1,
        max_queue_size=2,
        queue_timeout_seconds=0.02,
        provider_timeout_seconds=0.02,
        store_timeout_seconds=0.02,
        key_timeout_seconds=0.02,
    )
    controller = security.FairConversationAdmissionController(resources)
    first = security.ConversationAdmissionKey(
        authority_digest=AuthorityDigest("a" * 64),
        conversation_digest=IntegrityDigest("1" * 64),
    )
    second = security.ConversationAdmissionKey(
        authority_digest=AuthorityDigest("a" * 64),
        conversation_digest=IntegrityDigest("2" * 64),
    )
    third = security.ConversationAdmissionKey(
        authority_digest=AuthorityDigest("b" * 64),
        conversation_digest=IntegrityDigest("3" * 64),
    )
    lease = await controller.acquire(first)
    second_task = create_task(controller.acquire(second))
    third_task = create_task(controller.acquire(third))
    await sleep(0)
    assert await controller.snapshot() == (1, 2)
    with pytest.raises(ConversationLimitError):
        await controller.acquire(
            security.ConversationAdmissionKey(
                authority_digest=AuthorityDigest("c" * 64),
                conversation_digest=IntegrityDigest("4" * 64),
            )
        )
    await lease.release()
    second_lease = await second_task
    await second_lease.release()
    third_lease = await third_task
    async with third_lease:
        assert await controller.snapshot() == (1, 0)
    assert await controller.snapshot() == (0, 0)
    held = await controller.acquire(first)
    cancelled = create_task(controller.acquire(first))
    await sleep(0)
    cancelled.cancel()
    cancellation = await gather(cancelled, return_exceptions=True)
    assert isinstance(cancellation[0], CancelledError)
    assert await controller.snapshot() == (1, 0)
    await held.release()
    assert await controller.snapshot() == (0, 0)
    held = await controller.acquire(first)
    with pytest.raises(ConversationLimitError):
        await controller.acquire(first)
    await held.release()
    runner = security.ConversationEffectRunner(policy=resources)

    async def value() -> int:
        await sleep(0)
        return 7

    assert await runner.provider(value()) == 7
    assert await runner.store(value()) == 7
    assert await runner.key(value()) == 7
    barrier = Event()
    with pytest.raises(TimeoutError):
        await runner.provider(barrier.wait())


async def _exercise_workers() -> (
    tuple[security.ConversationMaintenanceWorker, ...]
):
    counts: dict[security.ConversationMaintenanceKind, int] = {
        kind: 1 for kind in security.ConversationMaintenanceKind
    }

    def operation(
        kind: security.ConversationMaintenanceKind,
    ) -> _Operation:
        async def run(limit: int) -> int:
            await sleep(0)
            return min(counts[kind], limit)

        return _Operation(
            kind=kind,
            runner=run,
        )

    operations = tuple(
        operation(kind) for kind in security.ConversationMaintenanceKind
    )
    worker = security.ConversationMaintenanceWorker(
        operations,
        batch_size=2,
        interval_seconds=0.01,
        shutdown_timeout_seconds=0.1,
    )
    with pytest.raises(ConversationValidationError):
        await worker.run_once()
    await worker.start()
    await sleep(0)
    health = await worker.health()
    assert health.state is security.ConversationWorkerState.RUNNING
    await worker.drain()
    assert (
        await worker.health()
    ).state is security.ConversationWorkerState.STOPPED
    await worker.drain()
    await worker.start()
    await worker.cancel()
    await worker.cancel()

    async def fail(limit: int) -> int:
        raise RuntimeError("secret-worker-sentinel")

    failed = security.ConversationMaintenanceWorker(
        (
            _Operation(
                kind=security.ConversationMaintenanceKind.RETENTION,
                runner=fail,
            ),
        ),
        batch_size=1,
        interval_seconds=0.01,
        shutdown_timeout_seconds=0.1,
    )
    await failed.start()
    await sleep(0)
    failed_health = await failed.health()
    assert failed_health.state is security.ConversationWorkerState.FAILED
    assert failed_health.failure is not None
    assert "secret-worker-sentinel" not in repr(failed_health)
    await failed.drain()
    await failed.start()
    await sleep(0)
    await failed.cancel()
    await worker.start()
    return (worker, failed)


def _migration(
    disposition: security.ConversationRollbackDisposition = (
        security.ConversationRollbackDisposition.RESOLVABLE
    ),
) -> security.ConversationMigrationContract:
    return security.ConversationMigrationContract(
        code_revision=2,
        revisions=tuple(
            security.ConversationSurfaceRevision(
                surface=surface,
                minimum_reader=1,
                maximum_reader=2,
                writer=2,
            )
            for surface in security.ConversationStateSurface
        ),
        rollback_disposition=disposition,
    )


async def _exercise_migration_and_readiness(
    ring: security.AsyncConversationKeyRing,
    workers: tuple[security.ConversationMaintenanceWorker, ...],
) -> None:
    contract = _migration()
    for surface in security.ConversationStateSurface:
        contract.require_readable(surface, 1)
        contract.require_readable(surface, 2)
        with pytest.raises(ConversationMigrationRequiredError):
            contract.require_readable(surface, 3)
    for operation in ResponseOperation:
        contract.require_operation(operation)
    assert contract.visible_transcript_fallback is False
    unavailable = _migration(
        security.ConversationRollbackDisposition.DETERMINISTICALLY_UNAVAILABLE
    )
    unavailable.require_operation(ResponseOperation.DELETE)
    with pytest.raises(ConversationMigrationRequiredError):
        unavailable.require_operation(ResponseOperation.CONTINUE)
    digest = IntegrityDigest("d" * 64)
    activation = security.ConversationActivationHealth(
        expected_digest=digest,
        loaded_digest=digest,
    )

    async def backend() -> security.ConversationBackendHealth:
        return security.ConversationBackendHealth(
            migration_ready=True,
            schema_version=1,
            application_version=2,
            outbox_lag=0,
            maximum_outbox_lag=10,
        )

    async def capability() -> security.ConversationCapabilityHealth:
        return security.ConversationCapabilityHealth(
            resolver_available=True,
            active_profiles=1,
            resolvable_profiles=2,
        )

    scope = AuthorityDigest(authority_digest(_authority()))
    checker = security.ConversationReadinessChecker(
        backend_probe=backend,
        key_ring=ring,
        authority=scope,
        workers=(workers[0],),
        capability_probe=capability,
        activation=activation,
    )
    assert (await checker.check()).ready

    async def bad_backend() -> security.ConversationBackendHealth:
        return security.ConversationBackendHealth(
            migration_ready=False,
            schema_version=1,
            application_version=2,
            outbox_lag=11,
            maximum_outbox_lag=10,
        )

    async def bad_capability() -> security.ConversationCapabilityHealth:
        return security.ConversationCapabilityHealth(
            resolver_available=False,
            active_profiles=1,
            resolvable_profiles=1,
        )

    failed_worker = workers[1]
    await failed_worker.start()
    await sleep(0)
    bad_checker = replace(
        checker,
        backend_probe=bad_backend,
        authority=AuthorityDigest("f" * 64),
        workers=(failed_worker,),
        capability_probe=bad_capability,
        activation=security.ConversationActivationHealth(
            expected_digest=IntegrityDigest("d" * 64),
            loaded_digest=IntegrityDigest("e" * 64),
        ),
    )
    report = await bad_checker.check()
    assert report.ready is False
    assert set(report.failures) == set(security.ConversationReadinessFailure)
    await failed_worker.drain()
    await workers[0].drain()


def _exercise_traceability_and_activation() -> None:
    threats = (
        "opaque-state-disclosure",
        "confused-deputy",
        "authority-spoofing",
        "public-upstream-id-confusion",
        "cross-lane-replay",
        "token-theft",
        "replay-branch-ambiguity",
        "stale-head-advancement",
        "key-compromise",
        "database-disclosure",
        "malicious-provider-items",
        "denial-of-service",
        "partial-deletion",
    )
    controls = tuple(
        security.ConversationThreatControl(
            threat_id=threat,
            controls=("exact-authority-pre-dispatch",),
            control_owners=(
                security.ConversationThreatControlOwnership(
                    control_id="exact-authority-pre-dispatch",
                    owner="conversation-runtime-operator",
                ),
            ),
            positive_tests=("phase11-hardening-positive",),
            negative_tests=("phase11-hardening-negative",),
            operator_detection="safe-failure-boundary-counter",
            incident_response="fence-dispatch-and-run-reconciliation",
            residual_risk="provider-control-plane-availability",
        )
        for threat in threats
    )
    assert len(controls) == 13
    security.validate_capability_profile_for_activation(
        _profile(CapabilityEvidenceState.CONFORMANT, test_only=False)
    )
    for profile in (
        _profile(CapabilityEvidenceState.TEST_ONLY, test_only=True),
        _profile(CapabilityEvidenceState.INCAPABLE, test_only=False),
    ):
        with pytest.raises(ConversationValidationError):
            security.validate_capability_profile_for_activation(profile)


@pytest.mark.parametrize("bad", [0, -1, True, "1"])
def test_hardening_models_reject_invalid_bounds(bad: object) -> None:
    """Reject malformed resource and lifecycle bounds before effects."""
    with pytest.raises(ConversationValidationError):
        security.ConversationResourcePolicy(max_items=cast(int, bad))
    with pytest.raises(ConversationValidationError):
        security.ConversationResourcePolicy(
            queue_timeout_seconds=cast(float, bad)
        )
    with pytest.raises(ConversationValidationError):
        security.ConversationKeyRotationPolicy(
            max_retiring_keys=cast(int, bad)
        )


def test_closed_hardening_models_fail_before_effects() -> None:
    """Close invalid startup, request, diagnostic, and migration models."""
    policy = _policy()
    factories: tuple[Callable[[], object], ...] = (
        lambda: security.ConversationCompactionPolicy(
            allowed_operations=frozenset({CompactionOperation.INLINE})
        ),
        lambda: security.ConversationCompactionPolicy(
            allowed_operations=frozenset({CompactionOperation.NONE}),
            minimum_inline_threshold=2,
            maximum_inline_threshold=1,
        ),
        lambda: security.ConversationTelemetryPolicy(enabled=cast(bool, 1)),
        lambda: security.ConversationTelemetryPolicy(enabled=False),
        lambda: security.ConversationResourcePolicy(
            max_checkpoint_bytes=2,
            max_conversation_bytes=1,
        ),
        lambda: replace(
            policy,
            default_mode=cast(ConversationMode, "bad"),
        ),
        lambda: replace(
            policy,
            capability_profiles=(SafeAlias("profile-1"),) * 2,
        ),
        lambda: replace(
            policy,
            backend=security.ConversationCheckpointBackend.PROCESS,
        ),
        lambda: replace(
            policy,
            allowed_modes=frozenset(
                {
                    ConversationMode.OFF,
                    ConversationMode.STATELESS,
                }
            ),
            backend=security.ConversationCheckpointBackend.CALLER_HELD,
        ),
        lambda: security.ConversationHardeningPolicy(
            default_mode=ConversationMode.STATELESS,
            allowed_modes=frozenset({ConversationMode.STATELESS}),
            allowed_reasoning_contexts=frozenset(ReasoningContext),
            compaction=policy.compaction,
            backend=security.ConversationCheckpointBackend.CALLER_HELD,
            retention=_retention(
                local=LocalResponseStorage.PROCESS_LOCAL,
                upstream=ProviderLaneStorage.STATELESS,
            ),
            resources=policy.resources,
            checkpoint_keys=policy.checkpoint_keys,
            envelope_keys=policy.envelope_keys,
            capability_profiles=(),
            telemetry=policy.telemetry,
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.SERVER_POLICY
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            mode=cast(ConversationMode, "bad"),
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            reasoning_context=cast(ReasoningContext, "bad"),
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            compaction_operation=cast(CompactionOperation, "bad"),
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            allowed_modes=cast(frozenset[ConversationMode], frozenset()),
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            inline_threshold=0,
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            inline_threshold=1,
            compaction_operation=CompactionOperation.NONE,
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            retention=cast(RetentionLimits, object()),
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            resources=cast(security.ConversationResourcePolicy, object()),
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            capability_profiles=(SafeAlias("profile-1"),) * 2,
        ),
        lambda: security.ConversationConfigurationLayer(
            source=ConfigurationSource.REQUEST,
            telemetry_enabled=cast(bool, 1),
        ),
        lambda: security.ConversationPolicySource(
            field="mode",
            source=cast(ConfigurationSource, "bad"),
        ),
        lambda: security.EffectiveConversationPolicy(
            mode=ConversationMode.STORED,
            allowed_modes=frozenset({ConversationMode.OFF}),
            reasoning_context=ReasoningContext.AUTO,
            allowed_reasoning_contexts=frozenset({ReasoningContext.AUTO}),
            compaction_operation=CompactionOperation.NONE,
            inline_threshold=None,
            backend=security.ConversationCheckpointBackend.PROCESS,
            retention=_retention(
                local=LocalResponseStorage.PROCESS_LOCAL,
                upstream=ProviderLaneStorage.OFF,
            ),
            resources=security.ConversationResourcePolicy(),
            capability_profiles=(),
            telemetry_enabled=False,
            sources=(),
        ),
        lambda: security.ConversationKeyHealth(
            active_checkpoint_keys=-1,
            active_envelope_keys=0,
            retiring_keys=0,
            retired_keys=0,
            compromised_keys=0,
            highest_revision=0,
        ),
        lambda: security.ConversationCorrelationKey(
            key_id="short", key_bytes=b"x"
        ),
        lambda: security.SafeConversationEvent(
            kind=cast(security.ConversationEventKind, "bad"),
            correlation_digest=_DIGEST,
        ),
        lambda: security.SafeConversationEvent(
            kind=security.ConversationEventKind.CREATE,
            correlation_digest=_DIGEST,
            mode=cast(ConversationMode, "bad"),
        ),
        lambda: security.SafeConversationEvent(
            kind=security.ConversationEventKind.CREATE,
            correlation_digest=_DIGEST,
            reasoning_context=cast(ReasoningContext, "bad"),
        ),
        lambda: security.SafeConversationEvent(
            kind=security.ConversationEventKind.CREATE,
            correlation_digest=_DIGEST,
            compaction=cast(CompactionOperation, "bad"),
        ),
        lambda: security.SafeConversationEvent(
            kind=security.ConversationEventKind.CREATE,
            correlation_digest=_DIGEST,
            item_count=-1,
        ),
        lambda: security.SafeConversationEvent(
            kind=security.ConversationEventKind.CREATE,
            correlation_digest=_DIGEST,
            revision=-1,
        ),
        lambda: security.SafeConversationEvent(
            kind=security.ConversationEventKind.CREATE,
            correlation_digest=_DIGEST,
            restarted=cast(bool, 1),
        ),
        lambda: security.SafeConversationEvent(
            kind=security.ConversationEventKind.CREATE,
            correlation_digest=_DIGEST,
            error_code=cast(ConversationErrorCode, "bad"),
        ),
        lambda: security.BoundedConversationTelemetry(max_events=0),
        lambda: security.ConversationSecurityContext(
            authority=cast(AuthorityScope, object()),
            deployment_id="deployment",
        ),
        lambda: security.ConversationPayloadDeduplicationPolicy(
            disposition=cast(
                security.ConversationDeduplicationDisposition, "bad"
            ),
            minimum_payload_bytes=1,
            tenant_isolation=True,
            authenticated_metadata=True,
            durable_refcounts=True,
            deletion_safe=True,
            rotation_safe=True,
        ),
        lambda: security.ConversationPayloadDeduplicationPolicy(
            disposition=security.ConversationDeduplicationDisposition.DISABLED,
            minimum_payload_bytes=1,
            tenant_isolation=cast(bool, 1),
            authenticated_metadata=False,
            durable_refcounts=False,
            deletion_safe=False,
            rotation_safe=False,
        ),
        lambda: security.FairConversationAdmissionController(
            cast(security.ConversationResourcePolicy, object())
        ),
        lambda: security.ConversationEffectRunner(
            policy=cast(security.ConversationResourcePolicy, object())
        ),
        lambda: _Operation(
            kind=cast(security.ConversationMaintenanceKind, "bad"),
            runner=cast(Callable[[int], Awaitable[int]], object()),
        ),
        lambda: security.ConversationMaintenanceHealth(
            state=cast(security.ConversationWorkerState, "bad"),
            completed_batches=0,
            processed_records=0,
            task_active=False,
        ),
        lambda: security.ConversationMaintenanceHealth(
            state=security.ConversationWorkerState.STOPPED,
            completed_batches=-1,
            processed_records=0,
            task_active=False,
        ),
        lambda: security.ConversationMaintenanceHealth(
            state=security.ConversationWorkerState.STOPPED,
            completed_batches=0,
            processed_records=0,
            task_active=cast(bool, 1),
        ),
        lambda: security.ConversationMaintenanceHealth(
            state=security.ConversationWorkerState.STOPPED,
            completed_batches=0,
            processed_records=0,
            task_active=False,
            failure=cast(security.SafeConversationException, object()),
        ),
        lambda: security.ConversationSurfaceRevision(
            surface=cast(security.ConversationStateSurface, "bad"),
            minimum_reader=1,
            maximum_reader=2,
            writer=1,
        ),
        lambda: security.ConversationSurfaceRevision(
            surface=security.ConversationStateSurface.CHECKPOINT,
            minimum_reader=0,
            maximum_reader=2,
            writer=1,
        ),
        lambda: security.ConversationSurfaceRevision(
            surface=security.ConversationStateSurface.CHECKPOINT,
            minimum_reader=2,
            maximum_reader=2,
            writer=1,
        ),
        lambda: replace(_migration(), code_revision=0),
        lambda: security.ConversationMigrationContract(
            code_revision=1,
            revisions=(),
            rollback_disposition=(
                security.ConversationRollbackDisposition.RESOLVABLE
            ),
        ),
        lambda: security.ConversationBackendHealth(
            migration_ready=cast(bool, 1),
            schema_version=1,
            application_version=1,
            outbox_lag=0,
            maximum_outbox_lag=0,
        ),
        lambda: security.ConversationBackendHealth(
            migration_ready=True,
            schema_version=-1,
            application_version=1,
            outbox_lag=0,
            maximum_outbox_lag=0,
        ),
        lambda: security.ConversationCapabilityHealth(
            resolver_available=cast(bool, 1),
            active_profiles=0,
            resolvable_profiles=0,
        ),
        lambda: security.ConversationCapabilityHealth(
            resolver_available=True,
            active_profiles=-1,
            resolvable_profiles=0,
        ),
        lambda: security.ConversationCapabilityHealth(
            resolver_available=True,
            active_profiles=2,
            resolvable_profiles=1,
        ),
        lambda: security.ConversationReadinessReport(ready=False, failures=()),
        lambda: security.ConversationThreatControl(
            threat_id="threat",
            controls=(),
            control_owners=(),
            positive_tests=("positive",),
            negative_tests=("negative",),
            operator_detection="detect",
            incident_response="respond",
            residual_risk="risk",
        ),
    )
    for factory in factories:
        with pytest.raises(ConversationValidationError):
            factory()
    invalid_keys: tuple[
        Callable[[], security.ConversationOperationalKey], ...
    ] = (
        lambda: security.ConversationOperationalKey(
            key_id="key",
            revision=0,
            status=security.ConversationOperationalKeyStatus.ACTIVE,
            purposes=frozenset(security.ConversationKeyPurpose),
            key_bytes=b"k" * 32,
            activated_at=_NOW,
        ),
        lambda: security.ConversationOperationalKey(
            key_id="key",
            revision=1,
            status=security.ConversationOperationalKeyStatus.ACTIVE,
            purposes=frozenset(security.ConversationKeyPurpose),
            key_bytes=b"k" * 32,
            activated_at=datetime(2026, 1, 1),
        ),
        lambda: security.ConversationOperationalKey(
            key_id="key",
            revision=1,
            status=security.ConversationOperationalKeyStatus.RETIRING,
            purposes=frozenset(security.ConversationKeyPurpose),
            key_bytes=b"k" * 32,
            activated_at=_NOW,
            read_until=datetime(2026, 1, 2),
        ),
        lambda: security.ConversationOperationalKey(
            key_id="key",
            revision=1,
            status=security.ConversationOperationalKeyStatus.RETIRING,
            purposes=frozenset(security.ConversationKeyPurpose),
            key_bytes=b"k" * 32,
            activated_at=_NOW,
        ),
        lambda: security.ConversationOperationalKey(
            key_id="key",
            revision=1,
            status=security.ConversationOperationalKeyStatus.ACTIVE,
            purposes=frozenset(security.ConversationKeyPurpose),
            key_bytes=b"k" * 32,
            activated_at=_NOW,
            read_until=_NOW + timedelta(minutes=1),
        ),
    )
    for invalid_key in invalid_keys:
        with pytest.raises(ConversationKeyPolicyError):
            invalid_key()
    active = _key(
        key_id="active",
        revision=2,
        status=security.ConversationOperationalKeyStatus.ACTIVE,
        material=b"a" * 32,
    )
    checkpoint_only = _key(
        key_id="active",
        revision=2,
        status=security.ConversationOperationalKeyStatus.ACTIVE,
        material=b"a" * 32,
        purposes=frozenset({security.ConversationKeyPurpose.CHECKPOINT}),
    )
    newer_retiring = _key(
        key_id="newer",
        revision=3,
        status=security.ConversationOperationalKeyStatus.RETIRING,
        material=b"n" * 32,
        read_until=_NOW + timedelta(minutes=2),
    )
    old_one = _key(
        key_id="old-1",
        revision=1,
        status=security.ConversationOperationalKeyStatus.RETIRING,
        material=b"1" * 32,
        read_until=_NOW + timedelta(minutes=2),
    )
    old_two = _key(
        key_id="old-2",
        revision=1,
        status=security.ConversationOperationalKeyStatus.RETIRING,
        material=b"2" * 32,
        read_until=_NOW + timedelta(minutes=2),
    )
    short_window = _key(
        key_id="old-1",
        revision=1,
        status=security.ConversationOperationalKeyStatus.RETIRING,
        material=b"1" * 32,
        read_until=_NOW + timedelta(seconds=30),
    )
    invalid_key_sets = (
        (active, active),
        (checkpoint_only,),
        (active, newer_retiring),
        (active, old_one, old_two),
        (active, short_window),
    )
    policies = (
        security.ConversationKeyRotationPolicy(),
        security.ConversationKeyRotationPolicy(),
        security.ConversationKeyRotationPolicy(),
        security.ConversationKeyRotationPolicy(max_retiring_keys=1),
        security.ConversationKeyRotationPolicy(),
    )
    for key_set, key_policy in zip(invalid_key_sets, policies, strict=True):
        with pytest.raises(ConversationKeyPolicyError):
            security.AsyncConversationKeyRing._validate_key_set(
                key_set, key_policy
            )
    with pytest.raises(ConversationValidationError):
        security.HardeningCheckpointKeyResolver(
            key_ring=cast(security.AsyncConversationKeyRing, object())
        )
    with pytest.raises(ConversationValidationError):
        security.HardeningEnvelopeKeyResolver(
            key_ring=cast(security.AsyncConversationKeyRing, object())
        )
    with pytest.raises(ConversationValidationError):
        security.conversation_correlation_digest(
            "value",
            namespace="namespace",
            key=cast(security.ConversationCorrelationKey, object()),
        )
    context = security.ConversationSecurityContext(
        authority=_authority(), deployment_id="deployment"
    )
    with pytest.raises(ConversationValidationError):
        security.authorize_conversation_target(
            cast(security.ConversationSecurityContext, object()),
            context,
            ConversationOperation.CREATE,
        )
    disabled = security.ConversationPayloadDeduplicationPolicy(
        disposition=security.ConversationDeduplicationDisposition.DISABLED,
        minimum_payload_bytes=1,
        tenant_isolation=False,
        authenticated_metadata=False,
        durable_refcounts=False,
        deletion_safe=False,
        rotation_safe=False,
    )
    with pytest.raises(ConversationValidationError):
        disabled.address(
            authority=AuthorityDigest("a" * 64),
            authenticated_payload_digest=IntegrityDigest("b" * 64),
            authenticated_metadata_digest=IntegrityDigest("c" * 64),
            payload_bytes=-1,
        )
    with pytest.raises(ConversationValidationError):
        security.ConversationAdmissionKey(
            authority_digest=AuthorityDigest("g" * 64),
            conversation_digest=IntegrityDigest("a" * 64),
        )
    counts = {"key": 2}
    security._decrement_count(counts, "key")
    assert counts == {"key": 1}
    with pytest.raises(ConversationValidationError):
        security.resolve_conversation_policy(
            cast(security.ConversationHardeningPolicy, object())
        )
    with pytest.raises(ConversationValidationError):
        security.resolve_conversation_policy(
            policy,
            (cast(security.ConversationConfigurationLayer, object()),),
        )
    empty_modes = _policy()
    object.__setattr__(empty_modes, "allowed_modes", frozenset())
    with pytest.raises(ConversationValidationError):
        security.resolve_conversation_policy(empty_modes)
    with pytest.raises(ConversationValidationError):
        security.resolve_conversation_policy(
            policy,
            (
                security.ConversationConfigurationLayer(
                    source=ConfigurationSource.REQUEST,
                    reasoning_context=ReasoningContext.ALL_TURNS,
                    allowed_reasoning_contexts=frozenset(
                        {ReasoningContext.AUTO}
                    ),
                ),
            ),
        )
    restricted_compaction = replace(
        policy,
        compaction=security.ConversationCompactionPolicy(
            allowed_operations=frozenset({CompactionOperation.NONE})
        ),
    )
    with pytest.raises(ConversationValidationError):
        security.resolve_conversation_policy(
            restricted_compaction,
            (
                security.ConversationConfigurationLayer(
                    source=ConfigurationSource.REQUEST,
                    compaction_operation=CompactionOperation.STANDALONE,
                ),
            ),
        )
    with pytest.raises(ConversationValidationError):
        _migration().require_readable(
            cast(security.ConversationStateSurface, "bad"), 1
        )
    with pytest.raises(ConversationValidationError):
        _migration().require_operation(cast(ResponseOperation, "bad"))
    with pytest.raises(ConversationValidationError):
        security.validate_capability_profile_for_activation(
            cast(ConversationCapabilityProfile, object())
        )


async def test_hardening_operational_edges_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settle cancellation, timeout, worker, key, and scheduler edges."""
    clock = _Clock()
    ring, scope = _ring(clock)
    with pytest.raises(ConversationKeyPolicyError):
        security.AsyncConversationKeyRing(
            {scope: cast(tuple[security.ConversationOperationalKey, ...], ())},
            clock=clock,
        )
    with pytest.raises(ConversationKeyPolicyError):
        security.AsyncConversationKeyRing(
            {
                scope: (
                    _key(
                        key_id="active",
                        revision=1,
                        status=(
                            security.ConversationOperationalKeyStatus.ACTIVE
                        ),
                        material=b"a" * 32,
                    ),
                )
            },
            clock=cast(security.ConversationSecurityClock, object()),
        )
    with pytest.raises(ConversationKeyMissingError):
        await ring.resolve_active(
            AuthorityDigest("f" * 64),
            security.ConversationKeyPurpose.CHECKPOINT,
        )
    with pytest.raises(ConversationKeyMissingError):
        await ring.resolve_read(
            scope,
            purpose=security.ConversationKeyPurpose.CHECKPOINT,
            key_id="missing",
            revision=1,
        )
    naive_ring, naive_scope = _ring(_Clock(datetime(2026, 1, 1)))
    with pytest.raises(ConversationValidationError):
        await naive_ring.resolve_read(
            naive_scope,
            purpose=security.ConversationKeyPurpose.CHECKPOINT,
            key_id="key-2",
            revision=2,
        )
    telemetry = security.BoundedConversationTelemetry(max_events=1)
    with pytest.raises(ConversationValidationError):
        await telemetry.emit(cast(security.SafeConversationEvent, object()))

    resources = security.ConversationResourcePolicy(
        max_global_concurrency=1,
        max_authority_concurrency=1,
        max_conversation_concurrency=1,
        max_queue_size=4,
        queue_timeout_seconds=0.05,
    )
    controller = security.FairConversationAdmissionController(resources)
    first = security.ConversationAdmissionKey(
        authority_digest=AuthorityDigest("a" * 64),
        conversation_digest=IntegrityDigest("1" * 64),
    )
    second = security.ConversationAdmissionKey(
        authority_digest=AuthorityDigest("a" * 64),
        conversation_digest=IntegrityDigest("2" * 64),
    )
    with pytest.raises(ConversationValidationError):
        await controller.acquire(
            cast(security.ConversationAdmissionKey, object())
        )
    lease = await controller.acquire(first)
    await lease.release()
    await lease.release()
    with pytest.raises(ConversationValidationError):
        await lease.__aenter__()
    with pytest.raises(ConversationValidationError):
        await controller._release(first)

    held = await controller.acquire(first)

    async def release_then_raise(
        futures: tuple[Future[None], ...], *, timeout: float
    ) -> tuple[set[Future[None]], set[Future[None]]]:
        assert futures and timeout > 0
        await held.release()
        raise RuntimeError("controlled-wait-failure")

    with monkeypatch.context() as context:
        context.setattr(security, "wait", release_then_raise)
        with pytest.raises(RuntimeError, match="controlled-wait-failure"):
            await controller.acquire(second)
    assert await controller.snapshot() == (0, 0)

    held = await controller.acquire(first)

    async def release_then_timeout(
        futures: tuple[Future[None], ...], *, timeout: float
    ) -> tuple[set[Future[None]], set[Future[None]]]:
        assert futures and timeout > 0
        await held.release()
        return set(), set(futures)

    with monkeypatch.context() as context:
        context.setattr(security, "wait", release_then_timeout)
        with pytest.raises(ConversationLimitError):
            await controller.acquire(second)
    assert await controller.snapshot() == (0, 0)

    fair_resources = replace(
        resources,
        max_global_concurrency=2,
        max_authority_concurrency=1,
    )
    fair = security.FairConversationAdmissionController(fair_resources)
    other_authority = security.ConversationAdmissionKey(
        authority_digest=AuthorityDigest("b" * 64),
        conversation_digest=IntegrityDigest("3" * 64),
    )
    first_lease = await fair.acquire(first)
    same_authority_task = create_task(fair.acquire(second))
    await sleep(0)
    other_lease = await fair.acquire(other_authority)
    await other_lease.release()
    await first_lease.release()
    await (await same_authority_task).release()

    conversation_fair = security.FairConversationAdmissionController(
        replace(
            resources,
            max_global_concurrency=2,
            max_authority_concurrency=2,
        )
    )
    first_lease = await conversation_fair.acquire(first)
    same_conversation_task = create_task(conversation_fair.acquire(first))
    await sleep(0)
    second_lease = await conversation_fair.acquire(second)
    await second_lease.release()
    await first_lease.release()
    await (await same_conversation_task).release()

    cancelled_future: Future[None] = get_running_loop().create_future()
    cancelled_future.cancel()
    cancelled_waiter = security._AdmissionWaiter(
        key=first, future=cancelled_future
    )
    controller._queues[first.authority_digest] = deque((cancelled_waiter,))
    controller._authority_order.append(first.authority_digest)
    controller._dispatch_locked()
    controller._queues[first.authority_digest] = deque()
    controller._authority_order.append(first.authority_digest)
    controller._dispatch_locked()
    live_future: Future[None] = get_running_loop().create_future()
    live_waiter = security._AdmissionWaiter(key=first, future=live_future)
    controller._queues[first.authority_digest] = deque((live_waiter,))
    absent_future: Future[None] = get_running_loop().create_future()
    absent_waiter = security._AdmissionWaiter(key=first, future=absent_future)
    assert not await controller._remove_waiter(absent_waiter)
    controller._remove_empty_queue(first.authority_digest)
    controller._queues[first.authority_digest] = deque()
    controller._authority_order.clear()
    controller._remove_empty_queue(first.authority_digest)

    runner = security.ConversationEffectRunner(
        policy=replace(resources, provider_timeout_seconds=0.01)
    )
    effect_barrier = Event()
    cancelled_effect = create_task(runner.provider(effect_barrier.wait()))
    await sleep(0)
    cancelled_effect.cancel()
    cancellation = await gather(cancelled_effect, return_exceptions=True)
    assert isinstance(cancellation[0], CancelledError)

    async def cancellation_failure() -> None:
        try:
            await Event().wait()
        finally:
            raise RuntimeError("cancel-failure")

    with pytest.raises(RuntimeError, match="cancel-failure"):
        await runner.provider(cancellation_failure())

    async def failing_wait(
        futures: tuple[Future[None], ...], *, timeout: float
    ) -> tuple[set[Future[None]], set[Future[None]]]:
        assert futures and timeout > 0
        await sleep(0)
        raise RuntimeError("outer-wait-failure")

    async def cleanup_failure() -> None:
        try:
            await Event().wait()
        finally:
            raise ValueError("effect-cleanup-failure")

    with monkeypatch.context() as context:
        context.setattr(security, "wait", failing_wait)
        with pytest.raises(ValueError, match="effect-cleanup-failure"):
            await runner.provider(cleanup_failure())

    async def zero(limit: int) -> int:
        return min(0, limit)

    operation = _Operation(
        kind=security.ConversationMaintenanceKind.RETENTION,
        runner=zero,
    )
    with pytest.raises(ConversationValidationError):
        await operation.run(limit=0)

    async def too_many(limit: int) -> int:
        return limit + 1

    invalid_count = replace(operation, runner=too_many)
    with pytest.raises(ConversationValidationError):
        await invalid_count.run(limit=1)

    invalid_workers: tuple[Callable[[], object], ...] = (
        lambda: security.ConversationMaintenanceWorker(
            (), batch_size=1, interval_seconds=1, shutdown_timeout_seconds=1
        ),
        lambda: security.ConversationMaintenanceWorker(
            (operation,),
            batch_size=1,
            interval_seconds=0,
            shutdown_timeout_seconds=1,
        ),
        lambda: security.ConversationMaintenanceWorker(
            (operation, operation),
            batch_size=1,
            interval_seconds=1,
            shutdown_timeout_seconds=1,
        ),
    )
    for factory in invalid_workers:
        with pytest.raises(ConversationValidationError):
            factory()

    interval_worker = security.ConversationMaintenanceWorker(
        (operation,),
        batch_size=1,
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.05,
    )
    await interval_worker.start()
    with pytest.raises(ConversationValidationError):
        await interval_worker.start()
    await sleep(0.01)
    assert (await interval_worker.health()).completed_batches > 1
    await interval_worker.cancel()

    worker_barrier = Event()

    async def block(limit: int) -> int:
        await worker_barrier.wait()
        return min(1, limit)

    blocked_worker = security.ConversationMaintenanceWorker(
        (replace(operation, runner=block),),
        batch_size=1,
        interval_seconds=1,
        shutdown_timeout_seconds=0.001,
    )
    await blocked_worker.start()
    await sleep(0)
    await blocked_worker.drain()
    assert (await blocked_worker.health()).state is (
        security.ConversationWorkerState.STOPPED
    )

    class _HardCancellation(BaseException):
        pass

    async def hard_block(limit: int) -> int:
        try:
            await Event().wait()
        finally:
            raise _HardCancellation()
        return limit

    hard_drain_worker = security.ConversationMaintenanceWorker(
        (
            replace(
                operation,
                runner=hard_block,
            ),
        ),
        batch_size=1,
        interval_seconds=1,
        shutdown_timeout_seconds=0.001,
    )
    await hard_drain_worker.start()
    await sleep(0)
    with pytest.raises(_HardCancellation):
        await hard_drain_worker.drain()

    async def reject_cancellation(limit: int) -> int:
        try:
            await Event().wait()
        finally:
            raise _HardCancellation()
        return limit

    resistant_worker = security.ConversationMaintenanceWorker(
        (replace(operation, runner=reject_cancellation),),
        batch_size=1,
        interval_seconds=1,
        shutdown_timeout_seconds=0.01,
    )
    await resistant_worker.start()
    await sleep(0)
    with pytest.raises(_HardCancellation):
        await resistant_worker.cancel()
    await resistant_worker.drain()

    class _ResistantStop:
        def __init__(self) -> None:
            self.stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def set(self) -> None:
            self.stopped = True

        async def wait(self) -> bool:
            try:
                await Event().wait()
            finally:
                raise RuntimeError("stop-cleanup-failure")

    async def run_after_gate(gate: Event, limit: int) -> int:
        await gate.wait()
        return min(0, limit)

    timeout_gate = Event()
    stop_timeout_worker = security.ConversationMaintenanceWorker(
        (
            replace(
                operation,
                runner=lambda limit: run_after_gate(timeout_gate, limit),
            ),
        ),
        batch_size=1,
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.01,
    )
    await stop_timeout_worker.start()
    await sleep(0)
    stop_timeout_worker._stop = cast(Event, _ResistantStop())
    timeout_gate.set()
    await sleep(0.01)
    assert (await stop_timeout_worker.health()).state is (
        security.ConversationWorkerState.FAILED
    )
    await stop_timeout_worker.drain()

    outer_gate = Event()
    stop_outer_worker = security.ConversationMaintenanceWorker(
        (
            replace(
                operation,
                runner=lambda limit: run_after_gate(outer_gate, limit),
            ),
        ),
        batch_size=1,
        interval_seconds=1,
        shutdown_timeout_seconds=0.01,
    )
    await stop_outer_worker.start()
    await sleep(0)
    stop_outer_worker._stop = cast(Event, _ResistantStop())
    with monkeypatch.context() as context:
        context.setattr(security, "wait", failing_wait)
        outer_gate.set()
        for _ in range(20):
            if (await stop_outer_worker.health()).state is (
                security.ConversationWorkerState.FAILED
            ):
                break
            await sleep(0.001)
        assert (await stop_outer_worker.health()).state is (
            security.ConversationWorkerState.FAILED
        )
    await stop_outer_worker.drain()

    async def backend() -> security.ConversationBackendHealth:
        return security.ConversationBackendHealth(
            migration_ready=True,
            schema_version=1,
            application_version=1,
            outbox_lag=0,
            maximum_outbox_lag=0,
        )

    async def capability() -> security.ConversationCapabilityHealth:
        return security.ConversationCapabilityHealth(
            resolver_available=True,
            active_profiles=0,
            resolvable_profiles=0,
        )

    with pytest.raises(ConversationValidationError):
        security.ConversationReadinessChecker(
            backend_probe=backend,
            key_ring=ring,
            authority=scope,
            workers=(cast(security.ConversationMaintenanceWorker, object()),),
            capability_probe=capability,
            activation=security.ConversationActivationHealth(
                expected_digest=_DIGEST,
                loaded_digest=_DIGEST,
            ),
        )


async def test_phase11_hardening_positive_runtime(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Drive concrete maintenance and hardening through a real coordinator."""
    record_property("conversation_acceptance_evidence", "runtime")
    scope = _authority(agent="agent-phase2", endpoint="endpoint-phase2")
    scope_digest = AuthorityDigest(authority_digest(scope))
    active = _key(
        key_id="runtime-key",
        revision=2,
        status=security.ConversationOperationalKeyStatus.ACTIVE,
        material=b"r" * 32,
    )
    ring = security.AsyncConversationKeyRing(
        {scope_digest: (active,)},
        clock=_Clock(),
    )
    store = conversation.InMemoryConversationStore()
    retention_operation = security.ConversationRetentionMaintenanceOperation(
        store=store,
        clock=_Clock(),
    )
    assert (
        retention_operation.kind
        is security.ConversationMaintenanceKind.RETENTION
    )
    assert await retention_operation.run(limit=1) == 0
    with pytest.raises(ConversationValidationError):
        await retention_operation.run(limit=0)

    async def oversized_sweep(
        self: conversation.InMemoryConversationStore,
        now: datetime,
        *,
        limit: int,
    ) -> conversation.SweepReceipt:
        assert self is store and now == _NOW and limit == 1
        return conversation.SweepReceipt(expired=2, deleted=0)

    with monkeypatch.context() as context:
        context.setattr(
            conversation.InMemoryConversationStore,
            "sweep",
            oversized_sweep,
        )
        with pytest.raises(ConversationValidationError):
            await retention_operation.run(limit=1)

    class _Publisher:
        def __init__(self, *, fail: bool = False) -> None:
            self.fail = fail
            self.published: list[object] = []

        async def publish(self, intent: object) -> None:
            if self.fail:
                raise RuntimeError("controlled-publication-failure")
            self.published.append(intent)

    publisher = _Publisher()
    outbox = security.ConversationOutboxMaintenanceOperation(
        store=store,
        authority=scope,
        publisher=cast(conversation.ConversationPublisher, publisher),
    )
    assert outbox.kind is security.ConversationMaintenanceKind.OUTBOX
    assert await outbox.run(limit=1) == 0
    with pytest.raises(ConversationValidationError):
        await outbox.run(limit=0)

    class _RecoveryWorker:
        def __init__(self) -> None:
            self.acknowledged = 0
            self.released = 0

        async def claim(self, *, limit: int) -> object:
            assert limit == 1
            return SimpleNamespace(
                records=(SimpleNamespace(intent="safe-intent"),)
            )

        async def acknowledge(self, record: object) -> None:
            assert record is not None
            self.acknowledged += 1

        async def release(self, record: object) -> None:
            assert record is not None
            self.released += 1

    recovered = _RecoveryWorker()
    outbox._worker = cast(
        conversation.ConversationOutboxRecoveryWorker, recovered
    )
    assert await outbox.run(limit=1) == 1
    assert recovered.acknowledged == 1
    failing = security.ConversationOutboxMaintenanceOperation(
        store=store,
        authority=scope,
        publisher=cast(
            conversation.ConversationPublisher, _Publisher(fail=True)
        ),
    )
    failed_recovery = _RecoveryWorker()
    failing._worker = cast(
        conversation.ConversationOutboxRecoveryWorker,
        failed_recovery,
    )
    with pytest.raises(RuntimeError, match="controlled-publication-failure"):
        await failing.run(limit=1)
    assert failed_recovery.released == 1

    reconciler = object.__new__(conversation.ProviderLifecycleReconciler)

    async def reconcile_once(
        self: conversation.ProviderLifecycleReconciler,
        *,
        limit: int,
    ) -> int:
        assert self is reconciler and limit == 2
        return 1

    with monkeypatch.context() as context:
        context.setattr(
            conversation.ProviderLifecycleReconciler,
            "run_once",
            reconcile_once,
        )
        lifecycle = security.ConversationLifecycleMaintenanceOperation(
            reconciler=reconciler
        )
        assert (
            lifecycle.kind
            is security.ConversationMaintenanceKind.RECONCILIATION
        )
        assert await lifecycle.run(limit=2) == 1

    pgsql = object.__new__(conversation.PgsqlConversationStore)

    async def garbage_collect(
        self: conversation.PgsqlConversationStore,
        *,
        limit: int,
    ) -> object:
        assert self is pgsql and limit == 3
        return SimpleNamespace(deleted_payloads=2)

    async def rotate_keys(
        self: conversation.PgsqlConversationStore,
        authority: AuthorityScope,
        *,
        limit: int,
    ) -> object:
        assert self is pgsql and authority == scope and limit == 3
        return SimpleNamespace(reencrypted=2)

    with monkeypatch.context() as context:
        context.setattr(
            conversation.PgsqlConversationStore,
            "garbage_collect",
            garbage_collect,
        )
        context.setattr(
            conversation.PgsqlConversationStore,
            "rotate_keys",
            rotate_keys,
        )
        payload_gc = security.ConversationPayloadGcMaintenanceOperation(
            store=pgsql
        )
        key_rotation = security.ConversationKeyRotationMaintenanceOperation(
            store=pgsql,
            authority=scope,
        )
        assert (
            payload_gc.kind is security.ConversationMaintenanceKind.PAYLOAD_GC
        )
        assert (
            key_rotation.kind
            is security.ConversationMaintenanceKind.KEY_ROTATION
        )
        assert await payload_gc.run(limit=3) == 2
        assert await key_rotation.run(limit=3) == 2

    worker = security.ConversationMaintenanceWorker(
        (retention_operation,),
        batch_size=1,
        interval_seconds=60,
        shutdown_timeout_seconds=0.1,
    )

    async def backend() -> security.ConversationBackendHealth:
        return security.ConversationBackendHealth(
            migration_ready=True,
            schema_version=1,
            application_version=1,
            outbox_lag=0,
            maximum_outbox_lag=0,
        )

    async def capability() -> security.ConversationCapabilityHealth:
        return security.ConversationCapabilityHealth(
            resolver_available=True,
            active_profiles=0,
            resolvable_profiles=0,
        )

    readiness = security.ConversationReadinessChecker(
        backend_probe=backend,
        key_ring=ring,
        authority=scope_digest,
        workers=(worker,),
        capability_probe=capability,
        activation=security.ConversationActivationHealth(
            expected_digest=_DIGEST,
            loaded_digest=_DIGEST,
        ),
    )
    policy = security.resolve_conversation_policy(
        security.ConversationHardeningPolicy(
            default_mode=ConversationMode.STATELESS,
            allowed_modes=frozenset({ConversationMode.STATELESS}),
            allowed_reasoning_contexts=frozenset(ReasoningContext),
            compaction=security.ConversationCompactionPolicy(
                allowed_operations=frozenset(CompactionOperation)
            ),
            backend=security.ConversationCheckpointBackend.PROCESS,
            retention=_retention(
                local=LocalResponseStorage.PROCESS_LOCAL,
                upstream=ProviderLaneStorage.STATELESS,
            ),
            resources=security.ConversationResourcePolicy(),
            checkpoint_keys=security.ConversationKeyRotationPolicy(),
            envelope_keys=security.ConversationKeyRotationPolicy(),
            capability_profiles=(),
            telemetry=security.ConversationTelemetryPolicy(),
        )
    )
    admission = security.FairConversationAdmissionController(policy.resources)
    telemetry = security.BoundedConversationTelemetry(max_events=8)
    hook = security.ConversationHardeningCoordinatorHook(
        policy=policy,
        admission=admission,
        admission_key=security.ConversationAdmissionKey(
            authority_digest=scope_digest,
            conversation_digest=IntegrityDigest("1" * 64),
        ),
        readiness=readiness,
        telemetry=telemetry,
    )
    await hook.close()
    await hook.start()
    with pytest.raises(ConversationValidationError):
        await hook.start()
    with pytest.raises(ConversationValidationError):
        await hook.reach(cast(CoordinatorAwaitBoundary, "bad"))
    with monkeypatch.context() as context:
        context.setattr(security, "current_task", lambda: None)
        with pytest.raises(ConversationValidationError):
            await hook.reach(CoordinatorAwaitBoundary.RESOLVE_AUTHORITY)
    await worker.drain()
    with pytest.raises(ConversationValidationError):
        await hook.reach(CoordinatorAwaitBoundary.VALIDATE_PLAN)
    await worker.start()
    assert await worker.run_once() == 0
    await hook.reach(CoordinatorAwaitBoundary.VALIDATE_PLAN)
    with pytest.raises(ConversationValidationError):
        await hook.close()
    await hook.reach(CoordinatorAwaitBoundary.ROLLBACK)
    await hook.reach(CoordinatorAwaitBoundary.RESOLVE_AUTHORITY)
    lane_binding = phase2_binding()
    plan = phase2_empty_stateless_plan(lane_binding)
    runtime = conversation.ConversationLaneRuntime(
        binding=lane_binding,
        capability_profile=conversation.fake_capability_profile(lane_binding),
        provider_script=conversation.DeterministicFakeProviderScript(
            results=(conversation.fake_provider_result(plan, turn=1),)
        ),
    )
    engine = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(runtime,),
        boundary_hook=hook,
        hardening_required=True,
    )
    assert engine.hardening_active
    direct = conversation.DirectConversationClient(
        conversation.DirectConversationRuntime(
            coordinator=engine,
            store=store,
            authority=scope,
            lane=lane_binding,
            retention=_retention(
                local=LocalResponseStorage.PROCESS_LOCAL,
                upstream=ProviderLaneStorage.STATELESS,
            ),
            id_namespace="hardening",
            hardening_required=True,
        )
    )
    result = await direct.create(
        "hardening",
        conversation.StatelessConversationSettings(),
    )
    assert result.output == "synthetic-output"
    assert await admission.snapshot() == (0, 0)
    assert tuple(event.kind for event in await telemetry.snapshot()) == (
        security.ConversationEventKind.MODE,
        security.ConversationEventKind.FAILURE_BOUNDARY,
        security.ConversationEventKind.MODE,
        security.ConversationEventKind.COMMIT,
        security.ConversationEventKind.COMMIT,
    )
    await engine.close()
    assert (
        await worker.health()
    ).state is security.ConversationWorkerState.STOPPED


@pytest.mark.skipif(
    _PGSQL_DSN is None,
    reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
)
async def test_phase11_pgsql_migration_restart_and_rollback(
    record_property: Callable[[str, object], None],
) -> None:
    """Preserve state through real application N/N+1 and rollback."""
    record_property("conversation_acceptance_evidence", "database")
    assert _PGSQL_DSN is not None
    schema = f"conv_p11_{uuid4().hex}"
    stores: list[conversation.PgsqlConversationStore] = []

    def new_store(
        *,
        application_version: int,
        minimum_schema_version: int,
        maximum_schema_version: int,
    ) -> conversation.PgsqlConversationStore:
        store = conversation.PgsqlConversationStore.from_settings(
            conversation.PgsqlConversationStoreSettings(
                dsn=_PGSQL_DSN,
                schema=schema,
                pool_minimum=1,
                pool_maximum=2,
            ),
            key_resolver=resolver,
            cipher=conversation.AesGcmConversationCipher(),
            policy=conversation.PgsqlConversationStorePolicy(
                application_version=application_version,
                minimum_schema_version=minimum_schema_version,
                maximum_schema_version=maximum_schema_version,
            ),
            clock=conversation.DeterministicFakeClock(_NOW),
        )
        stores.append(store)
        return store

    scope = _authority(agent="agent-phase2", endpoint="endpoint-phase2")
    digest = AuthorityDigest(authority_digest(scope))
    resolver = conversation.InMemoryConversationKeyResolver(
        {
            digest: (
                conversation.ConversationDataKey(
                    key_id="migration-key",
                    revision=1,
                    status=conversation.ConversationKeyStatus.CURRENT,
                    key_bytes=b"m" * 32,
                ),
            )
        }
    )
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_PGSQL_DSN, schema=schema),
    )
    try:
        current = new_store(
            application_version=1,
            minimum_schema_version=1,
            maximum_schema_version=1,
        )
        await current.open()
        initial = await current.readiness(scope)
        assert initial.schema_version == 1
        assert initial.application_version == 1
        assert initial.checkpoint_codec_version == int(
            conversation.CHECKPOINT_CODEC_VERSION
        )
        assert (
            initial.minimum_reader_version,
            initial.maximum_reader_version,
            initial.minimum_writer_version,
            initial.maximum_writer_version,
        ) == (1, 2, 1, 2)
        lane_binding = phase2_binding()
        plan = phase2_empty_stateless_plan(lane_binding)

        def migration_engine(
            store: conversation.PgsqlConversationStore,
            *,
            turn: int,
        ) -> conversation.RunScopedConversationCoordinator:
            return conversation.RunScopedConversationCoordinator(
                store=store,
                authority_resolver=(
                    conversation.DeterministicFakeAuthorityResolver(scope)
                ),
                clock=conversation.DeterministicFakeClock(_NOW),
                publisher=conversation.DeterministicFakePublisher(),
                observer=conversation.DeterministicFakeObserver(),
                retry_waiter=conversation.DeterministicFakeRetryWaiter(),
                lanes=(
                    conversation.ConversationLaneRuntime(
                        binding=lane_binding,
                        capability_profile=(
                            conversation.fake_capability_profile(lane_binding)
                        ),
                        provider_script=(
                            conversation.DeterministicFakeProviderScript(
                                results=(
                                    conversation.fake_provider_result(
                                        plan,
                                        turn=turn,
                                    ),
                                )
                            )
                        ),
                    ),
                ),
            )

        engine = migration_engine(current, turn=1)
        client = conversation.DirectConversationClient(
            conversation.DirectConversationRuntime(
                coordinator=engine,
                store=current,
                authority=scope,
                lane=lane_binding,
                retention=_retention(
                    local=LocalResponseStorage.DURABLE,
                    upstream=ProviderLaneStorage.STATELESS,
                ),
                id_namespace="migration",
            )
        )
        committed = await client.create(
            "migration",
            conversation.StatelessConversationSettings(),
        )
        checkpoint_id = committed.handle.checkpoint_id
        await engine.close()
        await current.close()

        n_plus_one = new_store(
            application_version=2,
            minimum_schema_version=1,
            maximum_schema_version=1,
        )
        await n_plus_one.open()
        upgraded = await n_plus_one.readiness(scope)
        assert upgraded.schema_version == 1
        assert upgraded.application_version == 2
        assert upgraded.checkpoint_codec_version == int(
            conversation.CHECKPOINT_CODEC_VERSION
        )
        assert (
            await n_plus_one.load(checkpoint_id, scope)
        ).identity.checkpoint_id == checkpoint_id
        migrated_engine = migration_engine(n_plus_one, turn=2)
        migrated_client = conversation.DirectConversationClient(
            conversation.DirectConversationRuntime(
                coordinator=migrated_engine,
                store=n_plus_one,
                authority=scope,
                lane=lane_binding,
                retention=_retention(
                    local=LocalResponseStorage.DURABLE,
                    upstream=ProviderLaneStorage.STATELESS,
                ),
                id_namespace="migration-n-plus-one",
            )
        )
        migrated_committed = await migrated_client.create(
            "migration-n-plus-one",
            conversation.StatelessConversationSettings(),
        )
        migrated_checkpoint_id = migrated_committed.handle.checkpoint_id
        await migrated_engine.close()
        await n_plus_one.close()

        rolled_back = new_store(
            application_version=1,
            minimum_schema_version=1,
            maximum_schema_version=1,
        )
        await rolled_back.open()
        rollback_readiness = await rolled_back.readiness(scope)
        assert rollback_readiness.schema_version == 1
        assert rollback_readiness.application_version == 1
        assert rollback_readiness.checkpoint_codec_version == int(
            conversation.CHECKPOINT_CODEC_VERSION
        )
        assert (
            await rolled_back.load(checkpoint_id, scope)
        ).identity.checkpoint_id == checkpoint_id
        assert (
            await rolled_back.load(migrated_checkpoint_id, scope)
        ).identity.checkpoint_id == migrated_checkpoint_id
        await rolled_back.close()

        await _force_pgsql_schema_version(_PGSQL_DSN, schema, 2)
        future = new_store(
            application_version=2,
            minimum_schema_version=1,
            maximum_schema_version=1,
        )
        with pytest.raises(ConversationMigrationRequiredError):
            await future.open()
    finally:
        cleanup = await gather(
            *(store.close() for store in stores),
            return_exceptions=True,
        )
        await _drop_pgsql_schema(_PGSQL_DSN, schema)
        assert not any(isinstance(result, BaseException) for result in cleanup)


async def test_phase11_hardening_negative_runtime(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject rollback, secret serialization, and unbounded settlement."""
    record_property("conversation_acceptance_evidence", "negative")
    ring, scope = _ring(_Clock())
    current = await ring.resolve_active(
        scope,
        security.ConversationKeyPurpose.CHECKPOINT,
    )
    rolled_back = _key(
        key_id="rollback",
        revision=1,
        status=security.ConversationOperationalKeyStatus.ACTIVE,
        material=b"z" * 32,
    )
    with pytest.raises(ConversationKeyPolicyError):
        await ring.replace_keys(scope, (rolled_back,))
    substituted = _key(
        key_id="substitute",
        revision=current.revision,
        status=security.ConversationOperationalKeyStatus.ACTIVE,
        material=b"s" * 32,
    )
    with pytest.raises(ConversationKeyPolicyError):
        await ring.replace_keys(scope, (substituted,))
    rematerialized = _key(
        key_id=current.key_id,
        revision=current.revision,
        status=current.status,
        material=b"x" * 32,
    )
    with pytest.raises(ConversationKeyPolicyError):
        await ring.replace_keys(scope, (rematerialized,))

    direct_scope = _authority(
        agent="agent-phase2",
        endpoint="endpoint-phase2",
    )
    direct_store = conversation.InMemoryConversationStore()
    direct_binding = phase2_binding()
    direct_plan = phase2_empty_stateless_plan(direct_binding)

    def direct_lane_runtime() -> conversation.ConversationLaneRuntime:
        return conversation.ConversationLaneRuntime(
            binding=direct_binding,
            capability_profile=conversation.fake_capability_profile(
                direct_binding
            ),
            provider_script=conversation.DeterministicFakeProviderScript(
                results=(
                    conversation.fake_provider_result(direct_plan, turn=1),
                )
            ),
        )

    class _ForgedHardeningHook:
        conversation_hardening_active = True

        def __init__(self) -> None:
            self.reach_count = 0

        async def reach(self, boundary: CoordinatorAwaitBoundary) -> None:
            assert isinstance(boundary, CoordinatorAwaitBoundary)
            self.reach_count += 1

    forged_hook = _ForgedHardeningHook()
    with pytest.raises(ConversationValidationError):
        conversation.RunScopedConversationCoordinator(
            store=direct_store,
            authority_resolver=(
                conversation.DeterministicFakeAuthorityResolver(direct_scope)
            ),
            clock=conversation.DeterministicFakeClock(_NOW),
            publisher=conversation.DeterministicFakePublisher(),
            observer=conversation.DeterministicFakeObserver(),
            retry_waiter=conversation.DeterministicFakeRetryWaiter(),
            lanes=(direct_lane_runtime(),),
            boundary_hook=forged_hook,
            hardening_required=True,
        )
    assert forged_hook.reach_count == 0

    forged_engine = conversation.RunScopedConversationCoordinator(
        store=direct_store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            direct_scope
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(direct_lane_runtime(),),
        boundary_hook=forged_hook,
    )
    assert not forged_engine.hardening_active
    with pytest.raises(ConversationValidationError):
        forged_engine.assert_hardening_hook(forged_hook)
    with pytest.raises(ConversationValidationError):
        conversation.DirectConversationRuntime(
            coordinator=forged_engine,
            store=direct_store,
            authority=direct_scope,
            lane=direct_binding,
            retention=_retention(
                local=LocalResponseStorage.PROCESS_LOCAL,
                upstream=ProviderLaneStorage.STATELESS,
            ),
            hardening_required=True,
        )
    forged_diagnostics = forged_engine.fake_provider_diagnostics(
        direct_binding.lane_id
    )
    assert forged_diagnostics.plans == ()
    assert forged_diagnostics.remaining_results == 1
    assert forged_hook.reach_count == 0
    await forged_engine.close()

    unprotected = phase2_coordinator(
        store=direct_store,
        scope=direct_scope,
        runtimes=(direct_lane_runtime(),),
    )
    with pytest.raises(ConversationValidationError):
        conversation.DirectConversationRuntime(
            coordinator=unprotected,
            store=direct_store,
            authority=direct_scope,
            lane=direct_binding,
            retention=_retention(
                local=LocalResponseStorage.PROCESS_LOCAL,
                upstream=ProviderLaneStorage.STATELESS,
            ),
            hardening_required=True,
        )
    await unprotected.close()

    correlation = security.ConversationCorrelationKey(
        key_id="correlation-key",
        key_bytes=b"secret-correlation-material".ljust(32, b"x"),
    )
    security_context = security.ConversationSecurityContext(
        authority=_authority(),
        deployment_id="private-deployment",
    )
    for value, sentinel in (
        (current, "aaaa"),
        (correlation, "secret-correlation"),
        (security_context, "private-deployment"),
    ):
        rendered = (
            repr(value),
            str(value),
            format(value),
            dumps(value, default=repr),
        )
        assert all(sentinel not in item for item in rendered)
        with pytest.raises(TypeError):
            cast(Callable[[object], object], asdict)(value)
        for serializer in (pickle_dumps, copy, deepcopy):
            with pytest.raises(
                TypeError,
                match="conversation .* serialization is prohibited",
            ):
                serializer(value)
        reduce_value = cast(Callable[[], object], getattr(value, "__reduce__"))
        with pytest.raises(
            TypeError,
            match="conversation .* serialization is prohibited",
        ):
            reduce_value()
        with pytest.raises(ConversationValidationError):
            format(value, "unsafe")

    resources = security.ConversationResourcePolicy(
        provider_timeout_seconds=0.001,
        cancellation_settlement_seconds=0.001,
    )
    runner = security.ConversationEffectRunner(policy=resources)
    release = Event()

    async def resist_cancellation() -> None:
        try:
            await Event().wait()
        except CancelledError:
            await release.wait()

    with pytest.raises(TimeoutError):
        await runner.provider(resist_cancellation())
    assert runner.quarantined_task_count == 1
    rejected_effect = sleep(0)
    with pytest.raises(ConversationValidationError):
        await runner.provider(rejected_effect)
    rejected_effect.close()
    release.set()
    await sleep(0)
    assert runner.quarantined_task_count == 0

    outer_started = Event()
    outer_release = Event()

    async def resist_outer_cancellation() -> None:
        outer_started.set()
        try:
            await Event().wait()
        except CancelledError:
            await outer_release.wait()

    outer_task = create_task(runner.provider(resist_outer_cancellation()))
    await outer_started.wait()
    outer_task.cancel()
    with pytest.raises(CancelledError):
        await outer_task
    assert runner.quarantined_task_count == 1
    outer_release.set()
    await sleep(0)
    assert runner.quarantined_task_count == 0

    worker_started = Event()
    worker_release = Event()

    async def cancellation_resistant_worker(limit: int) -> int:
        worker_started.set()
        while not worker_release.is_set():
            try:
                await worker_release.wait()
            except CancelledError:
                continue
        return min(limit, 0)

    resistant_operation = _Operation(
        kind=security.ConversationMaintenanceKind.RETENTION,
        runner=cancellation_resistant_worker,
    )
    quarantined_worker = security.ConversationMaintenanceWorker(
        (resistant_operation,),
        batch_size=1,
        interval_seconds=60,
        shutdown_timeout_seconds=0.001,
    )
    await quarantined_worker.start()
    await worker_started.wait()
    await quarantined_worker.cancel()
    quarantined_health = await quarantined_worker.health()
    assert quarantined_health.state is (
        security.ConversationWorkerState.QUARANTINED
    )
    assert quarantined_health.task_active
    with pytest.raises(ConversationValidationError):
        await quarantined_worker.start()
    worker_release.set()
    owned_task = quarantined_worker._task
    assert owned_task is not None
    await owned_task
    settled_health = await quarantined_worker.health()
    assert settled_health.state is security.ConversationWorkerState.STOPPED
    assert not settled_health.task_active

    drain_started = Event()
    drain_release = Event()

    async def cancellation_resistant_drain(limit: int) -> int:
        drain_started.set()
        while not drain_release.is_set():
            try:
                await drain_release.wait()
            except CancelledError:
                continue
        return min(limit, 0)

    drain_worker = security.ConversationMaintenanceWorker(
        (
            _Operation(
                kind=security.ConversationMaintenanceKind.RETENTION,
                runner=cancellation_resistant_drain,
            ),
        ),
        batch_size=1,
        interval_seconds=60,
        shutdown_timeout_seconds=0.001,
    )
    await drain_worker.start()
    await drain_started.wait()
    await drain_worker.drain()
    assert (await drain_worker.health()).state is (
        security.ConversationWorkerState.QUARANTINED
    )
    await drain_worker.drain()
    drain_release.set()
    drain_task = drain_worker._task
    assert drain_task is not None
    await drain_task
    assert (await drain_worker.health()).state is (
        security.ConversationWorkerState.STOPPED
    )

    lifecycle_worker = security.ConversationMaintenanceWorker(
        (resistant_operation,),
        batch_size=1,
        interval_seconds=60,
        shutdown_timeout_seconds=0.001,
    )
    pending_release = Event()

    async def pending_work() -> None:
        await pending_release.wait()

    pending_task = create_task(pending_work())
    await lifecycle_worker._record_stop_failure(
        pending_task,
        RuntimeError("pending-stop-failure"),
    )
    assert (await lifecycle_worker.health()).state is (
        security.ConversationWorkerState.QUARANTINED
    )
    pending_task.cancel()
    with pytest.raises(CancelledError):
        await pending_task
    assert (await lifecycle_worker.health()).state is (
        security.ConversationWorkerState.STOPPED
    )

    async def failed_work() -> None:
        raise RuntimeError("terminal-quarantine-failure")

    failed_task = create_task(failed_work())
    await sleep(0)
    lifecycle_worker._state = security.ConversationWorkerState.QUARANTINED
    lifecycle_worker._task = failed_task
    failed_health = await lifecycle_worker.health()
    assert failed_health.state is security.ConversationWorkerState.FAILED
    assert failed_health.failure is not None

    async def fail_wait(
        futures: tuple[Future[object], ...],
        *,
        timeout: float,
    ) -> tuple[set[Future[object]], set[Future[object]]]:
        assert futures and timeout > 0
        raise RuntimeError("controlled-wait-failure")

    with monkeypatch.context() as context:
        context.setattr(security, "wait", fail_wait)
        with pytest.raises(RuntimeError, match="controlled-wait-failure"):
            await runner.provider(Event().wait())
    await sleep(0)
    failed_future: Future[object] = get_running_loop().create_future()
    failed_future.set_exception(RuntimeError("consumed-background-failure"))
    security._consume_background_task(failed_future)

    invalid_factories: tuple[Callable[[], object], ...] = (
        lambda: security.ConversationRetentionMaintenanceOperation(
            store=cast(conversation.InMemoryConversationStore, object()),
            clock=_Clock(),
        ),
        lambda: security.ConversationOutboxMaintenanceOperation(
            store=conversation.InMemoryConversationStore(),
            authority=cast(AuthorityScope, object()),
            publisher=cast(conversation.ConversationPublisher, object()),
        ),
        lambda: security.ConversationLifecycleMaintenanceOperation(
            reconciler=cast(conversation.ProviderLifecycleReconciler, object())
        ),
        lambda: security.ConversationPayloadGcMaintenanceOperation(
            store=cast(conversation.PgsqlConversationStore, object())
        ),
        lambda: security.ConversationKeyRotationMaintenanceOperation(
            store=object.__new__(conversation.PgsqlConversationStore),
            authority=cast(AuthorityScope, object()),
        ),
        lambda: security.ConversationThreatControlOwnership(
            control_id="production-control",
            owner="runtime-owner",
        ),
    )
    for factory in invalid_factories:
        with pytest.raises(ConversationValidationError):
            factory()
    with pytest.raises(ConversationValidationError):
        security.ConversationThreatControl(
            threat_id="traceability",
            controls=("exact-control",),
            control_owners=(),
            positive_tests=("positive-runtime",),
            negative_tests=("negative-runtime",),
            operator_detection="safe-counter",
            incident_response="fence-and-reconcile",
            residual_risk="provider-availability",
        )
    with pytest.raises(ConversationValidationError):
        security._validate_mode_retention_backend(
            ConversationMode.STATELESS,
            security.ConversationCheckpointBackend.CALLER_HELD,
            _retention(
                local=LocalResponseStorage.DURABLE,
                upstream=ProviderLaneStorage.STATELESS,
            ),
        )

    async def zero(limit: int) -> int:
        return min(limit, 0)

    operation = _Operation(
        kind=security.ConversationMaintenanceKind.RETENTION,
        runner=zero,
    )
    first_worker = security.ConversationMaintenanceWorker(
        (operation,),
        batch_size=1,
        interval_seconds=60,
        shutdown_timeout_seconds=0.1,
    )
    second_worker = security.ConversationMaintenanceWorker(
        (operation,),
        batch_size=1,
        interval_seconds=60,
        shutdown_timeout_seconds=0.1,
    )

    async def backend() -> security.ConversationBackendHealth:
        return security.ConversationBackendHealth(
            migration_ready=True,
            schema_version=1,
            application_version=1,
            outbox_lag=0,
            maximum_outbox_lag=0,
        )

    async def capability() -> security.ConversationCapabilityHealth:
        return security.ConversationCapabilityHealth(
            resolver_available=True,
            active_profiles=0,
            resolvable_profiles=0,
        )

    checker = security.ConversationReadinessChecker(
        backend_probe=backend,
        key_ring=ring,
        authority=scope,
        workers=(first_worker, second_worker),
        capability_probe=capability,
        activation=security.ConversationActivationHealth(
            expected_digest=_DIGEST,
            loaded_digest=_DIGEST,
        ),
    )
    with pytest.raises(ConversationValidationError):
        replace(checker, probe_timeout_seconds=0)
    policy = security.resolve_conversation_policy(_policy())
    admission = security.FairConversationAdmissionController(policy.resources)
    admission_key = security.ConversationAdmissionKey(
        authority_digest=scope,
        conversation_digest=IntegrityDigest("9" * 64),
    )
    with pytest.raises(ConversationValidationError):
        security.ConversationHardeningCoordinatorHook(
            policy=policy,
            admission=admission,
            admission_key=replace(
                admission_key,
                authority_digest=AuthorityDigest("f" * 64),
            ),
            readiness=checker,
            telemetry=security.BoundedConversationTelemetry(max_events=1),
        )
    failing_hook = security.ConversationHardeningCoordinatorHook(
        policy=policy,
        admission=admission,
        admission_key=admission_key,
        readiness=checker,
        telemetry=security.BoundedConversationTelemetry(max_events=1),
    )
    await second_worker.start()
    with pytest.raises(ConversationValidationError):
        await failing_hook.start()
    assert (await first_worker.health()).state is (
        security.ConversationWorkerState.STOPPED
    )
    await second_worker.cancel()


async def test_normative_hardening_contract() -> None:
    """Prove the complete hardening boundary with bounded async effects."""
    _exercise_configuration()
    ring = await _exercise_keys()
    await _exercise_observability()
    _exercise_authority_and_deduplication()
    await _exercise_admission_and_effects()
    workers = await _exercise_workers()
    await _exercise_migration_and_readiness(ring, workers)
    _exercise_traceability_and_activation()
