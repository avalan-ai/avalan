"""Exercise the durable patch PostgreSQL schema and codec boundary."""

from asyncio import run
from importlib import import_module

import pytest

import avalan.patch.durable_retention as durable_retention
import avalan.task.stores.pgsql as task_pgsql
from avalan.patch.codec import decode_result
from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    AlgorithmDigest,
    Audience,
    ExpiryTick,
    PatchContextId,
    PatchDomainId,
    PatchExecutionId,
    PatchLineageId,
    PatchPlanId,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
    PatchStepId,
    PatchWorkspaceId,
)
from avalan.patch.durable_retention import (
    AesGcmDurableRetentionCipher,
    AesGcmDurableRetentionEnvelopeValidator,
    DurableRetentionBinding,
    DurableRetentionKey,
    InMemoryDurableRetentionKeyResolver,
    StaticDurableRetentionAuthorizer,
)
from avalan.patch.durable_store import (
    DurablePlanReference,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRetentionAccess,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
    DurableStepBinding,
    DurableStoreError,
    DurableStoreErrorCode,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.pgsql_store import (
    PgsqlDurablePatchStoreSettings,
    _decode_plan,
    _encode_plan,
)
from avalan.patch.policy import PatchPrincipalId, PatchTenantId, PolicyRouteId
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_schema_statements,
)


def _digest(token: str) -> AlgorithmDigest:
    """Return deterministic opaque digest evidence for a test plan."""
    return AlgorithmDigest("sha256", token * 64)


def _plan() -> DurablePlanReference:
    """Return one sealed non-content plan reference for codec coverage."""
    return DurablePlanReference(
        PatchPlanId("plan_" + "a" * 16),
        _digest("a"),
        _digest("b"),
        _digest("c"),
        PatchContextId("context_" + "a" * 16),
        PatchWorkspaceId("workspace_" + "a" * 16),
        PatchDomainId("domain_" + "a" * 16),
        (
            DurableStepBinding(
                PatchStepId("step_" + "a" * 16),
                PatchLineageId("lineage_" + "a" * 16),
            ),
        ),
    )


def test_patch_durable_schema_is_isolated_and_constrains_core_truth() -> None:
    """Expose independent durable patch tables and fail-closed constraints."""
    schema = "\n".join(task_pgsql_schema_statements())

    for table in (
        "patch_durable_domains",
        "patch_durable_requests",
        "patch_durable_grant_consumptions",
        "patch_durable_step_journal",
        "patch_durable_artifact_journal",
        "patch_durable_outbox",
        "patch_durable_retention",
    ):
        assert f'"{table}"' in schema
    for constraint in (
        "ck_patch_durable_domains_current_fence",
        "uq_patch_durable_requests_retransmission",
        "ck_patch_durable_requests_fence",
        "ck_patch_durable_requests_pending_shape",
        "ck_patch_durable_requests_terminal_shape",
        "uq_patch_durable_outbox_request_sequence",
        "uq_patch_durable_outbox_terminal",
        "ck_patch_durable_retention_kind",
    ):
        assert f'"{constraint}"' in schema
    assert '"ciphertext" BYTEA NOT NULL' in schema
    assert "before_content" not in schema


def test_plan_codec_is_exact_and_rejects_corruption() -> None:
    """Round trip sealed plan evidence and reject malformed durable bytes."""
    plan = _plan()
    assert _decode_plan(_encode_plan(plan)) == plan
    with pytest.raises(DurableStoreError):
        _decode_plan(b"patch-durable-plan-v0\x1fnot-a-plan")


def test_pgsql_store_settings_require_bounded_connection_inputs() -> None:
    """Reject empty DSNs and invalid pool bounds before database creation."""
    with pytest.raises(DurableStoreError):
        PgsqlDurablePatchStoreSettings(dsn="", pool_minimum=1, pool_maximum=1)
    with pytest.raises(DurableStoreError):
        PgsqlDurablePatchStoreSettings(
            dsn="postgresql://example", pool_minimum=2, pool_maximum=1
        )


def test_pgsql_migration_rejects_unknown_alembic_command() -> None:
    """Reject commands outside the closed migration command set."""

    class Config:
        """Record Alembic configuration options without loading Alembic."""

        def __init__(self) -> None:
            """Initialize empty option and attribute collections."""
            self.attributes: dict[str, object] = {}

        def set_main_option(self, name: str, value: str) -> None:
            """Accept one configuration option before command dispatch."""
            del name, value

    config_module = type("ConfigModule", (), {"Config": Config})
    command_module = type("CommandModule", (), {})

    def importer(name: str) -> object:
        """Return the exact imported module requested by migration setup."""
        match name:
            case "alembic.config":
                return config_module
            case "alembic.command":
                return command_module
            case _:
                raise AssertionError("unexpected module import")

    settings = PgsqlTaskMigrationSettings(
        url="postgresql://example.invalid/task",
        module_importer=importer,
    )
    with pytest.raises(AssertionError, match="unsupported Alembic command"):
        task_pgsql._run_alembic_command(settings, "downgrade")


def test_durable_migration_rejects_unknown_record_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject every future durable record version without partial mutation."""
    migration = import_module(
        "avalan.task.stores.pgsql_migrations.versions."
        "v20260809_0001_patch_durable_store"
    )
    with pytest.raises(NotImplementedError, match="forward-only"):
        migration.downgrade()
    plan_payload = _encode_plan(_plan())
    assert _decode_plan(plan_payload) == _plan()
    future_plan_payload = plan_payload.replace(
        b"patch-durable-plan-v1", b"patch-durable-plan-v2", 1
    )
    with pytest.raises(DurableStoreError) as plan_rejected:
        _decode_plan(future_plan_payload)
    assert plan_rejected.value.code is DurableStoreErrorCode.PLAN_MISMATCH
    assert future_plan_payload.startswith(b"patch-durable-plan-v2")

    terminal_fields = (
        "patch-result-v1",
        "1",
        "request_" + "a" * 16,
        "plan_" + "a" * 16,
        "request_completed",
        "committed",
        "committed",
        "committed",
        "true",
        "absent",
        "changed",
        "true",
        "established",
        "",
        "",
        "",
    )
    terminal_payload = "\x1f".join(terminal_fields).encode("ascii")
    assert decode_result(terminal_payload).schema_version == 1
    future_terminal_payload = terminal_payload.replace(
        b"patch-result-v1\x1f1\x1f", b"patch-result-v1\x1f2\x1f", 1
    )
    with pytest.raises(ValueError, match="schema version"):
        decode_result(future_terminal_payload)
    assert future_terminal_payload.startswith(b"patch-result-v1\x1f2\x1f")

    async def scenario() -> None:
        key = DurableRetentionKey(
            PatchRetentionKeyId("retention_" + "a" * 16), b"r" * 32
        )
        cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
        )
        backend = InMemoryDurablePatchBackend(
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.APPROVER,))
            ),
            retention_validator=AesGcmDurableRetentionEnvelopeValidator(
                cipher
            ),
        )
        store = InMemoryDurablePatchStore(backend)
        identity = DurableRequestIdentity(
            PatchTenantId("tenant-a"),
            PatchPrincipalId("principal-a"),
            PatchExecutionId("execution_" + "a" * 16),
            PolicyRouteId("route-a"),
            RetransmissionKey("retransmission-a"),
        )
        reservation = await store.reserve(identity, _digest("a"))
        retention_id = PatchRetentionRecordId("retained_" + "a" * 16)
        binding = DurableRetentionBinding(
            reservation.request_id,
            retention_id,
            DurableRetentionKind.SEALED_PLAN,
        )
        policy = DurableRetentionPolicy(ExpiryTick(100), False)
        known = await cipher.seal(b"known-version", binding)
        with monkeypatch.context() as future_version:
            future_version.setattr(
                durable_retention, "_RETENTION_SCHEMA_VERSION", 2
            )
            unknown = await cipher.seal(b"future-version", binding)
        future_record = DurableRetentionRecord(
            retention_id,
            DurableRetentionKind.SEALED_PLAN,
            unknown.key_id,
            unknown.value,
            policy,
        )
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, identity)
        )
        before = await store.inspect(access.request)
        with pytest.raises(DurableStoreError) as retention_rejected:
            await store.put_retention(reservation, future_record)
        assert (
            retention_rejected.value.code
            is DurableStoreErrorCode.RETENTION_DENIED
        )
        assert await store.inspect(access.request) == before
        with pytest.raises(DurableStoreError) as absent:
            await store.get_retention(access, retention_id, ExpiryTick(1))
        assert absent.value.code is DurableStoreErrorCode.RETENTION_DENIED

        known_record = DurableRetentionRecord(
            retention_id,
            DurableRetentionKind.SEALED_PLAN,
            known.key_id,
            known.value,
            policy,
        )
        await store.put_retention(reservation, known_record)
        assert (
            await store.get_retention(access, retention_id, ExpiryTick(1))
            == known_record
        )

    run(scenario())
