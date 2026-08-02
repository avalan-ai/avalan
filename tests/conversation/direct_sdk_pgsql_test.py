"""Exercise the public direct SDK over the durable PostgreSQL store."""

from asyncio import run, to_thread
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from multiprocessing import get_context
from multiprocessing.connection import Connection
from os import environ
from uuid import uuid4

import pytest
from phase2_fixtures import (
    authority,
    binding,
    empty_stateless_plan,
    next_stateless_plan,
    retention,
)

import avalan
import avalan.conversation as conversation
from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
_NOW = datetime(2026, 8, 2, tzinfo=UTC)
_OPAQUE_STATE = b"phase4-private-provider-state-sentinel"

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        _DSN is None,
        reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run durable direct SDK evidence on asyncio only."""
    return "asyncio"


def _key() -> conversation.ConversationDataKey:
    return conversation.ConversationDataKey(
        key_id="phase4-direct-sdk-key",
        revision=1,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"4" * 32,
    )


def _resolver() -> conversation.InMemoryConversationKeyResolver:
    scope = authority()
    return conversation.InMemoryConversationKeyResolver(
        {conversation.authority_digest(scope): (_key(),)}
    )


@dataclass(slots=True)
class _PgsqlHarness:
    dsn: str
    schema: str
    resolver: conversation.InMemoryConversationKeyResolver
    stores: list[conversation.PgsqlConversationStore] = field(
        default_factory=list
    )

    def store(
        self,
        *,
        fault_hook: conversation.PgsqlConversationFaultHook | None = None,
    ) -> conversation.PgsqlConversationStore:
        """Return one closed production PostgreSQL store instance."""
        store = conversation.PgsqlConversationStore.from_settings(
            conversation.PgsqlConversationStoreSettings(
                dsn=self.dsn,
                schema=self.schema,
                pool_minimum=1,
                pool_maximum=2,
            ),
            key_resolver=self.resolver,
            cipher=conversation.AesGcmConversationCipher(),
            clock=conversation.DeterministicFakeClock(_NOW),
            fault_hook=fault_hook,
        )
        self.stores.append(store)
        return store


class _CommitFailureHook:
    def __init__(self) -> None:
        self.failed = False

    async def reach(
        self,
        point: conversation.PgsqlConversationFaultPoint,
    ) -> None:
        if (
            not self.failed
            and point.boundary
            is conversation.PgsqlConversationFaultBoundary.COMMIT_BEFORE
            and point.operation == "checkpoint_atomic_commit"
        ):
            self.failed = True
            raise RuntimeError("private durable commit fault")


async def _drop_schema(dsn: str, schema: str) -> None:
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


@pytest.fixture
async def pgsql_harness() -> AsyncIterator[_PgsqlHarness]:
    """Yield one isolated migrated Phase 4 PostgreSQL schema."""
    assert _DSN is not None
    schema = f"conv_phase4_direct_{uuid4().hex}"
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    harness = _PgsqlHarness(
        dsn=_DSN,
        schema=schema,
        resolver=_resolver(),
    )
    try:
        yield harness
    finally:
        for store in harness.stores:
            await store.close()
        await _drop_schema(_DSN, schema)


def _with_opaque_state(
    result: conversation.ProviderResult,
) -> conversation.ProviderResult:
    message = result.items[0]
    reasoning_id = conversation.ProviderItemId(f"{message.item_id}-reasoning")
    reasoning_item = conversation.ProviderItem(
        item_id=reasoning_id,
        lane_id=message.lane_id,
        model_call_id=message.model_call_id,
        kind=conversation.ProviderItemKind.REASONING,
        order=message.order,
        provider_index=conversation.ProviderItemIndex(0),
        phase=conversation.ProviderItemPhase.ASSISTANT,
        caller=conversation.ProviderItemCaller.PROVIDER,
        canonical_input={
            "id": reasoning_id,
            "summary": (
                {
                    "text": "content-safe reasoning summary",
                    "type": "summary_text",
                },
            ),
            "type": "reasoning",
        },
        normalization_version=(
            conversation.PROVIDER_ITEM_NORMALIZATION_VERSION
        ),
        opaque_state=conversation.OpaqueProviderState(_value=_OPAQUE_STATE),
    )
    return replace(
        result,
        items=(
            reasoning_item,
            replace(
                message,
                order=conversation.ProviderItemOrder(message.order + 1),
                provider_index=conversation.ProviderItemIndex(1),
            ),
        ),
    )


def _coordinator(
    *,
    store: conversation.ConversationStore,
    lane: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
    publisher: conversation.DeterministicFakePublisher | None = None,
) -> conversation.RunScopedConversationCoordinator:
    scope = authority()
    return conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=publisher or conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.ConversationLaneRuntime(
                binding=lane,
                capability_profile=conversation.fake_capability_profile(lane),
                provider_script=conversation.DeterministicFakeProviderScript(
                    results=results
                ),
            ),
        ),
    )


def _client(
    *,
    store: conversation.ConversationStore,
    lane: conversation.ProviderLaneBinding,
    results: tuple[conversation.ProviderResult, ...],
    namespace: str,
    publisher: conversation.DeterministicFakePublisher | None = None,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.RunScopedConversationCoordinator,
]:
    coordinator = _coordinator(
        store=store,
        lane=lane,
        results=results,
        publisher=publisher,
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=store,
        authority=authority(),
        lane=lane,
        retention=retention(),
        id_namespace=namespace,
    )
    return avalan.DirectConversationClient(runtime), coordinator


def _assert_opaque_state_absent(value: object) -> None:
    sentinel = _OPAQUE_STATE.decode("utf-8")
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert sentinel not in str(current)
        assert sentinel not in repr(current)
        if isinstance(current, BaseException):
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)


async def _encrypted_payloads(
    store: conversation.PgsqlConversationStore,
) -> tuple[bytes, ...]:
    async with store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                "SELECT ciphertext FROM conversation_encrypted_payloads"
            )
            rows = await cursor.fetchall()
    return tuple(bytes(row["ciphertext"]) for row in rows)


async def test_public_pgsql_commit_failure_and_post_commit_recovery(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Withhold failed handles and recover one committed publication."""
    record_property("conversation_acceptance_evidence", "database")
    lane = binding("lane-direct-pg-failure", streaming=True)
    plan = empty_stateless_plan(lane)
    provider_result = _with_opaque_state(
        conversation.fake_provider_result(
            plan,
            turn=1,
            text="visible-before-durable-failure",
        )
    )
    failed_store = pgsql_harness.store(fault_hook=_CommitFailureHook())
    await failed_store.open()
    failed_client, failed_coordinator = _client(
        store=failed_store,
        lane=lane,
        results=(provider_result,),
        namespace="phase4-pg-failure",
    )
    failed_result: avalan.DirectConversationResult | None = None
    with pytest.raises(conversation.ConversationStorageError) as failure:
        failed_result = await failed_client.create(
            "durable failure input",
            avalan.StatelessConversationSettings(),
        )
    _assert_opaque_state_absent(failure.value)
    assert failed_result is None
    page = await failed_store.list_checkpoints(
        authority(), cursor=None, limit=10
    )
    assert page.checkpoints == ()
    assert await _encrypted_payloads(failed_store) == ()
    assert failed_coordinator.diagnostics.active_attempts == 0
    failed_provider = failed_coordinator.fake_provider_diagnostics(
        lane.lane_id
    )
    assert len(failed_provider.plans) == 1
    assert failed_provider.streams == ()

    publish_controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="publisher:publish",
                exception=conversation.ConversationPublicationError(),
            ),
        )
    )
    publishing_store = pgsql_harness.store()
    await publishing_store.open()
    first_publisher = conversation.DeterministicFakePublisher(
        publish_controller
    )
    publishing_client, publishing_coordinator = _client(
        store=publishing_store,
        lane=lane,
        results=(provider_result,),
        namespace="phase4-pg-publish",
        publisher=first_publisher,
    )
    key = conversation.RequestIdempotencyKey("phase4-pg-recovery-key")
    with pytest.raises(conversation.ConversationPublicationError) as publish:
        await publishing_client.create(
            "publication recovery input",
            avalan.StatelessConversationSettings(),
            idempotency_key=key,
        )
    _assert_opaque_state_absent(publish.value)
    assert (
        len(
            publishing_coordinator.fake_provider_diagnostics(
                lane.lane_id
            ).plans
        )
        == 1
    )

    recovered = await publishing_client.create(
        "publication recovery input",
        avalan.StatelessConversationSettings(),
        idempotency_key=key,
    )

    assert recovered.output == "visible-before-durable-failure"
    assert len(first_publisher.published) == 1
    assert (
        len(
            publishing_coordinator.fake_provider_diagnostics(
                lane.lane_id
            ).plans
        )
        == 1
    )
    page = await publishing_store.list_checkpoints(
        authority(),
        cursor=None,
        limit=10,
    )
    assert len(page.checkpoints) == 1
    assert page.checkpoints[0].identity.checkpoint_id == (
        recovered.handle.checkpoint_id
    )


async def test_public_pgsql_stream_failure_redacts_opaque_sidecar(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep an actual streamed private sidecar out of every public error."""
    record_property("conversation_acceptance_evidence", "database")
    lane = binding("lane-direct-pg-private-stream", streaming=True)
    provider_result = _with_opaque_state(
        conversation.fake_provider_result(
            empty_stateless_plan(lane),
            turn=1,
            text="safe-visible-output",
        )
    )
    store = pgsql_harness.store(fault_hook=_CommitFailureHook())
    await store.open()
    client, coordinator = _client(
        store=store,
        lane=lane,
        results=(provider_result,),
        namespace="phase4-pg-private-stream",
    )
    stream = await client.create(
        "private sidecar input",
        avalan.StatelessConversationSettings(),
        stream=True,
    )
    iterator = stream.__aiter__()

    assert await iterator.__anext__() == (
        avalan.DirectConversationOutputDelta(text_delta="safe-visible-output")
    )
    with pytest.raises(conversation.ConversationStorageError) as failure:
        await iterator.__anext__()

    _assert_opaque_state_absent(failure.value)
    _assert_opaque_state_absent(stream)
    assert stream.state is avalan.DirectConversationStreamState.FAILED
    with pytest.raises(avalan.ConversationHandleUnavailableError) as missing:
        _ = stream.committed_handle
    assert missing.value.state is avalan.DirectConversationStreamState.FAILED
    page = await store.list_checkpoints(authority(), cursor=None, limit=10)
    assert page.checkpoints == ()
    assert await _encrypted_payloads(store) == ()
    diagnostics = coordinator.fake_provider_diagnostics(lane.lane_id)
    assert diagnostics.streams[0].close_attempts == 1
    assert diagnostics.streams[0].closed


async def _fresh_process_continue(
    dsn: str,
    schema: str,
    handle_values: tuple[str, str, str],
) -> tuple[str, tuple[str, str, str]]:
    lane = binding("lane-direct-pg-restart", streaming=True)
    first_plan = empty_stateless_plan(lane)
    first_result = _with_opaque_state(
        conversation.fake_provider_result(
            first_plan,
            turn=1,
            text="durable-first",
        )
    )
    child_plan = next_stateless_plan(lane, first_result.items)
    child_result = conversation.fake_provider_result(
        child_plan,
        turn=2,
        text="durable-second",
    )
    store = conversation.PgsqlConversationStore.from_settings(
        conversation.PgsqlConversationStoreSettings(
            dsn=dsn,
            schema=schema,
            pool_minimum=1,
            pool_maximum=2,
        ),
        key_resolver=_resolver(),
        cipher=conversation.AesGcmConversationCipher(),
        clock=conversation.DeterministicFakeClock(_NOW),
    )
    try:
        await store.open()
        client, _ = _client(
            store=store,
            lane=lane,
            results=(child_result,),
            namespace="phase4-fresh-process-child",
        )
        parent = avalan.StatelessParent(
            handle=avalan.StatelessConversationHandle(
                conversation_id=conversation.ConversationId(handle_values[0]),
                checkpoint_id=conversation.CheckpointId(handle_values[1]),
                branch_id=conversation.ConversationBranchId(handle_values[2]),
            )
        )
        result = await client.continue_conversation(
            "fresh process continuation",
            avalan.StatelessConversationSettings(parent=parent),
        )
        return result.output, (
            str(result.handle.conversation_id),
            str(result.handle.checkpoint_id),
            str(result.handle.branch_id),
        )
    finally:
        await store.close()


def _fresh_process_target(
    dsn: str,
    schema: str,
    handle_values: tuple[str, str, str],
    connection: Connection,
) -> None:
    try:
        connection.send(
            (
                True,
                run(_fresh_process_continue(dsn, schema, handle_values)),
            )
        )
    except BaseException as error:
        connection.send((False, (type(error).__name__, str(error))))
    finally:
        connection.close()


async def test_public_handle_fresh_process_and_opaque_state_privacy(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Restore and continue using only trusted config and a public handle."""
    record_property("conversation_acceptance_evidence", "database")
    lane = binding("lane-direct-pg-restart", streaming=True)
    first_plan = empty_stateless_plan(lane)
    first_result = _with_opaque_state(
        conversation.fake_provider_result(
            first_plan,
            turn=1,
            text="durable-first",
        )
    )
    first_store = pgsql_harness.store()
    await first_store.open()
    client, _ = _client(
        store=first_store,
        lane=lane,
        results=(first_result,),
        namespace="phase4-fresh-process-parent",
    )
    first = await client.create(
        "fresh process parent",
        avalan.StatelessConversationSettings(),
    )
    assert type(first.handle) is avalan.StatelessConversationHandle
    _assert_opaque_state_absent(first)
    await first_store.close()

    handle_values = (
        str(first.handle.conversation_id),
        str(first.handle.checkpoint_id),
        str(first.handle.branch_id),
    )
    context = get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_fresh_process_target,
        args=(
            pgsql_harness.dsn,
            pgsql_harness.schema,
            handle_values,
            child_connection,
        ),
    )
    process.start()
    child_connection.close()
    await to_thread(process.join, 45)
    if process.is_alive():
        process.terminate()
        await to_thread(process.join, 5)
        pytest.fail("fresh-process direct SDK continuation timed out")
    assert process.exitcode == 0
    success, payload = parent_connection.recv()
    parent_connection.close()
    assert success, payload
    output, child_handle_values = payload
    assert output == "durable-second"
    assert child_handle_values[0] == handle_values[0]
    assert child_handle_values[1] != handle_values[1]

    audit_store = pgsql_harness.store()
    await audit_store.open()
    child_checkpoint = await audit_store.load(
        conversation.CheckpointId(child_handle_values[1]),
        authority(),
    )
    assert child_checkpoint.identity.parent_checkpoint_id == (
        first.handle.checkpoint_id
    )
    assert _OPAQUE_STATE not in repr(child_checkpoint).encode("utf-8")
    ciphertexts = await _encrypted_payloads(audit_store)
    assert ciphertexts
    assert all(_OPAQUE_STATE not in ciphertext for ciphertext in ciphertexts)
