"""Exercise remote continuation through an isolated PostgreSQL store."""

from asyncio import Event, run, to_thread
from dataclasses import dataclass
from os import environ
from pathlib import Path
from runpy import run_path
from sys import path as sys_path
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from patch_activation_support import patch_test_activation_factory

from avalan.patch.domain import Capability, DurationTicks, ExpiryTick
from avalan.patch.durable_approval import (
    DurableApprovalSigningKey,
    HmacDurableApprovalAuthority,
    PhaseFiveDurableApprovalIssuer,
)
from avalan.patch.durable_store import DurableCommitLease
from avalan.patch.pgsql_store import (
    PgsqlDurablePatchStore,
    PgsqlDurablePatchStoreSettings,
)
from avalan.patch.planner import (
    BoundedPlannerWorker,
    PlannerFacade,
    PlannerLimits,
)
from avalan.patch.policy import ApprovalService, RuntimeGrantStore
from avalan.patch.sandbox_commit import (
    SandboxPatchRuntimeBinder,
    SandboxPatchRuntimeSettings,
    SandboxPatchServiceConfiguration,
)
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchPersistenceBinding,
    RemotePatchRuntimeWitness,
)
from avalan.pgsql import (
    PgsqlDatabase,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.server.patch import (
    RemotePatchAuthority,
    RemotePatchAuthorityResolver,
    RemotePatchEditPart,
    RemotePatchOperation,
    RemotePatchTestClient,
    RemotePatchTestServerConfiguration,
    RemotePatchTestServerProfile,
    install_remote_patch_test_routes,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")

pytestmark = pytest.mark.skipif(
    _DSN is None,
    reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
)


@dataclass(frozen=True, slots=True)
class _Resolver(RemotePatchAuthorityResolver):
    """Resolve the one test-host authenticated authority tuple."""

    authority: RemotePatchAuthority

    async def __call__(self, _: Request) -> RemotePatchAuthority | None:
        """Return only the exact authenticated test authority."""
        return self.authority


class _FencedPgsqlStore(PgsqlDurablePatchStore):
    """Pause a selected worker at its second durable fence observation."""

    def __init__(
        self,
        database: PgsqlDatabase,
        signer: HmacDurableApprovalAuthority,
    ) -> None:
        """Open one owned PostgreSQL store with a test-only fence gate."""
        super().__init__(
            database,
            owns_database=True,
            approval_verifier=signer,
        )
        self.effect_reached = Event()
        self.release_effect = Event()
        self.checks = 0

    async def is_current_fence(
        self,
        lease: DurableCommitLease,
        now: ExpiryTick,
    ) -> bool:
        """Block once after fencing and before the selected worker effect."""
        self.checks += 1
        if self.checks == 2:
            self.effect_reached.set()
            await self.release_effect.wait()
        return await super().is_current_fence(lease, now)


def _phase_ten() -> dict[str, object]:
    """Load the selected-runtime fixture builders without production wiring."""
    sys_path.insert(0, "tests/patch")
    try:
        return run_path("tests/patch/phase_10_contract_test.py")
    finally:
        sys_path.remove("tests/patch")


async def _open_store(
    dsn: str,
    schema: str,
    signer: HmacDurableApprovalAuthority,
    *,
    fenced: bool,
) -> PgsqlDurablePatchStore:
    """Open one independently owned durable PostgreSQL store client."""
    settings = PgsqlDurablePatchStoreSettings(
        dsn=dsn,
        schema=schema,
        pool_minimum=1,
        pool_maximum=2,
    )
    store: PgsqlDurablePatchStore
    if fenced:
        store = _FencedPgsqlStore(settings.database(), signer)
    else:
        store = PgsqlDurablePatchStore.from_settings(
            settings,
            approval_verifier=signer,
        )
    await store.open()
    return store


async def _drop_schema(dsn: str, schema: str) -> None:
    """Remove only the isolated schema owned by one PostgreSQL test."""
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


def _configuration(
    root: Path,
    namespace: Path,
    store: PgsqlDurablePatchStore,
    signer: HmacDurableApprovalAuthority,
    handle_key: bytes,
    *,
    recovery_tick: int = 10,
) -> tuple[RemotePatchTestServerConfiguration, RemotePatchAuthority]:
    """Build the exact selected sandbox and trusted server scope bindings."""
    helpers = _phase_ten()
    settings_factory = helpers["_settings"]
    subject_factory = helpers["_runtime_subject"]
    policy_factory = helpers["_sandbox_corpus_policy"]
    clock_type = helpers["_RuntimeClock"]
    broker_type = helpers["_RuntimeBroker"]
    assert callable(settings_factory)
    assert callable(subject_factory)
    assert callable(policy_factory)
    assert callable(clock_type)
    assert callable(broker_type)
    settings = settings_factory(root, namespace)
    assert isinstance(settings, SandboxPatchRuntimeSettings)
    subject = subject_factory()
    policy = policy_factory()
    clock = clock_type()
    if recovery_tick > 10:
        clock.advance(recovery_tick)
    approvals = ApprovalService(broker_type(), clock, RuntimeGrantStore())
    service = SandboxPatchServiceConfiguration(
        subject,
        PlannerFacade(BoundedPlannerWorker(1), PlannerLimits()),
        approvals,
        PhaseFiveDurableApprovalIssuer(approvals, signer),
        clock,
        DurationTicks(10),
        DurationTicks(10),
    )
    binder = SandboxPatchRuntimeBinder.from_settings(
        settings,
        service,
        policy,
        PatchApprovalBinding(True),
        PatchCoordinatorBinding(True, store),
        PatchPersistenceBinding(True, store),
    )
    authority = RemotePatchAuthority(
        tenant=subject.tenant,
        principal=subject.principal,
        run=subject.run,
        session=subject.session,
        task=subject.task,
        agent=subject.agent,
        execution_scope=settings.context.identity.domain_id.value,
        route=policy.approval.route,
        context=settings.context.identity.context_id,
        workspace=settings.context.identity.workspace_id,
        policy_revision=policy.revision,
        disclosures=frozenset(),
        approval_route=policy.approval.route,
        correlation="pgsql-sandbox-remote-correlation",
        capabilities=frozenset(
            (
                Capability.READ_FOR_MUTATION,
                Capability.OBSERVE_MUTATION_PRECONDITIONS,
            )
        ),
    )
    return (
        RemotePatchTestServerConfiguration(
            profile=RemotePatchTestServerProfile(
                enabled=True,
                authenticated=True,
                loopback_only=True,
            ),
            authority_resolver=_Resolver(authority),
            expected_authority=authority,
            binder=binder,
            activation_factory=patch_test_activation_factory(),
            store=store,
            handle_key=handle_key,
            runtime_witness=RemotePatchRuntimeWitness(
                tenant=authority.tenant,
                principal=authority.principal,
                run=authority.run,
                session=authority.session,
                task=authority.task,
                agent=authority.agent,
                execution_scope=authority.execution_scope,
                route=authority.route,
                context=authority.context,
                workspace=authority.workspace,
                policy_revision=authority.policy_revision,
                disclosures=authority.disclosures,
                approval_route=authority.approval_route,
                capabilities=authority.capabilities,
            ),
        ),
        authority,
    )


async def _edit_and_await(
    configuration: RemotePatchTestServerConfiguration,
    authority: RemotePatchAuthority,
    key: str,
) -> tuple[RemotePatchOperation, RemotePatchOperation]:
    """Run one public remote edit and return its initial and final records."""
    app = FastAPI()
    controller = install_remote_patch_test_routes(app, configuration)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as http_client:
            client = RemotePatchTestClient(http_client, authority.correlation)
            operation = await client.edit(
                "note.txt",
                [RemotePatchEditPart(old_text="before\n", new_text="after\n")],
                key,
            )
            terminal = await client.await_result(operation.operation_handle)
        return operation, terminal
    finally:
        await controller.close()


async def _apply_and_await(
    configuration: RemotePatchTestServerConfiguration,
    authority: RemotePatchAuthority,
    key: str,
) -> tuple[RemotePatchOperation, RemotePatchOperation]:
    """Run one public remote apply and return its initial and final records."""
    app = FastAPI()
    controller = install_remote_patch_test_routes(app, configuration)
    document = "\n".join(
        (
            "*** Begin Patch v1",
            "*** Update File: note.txt",
            "@@",
            "-before",
            "+after",
            "*** End Patch",
        )
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1)),
            base_url="http://testserver",
        ) as http_client:
            client = RemotePatchTestClient(http_client, authority.correlation)
            operation = await client.apply(document, key)
            terminal = await client.await_result(operation.operation_handle)
        return operation, terminal
    finally:
        await controller.close()


def test_remote_pgsql_selected_sandbox_replays_terminal_after_restart(
    tmp_path: Path,
) -> None:
    """Replay a terminal public operation through a fresh PostgreSQL server."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_remote_pgsql_replay_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        root = tmp_path / "sandbox-view"
        namespace = tmp_path / "sandbox-private"
        root.mkdir()
        namespace.mkdir()
        note = root / "note.txt"
        note.write_text("before\n", encoding="utf-8")
        signer = HmacDurableApprovalAuthority(
            DurableApprovalSigningKey(b"p" * 32)
        )
        handle_key = b"r" * 32
        try:
            first_store = await _open_store(_DSN, schema, signer, fenced=False)
            try:
                first_configuration, authority = _configuration(
                    root,
                    namespace,
                    first_store,
                    signer,
                    handle_key,
                )
                operation, terminal = await _apply_and_await(
                    first_configuration,
                    authority,
                    "pgsql-terminal-replay",
                )
                assert operation.state == "pending"
                assert terminal.state == "completed"
                assert note.read_text(encoding="utf-8") == "after\n"
            finally:
                await first_store.aclose()
            restarted_store = await _open_store(
                _DSN, schema, signer, fenced=False
            )
            try:
                restarted_configuration, restarted_authority = _configuration(
                    root,
                    namespace,
                    restarted_store,
                    signer,
                    handle_key,
                )
                app = FastAPI()
                controller = install_remote_patch_test_routes(
                    app, restarted_configuration
                )
                document = "\n".join(
                    (
                        "*** Begin Patch v1",
                        "*** Update File: note.txt",
                        "@@",
                        "-before",
                        "+after",
                        "*** End Patch",
                    )
                )
                try:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(
                            app=app,
                            client=("127.0.0.1", 1),
                        ),
                        base_url="http://testserver",
                    ) as http_client:
                        client = RemotePatchTestClient(
                            http_client, restarted_authority.correlation
                        )
                        replay = await client.apply(
                            document, "pgsql-terminal-replay"
                        )
                        replay_terminal = await client.await_result(
                            replay.operation_handle
                        )
                        async with client.events(
                            replay.operation_handle
                        ) as events:
                            emitted = [event async for event in events]
                        terminal_events = [
                            event
                            for event in emitted
                            if event.lifecycle == "request_completed"
                        ]
                        assert len(terminal_events) == 1
                        terminal_event = terminal_events[0]
                        async with client.events(
                            replay.operation_handle
                        ) as replay_stream:
                            replayed = [event async for event in replay_stream]
                        replayed_terminals = [
                            event
                            for event in replayed
                            if event.lifecycle == "request_completed"
                        ]
                        assert len(replayed_terminals) == 1
                        assert replayed_terminals[0] == terminal_event
                        async with client.events(
                            replay.operation_handle,
                            after=terminal_event.cursor,
                        ) as resumed:
                            with pytest.raises(StopAsyncIteration):
                                await anext(resumed)
                finally:
                    await controller.close()
                assert replay.state == "completed"
                assert replay_terminal.state == "completed"
                assert replay_terminal.event_cursor == terminal_event.cursor
                assert terminal_event.lifecycle == "request_completed"
                assert note.read_text(encoding="utf-8") == "after\n"
            finally:
                await restarted_store.aclose()
        finally:
            await _drop_schema(_DSN, schema)

    run(scenario())


def test_remote_pgsql_fenced_worker_crash_attaches_same_key_once(
    tmp_path: Path,
) -> None:
    """Fence a killed real worker before a fresh server attaches its key."""
    assert _DSN is not None

    async def scenario() -> None:
        schema = "patch_remote_pgsql_fence_" + uuid4().hex
        await to_thread(
            task_pgsql_upgrade,
            PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
        )
        root = tmp_path / "sandbox-view"
        namespace = tmp_path / "sandbox-private"
        root.mkdir()
        namespace.mkdir()
        note = root / "note.txt"
        note.write_text("before\n", encoding="utf-8")
        signer = HmacDurableApprovalAuthority(
            DurableApprovalSigningKey(b"q" * 32)
        )
        handle_key = b"s" * 32
        try:
            store = await _open_store(_DSN, schema, signer, fenced=True)
            assert isinstance(store, _FencedPgsqlStore)
            controller = None
            try:
                configuration, authority = _configuration(
                    root,
                    namespace,
                    store,
                    signer,
                    handle_key,
                )
                app = FastAPI()
                controller = install_remote_patch_test_routes(
                    app, configuration
                )
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(
                        app=app,
                        client=("127.0.0.1", 1),
                    ),
                    base_url="http://testserver",
                ) as http_client:
                    client = RemotePatchTestClient(
                        http_client, authority.correlation
                    )
                    pending = await client.edit(
                        "note.txt",
                        [
                            RemotePatchEditPart(
                                old_text="before\n",
                                new_text="after\n",
                            )
                        ],
                        "pgsql-fenced-retransmission",
                    )
                    assert pending.state == "pending"
                    await store.effect_reached.wait()
                    attached = await client.edit(
                        "note.txt",
                        [
                            RemotePatchEditPart(
                                old_text="before\n",
                                new_text="after\n",
                            )
                        ],
                        "pgsql-fenced-retransmission",
                    )
                    assert attached.state == "pending"
                    assert store.checks >= 2
                    binder = configuration.binder
                    assert isinstance(binder, SandboxPatchRuntimeBinder)
                    runtime = binder.runtime
                    process = runtime._process._process
                    assert process is not None
                    process.terminate()
                    await process.wait()
                    store.release_effect.set()
                resolved = controller._operation(
                    authority,
                    "pgsql-fenced-retransmission",
                )
                await controller.close()
                controller = None
                snapshot = await store.inspect(resolved.access)
                assert snapshot.worker_bound
                assert snapshot.worker_reaped
                assert snapshot.terminal is None
                assert note.read_text(encoding="utf-8") == "before\n"
            finally:
                store.release_effect.set()
                if controller is not None:
                    await controller.close()
                await store.aclose()
            restarted_store = await _open_store(
                _DSN, schema, signer, fenced=False
            )
            try:
                restarted_configuration, restarted_authority = _configuration(
                    root,
                    namespace,
                    restarted_store,
                    signer,
                    handle_key,
                    recovery_tick=20,
                )
                app = FastAPI()
                restarted_controller = install_remote_patch_test_routes(
                    app, restarted_configuration
                )
                try:
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(
                            app=app,
                            client=("127.0.0.1", 1),
                        ),
                        base_url="http://testserver",
                    ) as http_client:
                        client = RemotePatchTestClient(
                            http_client, restarted_authority.correlation
                        )
                        replay = await client.edit(
                            "note.txt",
                            [
                                RemotePatchEditPart(
                                    old_text="before\n",
                                    new_text="after\n",
                                )
                            ],
                            "pgsql-fenced-retransmission",
                        )
                        terminal = await client.await_result(
                            replay.operation_handle
                        )
                        async with client.events(
                            replay.operation_handle
                        ) as events:
                            emitted = [event async for event in events]
                        terminal_events = [
                            event
                            for event in emitted
                            if event.lifecycle == "request_completed"
                        ]
                        assert len(terminal_events) == 1
                        terminal_event = terminal_events[0]
                        async with client.events(
                            replay.operation_handle
                        ) as replay_stream:
                            replayed = [event async for event in replay_stream]
                        replayed_terminals = [
                            event
                            for event in replayed
                            if event.lifecycle == "request_completed"
                        ]
                        assert len(replayed_terminals) == 1
                        assert replayed_terminals[0] == terminal_event
                        async with client.events(
                            replay.operation_handle,
                            after=terminal_event.cursor,
                        ) as resumed:
                            with pytest.raises(StopAsyncIteration):
                                await anext(resumed)
                finally:
                    await restarted_controller.close()
                assert replay.state == "pending"
                assert terminal.state == "completed"
                assert set(terminal.model_dump()) == {
                    "object",
                    "state",
                    "operation_handle",
                    "event_cursor",
                }
                assert terminal.event_cursor == terminal_event.cursor
                assert terminal_event.lifecycle == "request_completed"
                assert note.read_text(encoding="utf-8") == "before\n"
            finally:
                await restarted_store.aclose()
        finally:
            await _drop_schema(_DSN, schema)

    run(scenario())
