"""Exercise the dormant Phase 12 data-only diagnostic association boundary."""

from asyncio import Event, create_task, gather, run, sleep
from base64 import b64decode, b64encode, urlsafe_b64encode
from collections.abc import Iterator
from dataclasses import replace
from gc import get_referents
from json import dumps
from pathlib import Path
from runpy import run_path

import pytest

import avalan.patch.diagnostic_association as diagnostic_association
from avalan.patch import sandbox_commit
from avalan.patch.audience_projection import PatchAudienceProjectionSource
from avalan.patch.codec import encode_result
from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    ArtifactState,
    CommitStepState,
    CommitTruth,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    LineageState,
    MutationState,
    PatchApprovalId,
    PatchCommitOwnerId,
    PatchGrantId,
    PatchObserverCorrelationId,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PostconditionState,
    RequestedEffectOccurrence,
    WorkspaceChange,
)
from avalan.patch.durable_approval import (
    DurableApprovalSigningKey,
    HmacDurableApprovalAuthority,
)
from avalan.patch.durable_store import (
    DurableApproval,
    DurableArtifactState,
    DurablePatchStore,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableRetentionKind,
    DurableStoreLimits,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.policy import (
    DiagnosticPolicyId,
    PolicyDisclosure,
    SealedPlan,
    cleanup_sealed_authorities,
    seal_plan,
)

_PHASE_FIVE = run_path(
    str(Path("tests/patch/phase_5_contract_test.py").resolve())
)
_ERROR = diagnostic_association.DiagnosticAssociationErrorCode
_COMMAND = diagnostic_association.DiagnosticCommandClass
_OUTCOME = diagnostic_association.DiagnosticSandboxOutcome
_SEAL_CLEANUP_TICK = ExpiryTick(2**63 - 1)


@pytest.fixture(autouse=True)
def _phase_twelve_seal_lifecycle() -> Iterator[None]:
    """Release test-local plan seals at each Phase 12 lifecycle boundary."""
    cleanup_sealed_authorities(_SEAL_CLEANUP_TICK)
    yield
    cleanup_sealed_authorities(_SEAL_CLEANUP_TICK)


async def _terminal_store(
    *, diagnostic_policy: bool = True, disclosure: bool = True
) -> tuple[
    DurablePatchStore,
    SealedPlan,
    DurableRequestAccess,
    PatchObserverCorrelationId,
]:
    """Create an actual committed durable request from a sealed plan."""
    disclosures = (
        frozenset((PolicyDisclosure.DIAGNOSTIC_ASSOCIATION,))
        if disclosure
        else frozenset()
    )
    base = await _PHASE_FIVE["_sealed_plan"](disclosures=disclosures)
    plan = seal_plan(
        base.plan_id.__class__.new(),
        replace(
            base.binding,
            diagnostic_policy=(
                DiagnosticPolicyId("diagnostic-read-only")
                if diagnostic_policy
                else None
            ),
        ),
        base.candidate,
        base.review.expiry,
    )
    durable_plan = sandbox_commit._durable_plan(plan)
    subject = plan.binding.subject
    identity = DurableRequestIdentity(
        subject.tenant,
        subject.principal,
        plan.binding.request.execution_id,
        plan.binding.final.approval.route,
        RetransmissionKey("phase12-diagnostic"),
    )
    access = DurableRequestAccess(plan.binding.request.request_id, identity)
    authority = HmacDurableApprovalAuthority(
        DurableApprovalSigningKey(b"d" * 32)
    )
    store = InMemoryDurablePatchStore(
        InMemoryDurablePatchBackend(
            DurableStoreLimits(), approval_verifier=authority
        )
    )
    reservation = await store.reserve(
        identity,
        plan.binding.request_digest,
        plan.binding.request.request_id,
    )
    await store.persist_plan(reservation, durable_plan)
    artifacts = tuple(
        artifact_id
        for _, artifact_id in sandbox_commit._durable_artifacts(plan)
    )
    approval = authority.seal(
        DurableApproval(
            PatchGrantId.new(),
            PatchApprovalId.new(),
            identity,
            durable_plan.canonical_digest,
            durable_plan.plan_id,
            durable_plan.fingerprint_digest,
            durable_plan.review_digest,
            durable_plan.context_id,
            durable_plan.workspace_id,
            durable_plan.domain_id,
            plan.binding.final.revision.value,
            plan.binding.final.approval.broker,
            plan.binding.final.approval.reviewer_role,
            (subject.principal,),
            ExpiryTick(100),
            b"\x00" * 32,
        )
    )
    claim = await store.claim_commit(
        reservation,
        durable_plan,
        approval,
        PatchCommitOwnerId.new(),
        ExpiryTick(1),
        DurationTicks(50),
        artifacts,
    )
    assert claim.lease is not None
    cursor = (await store.inspect(access)).journal.cursor
    for binding in durable_plan.steps:
        cursor = (
            await store.append_step(
                claim.lease,
                cursor,
                binding.step_id,
                CommitStepState.PLANNED,
                ExpiryTick(2),
            )
        ).cursor
    for binding in durable_plan.steps:
        cursor = (
            await store.append_step(
                claim.lease,
                cursor,
                binding.step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(2),
            )
        ).cursor
    for artifact_id in artifacts:
        cursor = (
            await store.append_artifact(
                claim.lease,
                cursor,
                artifact_id,
                DurableArtifactState.PRESENT,
                ExpiryTick(2),
            )
        ).cursor
        cursor = (
            await store.append_artifact(
                claim.lease,
                cursor,
                artifact_id,
                DurableArtifactState.REMOVED,
                ExpiryTick(2),
            )
        ).cursor
    result = PatchResult(
        1,
        reservation.request_id,
        durable_plan.plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        PatchStatus.COMMITTED,
        CommitTruth(
            MutationState.COMMITTED,
            LineageState.COMMITTED,
            RequestedEffectOccurrence.TRUE,
            ArtifactState.CLEANED,
            WorkspaceChange.CHANGED,
            True,
            PostconditionState.ESTABLISHED,
        ),
        None,
    )
    correlation = PatchObserverCorrelationId.new()
    await store.settle(claim.lease, cursor, result, correlation, ExpiryTick(3))
    return store, plan, access, correlation


def _journal_bytes(snapshot: DurableRequestSnapshot) -> bytes:
    """Serialize the actual durable journal without local mutation literals."""
    return dumps(
        {
            "artifacts": tuple(
                (
                    item.cursor.revision.value,
                    item.artifact_id.value,
                    item.state.value,
                )
                for item in snapshot.journal.artifacts
            ),
            "cursor": snapshot.journal.cursor.revision.value,
            "steps": tuple(
                (
                    item.cursor.revision.value,
                    item.step_id.value,
                    item.lineage_id.value,
                    item.state.value,
                )
                for item in snapshot.journal.steps
            ),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _outbox_bytes(snapshot: DurableRequestSnapshot) -> bytes:
    """Serialize the actual terminal outbox without local mutation literals."""
    terminal = snapshot.terminal
    assert terminal is not None
    return dumps(
        (
            terminal.outbox.event_id.value,
            terminal.outbox.request_id.value,
            terminal.outbox.sequence.value,
            terminal.outbox.lifecycle.value,
            terminal.outbox.correlation_id.value,
        ),
        separators=(",", ":"),
    ).encode()


async def _host() -> tuple[
    diagnostic_association._TrustedDiagnosticHost,
    DurablePatchStore,
    SealedPlan,
    DurableRequestAccess,
]:
    """Create one trusted-only host over actual durable terminal truth."""
    store, plan, access, correlation = await _terminal_store()
    host = await diagnostic_association._trusted_diagnostic_host(
        store, plan, access, correlation
    )
    return host, store, plan, access


async def _receipt(
    host: diagnostic_association._TrustedDiagnosticHost,
) -> diagnostic_association.DiagnosticApprovalReceipt:
    """Obtain a capability and separately trusted approval receipt."""
    return await host._approve(await host._issue_capability())


def _signed_raw_payload(payload: bytes) -> bytes:
    """Sign raw test bytes for token-shape rejection coverage."""
    return (
        urlsafe_b64encode(payload)
        + b"."
        + urlsafe_b64encode(diagnostic_association._ROOT_SIGNER.sign(payload))
    )


@pytest.mark.parametrize(
    "outcome",
    (
        diagnostic_association.DiagnosticSandboxOutcome.SUCCEEDED,
        diagnostic_association.DiagnosticSandboxOutcome.FAILED,
    ),
)
def test_patch_e2e_026_preserves_actual_durable_terminal_bytes(
    monkeypatch: pytest.MonkeyPatch,
    outcome: diagnostic_association.DiagnosticSandboxOutcome,
) -> None:
    """Preserve terminal bytes for private integrity-probe containment."""

    async def fixed_probe(
        request: diagnostic_association._SealedReadOnlyProbeRequest,
    ) -> diagnostic_association.DiagnosticSandboxOutcome:
        """Fault the private integrity probe only for containment coverage."""
        request.__post_init__()
        return outcome

    async def scenario() -> None:
        """Associate independently without writing canonical patch truth."""
        host, store, plan, access = await _host()
        before = await store.inspect(access)
        assert before.terminal is not None
        original = (
            encode_result(before.terminal.result),
            _journal_bytes(before),
            _outbox_bytes(before),
        )
        monkeypatch.setattr(
            diagnostic_association, "_read_only_probe", fixed_probe
        )
        association = await host._associate(await _receipt(host))
        after = await store.inspect(access)
        claims = diagnostic_association._verify_payload(
            association, "association"
        )
        assert claims["outcome"] == outcome.value
        assert claims["request_id"] == plan.binding.request.request_id.value
        assert claims["plan_id"] == plan.plan_id.value
        assert after.terminal is not None
        assert (
            encode_result(after.terminal.result),
            _journal_bytes(after),
            _outbox_bytes(after),
        ) == original
        assert "secret-canary" not in repr(association)
        assert "stdout" not in repr(association)
        await host._aclose()

    run(scenario())


def test_e2e_026_cancel_timeout_own_task_until_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contain private integrity-probe timeout and cancellation only."""

    async def scenario() -> None:
        """Exercise bounded outcomes, busy fencing, and cleanup ownership."""
        release = Event()
        started = Event()

        async def stubborn_probe(
            request: diagnostic_association._SealedReadOnlyProbeRequest,
        ) -> diagnostic_association.DiagnosticSandboxOutcome:
            """Fault private integrity-probe cancellation containment only."""
            request.__post_init__()
            started.set()
            try:
                await release.wait()
            except BaseException:
                await release.wait()
            return diagnostic_association.DiagnosticSandboxOutcome.SUCCEEDED

        host, store, _, access = await _host()
        before = await store.inspect(access)
        assert before.terminal is not None
        original = (
            encode_result(before.terminal.result),
            _journal_bytes(before),
            _outbox_bytes(before),
        )
        monkeypatch.setattr(
            diagnostic_association, "_read_only_probe", stubborn_probe
        )
        monkeypatch.setattr(
            diagnostic_association,
            "_TRUSTED_READ_ONLY_PROBE_TIMEOUT_SECONDS",
            0.001,
        )
        monkeypatch.setattr(
            diagnostic_association,
            "_TRUSTED_PROBE_FINALIZATION_SECONDS",
            0.001,
        )
        timed = await host._associate(await _receipt(host))
        assert (
            diagnostic_association._verify_payload(timed, "association")[
                "outcome"
            ]
            == diagnostic_association.DiagnosticSandboxOutcome.TIMED_OUT.value
        )
        assert host._running_task is not None
        with pytest.raises(
            diagnostic_association.DiagnosticAssociationError
        ) as stale:
            await host._associate(await _receipt(host))
        assert stale.value.code is _ERROR.APPROVAL_INVALID
        release.set()
        await sleep(0)
        await host._aclose()
        assert host._running_task is None
        after = await store.inspect(access)
        assert after.terminal is not None
        assert (
            encode_result(after.terminal.result),
            _journal_bytes(after),
            _outbox_bytes(after),
        ) == original

        release = Event()
        started = Event()
        host, _, _, _ = await _host()
        task = create_task(host._associate(await _receipt(host)))
        await started.wait()
        task.cancel()
        cancelled = await task
        assert (
            diagnostic_association._verify_payload(cancelled, "association")[
                "outcome"
            ]
            == diagnostic_association.DiagnosticSandboxOutcome.CANCELLED.value
        )
        assert host._running_task is not None
        release.set()
        await sleep(0)
        await host._aclose()

    run(scenario())


def test_receipt_replay_concurrency_and_malformed_tokens_are_fenced() -> None:
    """Consume one receipt once and coarsen every malformed token failure."""

    async def scenario() -> None:
        """Exercise copied receipt fencing before its first await."""
        host, _, _, _ = await _host()
        receipt = await _receipt(host)
        first, second = await gather(
            host._associate(receipt),
            host._associate(
                diagnostic_association.DiagnosticApprovalReceipt(
                    bytes(receipt)
                )
            ),
            return_exceptions=True,
        )
        values = tuple(item for item in (first, second) if type(item) is bytes)
        errors = tuple(
            item for item in (first, second) if isinstance(item, Exception)
        )
        assert len(values) == 1
        assert len(errors) == 1
        assert isinstance(
            errors[0], diagnostic_association.DiagnosticAssociationError
        )
        for token in (
            None,
            "canary",
            b"",
            b".",
            b"bad.token.extra",
            b"%%%.",
            b"x" * 8193,
        ):
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ) as malformed:
                diagnostic_association._verify_payload(token, "approval")
            assert malformed.value.code is _ERROR.APPROVAL_INVALID
            assert "canary" not in str(malformed.value)
        await host._aclose()

    run(scenario())


def test_public_delivery_has_no_host_signer_store_or_plan_graph() -> None:
    """Deliver only bytes, never a bound method or trusted object graph."""

    async def scenario() -> None:
        """Prove ordinary delivered values are detached byte objects."""
        host, _, plan, access = await _host()
        capability = await host._issue_capability()
        receipt = await host._approve(capability)
        association = await host._associate(receipt)
        capability_claims = diagnostic_association._verify_payload(
            capability, "capability"
        )
        assert (
            capability_claims["execution_id"]
            == plan.binding.request.execution_id.value
        )
        assert capability_claims["request_id"] == access.request_id.value
        assert capability_claims["plan_id"] == plan.plan_id.value
        assert capability_claims["policy_id"] == host._policy_id.value
        assert (
            capability_claims["audience"]
            == diagnostic_association._DIAGNOSTIC_AUDIENCE
        )
        assert capability_claims["service_id"] == host._service_id
        assert capability_claims["binding"]
        receipt_claims = diagnostic_association._verify_payload(
            receipt, "approval"
        )
        for name in (
            "audience",
            "binding",
            "execution_id",
            "plan_id",
            "policy_id",
            "request_id",
            "service_id",
        ):
            assert receipt_claims[name] == capability_claims[name]
        assert receipt_claims["association_id"]
        other_host, _, _, _ = await _host()
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            await other_host._verify_capability(capability)
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            other_host._receipt(receipt)
        audience_claims = host._receipt(receipt)
        audience_claims["audience"] = "other_audience"
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            host._receipt(
                diagnostic_association.DiagnosticApprovalReceipt(
                    diagnostic_association._sign_payload(audience_claims)
                )
            )
        for value in (capability, receipt, association):
            assert type(value) is bytes
            assert get_referents(value) == []
            assert not hasattr(value, "__self__")
            assert not hasattr(value, "approve")
        assert not hasattr(
            diagnostic_association, "create_diagnostic_association_service"
        )
        assert not hasattr(
            diagnostic_association, "DiagnosticAssociationService"
        )
        await host._aclose()

    run(scenario())


def test_fixed_token_slots_bound_high_volume_and_reentrant_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep one capability, receipt, association, and probe per host."""

    async def scenario() -> None:
        """Exercise repeated, concurrent, stale, and reentrant slot access."""
        host, _, _, _ = await _host()
        capabilities = await gather(
            *(host._issue_capability() for _ in range(128))
        )
        capability = capabilities[0]
        assert all(item == capability for item in capabilities)
        capability_claims = diagnostic_association._verify_payload(
            capability, "capability"
        )
        receipts = await gather(
            *(host._approve(capability) for _ in range(128))
        )
        receipt = receipts[0]
        assert all(item == receipt for item in receipts)
        receipt_claims = diagnostic_association._verify_payload(
            receipt, "approval"
        )
        probe_calls = 0
        reentrant_errors = 0

        async def counted_probe(
            request: diagnostic_association._SealedReadOnlyProbeRequest,
        ) -> diagnostic_association.DiagnosticSandboxOutcome:
            """Count the private integrity probe without external execution."""
            nonlocal probe_calls, reentrant_errors
            request.__post_init__()
            probe_calls += 1
            try:
                await host._associate(receipt)
            except diagnostic_association.DiagnosticAssociationError:
                reentrant_errors += 1
            await sleep(0)
            return diagnostic_association.DiagnosticSandboxOutcome.SUCCEEDED

        with monkeypatch.context() as context:
            context.setattr(
                diagnostic_association, "_read_only_probe", counted_probe
            )
            attempts = await gather(
                *(host._associate(receipt) for _ in range(128)),
                return_exceptions=True,
            )
        associations = tuple(item for item in attempts if type(item) is bytes)
        failures = tuple(
            item for item in attempts if isinstance(item, Exception)
        )
        assert len(associations) == 1
        assert len(failures) == 127
        assert probe_calls == 1
        assert reentrant_errors == 1
        association_claims = diagnostic_association._verify_payload(
            associations[0], "association"
        )
        assert (
            association_claims["association_id"]
            == receipt_claims["association_id"]
        )
        assert host._consumed_receipt_id == receipt_claims["receipt_id"]
        assert host._capability == capability
        assert host._approval_receipt == receipt
        assert not hasattr(host, "_approved_capabilities")
        assert not hasattr(host, "_consumed_receipts")
        for name in (
            "_capability",
            "_capability_issuing",
            "_approval_receipt",
            "_approval_issuing",
            "_consumed_receipt_id",
        ):
            assert type(getattr(host, name)) not in (dict, list, set)
        assert capability_claims["capability_id"]
        assert receipt_claims["receipt_id"]
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            await host._associate(receipt)
        await host._aclose()

    run(scenario())


def test_fixed_token_slots_reserve_concurrent_and_invalid_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reserve fixed slots before await and reject every stale transition."""

    async def scenario() -> None:
        """Cover bounded reservation, scope, and consumed-state fences."""
        host, _, _, _ = await _host()
        entered = Event()
        release = Event()
        original_source = diagnostic_association._TrustedDiagnosticHost._source

        async def paused_source(
            value: diagnostic_association._TrustedDiagnosticHost,
        ) -> PatchAudienceProjectionSource:
            """Hold trusted inspection only while a slot is reserved."""
            entered.set()
            await release.wait()
            return await original_source(value)

        with monkeypatch.context() as context:
            context.setattr(
                diagnostic_association._TrustedDiagnosticHost,
                "_source",
                paused_source,
            )
            issuing = create_task(host._issue_capability())
            await entered.wait()
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ) as error:
                await host._issue_capability()
            assert error.value.code is _ERROR.CAPABILITY_INVALID
            release.set()
            capability = await issuing

        with pytest.raises(
            diagnostic_association.DiagnosticAssociationError
        ) as invalid_approval:
            await host._approve(
                diagnostic_association.DiagnosticCapability(b"x")
            )
        assert invalid_approval.value.code is _ERROR.APPROVAL_INVALID

        entered = Event()
        release = Event()
        with monkeypatch.context() as context:
            context.setattr(
                diagnostic_association._TrustedDiagnosticHost,
                "_source",
                paused_source,
            )
            approving = create_task(host._approve(capability))
            await entered.wait()
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ) as error:
                await host._approve(capability)
            assert error.value.code is _ERROR.APPROVAL_INVALID
            release.set()
            await approving

        scope_host, _, _, _ = await _host()
        scope_capability = await scope_host._issue_capability()
        object.__setattr__(scope_host, "_service_id", "other-service")
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            await scope_host._verify_capability(scope_capability)

        receipt_host, _, _, _ = await _host()
        receipt = await _receipt(receipt_host)
        object.__setattr__(receipt_host, "_service_id", "other-service")
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            receipt_host._receipt(receipt)

        busy_host, _, _, _ = await _host()
        busy_receipt = await _receipt(busy_host)
        pending = Event()

        async def pending_probe() -> (
            diagnostic_association.DiagnosticSandboxOutcome
        ):
            """Remain live to model host-owned prior work."""
            await pending.wait()
            return _OUTCOME.SUCCEEDED

        busy_task = create_task(pending_probe())
        busy_host._running_task = busy_task
        with pytest.raises(
            diagnostic_association.DiagnosticAssociationError
        ) as busy:
            await busy_host._associate(busy_receipt)
        assert busy.value.code is _ERROR.HOST_BUSY
        busy_task.cancel()
        await gather(busy_task, return_exceptions=True)
        busy_host._running_task = None

        failing_host, _, _, _ = await _host()
        failing_receipt = await _receipt(failing_host)

        async def unavailable_source(
            value: diagnostic_association._TrustedDiagnosticHost,
        ) -> PatchAudienceProjectionSource:
            """Raise a coarsened terminal error after atomic consumption."""
            del value
            raise diagnostic_association.DiagnosticAssociationError(
                _ERROR.TERMINAL_UNAVAILABLE
            )

        with monkeypatch.context() as context:
            context.setattr(
                diagnostic_association._TrustedDiagnosticHost,
                "_source",
                unavailable_source,
            )
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ) as unavailable:
                await failing_host._associate(failing_receipt)
        assert unavailable.value.code is _ERROR.TERMINAL_UNAVAILABLE
        assert failing_host._consumed_receipt_id is not None
        await host._aclose()
        await scope_host._aclose()
        await receipt_host._aclose()
        await busy_host._aclose()
        await failing_host._aclose()

    run(scenario())


def test_policy_selection_store_access_and_command_rejections() -> None:
    """Require sealed policy and reject substitutions and command classes."""

    async def scenario() -> None:
        """Check sealed policy, durable access, and fixed command shape."""
        for diagnostic_policy, disclosure in ((False, True), (True, False)):
            store, plan, access, correlation = await _terminal_store(
                diagnostic_policy=diagnostic_policy, disclosure=disclosure
            )
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ) as policy:
                await diagnostic_association._trusted_diagnostic_host(
                    store, plan, access, correlation
                )
            assert policy.value.code is _ERROR.POLICY_INVALID
        store, plan, access, correlation = await _terminal_store()
        with pytest.raises(
            diagnostic_association.DiagnosticAssociationError
        ) as unsealed:
            await diagnostic_association._trusted_diagnostic_host(
                store,
                replace(plan, plan_id=plan.plan_id.__class__.new()),
                access,
                correlation,
            )
        assert unsealed.value.code is _ERROR.POLICY_INVALID
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            await diagnostic_association._trusted_diagnostic_host(
                store,
                plan,
                DurableRequestAccess(PatchRequestId.new(), access.identity),
                correlation,
            )
        for command in _COMMAND:
            if command is _COMMAND.READ_ONLY_PROBE:
                continue
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ) as rejected:
                diagnostic_association._SealedReadOnlyProbeRequest(
                    command, (), (), "<detached-read-only-probe>", b"x" * 32
                )
            assert rejected.value.code is _ERROR.PROHIBITED_COMMAND
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            diagnostic_association._SealedReadOnlyProbeRequest(
                diagnostic_association.DiagnosticCommandClass.READ_ONLY_PROBE,
                ("test",),
                (),
                "<detached-read-only-probe>",
                b"x" * 32,
            )

    run(scenario())
    assert (
        diagnostic_association.diagnostic_retention_kind()
        is DurableRetentionKind.DIAGNOSTIC_ASSOCIATION
    )


def test_remediation_is_typed_as_a_new_patch_request_only() -> None:
    """Leave formatter and fixer remediation outside diagnostic authority."""
    assert not hasattr(diagnostic_association, "ReadOnlyDiagnosticSandbox")
    assert not hasattr(diagnostic_association, "DiagnosticSandboxExecutor")


def test_private_failure_fences_are_coarsened_and_owned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every trusted-host failure outcome without a public port."""

    async def scenario() -> None:
        """Drive claims, callbacks, finalizers, and source failures."""
        for constructor in (
            diagnostic_association.RemediationPatchAuthorization,
            diagnostic_association.FormatterFixerResult,
        ):
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ):
                type.__call__(constructor, None)

        host, _, _, access = await _host()
        capability = await host._issue_capability()
        receipt = await host._approve(capability)
        assert await host._approve(capability) == receipt

        capability_claims = await host._verify_capability(capability)
        capability_claims["service_id"] = "another-service"
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            await host._verify_capability(
                diagnostic_association.DiagnosticCapability(
                    diagnostic_association._sign_payload(capability_claims)
                )
            )
        receipt_claims = host._receipt(receipt)
        receipt_claims["service_id"] = "another-service"
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            host._receipt(
                diagnostic_association.DiagnosticApprovalReceipt(
                    diagnostic_association._sign_payload(receipt_claims)
                )
            )

        source = await host._source()
        snapshot = source._snapshot
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            diagnostic_association._terminal_truth(object(), access)
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            diagnostic_association._terminal_truth(
                object.__new__(PatchAudienceProjectionSource), access
            )
        object.__setattr__(snapshot, "pending", object())
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            diagnostic_association._terminal_truth(source, access)
        object.__setattr__(snapshot, "pending", None)
        terminal = snapshot.terminal
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            object.__setattr__(snapshot, "terminal", None)
            diagnostic_association._canonical_terminal_snapshot(
                source.plan, snapshot
            )
        object.__setattr__(snapshot, "terminal", terminal)
        terminal_claims = host._receipt(receipt)
        terminal_claims["request_id"] = "request-substitution"
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            diagnostic_association._require_payload_terminal(
                terminal_claims,
                snapshot,
                diagnostic_association._terminal_binding(source),
                "approval",
            )

        subject = host._plan.binding.subject
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            diagnostic_association._payload_string({}, "missing", "approval")
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            diagnostic_association._require_payload_subject(
                {
                    "principal": "other",
                    "tenant": subject.tenant.value,
                    "run": subject.run.value,
                    "session": subject.session.value,
                    "task": subject.task.value,
                    "agent": subject.agent.value,
                },
                subject,
                "approval",
            )
        assert (
            diagnostic_association._token_error_code("association")
            is _ERROR.ASSOCIATION_INVALID
        )
        for encoded in (
            b"[]",
            b'{"kind":"approval","version":"1","bad":1}',
            b'{"kind":"approval","version":"2"}',
        ):
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ):
                diagnostic_association._verify_payload(
                    _signed_raw_payload(encoded), "approval"
                )
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            diagnostic_association._verify_payload(b"e30=.eA==", "approval")
        for padding in range(128):
            outer_payload = {
                "kind": "approval",
                "outer_padding": "x" * padding,
                "version": "1",
            }
            canonical = diagnostic_association._sign_payload(outer_payload)
            payload_segment, signature_segment = canonical.split(b".")
            standard_signature = b64encode(
                b64decode(signature_segment, altchars=b"-_", validate=True)
            )
            if standard_signature != signature_segment:
                break
        else:
            raise AssertionError
        alternate_transport = (
            payload_segment.rstrip(b"=") + b"." + signature_segment,
            payload_segment + b"." + standard_signature,
            payload_segment.lower() + b"." + signature_segment,
            payload_segment + b"\n." + signature_segment,
            payload_segment + b"." + signature_segment + b" ",
        )
        for alternate in alternate_transport:
            if alternate == canonical:
                continue
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ) as error:
                diagnostic_association._verify_payload(alternate, "approval")
            assert error.value.code is _ERROR.APPROVAL_INVALID
            assert "outer_padding" not in str(error.value)
        valid_payload = {"kind": "approval", "version": "1"}
        valid = diagnostic_association._sign_payload(valid_payload)

        def broken_loads(value: bytes) -> object:
            """Raise a non-parser failure at the trusted decoder seam."""
            del value
            raise RuntimeError("decoder-canary")

        with monkeypatch.context() as context:
            context.setattr(diagnostic_association, "loads", broken_loads)
            with pytest.raises(
                diagnostic_association.DiagnosticAssociationError
            ) as error:
                diagnostic_association._verify_payload(valid, "approval")
            assert "decoder-canary" not in str(error.value)

        async def unavailable_probe(
            request: diagnostic_association._SealedReadOnlyProbeRequest,
        ) -> diagnostic_association.DiagnosticSandboxOutcome:
            """Fault private containment without external diagnostic work."""
            del request
            raise RuntimeError("unavailable")

        with monkeypatch.context() as context:
            context.setattr(
                diagnostic_association, "_read_only_probe", unavailable_probe
            )
            assert (
                await host._run_probe(
                    diagnostic_association._sealed_probe_request(b"x" * 32)
                )
                is diagnostic_association.DiagnosticSandboxOutcome.UNAVAILABLE
            )

        async def cancelled_source(
            value: diagnostic_association._TrustedDiagnosticHost,
        ) -> PatchAudienceProjectionSource:
            """Model cancellation before any diagnostic invocation starts."""
            del value
            raise asyncio.CancelledError

        with monkeypatch.context() as context:
            import asyncio

            context.setattr(
                diagnostic_association._TrustedDiagnosticHost,
                "_source",
                cancelled_source,
            )
            cancelled = await host._associate(receipt)
            assert (
                diagnostic_association._verify_payload(
                    cancelled, "association"
                )["outcome"]
                == _OUTCOME.CANCELLED.value
            )

        class RuntimeFailure:
            """Provide a private invalid projection host for coarsening."""

            async def issue_access(
                self, value: object, correlation: object
            ) -> object:
                """Fail without exposing durable diagnostics."""
                del value, correlation
                raise RuntimeError("source-canary")

        object.__setattr__(host, "_projection_host", RuntimeFailure())
        with pytest.raises(
            diagnostic_association.DiagnosticAssociationError
        ) as source_error:
            await host._source()
        assert "source-canary" not in str(source_error.value)
        host, _, _, _ = await _host()

        invalid_receipt = host._receipt(await _receipt(host))
        invalid_receipt["binding"] = "binding-substitution"
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            await host._associate(
                diagnostic_association.DiagnosticApprovalReceipt(
                    diagnostic_association._sign_payload(invalid_receipt)
                )
            )

        async def broken_run_probe(
            request: diagnostic_association._SealedReadOnlyProbeRequest,
        ) -> diagnostic_association.DiagnosticSandboxOutcome:
            """Fault private containment without external diagnostic work."""
            del request
            raise RuntimeError("run-probe-canary")

        with monkeypatch.context() as context:
            context.setattr(
                diagnostic_association._TrustedDiagnosticHost,
                "_run_probe",
                broken_run_probe,
            )
            unavailable = await host._associate(await _receipt(host))
            assert (
                diagnostic_association._verify_payload(
                    unavailable, "association"
                )["outcome"]
                == _OUTCOME.UNAVAILABLE.value
            )

        async def done_value() -> (
            diagnostic_association.DiagnosticSandboxOutcome
        ):
            """Return the successful fixed diagnostic outcome."""
            return diagnostic_association.DiagnosticSandboxOutcome.SUCCEEDED

        async def completed_finalizer() -> None:
            """Complete an already-owned finalizer for close coverage."""

        task = create_task(done_value())
        await task
        with pytest.raises(diagnostic_association.DiagnosticAssociationError):
            host._begin_finalization(-1, task)
        host._running_task = task
        host._finalizer = create_task(host._finalize(host._generation, task))
        await host._finalizer
        assert host._finalizer is None

        async def cancelled_value() -> (
            diagnostic_association.DiagnosticSandboxOutcome
        ):
            """Remain pending until cancellation for finalizer coverage."""
            await Event().wait()
            raise AssertionError

        cancelled_task = create_task(cancelled_value())
        cancelled_task.cancel()
        await gather(cancelled_task, return_exceptions=True)
        await host._finalize(host._generation, cancelled_task)
        host._task_completed(-1, cancelled_task)

        async def failed_value() -> (
            diagnostic_association.DiagnosticSandboxOutcome
        ):
            """Raise one bounded diagnostic worker failure."""
            raise RuntimeError("worker")

        failed_task = create_task(failed_value())
        await gather(failed_task, return_exceptions=True)
        await host._finalize(host._generation, failed_task)
        host._task_completed(-1, failed_task)

        wait_task: Event = Event()
        finalizer = create_task(wait_task.wait())
        waiting = create_task(host._wait_for_finalizer(finalizer))
        await sleep(0)
        waiting.cancel()
        await waiting
        finalizer.cancel()
        await gather(finalizer, return_exceptions=True)
        failed_finalizer = create_task(failed_value())
        await host._wait_for_finalizer(failed_finalizer)
        host._running_task = None
        host._finalizer = create_task(completed_finalizer())
        await host._finalizer
        await host._aclose()
        previous_finalizer = host._finalizer

        async def pending_value() -> (
            diagnostic_association.DiagnosticSandboxOutcome
        ):
            """Remain pending until the replacement finalizer cancels it."""
            await Event().wait()
            raise AssertionError

        pending = create_task(pending_value())
        host._running_task = pending
        replacement = host._begin_finalization(host._generation, pending)
        assert replacement is not previous_finalizer
        await replacement
        await host._aclose()

    run(scenario())
