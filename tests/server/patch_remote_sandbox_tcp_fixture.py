"""Host the selected sandbox runtime for authenticated TCP apply evidence."""

from dataclasses import dataclass
from os import environ
from pathlib import Path
from runpy import run_path
from sys import path as sys_path

from fastapi import FastAPI, Header, HTTPException, Request
from patch_activation_support import patch_test_activation_factory

from avalan.patch.domain import (
    Capability,
    DurationTicks,
    ExpiryTick,
    PatchArtifactId,
    PatchCommitOwnerId,
    PatchObserverCorrelationId,
    PatchResult,
)
from avalan.patch.durable_approval import (
    HmacDurableApprovalAuthority,
    PhaseFiveDurableApprovalIssuer,
)
from avalan.patch.durable_store import (
    DurableApproval,
    DurableCommitClaim,
    DurableCommitLease,
    DurableJournalCursor,
    DurablePlanReference,
    DurableReservation,
    DurableTerminalRecord,
    DurableWorkerBinding,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
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
from avalan.server.patch import (
    RemotePatchAuthority,
    RemotePatchAuthorityResolver,
    RemotePatchTestServerConfiguration,
    RemotePatchTestServerProfile,
    install_remote_patch_test_routes,
)


@dataclass(frozen=True, slots=True)
class _Resolver(RemotePatchAuthorityResolver):
    """Resolve the one authenticated child-process authority."""

    authority: RemotePatchAuthority

    async def __call__(self, _: Request) -> RemotePatchAuthority | None:
        """Return the exact configured loopback authority."""
        return self.authority


class _CountingStore(InMemoryDurablePatchStore):
    """Count the actual durable commit, worker, and terminal transitions."""

    def __init__(self, signer: HmacDurableApprovalAuthority) -> None:
        """Create one private durable backend with terminal counters."""
        super().__init__(InMemoryDurablePatchBackend(approval_verifier=signer))
        self.commit_claims = 0
        self.worker_bindings = 0
        self.settlements = 0

    async def claim_commit(
        self,
        reservation: DurableReservation,
        plan: DurablePlanReference,
        approval: DurableApproval,
        owner_id: PatchCommitOwnerId,
        now: ExpiryTick,
        lease_duration: DurationTicks,
        artifact_ids: tuple[PatchArtifactId, ...],
    ) -> DurableCommitClaim:
        """Count the real commit claim before delegating to the store."""
        self.commit_claims += 1
        return await super().claim_commit(
            reservation,
            plan,
            approval,
            owner_id,
            now,
            lease_duration,
            artifact_ids,
        )

    async def bind_worker(
        self,
        lease: DurableCommitLease,
        binding: DurableWorkerBinding,
        now: ExpiryTick,
    ) -> None:
        """Count the real selected-worker binding before delegation."""
        self.worker_bindings += 1
        await super().bind_worker(lease, binding, now)

    async def settle(
        self,
        lease: DurableCommitLease,
        expected: DurableJournalCursor,
        result: PatchResult,
        correlation_id: PatchObserverCorrelationId,
        now: ExpiryTick,
    ) -> DurableTerminalRecord:
        """Count the one durable terminal settlement before delegation."""
        self.settlements += 1
        return await super().settle(
            lease,
            expected,
            result,
            correlation_id,
            now,
        )


def _phase_ten() -> dict[str, object]:
    """Load selected sandbox fixtures without normal server activation."""
    sys_path.insert(0, "tests/patch")
    try:
        return run_path("tests/patch/phase_10_contract_test.py")
    finally:
        sys_path.remove("tests/patch")


def _configuration() -> RemotePatchTestServerConfiguration:
    """Build one complete selected-sandbox remote test configuration."""
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
    settings = settings_factory(
        Path(environ["AVALAN_PATCH_SANDBOX_TCP_ROOT"]),
        Path(environ["AVALAN_PATCH_SANDBOX_TCP_NAMESPACE"]),
    )
    assert isinstance(settings, SandboxPatchRuntimeSettings)
    subject = subject_factory()
    policy = policy_factory()
    signer = HmacDurableApprovalAuthority.random()
    store = _CountingStore(signer)
    clock = clock_type()
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
        correlation="sandbox-tcp-apply-correlation",
        capabilities=frozenset(
            (
                Capability.READ_FOR_MUTATION,
                Capability.OBSERVE_MUTATION_PRECONDITIONS,
            )
        ),
    )
    return RemotePatchTestServerConfiguration(
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
        handle_key=b"t" * 32,
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
        attestation_secret=environ["AVALAN_PATCH_SANDBOX_TCP_SECRET"],
    )


_CONFIGURATION = _configuration()
_STORE = _CONFIGURATION.store
assert isinstance(_STORE, _CountingStore)
app = FastAPI()
_CONTROLLER = install_remote_patch_test_routes(app, _CONFIGURATION)


@app.on_event("shutdown")
async def _close_controller() -> None:
    """Close the selected runtime when the isolated child stops."""
    await _CONTROLLER.close()


@app.get("/__avalan_test__/patch/v1/ready")
async def ready(
    x_avalan_test_attestation: str | None = Header(default=None),
) -> dict[str, int | bool]:
    """Return attested durable cardinalities for the public TCP test."""
    if x_avalan_test_attestation != environ["AVALAN_PATCH_SANDBOX_TCP_SECRET"]:
        raise HTTPException(status_code=404)
    return {
        "ready": True,
        "commit_claims": _STORE.commit_claims,
        "worker_bindings": _STORE.worker_bindings,
        "settlements": _STORE.settlements,
    }
