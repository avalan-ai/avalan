"""Exercise dormant policy, sealing, review, and approval boundaries."""

from asyncio import CancelledError, Event, create_task, gather, run
from dataclasses import replace
from json import dumps
from pathlib import Path
from typing import Callable

import pytest

from avalan.patch.coordinator import CommitLease
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ByteSize,
    Capability,
    ContextKind,
    DurationTicks,
    ExpiryTick,
    FileMode,
    LogicalPath,
    MetadataProfile,
    OperationType,
    PatchApprovalId,
    PatchContextId,
    PatchDomainId,
    PatchExecutionId,
    PatchFingerprint,
    PatchInput,
    PatchLimits,
    PatchLineageId,
    PatchPlanId,
    PatchProtocolId,
    PatchRequest,
    PatchRequestId,
    PatchTargetId,
    PatchWorkspaceId,
    ProposedBytes,
    SourceBytes,
)
from avalan.patch.parser import (
    CanonicalPatchRequest,
    PatchInputLimits,
    PatchRequestParser,
    RawPatchIngress,
    RawPatchInputKind,
    RawPatchInputState,
    RawProviderProfile,
    RawToolCallId,
)
from avalan.patch.planner import (
    LogicalText,
    PlannerCandidate,
    PlannerFile,
    PlannerParentMount,
    PlannerWorkspace,
    plan,
)
from avalan.patch.policy import (
    ApprovalClock,
    ApprovalDecisionState,
    ApprovalRequirements,
    ApprovalResult,
    ApprovalService,
    BrokerDecision,
    CapabilityMode,
    CompleteReviewArtifact,
    DiagnosticPolicyId,
    EffectiveLimits,
    ExecutionSubject,
    PatchAgentId,
    PatchPrincipalId,
    PatchRunId,
    PatchSessionId,
    PatchTaskId,
    PatchTenantId,
    Phase5ControlIngress,
    Phase5IngressBoundary,
    Phase5IngressSurface,
    PlanApprovalBroker,
    PlanBinding,
    PlanReviewRequest,
    PolicyAuthorizer,
    PolicyBrokerId,
    PolicyDisclosure,
    PolicyError,
    PolicyErrorCode,
    PolicyPathSelector,
    PolicyReviewerRole,
    PolicyRevision,
    PolicyRouteId,
    PolicyRule,
    PreauthorizationClass,
    PreflightAuthorization,
    PreflightRequest,
    PrivateArtifactRetention,
    ReviewerDecision,
    RuntimeGrantStore,
    RuntimePlanStore,
    SealedPlan,
    TrustedPatchPolicy,
    _integer,
    _most_restrictive_mode,
    _optional_metadata,
    cleanup_sealed_authorities,
    compose_limits,
    project_denial,
    project_model,
    project_reviewer,
    project_sdk_host,
    seal_plan,
)
from avalan.patch.target import (
    PrimitiveProbe,
    ProbeState,
    TargetHandshake,
    TargetIdentity,
    TargetPrimitive,
)

_SUPPORTED_TEST_ATOMICITY_CLASSES = frozenset(
    (
        "single_step",
        "dependency_ordered",
    )
)


def _limits() -> PatchLimits:
    """Return finite equal limits for each trusted policy boundary."""
    return PatchLimits(
        input_bytes=ByteSize(10_000),
        path_count=ByteSize(20),
        path_length=ByteSize(512),
        file_count=ByteSize(20),
        operation_count=ByteSize(20),
        snapshot_bytes=ByteSize(10_000),
        proposed_bytes=ByteSize(10_000),
        review_diff_bytes=ByteSize(10_000),
        planning_duration=DurationTicks(100),
        approval_duration=DurationTicks(100),
        commit_duration=DurationTicks(100),
    )


def _effective_limits() -> EffectiveLimits:
    """Return the exact strictest limit projection for test authority flows."""
    value = _limits()
    return compose_limits(value, value, value, value, value)


def _candidate() -> PlannerCandidate:
    """Return a fully planned update candidate with one exact match region."""
    text = LogicalText.from_bytes(b"before\n")
    source = PlannerFile(
        path=LogicalPath("note.txt"),
        bytes_value=SourceBytes(b"before\n"),
        metadata=MetadataProfile(FileMode(0o644), text.has_bom, "lf"),
        parent=None,
        mount_id="mount-a",
        identity="identity-note",
    )
    return plan(_edit_request(), PlannerWorkspace((source,), frozenset()))


def _edit_request(path: str = "note.txt") -> CanonicalPatchRequest:
    """Parse one fixed structured edit request for rooted-snapshot tests."""
    return PatchRequestParser(PatchInputLimits()).parse(
        RawPatchIngress(
            RawProviderProfile("phase5-provider"),
            RawToolCallId("phase5-call"),
            RawPatchInputKind.EDIT_JSON,
            RawPatchInputState.COMPLETE,
            b'{"path":"'
            + path.encode()
            + b'","edits":[{"old_text":"before\\n","new_text":"after\\n"}]}',
        )
    )


def _apply_request(*lines: str) -> CanonicalPatchRequest:
    """Parse one canonical apply document for planner-derived evidence."""
    return PatchRequestParser(PatchInputLimits()).parse(
        RawPatchIngress(
            RawProviderProfile("phase5-provider"),
            RawToolCallId("phase5-apply"),
            RawPatchInputKind.APPLY_JSON,
            RawPatchInputState.COMPLETE,
            dumps(
                {"patch": "\n".join(lines) + "\n"}, separators=(",", ":")
            ).encode(),
        )
    )


def _planner_file(path: str, value: bytes) -> PlannerFile:
    """Return a rooted-snapshot-shaped planner file without a target write."""
    text = LogicalText.from_bytes(value)
    return PlannerFile(
        LogicalPath(path),
        SourceBytes(value),
        MetadataProfile(FileMode(0o644), text.has_bom, "lf"),
        LogicalPath(path.rsplit("/", 1)[0]) if "/" in path else None,
        "mount-a",
        "identity-" + path,
    )


class _RootedTargetSpy:
    """Read only preauthorized relative files from one local test root."""

    def __init__(self, root: Path) -> None:
        """Bind the test spy to one rooted temporary directory."""
        self.root = root
        self.calls: list[tuple[LogicalPath, ...]] = []

    async def snapshot(
        self, paths: tuple[LogicalPath, ...]
    ) -> PlannerWorkspace:
        """Capture declared local paths after authorization without mutation.

        Returns:
            Return a rooted planner workspace without changing the target.
        """
        self.calls.append(paths)
        files = tuple(
            _planner_file(path.value, (self.root / path.value).read_bytes())
            for path in paths
            if (self.root / path.value).is_file()
        )
        return PlannerWorkspace(
            files,
            frozenset(),
            (PlannerParentMount(None, "mount-a"),),
        )


async def _preflight_then_snapshot(
    authorizer: PolicyAuthorizer,
    request: PreflightRequest,
    target: _RootedTargetSpy,
) -> tuple[PreflightAuthorization, PlannerWorkspace]:
    """Authorize all observations before invoking the rooted target spy."""
    preflight = await authorizer.authorize_preinspection(request)
    return preflight, await target.snapshot(preflight.paths)


def _policy(
    effect_mode: ApprovalMode = ApprovalMode.REQUIRE_REVIEW,
    *,
    disclosures: frozenset[PolicyDisclosure] = frozenset(),
) -> TrustedPatchPolicy:
    """Return a trusted ordinary-path policy with bounded reader authority."""
    reader = PreauthorizationClass("trusted-reader")
    effect = (
        CapabilityMode(Capability.UPDATE, effect_mode, reader)
        if effect_mode is ApprovalMode.PREAUTHORIZED
        else CapabilityMode(Capability.UPDATE, effect_mode)
    )
    return TrustedPatchPolicy(
        revision=PolicyRevision("policy-five"),
        enabled_operations=frozenset((OperationType.EDIT,)),
        rules=(
            PolicyRule(
                selector=PolicyPathSelector(None),
                modes=(
                    effect,
                    CapabilityMode(
                        Capability.READ_FOR_MUTATION,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    ),
                    CapabilityMode(
                        Capability.OBSERVE_MUTATION_PRECONDITIONS,
                        ApprovalMode.PREAUTHORIZED,
                        reader,
                    ),
                ),
                disclosures=disclosures,
                atomicity_classes=_SUPPORTED_TEST_ATOMICITY_CLASSES,
            ),
        ),
        limits=_limits(),
        approval=_requirements(effect_mode),
    )


def _policy_for(
    capabilities: frozenset[Capability],
    *,
    selector: PolicyPathSelector | None = None,
    mode: ApprovalMode = ApprovalMode.REQUIRE_REVIEW,
    approval: ApprovalRequirements | None = None,
    disclosures: frozenset[PolicyDisclosure] = frozenset(),
) -> TrustedPatchPolicy:
    """Return a policy whose effects and approval are runtime-owned inputs."""
    reader = PreauthorizationClass("trusted-reader")
    modes = tuple(
        CapabilityMode(
            item,
            mode,
            reader if mode is ApprovalMode.PREAUTHORIZED else None,
        )
        for item in sorted(capabilities, key=lambda item: item.value)
    )
    inspection = tuple(
        CapabilityMode(
            item,
            ApprovalMode.PREAUTHORIZED,
            reader,
        )
        for item in (
            Capability.READ_FOR_MUTATION,
            Capability.OBSERVE_MUTATION_PRECONDITIONS,
        )
        if item not in capabilities
    )
    selected = approval or _requirements(mode)
    return TrustedPatchPolicy(
        revision=PolicyRevision("policy-five"),
        enabled_operations=frozenset((OperationType.EDIT,)),
        rules=(
            PolicyRule(
                selector or PolicyPathSelector(None),
                modes + inspection,
                disclosures,
                atomicity_classes=_SUPPORTED_TEST_ATOMICITY_CLASSES,
            ),
        ),
        limits=_limits(),
        approval=selected,
    )


def _candidate_with(
    capabilities: frozenset[Capability],
    *,
    source: LogicalPath | None = LogicalPath("note.txt"),
    destination: LogicalPath | None = LogicalPath("note.txt"),
) -> PlannerCandidate:
    """Return one canonical candidate with test-selected final capabilities."""
    candidate = _candidate()
    final_path = destination or source
    assert final_path is not None
    lineage = replace(
        candidate.lineages[0],
        initial=replace(
            candidate.lineages[0].initial,
            path=source or final_path,
        ),
        final=replace(candidate.lineages[0].final, path=final_path),
        source_path=source,
        destination_path=destination,
        capabilities=capabilities,
    )
    return replace(
        candidate,
        lineages=(lineage,),
        final_files=(replace(candidate.final_files[0], path=final_path),),
    )


def _subject() -> ExecutionSubject:
    """Return fixed authenticated identities for exact grant bindings."""
    return ExecutionSubject(
        principal=PatchPrincipalId("principal-a"),
        tenant=PatchTenantId("tenant-a"),
        run=PatchRunId("run-a"),
        session=PatchSessionId("session-a"),
        task=PatchTaskId("task-a"),
        agent=PatchAgentId("agent-a"),
    )


def _target() -> TargetIdentity:
    """Return a trusted target identity without opening a target resource."""
    return TargetIdentity(
        context_id=PatchContextId("context_" + "a" * 16),
        workspace_id=PatchWorkspaceId("workspace_" + "a" * 16),
        domain_id=PatchDomainId("domain_" + "a" * 16),
        target_id=PatchTargetId("target_" + "a" * 16),
        protocol_id=PatchProtocolId("protocol_" + "a" * 16),
        filesystem_id="filesystem-a",
        mount_id="mount-a",
        policy_revision="policy-five",
        persistent_lease_id="persistent-lease-a",
        approval_channel_id=PatchApprovalId("approval_" + "a" * 16),
    )


def _requirements(mode: ApprovalMode) -> ApprovalRequirements:
    """Return exact bounded review requirements for the selected mode."""
    return ApprovalRequirements(
        mode=mode,
        route=PolicyRouteId("route-five"),
        broker=PolicyBrokerId("broker-five"),
        reviewer_role=PolicyReviewerRole("reviewer-five"),
        quorum=1,
        preauthorization=(
            PreauthorizationClass("trusted-reader")
            if mode is ApprovalMode.PREAUTHORIZED
            else None
        ),
    )


def _handshake(
    identity: TargetIdentity | None = None,
    *,
    effectful: bool = True,
) -> TargetHandshake:
    """Return a capability witness derived from one target handshake."""
    primitives = frozenset(
        (
            TargetPrimitive.ROOTED_CONTAINMENT,
            TargetPrimitive.NOFOLLOW_INSPECTION,
            TargetPrimitive.REGULAR_FILE_IDENTITY,
            TargetPrimitive.BOUNDED_READ,
        )
    )
    future = (
        TargetPrimitive.METADATA_PRESERVATION,
        TargetPrimitive.BOUNDED_WRITE,
        TargetPrimitive.REPLACE_PUBLICATION,
        TargetPrimitive.NOREPLACE_CREATE_MOVE,
        TargetPrimitive.DIRECTORY_ENTRY_DELETE,
        TargetPrimitive.SAME_FILESYSTEM_MOVE,
        TargetPrimitive.STAGING,
        TargetPrimitive.STRUCTURAL_VERIFICATION,
    )
    return TargetHandshake(
        identity or _target(),
        primitives,
        (),
        tuple(
            PrimitiveProbe(
                item,
                ProbeState.AVAILABLE if effectful else ProbeState.UNAVAILABLE,
            )
            for item in future
        ),
    )


async def _sealed_plan(
    mode: ApprovalMode = ApprovalMode.REQUIRE_REVIEW,
    *,
    disclosures: frozenset[PolicyDisclosure] = frozenset(),
    approval: ApprovalRequirements | None = None,
    external_effects: frozenset[Capability] | None = None,
) -> SealedPlan:
    """Run all non-effectful Phase 5 authority stages through sealing."""
    policy = _policy(mode, disclosures=disclosures)
    if approval is not None:
        policy = replace(policy, approval=approval)
    authorizer = PolicyAuthorizer(policy)
    effective = _effective_limits()
    preflight = await authorizer.authorize_preinspection(
        PreflightRequest(
            operation=OperationType.EDIT,
            paths=(LogicalPath("note.txt"),),
            external_effects=external_effects
            or frozenset((Capability.UPDATE,)),
            external_read_paths=frozenset((LogicalPath("note.txt"),)),
            effective_limits=effective,
        )
    )
    candidate = _candidate()
    final = await authorizer.authorize_final(
        preflight,
        candidate,
        _handshake(),
    )
    request = PatchRequest(
        schema_version=1,
        request_id=PatchRequestId("request_" + "a" * 16),
        execution_id=PatchExecutionId("execution_" + "a" * 16),
        operation=OperationType.EDIT,
        input_bytes=PatchInput(b"canonical-request"),
        logical_paths=(LogicalPath("note.txt"),),
    )
    return seal_plan(
        PatchPlanId("plan_" + "a" * 16),
        PlanBinding(
            request=request,
            request_digest=candidate.request_digest,
            subject=_subject(),
            context_kind=ContextKind.LOCAL,
            target=_target(),
            cwd=None,
            preflight=preflight,
            final=final,
        ),
        candidate,
        ExpiryTick(100),
    )


class _Clock(ApprovalClock):
    """Provide a deterministic asynchronous clock for approval tests."""

    def __init__(self, tick: int) -> None:
        """Initialize one current monotonic test tick."""
        self.tick = tick

    async def now(self) -> ExpiryTick:
        """Return the current test tick through the async clock boundary."""
        return ExpiryTick(self.tick)


class _Broker(PlanApprovalBroker):
    """Return a fixed typed reviewer decision without generic confirmation."""

    def __init__(
        self,
        state: ApprovalDecisionState,
        broker: PolicyBrokerId | None = None,
        reviewer_role: PolicyReviewerRole | None = None,
        reviewers: tuple[PatchPrincipalId, ...] = (
            PatchPrincipalId("reviewer-a"),
        ),
    ) -> None:
        """Initialize the one selected broker response state."""
        self.state = state
        self.broker = broker
        self.reviewer_role = reviewer_role
        self.reviewers = reviewers

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Return a review-bound decision using no mutable plan input."""
        return BrokerDecision(
            self.broker or request.requirements.broker,
            tuple(
                ReviewerDecision(
                    item,
                    request.subject.tenant,
                    self.reviewer_role or request.requirements.reviewer_role,
                    self.state,
                )
                for item in self.reviewers
            ),
        )


class _UnavailableBroker(PlanApprovalBroker):
    """Raise a typed availability failure from the asynchronous boundary."""

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Report a typed unavailable response without an approval decision."""
        raise PolicyError(PolicyErrorCode.APPROVAL_UNAVAILABLE)


class _CancelledBroker(PlanApprovalBroker):
    """Propagate the owning task cancellation without granting authority."""

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Cancel the awaiting interaction before it can issue a grant."""
        raise CancelledError


class _SequenceClock(ApprovalClock):
    """Return a fixed sequence of monotonic approval ticks."""

    def __init__(self, ticks: tuple[int, ...]) -> None:
        """Initialize finite monotonic ticks for one test flow."""
        self.ticks = ticks
        self.index = 0

    async def now(self) -> ExpiryTick:
        """Return the next trusted monotonic tick."""
        value = self.ticks[self.index]
        self.index += 1
        return ExpiryTick(value)


class _SuspendedBroker(PlanApprovalBroker):
    """Suspend one typed review until its explicit continuation is released."""

    def __init__(self) -> None:
        """Initialize deterministic review-start and continuation events."""
        self.started = Event()
        self.continue_review = Event()

    async def decide(self, request: PlanReviewRequest) -> BrokerDecision:
        """Await continuation before returning the selected broker result."""
        self.started.set()
        await self.continue_review.wait()
        return BrokerDecision(
            request.requirements.broker,
            (
                ReviewerDecision(
                    PatchPrincipalId("reviewer-suspended"),
                    request.subject.tenant,
                    request.requirements.reviewer_role,
                    ApprovalDecisionState.APPROVED,
                ),
            ),
        )


def test_patch_phase_5_requirements() -> None:
    """Seal a complete review and validate one opaque approval grant."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.path_count == ByteSize(20)

    async def execute() -> None:
        plan_value = await _sealed_plan(
            disclosures=frozenset(
                (
                    PolicyDisclosure.COMPLETE_REVIEW,
                    PolicyDisclosure.MODEL_DIFF,
                    PolicyDisclosure.MODEL_METADATA,
                    PolicyDisclosure.MODEL_MATCH_DETAILS,
                    PolicyDisclosure.AUDIT_PATHS,
                )
            )
        )
        sealed = plan_value
        review = project_reviewer(sealed)
        assert isinstance(review.artifact, CompleteReviewArtifact)
        assert review.artifact.lineages[0].regions[0].logical_start == 0
        assert review.artifact.diff.diff._value.endswith(b"+after\n")
        assert review.artifact.fingerprint == sealed.fingerprint
        model = project_model(sealed)
        assert model.diff is not None
        assert model.diff._value.endswith(b"+after\n")
        assert model.hashes_and_sizes and model.detailed_matches
        store = RuntimePlanStore()
        summary = await store.put(sealed)
        assert await store.get(sealed.plan_id) == sealed
        assert project_sdk_host(summary).observer_id == summary.observer_id
        grants = RuntimeGrantStore()
        service = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED), _Clock(1), grants
        )
        result = await service.await_review(
            PlanReviewRequest(
                sealed,
                _subject(),
                _requirements(ApprovalMode.REQUIRE_REVIEW),
            )
        )
        assert result.grant is not None
        await service.validate_grant(result.grant, sealed, _subject())

    run(execute())


def test_patch_phase_5_rooted_approval_e2es_without_commit(
    tmp_path: Path,
) -> None:
    """Exercise rooted snapshots, approval outcomes, and unconsumed grants."""
    note = tmp_path / "note.txt"
    note.write_bytes(b"before\n")

    async def execute() -> None:
        policy = _policy()
        authorizer = PolicyAuthorizer(policy)
        target = _RootedTargetSpy(tmp_path)
        request = PreflightRequest(
            OperationType.EDIT,
            (LogicalPath("note.txt"),),
            frozenset((Capability.UPDATE,)),
            frozenset((LogicalPath("note.txt"),)),
            _effective_limits(),
        )
        preflight, workspace = await _preflight_then_snapshot(
            authorizer, request, target
        )
        candidate = plan(_edit_request(), workspace)
        final = await authorizer.authorize_final(
            preflight, candidate, _handshake()
        )
        patch_request = PatchRequest(
            1,
            PatchRequestId("request_" + "b" * 16),
            PatchExecutionId("execution_" + "b" * 16),
            OperationType.EDIT,
            PatchInput(b"rooted-request"),
            (LogicalPath("note.txt"),),
        )
        plan_value = seal_plan(
            PatchPlanId("plan_" + "b" * 16),
            PlanBinding(
                patch_request,
                candidate.request_digest,
                _subject(),
                ContextKind.LOCAL,
                _target(),
                None,
                preflight,
                final,
            ),
            candidate,
            ExpiryTick(100),
        )
        review = PlanReviewRequest(
            plan_value,
            _subject(),
            plan_value.binding.final.approval,
        )
        grants = RuntimeGrantStore()
        service = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED), _Clock(1), grants
        )
        approved = await service.await_review(review)
        assert approved.grant is not None
        grant = approved.grant
        await gather(
            service.validate_grant(grant, plan_value, _subject()),
            service.validate_grant(grant, plan_value, _subject()),
        )
        assert await grants.get(grant.grant_id) == grant
        assert not hasattr(grants, "consume")
        note.write_bytes(b"workspace-changed-after-review\n")
        await service.validate_grant(grant, plan_value, _subject())
        assert target.calls == [(LogicalPath("note.txt"),)]
        for broker, clock, expected in (
            (
                _Broker(ApprovalDecisionState.DENIED),
                _Clock(1),
                ApprovalDecisionState.DENIED,
            ),
            (
                _UnavailableBroker(),
                _Clock(1),
                ApprovalDecisionState.UNAVAILABLE,
            ),
            (
                _Broker(ApprovalDecisionState.APPROVED),
                _Clock(100),
                ApprovalDecisionState.DENIED,
            ),
        ):
            assert (
                await ApprovalService(
                    broker, clock, RuntimeGrantStore()
                ).await_review(review)
            ).state is expected
        with pytest.raises(CancelledError):
            await ApprovalService(
                _CancelledBroker(), _Clock(1), RuntimeGrantStore()
            ).await_review(review)
        assert target.calls == [(LogicalPath("note.txt"),)]

    run(execute())


def test_patch_phase_5_default_denial_and_preinspection_are_closed() -> None:
    """Deny defaults and protected paths before a target can be consulted."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.path_count == ByteSize(20)

    async def execute() -> None:
        effective = _effective_limits()
        request = PreflightRequest(
            OperationType.EDIT,
            (LogicalPath("note.txt"),),
            frozenset((Capability.UPDATE,)),
            frozenset((LogicalPath("note.txt"),)),
            effective,
        )
        with pytest.raises(PolicyError) as empty:
            await PolicyAuthorizer(
                TrustedPatchPolicy.empty()
            ).authorize_preinspection(request)
        assert empty.value.code is PolicyErrorCode.DENIED
        with pytest.raises(PolicyError) as vcs:
            await PolicyAuthorizer(_policy()).authorize_preinspection(
                replace(
                    request,
                    paths=(LogicalPath(".git/config"),),
                    external_read_paths=frozenset(
                        (LogicalPath(".git/config"),)
                    ),
                )
            )
        assert vcs.value.code is PolicyErrorCode.PATH_DENIED
        with pytest.raises(PolicyError) as hidden:
            await PolicyAuthorizer(_policy()).authorize_preinspection(
                replace(
                    request,
                    paths=(LogicalPath(".env"),),
                    external_read_paths=frozenset((LogicalPath(".env"),)),
                )
            )
        assert hidden.value.code is PolicyErrorCode.PATH_DENIED

    run(execute())


def test_patch_phase_5_fingerprint_binds_plan_inputs_not_ephemeral_lease() -> (
    None
):
    """Bind persistent identity while excluding ephemeral commit leases."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.file_count == ByteSize(20)

    async def execute() -> None:
        sealed = await _sealed_plan()
        changed_subject = replace(
            sealed.binding,
            subject=replace(
                sealed.binding.subject,
                principal=PatchPrincipalId("principal-b"),
            ),
        )
        changed = seal_plan(
            sealed.plan_id,
            changed_subject,
            sealed.candidate,
            sealed.review.expiry,
        )
        changed_target = replace(
            sealed.binding,
            target=replace(
                sealed.binding.target,
                persistent_lease_id="persistent-lease-b",
            ),
        )
        persistent_lease_change = seal_plan(
            sealed.plan_id,
            changed_target,
            sealed.candidate,
            sealed.review.expiry,
        )
        commit_lease = CommitLease(
            sealed.binding.target.domain_id,
            sealed.binding.request.request_id,
            1,
        )
        replacement_commit_lease = replace(commit_lease, fence=2)
        same_plan = seal_plan(
            sealed.plan_id,
            sealed.binding,
            sealed.candidate,
            sealed.review.expiry,
        )
        assert changed.fingerprint != sealed.fingerprint
        assert persistent_lease_change.fingerprint != sealed.fingerprint
        assert replacement_commit_lease != commit_lease
        assert same_plan.fingerprint == sealed.fingerprint

    run(execute())


def test_patch_phase_5_disclosure_and_wrong_grant_remain_private() -> None:
    """Withhold model/reviewer details and reject cross-plan grants."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.operation_count == ByteSize(20)

    async def execute() -> None:
        sealed = await _sealed_plan()
        model = project_model(sealed)
        assert model.paths == ()
        assert model.diff is None
        assert not model.hashes_and_sizes and not model.detailed_matches
        assert b"after" not in repr(model).encode()
        with pytest.raises(PolicyError) as reviewer:
            project_reviewer(sealed)
        assert reviewer.value.code is PolicyErrorCode.PATH_DENIED
        grants = RuntimeGrantStore()
        service = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED), _Clock(1), grants
        )
        result = await service.await_review(
            PlanReviewRequest(
                sealed,
                _subject(),
                _requirements(ApprovalMode.REQUIRE_REVIEW),
            )
        )
        assert result.grant is not None
        other = seal_plan(
            PatchPlanId("plan_" + "b" * 16),
            sealed.binding,
            sealed.candidate,
            sealed.review.expiry,
        )
        with pytest.raises(PolicyError) as mismatch:
            await service.validate_grant(result.grant, other, _subject())
        assert mismatch.value.code is PolicyErrorCode.APPROVAL_MISMATCH

    run(execute())


def test_patch_phase_5_preauthorization_still_requires_every_gate() -> None:
    """Seal a preauthorized plan only after final target and policy checks."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.snapshot_bytes == ByteSize(10_000)

    async def execute() -> None:
        sealed = await _sealed_plan(ApprovalMode.PREAUTHORIZED)
        assert sealed.binding.final.approval.mode is ApprovalMode.PREAUTHORIZED
        denied = PolicyAuthorizer(_policy(ApprovalMode.PREAUTHORIZED))
        preflight = await denied.authorize_preinspection(
            PreflightRequest(
                OperationType.EDIT,
                (LogicalPath("note.txt"),),
                frozenset((Capability.UPDATE,)),
                frozenset((LogicalPath("note.txt"),)),
                _effective_limits(),
            )
        )
        with pytest.raises(PolicyError) as target:
            await denied.authorize_final(
                preflight,
                sealed.candidate,
                _handshake(effectful=False),
            )
        assert target.value.code is PolicyErrorCode.DENIED

    run(execute())


def test_patch_phase_5_capability_table_evidence() -> None:
    """Retain executable ownership for capability-table evidence."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.proposed_bytes == ByteSize(10_000)

    with pytest.raises(PolicyError):
        PolicyRevision("")
    with pytest.raises(PolicyError):
        PolicyRouteId("")
    with pytest.raises(PolicyError):
        PolicyBrokerId("")
    with pytest.raises(PolicyError):
        PolicyReviewerRole("")
    with pytest.raises(PolicyError):
        PreauthorizationClass("")
    with pytest.raises(PolicyError):
        DiagnosticPolicyId("x" * 129)
    with pytest.raises(PolicyError):
        PatchPrincipalId("")
    with pytest.raises(PolicyError):
        PatchTenantId("")
    with pytest.raises(PolicyError):
        PatchRunId("")
    with pytest.raises(PolicyError):
        PatchSessionId("")
    with pytest.raises(PolicyError):
        PatchTaskId("")
    with pytest.raises(PolicyError):
        PatchAgentId("")
    selector = PolicyPathSelector(LogicalPath("nested"))
    assert selector.matches(LogicalPath("nested"))
    assert not selector.matches(LogicalPath(".git/config"))
    with pytest.raises(PolicyError):
        CapabilityMode(Capability.UPDATE, ApprovalMode.PREAUTHORIZED)
    with pytest.raises(PolicyError):
        CapabilityMode(
            Capability.READ_FOR_MUTATION,
            ApprovalMode.REQUIRE_REVIEW,
        )
    update = CapabilityMode(Capability.UPDATE, ApprovalMode.REQUIRE_REVIEW)
    with pytest.raises(PolicyError):
        PolicyRule(PolicyPathSelector(None), (update, update))
    assert (
        PolicyRule(PolicyPathSelector(None), ()).mode_for(Capability.UPDATE)
        is None
    )
    duplicate_rule = PolicyRule(PolicyPathSelector(LogicalPath("same")), ())
    with pytest.raises(PolicyError):
        TrustedPatchPolicy(
            PolicyRevision("duplicate-policy"),
            rules=(duplicate_rule, duplicate_rule),
        )
    with pytest.raises(PolicyError):
        PreflightRequest(
            OperationType.EDIT,
            (),
            frozenset(),
            frozenset(),
            _effective_limits(),
        )
    with pytest.raises(PolicyError):
        _requirements(ApprovalMode.REQUIRE_REVIEW).__class__(
            ApprovalMode.REQUIRE_REVIEW,
            PolicyRouteId("zero-route"),
            PolicyBrokerId("zero-broker"),
            PolicyReviewerRole("zero-reviewer"),
            0,
        )
    with pytest.raises(PolicyError):
        _most_restrictive_mode(())
    with pytest.raises(PolicyError):
        _integer(-1)
    assert _optional_metadata(None) == b""
    assert (
        project_denial(PolicyError(PolicyErrorCode.PATH_DENIED)).code
        is PolicyErrorCode.PATH_DENIED
    )
    assert (
        project_denial(PolicyError(PolicyErrorCode.APPROVAL_MISMATCH)).code
        is PolicyErrorCode.DENIED
    )


def test_patch_phase_5_final_effect_evidence() -> None:
    """Retain executable ownership for final-effect authorization evidence."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.review_diff_bytes == ByteSize(10_000)

    async def execute() -> None:
        sealed = await _sealed_plan()
        authorizer = PolicyAuthorizer(_policy())
        request = PreflightRequest(
            OperationType.EDIT,
            (LogicalPath("note.txt"),),
            frozenset((Capability.UPDATE,)),
            frozenset((LogicalPath("note.txt"),)),
            _effective_limits(),
        )
        await authorizer.authorize_preinspection(
            replace(request, external_read_paths=frozenset())
        )
        await authorizer.authorize_preinspection(
            replace(
                request,
                paths=(LogicalPath("note.txt"), LogicalPath("other.txt")),
            )
        )
        with pytest.raises(PolicyError):
            seal_plan(
                PatchPlanId("plan_" + "d" * 16),
                replace(
                    sealed.binding,
                    final=replace(
                        sealed.binding.final,
                        effective_limits=EffectiveLimits(
                            replace(
                                _limits(),
                                review_diff_bytes=ByteSize(1),
                            )
                        ),
                    ),
                ),
                sealed.candidate,
                sealed.review.expiry,
            )
        with pytest.raises(PolicyError):
            await authorizer.authorize_final(
                replace(
                    sealed.binding.preflight,
                    revision=PolicyRevision("other-policy"),
                ),
                sealed.candidate,
                _handshake(),
            )
        reader = PreauthorizationClass("trusted-reader")
        no_update = TrustedPatchPolicy(
            PolicyRevision("policy-five"),
            frozenset((OperationType.EDIT,)),
            (
                PolicyRule(
                    PolicyPathSelector(None),
                    (
                        CapabilityMode(
                            Capability.READ_FOR_MUTATION,
                            ApprovalMode.PREAUTHORIZED,
                            reader,
                        ),
                        CapabilityMode(
                            Capability.OBSERVE_MUTATION_PRECONDITIONS,
                            ApprovalMode.PREAUTHORIZED,
                            reader,
                        ),
                    ),
                ),
            ),
            _limits(),
        )
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(no_update).authorize_preinspection(request)
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(no_update).authorize_final(
                sealed.binding.preflight,
                sealed.candidate,
                _handshake(),
            )
        denying = TrustedPatchPolicy(
            PolicyRevision("policy-five"),
            frozenset((OperationType.EDIT,)),
            (
                PolicyRule(
                    PolicyPathSelector(None),
                    (
                        CapabilityMode(Capability.UPDATE, ApprovalMode.DENY),
                        CapabilityMode(
                            Capability.READ_FOR_MUTATION,
                            ApprovalMode.PREAUTHORIZED,
                            reader,
                        ),
                        CapabilityMode(
                            Capability.OBSERVE_MUTATION_PRECONDITIONS,
                            ApprovalMode.PREAUTHORIZED,
                            reader,
                        ),
                    ),
                ),
            ),
            _limits(),
        )
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(denying).authorize_final(
                sealed.binding.preflight,
                sealed.candidate,
                _handshake(),
            )
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(
                replace(
                    _policy(),
                    approval=_requirements(ApprovalMode.PREAUTHORIZED),
                )
            ).authorize_final(
                sealed.binding.preflight,
                sealed.candidate,
                _handshake(),
            )
        tampered = _policy()
        object.__setattr__(
            tampered.rules[0].modes[1], "mode", ApprovalMode.REQUIRE_REVIEW
        )
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(tampered).authorize_preinspection(request)
        lineage = sealed.candidate.lineages[0]
        assert lineage.final.metadata is not None
        executable = replace(
            sealed.candidate,
            lineages=(
                replace(
                    lineage,
                    final=replace(
                        lineage.final,
                        metadata=replace(
                            lineage.final.metadata,
                            mode=FileMode(0o755),
                        ),
                    ),
                ),
            ),
        )
        with pytest.raises(PolicyError):
            await authorizer.authorize_final(
                sealed.binding.preflight,
                executable,
                _handshake(),
            )
        executable_policy = TrustedPatchPolicy(
            PolicyRevision("policy-five"),
            frozenset((OperationType.EDIT,)),
            (
                PolicyRule(
                    PolicyPathSelector(None),
                    _policy().rules[0].modes
                    + (
                        CapabilityMode(
                            Capability.UPDATE_EXECUTABLE,
                            ApprovalMode.REQUIRE_REVIEW,
                        ),
                    ),
                ),
            ),
            _limits(),
            _requirements(ApprovalMode.REQUIRE_REVIEW),
        )
        executable_authorizer = PolicyAuthorizer(executable_policy)
        executable_preflight = (
            await executable_authorizer.authorize_preinspection(
                replace(
                    request,
                    external_effects=frozenset(
                        (
                            Capability.UPDATE,
                            Capability.UPDATE_EXECUTABLE,
                        )
                    ),
                )
            )
        )
        with pytest.raises(PolicyError):
            await executable_authorizer.authorize_final(
                sealed.binding.preflight,
                executable,
                _handshake(),
            )
        final = await executable_authorizer.authorize_final(
            executable_preflight,
            executable,
            _handshake(),
        )
        assert Capability.UPDATE_EXECUTABLE in final.effects

    run(execute())


def test_patch_phase_5_review_artifact_evidence() -> None:
    """Retain executable ownership for detached review-artifact evidence."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.planning_duration == DurationTicks(100)

    async def execute() -> None:
        sealed = await _sealed_plan()
        with pytest.raises(PolicyError):
            replace(
                sealed.binding,
                target=replace(
                    sealed.binding.target,
                    policy_revision="wrong-policy",
                ),
            )
        with pytest.raises(PolicyError):
            replace(sealed.review, lineages=())
        with pytest.raises(PolicyError):
            replace(sealed, fingerprint=PatchFingerprint(b"x" * 32))
        with pytest.raises(PolicyError):
            PlanReviewRequest(
                sealed,
                replace(
                    _subject(),
                    principal=PatchPrincipalId("principal-other"),
                ),
                _requirements(ApprovalMode.REQUIRE_REVIEW),
            )
        reviewer = ReviewerDecision(
            PatchPrincipalId("reviewer-duplicate"),
            _subject().tenant,
            PolicyReviewerRole("reviewer-five"),
            ApprovalDecisionState.APPROVED,
        )
        with pytest.raises(PolicyError):
            BrokerDecision(PolicyBrokerId("broker-five"), ())
        with pytest.raises(PolicyError):
            BrokerDecision(
                PolicyBrokerId("broker-five"),
                (reviewer, reviewer),
            )

    run(execute())


def test_patch_phase_5_broker_evidence() -> None:
    """Retain executable ownership for typed broker approval evidence."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.approval_duration == DurationTicks(100)

    async def execute() -> None:
        sealed = await _sealed_plan()
        request = PlanReviewRequest(
            sealed,
            _subject(),
            _requirements(ApprovalMode.REQUIRE_REVIEW),
        )
        unavailable = ApprovalService(
            _UnavailableBroker(), _Clock(1), RuntimeGrantStore()
        )
        assert (
            await unavailable.await_review(request)
        ).state is ApprovalDecisionState.UNAVAILABLE
        wrong_broker = ApprovalService(
            _Broker(
                ApprovalDecisionState.APPROVED,
                PolicyBrokerId("broker-other"),
            ),
            _Clock(1),
            RuntimeGrantStore(),
        )
        assert (
            await wrong_broker.await_review(request)
        ).state is ApprovalDecisionState.DENIED
        denied = ApprovalService(
            _Broker(ApprovalDecisionState.DENIED),
            _Clock(1),
            RuntimeGrantStore(),
        )
        assert (
            await denied.await_review(request)
        ).state is ApprovalDecisionState.DENIED
        broker_unavailable = ApprovalService(
            _Broker(ApprovalDecisionState.UNAVAILABLE),
            _Clock(1),
            RuntimeGrantStore(),
        )
        assert (
            await broker_unavailable.await_review(request)
        ).state is ApprovalDecisionState.UNAVAILABLE
        quorum_two = replace(
            sealed.binding,
            final=replace(
                sealed.binding.final,
                approval=replace(
                    sealed.binding.final.approval,
                    quorum=2,
                ),
            ),
        )
        quorum_plan = seal_plan(
            PatchPlanId("plan_" + "c" * 16),
            quorum_two,
            sealed.candidate,
            sealed.review.expiry,
        )
        insufficient = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED),
            _Clock(1),
            RuntimeGrantStore(),
        )
        assert (
            await insufficient.await_review(
                PlanReviewRequest(
                    quorum_plan,
                    _subject(),
                    quorum_plan.binding.final.approval,
                )
            )
        ).state is ApprovalDecisionState.DENIED
        already_expired = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED),
            _Clock(100),
            RuntimeGrantStore(),
        )
        assert (
            await already_expired.await_review(request)
        ).state is ApprovalDecisionState.DENIED
        expires_after_decision = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED),
            _SequenceClock((1, 100)),
            RuntimeGrantStore(),
        )
        assert (
            await expires_after_decision.await_review(request)
        ).state is ApprovalDecisionState.DENIED

    run(execute())


def test_patch_phase_5_grant_evidence() -> None:
    """Retain executable ownership for opaque plan-bound grant evidence."""
    assert isinstance(PolicyAuthorizer(_policy()), PolicyAuthorizer)
    assert _effective_limits().value.commit_duration == DurationTicks(100)

    async def execute() -> None:
        sealed = await _sealed_plan()
        request = PlanReviewRequest(
            sealed,
            _subject(),
            _requirements(ApprovalMode.REQUIRE_REVIEW),
        )
        grants = RuntimeGrantStore()
        clock = _Clock(1)
        service = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED), clock, grants
        )
        result = await service.await_review(request)
        assert result.grant is not None
        grant = result.grant
        with pytest.raises(PolicyError):
            replace(grant, reviewers=())
        with pytest.raises(PolicyError):
            ApprovalResult(ApprovalDecisionState.APPROVED)
        with pytest.raises(PolicyError):
            ApprovalResult(ApprovalDecisionState.DENIED, grant)
        conflicting = replace(grant, approval_id=PatchApprovalId.new())
        with pytest.raises(PolicyError):
            await grants.put(conflicting)
        clock.tick = 100
        with pytest.raises(PolicyError) as expired:
            await service.validate_grant(grant, sealed, _subject())
        assert expired.value.code is PolicyErrorCode.APPROVAL_EXPIRED
        plans = RuntimePlanStore()
        await plans.put(sealed)
        different = seal_plan(
            sealed.plan_id,
            replace(sealed.binding, cwd=LogicalPath("other")),
            sealed.candidate,
            sealed.review.expiry,
        )
        with pytest.raises(PolicyError):
            await plans.put(different)

    run(execute())


def test_patch_phase_5_policy_owned_authority_and_seal_integrity() -> None:
    """Reject caller-controlled authority and changed sealed plan contents."""

    async def execute() -> None:
        candidate = _candidate()
        policy = _policy()
        authorizer = PolicyAuthorizer(policy)
        preflight = await authorizer.authorize_preinspection(
            PreflightRequest(
                OperationType.EDIT,
                (LogicalPath("note.txt"),),
                frozenset((Capability.UPDATE,)),
                frozenset((LogicalPath("note.txt"),)),
                _effective_limits(),
            )
        )
        final = await authorizer.authorize_final(
            preflight, candidate, _handshake()
        )
        assert final.approval == policy.approval
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(
                replace(
                    policy,
                    approval=_requirements(ApprovalMode.PREAUTHORIZED),
                )
            ).authorize_final(preflight, candidate, _handshake())
        with pytest.raises(PolicyError):
            await authorizer.authorize_final(
                preflight, candidate, _handshake(effectful=False)
            )
        without_update = await authorizer.authorize_preinspection(
            PreflightRequest(
                OperationType.EDIT,
                (LogicalPath("note.txt"),),
                frozenset(),
                frozenset(),
                _effective_limits(),
            )
        )
        with pytest.raises(PolicyError):
            await authorizer.authorize_final(
                without_update, candidate, _handshake()
            )
        with pytest.raises(PolicyError):
            await authorizer.authorize_final(
                preflight,
                replace(
                    candidate,
                    lineages=(
                        replace(
                            candidate.lineages[0],
                            atomicity_class="untrusted-atomicity",
                        ),
                    ),
                ),
                _handshake(),
            )
        sealed = await _sealed_plan()
        with pytest.raises(PolicyError):
            project_model(replace(sealed, plan_id=PatchPlanId.new()))
        object.__setattr__(
            sealed.candidate.lineages[0],
            "capabilities",
            frozenset((Capability.DELETE,)),
        )
        with pytest.raises(PolicyError):
            project_model(sealed)
        with pytest.raises(PolicyError):
            PlanReviewRequest(
                sealed,
                _subject(),
                sealed.binding.final.approval,
            )

    run(execute())


def test_patch_phase_5_capability_path_and_fingerprint_matrix() -> None:
    """Exercise policy capability, path, disclosure, and fingerprint rows."""

    async def execute() -> None:
        cases = (
            (frozenset((Capability.CREATE,)), None, LogicalPath("made.txt")),
            (
                frozenset((Capability.UPDATE,)),
                LogicalPath("note.txt"),
                LogicalPath("note.txt"),
            ),
            (
                frozenset((Capability.MOVE, Capability.UPDATE)),
                LogicalPath("note.txt"),
                LogicalPath("moved.txt"),
            ),
            (frozenset((Capability.DELETE,)), LogicalPath("note.txt"), None),
        )
        for effects, source, destination in cases:
            policy = _policy_for(effects)
            paths = tuple(
                dict.fromkeys(
                    item for item in (source, destination) if item is not None
                )
            )
            preflight = await PolicyAuthorizer(policy).authorize_preinspection(
                PreflightRequest(
                    OperationType.EDIT,
                    paths,
                    effects,
                    (
                        frozenset((source,))
                        if source is not None
                        else frozenset()
                    ),
                    _effective_limits(),
                )
            )
            final = await PolicyAuthorizer(policy).authorize_final(
                preflight,
                _candidate_with(
                    effects, source=source, destination=destination
                ),
                _handshake(),
            )
            assert final.effects == effects
        hidden_path = LogicalPath(".trusted/note.txt")
        hidden_policy = _policy_for(
            frozenset((Capability.UPDATE,)),
            selector=PolicyPathSelector(LogicalPath(".trusted"), True),
        )
        hidden_preflight = await PolicyAuthorizer(
            hidden_policy
        ).authorize_preinspection(
            PreflightRequest(
                OperationType.EDIT,
                (hidden_path,),
                frozenset((Capability.UPDATE,)),
                frozenset((hidden_path,)),
                _effective_limits(),
            )
        )
        assert (
            await PolicyAuthorizer(hidden_policy).authorize_final(
                hidden_preflight,
                _candidate_with(
                    frozenset((Capability.UPDATE,)),
                    source=hidden_path,
                    destination=hidden_path,
                ),
                _handshake(),
            )
        ).effects == frozenset((Capability.UPDATE,))
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(hidden_policy).authorize_preinspection(
                PreflightRequest(
                    OperationType.EDIT,
                    (LogicalPath(".git/config"),),
                    frozenset((Capability.UPDATE,)),
                    frozenset((LogicalPath(".git/config"),)),
                    _effective_limits(),
                )
            )
        calls = 0
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(
                TrustedPatchPolicy.empty()
            ).authorize_preinspection(
                PreflightRequest(
                    OperationType.EDIT,
                    (LogicalPath("absent.txt"),),
                    frozenset((Capability.UPDATE,)),
                    frozenset((LogicalPath("absent.txt"),)),
                    _effective_limits(),
                )
            )
        assert calls == 0
        sealed = await _sealed_plan(
            disclosures=frozenset(
                (
                    PolicyDisclosure.MODEL_DIFF,
                    PolicyDisclosure.MODEL_METADATA,
                    PolicyDisclosure.MODEL_MATCH_DETAILS,
                    PolicyDisclosure.AUDIT_PATHS,
                )
            )
        )
        assert project_model(sealed).paths == (LogicalPath("note.txt"),) * 2
        bound_changes = (
            replace(
                sealed.binding,
                subject=replace(
                    sealed.binding.subject,
                    run=PatchRunId("run-other"),
                ),
            ),
            replace(sealed.binding, cwd=LogicalPath("subdir")),
            replace(
                sealed.binding,
                diagnostic_policy=DiagnosticPolicyId("diagnostic-five"),
            ),
            replace(
                sealed.binding,
                final=replace(
                    sealed.binding.final,
                    disclosures=frozenset((PolicyDisclosure.MODEL_DIFF,)),
                ),
            ),
        )
        for binding in bound_changes:
            assert (
                seal_plan(
                    PatchPlanId.new(),
                    binding,
                    sealed.candidate,
                    sealed.review.expiry,
                ).fingerprint
                != sealed.fingerprint
            )

    run(execute())


def test_patch_phase_5_policy_table_rooted_spy_and_multilineage(
    tmp_path: Path,
) -> None:
    """Exercise paths, disclosures, modes, and target ordering together."""
    (tmp_path / "note.txt").write_bytes(b"before\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "present").write_bytes(b"canary\n")
    (tmp_path / ".trusted").mkdir()
    (tmp_path / ".trusted" / "note.txt").write_bytes(b"before\n")

    async def execute() -> None:
        policy = _policy()
        authorizer = PolicyAuthorizer(policy)
        spy = _RootedTargetSpy(tmp_path)
        allowed = PreflightRequest(
            OperationType.EDIT,
            (LogicalPath("note.txt"),),
            frozenset((Capability.UPDATE,)),
            frozenset((LogicalPath("note.txt"),)),
            _effective_limits(),
        )
        preflight, workspace = await _preflight_then_snapshot(
            authorizer, allowed, spy
        )
        assert spy.calls == [(LogicalPath("note.txt"),)]
        candidate = plan(_edit_request(), workspace)
        assert (
            await authorizer.authorize_final(
                preflight, candidate, _handshake()
            )
        ).effects == frozenset((Capability.UPDATE,))
        denials: list[PolicyErrorCode] = []
        for path in (LogicalPath(".git/present"), LogicalPath(".git/absent")):
            with pytest.raises(PolicyError) as rejected:
                await _preflight_then_snapshot(
                    authorizer,
                    replace(
                        allowed,
                        paths=(path,),
                        external_read_paths=frozenset((path,)),
                    ),
                    spy,
                )
            denials.append(project_denial(rejected.value).code)
        assert denials == [PolicyErrorCode.PATH_DENIED] * 2
        assert len(spy.calls) == 1
        hidden_path = LogicalPath(".trusted/note.txt")
        hidden_authorizer = PolicyAuthorizer(
            _policy_for(
                frozenset((Capability.UPDATE,)),
                selector=PolicyPathSelector(LogicalPath(".trusted"), True),
            )
        )
        hidden_preflight, hidden_workspace = await _preflight_then_snapshot(
            hidden_authorizer,
            PreflightRequest(
                OperationType.EDIT,
                (hidden_path,),
                frozenset((Capability.UPDATE,)),
                frozenset((hidden_path,)),
                _effective_limits(),
            ),
            spy,
        )
        hidden_candidate = plan(
            _edit_request(hidden_path.value), hidden_workspace
        )
        assert (
            await hidden_authorizer.authorize_final(
                hidden_preflight, hidden_candidate, _handshake()
            )
        ).effects == frozenset((Capability.UPDATE,))
        assert spy.calls[-1] == (hidden_path,)
        second = _candidate_with(
            frozenset((Capability.UPDATE,)),
            source=LogicalPath("other.txt"),
            destination=LogicalPath("other.txt"),
        )
        multi = replace(
            candidate,
            lineages=(
                candidate.lineages[0],
                replace(
                    second.lineages[0],
                    lineage_id=PatchLineageId("lineage_" + "b" * 16),
                ),
            ),
            final_files=(candidate.final_files[0], second.final_files[0]),
        )
        multi_preflight = await authorizer.authorize_preinspection(
            PreflightRequest(
                OperationType.EDIT,
                (LogicalPath("note.txt"), LogicalPath("other.txt")),
                frozenset((Capability.UPDATE,)),
                frozenset((LogicalPath("note.txt"), LogicalPath("other.txt"))),
                _effective_limits(),
            )
        )
        assert (
            len(
                (
                    await authorizer.authorize_final(
                        multi_preflight, multi, _handshake()
                    )
                ).effects
            )
            == 1
        )
        for disclosure in (
            PolicyDisclosure.MODEL_DIFF,
            PolicyDisclosure.MODEL_METADATA,
            PolicyDisclosure.MODEL_MATCH_DETAILS,
            PolicyDisclosure.AUDIT_PATHS,
        ):
            model = project_model(
                await _sealed_plan(disclosures=frozenset((disclosure,)))
            )
            match disclosure:
                case PolicyDisclosure.MODEL_DIFF:
                    assert model.diff is not None
                case PolicyDisclosure.MODEL_METADATA:
                    assert model.hashes_and_sizes
                case PolicyDisclosure.MODEL_MATCH_DETAILS:
                    assert model.detailed_matches
                case PolicyDisclosure.AUDIT_PATHS:
                    assert model.paths
        for mode in (
            ApprovalMode.REQUIRE_REVIEW,
            ApprovalMode.PREAUTHORIZED,
        ):
            assert (
                await _sealed_plan(mode)
            ).binding.final.approval.mode is mode
        with pytest.raises(PolicyError):
            await PolicyAuthorizer(
                _policy_for(
                    frozenset((Capability.UPDATE,)),
                    mode=ApprovalMode.DENY,
                )
            ).authorize_preinspection(allowed)

    run(execute())


def test_patch_phase_5_planner_transition_effects_and_path_closure() -> None:
    """Derive virtual effects and deny paths outside preflight authority."""

    async def execute() -> None:
        created = plan(
            _apply_request(
                "*** Begin Patch v1",
                "*** Add File: made.txt",
                "+before",
                "*** Update File: made.txt",
                "@@",
                "-before",
                "+after",
                "*** End Patch",
            ),
            PlannerWorkspace((), frozenset()),
        )
        moved = plan(
            _apply_request(
                "*** Begin Patch v1",
                "*** Update File: old.txt",
                "*** Move to: new.txt",
                "@@",
                "-before",
                "+after",
                "*** End Patch",
            ),
            PlannerWorkspace(
                (_planner_file("old.txt", b"before\n"),),
                frozenset(),
                (PlannerParentMount(None, "mount-a"),),
            ),
        )
        assert created.lineages[0].capabilities == frozenset(
            (Capability.CREATE,)
        )
        assert moved.lineages[0].capabilities == frozenset(
            (
                Capability.MOVE,
                Capability.UPDATE,
            )
        )
        for candidate, effects, paths, reads in (
            (
                created,
                frozenset((Capability.CREATE,)),
                (LogicalPath("made.txt"),),
                frozenset(),
            ),
            (
                moved,
                frozenset((Capability.MOVE, Capability.UPDATE)),
                (LogicalPath("old.txt"), LogicalPath("new.txt")),
                frozenset((LogicalPath("old.txt"),)),
            ),
        ):
            authorizer = PolicyAuthorizer(_policy_for(effects))
            preflight = await authorizer.authorize_preinspection(
                PreflightRequest(
                    OperationType.EDIT,
                    paths,
                    effects,
                    reads,
                    _effective_limits(),
                )
            )
            assert (
                await authorizer.authorize_final(
                    preflight, candidate, _handshake()
                )
            ).effects == effects
        authorizer = PolicyAuthorizer(_policy())
        preflight = await authorizer.authorize_preinspection(
            PreflightRequest(
                OperationType.EDIT,
                (LogicalPath("note.txt"),),
                frozenset((Capability.UPDATE,)),
                frozenset((LogicalPath("note.txt"),)),
                _effective_limits(),
            )
        )
        for candidate in (
            replace(
                _candidate(),
                lineages=(
                    replace(
                        _candidate().lineages[0],
                        destination_path=LogicalPath("not-preflighted.txt"),
                    ),
                ),
            ),
            replace(
                _candidate(),
                final_files=(
                    replace(
                        _candidate().final_files[0],
                        path=LogicalPath("not-preflighted.txt"),
                    ),
                ),
            ),
        ):
            with pytest.raises(PolicyError) as denied:
                await authorizer.authorize_final(
                    preflight, candidate, _handshake()
                )
            assert denied.value.code is PolicyErrorCode.DENIED

    run(execute())


def test_patch_phase_5_complete_fingerprint_and_seal_matrix() -> None:
    """Bind every durable plan field and reject every post-seal alteration."""

    async def execute() -> None:
        sealed = await _sealed_plan()
        binding = sealed.binding
        candidate = sealed.candidate
        lineage = candidate.lineages[0]

        def target_binding(target: TargetIdentity) -> PlanBinding:
            """Rebind all coupled durable target and policy identities."""
            revision = PolicyRevision(target.policy_revision)
            final = replace(
                binding.final,
                revision=revision,
                handshake=replace(binding.final.handshake, identity=target),
            )
            return replace(
                binding,
                target=target,
                preflight=replace(binding.preflight, revision=revision),
                final=final,
            )

        def resealed(
            value: PlanBinding = binding,
            planned: PlannerCandidate = candidate,
            expiry: ExpiryTick = sealed.review.expiry,
        ) -> PatchFingerprint:
            """Return a fresh fingerprint for one valid changed plan input."""
            return seal_plan(
                PatchPlanId.new(), value, planned, expiry
            ).fingerprint

        schema_request = replace(binding.request)
        object.__setattr__(schema_request, "schema_version", 2)
        changed_request_digest = AlgorithmDigest.from_bytes(b"request-digest")
        changed_initial = SourceBytes(b"other-before\n")
        changed_final = ProposedBytes(b"other-after\n")
        target_variants = (
            replace(
                binding.target,
                context_id=PatchContextId("context_" + "b" * 16),
            ),
            replace(
                binding.target,
                workspace_id=PatchWorkspaceId("workspace_" + "b" * 16),
            ),
            replace(
                binding.target,
                domain_id=PatchDomainId("domain_" + "b" * 16),
            ),
            replace(
                binding.target,
                target_id=PatchTargetId("target_" + "b" * 16),
            ),
            replace(
                binding.target,
                protocol_id=PatchProtocolId("protocol_" + "b" * 16),
            ),
            replace(binding.target, filesystem_id="filesystem-b"),
            replace(binding.target, mount_id="mount-b"),
            replace(binding.target, policy_revision="policy-six"),
            replace(
                binding.target,
                persistent_lease_id="persistent-lease-b",
            ),
            replace(
                binding.target,
                approval_channel_id=PatchApprovalId("approval_" + "b" * 16),
            ),
        )
        binding_variants = (
            replace(binding, request=schema_request),
            replace(
                binding,
                request=replace(
                    binding.request, operation=OperationType.APPLY
                ),
            ),
            replace(
                binding,
                request=replace(
                    binding.request,
                    request_id=PatchRequestId("request_" + "b" * 16),
                ),
            ),
            replace(
                binding,
                request=replace(
                    binding.request,
                    execution_id=PatchExecutionId("execution_" + "b" * 16),
                ),
            ),
            replace(
                binding,
                request=replace(
                    binding.request, input_bytes=PatchInput(b"changed-input")
                ),
            ),
            replace(
                binding,
                request=replace(
                    binding.request,
                    logical_paths=(LogicalPath("declared.txt"),),
                ),
            ),
            replace(binding, request_digest=changed_request_digest),
            replace(
                binding,
                subject=replace(
                    binding.subject,
                    principal=PatchPrincipalId("principal-b"),
                ),
            ),
            replace(
                binding,
                subject=replace(
                    binding.subject,
                    tenant=PatchTenantId("tenant-b"),
                ),
            ),
            replace(
                binding,
                subject=replace(binding.subject, run=PatchRunId("run-b")),
            ),
            replace(
                binding,
                subject=replace(
                    binding.subject,
                    session=PatchSessionId("session-b"),
                ),
            ),
            replace(
                binding,
                subject=replace(binding.subject, task=PatchTaskId("task-b")),
            ),
            replace(
                binding,
                subject=replace(
                    binding.subject, agent=PatchAgentId("agent-b")
                ),
            ),
            replace(binding, context_kind=ContextKind.SANDBOX),
            *(target_binding(item) for item in target_variants),
            replace(binding, cwd=LogicalPath("subdir")),
            replace(
                binding,
                preflight=replace(
                    binding.preflight,
                    paths=(
                        LogicalPath("note.txt"),
                        LogicalPath("declared.txt"),
                    ),
                ),
            ),
            replace(
                binding,
                preflight=replace(
                    binding.preflight,
                    effects=frozenset((Capability.UPDATE, Capability.DELETE)),
                ),
            ),
            replace(
                binding,
                preflight=replace(
                    binding.preflight,
                    effective_limits=EffectiveLimits(
                        replace(_limits(), input_bytes=ByteSize(9_999))
                    ),
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final,
                    handshake=replace(binding.final.handshake, probes=()),
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final, effects=frozenset((Capability.DELETE,))
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final,
                    disclosures=frozenset((PolicyDisclosure.AUDIT_PATHS,)),
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final,
                    effective_limits=EffectiveLimits(
                        replace(_limits(), proposed_bytes=ByteSize(9_999))
                    ),
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final,
                    approval=replace(
                        binding.final.approval,
                        route=PolicyRouteId("route-six"),
                    ),
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final,
                    approval=replace(
                        binding.final.approval,
                        broker=PolicyBrokerId("broker-six"),
                    ),
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final,
                    approval=replace(
                        binding.final.approval,
                        reviewer_role=PolicyReviewerRole("reviewer-six"),
                    ),
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final,
                    approval=replace(binding.final.approval, quorum=2),
                ),
            ),
            replace(
                binding,
                final=replace(
                    binding.final,
                    approval=ApprovalRequirements(
                        ApprovalMode.PREAUTHORIZED,
                        PolicyRouteId("route-six"),
                        PolicyBrokerId("broker-six"),
                        PolicyReviewerRole("reviewer-six"),
                        1,
                        PreauthorizationClass("trusted-reader"),
                    ),
                ),
            ),
            replace(
                binding,
                diagnostic_policy=DiagnosticPolicyId("diagnostic-five"),
            ),
        )
        candidate_variants = (
            replace(candidate, request_digest=changed_request_digest),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage,
                        lineage_id=PatchLineageId("lineage_" + "b" * 16),
                    ),
                ),
            ),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage,
                        initial=replace(
                            lineage.initial, path=LogicalPath("initial.txt")
                        ),
                    ),
                ),
            ),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage,
                        initial=replace(
                            lineage.initial,
                            bytes_value=changed_initial,
                            digest=changed_initial.digest(),
                            size=changed_initial.size(),
                        ),
                    ),
                ),
            ),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage,
                        final=replace(
                            lineage.final, path=LogicalPath("final.txt")
                        ),
                    ),
                ),
            ),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage,
                        final=replace(
                            lineage.final,
                            bytes_value=changed_final,
                            digest=changed_final.digest(),
                            size=changed_final.size(),
                        ),
                    ),
                ),
            ),
            replace(
                candidate,
                lineages=(
                    replace(lineage, source_path=LogicalPath("source.txt")),
                ),
            ),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage,
                        destination_path=LogicalPath("destination.txt"),
                    ),
                ),
            ),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage, capabilities=frozenset((Capability.DELETE,))
                    ),
                ),
            ),
            replace(candidate, lineages=(replace(lineage, matches=()),)),
            replace(
                candidate,
                lineages=(
                    replace(lineage, parent_paths=(LogicalPath("parent"),)),
                ),
            ),
            replace(
                candidate, lineages=(replace(lineage, mount_ids=("mount-b",)),)
            ),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage, lock_footprint=(LogicalPath("lock.txt"),)
                    ),
                ),
            ),
            replace(
                candidate,
                lineages=(replace(lineage, atomicity_class="atomic-b"),),
            ),
            replace(
                candidate, lineages=(replace(lineage, step_graph=("step-b",)),)
            ),
            replace(
                candidate,
                lineages=(replace(lineage, staging_class="staging-b"),),
            ),
            replace(
                candidate,
                lineages=(
                    replace(
                        lineage, diff_contribution=b"changed-contribution"
                    ),
                ),
            ),
            replace(
                candidate,
                final_files=(
                    replace(
                        candidate.final_files[0],
                        path=LogicalPath("final-file.txt"),
                    ),
                ),
            ),
            replace(
                candidate,
                diff=replace(candidate.diff, entries=(b"changed-entry",)),
            ),
            replace(
                candidate,
                diff=replace(
                    candidate.diff,
                    rendered=b"changed-diff\n",
                    digest=AlgorithmDigest.from_bytes(b"changed-diff\n"),
                ),
            ),
        )
        for index, binding_value in enumerate(binding_variants):
            assert resealed(binding_value) != sealed.fingerprint, index
        for candidate_value in candidate_variants:
            assert resealed(planned=candidate_value) != sealed.fingerprint
        assert resealed(expiry=ExpiryTick(99)) != sealed.fingerprint
        persistent_workspace_lease = target_binding(
            replace(
                binding.target,
                persistent_lease_id="persistent-lease-b",
            )
        )
        commit_lease = CommitLease(
            binding.target.domain_id,
            binding.request.request_id,
            1,
        )
        replacement_commit_lease = replace(commit_lease, fence=2)
        assert resealed(persistent_workspace_lease) != sealed.fingerprint
        assert replacement_commit_lease != commit_lease
        assert resealed() == sealed.fingerprint
        mutators: tuple[Callable[[SealedPlan], None], ...] = (
            lambda plan_value: object.__setattr__(
                plan_value, "plan_id", PatchPlanId.new()
            ),
            lambda plan_value: object.__setattr__(
                plan_value.binding.request, "schema_version", 2
            ),
            lambda plan_value: object.__setattr__(
                plan_value.binding.final.handshake,
                "identity",
                replace(
                    plan_value.binding.final.handshake.identity,
                    filesystem_id="filesystem-b",
                ),
            ),
            lambda plan_value: object.__setattr__(
                plan_value.candidate,
                "final_files",
                (
                    replace(
                        plan_value.candidate.final_files[0],
                        path=LogicalPath("tampered.txt"),
                    ),
                ),
            ),
            lambda plan_value: object.__setattr__(
                plan_value.candidate.lineages[0],
                "diff_contribution",
                b"tampered",
            ),
            lambda plan_value: object.__setattr__(
                plan_value.candidate.lineages[0].final,
                "bytes_value",
                ProposedBytes(b"tampered\n"),
            ),
            lambda plan_value: object.__setattr__(
                plan_value.review, "expiry", ExpiryTick(99)
            ),
        )
        for mutate in mutators:
            altered = await _sealed_plan()
            mutate(altered)
            with pytest.raises(PolicyError):
                project_model(altered)

    run(execute())


def test_patch_phase_5_approval_e2e_and_wrong_grant_matrix() -> None:
    """Run no-commit approval continuations and bound grant denials."""

    async def execute() -> None:
        quorum = replace(_requirements(ApprovalMode.REQUIRE_REVIEW), quorum=2)
        plan = await _sealed_plan(approval=quorum)
        request = PlanReviewRequest(plan, _subject(), quorum)
        service = ApprovalService(
            _Broker(
                ApprovalDecisionState.APPROVED,
                reviewers=(
                    PatchPrincipalId("reviewer-one"),
                    PatchPrincipalId("reviewer-two"),
                ),
            ),
            _Clock(1),
            RuntimeGrantStore(),
        )
        result = await service.await_review(request)
        assert result.grant is not None
        assert not hasattr(service, "commit")
        wrong_reviewer = ApprovalService(
            _Broker(
                ApprovalDecisionState.APPROVED,
                reviewer_role=PolicyReviewerRole("reviewer-other"),
            ),
            _Clock(1),
            RuntimeGrantStore(),
        )
        assert (
            await wrong_reviewer.await_review(request)
        ).state is ApprovalDecisionState.DENIED
        single = await _sealed_plan()
        single_request = PlanReviewRequest(
            single, _subject(), single.binding.final.approval
        )
        suspended = _SuspendedBroker()
        continuation = ApprovalService(
            suspended,
            _Clock(1),
            RuntimeGrantStore(),
        )
        waiting = create_task(continuation.await_review(single_request))
        await suspended.started.wait()
        assert not waiting.done()
        suspended.continue_review.set()
        assert (await waiting).state is ApprovalDecisionState.APPROVED
        for state in (
            ApprovalDecisionState.DENIED,
            ApprovalDecisionState.CANCELLED,
            ApprovalDecisionState.UNAVAILABLE,
        ):
            outcome = await ApprovalService(
                _Broker(state), _Clock(1), RuntimeGrantStore()
            ).await_review(request)
            assert outcome.state is not ApprovalDecisionState.APPROVED
        with pytest.raises(CancelledError):
            await ApprovalService(
                _CancelledBroker(), _Clock(1), RuntimeGrantStore()
            ).await_review(single_request)
        grants = RuntimeGrantStore()
        single_service = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED), _Clock(1), grants
        )
        grant_result = await single_service.await_review(single_request)
        assert grant_result.grant is not None
        grant = grant_result.grant
        alternate_subjects = (
            replace(_subject(), principal=PatchPrincipalId("principal-other")),
            replace(_subject(), tenant=PatchTenantId("tenant-other")),
            replace(_subject(), run=PatchRunId("run-other")),
            replace(_subject(), session=PatchSessionId("session-other")),
            replace(_subject(), task=PatchTaskId("task-other")),
            replace(_subject(), agent=PatchAgentId("agent-other")),
        )
        for subject in alternate_subjects:
            with pytest.raises(PolicyError):
                await single_service.validate_grant(grant, single, subject)
        alternate_policy = PolicyRevision("policy-six")
        alternate_target = replace(
            single.binding.target,
            workspace_id=PatchWorkspaceId("workspace_" + "b" * 16),
        )
        alternate_handshake = replace(
            single.binding.final.handshake,
            identity=alternate_target,
        )
        changed_policy_target = replace(
            single.binding.target,
            policy_revision=alternate_policy.value,
        )
        changed_policy_handshake = replace(
            single.binding.final.handshake,
            identity=changed_policy_target,
        )
        changed_policy_final = replace(
            single.binding.final,
            revision=alternate_policy,
            handshake=changed_policy_handshake,
        )
        mismatch_bindings = (
            replace(single.binding, context_kind=ContextKind.SANDBOX),
            replace(
                single.binding,
                target=alternate_target,
                final=replace(
                    single.binding.final,
                    handshake=alternate_handshake,
                ),
            ),
            replace(
                single.binding,
                preflight=replace(
                    single.binding.preflight,
                    revision=alternate_policy,
                ),
                target=changed_policy_target,
                final=changed_policy_final,
            ),
            replace(
                single.binding,
                request_digest=AlgorithmDigest.from_bytes(b"wrong-request"),
            ),
            replace(
                single.binding,
                final=replace(
                    single.binding.final,
                    effects=frozenset((Capability.DELETE,)),
                ),
            ),
            replace(
                single.binding,
                final=replace(
                    single.binding.final,
                    disclosures=frozenset((PolicyDisclosure.AUDIT_PATHS,)),
                ),
            ),
            replace(
                single.binding,
                final=replace(
                    single.binding.final,
                    approval=replace(
                        single.binding.final.approval,
                        route=PolicyRouteId("route-other"),
                    ),
                ),
            ),
            replace(
                single.binding,
                final=replace(
                    single.binding.final,
                    approval=replace(
                        single.binding.final.approval,
                        broker=PolicyBrokerId("broker-other"),
                    ),
                ),
            ),
            replace(
                single.binding,
                final=replace(
                    single.binding.final,
                    approval=replace(single.binding.final.approval, quorum=2),
                ),
            ),
        )
        for binding in mismatch_bindings:
            with pytest.raises(PolicyError):
                await single_service.validate_grant(
                    grant,
                    seal_plan(
                        single.plan_id,
                        binding,
                        single.candidate,
                        single.review.expiry,
                    ),
                    _subject(),
                )
        changed_diff = b"wrong-diff\n"
        with pytest.raises(PolicyError):
            await single_service.validate_grant(
                grant,
                seal_plan(
                    single.plan_id,
                    single.binding,
                    replace(
                        single.candidate,
                        diff=replace(
                            single.candidate.diff,
                            rendered=changed_diff,
                            digest=AlgorithmDigest.from_bytes(changed_diff),
                        ),
                    ),
                    single.review.expiry,
                ),
                _subject(),
            )
        with pytest.raises(PolicyError):
            await single_service.validate_grant(
                grant,
                seal_plan(
                    PatchPlanId.new(),
                    single.binding,
                    single.candidate,
                    single.review.expiry,
                ),
                _subject(),
            )
        assert not hasattr(ApprovalService, "approve_all")

    run(execute())


def test_patch_phase_5_private_canaries_and_n_plus_one_bounds() -> None:
    """Keep complete artifacts private and reject expanded finite inputs."""

    async def execute() -> None:
        sealed = await _sealed_plan()
        canary = b"phase-five-private-canary\n"
        private_candidate = replace(
            sealed.candidate,
            diff=replace(
                sealed.candidate.diff,
                entries=(canary,),
                rendered=canary,
                digest=AlgorithmDigest.from_bytes(canary),
            ),
        )
        private_plan = seal_plan(
            PatchPlanId.new(),
            sealed.binding,
            private_candidate,
            sealed.review.expiry,
        )
        with pytest.raises(PolicyError) as invalid_diff:
            seal_plan(
                PatchPlanId.new(),
                sealed.binding,
                replace(
                    sealed.candidate,
                    diff=replace(
                        sealed.candidate.diff,
                        digest=AlgorithmDigest.from_bytes(b"wrong-digest"),
                    ),
                ),
                sealed.review.expiry,
            )
        assert invalid_diff.value.code is PolicyErrorCode.INVALID_PLAN
        model = project_model(private_plan)
        assert canary not in repr(model).encode()
        with pytest.raises(PolicyError):
            project_reviewer(private_plan)
        summary = await RuntimePlanStore().put(private_plan)
        assert canary not in repr(project_sdk_host(summary)).encode()
        assert (
            canary
            not in repr(
                project_denial(PolicyError(PolicyErrorCode.APPROVAL_MISMATCH))
            ).encode()
        )
        expanded_paths = tuple(
            LogicalPath("nested/path-" + str(index))
            for index in range(_limits().path_count.value + 1)
        )
        with pytest.raises(PolicyError) as path_limit:
            PreflightRequest(
                OperationType.EDIT,
                expanded_paths,
                frozenset((Capability.UPDATE,)),
                frozenset(expanded_paths),
                _effective_limits(),
            )
        assert path_limit.value.code is PolicyErrorCode.DENIED
        with pytest.raises(PolicyError):
            replace(_requirements(ApprovalMode.REQUIRE_REVIEW), quorum=65)
        second = _candidate_with(
            frozenset((Capability.UPDATE,)),
            source=LogicalPath("second.txt"),
            destination=LogicalPath("second.txt"),
        )
        expanded_candidate = replace(
            sealed.candidate,
            lineages=(
                sealed.candidate.lineages[0],
                replace(
                    second.lineages[0],
                    lineage_id=PatchLineageId("lineage_" + "c" * 16),
                ),
            ),
            final_files=(
                sealed.candidate.final_files[0],
                second.final_files[0],
            ),
        )
        one_file_limit = EffectiveLimits(
            replace(
                _limits(), file_count=ByteSize(1), operation_count=ByteSize(1)
            )
        )
        with pytest.raises(PolicyError) as candidate_limit:
            seal_plan(
                PatchPlanId.new(),
                replace(
                    sealed.binding,
                    final=replace(
                        sealed.binding.final,
                        effective_limits=one_file_limit,
                    ),
                ),
                expanded_candidate,
                sealed.review.expiry,
            )
        assert candidate_limit.value.code is PolicyErrorCode.LIMIT_EXCEEDED
        requirements = _requirements(ApprovalMode.REQUIRE_REVIEW)
        object.__setattr__(requirements, "route", "raw-route")
        with pytest.raises(PolicyError):
            requirements.__post_init__()
        decision = ReviewerDecision(
            PatchPrincipalId("reviewer-raw"),
            PatchTenantId("tenant-a"),
            PolicyReviewerRole("reviewer-five"),
            ApprovalDecisionState.APPROVED,
        )
        object.__setattr__(decision, "reviewer_role", "raw-role")
        with pytest.raises(PolicyError):
            decision.__post_init__()
        review = PlanReviewRequest(
            sealed,
            _subject(),
            sealed.binding.final.approval,
        )
        object.__setattr__(review, "subject", "raw-subject")
        with pytest.raises(PolicyError):
            review.__post_init__()

    run(execute())


@pytest.mark.parametrize(
    "case",
    (
        "capability_denials",
        "target_spy",
        "scope_and_raw_inputs",
        "grant_matrix",
        "private_canaries",
        "n_plus_one",
        "approval_e2e",
    ),
)
def test_patch_phase_5_residual_negative_and_e2e_proof(
    tmp_path: Path, case: str
) -> None:
    """Prove the remaining precommit authority boundaries by case."""

    async def execute() -> None:
        request = PreflightRequest(
            OperationType.EDIT,
            (LogicalPath("note.txt"),),
            frozenset((Capability.UPDATE,)),
            frozenset((LogicalPath("note.txt"),)),
            _effective_limits(),
        )
        if case == "capability_denials":
            with pytest.raises(PolicyError):
                await PolicyAuthorizer(
                    TrustedPatchPolicy.empty()
                ).authorize_preinspection(request)
            reader = PreauthorizationClass("trusted-reader")
            missing_observation = TrustedPatchPolicy(
                PolicyRevision("policy-five"),
                frozenset((OperationType.EDIT,)),
                (
                    PolicyRule(
                        PolicyPathSelector(None),
                        (
                            CapabilityMode(
                                Capability.UPDATE,
                                ApprovalMode.REQUIRE_REVIEW,
                            ),
                            CapabilityMode(
                                Capability.READ_FOR_MUTATION,
                                ApprovalMode.PREAUTHORIZED,
                                reader,
                            ),
                        ),
                    ),
                ),
                _limits(),
            )
            with pytest.raises(PolicyError):
                await PolicyAuthorizer(
                    missing_observation
                ).authorize_preinspection(request)
            authorizer = PolicyAuthorizer(_policy())
            preflight = await authorizer.authorize_preinspection(request)
            with pytest.raises(PolicyError):
                await authorizer.authorize_final(
                    preflight, _candidate(), _handshake(effectful=False)
                )
            with pytest.raises(PolicyError):
                await authorizer.authorize_preinspection(
                    replace(
                        request,
                        paths=(LogicalPath(".git/config"),),
                        external_read_paths=frozenset(
                            (LogicalPath(".git/config"),)
                        ),
                    )
                )
            executable = replace(
                _candidate(),
                lineages=(
                    replace(
                        _candidate().lineages[0],
                        final=replace(
                            _candidate().lineages[0].final,
                            metadata=MetadataProfile(
                                FileMode(0o755), False, "lf"
                            ),
                        ),
                    ),
                ),
            )
            with pytest.raises(PolicyError):
                await authorizer.authorize_final(
                    preflight, executable, _handshake()
                )
            assert project_model(await _sealed_plan()).diff is None
        elif case == "target_spy":
            (tmp_path / ".git").mkdir()
            (tmp_path / ".git" / "present").write_bytes(b"private\n")
            spy = _RootedTargetSpy(tmp_path)
            errors: list[PolicyErrorCode] = []
            for path in (
                LogicalPath(".git/present"),
                LogicalPath(".git/absent"),
            ):
                with pytest.raises(PolicyError) as rejected:
                    await _preflight_then_snapshot(
                        PolicyAuthorizer(_policy()),
                        replace(
                            request,
                            paths=(path,),
                            external_read_paths=frozenset((path,)),
                        ),
                        spy,
                    )
                errors.append(project_denial(rejected.value).code)
            assert errors == [PolicyErrorCode.PATH_DENIED] * 2
            assert spy.calls == []
        elif case == "scope_and_raw_inputs":
            sealed = await _sealed_plan()
            boundary = Phase5IngressBoundary()

            class AlternateControlIngress(Phase5ControlIngress):
                """Represent an untrusted caller-controlled ingress subtype."""

            malformed = object.__new__(Phase5ControlIngress)
            object.__setattr__(malformed, "surface", "raw-surface")
            object.__setattr__(malformed, "payload", b"widen")
            with pytest.raises(PolicyError):
                malformed.__post_init__()
            for surface in Phase5IngressSurface:
                with pytest.raises(PolicyError):
                    boundary.reject_control_widening(
                        Phase5ControlIngress(
                            surface, {"requested_capability": "approve-all"}
                        )
                    )
            with pytest.raises(PolicyError):
                boundary.reject_control_widening(
                    AlternateControlIngress(
                        Phase5IngressSurface.TOOL_ARGUMENTS, b"widen"
                    )
                )
            for raw_artifact in (
                b"yes",
                "y",
                True,
                "approve-all",
                sealed.review.diff,
                project_model(sealed),
            ):
                with pytest.raises(PolicyError):
                    boundary.review_request(sealed, _subject(), raw_artifact)
            assert boundary.review_request(
                sealed, _subject(), sealed.review
            ) == PlanReviewRequest(
                sealed, _subject(), sealed.binding.final.approval
            )
            with pytest.raises(PolicyError):
                boundary.review_request(
                    sealed,
                    _subject(),
                    replace(sealed.review, expiry=ExpiryTick(99)),
                )
        elif case == "grant_matrix":
            sealed = await _sealed_plan()
            grants = RuntimeGrantStore()
            service = ApprovalService(
                _Broker(ApprovalDecisionState.APPROVED), _Clock(1), grants
            )
            outcome = await service.await_review(
                PlanReviewRequest(
                    sealed, _subject(), sealed.binding.final.approval
                )
            )
            assert outcome.grant is not None
            grant = outcome.grant
            for altered_grant in (
                replace(grant, _secret=b"x" * 32),
                replace(grant, plan_id=PatchPlanId.new()),
                replace(
                    grant,
                    binding=replace(
                        grant.binding,
                        request_digest=AlgorithmDigest.from_bytes(b"wrong"),
                    ),
                ),
                replace(
                    grant,
                    diff_digest=AlgorithmDigest.from_bytes(b"wrong-diff"),
                ),
            ):
                with pytest.raises(PolicyError):
                    await service.validate_grant(
                        altered_grant, sealed, _subject()
                    )
            with pytest.raises(PolicyError) as expired:
                await ApprovalService(
                    _Broker(ApprovalDecisionState.APPROVED),
                    _Clock(100),
                    grants,
                ).validate_grant(grant, sealed, _subject())
            assert expired.value.code is PolicyErrorCode.APPROVAL_EXPIRED
            with pytest.raises(PolicyError):
                await service.validate_grant(
                    grant,
                    sealed,
                    replace(_subject(), tenant=PatchTenantId("tenant-other")),
                )
        elif case == "private_canaries":
            sealed = await _sealed_plan()
            canary = b"private-model-event-audit-grant-canary"
            candidate = replace(
                sealed.candidate,
                diff=replace(
                    sealed.candidate.diff,
                    entries=(canary,),
                    rendered=canary,
                    digest=AlgorithmDigest.from_bytes(canary),
                ),
            )
            private = seal_plan(
                PatchPlanId.new(), sealed.binding, candidate, ExpiryTick(100)
            )
            summary = await RuntimePlanStore().put(private)
            public = (
                repr(project_model(private))
                + repr(project_sdk_host(summary))
                + repr(project_denial(PolicyError(PolicyErrorCode.DENIED)))
            ).encode()
            assert canary not in public
            with pytest.raises(PolicyError):
                project_reviewer(private)
        elif case == "n_plus_one":
            with pytest.raises(PolicyError):
                PreflightRequest(
                    OperationType.EDIT,
                    tuple(
                        LogicalPath("path-" + str(index))
                        for index in range(21)
                    ),
                    frozenset((Capability.UPDATE,)),
                    frozenset(),
                    _effective_limits(),
                )
            with pytest.raises(PolicyError):
                replace(_requirements(ApprovalMode.REQUIRE_REVIEW), quorum=65)
            sealed = await _sealed_plan()
            with pytest.raises(PolicyError):
                seal_plan(
                    PatchPlanId.new(),
                    replace(
                        sealed.binding,
                        final=replace(
                            sealed.binding.final,
                            effective_limits=EffectiveLimits(
                                replace(
                                    _limits(), review_diff_bytes=ByteSize(1)
                                )
                            ),
                        ),
                    ),
                    sealed.candidate,
                    sealed.review.expiry,
                )
        else:
            (tmp_path / "note.txt").write_bytes(b"before\n")
            spy = _RootedTargetSpy(tmp_path)
            authorizer = PolicyAuthorizer(_policy())
            preflight, workspace = await _preflight_then_snapshot(
                authorizer, request, spy
            )
            candidate = plan(_edit_request(), workspace)
            final = await authorizer.authorize_final(
                preflight, candidate, _handshake()
            )
            plan_value = seal_plan(
                PatchPlanId.new(),
                replace(
                    (await _sealed_plan()).binding,
                    preflight=preflight,
                    final=final,
                ),
                candidate,
                ExpiryTick(100),
            )
            review = PlanReviewRequest(
                plan_value, _subject(), plan_value.binding.final.approval
            )
            for broker, clock, expected in (
                (
                    _Broker(ApprovalDecisionState.DENIED),
                    _Clock(1),
                    ApprovalDecisionState.DENIED,
                ),
                (
                    _UnavailableBroker(),
                    _Clock(1),
                    ApprovalDecisionState.UNAVAILABLE,
                ),
                (
                    _Broker(ApprovalDecisionState.APPROVED),
                    _Clock(100),
                    ApprovalDecisionState.DENIED,
                ),
            ):
                assert (
                    await ApprovalService(
                        broker, clock, RuntimeGrantStore()
                    ).await_review(review)
                ).state is expected
            with pytest.raises(CancelledError):
                await ApprovalService(
                    _CancelledBroker(), _Clock(1), RuntimeGrantStore()
                ).await_review(review)
            assert spy.calls == [(LogicalPath("note.txt"),)]

    run(execute())


def test_patch_phase_5_private_artifact_retention_cleanup() -> None:
    """Bound seal, plan, and grant retention and remove expired artifacts."""

    async def execute() -> None:
        class AlternateRetention(PrivateArtifactRetention):
            """Represent an untrusted retention subtype at the boundary."""

        with pytest.raises(PolicyError):
            PrivateArtifactRetention(0)
        template = await _sealed_plan()
        assert cleanup_sealed_authorities(ExpiryTick(1_000)) >= 1
        retention = PrivateArtifactRetention(1)
        alternate_retention = AlternateRetention(1)
        with pytest.raises(PolicyError):
            seal_plan(
                PatchPlanId.new(),
                template.binding,
                template.candidate,
                ExpiryTick(100),
                alternate_retention,
            )
        with pytest.raises(PolicyError):
            RuntimePlanStore(alternate_retention)
        with pytest.raises(PolicyError):
            RuntimeGrantStore(alternate_retention)
        expired = seal_plan(
            PatchPlanId.new(),
            template.binding,
            template.candidate,
            ExpiryTick(2),
            retention,
        )
        with pytest.raises(PolicyError) as seal_limit:
            seal_plan(
                PatchPlanId.new(),
                template.binding,
                template.candidate,
                ExpiryTick(100),
                retention,
            )
        assert seal_limit.value.code is PolicyErrorCode.LIMIT_EXCEEDED
        active = seal_plan(
            PatchPlanId.new(),
            template.binding,
            template.candidate,
            ExpiryTick(100),
        )
        plans = RuntimePlanStore(retention)
        await plans.put(expired)
        with pytest.raises(PolicyError) as plan_limit:
            await plans.put(active)
        assert plan_limit.value.code is PolicyErrorCode.LIMIT_EXCEEDED
        assert await plans.cleanup_expired(ExpiryTick(2)) == 1
        assert await plans.get(expired.plan_id) is None
        await plans.put(active)
        grants = RuntimeGrantStore(retention)
        expired_grant_service = ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED), _Clock(1), grants
        )
        expired_grant = await expired_grant_service.await_review(
            PlanReviewRequest(
                expired, _subject(), expired.binding.final.approval
            )
        )
        assert expired_grant.grant is not None
        with pytest.raises(PolicyError) as grant_limit:
            await ApprovalService(
                _Broker(ApprovalDecisionState.APPROVED), _Clock(1), grants
            ).await_review(
                PlanReviewRequest(
                    active, _subject(), active.binding.final.approval
                )
            )
        assert grant_limit.value.code is PolicyErrorCode.LIMIT_EXCEEDED
        assert await grants.cleanup_expired(ExpiryTick(2)) == 1
        active_grant = await ApprovalService(
            _Broker(ApprovalDecisionState.APPROVED), _Clock(1), grants
        ).await_review(
            PlanReviewRequest(
                active, _subject(), active.binding.final.approval
            )
        )
        assert active_grant.grant is not None
        assert cleanup_sealed_authorities(ExpiryTick(100)) == 2

    run(execute())
