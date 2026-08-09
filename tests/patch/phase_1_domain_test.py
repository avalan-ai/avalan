"""Exercise the dormant immutable mutation-domain boundary."""

from dataclasses import FrozenInstanceError, replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Protocol, runtime_checkable

import pytest

import avalan.patch.codec as codec
from avalan.patch.codec import (
    decode_diagnostic,
    decode_event,
    decode_pending,
    decode_public_pending,
    decode_result,
    encode_diagnostic,
    encode_event,
    encode_pending,
    encode_public_pending,
    encode_result,
)
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalGrant,
    ArtifactState,
    AtomicityClass,
    Audience,
    ByteSize,
    Capability,
    CommitGraph,
    CommitStepJournal,
    CommitStepState,
    CommitTruth,
    ContextKind,
    DiffBytes,
    DomainFacade,
    DurationTicks,
    ErrorStage,
    ExpiryTick,
    FileMode,
    GrantSecret,
    LifecyclePhase,
    Lineage,
    LineageJournal,
    LineageState,
    LogicalPath,
    MatchStrategy,
    MetadataProfile,
    MutationPlan,
    MutationScope,
    MutationState,
    OperationType,
    PatchApprovalId,
    PatchContextId,
    PatchCredential,
    PatchDiagnostic,
    PatchDomainId,
    PatchErrorCode,
    PatchEventId,
    PatchExecutionId,
    PatchFingerprint,
    PatchGrantId,
    PatchInput,
    PatchInvariantError,
    PatchLifecycleEvent,
    PatchLimits,
    PatchLineageId,
    PatchObserverCorrelationId,
    PatchObserverId,
    PatchOperationId,
    PatchPending,
    PatchPendingOperationId,
    PatchPlanId,
    PatchProtocolId,
    PatchRequest,
    PatchRequestId,
    PatchResult,
    PatchStatus,
    PatchStepId,
    PatchTargetId,
    PatchValidationError,
    PatchWorkspaceId,
    PostconditionState,
    PrivateStagingName,
    ProjectionInput,
    ProposedBytes,
    RequestedEffectOccurrence,
    Retryability,
    ReviewArtifact,
    SequenceNumber,
    Snapshot,
    SourceBytes,
    VirtualFile,
    WorkspaceChange,
    _PatchIdentifier,
    advance_lifecycle,
    coarsen_error_code,
    derive_commit_truth,
    project_pending,
)

_ROOT = Path(__file__).resolve().parents[2]


@runtime_checkable
class _MutableConstructionHelper(Protocol):
    """Describe the dedicated type-negative runtime constructor helper."""

    def construct_mutable_scope(self, scope: MutationScope) -> MutationScope:
        """Attempt to construct a scope from mutable capabilities."""

    def construct_mutable_lineage(self, lineage: Lineage) -> Lineage:
        """Attempt to construct a lineage from mutable capabilities."""


def _mutable_construction_helper() -> _MutableConstructionHelper:
    """Load the exact negative type fixture for runtime boundary execution."""
    path = (
        _ROOT
        / "tests"
        / "patch_type_contracts"
        / "phase1_mutable_collection_negative.py"
    )
    spec = spec_from_file_location("phase1_mutable_collection_negative", path)
    assert spec is not None and spec.loader is not None
    module: ModuleType = module_from_spec(spec)
    spec.loader.exec_module(module)
    if not isinstance(module, _MutableConstructionHelper):
        raise AssertionError("mutable constructor fixture is incomplete")
    return module


def _request_id() -> PatchRequestId:
    """Return one deterministic request identifier."""
    return PatchRequestId("request_0123456789abcdef")


def _plan_id() -> PatchPlanId:
    """Return one deterministic plan identifier."""
    return PatchPlanId("plan_0123456789abcdef")


def _correlation_id() -> PatchObserverCorrelationId:
    """Return one deterministic observer correlation identifier."""
    return PatchObserverCorrelationId("correlation_0123456789abcdef")


def _step_id() -> PatchStepId:
    """Return one deterministic step identifier."""
    return PatchStepId("step_0123456789abcdef")


def _lineage_id() -> PatchLineageId:
    """Return one deterministic lineage identifier."""
    return PatchLineageId("lineage_0123456789abcdef")


def _diagnostic() -> PatchDiagnostic:
    """Return one stable commit diagnostic."""
    return PatchDiagnostic(
        stage=ErrorStage.COMMIT,
        code=PatchErrorCode.COMMIT_FAILED,
        retryability=Retryability.NOT_RETRYABLE,
    )


def _limits() -> PatchLimits:
    """Return one valid nonzero bounded plan limit set."""
    return PatchLimits(
        input_bytes=ByteSize(1),
        path_count=ByteSize(1),
        path_length=ByteSize(1),
        file_count=ByteSize(1),
        operation_count=ByteSize(1),
        snapshot_bytes=ByteSize(1),
        proposed_bytes=ByteSize(1),
        review_diff_bytes=ByteSize(1),
        planning_duration=DurationTicks(1),
        approval_duration=DurationTicks(1),
        commit_duration=DurationTicks(1),
    )


def _journal(
    *states: CommitStepState,
    lineage_id: PatchLineageId | None = None,
    artifact: ArtifactState = ArtifactState.CLEANED,
    postcondition: PostconditionState = PostconditionState.ESTABLISHED,
) -> LineageJournal:
    """Return one closed journal with deterministic distinct steps."""
    return LineageJournal(
        lineage_id=lineage_id or _lineage_id(),
        steps=tuple(
            CommitStepJournal(
                step_id=PatchStepId(f"step_{index:016x}"), state=state
            )
            for index, state in enumerate(states, start=1)
        ),
        postcondition=postcondition,
        artifact_state=artifact,
    )


def _plan(*journals: LineageJournal) -> MutationPlan:
    """Return one sealed plan whose graphs exactly bind the journals."""
    request = PatchRequest(
        schema_version=1,
        request_id=_request_id(),
        execution_id=PatchExecutionId("execution_0123456789abcdef"),
        operation=OperationType.EDIT,
        input_bytes=PatchInput(b"request"),
        logical_paths=(LogicalPath("file.txt"),),
    )
    limits = _limits()
    scope = MutationScope(
        context_kind=ContextKind.LOCAL,
        context_id=PatchContextId("context_0123456789abcdef"),
        workspace_id=PatchWorkspaceId("workspace_0123456789abcdef"),
        domain_id=PatchDomainId("domain_0123456789abcdef"),
        target_id=PatchTargetId("target_0123456789abcdef"),
        protocol_id=PatchProtocolId("protocol_0123456789abcdef"),
        capabilities=frozenset((Capability.UPDATE,)),
        disclosures=frozenset(),
        limits=limits,
    )
    lineages = tuple(
        Lineage(
            journal.lineage_id,
            LogicalPath(f"before-{index}.txt"),
            LogicalPath(f"after-{index}.txt"),
            frozenset((Capability.UPDATE,)),
            MatchStrategy.EXACT_BYTES,
            CommitGraph(
                tuple(step.step_id for step in journal.steps),
                AtomicityClass.SINGLE_STEP,
            ),
        )
        for index, journal in enumerate(journals, start=1)
    )
    review = ReviewArtifact(
        DiffBytes(b"diff"), AlgorithmDigest.from_bytes(b"diff"), ByteSize(4)
    )
    return MutationPlan(
        _plan_id(), request, scope, lineages, review, PatchFingerprint(b"plan")
    )


def _truth(*journals: LineageJournal) -> CommitTruth:
    """Derive request-wide truth from a matching sealed plan and journals."""
    return derive_commit_truth(_plan(*journals), journals)


def test_patch_phase_1_identifiers_are_distinct_and_bounded() -> None:
    """Construct all immutable identity values with their exact prefixes."""
    values = (
        _request_id(),
        PatchExecutionId("execution_0123456789abcdef"),
        _plan_id(),
        PatchOperationId("operation_0123456789abcdef"),
        _lineage_id(),
        _step_id(),
        PatchContextId("context_0123456789abcdef"),
        PatchWorkspaceId("workspace_0123456789abcdef"),
        PatchDomainId("domain_0123456789abcdef"),
        PatchTargetId("target_0123456789abcdef"),
        PatchProtocolId("protocol_0123456789abcdef"),
        PatchGrantId("grant_0123456789abcdef"),
        PatchApprovalId("approval_0123456789abcdef"),
        PatchEventId("event_0123456789abcdef"),
        PatchPendingOperationId("pending_0123456789abcdef"),
        _correlation_id(),
    )
    assert len(values) == 16
    assert PatchObserverId.new().value.startswith("observer_")
    with pytest.raises(PatchValidationError):
        PatchRequestId("plan_0123456789abcdef")


def test_patch_phase_1_values_preserve_canonical_order_and_bounds() -> None:
    """Construct bounded paths, sizes, digests, modes, and durations."""
    assert LogicalPath("nested/file.txt") == LogicalPath("nested/file.txt")
    assert ByteSize(0) < ByteSize(1)
    assert SequenceNumber(0) < SequenceNumber(1)
    assert DurationTicks(1).value == 1
    assert ExpiryTick(1).value == 1
    assert FileMode(0o644).value == 0o644
    assert AlgorithmDigest.from_bytes(b"source").algorithm == "sha256"
    for invalid in ("", "/root", "nested/../file", "nested\\file"):
        with pytest.raises(PatchValidationError):
            LogicalPath(invalid)


def test_patch_phase_1_closed_enums_and_metadata_reject_unknown_values() -> (
    None
):
    """Keep operation, authority, context, matching, and metadata closed."""
    assert OperationType.EDIT.value == "edit"
    assert Capability.UPDATE.value == "update"
    assert ContextKind.SANDBOX.value == "sandbox"
    assert MatchStrategy.EXACT_BYTES.value == "exact_bytes"
    assert MetadataProfile(FileMode(0o644), False, "lf").newline == "lf"
    with pytest.raises(ValueError):
        OperationType("replace")
    with pytest.raises(PatchValidationError):
        MetadataProfile(FileMode(0o644), False, "mixed")


def test_patch_phase_1_redacted_values_never_render_canaries() -> None:
    """Keep content, grants, staging names, and credentials redacted."""
    canary = b"PATCH-PRIVATE-CANARY"
    values = (
        PatchInput(canary),
        SourceBytes(canary),
        ProposedBytes(canary),
        DiffBytes(canary),
        PatchFingerprint(canary),
        GrantSecret(canary),
        PrivateStagingName(canary),
        PatchCredential(canary),
    )
    assert all("PATCH-PRIVATE-CANARY" not in repr(value) for value in values)
    assert all("PATCH-PRIVATE-CANARY" not in str(value) for value in values)
    assert SourceBytes(canary).size() == ByteSize(len(canary))


def test_patch_phase_1_request_scope_plan_contracts() -> None:
    """Construct structure-only request, scope, snapshot, and plan values."""
    limits = PatchLimits(
        input_bytes=ByteSize(1),
        path_count=ByteSize(1),
        path_length=ByteSize(1),
        file_count=ByteSize(1),
        operation_count=ByteSize(1),
        snapshot_bytes=ByteSize(1),
        proposed_bytes=ByteSize(1),
        review_diff_bytes=ByteSize(1),
        planning_duration=DurationTicks(1),
        approval_duration=DurationTicks(1),
        commit_duration=DurationTicks(1),
    )
    request = PatchRequest(
        schema_version=1,
        request_id=_request_id(),
        execution_id=PatchExecutionId("execution_0123456789abcdef"),
        operation=OperationType.EDIT,
        input_bytes=PatchInput(b"request"),
        logical_paths=(LogicalPath("file.txt"),),
    )
    scope = MutationScope(
        context_kind=ContextKind.LOCAL,
        context_id=PatchContextId("context_0123456789abcdef"),
        workspace_id=PatchWorkspaceId("workspace_0123456789abcdef"),
        domain_id=PatchDomainId("domain_0123456789abcdef"),
        target_id=PatchTargetId("target_0123456789abcdef"),
        protocol_id=PatchProtocolId("protocol_0123456789abcdef"),
        capabilities=frozenset((Capability.UPDATE,)),
        disclosures=frozenset(),
        limits=limits,
    )
    metadata = MetadataProfile(FileMode(0o644), False, "lf")
    snapshot = Snapshot(
        path=LogicalPath("file.txt"),
        present=True,
        size=ByteSize(6),
        digest=AlgorithmDigest.from_bytes(b"before"),
        metadata=metadata,
    )
    virtual = VirtualFile(
        LogicalPath("file.txt"), ProposedBytes(b"after"), metadata
    )
    graph = CommitGraph((_step_id(),), AtomicityClass.SINGLE_STEP)
    lineage = Lineage(
        _lineage_id(),
        LogicalPath("file.txt"),
        LogicalPath("file.txt"),
        frozenset((Capability.UPDATE,)),
        MatchStrategy.EXACT_BYTES,
        graph,
    )
    review = ReviewArtifact(
        DiffBytes(b"diff"), AlgorithmDigest.from_bytes(b"diff"), ByteSize(4)
    )
    plan = MutationPlan(
        _plan_id(),
        request,
        scope,
        (lineage,),
        review,
        PatchFingerprint(b"fingerprint"),
    )
    assert snapshot.present and virtual.path == LogicalPath("file.txt")
    assert plan.request is request
    with pytest.raises(FrozenInstanceError):
        setattr(request, "schema_version", 2)


def test_patch_phase_1_legal_lifecycle() -> None:
    """Walk each approval, commit, pending, settlement, and terminal branch."""
    legal = (
        (LifecyclePhase.RECEIVED, LifecyclePhase.PARSED),
        (LifecyclePhase.RECEIVED, LifecyclePhase.REQUEST_COMPLETED),
        (LifecyclePhase.PARSED, LifecyclePhase.SCOPE_BOUND),
        (LifecyclePhase.PARSED, LifecyclePhase.REQUEST_COMPLETED),
        (LifecyclePhase.SCOPE_BOUND, LifecyclePhase.PREFLIGHT_AUTHORIZED),
        (LifecyclePhase.SCOPE_BOUND, LifecyclePhase.REQUEST_COMPLETED),
        (LifecyclePhase.PREFLIGHT_AUTHORIZED, LifecyclePhase.PLANNED),
        (
            LifecyclePhase.PREFLIGHT_AUTHORIZED,
            LifecyclePhase.REQUEST_COMPLETED,
        ),
        (LifecyclePhase.PLANNED, LifecyclePhase.APPROVAL_REQUIRED),
        (LifecyclePhase.PLANNED, LifecyclePhase.APPROVED),
        (LifecyclePhase.PLANNED, LifecyclePhase.REQUEST_COMPLETED),
        (LifecyclePhase.APPROVAL_REQUIRED, LifecyclePhase.APPROVED),
        (LifecyclePhase.APPROVAL_REQUIRED, LifecyclePhase.REQUEST_COMPLETED),
        (LifecyclePhase.APPROVED, LifecyclePhase.COMMIT_READY),
        (LifecyclePhase.APPROVED, LifecyclePhase.REQUEST_COMPLETED),
        (LifecyclePhase.COMMIT_READY, LifecyclePhase.COMMIT_STARTED),
        (LifecyclePhase.COMMIT_STARTED, LifecyclePhase.SETTLED),
        (LifecyclePhase.COMMIT_STARTED, LifecyclePhase.SETTLEMENT_PENDING),
        (LifecyclePhase.SETTLEMENT_PENDING, LifecyclePhase.SETTLED),
        (LifecyclePhase.SETTLED, LifecyclePhase.REQUEST_COMPLETED),
    )
    assert all(
        advance_lifecycle(current, next) is next for current, next in legal
    )


def test_patch_phase_1_invalid_lifecycle() -> None:
    """Reject guessed terminal transitions and pre-approval mutation paths."""
    with pytest.raises(PatchValidationError):
        advance_lifecycle(
            LifecyclePhase.REQUEST_COMPLETED, LifecyclePhase.SETTLED
        )
    with pytest.raises(PatchValidationError):
        advance_lifecycle(
            LifecyclePhase.PLANNED, LifecyclePhase.COMMIT_STARTED
        )


@pytest.mark.parametrize(
    ("states", "expected"),
    (
        ((CommitStepState.NOT_COMMITTED,), MutationState.NOT_COMMITTED),
        ((CommitStepState.COMMITTED,), MutationState.COMMITTED),
        (
            (CommitStepState.COMMITTED, CommitStepState.NOT_COMMITTED),
            MutationState.PARTIALLY_COMMITTED,
        ),
        ((CommitStepState.UNKNOWN,), MutationState.INDETERMINATE),
    ),
)
def test_patch_phase_1_derives_every_commit_truth_vector(
    states: tuple[CommitStepState, ...], expected: MutationState
) -> None:
    """Derive all closed requested-effect mutation states mechanically."""
    truth = _truth(_journal(*states))
    assert truth.mutation_state is expected
    assert truth.requested_effect_occurred in set(RequestedEffectOccurrence)


def test_patch_phase_1_derives_request_wide_multi_lineage_truth() -> None:
    """Aggregate exact sealed graphs without losing known committed effects."""
    first = _journal(
        CommitStepState.COMMITTED,
        lineage_id=PatchLineageId("lineage_0000000000000001"),
    )
    second = _journal(
        CommitStepState.UNKNOWN,
        lineage_id=PatchLineageId("lineage_0000000000000002"),
        artifact=ArtifactState.UNKNOWN,
        postcondition=PostconditionState.UNKNOWN,
    )
    truth = _truth(first, second)
    assert truth.mutation_state is MutationState.INDETERMINATE
    assert truth.requested_effect_occurred is RequestedEffectOccurrence.TRUE
    assert truth.commit_set_exact is False
    assert truth.workspace_change is WorkspaceChange.CHANGED


def test_patch_phase_1_rejects_unsealed_or_contradictory_commit_truth() -> (
    None
):
    """Reject mismatched journal graphs and impossible derived fact vectors."""
    journal = _journal(CommitStepState.COMMITTED)
    mismatched = LineageJournal(
        lineage_id=journal.lineage_id,
        steps=(
            CommitStepJournal(
                PatchStepId("step_1111111111111111"),
                CommitStepState.COMMITTED,
            ),
        ),
        postcondition=PostconditionState.ESTABLISHED,
        artifact_state=ArtifactState.CLEANED,
    )
    with pytest.raises(PatchValidationError):
        derive_commit_truth(_plan(journal), (mismatched,))
    with pytest.raises(PatchValidationError):
        CommitTruth(
            mutation_state=MutationState.NOT_COMMITTED,
            lineage_state=LineageState.NOT_COMMITTED,
            requested_effect_occurred=RequestedEffectOccurrence.TRUE,
            artifact_state=ArtifactState.CLEANED,
            workspace_change=WorkspaceChange.CHANGED,
            commit_set_exact=True,
            postcondition=PostconditionState.ESTABLISHED,
        )


def test_patch_phase_1_supersession_does_not_rewrite_committed_history() -> (
    None
):
    """Keep an effect committed when its postcondition is superseded."""
    truth = _truth(
        _journal(
            CommitStepState.COMMITTED,
            postcondition=PostconditionState.SUPERSEDED,
        )
    )
    assert truth.mutation_state is MutationState.COMMITTED
    assert truth.postcondition is PostconditionState.SUPERSEDED
    assert truth.workspace_change is WorkspaceChange.CHANGED


def test_patch_phase_1_artifacts_remain_independent_of_requested_effects() -> (
    None
):
    """Keep cleaned, leaked, and unknown staging facts visible."""
    cleaned = _truth(_journal(CommitStepState.NOT_COMMITTED))
    leaked = _truth(
        _journal(CommitStepState.NOT_COMMITTED, artifact=ArtifactState.LEAKED)
    )
    unknown = _truth(
        _journal(CommitStepState.NOT_COMMITTED, artifact=ArtifactState.UNKNOWN)
    )
    assert cleaned.workspace_change is WorkspaceChange.UNCHANGED
    assert leaked.workspace_change is WorkspaceChange.CHANGED
    assert unknown.workspace_change is WorkspaceChange.UNKNOWN


def test_patch_phase_1_result_pending_and_status_combinations_are_closed() -> (
    None
):
    """Create valid terminal and nonterminal tagged outcomes only."""
    truth = _truth(_journal(CommitStepState.COMMITTED))
    result = PatchResult(
        1,
        _request_id(),
        _plan_id(),
        LifecyclePhase.REQUEST_COMPLETED,
        PatchStatus.COMMITTED,
        truth,
        None,
    )
    pending = PatchPending(
        1,
        PatchPendingOperationId("pending_0123456789abcdef"),
        _request_id(),
        _correlation_id(),
        LifecyclePhase.SETTLEMENT_PENDING,
    )
    assert isinstance(result, PatchResult)
    assert result.diagnostic is None
    assert isinstance(pending, PatchPending)
    with pytest.raises(PatchValidationError):
        PatchResult(
            1,
            _request_id(),
            _plan_id(),
            LifecyclePhase.REQUEST_COMPLETED,
            PatchStatus.COMMITTED,
            truth,
            _diagnostic(),
        )
    with pytest.raises(PatchValidationError):
        PatchResult(
            1,
            _request_id(),
            _plan_id(),
            LifecyclePhase.REQUEST_COMPLETED,
            PatchStatus.PARTIAL,
            truth,
            _diagnostic(),
        )


def test_patch_phase_1_event_projection() -> None:
    """Construct monotonic content-free events and a public-safe projection."""
    pending = PatchPending(
        1,
        PatchPendingOperationId("pending_0123456789abcdef"),
        _request_id(),
        _correlation_id(),
        LifecyclePhase.SETTLEMENT_PENDING,
    )
    event = PatchLifecycleEvent(
        1,
        PatchEventId("event_0123456789abcdef"),
        PatchObserverId.new(),
        _correlation_id(),
        _request_id(),
        SequenceNumber(1),
        LifecyclePhase.SETTLEMENT_PENDING,
    )
    projection = project_pending(ProjectionInput(Audience.PUBLIC, pending))
    assert event.sequence == SequenceNumber(1)
    assert projection.correlation_id == pending.correlation_id
    assert projection.pending_operation_id == pending.pending_operation_id
    assert not hasattr(projection, "request_id")
    assert not hasattr(projection, "plan_id")


def test_patch_phase_1_validation_errors_and_grants_remain_immutable() -> None:
    """Keep expected decoding errors distinct from immutable grant values."""
    grant = ApprovalGrant(
        PatchGrantId("grant_0123456789abcdef"),
        PatchApprovalId("approval_0123456789abcdef"),
        _plan_id(),
        ExpiryTick(1),
        GrantSecret(b"grant"),
    )
    assert grant.plan_id == _plan_id()
    with pytest.raises(PatchValidationError):
        Snapshot(LogicalPath("missing.txt"), False, ByteSize(1), None, None)
    with pytest.raises(FrozenInstanceError):
        setattr(grant, "expiry", ExpiryTick(2))


def test_patch_phase_1_rejects_mutable_authority_collections() -> None:
    """Reject caller-owned mutable collections at authority boundaries."""
    plan = _plan(_journal(CommitStepState.COMMITTED))
    helper = _mutable_construction_helper()
    with pytest.raises(PatchValidationError):
        helper.construct_mutable_scope(plan.scope)
    with pytest.raises(PatchValidationError):
        helper.construct_mutable_lineage(plan.lineages[0])


def test_patch_phase_1_codecs_round_trip_closed_domain_values() -> None:
    """Round-trip results, pending envelopes, diagnostics, and events."""
    journal = _journal(CommitStepState.COMMITTED)
    result = DomainFacade().settle(
        _plan(journal),
        (journal,),
        None,
    )
    pending = PatchPending(
        1,
        PatchPendingOperationId("pending_0123456789abcdef"),
        _request_id(),
        _correlation_id(),
        LifecyclePhase.SETTLEMENT_PENDING,
    )
    event = PatchLifecycleEvent(
        1,
        PatchEventId("event_0123456789abcdef"),
        PatchObserverId.new(),
        _correlation_id(),
        _request_id(),
        SequenceNumber(0),
        LifecyclePhase.RECEIVED,
    )
    assert decode_result(encode_result(result)) == result
    assert result.diagnostic is None
    assert decode_pending(encode_pending(pending)) == pending
    assert decode_event(encode_event(event)) == event
    assert decode_diagnostic(encode_diagnostic(_diagnostic())) == _diagnostic()
    projection = project_pending(ProjectionInput(Audience.PUBLIC, pending))
    assert (
        decode_public_pending(encode_public_pending(projection)) == projection
    )


def test_patch_phase_1_facade_e2e_returns_one_committed_terminal_result() -> (
    None
):
    """Feed an all-success journal through the dormant public domain facade."""
    journal = _journal(CommitStepState.COMMITTED)
    result, event = DomainFacade().settle_with_event(
        _plan(journal),
        (journal,),
        None,
        PatchEventId("event_0123456789abcdef"),
        PatchObserverId("observer_0123456789abcdef"),
        _correlation_id(),
        SequenceNumber(1),
    )
    assert result.status is PatchStatus.COMMITTED
    assert result.diagnostic is None
    assert result.lifecycle is LifecyclePhase.REQUEST_COMPLETED
    assert event.lifecycle is LifecyclePhase.REQUEST_COMPLETED


def test_patch_phase_1_facade_disjoint_truth() -> None:
    """Feed non-success histories through the same pure outcome boundary."""
    facade = DomainFacade()
    partial_journal = _journal(
        CommitStepState.COMMITTED,
        CommitStepState.NOT_COMMITTED,
        artifact=ArtifactState.LEAKED,
    )
    partial = facade.settle(
        _plan(partial_journal),
        (partial_journal,),
        _diagnostic(),
    )
    unknown_journal = _journal(
        CommitStepState.UNKNOWN, artifact=ArtifactState.UNKNOWN
    )
    unknown = facade.settle(
        _plan(unknown_journal),
        (unknown_journal,),
        _diagnostic(),
    )
    pending = facade.pending(
        _request_id(),
        PatchPendingOperationId("pending_0123456789abcdef"),
        _correlation_id(),
    )
    assert partial.status is PatchStatus.PARTIAL
    assert partial.truth.artifact_state is ArtifactState.LEAKED
    assert unknown.status is PatchStatus.INDETERMINATE
    assert pending.lifecycle is LifecyclePhase.SETTLEMENT_PENDING


def test_patch_phase_1_pending_roundtrip() -> None:
    """Restore a public-safe pending envelope without any mutation surface."""
    pending = PatchPending(
        1,
        PatchPendingOperationId("pending_0123456789abcdef"),
        _request_id(),
        _correlation_id(),
        LifecyclePhase.SETTLEMENT_PENDING,
    )
    restored = decode_pending(encode_pending(pending))
    assert restored.correlation_id == pending.correlation_id
    assert restored.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
    assert not hasattr(restored, "plan_id")


def test_patch_phase_1_error_catalog_is_complete_and_exact() -> None:
    """Freeze every stable error code from the canonical contract catalog."""
    assert {code.value for code in PatchErrorCode} == {
        "patch.invalid_request",
        "patch.invalid_patch",
        "patch.unsupported_operation",
        "patch.conflicting_operations",
        "patch.no_effect",
        "patch.limit_exceeded",
        "patch.context_unavailable",
        "patch.backend_unavailable",
        "patch.capability_unavailable",
        "patch.capability_required",
        "patch.precondition_observation_required",
        "patch.path_denied",
        "patch.traversal_denied",
        "patch.link_denied",
        "patch.alias_denied",
        "patch.mount_denied",
        "patch.special_file_denied",
        "patch.parent_missing",
        "patch.source_missing",
        "patch.destination_exists",
        "patch.metadata_unsupported",
        "patch.unsupported_content",
        "patch.encoding_unsupported",
        "patch.representation_unsupported",
        "patch.match_not_found",
        "patch.ambiguous_match",
        "patch.overlapping_edits",
        "patch.approval_required",
        "patch.approval_denied",
        "patch.approval_unavailable",
        "patch.approval_expired",
        "patch.approval_mismatch",
        "patch.stale",
        "patch.cancelled",
        "patch.timeout",
        "patch.commit_failed",
        "patch.partial_commit",
        "patch.indeterminate",
        "patch.verification_failed",
        "patch.staging_artifact_leaked",
        "patch.staging_artifact_unknown",
        "patch.diagnostic_failed",
    }


def test_patch_phase_1_rejects_invalid_domain_boundary_values() -> None:
    """Reject malformed scalar, structure, and immutable value boundaries."""

    class _PrefixlessIdentifier(_PatchIdentifier):
        """Exercise the required identity-prefix subclass contract."""

    for factory, value in (
        (ByteSize, -1),
        (SequenceNumber, -1),
        (DurationTicks, 0),
        (ExpiryTick, 0),
        (FileMode, 0o1000),
    ):
        with pytest.raises(PatchValidationError):
            factory(value)
    with pytest.raises(PatchValidationError):
        AlgorithmDigest("sha1", "0" * 64)
    invalid_source: SourceBytes = object.__new__(SourceBytes)
    object.__setattr__(invalid_source, "_value", "not-bytes")
    with pytest.raises(PatchValidationError):
        invalid_source.__post_init__()
    with pytest.raises(NotImplementedError):
        _PrefixlessIdentifier._identifier_prefix()

    journal = _journal(CommitStepState.COMMITTED)
    plan = _plan(journal)
    request = plan.request
    with pytest.raises(PatchValidationError):
        PatchRequest(
            2,
            request.request_id,
            request.execution_id,
            request.operation,
            request.input_bytes,
            request.logical_paths,
        )
    with pytest.raises(PatchValidationError):
        PatchRequest(
            1,
            request.request_id,
            request.execution_id,
            request.operation,
            request.input_bytes,
            (LogicalPath("same.txt"), LogicalPath("same.txt")),
        )
    with pytest.raises(PatchValidationError):
        replace(_limits(), input_bytes=ByteSize(0))
    with pytest.raises(PatchValidationError):
        Snapshot(LogicalPath("missing.txt"), True, ByteSize(0), None, None)
    with pytest.raises(PatchValidationError):
        CommitGraph((), AtomicityClass.SINGLE_STEP)
    with pytest.raises(PatchValidationError):
        Lineage(
            _lineage_id(),
            None,
            None,
            frozenset(),
            None,
            CommitGraph((_step_id(),), AtomicityClass.SINGLE_STEP),
        )
    with pytest.raises(PatchInvariantError):
        ReviewArtifact(
            DiffBytes(b"diff"),
            AlgorithmDigest.from_bytes(b"other"),
            ByteSize(4),
        )
    with pytest.raises(PatchValidationError):
        MutationPlan(
            plan.plan_id,
            plan.request,
            plan.scope,
            (),
            plan.review,
            plan.fingerprint,
        )
    with pytest.raises(PatchValidationError):
        LineageJournal(
            _lineage_id(),
            (),
            PostconditionState.UNKNOWN,
            ArtifactState.ABSENT,
        )


def test_patch_phase_1_rejects_impossible_truth_and_outcome_combinations() -> (
    None
):
    """Reject contradictory aggregate facts and nonterminal projections."""
    with pytest.raises(PatchValidationError):
        CommitTruth(
            MutationState.COMMITTED,
            LineageState.NOT_COMMITTED,
            RequestedEffectOccurrence.TRUE,
            ArtifactState.CLEANED,
            WorkspaceChange.CHANGED,
            True,
            PostconditionState.ESTABLISHED,
        )
    with pytest.raises(PatchValidationError):
        CommitTruth(
            MutationState.COMMITTED,
            LineageState.COMMITTED,
            RequestedEffectOccurrence.TRUE,
            ArtifactState.CLEANED,
            WorkspaceChange.CHANGED,
            False,
            PostconditionState.ESTABLISHED,
        )
    with pytest.raises(PatchValidationError):
        CommitTruth(
            MutationState.INDETERMINATE,
            LineageState.INDETERMINATE,
            RequestedEffectOccurrence.FALSE,
            ArtifactState.UNKNOWN,
            WorkspaceChange.UNKNOWN,
            False,
            PostconditionState.UNKNOWN,
        )
    with pytest.raises(PatchValidationError):
        CommitTruth(
            MutationState.NOT_COMMITTED,
            LineageState.NOT_COMMITTED,
            RequestedEffectOccurrence.FALSE,
            ArtifactState.CLEANED,
            WorkspaceChange.UNCHANGED,
            True,
            PostconditionState.ESTABLISHED,
        )
    with pytest.raises(PatchValidationError):
        CommitTruth(
            MutationState.NOT_COMMITTED,
            LineageState.NOT_COMMITTED,
            RequestedEffectOccurrence.FALSE,
            ArtifactState.CLEANED,
            WorkspaceChange.CHANGED,
            True,
            PostconditionState.UNKNOWN,
        )

    journal = _journal(CommitStepState.COMMITTED)
    result = DomainFacade().settle(_plan(journal), (journal,), None)
    pending = DomainFacade().pending(
        _request_id(),
        PatchPendingOperationId("pending_0123456789abcdef"),
        _correlation_id(),
    )
    event = PatchLifecycleEvent(
        1,
        PatchEventId("event_0123456789abcdef"),
        PatchObserverId("observer_0123456789abcdef"),
        _correlation_id(),
        _request_id(),
        SequenceNumber(1),
        LifecyclePhase.SETTLEMENT_PENDING,
    )
    with pytest.raises(PatchValidationError):
        replace(result, lifecycle=LifecyclePhase.SETTLED)
    with pytest.raises(PatchValidationError):
        replace(pending, lifecycle=LifecyclePhase.REQUEST_COMPLETED)
    with pytest.raises(PatchValidationError):
        replace(event, schema_version=2)
    with pytest.raises(PatchValidationError):
        project_pending(ProjectionInput(Audience.PUBLIC, result))


def test_patch_phase_1_derivation_binds_journals_and_tracks_artifacts() -> (
    None
):
    """Reject unsealed journals and retain each independent artifact state."""
    planned = _journal(CommitStepState.PLANNED)
    with pytest.raises(PatchValidationError):
        derive_commit_truth(_plan(planned), (planned,))

    committed = _journal(CommitStepState.COMMITTED)
    with pytest.raises(PatchValidationError):
        derive_commit_truth(_plan(committed), ())

    first = _journal(
        CommitStepState.COMMITTED,
        lineage_id=PatchLineageId("lineage_0000000000000001"),
    )
    second = _journal(
        CommitStepState.NOT_COMMITTED,
        lineage_id=PatchLineageId("lineage_0000000000000002"),
    )
    with pytest.raises(PatchValidationError):
        derive_commit_truth(_plan(first, second), (first, first))

    staged = _truth(
        _journal(CommitStepState.NOT_COMMITTED, artifact=ArtifactState.STAGED)
    )
    cleaned = _truth(
        _journal(CommitStepState.NOT_COMMITTED, artifact=ArtifactState.CLEANED)
    )
    absent = _truth(
        _journal(CommitStepState.NOT_COMMITTED, artifact=ArtifactState.ABSENT)
    )
    failed = DomainFacade().settle(
        _plan(committed),
        (_journal(CommitStepState.NOT_COMMITTED),),
        _diagnostic(),
    )
    assert staged.artifact_state is ArtifactState.STAGED
    assert cleaned.artifact_state is ArtifactState.CLEANED
    assert absent.artifact_state is ArtifactState.ABSENT
    assert failed.status is PatchStatus.COMMIT_FAILED


def test_patch_phase_1_coarsens_only_public_error_details() -> None:
    """Coarsen protected path and content categories for public audiences."""
    assert (
        coarsen_error_code(PatchErrorCode.SOURCE_MISSING, Audience.PUBLIC)
        is PatchErrorCode.PATH_DENIED
    )
    assert (
        coarsen_error_code(PatchErrorCode.UNSUPPORTED_CONTENT, Audience.MODEL)
        is PatchErrorCode.INVALID_REQUEST
    )
    assert (
        coarsen_error_code(PatchErrorCode.SOURCE_MISSING, Audience.AUDIT)
        is PatchErrorCode.SOURCE_MISSING
    )


def test_patch_phase_1_codecs_reject_noncanonical_internal_envelopes() -> None:
    """Reject malformed codec bytes while accepting canonical false values."""
    committed = _journal(CommitStepState.COMMITTED)
    result = DomainFacade().settle(_plan(committed), (committed,), None)
    unknown = _journal(
        CommitStepState.UNKNOWN,
        artifact=ArtifactState.UNKNOWN,
        postcondition=PostconditionState.UNKNOWN,
    )
    unknown_result = DomainFacade().settle(
        _plan(unknown), (unknown,), _diagnostic()
    )
    event = PatchLifecycleEvent(
        1,
        PatchEventId("event_0123456789abcdef"),
        PatchObserverId("observer_0123456789abcdef"),
        _correlation_id(),
        _request_id(),
        SequenceNumber(0),
        LifecyclePhase.RECEIVED,
    )
    with pytest.raises(ValueError):
        codec._encode(("field", "bad\nfield"))
    with pytest.raises(ValueError):
        decode_result(b"not-an-envelope")
    with pytest.raises(ValueError):
        decode_result(b"\xff")
    with pytest.raises(ValueError):
        decode_result(
            encode_result(result).replace(
                b"patch-result-v1\x1f1\x1f", b"patch-result-v1\x1f2\x1f", 1
            )
        )
    assert decode_result(encode_result(unknown_result)) == unknown_result
    with pytest.raises(ValueError):
        decode_result(
            encode_result(result).replace(
                b"\x1f\x1f\x1f", b"\x1f\x1fpatch.commit_failed\x1f", 1
            )
        )
    with pytest.raises(ValueError):
        decode_result(
            encode_result(unknown_result).replace(
                b"\x1ffalse\x1f", b"\x1fmaybe\x1f", 1
            )
        )
    with pytest.raises(ValueError):
        decode_event(
            encode_event(event).replace(b"\x1f0\x1f", b"\x1f00\x1f", 1)
        )
