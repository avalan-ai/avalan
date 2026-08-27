"""Exercise Phase 12 audit, metric, telemetry, and retention boundaries."""

from asyncio import CancelledError, Event, create_task, run
from collections.abc import Awaitable, Callable, Iterator
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, replace
from json import loads
from pathlib import Path
from pickle import dumps
from runpy import run_path
from typing import TypeAlias

import pytest

import avalan.patch.audience_projection as audience_projection
import avalan.patch.pgsql_store as pgsql_durable
from avalan.patch import sandbox_commit
from avalan.patch.audience_projection import (
    AudiencePayload,
    AudienceProjectionError,
    AudienceProjectionHost,
    PatchAudienceProjectionSource,
    create_audit_record_boundary,
    create_metrics_record_boundary,
    create_server_record_boundary,
    create_telemetry_record_boundary,
)
from avalan.patch.audience_retention import (
    AudienceRetainedValue,
    AudienceRetentionCleanupReceipt,
    AudienceRetentionClock,
    AudienceRetentionError,
    AudienceRetentionPolicy,
    AudienceRetentionReadAuthority,
    AudienceRetentionReadReceipt,
    AudienceRetentionService,
    AudienceRetentionWarning,
    AudienceRetentionWriter,
    AudienceRetentionWriteReceipt,
    _retention_audience,
)
from avalan.patch.coordinator import RetransmissionKey
from avalan.patch.domain import (
    AlgorithmDigest,
    ApprovalMode,
    ArtifactState,
    Audience,
    ByteSize,
    Capability,
    CommitStepState,
    CommitTruth,
    DurationTicks,
    ErrorStage,
    ExpiryTick,
    LifecyclePhase,
    LineageState,
    LogicalPath,
    MutationState,
    PatchDiagnostic,
    PatchErrorCode,
    PatchEventId,
    PatchObserverCorrelationId,
    PatchPendingOperationId,
    PatchPlanId,
    PatchPublicCorrelationId,
    PatchResult,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
    PatchStatus,
    PatchValidationError,
    PostconditionState,
    RequestedEffectOccurrence,
    Retryability,
    SequenceNumber,
    WorkspaceChange,
    coarsen_error_code,
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
    DurableArtifactJournalEntry,
    DurableArtifactState,
    DurableJournal,
    DurableJournalCursor,
    DurableOutboxRecord,
    DurablePendingRecord,
    DurableRequestAccess,
    DurableRequestIdentity,
    DurableRequestSnapshot,
    DurableReservation,
    DurableRetentionAccess,
    DurableRetentionCleanup,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
    DurableStepJournalEntry,
    DurableStoreError,
    DurableStoreErrorCode,
    DurableStoreLimits,
    DurableTerminalRecord,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.pgsql_store import PgsqlDurablePatchStore
from avalan.patch.planner import Match, MatchKind, TextSpan
from avalan.patch.policy import (
    CapabilityWarning,
    PolicyDisclosure,
    SealedPlan,
    cleanup_sealed_authorities,
)

_PHASE_FIVE = run_path(
    str(Path("tests/patch/phase_5_contract_test.py").resolve())
)

_DeliveryValue: TypeAlias = int | str | AudiencePayload
_Delivery: TypeAlias = dict[str, _DeliveryValue]
_RetentionRowValue: TypeAlias = bool | bytes | int | str
_RetentionRow: TypeAlias = dict[str, _RetentionRowValue]
_RetentionIdentityRow: TypeAlias = dict[str, str]
_SEAL_CLEANUP_TICK = ExpiryTick(2**63 - 1)


@pytest.fixture(autouse=True)
def _phase_twelve_seal_lifecycle() -> Iterator[None]:
    """Release test-local plan seals at each Phase 12 lifecycle boundary."""
    cleanup_sealed_authorities(_SEAL_CLEANUP_TICK)
    yield
    cleanup_sealed_authorities(_SEAL_CLEANUP_TICK)


class _ManualClock(AudienceRetentionClock):
    """Provide a manually advanced async clock for retention tests."""

    def __init__(self, value: int) -> None:
        """Initialize the current trusted tick."""
        self.value = value

    async def now(self) -> ExpiryTick:
        """Return the manually configured current tick."""
        return ExpiryTick(self.value)


class _RetentionIdSubstitute(PatchRetentionRecordId):
    """Model a same-shape identifier that lacks exact retention authority."""


class _RetainedValueSubstitute(AudienceRetainedValue):
    """Model a same-shape value that lacks exact retention authority."""


class _RetentionAccessSubstitute(DurableRetentionAccess):
    """Model a same-shape access value that lacks exact authentication."""


class _ReservationSubstitute(DurableReservation):
    """Model a same-shape reservation that lacks exact writer authority."""


class _ExpiryTickSubstitute(ExpiryTick):
    """Model a clock value that lacks exact manual-clock identity."""


class _InvalidClock(AudienceRetentionClock):
    """Return a valid subtype that fails the exact clock boundary."""

    async def now(self) -> ExpiryTick:
        """Return a non-exact expiry tick."""
        return _ExpiryTickSubstitute(1)


class _CleanupSubstitute(DurableRetentionCleanup):
    """Model cleanup accounting that lacks the exact store result type."""


class _SnapshotStore:
    """Return one host-owned snapshot only to its exact request access."""

    def __init__(
        self,
        access: DurableRequestAccess,
        snapshot: DurableRequestSnapshot,
    ) -> None:
        """Bind one exact access object to one immutable durable snapshot."""
        self.access = access
        self.snapshot = snapshot

    async def inspect(
        self, access: DurableRequestAccess
    ) -> DurableRequestSnapshot:
        """Return only the store-owned snapshot for the exact access object."""
        if access is not self.access:
            raise RuntimeError("wrong-access")
        return self.snapshot


class _AllowRetentionWriter:
    """Authorize only the configured test retention purposes."""

    async def authorize(
        self,
        reservation: DurableReservation,
        kind: DurableRetentionKind,
    ) -> bool:
        """Allow exact typed requests without exposing request data."""
        return (
            type(reservation) is DurableReservation
            and type(kind) is DurableRetentionKind
        )


class _DenyRetentionWriter:
    """Deny every write before encryption or durable persistence."""

    async def authorize(
        self,
        reservation: DurableReservation,
        kind: DurableRetentionKind,
    ) -> bool:
        """Return a bounded authorization denial."""
        del reservation, kind
        return False


class _ToggleRetentionWriter:
    """Authorize issuance once and allow tests to deny the subsequent write."""

    def __init__(self) -> None:
        """Begin with authorization enabled."""
        self.allowed = True

    async def authorize(
        self,
        reservation: DurableReservation,
        kind: DurableRetentionKind,
    ) -> bool:
        """Return the current bounded authorization decision."""
        del reservation, kind
        return self.allowed


class _FailingRetentionWriter:
    """Fail authorization without exposing its internal cause."""

    async def authorize(
        self,
        reservation: DurableReservation,
        kind: DurableRetentionKind,
    ) -> bool:
        """Raise a non-semantic authorization failure."""
        del reservation, kind
        raise RuntimeError("retention-authorizer-failure")


def _identity(plan: SealedPlan) -> DurableRequestIdentity:
    """Return the durable identity matching the sealed plan subject."""
    binding = plan.binding
    return DurableRequestIdentity(
        binding.subject.tenant,
        binding.subject.principal,
        binding.request.execution_id,
        binding.final.approval.route,
        RetransmissionKey("phase12-retention"),
    )


def _retention_policy(
    *kinds: DurableRetentionKind,
    delete_on_terminal: bool = False,
) -> AudienceRetentionPolicy:
    """Return a bounded policy-owned audience retention configuration."""
    return AudienceRetentionPolicy(
        frozenset(kinds),
        DurationTicks(4),
        delete_on_terminal,
    )


def _result(plan: SealedPlan, status: PatchStatus) -> PatchResult:
    """Return an immutable result matching one requested terminal status."""
    request = plan.binding.request
    states = {
        PatchStatus.COMMITTED: MutationState.COMMITTED,
        PatchStatus.PARTIAL: MutationState.PARTIALLY_COMMITTED,
        PatchStatus.INDETERMINATE: MutationState.INDETERMINATE,
    }
    mutation = states.get(status, MutationState.NOT_COMMITTED)
    occurrence = (
        RequestedEffectOccurrence.TRUE
        if mutation
        in {MutationState.COMMITTED, MutationState.PARTIALLY_COMMITTED}
        else (
            RequestedEffectOccurrence.UNKNOWN
            if mutation is MutationState.INDETERMINATE
            else RequestedEffectOccurrence.FALSE
        )
    )
    truth = CommitTruth(
        mutation,
        LineageState(mutation.value),
        occurrence,
        (
            ArtifactState.CLEANED
            if occurrence is RequestedEffectOccurrence.TRUE
            else ArtifactState.ABSENT
        ),
        (
            WorkspaceChange.CHANGED
            if occurrence is RequestedEffectOccurrence.TRUE
            else (
                WorkspaceChange.UNKNOWN
                if occurrence is RequestedEffectOccurrence.UNKNOWN
                else WorkspaceChange.UNCHANGED
            )
        ),
        mutation is not MutationState.INDETERMINATE,
        (
            PostconditionState.ESTABLISHED
            if occurrence is RequestedEffectOccurrence.TRUE
            else PostconditionState.UNKNOWN
        ),
    )
    diagnostic = (
        None
        if status is PatchStatus.COMMITTED
        else PatchDiagnostic(
            ErrorStage.COMMIT,
            PatchErrorCode.COMMIT_FAILED,
            Retryability.NOT_RETRYABLE,
        )
    )
    return PatchResult(
        1,
        request.request_id,
        getattr(plan, "plan_id"),
        LifecyclePhase.REQUEST_COMPLETED,
        status,
        truth,
        diagnostic,
    )


def _truth(
    mutation: MutationState,
    artifact: ArtifactState,
    postcondition: PostconditionState,
) -> CommitTruth:
    """Return one internally consistent terminal truth for matrix evidence."""
    occurrence = (
        RequestedEffectOccurrence.TRUE
        if mutation
        in {MutationState.COMMITTED, MutationState.PARTIALLY_COMMITTED}
        else (
            RequestedEffectOccurrence.UNKNOWN
            if mutation is MutationState.INDETERMINATE
            else RequestedEffectOccurrence.FALSE
        )
    )
    workspace_change = (
        WorkspaceChange.CHANGED
        if occurrence is RequestedEffectOccurrence.TRUE
        or artifact in {ArtifactState.STAGED, ArtifactState.LEAKED}
        else (
            WorkspaceChange.UNKNOWN
            if occurrence is RequestedEffectOccurrence.UNKNOWN
            or artifact is ArtifactState.UNKNOWN
            else WorkspaceChange.UNCHANGED
        )
    )
    return CommitTruth(
        mutation,
        LineageState(mutation.value),
        occurrence,
        artifact,
        workspace_change,
        mutation is not MutationState.INDETERMINATE,
        postcondition,
    )


def _terminal_result(
    plan: SealedPlan,
    status: PatchStatus,
    truth: CommitTruth,
    stage: ErrorStage,
) -> PatchResult:
    """Return one terminal record with a bounded stable error category."""
    return PatchResult(
        1,
        plan.binding.request.request_id,
        plan.plan_id,
        LifecyclePhase.REQUEST_COMPLETED,
        status,
        truth,
        (
            None
            if status is PatchStatus.COMMITTED
            else PatchDiagnostic(
                stage,
                PatchErrorCode.COMMIT_FAILED,
                Retryability.NOT_RETRYABLE,
            )
        ),
    )


async def _source(
    *,
    disclosures: frozenset[PolicyDisclosure] = frozenset(),
    mode: ApprovalMode = ApprovalMode.REQUIRE_REVIEW,
    terminal: bool = True,
    cancellation_requested: bool = False,
) -> PatchAudienceProjectionSource:
    """Return a sealed plan and matching terminal or pending durable truth."""
    plan = await _PHASE_FIVE["_sealed_plan"](
        mode=mode, disclosures=disclosures
    )
    durable_plan = sandbox_commit._durable_plan(plan)
    identity = _identity(plan)
    reservation = DurableReservation(
        plan.binding.request.request_id,
        identity,
        plan.binding.request_digest,
        False,
    )
    planned_steps = tuple(
        DurableStepJournalEntry(
            DurableJournalCursor(
                reservation.request_id, SequenceNumber(index)
            ),
            item.step_id,
            item.lineage_id,
            CommitStepState.PLANNED,
        )
        for index, item in enumerate(durable_plan.steps, start=1)
    )
    steps = planned_steps + tuple(
        DurableStepJournalEntry(
            DurableJournalCursor(
                reservation.request_id,
                SequenceNumber(len(planned_steps) + index),
            ),
            item.step_id,
            item.lineage_id,
            CommitStepState.COMMITTED,
        )
        for index, item in enumerate(durable_plan.steps, start=1)
    )
    artifact_bindings = sandbox_commit._durable_artifacts(plan)
    artifact_states = tuple(
        (binding, state)
        for binding in artifact_bindings
        for state in (
            DurableArtifactState.INTENDED,
            DurableArtifactState.PRESENT,
            DurableArtifactState.REMOVED,
        )
    )
    artifacts = tuple(
        DurableArtifactJournalEntry(
            DurableJournalCursor(
                reservation.request_id,
                SequenceNumber(len(steps) + index),
            ),
            item[1],
            state,
        )
        for index, (item, state) in enumerate(artifact_states, start=1)
    )
    journal = DurableJournal(
        DurableJournalCursor(
            reservation.request_id, SequenceNumber(len(steps) + len(artifacts))
        ),
        steps,
        artifacts,
    )
    if terminal:
        result = _result(plan, PatchStatus.COMMITTED)
        correlation = PatchObserverCorrelationId.new()
        terminal_record = DurableTerminalRecord(
            result,
            DurableOutboxRecord(
                PatchEventId.new(),
                reservation.request_id,
                SequenceNumber(1),
                LifecyclePhase.REQUEST_COMPLETED,
                correlation,
            ),
            None,
        )
        snapshot = DurableRequestSnapshot(
            reservation,
            durable_plan,
            LifecyclePhase.REQUEST_COMPLETED,
            None,
            journal,
            None,
            terminal_record,
            False,
            False,
            cancellation_requested,
            SequenceNumber(1),
        )
    else:
        correlation = PatchObserverCorrelationId.new()
        pending = DurablePendingRecord(
            reservation.request_id,
            reservation.identity.execution_id,
            PatchPendingOperationId.new(),
            correlation,
            SequenceNumber(1),
            SequenceNumber(1),
            cancellation_requested,
            DurationTicks(10),
        )
        snapshot = DurableRequestSnapshot(
            reservation,
            durable_plan,
            LifecyclePhase.SETTLEMENT_PENDING,
            None,
            journal,
            pending,
            None,
            False,
            False,
            cancellation_requested,
            SequenceNumber(1),
        )
    access = DurableRequestAccess(reservation.request_id, identity)
    host = AudienceProjectionHost(_SnapshotStore(access, snapshot))
    witness = await host.issue_access(access, correlation)
    return await host.source(plan, witness)


def _delivery(value: bytes) -> _Delivery:
    """Return the detached JSON payload for lower-boundary assertions."""
    decoded = loads(value)
    assert isinstance(decoded, dict)
    return decoded


def test_audit_metrics_telemetry_server_truth_and_privacy() -> None:
    """Project durable truth once per audience without protected canaries."""

    async def scenario() -> None:
        source = await _source(
            disclosures=frozenset(
                (
                    PolicyDisclosure.AUDIT_PATHS,
                    PolicyDisclosure.EVENT_METRICS,
                    PolicyDisclosure.AUDIT_EXACT_TRUTH,
                    PolicyDisclosure.METRICS_EXACT_TRUTH,
                    PolicyDisclosure.TELEMETRY_EXACT_TRUTH,
                    PolicyDisclosure.SERVER_EXACT_TRUTH,
                    PolicyDisclosure.DIAGNOSTIC_ASSOCIATION,
                )
            )
        )
        boundaries = (
            create_audit_record_boundary(source),
            create_metrics_record_boundary(source),
            create_telemetry_record_boundary(source),
            create_server_record_boundary(source),
        )
        values = tuple(
            _delivery(boundary.project(boundary.authority()))
            for boundary in boundaries
        )
        audit, metrics, telemetry, server = values
        audit_payload = audit["payload"]
        assert isinstance(audit_payload, dict)
        assert audit["audience"] == "audit"
        assert audit_payload["terminal"] is True
        assert audit_payload["matching_exact"] is True
        assert audit_payload["commit_set_exact"] is True
        assert audit_payload["paths_redacted"] is False
        assert audit_payload["authenticated"] == {
            "tenant": source.plan.binding.subject.tenant.value,
            "principal": source.plan.binding.subject.principal.value,
            "run": source.plan.binding.subject.run.value,
            "session": source.plan.binding.subject.session.value,
            "task": source.plan.binding.subject.task.value,
            "agent": source.plan.binding.subject.agent.value,
        }
        lineages = audit_payload["lineages"]
        assert isinstance(lineages, list)
        assert len(lineages) == len(source.plan.candidate.lineages)
        assert all(
            "source_path" in item and "destination_path" in item
            for item in lineages
        )
        assert all(
            "move" not in item["operation_classes"] for item in lineages
        )
        for value in (metrics, telemetry, server):
            payload = value["payload"]
            assert isinstance(payload, dict)
            assert "authenticated" not in payload
            assert "context" not in payload
            assert "source_path" not in str(value)
            assert payload["matching_exact"] is True
            assert payload["commit_set_exact"] is True
            assert "lineages" in payload
        server_payload = server["payload"]
        assert isinstance(server_payload, dict)
        assert server_payload["server_activation"] == "absent"
        assert len({item["correlation_id"] for item in values}) == 4
        protected = (
            source.plan.binding.request.input_bytes._value.decode(),
            source.plan.review.diff.diff._value.decode(),
            source.plan.review.diff.digest.value,
            source.plan.fingerprint._value.hex(),
        )
        for value in values[1:]:
            encoded = str(value)
            assert all(canary not in encoded for canary in protected)
            assert "source_digest" not in encoded
            assert "terminal_digest" not in encoded
            assert "issuer_id" not in encoded

    run(scenario())


def test_default_lower_audiences_coarsen_exact_and_existence_truth() -> None:
    """Keep default lower records free of exact matching and source oracles."""

    async def scenario() -> None:
        source = await _source(
            disclosures=frozenset((PolicyDisclosure.EVENT_METRICS,))
        )
        assert source._snapshot.terminal is not None
        object.__setattr__(
            source._snapshot.terminal.result,
            "diagnostic",
            PatchDiagnostic(
                ErrorStage.PREFLIGHT,
                PatchErrorCode.SOURCE_MISSING,
                Retryability.NOT_RETRYABLE,
            ),
        )
        compatible = replace(
            source.plan.candidate.lineages[0],
            matches=(
                Match(
                    MatchKind.NEWLINE_COMPATIBLE,
                    TextSpan(0, 1, 0, 1),
                ),
            ),
        )
        object.__setattr__(
            source.plan,
            "candidate",
            replace(source.plan.candidate, lineages=(compatible,)),
        )
        audit = create_audit_record_boundary(source)
        metrics = create_metrics_record_boundary(source)
        telemetry = create_telemetry_record_boundary(source)
        server = create_server_record_boundary(source)
        audit_payload = _delivery(audit.project(audit.authority()))["payload"]
        assert isinstance(audit_payload, dict)
        assert "matching_exact" not in audit_payload
        assert "commit_set_exact" not in audit_payload
        assert "error_code" not in audit_payload
        assert "error_stage" not in audit_payload
        for boundary in (metrics, telemetry, server):
            payload = _delivery(boundary.project(boundary.authority()))[
                "payload"
            ]
            assert isinstance(payload, dict)
            assert (
                payload["error_category"] == PatchErrorCode.PATH_DENIED.value
            )
            assert "matching_exact" not in payload
            assert "commit_set_exact" not in payload
            assert "status" not in payload
            assert "mutation_state" not in payload
            assert "postcondition" not in payload
            assert "error_code" not in payload
            assert "error_stage" not in payload
            assert "lineages" not in payload
            assert "source_missing" not in str(payload)

    run(scenario())


def test_exact_truth_disclosures_are_one_audience_at_a_time() -> None:
    """Keep exact truth unavailable outside its sealed audience capability."""

    async def scenario() -> None:
        boundaries = (
            (
                PolicyDisclosure.AUDIT_EXACT_TRUTH,
                create_audit_record_boundary,
                "audit",
            ),
            (
                PolicyDisclosure.METRICS_EXACT_TRUTH,
                create_metrics_record_boundary,
                "metrics",
            ),
            (
                PolicyDisclosure.TELEMETRY_EXACT_TRUTH,
                create_telemetry_record_boundary,
                "telemetry",
            ),
            (
                PolicyDisclosure.SERVER_EXACT_TRUTH,
                create_server_record_boundary,
                "server",
            ),
        )
        for disclosure, selected, selected_name in boundaries:
            source = await _source(disclosures=frozenset((disclosure,)))
            projections = {
                "audit": create_audit_record_boundary(source),
                "metrics": create_metrics_record_boundary(source),
                "telemetry": create_telemetry_record_boundary(source),
                "server": create_server_record_boundary(source),
            }
            for name, boundary in projections.items():
                payload = _delivery(boundary.project(boundary.authority()))[
                    "payload"
                ]
                assert isinstance(payload, dict)
                if name == selected_name:
                    assert "matching_exact" in payload
                    assert "commit_set_exact" in payload
                else:
                    assert "matching_exact" not in payload
                    assert "commit_set_exact" not in payload

    run(scenario())


def test_public_error_coarsening_has_no_source_or_state_oracle() -> None:
    """Map every lower-audience error code to one non-oracular category."""
    for code in PatchErrorCode:
        assert (
            coarsen_error_code(code, Audience.PUBLIC)
            is PatchErrorCode.PATH_DENIED
        )
        assert (
            coarsen_error_code(code, Audience.MODEL)
            is PatchErrorCode.PATH_DENIED
        )
        assert coarsen_error_code(code, Audience.AUDIT) is code
        assert coarsen_error_code(code, Audience.APPROVER) is code


def test_unauthorized_projection_and_failure_surfaces_exclude_canaries(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep exact content facts and injected failures out of lower surfaces."""

    async def scenario() -> None:
        source = await _source()
        lineage = source.plan.candidate.lineages[0]
        protected = (
            source.plan.binding.request.input_bytes._value.decode("utf-8"),
            lineage.initial.bytes_value._value.decode("utf-8"),
            lineage.final.bytes_value._value.decode("utf-8"),
            source.plan.review.diff.diff._value.decode("utf-8"),
            lineage.initial.digest.value,
            lineage.final.digest.value,
            source.plan.review.diff.digest.value,
            source.plan.fingerprint._value.hex(),
            lineage.lineage_id.value,
            lineage.source_path.value,
            lineage.destination_path.value,
        )
        audit = create_audit_record_boundary(source)
        metrics = create_metrics_record_boundary(source)
        telemetry = create_telemetry_record_boundary(source)
        server = create_server_record_boundary(source)
        deliveries = (
            audit.project(audit.authority()),
            metrics.project(metrics.authority()),
            telemetry.project(telemetry.authority()),
            server.project(server.authority()),
        )
        for delivery in deliveries:
            rendered = delivery.decode("utf-8")
            assert all(canary not in rendered for canary in protected)
            assert all(
                field not in rendered
                for field in (
                    "digest",
                    "size",
                    "temporary",
                    "credential",
                    "grant",
                    "match_kind",
                    "match_span",
                    "timing",
                )
            )

        class FailingStore:
            """Raise every protected canary through the host read seam."""

            async def inspect(
                self, access: DurableRequestAccess
            ) -> DurableRequestSnapshot:
                """Fail before returning an untrusted durable snapshot."""
                del access
                raise RuntimeError("\n".join(protected))

        with caplog.at_level("DEBUG"):
            with pytest.raises(AudienceProjectionError) as raised:
                await AudienceProjectionHost(FailingStore()).issue_access(
                    source._access._access,
                    source._access._correlation,
                )
        rendered_error = f"{raised.value!s}\n{raised.value!r}\n{caplog.text}"
        assert all(canary not in rendered_error for canary in protected)
        assert raised.value.__cause__ is None

    run(scenario())


def test_projection_durable_history_validation_rejects_malformed_truth() -> (
    None
):
    """Reject malformed terminal, pending, step, and artifact histories."""

    async def scenario() -> None:
        malformed_terminal = await _source()
        assert malformed_terminal._snapshot.terminal is not None
        object.__setattr__(
            malformed_terminal._snapshot.terminal.result,
            "truth",
            object(),
        )
        with pytest.raises(AudienceProjectionError, match="source is invalid"):
            audience_projection._validate_source_truth(
                malformed_terminal.plan,
                malformed_terminal._snapshot,
            )

        malformed_pending = await _source(terminal=False)
        assert malformed_pending._snapshot.pending is not None
        object.__setattr__(
            malformed_pending._snapshot.pending,
            "next_check_after",
            object(),
        )
        with pytest.raises(AudienceProjectionError, match="source is invalid"):
            audience_projection._validate_source_truth(
                malformed_pending.plan,
                malformed_pending._snapshot,
            )

        wrong_pending_lifecycle = await _source(terminal=False)
        object.__setattr__(
            wrong_pending_lifecycle._snapshot,
            "lifecycle",
            LifecyclePhase.COMMIT_STARTED,
        )
        with pytest.raises(AudienceProjectionError, match="lifecycle"):
            audience_projection._validate_source_truth(
                wrong_pending_lifecycle.plan,
                wrong_pending_lifecycle._snapshot,
            )

        invalid_journal = await _source()
        assert invalid_journal._snapshot.plan is not None
        object.__setattr__(invalid_journal._snapshot, "journal", object())
        with pytest.raises(AudienceProjectionError, match="journal"):
            audience_projection._complete_journal_mutation_state(
                invalid_journal._snapshot,
                invalid_journal._snapshot.plan,
            )

        incomplete_steps = await _source()
        assert incomplete_steps._snapshot.plan is not None
        object.__setattr__(
            incomplete_steps._snapshot.journal,
            "steps",
            incomplete_steps._snapshot.journal.steps[:1],
        )
        with pytest.raises(AudienceProjectionError, match="incomplete"):
            audience_projection._complete_journal_mutation_state(
                incomplete_steps._snapshot,
                incomplete_steps._snapshot.plan,
            )

        unplanned_steps = await _source()
        assert unplanned_steps._snapshot.plan is not None
        object.__setattr__(
            unplanned_steps._snapshot.journal.steps[0],
            "state",
            CommitStepState.COMMITTED,
        )
        with pytest.raises(AudienceProjectionError, match="incomplete"):
            audience_projection._complete_journal_mutation_state(
                unplanned_steps._snapshot,
                unplanned_steps._snapshot.plan,
            )

        invalid_binding = await _source()
        assert invalid_binding._snapshot.plan is not None
        object.__setattr__(
            invalid_binding._snapshot.journal.steps[0],
            "lineage_id",
            invalid_binding._snapshot.journal.steps[0].lineage_id.new(),
        )
        with pytest.raises(
            AudienceProjectionError, match="journal is invalid"
        ):
            audience_projection._complete_journal_mutation_state(
                invalid_binding._snapshot,
                invalid_binding._snapshot.plan,
            )

        repeated_terminal = await _source()
        assert repeated_terminal._snapshot.plan is not None
        terminal_entry = repeated_terminal._snapshot.journal.steps[1]
        object.__setattr__(
            repeated_terminal._snapshot.journal,
            "steps",
            repeated_terminal._snapshot.journal.steps
            + (
                DurableStepJournalEntry(
                    DurableJournalCursor(
                        repeated_terminal._snapshot.reservation.request_id,
                        SequenceNumber(3),
                    ),
                    terminal_entry.step_id,
                    terminal_entry.lineage_id,
                    CommitStepState.COMMITTED,
                ),
            ),
        )
        with pytest.raises(
            AudienceProjectionError, match="journal is invalid"
        ):
            audience_projection._complete_journal_mutation_state(
                repeated_terminal._snapshot,
                repeated_terminal._snapshot.plan,
            )

        planned_only = await _source()
        assert planned_only._snapshot.plan is not None
        object.__setattr__(
            planned_only._snapshot.journal.steps[1],
            "state",
            CommitStepState.PLANNED,
        )
        with pytest.raises(
            AudienceProjectionError, match="journal is invalid"
        ):
            audience_projection._complete_journal_mutation_state(
                planned_only._snapshot,
                planned_only._snapshot.plan,
            )

        for state, expected in (
            (CommitStepState.NOT_COMMITTED, MutationState.NOT_COMMITTED),
            (CommitStepState.UNKNOWN, MutationState.INDETERMINATE),
        ):
            terminal_state = await _source()
            assert terminal_state._snapshot.plan is not None
            object.__setattr__(
                terminal_state._snapshot.journal.steps[1], "state", state
            )
            assert (
                audience_projection._complete_journal_mutation_state(
                    terminal_state._snapshot,
                    terminal_state._snapshot.plan,
                )
                is expected
            )

        partial = await _source()
        assert partial._snapshot.plan is not None
        binding = partial._snapshot.plan.steps[0]
        additional = replace(binding, step_id=binding.step_id.new())
        expanded_plan = replace(
            partial._snapshot.plan,
            steps=(binding, additional),
        )
        entries = (
            DurableStepJournalEntry(
                DurableJournalCursor(
                    partial._snapshot.reservation.request_id,
                    SequenceNumber(1),
                ),
                binding.step_id,
                binding.lineage_id,
                CommitStepState.PLANNED,
            ),
            DurableStepJournalEntry(
                DurableJournalCursor(
                    partial._snapshot.reservation.request_id,
                    SequenceNumber(2),
                ),
                binding.step_id,
                binding.lineage_id,
                CommitStepState.COMMITTED,
            ),
            DurableStepJournalEntry(
                DurableJournalCursor(
                    partial._snapshot.reservation.request_id,
                    SequenceNumber(3),
                ),
                additional.step_id,
                additional.lineage_id,
                CommitStepState.PLANNED,
            ),
            DurableStepJournalEntry(
                DurableJournalCursor(
                    partial._snapshot.reservation.request_id,
                    SequenceNumber(4),
                ),
                additional.step_id,
                additional.lineage_id,
                CommitStepState.NOT_COMMITTED,
            ),
        )
        object.__setattr__(partial._snapshot.journal, "steps", entries)
        assert (
            audience_projection._complete_journal_mutation_state(
                partial._snapshot, expanded_plan
            )
            is MutationState.PARTIALLY_COMMITTED
        )

        incomplete_artifacts = await _source()
        object.__setattr__(
            incomplete_artifacts._snapshot.journal,
            "artifacts",
            incomplete_artifacts._snapshot.journal.artifacts[:1],
        )
        with pytest.raises(AudienceProjectionError, match="incomplete"):
            audience_projection._complete_journal_artifact_state(
                incomplete_artifacts.plan, incomplete_artifacts._snapshot
            )

        mismatched_artifacts = await _source()
        object.__setattr__(
            mismatched_artifacts._snapshot.journal.artifacts[0],
            "artifact_id",
            mismatched_artifacts._snapshot.journal.artifacts[
                0
            ].artifact_id.new(),
        )
        with pytest.raises(AudienceProjectionError, match="incomplete"):
            audience_projection._complete_journal_artifact_state(
                mismatched_artifacts.plan, mismatched_artifacts._snapshot
            )

        invalid_artifacts = await _source()
        object.__setattr__(
            invalid_artifacts._snapshot.journal.artifacts[1],
            "state",
            DurableArtifactState.REMOVED,
        )
        with pytest.raises(
            AudienceProjectionError, match="journal is invalid"
        ):
            audience_projection._complete_journal_artifact_state(
                invalid_artifacts.plan, invalid_artifacts._snapshot
            )

        disclosure = await _source()
        with pytest.raises(AudienceProjectionError, match="disclosure"):
            audience_projection._exact_truth_authorized(disclosure, object())
        pending_exact = await _source(
            disclosures=frozenset(
                (
                    PolicyDisclosure.METRICS_EXACT_TRUTH,
                    PolicyDisclosure.EVENT_METRICS,
                )
            ),
            terminal=False,
        )
        payload = audience_projection._metrics_body(pending_exact)
        lineages = payload["lineages"]
        assert isinstance(lineages, tuple)
        assert "aggregate_postcondition" not in lineages[0]

    run(scenario())


def test_pgsql_audience_retention_read_requires_exact_kind_and_audience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise PostgreSQL exact-kind reader authorization without a server."""

    class Cursor:
        """Return the configured retention row to the parameterized query."""

        def __init__(self, row: _RetentionRow | None) -> None:
            """Store the next query row without exposing plaintext."""
            self.row = row

        async def execute(self, statement: str, parameters: object) -> None:
            """Accept the closed SQL statement and typed parameters."""
            del statement, parameters

        async def fetchone(self) -> _RetentionRow | None:
            """Return the configured row once."""
            return self.row

    async def scenario() -> None:
        source = await _source()
        reservation = source._snapshot.reservation
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, reservation.identity)
        )
        key = DurableRetentionKey(
            PatchRetentionKeyId("retention_" + "a" * 16), b"p" * 32
        )
        cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
        )
        retention_id = PatchRetentionRecordId("retained_" + "a" * 16)
        sealed = await cipher.seal(
            b"pgsql-audience-canary",
            DurableRetentionBinding(
                reservation.request_id,
                retention_id,
                DurableRetentionKind.AUDIT_PROJECTION,
            ),
        )
        record = DurableRetentionRecord(
            retention_id,
            DurableRetentionKind.AUDIT_PROJECTION,
            sealed.key_id,
            sealed.value,
            DurableRetentionPolicy(ExpiryTick(10), False),
        )
        row: _RetentionRow = {
            "retention_id": record.retention_id.value,
            "kind": record.kind.value,
            "key_id": record.key_id.value,
            "ciphertext": record.value._ciphertext,
            "ciphertext_digest": record.value.digest().value,
            "expires_at": record.policy.expires_at.value,
            "delete_on_terminal": record.policy.delete_on_terminal,
        }
        identity = reservation.identity
        identity_row: _RetentionIdentityRow = {
            "tenant_id": identity.tenant_id.value,
            "principal_id": identity.principal_id.value,
            "execution_id": identity.execution_id.value,
            "route_id": identity.route_id.value,
            "retransmission_key": identity.retransmission_key.value,
        }
        cursor = Cursor(row)

        async def transaction(
            operation: str,
            callback: Callable[[object], Awaitable[object]],
        ) -> object:
            """Run one store transaction against the configured fake cursor."""
            del operation
            return await callback(cursor)

        async def select_access(
            value: object, request: DurableRequestAccess
        ) -> _RetentionIdentityRow:
            """Return the exact authenticated identity row."""
            del value
            assert request is access.request
            return identity_row

        store = PgsqlDurablePatchStore(
            type("Pool", (), {"connection": lambda self: None})(),
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.AUDIT,))
            ),
            retention_validator=AesGcmDurableRetentionEnvelopeValidator(
                cipher
            ),
        )
        monkeypatch.setattr(store, "_transaction", transaction)
        monkeypatch.setattr(
            pgsql_durable, "_select_access_for_update", select_access
        )
        assert (
            await store.get_retention_for_audience(
                access,
                retention_id,
                DurableRetentionKind.AUDIT_PROJECTION,
                Audience.AUDIT,
                ExpiryTick(1),
            )
            == record
        )
        for kind, audience, now in (
            (
                DurableRetentionKind.METRICS_PROJECTION,
                Audience.PUBLIC,
                ExpiryTick(1),
            ),
            (
                DurableRetentionKind.AUDIT_PROJECTION,
                Audience.PUBLIC,
                ExpiryTick(1),
            ),
            (
                DurableRetentionKind.AUDIT_PROJECTION,
                Audience.AUDIT,
                ExpiryTick(10),
            ),
        ):
            cursor = Cursor(row)
            with pytest.raises(DurableStoreError) as raised:
                await store.get_retention_for_audience(
                    access, retention_id, kind, audience, now
                )
            assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        cursor = Cursor(None)
        with pytest.raises(DurableStoreError) as raised:
            await store.get_retention_for_audience(
                access,
                retention_id,
                DurableRetentionKind.AUDIT_PROJECTION,
                Audience.AUDIT,
                ExpiryTick(1),
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

    run(scenario())


def test_host_reads_store_truth_and_rejects_forged_bindings() -> None:
    """Reject forged projection construction inputs."""

    async def scenario() -> None:
        source = await _source()
        access = source._access._access
        correlation = source._access._correlation
        host = AudienceProjectionHost(source._access._store)
        assert repr(source._access) == "AudienceProjectionAccess(<opaque>)"
        for method in (
            source._access.__copy__,
            lambda: source._access.__deepcopy__({}),
            source._access.__reduce__,
            lambda: source._access.__reduce_ex__(4),
        ):
            with pytest.raises(AudienceProjectionError):
                method()
        with pytest.raises(AudienceProjectionError, match="host is invalid"):
            AudienceProjectionHost(object())
        with pytest.raises(AudienceProjectionError, match="access is invalid"):
            await host.issue_access(object(), correlation)
        issued = await host.issue_access(access, correlation)
        rebuilt = await host.source(source.plan, issued)
        assert rebuilt.plan is source.plan
        assert rebuilt._snapshot is source._snapshot
        with pytest.raises(AudienceProjectionError, match="host-issued"):
            audience_projection.AudienceProjectionAccess(object())
        with pytest.raises(AudienceProjectionError, match="unavailable"):
            await host.issue_access(
                DurableRequestAccess(access.request_id.new(), access.identity),
                correlation,
            )
        with pytest.raises(AudienceProjectionError, match="correlation"):
            await host.issue_access(access, PatchObserverCorrelationId.new())
        unavailable_store = _SnapshotStore(access, source._snapshot)
        unavailable_host = AudienceProjectionHost(unavailable_store)
        unavailable = await unavailable_host.issue_access(access, correlation)
        object.__setattr__(unavailable_store, "inspect", None)
        with pytest.raises(
            AudienceProjectionError, match="source is unavailable"
        ):
            await unavailable_host.source(source.plan, unavailable)
        other_store = _SnapshotStore(access, source._snapshot)
        other_host = AudienceProjectionHost(other_store)
        other_access = await other_host.issue_access(access, correlation)
        with pytest.raises(AudienceProjectionError, match="source is invalid"):
            await host.source(source.plan, other_access)
        forged = await _source()
        object.__setattr__(
            forged._snapshot,
            "reservation",
            replace(
                forged._snapshot.reservation,
                request_id=forged._access._access.request_id.new(),
            ),
        )
        forged_host = AudienceProjectionHost(forged._access._store)
        with pytest.raises(AudienceProjectionError, match="store truth"):
            await forged_host.issue_access(
                forged._access._access,
                forged._access._correlation,
            )
        forged_status = await _source()
        assert forged_status._snapshot.terminal is not None
        object.__setattr__(
            forged_status._snapshot.terminal,
            "result",
            _terminal_result(
                forged_status.plan,
                PatchStatus.COMMIT_FAILED,
                _truth(
                    MutationState.NOT_COMMITTED,
                    ArtifactState.ABSENT,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.COMMIT,
            ),
        )
        status_host = AudienceProjectionHost(forged_status._access._store)
        status_access = await status_host.issue_access(
            forged_status._access._access,
            forged_status._access._correlation,
        )
        with pytest.raises(AudienceProjectionError, match="terminal"):
            await status_host.source(forged_status.plan, status_access)

        for same_request in (True, False):
            terminal_with_pending = await _source()
            terminal_pending_host = AudienceProjectionHost(
                terminal_with_pending._access._store
            )
            terminal_pending_access = await terminal_pending_host.issue_access(
                terminal_with_pending._access._access,
                terminal_with_pending._access._correlation,
            )
            reservation = terminal_with_pending._snapshot.reservation
            terminal_record = terminal_with_pending._snapshot.terminal
            assert terminal_record is not None
            pending_request_id = (
                reservation.request_id
                if same_request
                else reservation.request_id.new()
            )
            pending = DurablePendingRecord(
                pending_request_id,
                reservation.identity.execution_id,
                PatchPendingOperationId.new(),
                terminal_with_pending._access._correlation,
                SequenceNumber(1),
                SequenceNumber(1),
                False,
                DurationTicks(10),
            )
            object.__setattr__(
                terminal_with_pending._snapshot, "pending", pending
            )
            with pytest.raises(AudienceProjectionError, match="store truth"):
                await terminal_pending_host.issue_access(
                    terminal_with_pending._access._access,
                    terminal_with_pending._access._correlation,
                )
            with pytest.raises(AudienceProjectionError, match="store truth"):
                await terminal_pending_host.source(
                    terminal_with_pending.plan, terminal_pending_access
                )
            assert terminal_with_pending._snapshot.terminal is terminal_record

        malformed_terminal_pending = await _source()
        malformed_pending_host = AudienceProjectionHost(
            malformed_terminal_pending._access._store
        )
        malformed_pending_access = await malformed_pending_host.issue_access(
            malformed_terminal_pending._access._access,
            malformed_terminal_pending._access._correlation,
        )
        object.__setattr__(
            malformed_terminal_pending._snapshot, "pending", object()
        )
        with pytest.raises(AudienceProjectionError, match="store truth"):
            await malformed_pending_host.issue_access(
                malformed_terminal_pending._access._access,
                malformed_terminal_pending._access._correlation,
            )
        with pytest.raises(AudienceProjectionError, match="store truth"):
            await malformed_pending_host.source(
                malformed_terminal_pending.plan, malformed_pending_access
            )
        assert malformed_terminal_pending._snapshot.terminal is not None

        for field in ("fingerprint_digest", "review_digest"):
            digest_forgery = await _source()
            assert digest_forgery._snapshot.plan is not None
            object.__setattr__(
                digest_forgery._snapshot,
                "plan",
                replace(
                    digest_forgery._snapshot.plan,
                    **{field: AlgorithmDigest("sha256", "f" * 64)},
                ),
            )
            digest_host = AudienceProjectionHost(digest_forgery._access._store)
            digest_access = await digest_host.issue_access(
                digest_forgery._access._access,
                digest_forgery._access._correlation,
            )
            with pytest.raises(
                AudienceProjectionError, match="does not match"
            ):
                await digest_host.source(digest_forgery.plan, digest_access)

        step_forgery = await _source()
        assert step_forgery._snapshot.plan is not None
        first_binding = step_forgery._snapshot.plan.steps[0]
        object.__setattr__(
            step_forgery._snapshot,
            "plan",
            replace(
                step_forgery._snapshot.plan,
                steps=(
                    replace(
                        first_binding,
                        step_id=first_binding.step_id.new(),
                    ),
                ),
            ),
        )
        step_host = AudienceProjectionHost(step_forgery._access._store)
        step_access = await step_host.issue_access(
            step_forgery._access._access,
            step_forgery._access._correlation,
        )
        with pytest.raises(AudienceProjectionError, match="does not match"):
            await step_host.source(step_forgery.plan, step_access)

        journal_forgery = await _source()
        journal_host = AudienceProjectionHost(journal_forgery._access._store)
        journal_access = await journal_host.issue_access(
            journal_forgery._access._access,
            journal_forgery._access._correlation,
        )
        object.__setattr__(
            journal_forgery._snapshot.journal,
            "steps",
            journal_forgery._snapshot.journal.steps[1:],
        )
        with pytest.raises(AudienceProjectionError, match="store truth"):
            await journal_host.issue_access(
                journal_forgery._access._access,
                journal_forgery._access._correlation,
            )
        with pytest.raises(AudienceProjectionError, match="store truth"):
            await journal_host.source(journal_forgery.plan, journal_access)

        outbox_forgery = await _source()
        assert outbox_forgery._snapshot.terminal is not None
        object.__setattr__(
            outbox_forgery._snapshot.terminal,
            "outbox",
            replace(
                outbox_forgery._snapshot.terminal.outbox,
                sequence=SequenceNumber(2),
            ),
        )
        outbox_host = AudienceProjectionHost(outbox_forgery._access._store)
        outbox_access = await outbox_host.issue_access(
            outbox_forgery._access._access,
            outbox_forgery._access._correlation,
        )
        with pytest.raises(AudienceProjectionError, match="terminal"):
            await outbox_host.source(outbox_forgery.plan, outbox_access)

    run(scenario())


def test_projection_store_malformed_truth_is_bounded_before_delivery(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Coarsen malformed store shapes and faults before projection issuance."""
    canary = "projection-store-canary"

    def assert_bounded(error: AudienceProjectionError) -> None:
        """Require one redacted boundary error without an exception cause."""
        assert type(error) is AudienceProjectionError
        assert str(error) in {
            "projection store truth is invalid",
            "projection access is unavailable",
            "projection source is unavailable",
            "projection source is invalid",
        }
        assert canary not in str(error)
        assert canary not in repr(error)
        assert error.__cause__ is None

    async def scenario() -> None:
        mutations: tuple[
            tuple[str, Callable[[DurableRequestSnapshot], None]], ...
        ] = (
            (
                "terminal",
                lambda snapshot: object.__setattr__(
                    snapshot, "terminal", object()
                ),
            ),
            (
                "result",
                lambda snapshot: object.__setattr__(
                    snapshot.terminal, "result", object()
                ),
            ),
            (
                "outbox",
                lambda snapshot: object.__setattr__(
                    snapshot.terminal, "outbox", object()
                ),
            ),
            (
                "result_truth",
                lambda snapshot: object.__setattr__(
                    snapshot.terminal.result, "truth", object()
                ),
            ),
            (
                "outbox_sequence",
                lambda snapshot: object.__setattr__(
                    snapshot.terminal.outbox, "sequence", object()
                ),
            ),
            (
                "pending",
                lambda snapshot: object.__setattr__(
                    snapshot, "pending", object()
                ),
            ),
            (
                "pending_request",
                lambda snapshot: object.__setattr__(
                    snapshot.pending, "request_id", object()
                ),
            ),
            (
                "plan",
                lambda snapshot: object.__setattr__(
                    snapshot, "plan", object()
                ),
            ),
            (
                "plan_steps",
                lambda snapshot: object.__setattr__(
                    snapshot.plan, "steps", object()
                ),
            ),
            (
                "plan_step",
                lambda snapshot: object.__setattr__(
                    snapshot.plan.steps[0], "step_id", object()
                ),
            ),
            (
                "journal",
                lambda snapshot: object.__setattr__(
                    snapshot, "journal", object()
                ),
            ),
            (
                "journal_steps",
                lambda snapshot: object.__setattr__(
                    snapshot.journal, "steps", object()
                ),
            ),
            (
                "journal_step",
                lambda snapshot: object.__setattr__(
                    snapshot.journal.steps[0], "cursor", object()
                ),
            ),
            (
                "artifact_container",
                lambda snapshot: object.__setattr__(
                    snapshot.journal, "artifacts", object()
                ),
            ),
            (
                "artifact_entry",
                lambda snapshot: object.__setattr__(
                    snapshot.journal, "artifacts", (object(),)
                ),
            ),
        )
        for name, mutate in mutations:
            source = await _source(terminal=not name.startswith("pending"))
            host = AudienceProjectionHost(source._access._store)
            authority = await host.issue_access(
                source._access._access, source._access._correlation
            )
            mutate(source._snapshot)
            terminal = source._snapshot.terminal
            with pytest.raises(AudienceProjectionError) as issued:
                await host.issue_access(
                    source._access._access, source._access._correlation
                )
            assert_bounded(issued.value)
            with pytest.raises(AudienceProjectionError) as sourced:
                await host.source(source.plan, authority)
            assert_bounded(sourced.value)
            assert source._snapshot.terminal is terminal, name

        malformed_snapshot = await _source()
        malformed_snapshot_host = AudienceProjectionHost(
            malformed_snapshot._access._store
        )
        malformed_snapshot_access = await malformed_snapshot_host.issue_access(
            malformed_snapshot._access._access,
            malformed_snapshot._access._correlation,
        )
        object.__setattr__(
            malformed_snapshot._access._store, "snapshot", object()
        )
        with pytest.raises(AudienceProjectionError) as issued:
            await malformed_snapshot_host.issue_access(
                malformed_snapshot._access._access,
                malformed_snapshot._access._correlation,
            )
        assert_bounded(issued.value)
        with pytest.raises(AudienceProjectionError) as sourced:
            await malformed_snapshot_host.source(
                malformed_snapshot.plan, malformed_snapshot_access
            )
        assert_bounded(sourced.value)

        invalid_plan = await _source()
        with pytest.raises(AudienceProjectionError) as raised:
            audience_projection._validate_source_truth(
                object(), invalid_plan._snapshot
            )
        assert_bounded(raised.value)

        failures: tuple[Exception, ...] = (
            AttributeError(canary),
            TypeError(canary),
            ValueError(canary),
            AssertionError(canary),
            PatchValidationError(canary),
        )
        for failure in failures:
            source = await _source()
            host = AudienceProjectionHost(source._access._store)
            authority = await host.issue_access(
                source._access._access, source._access._correlation
            )

            async def fail(
                access: DurableRequestAccess,
            ) -> DurableRequestSnapshot:
                """Raise one untrusted store fault without returning truth."""
                del access
                raise failure

            object.__setattr__(source._access._store, "inspect", fail)
            with pytest.raises(AudienceProjectionError) as issued:
                await host.issue_access(
                    source._access._access, source._access._correlation
                )
            assert_bounded(issued.value)
            with pytest.raises(AudienceProjectionError) as sourced:
                await host.source(source.plan, authority)
            assert_bounded(sourced.value)

        for interruption in (
            CancelledError(),
            SystemExit(),
            KeyboardInterrupt(),
        ):
            source = await _source()
            host = AudienceProjectionHost(source._access._store)
            authority = await host.issue_access(
                source._access._access, source._access._correlation
            )

            async def interrupt(
                access: DurableRequestAccess,
            ) -> DurableRequestSnapshot:
                """Propagate one control-flow interruption without truth."""
                del access
                raise interruption

            object.__setattr__(source._access._store, "inspect", interrupt)
            with pytest.raises(type(interruption)):
                await host.issue_access(
                    source._access._access, source._access._correlation
                )
            with pytest.raises(type(interruption)):
                await host.source(source.plan, authority)

        assert canary not in caplog.text

    run(scenario())


def test_patch_e2e_025_audience_privacy_result_matrix() -> None:
    """Project every durable outcome to its authorized data-only audience."""

    async def scenario() -> None:
        matrix = (
            (
                PatchStatus.REJECTED,
                _truth(
                    MutationState.NOT_COMMITTED,
                    ArtifactState.ABSENT,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.INPUT,
                "not_observed",
            ),
            (
                PatchStatus.DENIED,
                _truth(
                    MutationState.NOT_COMMITTED,
                    ArtifactState.STAGED,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.SCOPE,
                "not_observed",
            ),
            (
                PatchStatus.APPROVAL_DENIED,
                _truth(
                    MutationState.NOT_COMMITTED,
                    ArtifactState.CLEANED,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.APPROVAL,
                "denied",
            ),
            (
                PatchStatus.APPROVAL_UNAVAILABLE,
                _truth(
                    MutationState.NOT_COMMITTED,
                    ArtifactState.LEAKED,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.APPROVAL,
                "unavailable",
            ),
            (
                PatchStatus.STALE,
                _truth(
                    MutationState.NOT_COMMITTED,
                    ArtifactState.UNKNOWN,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.REVALIDATION,
                "not_observed",
            ),
            (
                PatchStatus.CANCELLED,
                _truth(
                    MutationState.NOT_COMMITTED,
                    ArtifactState.ABSENT,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.SETTLEMENT,
                "cancelled",
            ),
            (
                PatchStatus.COMMIT_FAILED,
                _truth(
                    MutationState.NOT_COMMITTED,
                    ArtifactState.CLEANED,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.COMMIT,
                "not_observed",
            ),
            (
                PatchStatus.COMMITTED,
                _truth(
                    MutationState.COMMITTED,
                    ArtifactState.CLEANED,
                    PostconditionState.ESTABLISHED,
                ),
                ErrorStage.COMMIT,
                "approved",
            ),
            (
                PatchStatus.PARTIAL,
                _truth(
                    MutationState.PARTIALLY_COMMITTED,
                    ArtifactState.LEAKED,
                    PostconditionState.SUPERSEDED,
                ),
                ErrorStage.COMMIT,
                "approved",
            ),
            (
                PatchStatus.INDETERMINATE,
                _truth(
                    MutationState.INDETERMINATE,
                    ArtifactState.UNKNOWN,
                    PostconditionState.UNKNOWN,
                ),
                ErrorStage.SETTLEMENT,
                "approved",
            ),
        )
        for status, truth, stage, approval in matrix:
            source = await _source(
                disclosures=frozenset(
                    (
                        PolicyDisclosure.EVENT_METRICS,
                        PolicyDisclosure.AUDIT_EXACT_TRUTH,
                        PolicyDisclosure.METRICS_EXACT_TRUTH,
                        PolicyDisclosure.TELEMETRY_EXACT_TRUTH,
                        PolicyDisclosure.SERVER_EXACT_TRUTH,
                    )
                ),
                cancellation_requested=status is PatchStatus.CANCELLED,
            )
            assert source._snapshot.terminal is not None
            result = _terminal_result(source.plan, status, truth, stage)
            object.__setattr__(source._snapshot.terminal, "result", result)
            audit = create_audit_record_boundary(source)
            metrics = create_metrics_record_boundary(source)
            telemetry = create_telemetry_record_boundary(source)
            server = create_server_record_boundary(source)
            deliveries = (
                _delivery(audit.project(audit.authority())),
                _delivery(metrics.project(metrics.authority())),
                _delivery(telemetry.project(telemetry.authority())),
                _delivery(server.project(server.authority())),
            )
            for delivery in deliveries:
                payload = delivery["payload"]
                assert isinstance(payload, dict)
                assert payload["terminal"] is True
                assert payload["status"] == status.value
                assert payload["mutation_state"] == truth.mutation_state.value
                assert payload["artifact_state"] == truth.artifact_state.value
                assert payload["postcondition"] == truth.postcondition.value
                assert payload["commit_set_exact"] is truth.commit_set_exact
                assert payload["approval"] == {
                    "required": True,
                    "outcome": approval,
                }
                assert payload["cancellation_requested"] is (
                    status is PatchStatus.CANCELLED
                )
                assert payload["warning_categories"] == []
                assert payload["error_stage"] == (
                    None if status is PatchStatus.COMMITTED else stage.value
                )
                assert payload["diagnostic_association"] == {
                    "supported": False,
                    "executed": False,
                }
                lineages = payload["lineages"]
                assert isinstance(lineages, list)
                assert len(lineages) == len(source.plan.candidate.lineages)
                assert len({item["lineage_id"] for item in lineages}) == len(
                    lineages
                )

        move_source = await _source(
            disclosures=frozenset(
                (
                    PolicyDisclosure.AUDIT_PATHS,
                    PolicyDisclosure.EVENT_METRICS,
                    PolicyDisclosure.AUDIT_EXACT_TRUTH,
                )
            )
        )
        original = move_source.plan.candidate.lineages[0]
        moved = replace(
            original,
            source_path=LogicalPath("audit-source-endpoint"),
            destination_path=LogicalPath("audit-destination-endpoint"),
            capabilities=frozenset((Capability.MOVE,)),
            matches=(
                Match(
                    MatchKind.NEWLINE_COMPATIBLE,
                    TextSpan(0, 1, 0, 1),
                ),
            ),
        )
        object.__setattr__(
            move_source.plan,
            "candidate",
            replace(move_source.plan.candidate, lineages=(moved,)),
        )
        object.__setattr__(
            move_source.plan,
            "review",
            replace(
                move_source.plan.review,
                warnings=(CapabilityWarning(Capability.MOVE),),
            ),
        )
        move_body = audience_projection._audit_body(move_source)
        assert move_body["matching_exact"] is False
        assert move_body["commit_set_exact"] is True
        assert move_body["operation_classes"] == ("move",)
        assert move_body["warning_categories"] == ("move",)
        move_lineages = move_body["lineages"]
        assert isinstance(move_lineages, tuple)
        assert len(move_lineages) == 1
        move_lineage = move_lineages[0]
        assert move_lineage["operation_classes"] == ("move",)
        assert move_lineage["source_path"] == "audit-source-endpoint"
        assert move_lineage["destination_path"] == "audit-destination-endpoint"
        assert str(move_lineages).count("audit-source-endpoint") == 1
        assert str(move_lineages).count("audit-destination-endpoint") == 1

        assert move_source._snapshot.terminal is not None
        indeterminate = _terminal_result(
            move_source.plan,
            PatchStatus.INDETERMINATE,
            _truth(
                MutationState.INDETERMINATE,
                ArtifactState.UNKNOWN,
                PostconditionState.UNKNOWN,
            ),
            ErrorStage.SETTLEMENT,
        )
        object.__setattr__(
            move_source._snapshot.terminal,
            "result",
            indeterminate,
        )
        indeterminate_body = audience_projection._audit_body(move_source)
        assert indeterminate_body["matching_exact"] is False
        assert indeterminate_body["commit_set_exact"] is False

    run(scenario())


def test_pending_and_cross_audience_authority_fail_closed() -> None:
    """Keep pending nonterminal and reject audience-boundary substitution."""

    async def scenario() -> None:
        source = await _source(
            disclosures=frozenset((PolicyDisclosure.METRICS_EXACT_TRUTH,)),
            terminal=False,
            cancellation_requested=True,
        )
        audit = create_audit_record_boundary(source)
        metrics = create_metrics_record_boundary(source)
        pending = _delivery(metrics.project(metrics.authority()))["payload"]
        assert isinstance(pending, dict)
        assert pending["terminal"] is False
        assert pending["lifecycle"] == "settlement_pending"
        assert pending["cancellation_requested"] is True
        assert "status" not in pending
        with pytest.raises(AudienceProjectionError, match="not issued"):
            audit.project(metrics.authority())
        for value in (source, audit, audit.authority()):
            with pytest.raises(AudienceProjectionError):
                copy(value)
            with pytest.raises(AudienceProjectionError):
                deepcopy(value)
            with pytest.raises(AudienceProjectionError):
                dumps(value)

    run(scenario())


def test_retention_uses_manual_clock_and_bounds_failures() -> None:
    """Encrypt, access, expire, clean, and warn without changing truth."""

    async def scenario() -> None:
        source = await _source()
        identity = source._snapshot.reservation.identity
        key = DurableRetentionKey(
            PatchRetentionKeyId("retention_" + "a" * 16),
            b"a" * 32,
        )
        cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
        )
        backend = InMemoryDurablePatchBackend(
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.AUDIT,))
            ),
            retention_validator=AesGcmDurableRetentionEnvelopeValidator(
                cipher
            ),
        )
        store = InMemoryDurablePatchStore(backend)
        reservation = await store.reserve(
            identity,
            AlgorithmDigest("sha256", "c" * 64),
            source._snapshot.reservation.request_id,
        )
        clock = _ManualClock(1)
        service = AudienceRetentionService(
            store,
            cipher,
            clock,
            _retention_policy(
                DurableRetentionKind.AUDIT_PROJECTION,
                DurableRetentionKind.METRICS_PROJECTION,
                delete_on_terminal=True,
            ),
            _AllowRetentionWriter(),
        )
        canary = b"audit-retention-private-canary"
        writer = await service.issue_writer(
            reservation, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert writer is not None
        write = await service.retain(writer, AudienceRetainedValue(canary))
        assert write.warning is None
        assert write.retention_id is not None
        assert canary.decode() not in repr(write)
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, identity)
        )
        reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert reader is not None
        read = await service.open(reader, write.retention_id)
        assert read.warning is None
        assert read.value is not None
        assert read.value.read() == canary
        assert canary.decode() not in repr(read.value)
        assert str(read.value) == "<redacted>"
        backend.retention_authorizer = StaticDurableRetentionAuthorizer(
            frozenset((Audience.PUBLIC,))
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.get_retention_for_audience(
                access,
                write.retention_id,
                DurableRetentionKind.AUDIT_PROJECTION,
                Audience.AUDIT,
                ExpiryTick(1),
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED
        backend.retention_authorizer = StaticDurableRetentionAuthorizer(
            frozenset((Audience.AUDIT,))
        )
        with pytest.raises(AudienceRetentionError):
            AudienceRetentionWriteReceipt(
                _RetentionIdSubstitute("retained_" + "b" * 16), None
            )
        with pytest.raises(AudienceRetentionError):
            AudienceRetentionReadReceipt(
                _RetainedValueSubstitute(b"value"), None
            )
        invalid_clock_service = AudienceRetentionService(
            store,
            cipher,
            _InvalidClock(),
            _retention_policy(DurableRetentionKind.AUDIT_PROJECTION),
            _AllowRetentionWriter(),
        )
        invalid_clock_reader = (
            await invalid_clock_service.issue_read_authority(
                reservation, access, DurableRetentionKind.AUDIT_PROJECTION
            )
        )
        assert invalid_clock_reader is not None
        assert await invalid_clock_service.open(
            invalid_clock_reader, write.retention_id
        ) == AudienceRetentionReadReceipt(
            None, AudienceRetentionWarning.ACCESS_FAILED
        )
        assert await AudienceRetentionService(
            store,
            cipher,
            _InvalidClock(),
            _retention_policy(DurableRetentionKind.AUDIT_PROJECTION),
            _AllowRetentionWriter(),
        ).cleanup() == AudienceRetentionCleanupReceipt(
            False, AudienceRetentionWarning.CLEANUP_FAILED
        )
        broken_clock = _ManualClock(1)
        object.__setattr__(broken_clock, "now", None)
        with pytest.raises(AudienceRetentionError, match="service is invalid"):
            AudienceRetentionService(
                store,
                cipher,
                broken_clock,
                _retention_policy(DurableRetentionKind.AUDIT_PROJECTION),
                _AllowRetentionWriter(),
            )
        forged_reader = object.__new__(AudienceRetentionReadAuthority)
        assert await service.open(
            forged_reader, write.retention_id
        ) == AudienceRetentionReadReceipt(
            None, AudienceRetentionWarning.ACCESS_FAILED
        )
        assert await service.open(reader, write.retention_id) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )
        clock.value = 5
        expired_reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert expired_reader is not None
        expired = await service.open(expired_reader, write.retention_id)
        assert expired == AudienceRetentionReadReceipt(
            None, AudienceRetentionWarning.ACCESS_FAILED
        )
        later_writer = await service.issue_writer(
            reservation, DurableRetentionKind.METRICS_PROJECTION
        )
        assert later_writer is not None
        later = await service.retain(
            later_writer, AudienceRetainedValue(b"metrics")
        )
        assert later.warning is None
        assert source._snapshot.terminal is not None
        backend.records[identity].terminal = source._snapshot.terminal
        assert await service.cleanup() == AudienceRetentionCleanupReceipt(
            True, None
        )
        assert not backend.records[identity].retention
        failed = await service.retain(
            None,
            AudienceRetainedValue(b"invalid-kind"),
        )
        assert failed == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )
        with pytest.raises(AudienceRetentionError):
            AudienceRetentionPolicy(
                frozenset((DurableRetentionKind.AUDIT_PROJECTION,)),
                DurationTicks(86_401),
                False,
            )
        with pytest.raises(AudienceRetentionError):
            AudienceRetainedValue(b"x").__reduce__()

    run(scenario())


def test_retention_read_authority_is_one_shot_without_service_registry() -> (
    None
):
    """Claim readers before I/O without retaining them in the service."""

    class GateAuthorizer:
        """Gate one authorization call to exercise concurrent cancellation."""

        def __init__(self) -> None:
            """Begin with immediate successful authorization."""
            self.block = False
            self.fail = False
            self.entered = Event()
            self.released = Event()
            self.released.set()

        async def authorize(
            self,
            reservation: DurableReservation,
            kind: DurableRetentionKind,
        ) -> bool:
            """Return the configured decision after an optional async gate."""
            del reservation, kind
            if self.block:
                self.entered.set()
                await self.released.wait()
            if self.fail:
                raise RuntimeError("retention-read-authorizer-failure")
            return True

    async def scenario() -> None:
        source = await _source()
        key = DurableRetentionKey(
            PatchRetentionKeyId("retention_" + "f" * 16), b"f" * 32
        )
        cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
        )
        backend = InMemoryDurablePatchBackend(
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.AUDIT,))
            ),
            retention_validator=AesGcmDurableRetentionEnvelopeValidator(
                cipher
            ),
        )
        store = InMemoryDurablePatchStore(backend)
        reservation = await store.reserve(
            source._snapshot.reservation.identity,
            AlgorithmDigest("sha256", "f" * 64),
            source._snapshot.reservation.request_id,
        )
        clock = _ManualClock(1)
        authorizer = GateAuthorizer()
        service = AudienceRetentionService(
            store,
            cipher,
            clock,
            _retention_policy(DurableRetentionKind.AUDIT_PROJECTION),
            authorizer,
        )
        writer = await service.issue_writer(
            reservation, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert writer is not None
        write = await service.retain(writer, AudienceRetainedValue(b"bounded"))
        assert write.retention_id is not None
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, reservation.identity)
        )

        assert type(service).__slots__ == (
            "_store",
            "_cipher",
            "_clock",
            "_policy",
            "_authorizer",
            "_issuer",
        )
        assert not hasattr(service, "_opened_read_authorities")
        for _ in range(128):
            reader = await service.issue_read_authority(
                reservation, access, DurableRetentionKind.AUDIT_PROJECTION
            )
            assert reader is not None
            assert (
                await service.open(reader, write.retention_id)
            ).warning is None
            del reader
        assert not hasattr(service, "_opened_read_authorities")

        immutable_reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert immutable_reader is not None
        assert (
            await service.open(immutable_reader, write.retention_id)
        ).warning is None
        with pytest.raises(FrozenInstanceError):
            setattr(immutable_reader, "_consumed", False)
        assert await service.open(immutable_reader, write.retention_id) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )

        concurrent_reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert concurrent_reader is not None
        authorizer.block = True
        authorizer.released.clear()
        first = create_task(
            service.open(concurrent_reader, write.retention_id)
        )
        await authorizer.entered.wait()
        second = await service.open(concurrent_reader, write.retention_id)
        assert second == AudienceRetentionReadReceipt(
            None, AudienceRetentionWarning.ACCESS_FAILED
        )
        authorizer.block = False
        authorizer.released.set()
        assert (await first).warning is None

        authorizer.entered.clear()
        authorizer.released.clear()
        cancelled_reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert cancelled_reader is not None
        authorizer.block = True
        cancelled = create_task(
            service.open(cancelled_reader, write.retention_id)
        )
        await authorizer.entered.wait()
        cancelled.cancel()
        with pytest.raises(CancelledError):
            await cancelled
        assert await service.open(cancelled_reader, write.retention_id) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )
        authorizer.block = False
        authorizer.released.set()

        failed_reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert failed_reader is not None
        authorizer.fail = True
        assert await service.open(failed_reader, write.retention_id) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )
        authorizer.fail = False
        assert await service.open(failed_reader, write.retention_id) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )

    run(scenario())


def test_projection_closed_inputs_and_conformance_helpers() -> None:
    """Exercise closed source, approval, lineage, and category branches."""

    async def scenario() -> None:
        source = await _source()
        assert repr(source) == "PatchAudienceProjectionSource(<redacted>)"
        for method in (source.__reduce__, lambda: source.__reduce_ex__(4)):
            with pytest.raises(AudienceProjectionError):
                method()
        for boundary in (
            create_audit_record_boundary(source),
            create_metrics_record_boundary(source),
            create_telemetry_record_boundary(source),
            create_server_record_boundary(source),
        ):
            assert "<opaque>" in repr(boundary.authority())
            for method in (
                boundary.__reduce__,
                lambda boundary=boundary: boundary.__reduce_ex__(4),
                boundary.authority().__reduce__,
                lambda boundary=boundary: boundary.authority().__reduce_ex__(
                    4
                ),
            ):
                with pytest.raises(AudienceProjectionError):
                    method()
        with pytest.raises(AudienceProjectionError, match="host-issued"):
            PatchAudienceProjectionSource(object())
        with pytest.raises(
            AudienceProjectionError, match="authority is invalid"
        ):
            audience_projection.AuditRecordAuthority(object(), object())
        with pytest.raises(AudienceProjectionError, match="exceeds"):
            audience_projection._delivery(
                "audit",
                PatchPublicCorrelationId.new(),
                {"padding": "x" * 1_048_576},
            )

        invalid_seal = await _source()
        object.__setattr__(invalid_seal.plan, "binding", object())
        with pytest.raises(AudienceProjectionError, match="source is invalid"):
            audience_projection._validate_source_truth(
                invalid_seal.plan,
                invalid_seal._snapshot,
            )

        durable_mismatch = await _source()
        assert durable_mismatch._snapshot.plan is not None
        object.__setattr__(
            durable_mismatch._snapshot,
            "plan",
            replace(
                durable_mismatch._snapshot.plan,
                plan_id=PatchPlanId.new(),
            ),
        )
        with pytest.raises(AudienceProjectionError, match="does not match"):
            audience_projection._validate_source_truth(
                durable_mismatch.plan,
                durable_mismatch._snapshot,
            )

        invalid_terminal = await _source()
        assert invalid_terminal._snapshot.terminal is not None
        object.__setattr__(
            invalid_terminal._snapshot.terminal,
            "result",
            replace(
                invalid_terminal._snapshot.terminal.result,
                request_id=invalid_terminal.plan.binding.request.request_id.new(),
            ),
        )
        with pytest.raises(AudienceProjectionError, match="source is invalid"):
            audience_projection._validate_source_truth(
                invalid_terminal.plan,
                invalid_terminal._snapshot,
            )

        invalid_pending = await _source(terminal=False)
        object.__setattr__(invalid_pending._snapshot, "pending", None)
        with pytest.raises(AudienceProjectionError, match="source is invalid"):
            audience_projection._validate_source_truth(
                invalid_pending.plan,
                invalid_pending._snapshot,
            )

        missing_store_plan = await _source()
        object.__setattr__(missing_store_plan._snapshot, "plan", None)
        with pytest.raises(AudienceProjectionError, match="source is invalid"):
            audience_projection._validate_source_truth(
                missing_store_plan.plan,
                missing_store_plan._snapshot,
            )
        missing_correlation = await _source(terminal=False)
        assert missing_correlation._snapshot.pending is not None
        object.__setattr__(
            missing_correlation._snapshot,
            "pending",
            replace(
                missing_correlation._snapshot.pending,
                correlation_id=PatchObserverCorrelationId.new(),
            ),
        )
        with pytest.raises(AudienceProjectionError, match="correlation"):
            audience_projection._validate_access_snapshot(
                missing_correlation._access._access,
                missing_correlation._access._correlation,
                missing_correlation._snapshot,
            )
        access_mismatch = await _source()
        with pytest.raises(AudienceProjectionError, match="store truth"):
            audience_projection._validate_access_snapshot(
                DurableRequestAccess(
                    access_mismatch._access._access.request_id.new(),
                    access_mismatch._access._access.identity,
                ),
                access_mismatch._access._correlation,
                access_mismatch._snapshot,
            )

        assert (
            audience_projection._approval(
                await _source(mode=ApprovalMode.PREAUTHORIZED)
            )["outcome"]
            == "preauthorized"
        )
        denied_mode = await _source()
        object.__setattr__(
            denied_mode.plan.binding.final,
            "approval",
            replace(
                denied_mode.plan.binding.final.approval,
                mode=ApprovalMode.DENY,
            ),
        )
        assert (
            audience_projection._approval(denied_mode)["outcome"]
            == "not_available"
        )
        assert (
            audience_projection._approval(await _source(terminal=False))[
                "outcome"
            ]
            == "pending"
        )
        for status, expected in (
            (PatchStatus.APPROVAL_DENIED, "denied"),
            (PatchStatus.APPROVAL_UNAVAILABLE, "unavailable"),
            (PatchStatus.CANCELLED, "cancelled"),
            (PatchStatus.STALE, "not_observed"),
        ):
            approval_source = await _source()
            assert approval_source._snapshot.terminal is not None
            object.__setattr__(
                approval_source._snapshot.terminal,
                "result",
                _result(approval_source.plan, status),
            )
            assert (
                audience_projection._approval(approval_source)["outcome"]
                == expected
            )

        lineage = source.plan.candidate.lineages[0]
        inexact = replace(
            lineage,
            matches=(
                Match(
                    MatchKind.NEWLINE_COMPATIBLE,
                    TextSpan(0, 1, 0, 1),
                ),
            ),
        )
        unmatched = replace(lineage, matches=())
        assert audience_projection._matching_exact((unmatched,)) is None
        assert audience_projection._matching_exact((lineage, inexact)) is False
        assert audience_projection._lineage_matching_exact(unmatched) is None
        classes = audience_projection._lineage_operation_classes(
            replace(
                lineage,
                capabilities=frozenset(
                    (
                        Capability.CREATE,
                        Capability.UPDATE_EXECUTABLE,
                        Capability.DELETE,
                        Capability.MOVE,
                    )
                ),
            )
        )
        assert classes == ("create", "delete", "move", "update")
        assert audience_projection._operation_classes((lineage,)) == (
            "update",
        )
        assert (
            audience_projection._lineage_operation_classes(
                replace(lineage, capabilities=frozenset())
            )
            == ()
        )

        assert source._snapshot.plan is not None
        step = source._snapshot.plan.steps[0]
        assert (
            audience_projection._journal_mutation_state(source._snapshot)
            is MutationState.COMMITTED
        )
        for states, expected in (
            ((CommitStepState.NOT_COMMITTED,), "not_committed"),
            ((CommitStepState.UNKNOWN,), "indeterminate"),
            (
                (CommitStepState.COMMITTED, CommitStepState.NOT_COMMITTED),
                "partially_committed",
            ),
        ):
            entries = tuple(
                DurableStepJournalEntry(
                    DurableJournalCursor(
                        source._snapshot.reservation.request_id,
                        SequenceNumber(index),
                    ),
                    step.step_id,
                    step.lineage_id,
                    value,
                )
                for index, value in enumerate(states, start=1)
            )
            object.__setattr__(
                source._snapshot,
                "journal",
                DurableJournal(
                    DurableJournalCursor(
                        source._snapshot.reservation.request_id,
                        SequenceNumber(len(entries)),
                    ),
                    entries,
                    (),
                ),
            )
            assert (
                audience_projection._lineage_states(source)[
                    step.lineage_id.value
                ]
                == expected
            )
            assert (
                audience_projection._journal_mutation_state(
                    source._snapshot
                ).value
                == expected
            )

        paths_without_exact = await _source(
            disclosures=frozenset((PolicyDisclosure.AUDIT_PATHS,))
        )
        paths_body = audience_projection._audit_body(paths_without_exact)
        paths_lineages = paths_body["lineages"]
        assert isinstance(paths_lineages, tuple)
        assert "matching_exact" not in paths_lineages[0]
        assert "source_path" in paths_lineages[0]

    run(scenario())


def test_retention_writer_authority_and_store_limits_fail_closed() -> None:
    """Bind retained data to host authority, policy TTL, and store bounds."""

    async def scenario() -> None:
        source = await _source()
        identity = source._snapshot.reservation.identity
        key = DurableRetentionKey(
            PatchRetentionKeyId("retention_" + "c" * 16), b"c" * 32
        )
        cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
        )
        backend = InMemoryDurablePatchBackend(
            DurableStoreLimits(
                max_retention_records=1,
                max_retention_bytes=ByteSize(64),
            ),
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.AUDIT,))
            ),
            retention_validator=AesGcmDurableRetentionEnvelopeValidator(
                cipher
            ),
        )
        store = InMemoryDurablePatchStore(backend)
        reservation = await store.reserve(
            identity,
            AlgorithmDigest("sha256", "d" * 64),
            source._snapshot.reservation.request_id,
        )
        clock = _ManualClock(1)
        policy = _retention_policy(DurableRetentionKind.AUDIT_PROJECTION)
        authorizer = _ToggleRetentionWriter()
        service = AudienceRetentionService(
            store, cipher, clock, policy, authorizer
        )
        default_denied = AudienceRetentionService(store, cipher, clock, policy)
        assert (
            await default_denied.issue_writer(
                reservation, DurableRetentionKind.AUDIT_PROJECTION
            )
            is None
        )
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, reservation.identity)
        )
        assert (
            await default_denied.issue_read_authority(
                reservation, access, DurableRetentionKind.AUDIT_PROJECTION
            )
            is None
        )
        with pytest.raises(AudienceRetentionError, match="authorization"):
            await default_denied._authorizer.authorize(
                _ReservationSubstitute(
                    reservation.request_id,
                    reservation.identity,
                    reservation.canonical_digest,
                    reservation.replayed,
                ),
                DurableRetentionKind.AUDIT_PROJECTION,
            )
        with pytest.raises(AudienceRetentionError, match="service-issued"):
            AudienceRetentionWriter(object())
        with pytest.raises(AudienceRetentionError, match="service-issued"):
            AudienceRetentionReadAuthority(object())
        writer = await service.issue_writer(
            reservation, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert writer is not None
        assert repr(writer) == "AudienceRetentionWriter(<opaque>)"
        for method in (
            writer.__copy__,
            lambda: writer.__deepcopy__({}),
            writer.__reduce__,
            lambda: writer.__reduce_ex__(4),
        ):
            with pytest.raises(AudienceRetentionError):
                method()
        denied_reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert denied_reader is not None
        assert (
            repr(denied_reader) == "AudienceRetentionReadAuthority(<opaque>)"
        )
        assert (
            await service.issue_writer(
                reservation, DurableRetentionKind.TELEMETRY_PROJECTION
            )
            is None
        )
        for kind in (
            DurableRetentionKind.SEALED_PLAN,
            DurableRetentionKind.REVIEW_ARTIFACT,
            DurableRetentionKind.PRIVATE_STAGING,
        ):
            assert await service.issue_writer(reservation, kind) is None
            assert (
                await service.issue_read_authority(
                    reservation,
                    DurableRetentionAccess(
                        DurableRequestAccess(
                            reservation.request_id, reservation.identity
                        )
                    ),
                    kind,
                )
                is None
            )
        authorizer.allowed = False
        assert await service.retain(
            writer, AudienceRetainedValue(b"denied-before-encryption")
        ) == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )
        assert await service.open(
            denied_reader, PatchRetentionRecordId.new()
        ) == AudienceRetentionReadReceipt(
            None, AudienceRetentionWarning.ACCESS_FAILED
        )
        assert not backend.records[identity].retention
        authorizer.allowed = True
        other_service = AudienceRetentionService(
            store, cipher, clock, policy, authorizer
        )
        assert await other_service.retain(
            writer, AudienceRetainedValue(b"cross-service")
        ) == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )
        first = await service.retain(writer, AudienceRetainedValue(b"bounded"))
        assert first.warning is None
        assert first.retention_id is not None
        reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert reader is not None
        assert (
            await service.open(reader, first.retention_id)
        ).value is not None
        for method in (
            reader.__copy__,
            lambda: reader.__deepcopy__({}),
            reader.__reduce__,
            lambda: reader.__reduce_ex__(4),
        ):
            with pytest.raises(AudienceRetentionError):
                method()
        other_reader = await other_service.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert other_reader is not None
        assert await service.open(other_reader, first.retention_id) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )
        assert (
            await service.issue_read_authority(
                reservation,
                DurableRetentionAccess(
                    DurableRequestAccess(
                        reservation.request_id.new(), reservation.identity
                    )
                ),
                DurableRetentionKind.AUDIT_PROJECTION,
            )
            is None
        )
        broad = AudienceRetentionService(
            store,
            cipher,
            clock,
            _retention_policy(
                DurableRetentionKind.AUDIT_PROJECTION,
                DurableRetentionKind.METRICS_PROJECTION,
            ),
            authorizer,
        )
        metrics_reader = await broad.issue_read_authority(
            reservation, access, DurableRetentionKind.METRICS_PROJECTION
        )
        assert metrics_reader is not None
        assert await broad.open(metrics_reader, first.retention_id) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )
        assert await service.retain(
            writer, AudienceRetainedValue(b"count-bound")
        ) == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )
        byte_backend = InMemoryDurablePatchBackend(
            DurableStoreLimits(
                max_retention_records=2,
                max_retention_bytes=ByteSize(16),
            ),
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.AUDIT,))
            ),
            retention_validator=AesGcmDurableRetentionEnvelopeValidator(
                cipher
            ),
        )
        byte_store = InMemoryDurablePatchStore(byte_backend)
        byte_reservation = await byte_store.reserve(
            identity,
            AlgorithmDigest("sha256", "e" * 64),
            source._snapshot.reservation.request_id,
        )
        byte_service = AudienceRetentionService(
            byte_store, cipher, clock, policy, authorizer
        )
        byte_writer = await byte_service.issue_writer(
            byte_reservation, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert byte_writer is not None
        assert await byte_service.retain(
            byte_writer, AudienceRetainedValue(b"x")
        ) == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )
        assert not byte_backend.records[identity].retention
        overflow_clock = _ManualClock(2**63 - 1)
        overflow_service = AudienceRetentionService(
            store, cipher, overflow_clock, policy, authorizer
        )
        overflow_writer = await overflow_service.issue_writer(
            reservation, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert overflow_writer is not None
        assert await overflow_service.retain(
            overflow_writer, AudienceRetainedValue(b"huge-expiry")
        ) == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )
        failing = AudienceRetentionService(
            store,
            cipher,
            clock,
            policy,
            _FailingRetentionWriter(),
        )
        assert (
            await failing.issue_writer(
                reservation, DurableRetentionKind.AUDIT_PROJECTION
            )
            is None
        )
        assert (
            await failing.issue_read_authority(
                reservation, access, DurableRetentionKind.AUDIT_PROJECTION
            )
            is None
        )
        with pytest.raises(AudienceRetentionError, match="kind is invalid"):
            _retention_audience(DurableRetentionKind.SEALED_PLAN)

    run(scenario())


def test_retention_dependency_faults_are_independent_warnings() -> None:
    """Bound encryption, store access, and cleanup dependency failures."""

    class FailingResolver:
        """Fail every exact key lookup without returning a secret."""

        async def active_key(self) -> DurableRetentionKey:
            """Fail active-key resolution."""
            raise RuntimeError("key-failure")

        async def read_key(
            self, key_id: PatchRetentionKeyId
        ) -> DurableRetentionKey:
            """Fail read-key resolution."""
            del key_id
            raise RuntimeError("key-failure")

    class FailingStore:
        """Fail each retention operation without returning internal detail."""

        async def put_retention(
            self,
            reservation: DurableReservation,
            record: DurableRetentionRecord,
        ) -> None:
            """Fail persistence after receiving typed encrypted data."""
            del reservation, record
            raise RuntimeError("write-failure")

        async def get_retention_for_audience(
            self,
            access: DurableRetentionAccess,
            retention_id: PatchRetentionRecordId,
            kind: DurableRetentionKind,
            audience: Audience,
            now: ExpiryTick,
        ) -> DurableRetentionRecord:
            """Fail authorized ciphertext lookup."""
            del access, retention_id, kind, audience, now
            raise RuntimeError("access-failure")

        async def cleanup_retention(
            self, now: ExpiryTick
        ) -> DurableRetentionCleanup:
            """Fail bounded retention cleanup."""
            del now
            raise RuntimeError("cleanup-failure")

    async def scenario() -> None:
        source = await _source()
        key = DurableRetentionKey(
            PatchRetentionKeyId("retention_" + "b" * 16), b"b" * 32
        )
        cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
        )
        clock = _ManualClock(1)
        service = AudienceRetentionService(
            FailingStore(),
            cipher,
            clock,
            _retention_policy(DurableRetentionKind.TELEMETRY_PROJECTION),
            _AllowRetentionWriter(),
        )
        reservation = source._snapshot.reservation
        value = AudienceRetainedValue(b"fault-canary")
        writer = await service.issue_writer(
            reservation, DurableRetentionKind.TELEMETRY_PROJECTION
        )
        assert writer is not None
        write = await service.retain(writer, value)
        assert write == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, reservation.identity)
        )
        reader = await service.issue_read_authority(
            reservation, access, DurableRetentionKind.TELEMETRY_PROJECTION
        )
        assert reader is not None
        assert await service.open(reader, PatchRetentionRecordId.new()) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )
        assert await service.cleanup() == AudienceRetentionCleanupReceipt(
            False, AudienceRetentionWarning.CLEANUP_FAILED
        )
        failed_cipher = AesGcmDurableRetentionCipher(FailingResolver())
        encryption = AudienceRetentionService(
            FailingStore(),
            failed_cipher,
            clock,
            _retention_policy(DurableRetentionKind.CLI_REVIEW),
            _AllowRetentionWriter(),
        )
        encryption_writer = await encryption.issue_writer(
            reservation, DurableRetentionKind.CLI_REVIEW
        )
        assert encryption_writer is not None
        assert await encryption.retain(
            encryption_writer,
            value,
        ) == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.ENCRYPTION_FAILED
        )
        denied = AudienceRetentionService(
            FailingStore(),
            cipher,
            clock,
            _retention_policy(DurableRetentionKind.CLI_REVIEW),
            _DenyRetentionWriter(),
        )
        assert (
            await denied.issue_writer(
                reservation, DurableRetentionKind.CLI_REVIEW
            )
            is None
        )
        assert await denied.retain(
            None, value
        ) == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )
        clock_failed = AudienceRetentionService(
            FailingStore(),
            cipher,
            _InvalidClock(),
            _retention_policy(DurableRetentionKind.CLI_REVIEW),
            _AllowRetentionWriter(),
        )
        clock_writer = await clock_failed.issue_writer(
            reservation, DurableRetentionKind.CLI_REVIEW
        )
        assert clock_writer is not None
        assert await clock_failed.retain(
            clock_writer, value
        ) == AudienceRetentionWriteReceipt(
            None, AudienceRetentionWarning.WRITE_FAILED
        )

        retention_id = PatchRetentionRecordId.new()
        encrypted = await cipher.seal(
            b"non-audience-kind",
            DurableRetentionBinding(
                reservation.request_id,
                retention_id,
                DurableRetentionKind.SEALED_PLAN,
            ),
        )
        invalid_kind = DurableRetentionRecord(
            retention_id,
            DurableRetentionKind.SEALED_PLAN,
            encrypted.key_id,
            encrypted.value,
            DurableRetentionPolicy(ExpiryTick(10), False),
        )

        class InvalidResultStore(FailingStore):
            """Return typed but policy-ineligible stored records."""

            async def get_retention_for_audience(
                self,
                access: DurableRetentionAccess,
                requested_id: PatchRetentionRecordId,
                kind: DurableRetentionKind,
                audience: Audience,
                now: ExpiryTick,
            ) -> DurableRetentionRecord:
                """Return the non-audience record after typed access."""
                del access, requested_id, kind, audience, now
                return invalid_kind

            async def cleanup_retention(
                self, now: ExpiryTick
            ) -> DurableRetentionCleanup:
                """Return a cleanup subclass rejected by the boundary."""
                del now
                return _CleanupSubstitute(0, ByteSize(0))

        malformed = AudienceRetentionService(
            InvalidResultStore(),
            cipher,
            clock,
            _retention_policy(DurableRetentionKind.AUDIT_PROJECTION),
            _AllowRetentionWriter(),
        )
        malformed_reader = await malformed.issue_read_authority(
            reservation, access, DurableRetentionKind.AUDIT_PROJECTION
        )
        assert malformed_reader is not None
        assert await malformed.open(malformed_reader, retention_id) == (
            AudienceRetentionReadReceipt(
                None, AudienceRetentionWarning.ACCESS_FAILED
            )
        )
        assert await malformed.cleanup() == AudienceRetentionCleanupReceipt(
            False, AudienceRetentionWarning.CLEANUP_FAILED
        )
        for item in (value,):
            with pytest.raises(AudienceRetentionError):
                copy(item)
            with pytest.raises(AudienceRetentionError):
                deepcopy(item)
            with pytest.raises(AudienceRetentionError):
                dumps(item)
            with pytest.raises(AudienceRetentionError):
                item.__reduce_ex__(4)

        for factory in (
            lambda: AudienceRetainedValue("invalid"),
            lambda: AudienceRetentionWriteReceipt(None, None),
            lambda: AudienceRetentionWriteReceipt(
                PatchRetentionRecordId.new(),
                AudienceRetentionWarning.WRITE_FAILED,
            ),
            lambda: AudienceRetentionReadReceipt(None, None),
            lambda: AudienceRetentionReadReceipt(
                value, AudienceRetentionWarning.ACCESS_FAILED
            ),
            lambda: AudienceRetentionCleanupReceipt(
                True, AudienceRetentionWarning.CLEANUP_FAILED
            ),
        ):
            with pytest.raises(AudienceRetentionError):
                factory()

    run(scenario())
