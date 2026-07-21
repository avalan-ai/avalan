"""Exercise trusted interaction policy boundaries and fixed limits."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import get_type_hints

import pytest

from avalan import interaction as interaction_api
from avalan.interaction import (
    INTERACTION_SETTLEMENT_PRECEDENCE,
    MAX_EQUIVALENT_INTERACTIONS_PER_BRANCH,
    MAX_PENDING_INTERACTIONS_PER_PROCESS,
    MAX_RESOLUTION_IDEMPOTENCY_KEY_BYTES,
    MAX_RESOLUTION_IDEMPOTENCY_KEY_CHARACTERS,
    MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST,
    MAX_UNRESOLVED_INTERACTIONS_PER_RUN,
    MAX_UNRESOLVED_REQUIRED_INTERACTIONS_PER_BRANCH,
    AcquireControllerActivity,
    ActiveControlLeaseNonce,
    BranchId,
    ControllerActivityAction,
    ControllerId,
    DeadlineTiePolicy,
    DisconnectControllerActivity,
    HandlerLossDisposition,
    InputErrorCode,
    InputRequestId,
    InputValidationError,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionBranchAuthorizationTarget,
    InteractionClock,
    InteractionDisclosure,
    InteractionOperation,
    InteractionPolicy,
    InteractionScopeAuthorizationTarget,
    InteractionSettlement,
    InteractionTime,
    PrincipalScope,
    PulseControllerActivity,
    QuestionId,
    QuestionType,
    ReleaseControllerActivity,
    ResolutionIdempotencyKey,
    RunId,
    SequencedControllerActivity,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    UserId,
    select_interaction_settlement,
    validate_resolution_idempotency_key,
)

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


def test_trusted_time_and_fixed_runtime_policy() -> None:
    """Keep deadline equality and handler-loss outcomes deterministic."""
    observed = InteractionTime.from_clock(
        wall_time=_NOW,
        monotonic_seconds=12.5,
    )
    policy = InteractionPolicy()

    assert observed.monotonic_seconds == 12.5
    assert observed.monotonic_deadline_reached(12.5)
    assert not observed.monotonic_deadline_reached(12.5001)
    assert observed.wall_deadline_reached(_NOW)
    assert not observed.wall_deadline_reached(_NOW.replace(microsecond=1))
    assert get_type_hints(InteractionClock.read)["return"] is InteractionTime
    assert not hasattr(interaction_api, "_InteractionTime")
    assert not hasattr(InteractionTime, "_from_clock")
    with pytest.raises(FrozenInstanceError):
        observed.wall_time = _NOW  # type: ignore[misc]
    assert policy.deadline_tie is DeadlineTiePolicy.DEADLINE_FIRST
    assert policy.attached_loss_with_resumer is HandlerLossDisposition.DETACH
    assert (
        policy.attached_loss_without_resumer
        is HandlerLossDisposition.UNAVAILABLE
    )
    with pytest.raises(InputValidationError):
        InteractionTime(
            wall_time=_NOW,
            monotonic_seconds=0,
            _token=object(),
        )
    with pytest.raises(InputValidationError):
        InteractionTime.from_clock(
            wall_time=_NOW,
            monotonic_seconds=-0.1,
        )
    with pytest.raises(InputValidationError):
        InteractionPolicy(
            attached_loss_without_resumer=HandlerLossDisposition.DETACH
        )


def test_idempotency_keys_have_character_and_utf8_bounds() -> None:
    """Reject unbounded transport keys before they enter a store ledger."""
    valid = "k" * MAX_RESOLUTION_IDEMPOTENCY_KEY_CHARACTERS
    assert validate_resolution_idempotency_key(valid) == valid

    with pytest.raises(InputValidationError) as characters:
        validate_resolution_idempotency_key(valid + "k")
    assert characters.value.code is InputErrorCode.OUT_OF_BOUNDS

    bytes_overflow = "a" * (MAX_RESOLUTION_IDEMPOTENCY_KEY_BYTES // 4) + "😀"
    with pytest.raises(InputValidationError) as encoded:
        validate_resolution_idempotency_key(bytes_overflow)
    assert encoded.value.code is InputErrorCode.OUT_OF_BOUNDS


def test_controller_activity_has_explicit_lease_actions() -> None:
    """Make acquire nonce-free and every later action strictly sequenced."""
    acquire = AcquireControllerActivity(
        request_id=InputRequestId("request-1"),
        controller_id=ControllerId("controller-1"),
    )
    activities: tuple[SequencedControllerActivity, ...] = (
        PulseControllerActivity(
            request_id=InputRequestId("request-1"),
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=1,
        ),
        ReleaseControllerActivity(
            request_id=InputRequestId("request-1"),
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=2,
        ),
        DisconnectControllerActivity(
            request_id=InputRequestId("request-1"),
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=3,
        ),
    )

    assert acquire.action is ControllerActivityAction.ACQUIRE
    assert not hasattr(acquire, "lease_nonce")
    for sequence, evidence in enumerate(activities, start=1):
        assert evidence.action is tuple(ControllerActivityAction)[sequence]
        assert evidence.sequence == sequence

    with pytest.raises(InputValidationError):
        PulseControllerActivity(
            request_id=InputRequestId("request-1"),
            controller_id=ControllerId("controller-1"),
            lease_nonce=ActiveControlLeaseNonce("lease-1"),
            sequence=0,
        )


def test_denied_authorization_discloses_nothing() -> None:
    """Reject denied decisions that would disclose interaction metadata."""
    actor = InteractionActor(
        principal=PrincipalScope(user_id=UserId("reviewer"))
    )
    target = InteractionBranchAuthorizationTarget(
        run_id=RunId("run-1"),
        branch_id=BranchId("child"),
        parent_branch_id=BranchId("parent"),
        principal=PrincipalScope(user_id=UserId("owner")),
    )
    denied = InteractionAuthorizationDecision(
        actor=actor,
        operation=InteractionOperation.REGISTER_BRANCH,
        target=target,
        allowed=False,
        disclosure=InteractionDisclosure.NONE,
    )
    assert denied.disclosure is InteractionDisclosure.NONE

    with pytest.raises(InputValidationError):
        InteractionAuthorizationDecision(
            actor=actor,
            operation=InteractionOperation.REGISTER_BRANCH,
            target=target,
            allowed=False,
            disclosure=InteractionDisclosure.TERMINAL_METADATA,
        )


def test_idempotency_key_static_identity_is_preserved() -> None:
    """Return the dedicated opaque key type from runtime validation."""
    key = validate_resolution_idempotency_key("key-1")
    typed: ResolutionIdempotencyKey = key
    assert typed == "key-1"


@pytest.mark.parametrize(
    ("absolute", "advisory", "candidate", "expected"),
    (
        (False, False, False, None),
        (False, False, True, InteractionSettlement.CANDIDATE_RESOLUTION),
        (False, True, False, InteractionSettlement.ADVISORY_TIMEOUT),
        (False, True, True, InteractionSettlement.ADVISORY_TIMEOUT),
        (True, False, False, InteractionSettlement.ABSOLUTE_EXPIRY),
        (True, False, True, InteractionSettlement.ABSOLUTE_EXPIRY),
        (True, True, False, InteractionSettlement.ABSOLUTE_EXPIRY),
        (True, True, True, InteractionSettlement.ABSOLUTE_EXPIRY),
    ),
)
def test_settlement_priority_is_total(
    absolute: bool,
    advisory: bool,
    candidate: bool,
    expected: InteractionSettlement | None,
) -> None:
    """Prefer absolute expiry, then advisory timeout, then submission."""
    assert INTERACTION_SETTLEMENT_PRECEDENCE == (
        InteractionSettlement.ABSOLUTE_EXPIRY,
        InteractionSettlement.ADVISORY_TIMEOUT,
        InteractionSettlement.CANDIDATE_RESOLUTION,
    )
    assert (
        select_interaction_settlement(
            absolute_expiry_due=absolute,
            advisory_timeout_due=advisory,
            candidate_resolution_present=candidate,
        )
        is expected
    )


def test_policy_centralizes_every_shared_capacity_limit() -> None:
    """Expose exact defaults and reject zero or above-contract limits."""
    policy = InteractionPolicy()
    assert policy.maximum_unresolved_interactions_per_run == 8
    assert policy.maximum_unresolved_required_interactions_per_branch == 1
    assert policy.maximum_equivalent_interactions_per_branch == 3
    assert policy.maximum_pending_interactions_per_process == 1_024
    assert policy.maximum_idempotency_keys_per_request == 32

    limits = (
        (
            "maximum_unresolved_interactions_per_run",
            MAX_UNRESOLVED_INTERACTIONS_PER_RUN,
        ),
        (
            "maximum_unresolved_required_interactions_per_branch",
            MAX_UNRESOLVED_REQUIRED_INTERACTIONS_PER_BRANCH,
        ),
        (
            "maximum_equivalent_interactions_per_branch",
            MAX_EQUIVALENT_INTERACTIONS_PER_BRANCH,
        ),
        (
            "maximum_pending_interactions_per_process",
            MAX_PENDING_INTERACTIONS_PER_PROCESS,
        ),
        (
            "maximum_idempotency_keys_per_request",
            MAX_RESOLUTION_IDEMPOTENCY_KEYS_PER_REQUEST,
        ),
    )
    for field_name, maximum in limits:
        with pytest.raises(InputValidationError) as below:
            InteractionPolicy(**{field_name: 0})  # type: ignore[arg-type]
        with pytest.raises(InputValidationError) as above:
            InteractionPolicy(
                **{field_name: maximum + 1}  # type: ignore[arg-type]
            )
        assert below.value.code is InputErrorCode.OUT_OF_BOUNDS
        assert above.value.code is InputErrorCode.OUT_OF_BOUNDS


def test_classification_request_repr_omits_submitted_content() -> None:
    """Keep classifier input private and its public output untrusted."""
    request = TaskInputClassificationRequest(
        value="sensitive submitted value",
        request_id=InputRequestId("request-1"),
        candidate_digest="0" * 64,
        question_id=QuestionId("question-1"),
        semantic_type=QuestionType.TEXT,
        policy_revision="policy-1",
    )

    assert "sensitive submitted value" not in repr(request)
    output = TaskInputClassification(
        decision=TaskInputClassificationDecision.ALLOW,
        classifier_id="classifier-1",
        classification_id="classification-1",
        policy_revision=request.policy_revision,
        request_id=request.request_id,
        candidate_digest=request.candidate_digest,
        question_id=request.question_id,
        semantic_type=request.semantic_type,
    )
    assert output.request_id == request.request_id
    assert output.candidate_digest == request.candidate_digest
    assert not hasattr(TaskInputClassification, "_from_classifier")
    with pytest.raises(TypeError):
        TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id="classifier-1",
            classification_id="classification-1",
            policy_revision="policy-1",
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
            _token=object(),  # type: ignore[call-arg]
        )


def test_scope_authorization_target_covers_empty_list_and_mutation() -> None:
    """Authorize a scope independently from any matching request record."""
    actor = InteractionActor(principal=PrincipalScope(user_id=UserId("owner")))
    target = InteractionScopeAuthorizationTarget(
        run_id=RunId("run-1"),
        branch_id=BranchId("branch-1"),
        include_descendants=True,
        principal=actor.principal,
    )

    for operation in (
        InteractionOperation.LIST,
        InteractionOperation.CANCEL_SCOPE,
        InteractionOperation.SUPERSEDE,
    ):
        decision = InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.NONE,
        )
    with pytest.raises(InputValidationError):
        InteractionAuthorizationDecision(
            actor=actor,
            operation=InteractionOperation.SUPERSEDE,
            target=InteractionBranchAuthorizationTarget(
                run_id=RunId("run-1"),
                branch_id=BranchId("branch-1"),
                parent_branch_id=BranchId("root"),
                principal=actor.principal,
            ),
            allowed=True,
            disclosure=InteractionDisclosure.NONE,
        )
        assert decision.target is target
    exact_branch_target = InteractionScopeAuthorizationTarget(
        run_id=RunId("run-1"),
        branch_id=BranchId("branch-1"),
        principal=actor.principal,
    )
    branch_inspection = InteractionAuthorizationDecision(
        actor=actor,
        operation=InteractionOperation.INSPECT_BRANCH,
        target=exact_branch_target,
        allowed=True,
        disclosure=InteractionDisclosure.NONE,
    )
    assert branch_inspection.target is exact_branch_target
    for invalid_target in (
        target,
        InteractionScopeAuthorizationTarget(
            run_id=RunId("run-1"),
            principal=actor.principal,
        ),
        InteractionBranchAuthorizationTarget(
            run_id=RunId("run-1"),
            branch_id=BranchId("branch-1"),
            parent_branch_id=BranchId("root"),
            principal=actor.principal,
        ),
    ):
        with pytest.raises(InputValidationError):
            InteractionAuthorizationDecision(
                actor=actor,
                operation=InteractionOperation.INSPECT_BRANCH,
                target=invalid_target,
                allowed=True,
                disclosure=InteractionDisclosure.NONE,
            )
    with pytest.raises(InputValidationError):
        InteractionAuthorizationDecision(
            actor=actor,
            operation=InteractionOperation.CANCEL_SCOPE,
            target=InteractionBranchAuthorizationTarget(
                run_id=RunId("run-1"),
                branch_id=BranchId("branch-1"),
                parent_branch_id=BranchId("root"),
                principal=actor.principal,
            ),
            allowed=True,
            disclosure=InteractionDisclosure.NONE,
        )
