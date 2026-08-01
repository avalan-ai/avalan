"""Exercise closed Phase 2 runtime and fake-adapter validation branches."""

import asyncio as asyncio_module
from asyncio import CancelledError, all_tasks, create_task, sleep
from collections.abc import Awaitable, Callable, Iterator
from copy import copy
from dataclasses import replace
from datetime import datetime
from dis import get_instructions
from inspect import getsource, signature
from types import CodeType, MappingProxyType
from typing import cast

import pytest
from phase2_fixtures import (
    NOW,
    authority,
    binding,
    empty_stateless_plan,
    request,
    root_identity,
)

import avalan.conversation as conversation
from avalan.conversation import fakes as fakes_module
from avalan.conversation.runtime import request_digest_value

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run fake adapter validation on asyncio only."""
    return "asyncio"


def _snapshot_and_candidate() -> tuple[
    conversation.ConversationRunRequest,
    conversation.StatelessProviderLaneSnapshot,
    conversation.OutwardTurnCheckpointCandidate,
]:
    scope = authority()
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    run = request(
        scope=scope,
        identity=root_identity("validation"),
        advance=conversation.FirstTurnAdvance(),
        response_suffix="validation",
        key="key-validation",
    )
    execution_receipt = conversation.provider_lane_execution_receipt(
        authority=scope,
        identity=run.identity,
        binding=lane_binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=result.items,
        reasoning=result.reasoning,
        usage=result.usage,
        upstream_response_id=None,
    )
    snapshot = conversation.StatelessProviderLaneSnapshot(
        binding=lane_binding,
        ledger=conversation.ProviderItemLedger(
            lane_id=lane_binding.lane_id,
            normalization_version=conversation.ConversationCodecVersion(1),
            items=result.items,
        ),
        reasoning=result.reasoning,
        lifecycle=conversation.ProviderLaneLifecycle.COMMITTED,
        retention_policy=conversation.ChildLaneRetentionPolicy.RETAIN,
        execution_receipt=execution_receipt,
    )
    candidate = conversation.build_checkpoint_candidate(
        run,
        parent=None,
        completed_lanes=(snapshot,),
        created_at=NOW,
    )
    assert isinstance(candidate, conversation.OutwardTurnCheckpointCandidate)
    return run, snapshot, candidate


def _commit() -> conversation.AtomicConversationCommit:
    run, snapshot, candidate = _snapshot_and_candidate()
    assert snapshot.execution_receipt is not None
    output_candidate = conversation.ProviderLaneOutputCandidate(
        lane_id=snapshot.lane_id,
        binding=snapshot.binding,
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
        completed_items=snapshot.ledger.items,
        reasoning=snapshot.reasoning,
        usage=conversation.ProviderUsage(input_tokens=10, output_tokens=5),
        execution_receipt=snapshot.execution_receipt,
    )
    return conversation.AtomicConversationCommit(
        candidate=candidate,
        idempotency=conversation.RequestIdempotencyIdentity(
            authority=run.semantics.authority,
            operation=conversation.ConversationOperation.CREATE,
            key=run.idempotency_key,
            request_digest=conversation.CanonicalRequestDigest(
                "validation-digest"
            ),
        ),
        owner_token="validation-owner",
        output_candidates=(output_candidate,),
        committed_at=NOW,
        result_mode=conversation.ConversationMode.STATELESS,
        provisional_response_id=run.provisional_response_id,
        public_response_id=run.public_response_id,
        outbox_intent_id="validation-outbox",
    )


def test_execution_receipt_settlement_and_snapshot_validation_is_closed() -> (
    None
):
    """Reject malformed receipts, settlement states, and snapshot proofs."""
    digest = conversation.IntegrityDigest("0" * 64)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLaneExecutionReceipt(
            schema_version=2,
            digest=digest,
            item_count=0,
            opaque_byte_count=0,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLaneExecutionReceipt(
            schema_version=1,
            digest=conversation.IntegrityDigest("short"),
            item_count=0,
            opaque_byte_count=0,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLaneExecutionReceipt(
            schema_version=1,
            digest=conversation.IntegrityDigest("z" * 64),
            item_count=0,
            opaque_byte_count=0,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.ProviderLaneExecutionReceipt(
            schema_version=1,
            digest=digest,
            item_count=-1,
            opaque_byte_count=0,
        )

    run, stateless, _candidate = _snapshot_and_candidate()
    with pytest.raises(conversation.ConversationValidationError):
        conversation.provider_lane_execution_receipt(
            authority=cast(conversation.AuthorityScope, object()),
            identity=run.identity,
            binding=stateless.binding,
            mode=conversation.ConversationMode.STATELESS,
            scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
            completed_items=stateless.ledger.items,
            reasoning=stateless.reasoning,
            usage=conversation.ProviderUsage(),
            upstream_response_id=None,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.provider_lane_execution_receipt(
            authority=run.semantics.authority,
            identity=run.identity,
            binding=stateless.binding,
            mode=conversation.ConversationMode.STATELESS,
            scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
            completed_items=stateless.ledger.items,
            reasoning=stateless.reasoning,
            usage=conversation.ProviderUsage(),
            upstream_response_id=conversation.UpstreamResponseId(
                "unexpected-upstream"
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            stateless,
            execution_receipt=cast(
                conversation.ProviderLaneExecutionReceipt,
                object(),
            ),
        )
    stored = conversation.StoredProviderLaneSnapshot(
        binding=stateless.binding,
        upstream_response_id=conversation.UpstreamResponseId(
            "validation-upstream"
        ),
        reasoning=stateless.reasoning,
        lifecycle=stateless.lifecycle,
        retention_policy=stateless.retention_policy,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            stored,
            execution_receipt=cast(
                conversation.ProviderLaneExecutionReceipt,
                object(),
            ),
        )

    with pytest.raises(conversation.ConversationValidationError):
        conversation.IdempotencySettlementResolution(
            disposition=cast(
                conversation.IdempotencySettlementDisposition,
                "invalid",
            )
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.IdempotencySettlementResolution(
            disposition=(conversation.IdempotencySettlementDisposition.LEASED)
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.IdempotencySettlementResolution(
            disposition=(conversation.IdempotencySettlementDisposition.LEASED),
            lease_expires_at=datetime.min,
            lease_owner_token="owner",
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.IdempotencySettlementResolution(
            disposition=(conversation.IdempotencySettlementDisposition.LEASED),
            lease_expires_at=NOW,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.StoreCloseResolution(
            disposition=cast(conversation.StoreCloseDisposition, "invalid")
        )


def test_execution_staging_values_are_closed_and_redacted() -> None:
    """Reject malformed provenance values and redact store-owned handles."""
    commit = _commit()
    output = commit.output_candidates[0]
    lane = conversation.ProviderLaneExecutionReservation(
        binding=output.binding,
        mode=output.mode,
        scope=output.scope,
    )
    assert str(lane.binding.lane_id) in repr(lane)
    for values in (
        {"binding": cast(conversation.ProviderLaneBinding, object())},
        {"mode": conversation.ConversationMode.OFF},
        {"scope": cast(conversation.ProviderLaneOutputScope, "invalid")},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(lane, **values)

    reservation = conversation.ConversationExecutionReservation(
        idempotency=commit.idempotency,
        identity=commit.candidate.checkpoint.identity,
        lanes=(lane,),
    )
    assert str(reservation.identity.checkpoint_id) in repr(reservation)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            reservation,
            lanes=cast(
                tuple[conversation.ProviderLaneExecutionReservation, ...],
                [],
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(reservation, lanes=(lane, lane))
    wrong_agent = conversation.ProviderLaneExecutionReservation(
        binding=binding("wrong-agent-lane", agent="wrong-agent"),
        mode=conversation.ConversationMode.STATELESS,
        scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(reservation, lanes=(wrong_agent,))

    stage = conversation.ProviderLaneExecutionStage(
        idempotency=commit.idempotency,
        owner_token=commit.owner_token,
        identity=commit.candidate.checkpoint.identity,
        binding=output.binding,
        mode=output.mode,
        scope=output.scope,
        completed_items=output.completed_items,
        reasoning=output.reasoning,
        usage=output.usage,
        execution_receipt=output.execution_receipt,
        upstream_response_id=output.upstream_response_id,
    )
    assert str(stage.identity.checkpoint_id) in repr(stage)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            stage,
            completed_items=cast(tuple[conversation.ProviderItem, ...], []),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(stage, mode=conversation.ConversationMode.STORED)
    mismatched_receipt = conversation.ProviderLaneExecutionReceipt(
        schema_version=1,
        digest=conversation.IntegrityDigest("1" * 64),
        item_count=stage.execution_receipt.item_count,
        opaque_byte_count=stage.execution_receipt.opaque_byte_count,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(stage, execution_receipt=mismatched_receipt)

    attestation = conversation.ProviderLaneExecutionAttestation(
        schema_version=1,
        staging_id="opaque-staging-handle",
        lane_id=output.lane_id,
    )
    rendered = repr(attestation)
    assert "opaque-staging-handle" not in rendered
    assert "<redacted>" in rendered
    with pytest.raises(conversation.ConversationValidationError):
        replace(attestation, schema_version=2)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            commit,
            execution_attestations=cast(
                tuple[conversation.ProviderLaneExecutionAttestation, ...],
                (object(),),
            ),
        )


def test_checkpoint_observation_carries_context_and_rejects_bad_ids() -> None:
    """Expose safe checkpoint context and reject malformed identifiers."""
    checkpoint = _commit().candidate.checkpoint
    observation = conversation.checkpoint_observation(
        "checkpoint-staged", checkpoint
    )

    assert observation.authority_scope_digest
    assert (
        observation.parent_checkpoint_id
        == checkpoint.identity.parent_checkpoint_id
    )
    assert observation.lane_ids == tuple(
        str(lane.lane_id) for lane in checkpoint.content.lanes
    )
    projected = observation.to_mapping()
    assert projected["event"] == "checkpoint-staged"
    assert projected["authority_scope_digest"] == (
        observation.authority_scope_digest
    )
    assert projected["parent_checkpoint_id"] is None
    assert projected["lane_ids"] == observation.lane_ids
    assert "opaque-secret-sentinel" not in repr(projected)

    with pytest.raises(conversation.ConversationValidationError):
        replace(
            observation,
            authority_scope_digest=conversation.AuthorityDigest(""),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            observation,
            parent_checkpoint_id=conversation.CheckpointId(""),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(observation, lane_ids=cast(tuple[str, ...], []))
    with pytest.raises(conversation.ConversationValidationError):
        replace(observation, lane_ids=("duplicate", "duplicate"))


def test_runtime_value_objects_reject_every_open_variant() -> None:
    """Reject invalid run, publication, page, sweep, and policy values."""
    run, _snapshot, candidate = _snapshot_and_candidate()
    lane = run.lanes[0]
    invalid_lane_values = (
        {"mode": conversation.ConversationMode.OFF},
        {"mode": cast(conversation.ConversationMode, "bad")},
        {"reasoning_context": cast(conversation.ReasoningContext, "bad")},
    )
    for lane_values in invalid_lane_values:
        with pytest.raises(conversation.ConversationValidationError):
            replace(lane, **lane_values)

    invalid_runs = (
        {
            "semantics": cast(
                conversation.ConversationRequestSemantics, object()
            )
        },
        {"identity": cast(conversation.CheckpointIdentity, object())},
        {"advance": cast(conversation.ConversationAdvance, object())},
        {"lanes": ()},
        {"lanes": cast(tuple[conversation.ConversationLaneRequest, ...], [])},
        {
            "lanes": cast(
                tuple[conversation.ConversationLaneRequest, ...], (object(),)
            )
        },
        {"lanes": (lane, lane)},
        {
            "visible_delta": cast(
                tuple[conversation.VisibleTranscriptEntry, ...], []
            )
        },
        {
            "visible_delta": cast(
                tuple[conversation.VisibleTranscriptEntry, ...], (object(),)
            )
        },
        {"retention": cast(conversation.RetentionLimits, object())},
        {"boundary": cast(conversation.ConversationCommitBoundary, "bad")},
        {"provisional_response_id": None},
        {"public_response_id": None},
    )
    for run_values in invalid_runs:
        with pytest.raises(conversation.ConversationValidationError):
            replace(run, **run_values)

    bad_parent_semantics = replace(
        run.semantics,
        parent_checkpoint_id=conversation.CheckpointId("unexpected-parent"),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(run, semantics=bad_parent_semantics)

    child = conversation.CheckpointIdentity(
        conversation_id=run.identity.conversation_id,
        logical_turn_id=conversation.LogicalTurnId("validation-child-turn"),
        execution_segment_id=conversation.ExecutionSegmentId(
            "validation-child-segment"
        ),
        checkpoint_id=conversation.CheckpointId("validation-child"),
        branch_id=run.identity.branch_id,
        sequence=conversation.CheckpointSequence(1),
        parent_checkpoint_id=conversation.CheckpointId("validation-parent"),
        parent_sequence=conversation.CheckpointSequence(0),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(run, identity=child)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            run,
            advance=conversation.ResetAdvance(
                parent_checkpoint_id=conversation.CheckpointId(
                    "validation-parent"
                )
            ),
            semantics=replace(
                run.semantics,
                parent_checkpoint_id=conversation.CheckpointId(
                    "different-parent"
                ),
            ),
        )
    child_semantics = replace(
        run.semantics,
        parent_checkpoint_id=conversation.CheckpointId("validation-parent"),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            run,
            identity=child,
            advance=conversation.OrdinaryChildAdvance(
                parent_checkpoint_id=conversation.CheckpointId(
                    "different-parent"
                )
            ),
            semantics=replace(
                child_semantics,
                parent_checkpoint_id=conversation.CheckpointId(
                    "different-parent"
                ),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            run,
            identity=child,
            advance=conversation.ExplicitBranchAdvance(
                parent_checkpoint_id=conversation.CheckpointId(
                    "validation-parent"
                ),
                branch_id=conversation.ConversationBranchId(
                    "different-branch"
                ),
            ),
            semantics=child_semantics,
        )

    invalid_publication_values = (
        conversation.OutboxRecord,
        conversation.PublicResponseRecord,
        conversation.StoreLimits,
        conversation.SweepReceipt,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutboxRecord(
            intent=cast(conversation.PublicationIntent, object()),
            authority_digest=conversation.AuthorityDigest(
                "validation-authority"
            ),
            state=conversation.OutboxState.PENDING,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutboxRecord(
            intent=conversation.PublicationIntent(
                intent_id="intent",
                public_response_id=conversation.PublicResponseId("response"),
                checkpoint_id=conversation.CheckpointId("checkpoint"),
                lane_outputs=(_commit().output_candidates[0].public_output,),
            ),
            authority_digest=conversation.AuthorityDigest(
                "validation-authority"
            ),
            state=conversation.OutboxState.PENDING,
            attempts=-1,
        )
    intent = conversation.PublicationIntent(
        intent_id="validation-intent",
        public_response_id=conversation.PublicResponseId(
            "validation-response"
        ),
        checkpoint_id=conversation.CheckpointId("validation-checkpoint"),
        lane_outputs=(_commit().output_candidates[0].public_output,),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(intent, lane_outputs=())
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            intent,
            lane_outputs=cast(
                tuple[conversation.ProviderLaneOutput, ...], (object(),)
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(intent, lane_outputs=intent.lane_outputs * 2)
    pending_record = conversation.OutboxRecord(
        intent=intent,
        authority_digest=conversation.AuthorityDigest("validation-authority"),
        state=conversation.OutboxState.PENDING,
    )
    claimed_record = replace(
        pending_record,
        state=conversation.OutboxState.CLAIMED,
        attempts=1,
        lease_owner="validation-lease-owner",
        lease_expires_at=NOW,
    )
    claimed_batch = conversation.OutboxRecoveryBatch(
        disposition=conversation.OutboxRecoveryDisposition.CLAIMED,
        limit=1,
        records=(claimed_record,),
    )
    assert claimed_batch.records == (claimed_record,)
    assert (
        conversation.OutboxRecoveryBatch(
            disposition=conversation.OutboxRecoveryDisposition.EMPTY,
            limit=1,
            records=(),
        ).records
        == ()
    )
    for recovery_values in (
        {"disposition": cast(conversation.OutboxRecoveryDisposition, "bad")},
        {"limit": 0},
        {"records": (pending_record,)},
        {"records": (claimed_record, claimed_record), "limit": 2},
        {
            "disposition": conversation.OutboxRecoveryDisposition.EMPTY,
            "records": (claimed_record,),
        },
        {
            "disposition": conversation.OutboxRecoveryDisposition.CLAIMED,
            "records": (),
        },
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(claimed_batch, **recovery_values)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            pending_record,
            state=cast(conversation.OutboxState, "invalid"),
        )
    target = conversation.OutboxClaimTarget(
        authority=authority(),
        checkpoint_id=intent.checkpoint_id,
        public_response_id=intent.public_response_id,
        intent_id=intent.intent_id,
    )
    invalid_targets: tuple[Callable[[], object], ...] = (
        lambda: replace(
            target,
            authority=cast(conversation.AuthorityScope, object()),
        ),
        lambda: replace(
            target,
            checkpoint_id=conversation.CheckpointId(""),
        ),
        lambda: replace(
            target,
            public_response_id=conversation.PublicResponseId(""),
        ),
        lambda: replace(target, intent_id=""),
    )
    for invalid_target in invalid_targets:
        with pytest.raises(conversation.ConversationValidationError):
            invalid_target()
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutboxClaimResolution(
            disposition=cast(
                conversation.OutboxClaimDisposition,
                "invalid",
            )
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutboxClaimResolution(
            disposition=conversation.OutboxClaimDisposition.CLAIMED
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutboxClaimResolution(
            disposition=conversation.OutboxClaimDisposition.CLAIMED,
            record=pending_record,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.OutboxClaimResolution(
            disposition=(conversation.OutboxClaimDisposition.ACTIVELY_LEASED),
            record=pending_record,
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            pending_record,
            authority_digest=conversation.AuthorityDigest(""),
        )
    for record_values in (
        {"lease_expires_at": NOW},
        {
            "state": conversation.OutboxState.CLAIMED,
            "lease_owner": None,
            "lease_expires_at": NOW,
        },
        {
            "state": conversation.OutboxState.CLAIMED,
            "lease_owner": "lease-owner",
            "lease_expires_at": None,
        },
        {
            "state": conversation.OutboxState.PUBLISHED,
            "published_at": None,
        },
        {"published_at": datetime.min},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            replace(pending_record, **record_values)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PublicResponseRecord(
            public_response_id=conversation.PublicResponseId("response"),
            checkpoint_id=conversation.CheckpointId("checkpoint"),
            authority_digest="digest",
            tombstoned=cast(bool, 1),
        )
    for field in conversation.StoreLimits.__dataclass_fields__:
        with pytest.raises(conversation.ConversationValidationError):
            conversation.StoreLimits(**{field: 0})
    with pytest.raises(conversation.ConversationValidationError):
        conversation.SweepReceipt(expired=-1, deleted=0)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.SweepReceipt(expired=0, deleted=-1)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PruneReceipt(outbox_records=-1, idempotency_records=0)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.PruneReceipt(outbox_records=0, idempotency_records=-1)
    assert invalid_publication_values
    assert candidate.checkpoint.integrity is not None


def test_atomic_commit_and_receipt_validation_is_closed() -> None:
    """Reject incomplete atomic tuples and impossible receipt variants."""
    commit = _commit()
    invalid_commits = (
        {"candidate": cast(conversation.CheckpointCandidate, object())},
        {
            "idempotency": cast(
                conversation.RequestIdempotencyIdentity, object()
            )
        },
        {"committed_at": datetime.min},
        {"owner_token": ""},
        {"output_candidates": ()},
        {
            "output_candidates": cast(
                tuple[conversation.ProviderLaneOutputCandidate, ...], []
            )
        },
        {
            "output_candidates": cast(
                tuple[conversation.ProviderLaneOutputCandidate, ...],
                (object(),),
            )
        },
        {"output_candidates": commit.output_candidates * 2},
        {"result_mode": conversation.ConversationMode.OFF},
        {"provisional_response_id": None},
        {"head_id": conversation.NamedHeadId("head")},
        {"expected_head_revision": conversation.NamedHeadRevision(0)},
    )
    for values in invalid_commits:
        with pytest.raises(conversation.ConversationValidationError):
            replace(commit, **values)
    output = commit.output_candidates[0]
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            output,
            execution_receipt=cast(
                conversation.ProviderLaneExecutionReceipt,
                object(),
            ),
        )
    different_receipt = conversation.ProviderLaneExecutionReceipt(
        schema_version=1,
        digest=conversation.IntegrityDigest("0" * 64),
        item_count=len(output.completed_items),
        opaque_byte_count=0,
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            commit,
            output_candidates=(
                replace(output, execution_receipt=different_receipt),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            commit,
            provisional_response_id=None,
            public_response_id=None,
            outbox_intent_id=None,
        )
    wrong_item = replace(
        commit.output_candidates[0].completed_items[0],
        lane_id=conversation.ProviderLaneId("outside-checkpoint"),
    )
    wrong_output = replace(
        commit.output_candidates[0],
        lane_id=conversation.ProviderLaneId("outside-checkpoint"),
        binding=replace(
            commit.output_candidates[0].binding,
            lane_id=conversation.ProviderLaneId("outside-checkpoint"),
        ),
        completed_items=(wrong_item,),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(commit, output_candidates=(wrong_output,))
    with_head = replace(
        commit,
        head_id=conversation.NamedHeadId("head"),
        expected_head_revision=conversation.NamedHeadRevision(0),
    )
    assert with_head.head_id == conversation.NamedHeadId("head")

    checkpoint = commit.candidate.checkpoint
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AtomicCommitReceipt(
            checkpoint=cast(conversation.ConversationCheckpoint, object()),
            result=None,
            outbox=None,
            output_candidates=commit.output_candidates,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AtomicCommitReceipt(
            checkpoint=checkpoint,
            result=None,
            outbox=None,
            output_candidates=(),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AtomicCommitReceipt(
            checkpoint=checkpoint,
            result=None,
            outbox=None,
            output_candidates=cast(
                tuple[conversation.ProviderLaneOutputCandidate, ...],
                (object(),),
            ),
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AtomicCommitReceipt(
            checkpoint=checkpoint,
            result=cast(conversation.ConversationResult, object()),
            outbox=None,
            output_candidates=commit.output_candidates,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AtomicCommitReceipt(
            checkpoint=checkpoint,
            result=None,
            outbox=cast(conversation.OutboxRecord, object()),
            output_candidates=commit.output_candidates,
        )
    assert commit.public_response_id is not None
    assert checkpoint.integrity is not None
    result = conversation.ConversationResult(
        handle=conversation.StatelessConversationHandle(
            conversation_id=checkpoint.identity.conversation_id,
            checkpoint_id=checkpoint.identity.checkpoint_id,
            branch_id=checkpoint.identity.branch_id,
        ),
        reasoning=commit.output_candidates[-1].reasoning,
        checkpoint_digest=checkpoint.integrity.digest,
        lane_outputs=tuple(
            item.public_output for item in commit.output_candidates
        ),
        public_response_id=commit.public_response_id,
    )
    outbox = conversation.OutboxRecord(
        intent=conversation.PublicationIntent(
            intent_id="receipt-outbox",
            public_response_id=conversation.PublicResponseId(
                "different-response"
            ),
            checkpoint_id=checkpoint.identity.checkpoint_id,
            lane_outputs=result.lane_outputs,
        ),
        authority_digest=conversation.authority_digest(checkpoint.authority),
        state=conversation.OutboxState.PENDING,
    )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AtomicCommitReceipt(
            checkpoint=checkpoint,
            result=None,
            outbox=outbox,
            output_candidates=commit.output_candidates,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.AtomicCommitReceipt(
            checkpoint=checkpoint,
            result=result,
            outbox=outbox,
            output_candidates=commit.output_candidates,
        )


def test_resolution_page_failure_and_digest_validation_is_closed() -> None:
    """Reject impossible resolution, page, failure, and digest values."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.IdempotencyResolution(
            disposition=cast(conversation.IdempotencyDisposition, "bad")
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.IdempotencyResolution(
            disposition=conversation.IdempotencyDisposition.EXECUTE,
            checkpoint_id=conversation.CheckpointId("checkpoint"),
            owner_token="owner",
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.IdempotencyResolution(
            disposition=conversation.IdempotencyDisposition.EXECUTE
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.IdempotencyResolution(
            disposition=conversation.IdempotencyDisposition.REPLAY_COMMITTED
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.CheckpointPage(
            checkpoints=cast(
                tuple[conversation.ConversationCheckpoint, ...], []
            ),
            next_cursor=None,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.CheckpointPage(
            checkpoints=cast(
                tuple[conversation.ConversationCheckpoint, ...], (object(),)
            ),
            next_cursor=None,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.FailureDisposition(
            boundary=cast(conversation.FailureBoundary, "bad"),
            retry_rule=conversation.RetryRule.NEVER,
            fence_dispatch=False,
            preserve_parent=True,
            reconciliation_required=False,
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.FailureDisposition(
            boundary=conversation.FailureBoundary.PROVIDER_REJECTION,
            retry_rule=conversation.RetryRule.NEVER,
            fence_dispatch=cast(bool, 1),
            preserve_parent=True,
            reconciliation_required=False,
        )
    with pytest.raises(conversation.ConversationValidationError):
        request_digest_value(1)
    assert request_digest_value("digest") == (
        conversation.CanonicalRequestDigest("digest")
    )


def test_provider_runtime_contracts_reject_untyped_values() -> None:
    """Reject invalid usage, plans, and result variants."""
    lane_binding = binding()
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    for values in (
        {"input_tokens": -1},
        {"output_tokens": -1},
        {"input_tokens": cast(int, True)},
    ):
        with pytest.raises(conversation.ConversationValidationError):
            conversation.ProviderUsage(**values)
    with pytest.raises(conversation.ConversationValidationError):
        replace(plan, binding=cast(conversation.ProviderLaneBinding, object()))
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            plan,
            ledger=replace(
                plan.ledger, lane_id=conversation.ProviderLaneId("other-lane")
            ),
        )
    stored = conversation.StoredProviderPlan(
        binding=lane_binding,
        upstream_response_id=conversation.UpstreamResponseId("upstream"),
        reasoning=plan.reasoning,
    )
    first = conversation.FirstStoredProviderPlan(
        binding=lane_binding,
        reasoning=plan.reasoning,
    )
    for value in (stored, first):
        with pytest.raises(conversation.ConversationValidationError):
            replace(
                value,
                reasoning=cast(
                    conversation.EffectiveReasoningMetadata, object()
                ),
            )
    with pytest.raises(conversation.ConversationValidationError):
        replace(result, items=cast(tuple[conversation.ProviderItem, ...], []))
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            result,
            items=cast(tuple[conversation.ProviderItem, ...], (object(),)),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            result,
            reasoning=cast(conversation.EffectiveReasoningMetadata, object()),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(result, usage=cast(conversation.ProviderUsage, object()))

    candidate = _commit().output_candidates[0]
    invalid_candidate_factories: tuple[Callable[[], object], ...] = (
        lambda: replace(
            candidate,
            binding=cast(conversation.ProviderLaneBinding, object()),
        ),
        lambda: replace(
            candidate,
            binding=replace(
                candidate.binding,
                lane_id=conversation.ProviderLaneId("other"),
            ),
        ),
        lambda: replace(candidate, mode=conversation.ConversationMode.OFF),
        lambda: replace(
            candidate,
            scope=cast(conversation.ProviderLaneOutputScope, "invalid"),
        ),
        lambda: replace(
            candidate,
            completed_items=cast(tuple[conversation.ProviderItem, ...], []),
        ),
        lambda: replace(
            candidate,
            completed_items=(
                replace(
                    result.items[0],
                    lane_id=conversation.ProviderLaneId("other"),
                ),
            ),
        ),
        lambda: replace(
            candidate,
            reasoning=cast(
                conversation.EffectiveReasoningMetadata,
                object(),
            ),
        ),
        lambda: replace(
            candidate,
            usage=cast(conversation.ProviderUsage, object()),
        ),
        lambda: replace(
            candidate,
            upstream_response_id=conversation.UpstreamResponseId("private"),
        ),
    )
    for factory in invalid_candidate_factories:
        with pytest.raises(conversation.ConversationValidationError):
            factory()
    stored_candidate = replace(
        candidate,
        mode=conversation.ConversationMode.STORED,
        upstream_response_id=conversation.UpstreamResponseId("private"),
    )
    assert "private" not in repr(stored_candidate)
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            stored_candidate,
            scope=conversation.ProviderLaneOutputScope.CUMULATIVE,
        )
    public_output = candidate.public_output
    invalid_public_output_factories: tuple[Callable[[], object], ...] = (
        lambda: replace(
            public_output,
            binding_alias=conversation.SafeAlias(""),
        ),
        lambda: replace(
            public_output,
            mode=conversation.ConversationMode.OFF,
        ),
        lambda: replace(
            public_output,
            scope=cast(conversation.ProviderLaneOutputScope, "invalid"),
        ),
        lambda: replace(
            public_output,
            items=cast(tuple[conversation.ProviderItem, ...], []),
        ),
        lambda: replace(
            public_output,
            items=(
                replace(
                    result.items[0],
                    lane_id=conversation.ProviderLaneId("other"),
                ),
            ),
        ),
        lambda: replace(
            public_output,
            reasoning=cast(
                conversation.EffectiveReasoningMetadata,
                object(),
            ),
        ),
        lambda: replace(
            public_output,
            usage=cast(conversation.ProviderUsage, object()),
        ),
    )
    for factory in invalid_public_output_factories:
        with pytest.raises(conversation.ConversationValidationError):
            factory()
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            public_output,
            mode=conversation.ConversationMode.STORED,
            scope=conversation.ProviderLaneOutputScope.CUMULATIVE,
        )
    checkpoint = _commit().candidate.checkpoint
    conversation_result = conversation.ConversationResult(
        handle=conversation.StatelessConversationHandle(
            conversation_id=checkpoint.identity.conversation_id,
            checkpoint_id=checkpoint.identity.checkpoint_id,
            branch_id=checkpoint.identity.branch_id,
        ),
        reasoning=candidate.reasoning,
        checkpoint_digest=conversation.IntegrityDigest("safe-digest"),
        lane_outputs=(public_output,),
    )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            conversation_result,
            lane_outputs=cast(tuple[conversation.ProviderLaneOutput, ...], []),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            conversation_result,
            lane_outputs=(public_output, public_output),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            conversation_result,
            public_response_id=conversation.PublicResponseId(""),
        )
    stored_handle = conversation.StoredConversationHandle(
        conversation_id=checkpoint.identity.conversation_id,
        checkpoint_id=checkpoint.identity.checkpoint_id,
        branch_id=checkpoint.identity.branch_id,
        public_response_id=conversation.PublicResponseId("stored-response"),
    )
    retained_stored_result = replace(
        conversation_result,
        handle=stored_handle,
        public_response_id=conversation.PublicResponseId("stored-response"),
    )
    assert retained_stored_result.handle is stored_handle
    stored_public_output = stored_candidate.public_output
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            conversation_result,
            lane_outputs=(stored_public_output,),
        )
    with pytest.raises(conversation.ConversationValidationError):
        replace(
            conversation_result,
            handle=stored_handle,
            lane_outputs=(stored_public_output,),
            public_response_id=conversation.PublicResponseId(
                "different-response"
            ),
        )


async def test_fake_fault_clock_retry_observer_and_publisher_validation() -> (
    None
):
    """Exercise deterministic fake validation and one-shot fault behavior."""
    with pytest.raises(conversation.ConversationValidationError):
        conversation.FaultAction(label="")
    with pytest.raises(conversation.ConversationValidationError):
        conversation.FaultAction(label=cast(str, object()))
    with pytest.raises(conversation.ConversationValidationError):
        conversation.FaultAction(label="fault", pause=cast(bool, object()))
    with pytest.raises(conversation.ConversationValidationError):
        conversation.FaultAction(
            label="fault", exception=cast(BaseException, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFaultController(
            (cast(conversation.FaultAction, object()),)
        )
    action = conversation.FaultAction(label="duplicate")
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFaultController((action, action))
    controller = conversation.DeterministicFaultController(
        (
            conversation.FaultAction(
                label="once", exception=RuntimeError("once")
            ),
        )
    )
    with pytest.raises(RuntimeError, match="once"):
        await controller.reach("once")
    await controller.reach("once")
    assert controller.visited == ("once", "once")
    with pytest.raises(conversation.ConversationValidationError):
        await controller.reach("")

    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFakeClock(datetime.min)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFakeAuthorityResolver(
            cast(conversation.AuthorityScope, object())
        )
    clock = conversation.DeterministicFakeClock(NOW)
    with pytest.raises(conversation.ConversationValidationError):
        clock.set(datetime.min)
    clock.set(NOW)
    assert await clock.now() == NOW
    retry = conversation.DeterministicFakeRetryWaiter()
    with pytest.raises(conversation.ConversationValidationError):
        await retry.wait(0)
    await retry.wait(1)
    assert retry.attempts == (1,)
    observer = conversation.DeterministicFakeObserver()
    with pytest.raises(conversation.ConversationValidationError):
        await observer.publish(
            cast(conversation.ConversationObservation, object())
        )
    publisher = conversation.DeterministicFakePublisher()
    with pytest.raises(conversation.ConversationValidationError):
        await publisher.publish(cast(conversation.PublicationIntent, object()))
    first_intent = conversation.PublicationIntent(
        intent_id="same-intent",
        public_response_id=conversation.PublicResponseId("response-one"),
        checkpoint_id=conversation.CheckpointId("checkpoint-one"),
        lane_outputs=(_commit().output_candidates[0].public_output,),
    )
    await publisher.publish(first_intent)
    with pytest.raises(conversation.ConversationConflictError):
        await publisher.publish(
            replace(
                first_intent,
                checkpoint_id=conversation.CheckpointId("checkpoint-two"),
            )
        )


async def test_fault_pause_uses_only_captured_label_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every caller collection and replacement callable inert."""
    source = getsource(fakes_module)
    assert "Event" not in source
    assert "_waiters" not in source

    calls = {
        "iter": 0,
        "append": 0,
        "remove": 0,
        "add": 0,
        "contains": 0,
        "sleep": 0,
        "await": 0,
        "global": 0,
        "custom": 0,
        "attribute": 0,
    }

    class _SetSpy(set[str]):
        def __iter__(self) -> Iterator[str]:
            calls["iter"] += 1
            return super().__iter__()

        def __contains__(self, value: object) -> bool:
            calls["contains"] += 1
            return super().__contains__(value)

        def add(self, value: str) -> None:
            calls["add"] += 1
            super().add(value)

        def remove(self, value: str) -> None:
            calls["remove"] += 1
            super().remove(value)

        def append(self, value: str) -> None:
            calls["append"] += 1

        def __await__(self) -> Iterator[None]:
            calls["await"] += 1
            return iter(())

    class _IterableSpy:
        def __iter__(self) -> Iterator[object]:
            calls["iter"] += 1
            return iter(())

        def __contains__(self, _value: object) -> bool:
            calls["contains"] += 1
            return False

        def __await__(self) -> Iterator[None]:
            calls["await"] += 1
            return iter(())

    action_factory = cast(
        Callable[..., conversation.FaultAction], conversation.FaultAction
    )
    with pytest.raises(TypeError):
        action_factory(label="caller-entered", entered=_SetSpy())
    with pytest.raises(TypeError):
        action_factory(label="caller-barrier", barrier=_SetSpy())
    with pytest.raises(conversation.ConversationValidationError):
        conversation.FaultAction(
            label="caller-pause", pause=cast(bool, _IterableSpy())
        )

    label = "provider:dispatch"
    controller = conversation.DeterministicFaultController(
        (conversation.FaultAction(label=label, pause=True),)
    )
    canonical_reach = controller.reach
    canonical_wait = controller.wait_until_entered
    canonical_release = controller.release
    canonical_close = controller.close
    canonical_visited = type(controller).visited.fget
    assert canonical_visited is not None
    canonical_validate = fakes_module._validate_fault_controller
    cancelled_controller = conversation.DeterministicFaultController(
        (conversation.FaultAction(label=label, pause=True),)
    )
    cancelled_reach = cancelled_controller.reach
    cancelled_wait = cancelled_controller.wait_until_entered
    subclass_controller = conversation.DeterministicFaultController(
        (conversation.FaultAction(label=label, pause=True),)
    )
    iterable_controller = conversation.DeterministicFaultController(
        (conversation.FaultAction(label=label, pause=True),)
    )

    def custom_effect(*_args: object, **_kwargs: object) -> None:
        calls["custom"] += 1

    reach_with_extra = cast(Callable[..., Awaitable[None]], controller.reach)
    wait_with_extra = cast(
        Callable[..., Awaitable[None]], controller.wait_until_entered
    )
    release_with_extra = cast(Callable[..., None], controller.release)
    close_with_extra = cast(Callable[..., None], controller.close)
    with pytest.raises(TypeError):
        await reach_with_extra(label, custom_effect)
    with pytest.raises(TypeError):
        await wait_with_extra(label, custom_effect)
    with pytest.raises(TypeError):
        release_with_extra(label, custom_effect)
    with pytest.raises(TypeError):
        close_with_extra(custom_effect)
    controller_factory = cast(
        Callable[..., conversation.DeterministicFaultController],
        conversation.DeterministicFaultController,
    )
    with pytest.raises(TypeError):
        controller_factory((), custom_effect)

    def controller_getattribute(self: object, name: str) -> object:
        calls["attribute"] += 1
        return object.__getattribute__(self, name)

    controller_subclass = type(
        "_ControllerSubclass",
        (conversation.DeterministicFaultController,),
        {"__getattribute__": controller_getattribute},
    )
    subclass_factory = cast(
        Callable[..., conversation.DeterministicFaultController],
        controller_subclass,
    )
    with pytest.raises(conversation.ConversationValidationError):
        subclass_factory()
    assert calls["custom"] == 0
    assert calls["attribute"] == 0

    def replacement(name: str, result: object = None) -> object:
        def invoke(*_args: object, **_kwargs: object) -> object:
            calls[name] += 1
            return result

        return invoke

    async def replacement_sleep(*_args: object, **_kwargs: object) -> None:
        calls["sleep"] += 1

    replacements = {
        "DeterministicFaultController": replacement("iter"),
        "FaultAction": replacement("iter"),
        "_ATTRIBUTE_ERROR_TYPE": replacement("global"),
        "_BASE_EXCEPTION_TYPE": replacement("global"),
        "_BUILTIN_BOOL_TYPE": replacement("global"),
        "_BUILTIN_DICT_TYPE": replacement("iter"),
        "_BUILTIN_FROZENSET_TYPE": replacement("iter"),
        "_BUILTIN_LIST_TYPE": replacement("iter"),
        "_BUILTIN_SET_TYPE": replacement("iter"),
        "_BUILTIN_STR_TYPE": replacement("global"),
        "_BUILTIN_TUPLE_EXACT_TYPE": replacement("global"),
        "_BUILTIN_TUPLE_TYPE": replacement("global"),
        "_DETERMINISTIC_FAULT_CONTROLLER_REACH": replacement_sleep,
        "_DETERMINISTIC_FAULT_CONTROLLER_TYPE": replacement("iter"),
        "_DICT_SETITEM": replacement("global"),
        "_EXACT_TYPE": replacement("global"),
        "_FAULT_RENDEZVOUS_MAX_YIELDS": replacement("iter"),
        "_DICT_CONTAINS": replacement("contains", False),
        "_DICT_ITEMS": replacement("iter", ()),
        "_DICT_POP": replacement("remove"),
        "_FROZENSET_CONTAINS": replacement("contains", False),
        "_FROZENSET_ITER": replacement("iter", iter(())),
        "_INSTANCE_CHECK": replacement("global"),
        "_LENGTH": replacement("global"),
        "_LIST_APPEND": replacement("append"),
        "_LIST_CONTAINS": replacement("contains", False),
        "_LIST_ITER": replacement("iter", iter(())),
        "_OBJECT_GETATTRIBUTE": replacement("global"),
        "_OBJECT_SETATTR": replacement("global"),
        "_SET_ADD": replacement("add"),
        "_SET_CONTAINS": replacement("contains", False),
        "_SET_ITER": replacement("iter", iter(())),
        "_STR_STRIP": replacement("global"),
        "_VALIDATION_ERROR_TYPE": replacement("global"),
        "_ASYNCIO_SLEEP": replacement_sleep,
        "_FAULT_ACTION_POST_INIT": replacement("global"),
        "_asyncio_sleep": replacement_sleep,
        "_controller_or_default": replacement("iter"),
        "_fault_controller_init": replacement("iter"),
        "_fault_controller_reach": replacement_sleep,
        "_fault_controller_wait_until_entered": replacement_sleep,
        "_fault_controller_release": replacement("add"),
        "_fault_controller_close": replacement("remove"),
        "_fault_controller_collections": replacement("iter"),
        "_fault_controller_visited": replacement("iter"),
        "_public_fault_controller_close": replacement("global"),
        "_public_fault_controller_init": replacement("global"),
        "_public_fault_controller_reach": replacement("global"),
        "_public_fault_controller_release": replacement("global"),
        "_public_fault_controller_visited": replacement("global"),
        "_public_fault_controller_wait_until_entered": replacement("global"),
        "_reach_fault_controller": replacement_sleep,
        "_validate_fault_action": replacement("iter"),
        "_validate_fault_controller": replacement("iter"),
        "_validate_fault_controller_collections": replacement("iter"),
        "_validate_fault_controller_state": replacement("iter"),
        "_validate_public_fault_controller": replacement("global"),
        "cast": replacement("iter"),
    }
    for name, value in replacements.items():
        monkeypatch.setattr(fakes_module, name, value)
    for name in (
        "getattr",
        "setattr",
        "type",
        "len",
        "any",
        "isinstance",
        "tuple",
        "str",
        "bool",
        "BaseException",
        "AttributeError",
        "ConversationValidationError",
        "object",
    ):
        monkeypatch.setattr(
            fakes_module, name, replacement("global"), raising=False
        )
    monkeypatch.setattr(asyncio_module, "sleep", replacement_sleep)
    monkeypatch.setattr(type(controller), "__init__", replacement("global"))
    monkeypatch.setattr(
        type(controller),
        "visited",
        property(cast(Callable[[object], object], replacement("global"))),
    )
    monkeypatch.setattr(type(controller), "reach", replacement_sleep)
    monkeypatch.setattr(
        type(controller), "wait_until_entered", replacement_sleep
    )
    monkeypatch.setattr(type(controller), "release", replacement("add"))
    monkeypatch.setattr(type(controller), "close", replacement("remove"))

    task = create_task(canonical_reach(label))
    await canonical_wait(label)
    canonical_release(label)
    await task
    canonical_close()

    cancelled_task = create_task(cancelled_reach(label))
    await cancelled_wait(label)
    cancelled_task.cancel()
    with pytest.raises(CancelledError):
        await cancelled_task

    assert canonical_visited(controller) == (label,)
    assert cancelled_task.cancelled()
    assert calls == {name: 0 for name in calls}

    object.__setattr__(subclass_controller, "_entered_labels", _SetSpy())
    with pytest.raises(conversation.ConversationValidationError):
        canonical_validate(subclass_controller)
    object.__setattr__(iterable_controller, "_paused_labels", _IterableSpy())
    with pytest.raises(conversation.ConversationValidationError):
        canonical_validate(iterable_controller)
    with pytest.raises(AttributeError):
        object.__setattr__(controller._entered_labels, "add", _SetSpy.add)
    assert calls == {name: 0 for name in calls}


def test_fault_pause_critical_bytecode_is_closed() -> None:
    """Reject global or dynamic attribute lookup in the pause call graph."""
    visited_getter = conversation.DeterministicFaultController.visited.fget
    assert visited_getter is not None
    critical = (
        conversation.FaultAction.__post_init__,
        fakes_module._validate_fault_action,
        fakes_module._validate_fault_controller_state,
        fakes_module._fault_controller_init,
        fakes_module._fault_controller_collections,
        fakes_module._validate_fault_controller_collections,
        fakes_module._fault_controller_visited,
        fakes_module._fault_controller_reach,
        fakes_module._fault_controller_wait_until_entered,
        fakes_module._fault_controller_release,
        fakes_module._fault_controller_close,
        fakes_module._validate_public_fault_controller,
        conversation.DeterministicFaultController.__init__,
        visited_getter,
        conversation.DeterministicFaultController.reach,
        conversation.DeterministicFaultController.wait_until_entered,
        conversation.DeterministicFaultController.release,
        conversation.DeterministicFaultController.close,
        fakes_module._validate_fault_controller,
        fakes_module._controller_or_default,
        fakes_module._reach_fault_controller,
    )
    forbidden = {"LOAD_GLOBAL", "LOAD_NAME", "LOAD_ATTR", "LOAD_METHOD"}

    def nested(code: CodeType) -> Iterator[CodeType]:
        yield code
        for value in code.co_consts:
            if type(value) is CodeType:
                yield from nested(value)

    for function in critical:
        for code in nested(function.__code__):
            bad = tuple(
                (instruction.opname, instruction.argval)
                for instruction in get_instructions(code)
                if instruction.opname in forbidden
            )
            assert bad == (), (function.__qualname__, code.co_name, bad)

    controller = conversation.DeterministicFaultController()
    assert tuple(
        signature(conversation.DeterministicFaultController).parameters
    ) == ("actions",)
    assert tuple(signature(controller.reach).parameters) == ("label",)
    assert tuple(signature(controller.wait_until_entered).parameters) == (
        "label",
    )
    assert tuple(signature(controller.release).parameters) == ("label",)
    assert tuple(signature(controller.close).parameters) == ()


async def test_fault_pause_lifecycle_and_mutation_are_closed() -> None:
    """Cover pause ordering, cancellation, close, and yielded mutation."""
    label = "provider:dispatch"
    action = conversation.FaultAction(label=label, pause=True)

    release_first = conversation.DeterministicFaultController((action,))
    release_first.release(label)
    await release_first.reach(label)
    await release_first.wait_until_entered(label)
    assert release_first._entered_labels == {label}
    assert release_first._released_labels == {label}
    assert release_first._completed_labels == {label}
    with pytest.raises(conversation.ConversationValidationError):
        release_first.release(label)

    entered_first = conversation.DeterministicFaultController((action,))
    entered_task = create_task(entered_first.reach(label))
    await entered_first.wait_until_entered(label)
    entered_first.release(label)
    with pytest.raises(conversation.ConversationValidationError):
        entered_first.release(label)
    await entered_task

    baseline_tasks = all_tasks()
    cancelled = conversation.DeterministicFaultController((action,))
    cancelled_task = create_task(cancelled.reach(label))
    await cancelled.wait_until_entered(label)
    cancelled_task.cancel()
    with pytest.raises(CancelledError):
        await cancelled_task
    await sleep(0)
    assert cancelled_task.cancelled()
    assert cancelled._completed_labels == {label}
    assert all_tasks() == baseline_tasks
    with pytest.raises(conversation.ConversationValidationError):
        cancelled.release(label)

    closed = conversation.DeterministicFaultController((action,))
    closed_task = create_task(closed.reach(label))
    await closed.wait_until_entered(label)
    closed.close()
    closed.close()
    with pytest.raises(conversation.ConversationValidationError):
        await closed_task
    assert closed._completed_labels == {label}
    with pytest.raises(conversation.ConversationValidationError):
        await closed.reach(label)
    with pytest.raises(conversation.ConversationValidationError):
        await closed.wait_until_entered(label)
    with pytest.raises(conversation.ConversationValidationError):
        closed.release(label)

    closed_wait = conversation.DeterministicFaultController((action,))
    closed_wait_task = create_task(closed_wait.wait_until_entered(label))
    await sleep(0)
    closed_wait.close()
    with pytest.raises(conversation.ConversationValidationError):
        await closed_wait_task

    for invalid_label in ("", " invalid", cast(str, object())):
        invalid = conversation.DeterministicFaultController((action,))
        with pytest.raises(conversation.ConversationValidationError):
            await invalid.reach(invalid_label)
        with pytest.raises(conversation.ConversationValidationError):
            await invalid.wait_until_entered(invalid_label)
        with pytest.raises(conversation.ConversationValidationError):
            invalid.release(invalid_label)
    unknown = conversation.DeterministicFaultController((action,))
    with pytest.raises(conversation.ConversationValidationError):
        await unknown.wait_until_entered("unknown-pause")
    with pytest.raises(conversation.ConversationValidationError):
        unknown.release("unknown-pause")

    for attribute in (
        "_entered_labels",
        "_released_labels",
        "_paused_labels",
        "_collections",
    ):
        mutated = conversation.DeterministicFaultController((action,))
        mutated_task = create_task(mutated.reach(label))
        await mutated.wait_until_entered(label)
        current = getattr(mutated, attribute)
        replacement = (
            frozenset(tuple(current))
            if type(current) is frozenset
            else (
                tuple(list(current))
                if type(current) is tuple
                else set(current)
            )
        )
        object.__setattr__(mutated, attribute, replacement)
        with pytest.raises(conversation.ConversationValidationError):
            await mutated_task

    for attribute in (
        "_entered_labels",
        "_released_labels",
        "_completed_labels",
    ):
        malformed = conversation.DeterministicFaultController((action,))
        malformed_task = create_task(malformed.reach(label))
        await malformed.wait_until_entered(label)
        cast(set[object], getattr(malformed, attribute)).add(object())
        with pytest.raises(conversation.ConversationValidationError):
            await malformed_task

    waiting = conversation.DeterministicFaultController((action,))
    wait_task = create_task(waiting.wait_until_entered(label))
    await sleep(0)
    waiting._entered_labels = set()
    with pytest.raises(conversation.ConversationValidationError):
        await wait_task


async def test_fake_provider_stream_and_factory_validation() -> None:
    """Exercise canonical stream state and inert script rejection."""
    lane_binding = binding(streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFakeProviderScript(results=())
    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFakeProviderScript(
            results=(cast(conversation.ProviderResult, object()),)
        )
    script = conversation.DeterministicFakeProviderScript(results=(result,))
    provider = fakes_module._build_deterministic_fake_provider_runtime(script)
    with pytest.raises(conversation.ConversationValidationError):
        await fakes_module._dispatch_deterministic_fake_provider(
            provider,
            script,
            cast(conversation.ProviderPlan, object()),
        )
    stream = await fakes_module._open_deterministic_fake_provider_stream(
        provider, script, plan
    )
    with pytest.raises(conversation.ConversationValidationError):
        await fakes_module._terminal_deterministic_fake_provider_stream(
            provider, script, stream
        )
    items: list[conversation.ProviderItem] = []
    while True:
        try:
            items.append(
                await fakes_module._next_deterministic_fake_provider_item(
                    provider, script, stream
                )
            )
        except StopAsyncIteration:
            break
    assert tuple(items) == result.items
    assert (
        await fakes_module._terminal_deterministic_fake_provider_stream(
            provider, script, stream
        )
        == result
    )
    await fakes_module._close_deterministic_fake_provider_stream(
        provider, script, stream
    )
    with pytest.raises(StopAsyncIteration):
        await fakes_module._next_deterministic_fake_provider_item(
            provider, script, stream
        )
    with pytest.raises(conversation.ConversationConflictError):
        await fakes_module._open_deterministic_fake_provider_stream(
            provider, script, plan
        )
    diagnostics = fakes_module._deterministic_fake_provider_diagnostics(
        provider, script
    )
    assert diagnostics.plans == (plan, plan)
    assert diagnostics.streams[0].closed
    with pytest.raises(conversation.ConversationValidationError):
        conversation.fake_capability_profile(
            cast(conversation.ProviderLaneBinding, object())
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.fake_provider_result(
            cast(conversation.ProviderPlan, object()), turn=1
        )
    with pytest.raises(conversation.ConversationValidationError):
        conversation.fake_provider_result(plan, turn=0)
    stored_plan = conversation.FirstStoredProviderPlan(
        binding=lane_binding,
        reasoning=plan.reasoning,
    )
    stored_result = conversation.fake_provider_result(stored_plan, turn=2)
    assert stored_result.upstream_response_id is not None


async def test_canonical_fake_internal_state_validation_is_closed() -> None:
    """Reject every malformed inert script and canonical runtime member."""
    lane_binding = binding(streaming=True)
    plan = empty_stateless_plan(lane_binding)
    result = conversation.fake_provider_result(plan, turn=1)
    item = result.items[0]
    script = conversation.DeterministicFakeProviderScript(results=(result,))

    with pytest.raises(conversation.ConversationValidationError):
        conversation.DeterministicFaultController(
            cast(tuple[conversation.FaultAction, ...], [])
        )
    action = conversation.FaultAction(label="canonical-action")
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_fault_action(object())
    raw_action = object.__new__(conversation.FaultAction)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_fault_action(raw_action)
    with pytest.raises(conversation.ConversationValidationError):
        raw_action.__post_init__()
    for field_name, value in (
        ("label", " invalid"),
        ("label", object()),
        ("pause", object()),
        ("exception", object()),
    ):
        malformed_action = copy(action)
        object.__setattr__(malformed_action, field_name, value)
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._validate_fault_action(malformed_action)

    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_fault_controller(object())
    raw_controller = object.__new__(conversation.DeterministicFaultController)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_fault_controller(raw_controller)
    for actions, visited in (
        (object(), []),
        ({}, [object()]),
        ({1: action}, []),
        ({"different-label": action}, []),
        ({action.label: object()}, []),
    ):
        controller = conversation.DeterministicFaultController()
        object.__setattr__(controller, "_actions", actions)
        object.__setattr__(controller, "_visited", visited)
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._validate_fault_controller(controller)

    pause_action = conversation.FaultAction(
        label="canonical-pause", pause=True
    )

    def refresh_collections(
        controller: conversation.DeterministicFaultController,
    ) -> None:
        object.__setattr__(
            controller,
            "_collections",
            (
                controller._actions,
                controller._visited,
                controller._scheduled_labels,
                controller._paused_labels,
                controller._entered_labels,
                controller._released_labels,
                controller._completed_labels,
            ),
        )

    malformed_state_values: tuple[
        tuple[tuple[conversation.FaultAction, ...], dict[str, object]], ...
    ] = (
        (
            (),
            {
                "_scheduled_labels": cast(
                    frozenset[str], frozenset((object(),))
                )
            },
        ),
        (
            (),
            {"_paused_labels": cast(frozenset[str], frozenset((object(),)))},
        ),
        ((), {"_visited": cast(list[str], [object()])}),
        ((), {"_paused_labels": frozenset(("outside",))}),
        ((), {"_entered_labels": {"outside"}}),
        (
            (),
            {
                "_scheduled_labels": frozenset((pause_action.label,)),
                "_paused_labels": frozenset((pause_action.label,)),
                "_completed_labels": {pause_action.label},
            },
        ),
        (
            (action,),
            {
                "_actions": cast(
                    dict[str, conversation.FaultAction],
                    {object(): action},
                )
            },
        ),
        (
            (action,),
            {"_actions": {"different-label": action}},
        ),
        ((action,), {"_actions": {}}),
        (
            (pause_action,),
            {"_entered_labels": {pause_action.label}},
        ),
    )
    for scheduled_actions, values in malformed_state_values:
        controller = conversation.DeterministicFaultController(
            scheduled_actions
        )
        for field_name, value in values.items():
            object.__setattr__(controller, field_name, value)
        refresh_collections(controller)
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._validate_fault_controller(controller)

    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._canonical_provider_result(object())
    raw_result = object.__new__(conversation.ProviderResult)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._canonical_provider_result(raw_result)
    malformed_results: tuple[tuple[str, object], ...] = (
        ("items", []),
        ("reasoning", object()),
        ("usage", object()),
        ("upstream_response_id", object()),
    )
    for field_name, value in malformed_results:
        malformed_result = copy(result)
        object.__setattr__(malformed_result, field_name, value)
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._canonical_provider_result(malformed_result)
    raw_reasoning = object.__new__(conversation.EffectiveReasoningMetadata)
    malformed_result = copy(result)
    object.__setattr__(malformed_result, "reasoning", raw_reasoning)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._canonical_provider_result(malformed_result)
    for requested, effective in (
        (object(), None),
        (conversation.ReasoningContext.AUTO, object()),
    ):
        malformed_reasoning = copy(result.reasoning)
        object.__setattr__(malformed_reasoning, "requested", requested)
        object.__setattr__(malformed_reasoning, "effective", effective)
        malformed_result = copy(result)
        object.__setattr__(
            malformed_result,
            "reasoning",
            malformed_reasoning,
        )
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._canonical_provider_result(malformed_result)
    raw_usage = object.__new__(conversation.ProviderUsage)
    malformed_result = copy(result)
    object.__setattr__(malformed_result, "usage", raw_usage)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._canonical_provider_result(malformed_result)

    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._canonical_provider_item(object())
    raw_item = object.__new__(conversation.ProviderItem)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._canonical_provider_item(raw_item)
    malformed_items: tuple[tuple[str, object], ...] = (
        ("item_id", object()),
        ("lane_id", object()),
        ("model_call_id", object()),
        ("kind", object()),
        ("order", True),
        ("provider_index", True),
        ("phase", object()),
        ("caller", object()),
        ("canonical_input", {}),
        ("normalization_version", True),
        ("call_id", object()),
        ("complete", False),
        ("opaque_state", object()),
    )
    for field_name, value in malformed_items:
        malformed_item = copy(item)
        object.__setattr__(malformed_item, field_name, value)
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._canonical_provider_item(malformed_item)
    raw_opaque = object.__new__(conversation.OpaqueProviderState)
    malformed_item = copy(item)
    object.__setattr__(malformed_item, "opaque_state", raw_opaque)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._canonical_provider_item(malformed_item)
    opaque_item = copy(item)
    object.__setattr__(
        opaque_item,
        "opaque_state",
        conversation.OpaqueProviderState(_value=b"canonical"),
    )
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._canonical_provider_item(opaque_item)

    for value in (
        None,
        True,
        1,
        "value",
        1.5,
        (None, True, 1, "value"),
        MappingProxyType({"nested": (1, 2)}),
    ):
        fakes_module._validate_frozen_json_value(value)
    for value in (
        float("nan"),
        MappingProxyType({1: "invalid"}),
        [],
    ):
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._validate_frozen_json_value(value)
    cyclic_data: dict[str, object] = {}
    cyclic_value = MappingProxyType(cyclic_data)
    cyclic_data["cycle"] = cyclic_value
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_frozen_json_value(cyclic_value)

    state = fakes_module._build_deterministic_fake_provider_runtime(script)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_deterministic_fake_provider_runtime(
            object(), script
        )
    raw_state = object.__new__(fakes_module._DeterministicFakeProviderRuntime)
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_deterministic_fake_provider_runtime(
            raw_state, script
        )
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_deterministic_fake_provider_runtime(
            state, copy(script)
        )
    for field_name, value in (
        ("results", ()),
        ("plans", ()),
        ("plans", [object()]),
        ("streams", ()),
    ):
        malformed_state = (
            fakes_module._build_deterministic_fake_provider_runtime(script)
        )
        object.__setattr__(malformed_state, field_name, value)
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._validate_deterministic_fake_provider_runtime(
                malformed_state, script
            )

    stream_state = await fakes_module._open_deterministic_fake_provider_stream(
        state, script, plan
    )
    other_state = fakes_module._build_deterministic_fake_provider_runtime(
        script
    )
    other_stream = await fakes_module._open_deterministic_fake_provider_stream(
        other_state, script, plan
    )
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._owned_deterministic_fake_provider_stream(
            state, other_stream
        )
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_deterministic_fake_provider_stream(object())
    raw_stream = object.__new__(
        fakes_module._DeterministicFakeProviderStreamState
    )
    with pytest.raises(conversation.ConversationValidationError):
        fakes_module._validate_deterministic_fake_provider_stream(raw_stream)
    for field_name, value in (
        ("result", object()),
        ("index", True),
        ("index", -1),
        ("index", len(result.items) + 1),
        ("close_attempts", True),
        ("close_attempts", -1),
        ("closed", 0),
    ):
        malformed_stream = copy(stream_state)
        object.__setattr__(malformed_stream, field_name, value)
        with pytest.raises(conversation.ConversationValidationError):
            fakes_module._validate_deterministic_fake_provider_stream(
                malformed_stream
            )
