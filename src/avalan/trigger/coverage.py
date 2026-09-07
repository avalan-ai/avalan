"""Validate immutable slot decisions and compressed schedule coverage."""

from .definition import integer
from .error import TriggerError, TriggerErrorCode
from .records import TriggerCoverageSpan, TriggerDefinition, TriggerOccurrence
from .schedule import SearchLimits, latest_due, next_occurrence

from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import TypeAlias

TriggerDecision: TypeAlias = TriggerOccurrence | TriggerCoverageSpan


def _first(decision: TriggerDecision) -> datetime:
    return (
        decision.scheduled_at
        if isinstance(decision, TriggerOccurrence)
        else decision.first_at
    )


def decisions_overlap(left: TriggerDecision, right: TriggerDecision) -> bool:
    """Compare half-open ranges and exact slots without timestamp rounding."""
    if (left.owner_scope_id, left.trigger_id, left.revision) != (
        right.owner_scope_id,
        right.trigger_id,
        right.revision,
    ):
        return False
    if isinstance(left, TriggerOccurrence):
        if isinstance(right, TriggerOccurrence):
            return left.scheduled_at == right.scheduled_at
        return right.first_at <= left.scheduled_at < right.until_at
    if isinstance(right, TriggerOccurrence):
        return left.first_at <= right.scheduled_at < left.until_at
    return left.first_at < right.until_at and right.first_at < left.until_at


def validate_decisions(
    definition: TriggerDefinition,
    decisions: tuple[TriggerDecision, ...],
    *,
    existing: Sequence[TriggerDecision] = (),
    limits: SearchLimits = SearchLimits(),
    count_limit: int = 1000,
) -> None:
    """Reject off-schedule, overlapping or falsely counted decisions.

    PostgreSQL supplies only intersecting historical rows under its trigger
    lock. This function does not enumerate arbitrary retained history. Null
    span counts avoid enumeration; exact claims share a finite slot budget.
    """
    assert isinstance(decisions, tuple)
    integer(len(decisions), 0, 1000, "decisions")
    integer(count_limit, 1, 100000, "count_limit")
    remaining = count_limit
    accepted: list[TriggerDecision] = []
    for decision in decisions:
        if (
            decision.owner_scope_id != definition.owner_scope_id
            or decision.trigger_id != definition.trigger_id
            or decision.revision != definition.revision
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "decision.identity"
            )
        first = _first(decision)
        if (
            latest_due(
                definition.schedule,
                first,
                first_undecided=first,
                anchor=definition.resolved_anchor,
                limits=limits,
            )
            != first
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_SCHEDULE, "decision.first_at"
            )
        if any(
            decisions_overlap(decision, other)
            for other in (*existing, *accepted)
        ):
            raise TriggerError(TriggerErrorCode.CONFLICT, "decision.coverage")
        if isinstance(decision, TriggerCoverageSpan):
            # first < until ensures subtraction stays within datetime range.
            final = latest_due(
                definition.schedule,
                decision.until_at - timedelta(microseconds=1),
                first_undecided=first,
                anchor=definition.resolved_anchor,
                limits=limits,
            )
            if final is None or final > decision.decided_at:
                raise TriggerError(
                    TriggerErrorCode.INVALID_CONFIG, "span.until_at"
                )
            if decision.exact_count is not None:
                slot: datetime | None = first
                observed = 0
                while slot is not None and slot < decision.until_at:
                    if remaining == 0:
                        raise TriggerError(
                            TriggerErrorCode.SEARCH_BUDGET_EXHAUSTED,
                            "span.exact_count",
                        )
                    remaining -= 1
                    observed += 1
                    if slot == final:
                        break
                    slot = next_occurrence(
                        definition.schedule,
                        slot,
                        anchor=definition.resolved_anchor,
                        limits=limits,
                    )
                if observed != decision.exact_count:
                    raise TriggerError(
                        TriggerErrorCode.INVALID_CONFIG, "span.exact_count"
                    )
        accepted.append(decision)
