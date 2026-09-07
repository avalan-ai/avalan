"""Calculate bounded immutable admission proposals outside store locks."""

from .definition import (
    MisfirePolicy,
    RecurringPolicy,
    integer,
    timestamp,
    utc_text,
)
from .error import TriggerError, TriggerErrorCode
from .records import (
    OCCURRENCE_NAMESPACE,
    OccurrenceDisposition,
    TriggerSnapshot,
    occurrence_id,
)
from .schedule import SearchLimits
from .search import ScheduleSearch

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from json import dumps
from typing import TypeAlias
from uuid import uuid5


@dataclass(frozen=True, slots=True, kw_only=True)
class AdmissionLimits:
    decisions: int = 10
    admissions: int = 10
    search: SearchLimits = field(default_factory=SearchLimits)

    def __post_init__(self) -> None:
        integer(self.decisions, 1, 100, "admission.decisions")
        integer(self.admissions, 1, 100, "admission.admissions")
        assert isinstance(self.search, SearchLimits)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedOccurrence:
    scheduled_at: datetime
    disposition: OccurrenceDisposition

    def __post_init__(self) -> None:
        object.__setattr__(self, "scheduled_at", timestamp(self.scheduled_at))
        if not isinstance(
            self.disposition, OccurrenceDisposition
        ) or self.disposition not in (
            OccurrenceDisposition.ADMITTED,
            OccurrenceDisposition.EXPIRED,
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "plan.disposition"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class PlannedCoverage:
    first_at: datetime
    until_at: datetime
    disposition: OccurrenceDisposition

    def __post_init__(self) -> None:
        object.__setattr__(self, "first_at", timestamp(self.first_at))
        object.__setattr__(self, "until_at", timestamp(self.until_at))
        if (
            self.first_at >= self.until_at
            or not isinstance(self.disposition, OccurrenceDisposition)
            or self.disposition
            not in (
                OccurrenceDisposition.EXPIRED,
                OccurrenceDisposition.SKIPPED_MISFIRE,
                OccurrenceDisposition.COALESCED,
            )
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "plan.coverage"
            )


PlannedDecision: TypeAlias = PlannedOccurrence | PlannedCoverage


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerAdmissionPlan:
    snapshot: TriggerSnapshot
    prepared_at: datetime
    decisions: tuple[PlannedDecision, ...]
    next_at: datetime | None
    limits: AdmissionLimits

    def __post_init__(self) -> None:
        assert isinstance(self.snapshot, TriggerSnapshot)
        assert isinstance(self.decisions, tuple)
        assert isinstance(self.limits, AdmissionLimits)
        integer(
            len(self.decisions), 0, self.limits.decisions, "plan.decisions"
        )
        object.__setattr__(self, "prepared_at", timestamp(self.prepared_at))
        if self.next_at is not None:
            object.__setattr__(self, "next_at", timestamp(self.next_at))
        for decision in self.decisions:
            assert isinstance(decision, PlannedOccurrence | PlannedCoverage)
        if len(self.request_ids) != len(set(self.request_ids)):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "plan.identities"
            )

    def identity(self, decision: PlannedDecision) -> str:
        """Derive stable identities before any submission allocation."""
        state = self.snapshot.state
        if isinstance(decision, PlannedOccurrence):
            return occurrence_id(
                state.owner_scope_id,
                state.trigger_id,
                state.revision,
                decision.scheduled_at,
            )
        text = dumps(
            [
                "span",
                state.owner_scope_id.value,
                state.trigger_id,
                state.revision,
                utc_text(decision.first_at),
                utc_text(decision.until_at),
                decision.disposition.value,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return str(uuid5(OCCURRENCE_NAMESPACE, text))

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(self.identity(decision) for decision in self.decisions)

    @property
    def admission_ids(self) -> tuple[str, ...]:
        return tuple(
            self.identity(decision)
            for decision in self.decisions
            if isinstance(decision, PlannedOccurrence)
            and decision.disposition is OccurrenceDisposition.ADMITTED
        )


def plan_admission(
    snapshot: TriggerSnapshot,
    decision_time: datetime,
    *,
    limits: AdmissionLimits = AdmissionLimits(),
) -> TriggerAdmissionPlan:
    """Plan a gap-free prefix without performing task or backend work."""
    now = timestamp(decision_time)
    current = snapshot.state.next_at
    definition = snapshot.definition
    search = ScheduleSearch(limits.search)
    decisions: list[PlannedDecision] = []
    admissions = 0
    latest = (
        None
        if current is None
        else search.latest(
            definition.schedule,
            now,
            first=current,
            anchor=definition.resolved_anchor,
        )
    )
    if latest is not None:
        assert current is not None
        policy = definition.policy
        if isinstance(policy, RecurringPolicy) and latest != current:
            if policy.misfire is MisfirePolicy.SKIP:
                following = search.next(
                    definition.schedule,
                    latest,
                    anchor=definition.resolved_anchor,
                )
                # Recurring schedules have a next slot or raise a typed bound.
                assert following is not None
                decisions.append(
                    PlannedCoverage(
                        first_at=current,
                        until_at=following,
                        disposition=OccurrenceDisposition.SKIPPED_MISFIRE,
                    )
                )
                current = following
            elif policy.misfire is MisfirePolicy.LATEST:
                decisions.append(
                    PlannedCoverage(
                        first_at=current,
                        until_at=latest,
                        disposition=OccurrenceDisposition.COALESCED,
                    )
                )
                current = latest
        while (
            current is not None
            and current <= now
            and len(decisions) < limits.decisions
        ):
            # Subtract instants to avoid overflowing a grace deadline.
            expired = now - current > timedelta(
                seconds=policy.misfire_grace_seconds
            )
            if not expired and admissions >= limits.admissions:
                break
            if (
                expired
                and isinstance(policy, RecurringPolicy)
                and policy.misfire is MisfirePolicy.ALL
            ):
                boundary = now - timedelta(
                    seconds=policy.misfire_grace_seconds
                )
                # Expiration is strict; equality belongs to the grace interval.
                if boundary > current:
                    last_expired = search.latest(
                        definition.schedule,
                        boundary - timedelta(microseconds=1),
                        first=current,
                        anchor=definition.resolved_anchor,
                    )
                    if last_expired is not None and last_expired != current:
                        following = search.next(
                            definition.schedule,
                            last_expired,
                            anchor=definition.resolved_anchor,
                        )
                        assert following is not None
                        decisions.append(
                            PlannedCoverage(
                                first_at=current,
                                until_at=following,
                                disposition=OccurrenceDisposition.EXPIRED,
                            )
                        )
                        current = following
                        continue
            decisions.append(
                PlannedOccurrence(
                    scheduled_at=current,
                    disposition=(
                        OccurrenceDisposition.EXPIRED
                        if expired
                        else OccurrenceDisposition.ADMITTED
                    ),
                )
            )
            admissions += not expired
            current = search.next(
                definition.schedule, current, anchor=definition.resolved_anchor
            )
    return TriggerAdmissionPlan(
        snapshot=snapshot,
        prepared_at=now,
        decisions=tuple(decisions),
        next_at=current,
        limits=limits,
    )
