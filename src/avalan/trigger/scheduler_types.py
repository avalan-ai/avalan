"""Define finite scheduler budgets and explicit lifecycle outcomes."""

from .admission import TriggerAdmissionResult
from .definition import integer
from .error import TriggerErrorCode
from .preparation import TriggerPreparationFailure
from .records import TriggerCoverageSpan, TriggerOccurrence

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerSchedulerSettings:
    """Bound each tick, preparation, transaction and shutdown operation."""

    discovery_limit: int = 100
    decisions_per_tick: int = 100
    decisions_per_trigger: int = 10
    admissions_per_tick: int = 100
    admissions_per_trigger: int = 10
    candidate_evaluations: int = 10000
    search_years: int = 8
    search_candidates: int = 10000
    tick_timeout_seconds: int = 5
    preparation_timeout_seconds: int = 2
    transaction_timeout_seconds: int = 2
    poll_interval_seconds: int = 1
    admission_retry_attempts: int = 5
    retry_base_seconds: int = 1
    retry_max_seconds: int = 60
    stale_plan_retries: int = 2
    shutdown_timeout_seconds: int = 10
    diagnostic_bytes: int = 512
    orphan_grace_seconds: int = 86400

    def __post_init__(self) -> None:
        for name, maximum in (
            ("discovery_limit", 1000),
            ("decisions_per_tick", 1000),
            ("decisions_per_trigger", 100),
            ("admissions_per_tick", 1000),
            ("admissions_per_trigger", 100),
            ("candidate_evaluations", 100000),
            ("search_years", 400),
            ("search_candidates", 100000),
            ("tick_timeout_seconds", 60),
            ("preparation_timeout_seconds", 30),
            ("transaction_timeout_seconds", 30),
            ("poll_interval_seconds", 60),
            ("admission_retry_attempts", 20),
            ("retry_base_seconds", 60),
            ("retry_max_seconds", 3600),
            ("stale_plan_retries", 10),
            ("shutdown_timeout_seconds", 60),
            ("diagnostic_bytes", 4096),
            ("orphan_grace_seconds", 2592000),
        ):
            integer(getattr(self, name), 1, maximum, "scheduler." + name)


class TriggerTickStop(StrEnum):
    COMPLETE = "complete"
    WORK_LIMIT = "work_limit"
    TIME_LIMIT = "time_limit"
    STOP_REQUESTED = "stop_requested"


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerSchedulerDiagnostic:
    """Expose a safe classified failure independently of retained resources."""

    trigger_name: str | None
    code: TriggerErrorCode
    detail: str

    def __post_init__(self) -> None:
        assert self.trigger_name is None or (
            isinstance(self.trigger_name, str) and self.trigger_name
        )
        assert isinstance(self.code, TriggerErrorCode)
        assert isinstance(self.detail, str)


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerProcessResult:
    """Report observed decisions without inventing compressed slot counts.

    Admission associations may have been committed by a competing process.
    Distinct immutable decision identities are counted once within a tick.
    Unknown resources and pending owned operations prevent settled shutdown.
    """

    occurrences: tuple[TriggerOccurrence, ...] = ()
    ranges: tuple[TriggerCoverageSpan, ...] = ()
    conflicts: int = 0
    admission_retries: int = 0
    errors: tuple[TriggerSchedulerDiagnostic, ...] = ()
    unresolved: tuple[TriggerAdmissionResult, ...] = ()
    preparation_failures: tuple[TriggerPreparationFailure, ...] = ()
    remaining_work: bool = False
    pending_operations: int = 0
    stop: TriggerTickStop = TriggerTickStop.COMPLETE

    def __post_init__(self) -> None:
        # Operations run serially; the first timeout ends the invocation.
        # At most one prior completion can precede a full discovery batch.
        integer(self.conflicts, 0, 1001, "result.conflicts")
        integer(self.admission_retries, 0, 1000, "result.admission_retries")
        integer(self.pending_operations, 0, 1000, "result.pending_operations")
        assert type(self.remaining_work) is bool
        assert isinstance(self.stop, TriggerTickStop)
        for values, kind in (
            (self.occurrences, TriggerOccurrence),
            (self.ranges, TriggerCoverageSpan),
            (self.errors, TriggerSchedulerDiagnostic),
            (self.unresolved, TriggerAdmissionResult),
            (self.preparation_failures, TriggerPreparationFailure),
        ):
            assert isinstance(values, tuple)
            assert all(isinstance(value, kind) for value in values)

    @property
    def admitted(self) -> int:
        return sum(item.run_id is not None for item in self.occurrences)

    @property
    def skipped(self) -> int | None:
        if any(item.exact_count is None for item in self.ranges):
            return None
        return sum(item.run_id is None for item in self.occurrences) + sum(
            item.exact_count
            for item in self.ranges
            if item.exact_count is not None
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerShutdownResult:
    """Distinguish stopped admission from resources needing later recovery."""

    pending_operations: int
    unresolved: tuple[TriggerAdmissionResult, ...] = ()
    preparation_failures: tuple[TriggerPreparationFailure, ...] = ()
    errors: tuple[TriggerSchedulerDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        integer(
            self.pending_operations, 0, 1000, "shutdown.pending_operations"
        )

    @property
    def settled(self) -> bool:
        return not (
            self.pending_operations
            or self.unresolved
            or self.preparation_failures
            or self.errors
        )
