from ...pgsql import assert_pgsql_identifier
from ...types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ...types import (
    assert_non_negative_int as _assert_non_negative_int,
)
from ...types import (
    assert_optional_positive_number as _assert_optional_positive_number,
)
from ...types import (
    assert_positive_int as _assert_positive_int,
)

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class TaskPgsqlBenchmarkOperation(StrEnum):
    RUN_CREATION = "run_creation"
    IDEMPOTENCY_RESERVATION = "idempotency_reservation"
    ENQUEUE = "enqueue"
    CLAIM = "claim"
    HEARTBEAT = "heartbeat"
    EVENT_APPEND = "event_append"
    INSPECTION_FETCH = "inspection_fetch"
    RETENTION_SWEEP = "retention_sweep"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPgsqlBenchmarkSettings:
    worker_count: int
    run_count: int
    queue_count: int
    pool_size: int
    postgresql_version: str
    thresholds: Mapping[TaskPgsqlBenchmarkOperation, float] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _assert_positive_int(self.worker_count, "worker_count")
        _assert_positive_int(self.run_count, "run_count")
        _assert_positive_int(self.queue_count, "queue_count")
        _assert_positive_int(self.pool_size, "pool_size")
        _assert_non_empty_string(
            self.postgresql_version,
            "postgresql_version",
        )
        assert isinstance(self.thresholds, Mapping)
        for operation, threshold in self.thresholds.items():
            assert isinstance(operation, TaskPgsqlBenchmarkOperation)
            assert isinstance(threshold, int | float)
            assert not isinstance(threshold, bool)
            assert threshold > 0


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPgsqlQueueLoadProfile:
    worker_count: int
    run_count: int
    queue_count: int
    lease_seconds: int
    max_attempts: int
    abandon_limit: int
    pool_size: int
    postgresql_version: str
    retry_delay_seconds: int = 0
    min_claims_per_second: float | None = None

    def __post_init__(self) -> None:
        _assert_positive_int(self.worker_count, "worker_count")
        _assert_positive_int(self.run_count, "run_count")
        _assert_positive_int(self.queue_count, "queue_count")
        _assert_positive_int(self.lease_seconds, "lease_seconds")
        _assert_positive_int(self.max_attempts, "max_attempts")
        _assert_positive_int(self.abandon_limit, "abandon_limit")
        assert (
            self.abandon_limit <= self.run_count
        ), "abandon_limit must not exceed run_count"
        _assert_positive_int(self.pool_size, "pool_size")
        _assert_non_empty_string(
            self.postgresql_version,
            "postgresql_version",
        )
        _assert_non_negative_int(
            self.retry_delay_seconds,
            "retry_delay_seconds",
        )
        _assert_optional_positive_number(
            self.min_claims_per_second,
            "min_claims_per_second",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPgsqlEventVolumeProfile:
    run_count: int
    max_events_per_run: int
    retention_days: int
    max_unpartitioned_event_rows: int
    postgresql_version: str
    partitioning_enabled: bool = False

    def __post_init__(self) -> None:
        _assert_positive_int(self.run_count, "run_count")
        _assert_positive_int(
            self.max_events_per_run,
            "max_events_per_run",
        )
        _assert_positive_int(self.retention_days, "retention_days")
        _assert_positive_int(
            self.max_unpartitioned_event_rows,
            "max_unpartitioned_event_rows",
        )
        _assert_non_empty_string(
            self.postgresql_version,
            "postgresql_version",
        )
        assert isinstance(self.partitioning_enabled, bool)

    @property
    def expected_event_rows(self) -> int:
        return self.run_count * self.max_events_per_run


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskPgsqlBenchmarkCase:
    operation: TaskPgsqlBenchmarkOperation
    statement: str
    parameters: tuple[tuple[str, object], ...] = ()
    expected_indexes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.operation, TaskPgsqlBenchmarkOperation)
        _assert_non_empty_string(self.statement, "statement")
        assert ";" not in self.statement
        assert isinstance(self.parameters, tuple)
        for name, value in self.parameters:
            assert_pgsql_identifier(name, "parameter")
            assert value is not None
        assert isinstance(self.expected_indexes, tuple)
        for index_name in self.expected_indexes:
            assert_pgsql_identifier(index_name, "index_name")

    @property
    def parameter_map(self) -> dict[str, object]:
        return dict(self.parameters)


def task_pgsql_benchmark_cases() -> tuple[TaskPgsqlBenchmarkCase, ...]:
    return (
        TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.RUN_CREATION,
            statement=(
                'SELECT "definition_id" FROM "task_definitions" '
                'WHERE "name" = :task_name '
                'AND "version" = :task_version '
                'AND "spec_hash" = :spec_hash'
            ),
            parameters=(
                ("task_name", "benchmark-task"),
                ("task_version", "1"),
                ("spec_hash", "benchmark-spec-hash"),
            ),
            expected_indexes=("uq_task_definitions_identity",),
        ),
        TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.IDEMPOTENCY_RESERVATION,
            statement=(
                'SELECT "run_id" FROM "task_idempotency_keys" '
                'WHERE "identity_key" = :identity_key'
            ),
            parameters=(("identity_key", "benchmark-identity"),),
            expected_indexes=("uq_task_idempotency_keys_identity",),
        ),
        TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.ENQUEUE,
            statement=(
                'SELECT "queue_item_id" FROM "task_queue_items" '
                'WHERE "run_id" = :run_id '
                "AND \"state\" IN ('available', 'claimed')"
            ),
            parameters=(("run_id", "benchmark-run"),),
            expected_indexes=("uq_task_queue_items_one_active_per_run",),
        ),
        TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.CLAIM,
            statement=(
                'SELECT "queue_item_id" FROM "task_queue_items" '
                'WHERE "queue_name" = :queue_name '
                "AND \"state\" = 'available' "
                'AND "available_at" <= CURRENT_TIMESTAMP '
                'ORDER BY "priority" DESC, "available_at", "queue_item_id" '
                "LIMIT :limit FOR UPDATE SKIP LOCKED"
            ),
            parameters=(("queue_name", "default"), ("limit", 1)),
            expected_indexes=("ix_task_queue_items_claimable",),
        ),
        TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.HEARTBEAT,
            statement=(
                'SELECT "queue_item_id" FROM "task_queue_items" '
                'WHERE "queue_item_id" = :queue_item_id '
                "AND \"state\" = 'claimed' "
                'AND "claim_token" = :claim_token '
                'AND "lease_expires_at" > CURRENT_TIMESTAMP'
            ),
            parameters=(
                ("queue_item_id", "benchmark-queue-item"),
                ("claim_token", "benchmark-claim-token"),
            ),
            expected_indexes=("task_queue_items_pkey",),
        ),
        TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.EVENT_APPEND,
            statement=(
                'SELECT COALESCE(MAX("sequence"), 0) '
                'FROM "task_events" WHERE "run_id" = :run_id'
            ),
            parameters=(("run_id", "benchmark-run"),),
            expected_indexes=("ix_task_events_by_run_sequence",),
        ),
        TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.INSPECTION_FETCH,
            statement=(
                'SELECT r."run_id", a."attempt_id" '
                'FROM "task_runs" AS r '
                'LEFT JOIN "task_attempts" AS a '
                'ON a."run_id" = r."run_id" '
                'WHERE r."run_id" = :run_id '
                'ORDER BY a."attempt_number"'
            ),
            parameters=(("run_id", "benchmark-run"),),
            expected_indexes=(
                "task_runs_pkey",
                "ix_task_attempts_by_run_order",
            ),
        ),
        TaskPgsqlBenchmarkCase(
            operation=TaskPgsqlBenchmarkOperation.RETENTION_SWEEP,
            statement=(
                'SELECT "storage_key" FROM "task_artifact_bytes" '
                'WHERE "deleted_at" IS NULL '
                'AND "retention_deadline_at" <= CURRENT_TIMESTAMP '
                'ORDER BY "retention_deadline_at", "storage_key" '
                "LIMIT :limit"
            ),
            parameters=(("limit", 100),),
            expected_indexes=("ix_task_artifact_bytes_retention_deadline",),
        ),
    )


def task_pgsql_explain_statement(
    benchmark_case: TaskPgsqlBenchmarkCase,
    *,
    analyze: bool = False,
) -> str:
    assert isinstance(benchmark_case, TaskPgsqlBenchmarkCase)
    assert isinstance(analyze, bool)
    prefix = (
        "EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)"
        if analyze
        else "EXPLAIN (FORMAT TEXT)"
    )
    return f"{prefix} {benchmark_case.statement}"


def task_pgsql_benchmark_metadata(
    settings: TaskPgsqlBenchmarkSettings,
) -> dict[str, object]:
    assert isinstance(settings, TaskPgsqlBenchmarkSettings)
    return {
        "worker_count": settings.worker_count,
        "run_count": settings.run_count,
        "queue_count": settings.queue_count,
        "pool_size": settings.pool_size,
        "postgresql_version": settings.postgresql_version,
        "thresholds": {
            operation.value: threshold
            for operation, threshold in settings.thresholds.items()
        },
    }


def task_pgsql_queue_load_metadata(
    profile: TaskPgsqlQueueLoadProfile,
    *,
    elapsed_seconds: float | None = None,
) -> dict[str, object]:
    assert isinstance(profile, TaskPgsqlQueueLoadProfile)
    _assert_optional_positive_number(elapsed_seconds, "elapsed_seconds")
    metadata: dict[str, object] = {
        "worker_count": profile.worker_count,
        "run_count": profile.run_count,
        "queue_count": profile.queue_count,
        "lease_seconds": profile.lease_seconds,
        "max_attempts": profile.max_attempts,
        "abandon_limit": profile.abandon_limit,
        "pool_size": profile.pool_size,
        "postgresql_version": profile.postgresql_version,
        "retry_delay_seconds": profile.retry_delay_seconds,
    }
    if profile.min_claims_per_second is not None:
        metadata["min_claims_per_second"] = profile.min_claims_per_second
    if elapsed_seconds is not None:
        metadata["elapsed_seconds"] = elapsed_seconds
        metadata["claims_per_second"] = profile.run_count / elapsed_seconds
    return metadata


def task_pgsql_event_volume_metadata(
    profile: TaskPgsqlEventVolumeProfile,
) -> dict[str, object]:
    assert isinstance(profile, TaskPgsqlEventVolumeProfile)
    return {
        "run_count": profile.run_count,
        "max_events_per_run": profile.max_events_per_run,
        "expected_event_rows": profile.expected_event_rows,
        "retention_days": profile.retention_days,
        "max_unpartitioned_event_rows": profile.max_unpartitioned_event_rows,
        "partitioning_enabled": profile.partitioning_enabled,
        "postgresql_version": profile.postgresql_version,
    }


def task_pgsql_queue_load_issues(
    profile: TaskPgsqlQueueLoadProfile,
    *,
    claimed_run_ids: Iterable[str],
    attempt_count: int,
    stale_token_commits: int,
    reaped_claims: int,
    elapsed_seconds: float,
) -> tuple[str, ...]:
    assert isinstance(profile, TaskPgsqlQueueLoadProfile)
    assert isinstance(claimed_run_ids, Iterable)
    assert not isinstance(claimed_run_ids, str)
    _assert_non_negative_int(attempt_count, "attempt_count")
    _assert_non_negative_int(
        stale_token_commits,
        "stale_token_commits",
    )
    _assert_non_negative_int(reaped_claims, "reaped_claims")
    _assert_optional_positive_number(elapsed_seconds, "elapsed_seconds")
    claims = tuple(claimed_run_ids)
    for run_id in claims:
        _assert_non_empty_string(run_id, "claimed_run_id")
    unique_claims = set(claims)
    issues: list[str] = []
    if len(unique_claims) != len(claims):
        issues.append("queue.duplicate_claim")
    if len(claims) > profile.run_count:
        issues.append("queue.claims_extra")
    if len(unique_claims) < profile.run_count:
        issues.append("queue.claims_missing")
    if attempt_count < len(unique_claims):
        issues.append("queue.lost_attempt")
    if attempt_count > len(unique_claims):
        issues.append("queue.extra_attempt")
    if stale_token_commits:
        issues.append("queue.stale_token_commit")
    if reaped_claims > profile.abandon_limit:
        issues.append("queue.reaper_unbounded")
    if (
        profile.min_claims_per_second is not None
        and len(claims) / elapsed_seconds < profile.min_claims_per_second
    ):
        issues.append("queue.claim_throughput_low")
    return tuple(dict.fromkeys(issues))


def task_pgsql_event_volume_issues(
    profile: TaskPgsqlEventVolumeProfile,
    *,
    append_plan_lines: Iterable[str],
    fetch_plan_lines: Iterable[str],
) -> tuple[str, ...]:
    assert isinstance(profile, TaskPgsqlEventVolumeProfile)
    issues: list[str] = []
    if (
        profile.expected_event_rows > profile.max_unpartitioned_event_rows
        and not profile.partitioning_enabled
    ):
        issues.append("event.partitioning_required")
    issues.extend(
        _event_plan_issues(
            append_plan_lines,
            prefix="event.append",
        )
    )
    issues.extend(
        _event_plan_issues(
            fetch_plan_lines,
            prefix="event.fetch",
        )
    )
    return tuple(dict.fromkeys(issues))


def task_pgsql_plan_issues(
    benchmark_case: TaskPgsqlBenchmarkCase,
    plan_lines: Iterable[str],
) -> tuple[str, ...]:
    assert isinstance(benchmark_case, TaskPgsqlBenchmarkCase)
    assert isinstance(plan_lines, Iterable)
    lines = tuple(_safe_plan_line(line) for line in plan_lines)
    plan = "\n".join(line for line in lines if line)
    issues: list[str] = []
    for index_name in benchmark_case.expected_indexes:
        if index_name not in plan:
            issues.append(f"plan.missing_index.{index_name}")
    if benchmark_case.expected_indexes and "Seq Scan" in plan:
        issues.append("plan.unbounded_scan")
    if (
        benchmark_case.operation == TaskPgsqlBenchmarkOperation.CLAIM
        and "FOR UPDATE SKIP LOCKED" not in benchmark_case.statement.upper()
    ):
        issues.append("plan.duplicate_claim_risk")
    return tuple(dict.fromkeys(issues))


def _event_plan_issues(
    plan_lines: Iterable[str],
    *,
    prefix: str,
) -> tuple[str, ...]:
    assert isinstance(plan_lines, Iterable)
    assert not isinstance(plan_lines, str)
    _assert_non_empty_string(prefix, "prefix")
    lines = tuple(_safe_plan_line(line) for line in plan_lines)
    plan = "\n".join(line for line in lines if line)
    issues: list[str] = []
    if "ix_task_events_by_run_sequence" not in plan:
        issues.append(f"{prefix}_missing_index")
    if "Seq Scan" in plan:
        issues.append(f"{prefix}_unbounded_scan")
    return tuple(issues)


def _safe_plan_line(line: object) -> str:
    assert isinstance(line, str)
    return line.strip()
