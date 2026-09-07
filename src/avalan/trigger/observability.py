"""Publish bounded trigger observations independently of committed work."""

from .codec import record_payload
from .preparation import TriggerPreparationService
from .records import OccurrenceDisposition, TriggerEvent
from .scheduler import SchedulerOwnedResource, TriggerScheduler
from .scheduler_types import (
    TriggerProcessResult,
    TriggerSchedulerSettings,
    TriggerTickStop,
)

from asyncio import (
    CancelledError,
    Task,
    create_task,
    get_running_loop,
    shield,
    to_thread,
    wait,
)
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from functools import partial
from json import dumps
from pathlib import Path
from typing import Protocol


async def _owned_thread_write(write: Callable[[], object]) -> None:
    """Keep a started write owned until its thread actually finishes."""
    task = create_task(to_thread(write))
    cancellation: CancelledError | None = None
    while True:
        try:
            await shield(task)
            break
        except CancelledError as error:
            if task.cancelled():
                raise
            cancellation = error
        except Exception as error:
            if cancellation is not None:
                raise cancellation from error
            raise
    if cancellation is not None:
        raise cancellation


class TriggerTickSink(Protocol):
    async def emit_tick(self, result: TriggerProcessResult) -> None: ...


@dataclass(frozen=True, slots=True)
class TriggerSinkResult:
    delivered: int
    failed: int


async def replay_events(
    events: tuple[TriggerEvent, ...], path: Path
) -> TriggerSinkResult:
    """Append committed event IDs for idempotent downstream replay."""
    assert len(events) <= 200
    delivered = 0
    for event in events:
        try:
            await _owned_thread_write(partial(_append_event, path, event))
        except (OSError, UnicodeError):
            return TriggerSinkResult(delivered, len(events) - delivered)
        delivered += 1
    return TriggerSinkResult(delivered, 0)


def _append_event(path: Path, event: TriggerEvent) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            dumps(record_payload(event), sort_keys=True, separators=(",", ":"))
            + "\n"
        )


@dataclass(slots=True)
class TriggerMetrics:
    """Export aggregate observed decisions for a textfile collector.

    These process-local counters may observe recovered decisions again;
    durable event/occurrence identities remain the authoritative ledger.
    No task queue wait or execution duration is inferred from admission.
    """

    path: Path
    decisions: dict[OccurrenceDisposition, int] = field(default_factory=dict)
    conflicts: int = 0
    admission_retries: int = 0
    healthy: bool = True
    errors: int = 0
    pending: int = 0
    remaining: bool = False
    dispatch_lag_count: int = 0
    dispatch_lag_sum: float = 0.0

    async def emit_tick(self, result: TriggerProcessResult) -> None:
        for occurrence in result.occurrences:
            kind = occurrence.disposition
            self.decisions[kind] = self.decisions.get(kind, 0) + 1
            if occurrence.run_id is not None:
                self.dispatch_lag_count += 1
                self.dispatch_lag_sum += (
                    occurrence.decided_at - occurrence.scheduled_at
                ).total_seconds()
        for span in result.ranges:
            # Unknown compressed counts cannot truthfully become counters.
            if span.exact_count is not None:
                self.decisions[span.disposition] = (
                    self.decisions.get(span.disposition, 0) + span.exact_count
                )
        self.admission_retries += result.admission_retries
        self.healthy = not (
            result.errors
            or result.unresolved
            or result.pending_operations
            or result.preparation_failures
        )
        self.conflicts += result.conflicts
        self.errors += len(result.errors)
        self.pending = result.pending_operations + len(result.unresolved)
        self.remaining = result.remaining_work
        await _owned_thread_write(
            partial(self.path.write_text, self.render(), encoding="utf-8")
        )

    def render(self) -> str:
        lines = [
            "# TYPE avalan_trigger_observed_decisions_total counter",
            *(
                f'avalan_trigger_observed_decisions_total{{disposition="{kind.value}"}}'
                f" {self.decisions.get(kind, 0)}"
                for kind in OccurrenceDisposition
            ),
            "# TYPE avalan_trigger_conflicts_total counter",
            f"avalan_trigger_conflicts_total {self.conflicts}",
            "# TYPE avalan_trigger_admission_retries_total counter",
            f"avalan_trigger_admission_retries_total {self.admission_retries}",
            "# TYPE avalan_trigger_scheduler_healthy gauge",
            f"avalan_trigger_scheduler_healthy {int(self.healthy)}",
            "# TYPE avalan_trigger_errors_total counter",
            f"avalan_trigger_errors_total {self.errors}",
            "# TYPE avalan_trigger_pending gauge",
            f"avalan_trigger_pending {self.pending}",
            "# TYPE avalan_trigger_remaining gauge",
            f"avalan_trigger_remaining {int(self.remaining)}",
            "# TYPE avalan_trigger_dispatch_lag_seconds summary",
            (
                "avalan_trigger_dispatch_lag_seconds_count"
                f" {self.dispatch_lag_count}"
            ),
            f"avalan_trigger_dispatch_lag_seconds_sum {self.dispatch_lag_sum}",
        ]
        return "\n".join(lines) + "\n"


class _ObservationOwner:
    """Retain at most one external write until it actually settles."""

    def __init__(self) -> None:
        self.task: Task[None] | None = None
        self.failures = 0

    def collect(self) -> None:
        if self.task is not None and self.task.done():
            task, self.task = self.task, None
            try:
                task.result()
            except (CancelledError, Exception):
                self.failures += 1

    def start(self, work: Coroutine[None, None, None]) -> Task[None]:
        assert self.task is None
        self.task = create_task(work)
        return self.task

    async def aclose(self) -> None:
        # The scheduler's CLOSE operation owns and bounds this join. A
        # cancelled wait must not abandon the underlying sink/thread.
        task = self.task
        if task is not None:
            task.cancel()
            while not task.done():
                try:
                    await wait((task,))
                except CancelledError:
                    continue
            self.collect()


class ObservedTriggerScheduler(TriggerScheduler):
    """Keep bounded optional observation owned after committed admission."""

    def __init__(
        self,
        preparation: TriggerPreparationService,
        *,
        settings: TriggerSchedulerSettings = TriggerSchedulerSettings(),
        owned_resources: tuple[SchedulerOwnedResource, ...] = (),
    ) -> None:
        self._observation = _ObservationOwner()
        self.sink: TriggerTickSink | None = None
        super().__init__(
            preparation,
            settings=settings,
            owned_resources=(self._observation, *owned_resources),
        )

    @property
    def sink_failures(self) -> int:
        self._observation.collect()
        return self._observation.failures

    async def _process_once(self) -> TriggerProcessResult:
        deadline = (
            get_running_loop().time() + self.settings.tick_timeout_seconds
        )
        self._observation.collect()
        if self._observation.task is not None:
            return TriggerProcessResult(
                pending_operations=1,
                remaining_work=True,
                stop=TriggerTickStop.TIME_LIMIT,
            )
        result = await super()._process_once()
        # The base reducer releases its guard before returning. Keep it
        # held through observation; the owned task's callback releases it.
        self._processing = True
        if self.sink is None or self._stopping:
            return result
        task = self._observation.start(self.sink.emit_tick(result))
        try:
            done, _ = await wait(
                (task,), timeout=max(0.0, deadline - get_running_loop().time())
            )
        except CancelledError:
            task.cancel()
            raise
        if not done:
            task.cancel()
            return replace(
                result,
                pending_operations=result.pending_operations + 1,
                remaining_work=True,
                stop=TriggerTickStop.TIME_LIMIT,
            )
        self._observation.collect()
        return result
