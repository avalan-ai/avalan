"""Coordinate bounded trigger preparation and atomic admission."""

from .admission import (
    PreparedTriggerAdmission,
    TriggerAdmissionCancelledError,
    TriggerAdmissionResult,
    TriggerCommitOutcome,
)
from .error import TriggerError, TriggerErrorCode
from .plan import AdmissionLimits, TriggerAdmissionPlan, plan_admission
from .preparation import (
    TriggerPreparationCancelledError,
    TriggerPreparationFailure,
    TriggerPreparationService,
)
from .records import TriggerCoverageSpan, TriggerOccurrence, TriggerSnapshot
from .schedule import SearchLimits
from .scheduler_operations import (
    SchedulerOperationCompletion,
    SchedulerOperationKind,
    SchedulerOperations,
    SchedulerOperationTimeout,
)
from .scheduler_types import (
    TriggerProcessResult,
    TriggerSchedulerDiagnostic,
    TriggerSchedulerSettings,
    TriggerShutdownResult,
    TriggerTickStop,
)
from .search_budget import SearchWorkBudget, SearchWorkExhausted, schedule_work
from .store import TriggerDiscoveryCursor
from .stores.pgsql import PgsqlTriggerStore, decision_time

from asyncio import (
    CancelledError,
    Event,
    Task,
    create_task,
    get_running_loop,
    shield,
    wait,
)
from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol, TypeVar

_Value = TypeVar("_Value")


class SchedulerOwnedResource(Protocol):
    """Close only resources explicitly transferred to this scheduler."""

    async def aclose(self) -> None: ...


class TriggerSchedulerCancelledError(CancelledError):
    """Preserve recovery evidence when a serving task is cancelled."""

    def __init__(self, result: TriggerShutdownResult) -> None:
        self.result = result
        super().__init__("trigger scheduler cancelled")


@dataclass(slots=True)
class _Tick:
    deadline: float
    occurrences: dict[str, TriggerOccurrence] = field(default_factory=dict)
    ranges: dict[str, TriggerCoverageSpan] = field(default_factory=dict)
    errors: list[TriggerSchedulerDiagnostic] = field(default_factory=list)
    conflicts: int = 0
    admission_retries: int = 0
    decisions: int = 0
    admissions: int = 0
    remaining: bool = False
    stop: TriggerTickStop = TriggerTickStop.COMPLETE

    def observe(self, result: TriggerAdmissionResult) -> None:
        for resolved in result.resolved:
            for value in resolved.decisions:
                if isinstance(value, TriggerOccurrence):
                    self.occurrences[value.occurrence_id] = value
                else:
                    self.ranges[value.span_id] = value


class TriggerScheduler:
    """Own one fair bounded scheduling loop over a validated service."""

    def __init__(
        self,
        preparation: TriggerPreparationService,
        *,
        settings: TriggerSchedulerSettings = TriggerSchedulerSettings(),
        owned_resources: tuple[SchedulerOwnedResource, ...] = (),
    ) -> None:
        assert isinstance(preparation, TriggerPreparationService)
        assert isinstance(settings, TriggerSchedulerSettings)
        assert isinstance(owned_resources, tuple)
        self.preparation = preparation
        self.settings = settings
        self.store = preparation.admission.store
        self.admission = preparation.admission
        self._operations = SchedulerOperations()
        self._resources = owned_resources
        self._closed_resources: set[int] = set()
        self._unresolved: list[TriggerAdmissionResult] = []
        self._preparation_failures: list[TriggerPreparationFailure] = []
        self._processing = False
        self._serving = False
        self._stopping = False
        self._serving_tick: Task[TriggerProcessResult] | None = None
        self._active_serve: Task[None] | None = None
        self._serve_stop = Event()
        self._active_tick: Task[TriggerProcessResult] | None = None
        self.last_shutdown: TriggerShutdownResult | None = None
        self._shutdown_task: Task[TriggerShutdownResult] | None = None
        self._discovery_cursor: TriggerDiscoveryCursor | None = None

    async def _clock(self) -> datetime:
        if isinstance(self.store, PgsqlTriggerStore):
            async with self.store.transaction() as unit:
                return await decision_time(unit)
        return self.store._clock()

    async def _call(
        self,
        value: Awaitable[_Value],
        kind: SchedulerOperationKind,
        tick: _Tick,
        *,
        snapshot: TriggerSnapshot | None = None,
        prepared: PreparedTriggerAdmission | None = None,
        plan: TriggerAdmissionPlan | None = None,
        resource_index: int | None = None,
    ) -> _Value:
        limit = (
            self.settings.preparation_timeout_seconds
            if kind == SchedulerOperationKind.PREPARATION
            else self.settings.transaction_timeout_seconds
        )
        return await self._operations.call(
            value,
            kind=kind,
            timeout=max(
                0.0, min(limit, tick.deadline - get_running_loop().time())
            ),
            snapshot=snapshot,
            plan=prepared.plan if prepared else plan,
            prepared=prepared,
            resource_index=resource_index,
        )

    def _diagnostic(
        self, snapshot: TriggerSnapshot | None, code: TriggerErrorCode
    ) -> TriggerSchedulerDiagnostic:
        # Never copy exception messages, configuration or backend keys.
        return TriggerSchedulerDiagnostic(
            trigger_name=snapshot.definition.name if snapshot else None,
            code=code,
            detail=code.value.encode()[
                : self.settings.diagnostic_bytes
            ].decode("utf-8", errors="ignore"),
        )

    async def _failure(
        self, snapshot: TriggerSnapshot, error: Exception, tick: _Tick
    ) -> None:
        if isinstance(error, SearchWorkExhausted):
            raise error
        code = (
            error.code
            if isinstance(error, TriggerError)
            else TriggerErrorCode.ADMISSION_RETRYABLE
        )
        if code == TriggerErrorCode.CONFLICT:
            tick.conflicts += 1
            tick.remaining = True
            return
        tick.errors.append(self._diagnostic(snapshot, code))
        if (
            isinstance(error, TriggerPreparationFailure)
            and error.resources_uncertain
        ):
            self._preparation_failures.append(error)
        permanent = code not in {
            TriggerErrorCode.ADMISSION_RETRYABLE,
            TriggerErrorCode.COMMIT_UNKNOWN,
        }
        try:
            await self._call(
                self.store.fail(
                    snapshot.definition.name,
                    expected_generation=snapshot.state.generation,
                    error_code=code,
                    attempts=self.settings.admission_retry_attempts,
                    base_seconds=self.settings.retry_base_seconds,
                    max_seconds=self.settings.retry_max_seconds,
                    permanent=permanent,
                ),
                SchedulerOperationKind.FAILURE,
                tick,
                snapshot=snapshot,
            )
        except TriggerError as conflict:
            if conflict.code != TriggerErrorCode.CONFLICT:
                raise
            tick.conflicts += 1
        tick.remaining = tick.remaining or not permanent

    async def _settle(
        self,
        prepared: PreparedTriggerAdmission,
        tick: _Tick,
        *,
        known: TriggerAdmissionResult | None = None,
    ) -> TriggerAdmissionResult:
        if known is None:
            known = next(
                (
                    value
                    for value in self._unresolved
                    if value.plan.request_ids == prepared.plan.request_ids
                ),
                None,
            )
        if (
            known is not None
            and known.outcome == TriggerCommitOutcome.COMMITTED
        ):
            self._remember(
                replace(known, prepared=prepared, cleanup_pending=True)
            )
            tick.observe(known)
        try:
            result = await self._call(
                self.preparation.release_unused_admission(prepared),
                SchedulerOperationKind.CLEANUP,
                tick,
                snapshot=prepared.plan.snapshot,
                prepared=prepared,
            )
        except SchedulerOperationTimeout:
            raise
        except Exception:
            result = TriggerAdmissionResult(
                plan=prepared.plan,
                outcome=TriggerCommitOutcome.UNKNOWN,
                prepared=prepared,
                error_code=TriggerErrorCode.COMMIT_UNKNOWN,
                cleanup_pending=True,
            )
        if (
            known is not None
            and known.outcome == TriggerCommitOutcome.COMMITTED
            and result.outcome != TriggerCommitOutcome.COMMITTED
        ):
            result = replace(known, prepared=prepared, cleanup_pending=True)
        self._remember(result)
        return result

    def _remember(self, result: TriggerAdmissionResult) -> None:
        self._unresolved = [
            previous
            for previous in self._unresolved
            if previous.plan.request_ids != result.plan.request_ids
        ]
        if (
            result.outcome == TriggerCommitOutcome.UNKNOWN
            or result.cleanup_pending
        ):
            self._unresolved.append(result)

    async def _reconcile(self, tick: _Tick) -> None:
        for previous in tuple(self._unresolved):
            if previous.prepared is not None:
                result = await self._settle(
                    previous.prepared, tick, known=previous
                )
            else:
                result = await self._call(
                    self.admission.recover(previous.plan),
                    SchedulerOperationKind.RECOVERY,
                    tick,
                    plan=previous.plan,
                    snapshot=previous.plan.snapshot,
                )
                self._remember(result)
            if result.outcome == TriggerCommitOutcome.COMMITTED:
                tick.observe(result)

    async def _completed(
        self, values: tuple[SchedulerOperationCompletion, ...], tick: _Tick
    ) -> None:
        for value in values:
            error = value.error
            if isinstance(error, TriggerPreparationCancelledError):
                if not self._stopping and value.operation.snapshot is not None:
                    await self._failure(
                        value.operation.snapshot, error.preparation, tick
                    )
                elif error.preparation.resources_uncertain:
                    self._preparation_failures.append(error.preparation)
            elif isinstance(error, TriggerAdmissionCancelledError):
                result = error.result
                prepared = result.prepared or value.operation.prepared
                if prepared is not None:
                    settled = await self._settle(prepared, tick, known=result)
                    tick.observe(settled)
                    if (
                        settled.outcome == TriggerCommitOutcome.NOT_COMMITTED
                        and not self._stopping
                    ):
                        await self._failure(
                            prepared.plan.snapshot,
                            TriggerError(
                                TriggerErrorCode.ADMISSION_RETRYABLE,
                                "admission.timeout",
                            ),
                            tick,
                        )
                else:
                    self._remember(result)
            elif isinstance(error, CancelledError):
                if value.operation.plan is not None:
                    self._remember(
                        TriggerAdmissionResult(
                            plan=value.operation.plan,
                            outcome=TriggerCommitOutcome.UNKNOWN,
                            error_code=TriggerErrorCode.COMMIT_UNKNOWN,
                        )
                    )
                if value.operation.prepared is not None:
                    settled = await self._settle(
                        value.operation.prepared, tick
                    )
                    tick.observe(settled)
                    if (
                        settled.outcome == TriggerCommitOutcome.NOT_COMMITTED
                        and not self._stopping
                    ):
                        await self._failure(
                            value.operation.prepared.plan.snapshot,
                            TriggerError(
                                TriggerErrorCode.ADMISSION_RETRYABLE,
                                "admission.timeout",
                            ),
                            tick,
                        )
                elif (
                    value.operation.kind == SchedulerOperationKind.PREPARATION
                    and value.operation.snapshot is not None
                    and not self._stopping
                ):
                    await self._failure(
                        value.operation.snapshot,
                        TriggerError(
                            TriggerErrorCode.ADMISSION_RETRYABLE,
                            "preparation.timeout",
                        ),
                        tick,
                    )
            elif error is not None:
                if not isinstance(error, Exception):
                    raise error
                if value.operation.kind == SchedulerOperationKind.RECOVERY:
                    assert value.operation.plan is not None
                    self._remember(
                        TriggerAdmissionResult(
                            plan=value.operation.plan,
                            outcome=TriggerCommitOutcome.UNKNOWN,
                            error_code=TriggerErrorCode.COMMIT_UNKNOWN,
                        )
                    )
                    continue
                snapshot = value.operation.snapshot
                prepared = value.operation.prepared
                if prepared is not None:
                    recovered = await self._settle(prepared, tick)
                    if recovered.outcome == TriggerCommitOutcome.COMMITTED:
                        tick.observe(recovered)
                        continue
                    if recovered.outcome == TriggerCommitOutcome.UNKNOWN:
                        continue
                if snapshot is not None:
                    await self._failure(snapshot, error, tick)
                else:
                    tick.errors.append(
                        self._diagnostic(
                            None, TriggerErrorCode.CAPABILITY_UNAVAILABLE
                        )
                    )
            elif value.operation.kind == SchedulerOperationKind.CLOSE:
                assert value.operation.resource_index is not None
                self._closed_resources.add(value.operation.resource_index)
            elif isinstance(value.value, PreparedTriggerAdmission):
                await self._settle(value.value, tick)
            elif isinstance(value.value, TriggerAdmissionResult):
                self._remember(value.value)
                tick.observe(value.value)
                if value.value.prepared is not None:
                    await self._settle(
                        value.value.prepared, tick, known=value.value
                    )

    async def _trigger(self, snapshot: TriggerSnapshot, tick: _Tick) -> None:
        if any(
            value.plan.snapshot.state.trigger_id == snapshot.state.trigger_id
            for value in self._unresolved
        ):
            tick.remaining = True
            return
        for attempt in range(self.settings.stale_plan_retries + 1):
            now = await self._call(
                self._clock(), SchedulerOperationKind.DISCOVERY, tick
            )
            limits = AdmissionLimits(
                decisions=min(
                    self.settings.decisions_per_trigger,
                    self.settings.decisions_per_tick - tick.decisions,
                ),
                admissions=min(
                    self.settings.admissions_per_trigger,
                    self.settings.admissions_per_tick - tick.admissions,
                ),
                search=SearchLimits(
                    years=self.settings.search_years,
                    candidates=self.settings.search_candidates,
                ),
            )
            plan = plan_admission(snapshot, now, limits=limits)
            if not plan.decisions:
                return
            recovered = await self._call(
                self.admission.recover(plan),
                SchedulerOperationKind.RECOVERY,
                tick,
                snapshot=snapshot,
                plan=plan,
            )
            if recovered.outcome == TriggerCommitOutcome.UNKNOWN:
                self._remember(recovered)
                tick.remaining = True
                return
            if recovered.outcome == TriggerCommitOutcome.COMMITTED:
                tick.observe(recovered)
                tick.decisions += len(plan.decisions)
                tick.admissions += len(plan.admission_ids)
                return
            # Count one durable retry dispatch per discovered trigger,
            # excluding internal stale-plan replans and recovered decisions.
            if attempt == 0 and snapshot.state.failure_count:
                tick.admission_retries += 1
            prepared = await self._call(
                self.preparation.prepare_admission(plan),
                SchedulerOperationKind.PREPARATION,
                tick,
                snapshot=snapshot,
            )
            try:
                result = await self._call(
                    self.admission.admit(prepared),
                    SchedulerOperationKind.ADMISSION,
                    tick,
                    snapshot=snapshot,
                    prepared=prepared,
                )
            except SchedulerOperationTimeout:
                raise
            except Exception as error:
                settled = await self._settle(prepared, tick)
                if settled.outcome == TriggerCommitOutcome.COMMITTED:
                    tick.observe(settled)
                    tick.decisions += len(plan.decisions)
                    tick.admissions += len(plan.admission_ids)
                elif settled.outcome == TriggerCommitOutcome.NOT_COMMITTED:
                    await self._failure(snapshot, error, tick)
                else:
                    tick.remaining = True
                return
            settled = await self._settle(prepared, tick, known=result)
            if settled.outcome == TriggerCommitOutcome.COMMITTED:
                tick.observe(settled)
                tick.decisions += len(plan.decisions)
                tick.admissions += len(plan.admission_ids)
                tick.remaining = tick.remaining or (
                    plan.next_at is not None and plan.next_at <= now
                )
                return
            if settled.outcome == TriggerCommitOutcome.UNKNOWN:
                tick.remaining = True
                return
            if result.contended or (
                result.error_code == TriggerErrorCode.CONFLICT
                and not result.replan_required
            ):
                tick.conflicts += 1
                tick.remaining = True
                return
            if result.replan_required:
                tick.remaining = True
                if attempt < self.settings.stale_plan_retries:
                    fresh = await self._call(
                        self.store.inspect(snapshot.definition.name),
                        SchedulerOperationKind.DISCOVERY,
                        tick,
                    )
                    if fresh is not None:
                        snapshot = fresh
                        continue
                return
            await self._failure(
                snapshot,
                TriggerError(
                    result.error_code or TriggerErrorCode.ADMISSION_RETRYABLE,
                    "admission",
                ),
                tick,
            )
            return

    async def process_once(self) -> TriggerProcessResult:
        """Own one invocation without taking ownership of its caller task."""
        if self._processing:
            raise TriggerError(
                TriggerErrorCode.CONFLICT, "scheduler.process_once"
            )
        self._processing = True
        active = create_task(self._process_once())
        self._active_tick = active
        active.add_done_callback(self._tick_finished)
        try:
            return await shield(active)
        except CancelledError:
            active.cancel()
            if active.done():
                original = self._operations.cancelled_error
                if original is not None:
                    raise original
            raise
        finally:
            if active.done() and self._active_tick is active:
                self._active_tick = None

    def _tick_finished(self, active: Task[TriggerProcessResult]) -> None:
        if self._active_tick is active:
            self._processing = False

    async def _process_once(self) -> TriggerProcessResult:
        """Process a bounded fair prefix inside the scheduler-owned task."""
        tick = _Tick(
            deadline=get_running_loop().time()
            + self.settings.tick_timeout_seconds
        )
        budget = SearchWorkBudget(
            remaining=self.settings.candidate_evaluations,
            deadline=tick.deadline,
            clock=get_running_loop().time,
        )
        with schedule_work(budget):
            try:
                await self._completed(self._operations.completed(), tick)
                if not self._operations.pending:
                    await self._reconcile(tick)
                if self._stopping:
                    tick.stop = TriggerTickStop.STOP_REQUESTED
                elif self._operations.pending:
                    tick.stop = TriggerTickStop.TIME_LIMIT
                    tick.remaining = True
                else:
                    now = await self._call(
                        self._clock(), SchedulerOperationKind.DISCOVERY, tick
                    )
                    snapshots = await self._call(
                        self.store.discover(
                            decision_time=now,
                            limit=self.settings.discovery_limit,
                            after=self._discovery_cursor,
                        ),
                        SchedulerOperationKind.DISCOVERY,
                        tick,
                    )
                    tick.remaining = (
                        len(snapshots) == self.settings.discovery_limit
                    )
                    for snapshot in snapshots:
                        if (
                            tick.decisions >= self.settings.decisions_per_tick
                            or tick.admissions
                            >= self.settings.admissions_per_tick
                        ):
                            tick.stop = TriggerTickStop.WORK_LIMIT
                            tick.remaining = True
                            break
                        try:
                            previous = self._discovery_cursor
                            wrapped = previous is None or (
                                snapshot.state.last_processed_at,
                                snapshot.state.trigger_id,
                            ) <= (
                                previous.last_processed_at,
                                previous.trigger_id,
                            )
                            self._discovery_cursor = TriggerDiscoveryCursor(
                                last_processed_at=snapshot.state.last_processed_at,
                                trigger_id=snapshot.state.trigger_id,
                                round_started_at=(
                                    now
                                    if wrapped
                                    or previous is None
                                    or snapshot.state.last_processed_at
                                    > previous.round_started_at
                                    else previous.round_started_at
                                ),
                            )
                            await self._trigger(snapshot, tick)
                        except SchedulerOperationTimeout:
                            raise
                        except Exception as error:
                            await self._failure(snapshot, error, tick)
            except SchedulerOperationTimeout:
                tick.stop = TriggerTickStop.TIME_LIMIT
                tick.remaining = True
            except SearchWorkExhausted as error:
                tick.stop = (
                    TriggerTickStop.TIME_LIMIT
                    if error.path == "search.deadline"
                    else TriggerTickStop.WORK_LIMIT
                )
                tick.remaining = True
            except Exception:
                tick.errors.append(
                    self._diagnostic(
                        None, TriggerErrorCode.ADMISSION_RETRYABLE
                    )
                )
                tick.remaining = True
            finally:
                self._processing = False
        return TriggerProcessResult(
            occurrences=tuple(tick.occurrences.values()),
            ranges=tuple(tick.ranges.values()),
            conflicts=tick.conflicts,
            admission_retries=tick.admission_retries,
            errors=tuple(tick.errors),
            unresolved=tuple(self._unresolved),
            preparation_failures=tuple(self._preparation_failures),
            remaining_work=tick.remaining,
            pending_operations=len(self._operations.pending),
            stop=tick.stop,
        )

    async def shutdown(self) -> TriggerShutdownResult:
        """Join one owned shutdown operation without cancelling its cleanup."""
        if self._shutdown_task is None or self._shutdown_task.done():
            self._shutdown_task = create_task(self._shutdown())
        try:
            return await shield(self._shutdown_task)
        except CancelledError as error:
            result = TriggerShutdownResult(
                pending_operations=max(1, len(self._operations.pending)),
                unresolved=tuple(self._unresolved),
                preparation_failures=tuple(self._preparation_failures),
            )
            raise TriggerSchedulerCancelledError(result) from error

    async def _shutdown(self) -> TriggerShutdownResult:
        """Stop owned work before closing explicitly owned resources."""
        self._stopping = True
        self._serve_stop.set()
        deadline = (
            get_running_loop().time() + self.settings.shutdown_timeout_seconds
        )
        tick = _Tick(deadline=deadline)
        loop = self._active_serve
        if loop is not None and not loop.done():
            loop.cancel()
            await wait(
                (loop,), timeout=max(0, deadline - get_running_loop().time())
            )
        serving = self._active_tick
        if serving is not None and not serving.done():
            serving.cancel()
            await wait(
                (serving,),
                timeout=max(0, deadline - get_running_loop().time()),
            )
        budget = SearchWorkBudget(
            remaining=self.settings.candidate_evaluations,
            deadline=deadline,
            clock=get_running_loop().time,
        )
        with schedule_work(budget):
            try:
                if (
                    serving is not None
                    and serving.done()
                    and not serving.cancelled()
                ):
                    try:
                        serving.result()
                    except Exception:
                        tick.errors.append(
                            self._diagnostic(
                                None, TriggerErrorCode.CAPABILITY_UNAVAILABLE
                            )
                        )
                await self._completed(
                    await self._operations.stop(
                        max(0, deadline - get_running_loop().time())
                    ),
                    tick,
                )
                if not self._operations.pending:
                    await self._reconcile(tick)
                if (
                    not self._operations.pending
                    and (serving is None or serving.done())
                    and (loop is None or loop.done())
                ):
                    for index, resource in enumerate(self._resources):
                        if index not in self._closed_resources:
                            await self._call(
                                resource.aclose(),
                                SchedulerOperationKind.CLOSE,
                                tick,
                                resource_index=index,
                            )
                            self._closed_resources.add(index)
            except SchedulerOperationTimeout:
                # Timed-out work remains owned and pending; the result below
                # reports unsettled shutdown instead of claiming it closed.
                pass
            except Exception:
                tick.errors.append(
                    self._diagnostic(
                        None, TriggerErrorCode.CAPABILITY_UNAVAILABLE
                    )
                )
        self.last_shutdown = TriggerShutdownResult(
            pending_operations=len(self._operations.pending)
            + (1 if serving is not None and not serving.done() else 0)
            + (1 if loop is not None and not loop.done() else 0),
            unresolved=tuple(self._unresolved),
            preparation_failures=tuple(self._preparation_failures),
            errors=tuple(tick.errors),
        )
        return self.last_shutdown

    async def _wait_seconds(self, result: TriggerProcessResult) -> float:
        """Translate the authoritative next instant into a monotonic wait."""
        maximum = float(self.settings.poll_interval_seconds)
        if (
            result.errors
            or result.unresolved
            or result.pending_operations
            or result.conflicts
        ):
            return maximum
        tick = _Tick(
            deadline=get_running_loop().time()
            + self.settings.transaction_timeout_seconds
        )
        eligible = await self._call(
            self.store.next_eligible_at(),
            SchedulerOperationKind.DISCOVERY,
            tick,
        )
        if eligible is None:
            return maximum
        now = await self._call(
            self._clock(), SchedulerOperationKind.DISCOVERY, tick
        )
        delay = (eligible - now).total_seconds()
        if delay <= 0:
            return 0.0 if result.remaining_work else maximum
        return min(maximum, delay)

    async def serve(
        self, *, stop: Event | None = None
    ) -> TriggerShutdownResult:
        """Own a serving scope without taking ownership of its caller."""
        if self._serving or self._stopping:
            raise TriggerError(TriggerErrorCode.CONFLICT, "scheduler.serve")
        self._serving = True
        active = create_task(self._serve(stop or Event()))
        self._active_serve = active
        try:
            await shield(active)
        except CancelledError as error:
            raise TriggerSchedulerCancelledError(
                await self.shutdown()
            ) from error
        except BaseException:
            await self.shutdown()
            raise
        finally:
            self._serving = False
        return await self.shutdown()

    async def _serve(self, signal: Event) -> None:
        """Stop all loop reads before shutdown closes shared resources."""
        waiter = create_task(signal.wait())
        shutdown_waiter = create_task(self._serve_stop.wait())
        try:
            while not signal.is_set() and not self._serve_stop.is_set():
                self._serving_tick = create_task(self.process_once())
                done, _ = await wait(
                    (self._serving_tick, waiter, shutdown_waiter),
                    return_when="FIRST_COMPLETED",
                )
                if waiter in done or shutdown_waiter in done:
                    break
                processed = self._serving_tick.result()
                await wait(
                    (waiter, shutdown_waiter),
                    timeout=await self._wait_seconds(processed),
                    return_when="FIRST_COMPLETED",
                )
        except CancelledError:
            # Shutdown owns the loop child and joins it before closing. The
            # public caller still receives its own cancellation unchanged.
            if not self._stopping:
                raise
        finally:
            for pending in (waiter, shutdown_waiter):
                pending.cancel()
                try:
                    await pending
                except CancelledError:
                    # Join the waiter cancelled above. Cancellation of the
                    # serving caller is preserved by the outer handler.
                    pass
