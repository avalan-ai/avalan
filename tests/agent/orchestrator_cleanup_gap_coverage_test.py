"""Cover deterministic orchestrator cleanup convergence boundaries."""

from asyncio import CancelledError, Event, Task, create_task, gather
from dataclasses import dataclass
from logging import getLogger
from typing import Any
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock, create_autospec, patch

from avalan.agent.engine import EngineAgent
from avalan.agent.execution import AgentExecution, AgentExecutionStatus
from avalan.agent.orchestrator import (
    Orchestrator,
    _PendingProviderCleanup,
    _ProviderCleanupState,
)
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.event.manager import EventManager
from avalan.memory.manager import MemoryManager
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.tool.manager import ToolManager


@dataclass(frozen=True)
class _ResponseProbe:
    """Retain a strict response double and its cleanup methods."""

    value: TextGenerationResponse
    cancel: AsyncMock
    aclose: AsyncMock


@dataclass(frozen=True)
class _ExecutionProbe:
    """Retain a strict execution double and its settlement method."""

    value: AgentExecution
    settle_provider_exit: AsyncMock


@dataclass(frozen=True)
class _EngineAgentProbe:
    """Retain a strict engine-agent double and its cleanup methods."""

    value: EngineAgent
    acknowledge_provider_handoff: Mock
    drain_pending_provider_cleanups: AsyncMock
    sync_messages: AsyncMock


def _response(*, cleanup_complete: bool = True) -> _ResponseProbe:
    """Return a signature-enforced provider response cleanup probe."""
    response = create_autospec(
        TextGenerationResponse,
        instance=True,
        spec_set=True,
    )
    response.cleanup_complete = cleanup_complete
    cancel = response.cancel
    aclose = response.aclose
    assert isinstance(cancel, AsyncMock)
    assert isinstance(aclose, AsyncMock)
    cancel.return_value = None
    aclose.return_value = None
    return _ResponseProbe(
        value=response,
        cancel=cancel,
        aclose=aclose,
    )


def _execution(
    *,
    status: AgentExecutionStatus = AgentExecutionStatus.COMPLETED,
) -> _ExecutionProbe:
    """Return a signature-enforced execution terminalization probe."""
    execution = create_autospec(
        AgentExecution,
        instance=True,
        spec_set=True,
    )
    execution.status = status
    settle_provider_exit = execution.settle_provider_exit
    assert isinstance(settle_provider_exit, AsyncMock)
    settle_provider_exit.return_value = ()
    return _ExecutionProbe(
        value=execution,
        settle_provider_exit=settle_provider_exit,
    )


def _engine_agent() -> _EngineAgentProbe:
    """Return a signature-enforced engine-agent cleanup probe."""
    engine_agent = create_autospec(
        EngineAgent,
        instance=True,
        spec_set=True,
    )
    acknowledge_provider_handoff = engine_agent.acknowledge_provider_handoff
    drain_pending_provider_cleanups = (
        engine_agent.drain_pending_provider_cleanups
    )
    sync_messages = engine_agent.sync_messages
    assert isinstance(acknowledge_provider_handoff, Mock)
    assert isinstance(drain_pending_provider_cleanups, AsyncMock)
    assert isinstance(sync_messages, AsyncMock)
    acknowledge_provider_handoff.return_value = None
    drain_pending_provider_cleanups.return_value = ()
    sync_messages.return_value = None
    return _EngineAgentProbe(
        value=engine_agent,
        acknowledge_provider_handoff=acknowledge_provider_handoff,
        drain_pending_provider_cleanups=drain_pending_provider_cleanups,
        sync_messages=sync_messages,
    )


def _pending_cleanup(
    *,
    response: _ResponseProbe | None = None,
    execution: _ExecutionProbe | None = None,
    engine_agent: _EngineAgentProbe | None = None,
) -> _PendingProviderCleanup:
    """Return a pending cleanup owner with deterministic collaborators."""
    response = response or _response()
    execution = execution or _execution()
    engine_agent = engine_agent or _engine_agent()
    return _PendingProviderCleanup(
        response.value,
        execution.value,
        engine_agent.value,
    )


def _orchestrator() -> Orchestrator:
    """Return a real orchestrator with strict boundary collaborators."""
    model_manager = create_autospec(
        ModelManager,
        instance=True,
        spec_set=True,
    )
    memory = create_autospec(
        MemoryManager,
        instance=True,
        spec_set=True,
    )
    tool = create_autospec(
        ToolManager,
        instance=True,
        spec_set=True,
    )
    event_manager = create_autospec(
        EventManager,
        instance=True,
        spec_set=True,
    )
    event_manager_close = event_manager.aclose
    assert isinstance(event_manager_close, AsyncMock)
    event_manager_close.return_value = None
    return Orchestrator(
        getLogger(),
        model_manager,
        memory,
        tool,
        event_manager,
        [],
        exit_memory=False,
    )


class _ZeroDeadlinePendingProviderCleanup(_PendingProviderCleanup):
    """Use the production poller with a deterministic zero test deadline."""

    @staticmethod
    async def _poll_task(
        task: Task[Any],
        stage: str,
        *,
        timeout: float = 0.0,
    ) -> tuple[bool, tuple[BaseException, ...]]:
        """Poll a real cleanup task without wall-clock delay."""
        assert timeout >= 0.0
        return await _PendingProviderCleanup._poll_task(
            task,
            stage,
            timeout=0.0,
        )


class PendingProviderCleanupGapCoverageTest(IsolatedAsyncioTestCase):
    """Exercise retained-task and convergence edge outcomes."""

    async def test_completed_owner_ignores_cleanup_and_inactive_attempts(
        self,
    ) -> None:
        owner = _pending_cleanup()
        owner.complete_transfer()

        owner.require_cleanup(cancelled=True)

        self.assertIs(owner.state, _ProviderCleanupState.COMPLETE)
        self.assertIsNone(owner.cancelled)
        self.assertEqual(await owner._run_attempt(), ())
        self.assertEqual(await owner.converge(), ())

    async def test_observer_ignores_cancelled_and_unobservable_tasks(
        self,
    ) -> None:
        cancelled_task = create_task(self._raise_cancelled())
        with self.assertRaises(CancelledError):
            await cancelled_task
        owner = _pending_cleanup()

        owner._observe_task(cancelled_task)
        pending = Event()

        async def wait_for_release() -> None:
            await pending.wait()

        pending_task = create_task(wait_for_release())
        owner._observe_task(pending_task)
        pending_task.cancel()
        with self.assertRaises(CancelledError):
            await pending_task

    async def test_poll_task_reports_stubborn_cancellation(self) -> None:
        started = Event()
        release = Event()

        async def suppress_cancellation() -> None:
            started.set()
            try:
                await release.wait()
            except CancelledError:
                await release.wait()

        task = create_task(suppress_cancellation())
        await started.wait()

        done, failures = await _PendingProviderCleanup._poll_task(
            task,
            "stubborn",
            timeout=0.0,
        )

        self.assertFalse(done)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], TimeoutError)
        self.assertFalse(task.done())
        release.set()
        await task

    async def test_poll_task_distinguishes_cancel_outcomes(self) -> None:
        cancelled_started = Event()
        cancelled_release = Event()

        async def propagate_cancellation() -> None:
            cancelled_started.set()
            await cancelled_release.wait()

        cancelled_task = create_task(propagate_cancellation())
        await cancelled_started.wait()
        (
            cancelled_done,
            cancelled_failures,
        ) = await _PendingProviderCleanup._poll_task(
            cancelled_task,
            "cancelled",
            timeout=0.0,
        )

        failed_started = Event()
        failed_release = Event()
        cancellation_failure = RuntimeError("cancel handler failed")

        async def fail_during_cancellation() -> None:
            failed_started.set()
            try:
                await failed_release.wait()
            except CancelledError as error:
                raise cancellation_failure from error

        failed_task = create_task(fail_during_cancellation())
        await failed_started.wait()
        (
            failed_done,
            failed_failures,
        ) = await _PendingProviderCleanup._poll_task(
            failed_task,
            "failing cancellation",
            timeout=0.0,
        )

        suppressed_started = Event()
        suppressed_release = Event()

        async def finish_during_cancellation() -> None:
            suppressed_started.set()
            try:
                await suppressed_release.wait()
            except CancelledError:
                return

        suppressed_task = create_task(finish_during_cancellation())
        await suppressed_started.wait()
        (
            suppressed_done,
            suppressed_failures,
        ) = await _PendingProviderCleanup._poll_task(
            suppressed_task,
            "suppressed cancellation",
            timeout=0.0,
        )

        self.assertTrue(cancelled_done)
        self.assertEqual(len(cancelled_failures), 1)
        self.assertIsInstance(cancelled_failures[0], TimeoutError)
        self.assertTrue(failed_done)
        self.assertEqual(len(failed_failures), 2)
        self.assertIsInstance(failed_failures[0], TimeoutError)
        self.assertIs(failed_failures[1], cancellation_failure)
        self.assertTrue(suppressed_done)
        self.assertEqual(len(suppressed_failures), 1)
        self.assertIsInstance(suppressed_failures[0], TimeoutError)

    async def test_run_attempt_retains_operation_when_poll_is_incomplete(
        self,
    ) -> None:
        response = _response(cleanup_complete=True)
        execution = _execution(status=AgentExecutionStatus.COMPLETED)
        engine_agent = _engine_agent()
        cancel_started = Event()
        cancellation_observed = Event()
        release = Event()

        async def stubborn_cancel() -> None:
            cancel_started.set()
            try:
                await release.wait()
            except CancelledError:
                cancellation_observed.set()
                await release.wait()

        response.cancel.side_effect = stubborn_cancel
        owner = _ZeroDeadlinePendingProviderCleanup(
            response.value,
            execution.value,
            engine_agent.value,
        )
        owner.require_cleanup(cancelled=False)

        failures = await owner._run_attempt()

        self.assertTrue(cancel_started.is_set())
        self.assertTrue(cancellation_observed.is_set())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], TimeoutError)
        retained_task = owner._cancel_task
        self.assertIsNotNone(retained_task)
        assert retained_task is not None
        self.assertFalse(retained_task.done())
        self.assertIs(owner.state, _ProviderCleanupState.CLEANUP_REQUIRED)
        release.set()
        await retained_task

    async def test_converge_reports_cancelled_retained_attempt(self) -> None:
        owner = _pending_cleanup()
        owner.require_cleanup(cancelled=False)
        cancelled_task = create_task(self._raise_cancelled_tuple())
        with self.assertRaises(CancelledError):
            await cancelled_task
        owner._attempt_task = cancelled_task

        failures = await owner.converge()

        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], CancelledError)
        self.assertIsNone(owner._attempt_task)

    async def test_run_sync_propagates_engine_memory_failure(self) -> None:
        engine_agent = _engine_agent()
        sync_failure = RuntimeError("engine memory synchronization failed")
        engine_agent.sync_messages.side_effect = sync_failure
        owner = _pending_cleanup(engine_agent=engine_agent)
        owner.complete_transfer()

        with self.assertRaises(RuntimeError) as raised:
            await owner._run_sync()

        self.assertIs(raised.exception, sync_failure)
        engine_agent.sync_messages.assert_awaited_once_with(owner.execution)

    @staticmethod
    async def _raise_cancelled() -> None:
        """Raise cancellation as one completed observer input."""
        raise CancelledError

    @staticmethod
    async def _raise_cancelled_tuple() -> tuple[BaseException, ...]:
        """Raise cancellation as one retained convergence attempt."""
        raise CancelledError


class OrchestratorSettlementGapCoverageTest(IsolatedAsyncioTestCase):
    """Exercise failure aggregation across provider ownership boundaries."""

    async def test_engine_call_failure_collects_stage_errors(self) -> None:
        orchestrator = _orchestrator()
        engine_agent = _engine_agent()
        execution = _execution(status=AgentExecutionStatus.RUNNING)
        drain_failure = RuntimeError("engine drain failed")
        settlement_failure = ValueError("execution settlement failed")
        engine_agent.drain_pending_provider_cleanups.side_effect = (
            drain_failure
        )
        execution.settle_provider_exit.side_effect = settlement_failure
        primary_failure = OSError("engine call failed")

        await orchestrator._settle_engine_call_failure(
            engine_agent.value,
            execution.value,
            primary_failure,
        )

        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            [
                (
                    "post-provider cleanup failure: RuntimeError: "
                    "engine drain failed"
                ),
                (
                    "post-provider cleanup failure: ValueError: "
                    "execution settlement failed"
                ),
            ],
        )

    async def test_engine_call_failure_handles_cancelled_cleanup_task(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        primary_failure = RuntimeError("engine call interrupted")
        engine_agent = _engine_agent()
        execution = _execution(status=AgentExecutionStatus.RUNNING)
        drain_started = Event()
        settlement_started = Event()
        release = Event()

        async def drain_engine(
            selected_execution: AgentExecution | None = None,
            *,
            abandon_unclaimed: bool = False,
        ) -> tuple[BaseException, ...]:
            self.assertIs(selected_execution, execution.value)
            self.assertTrue(abandon_unclaimed)
            drain_started.set()
            await release.wait()
            return ()

        async def settle_execution(
            *,
            cancelled: bool,
        ) -> tuple[BaseException, ...]:
            self.assertFalse(cancelled)
            settlement_started.set()
            await release.wait()
            return ()

        engine_agent.drain_pending_provider_cleanups.side_effect = drain_engine
        execution.settle_provider_exit.side_effect = settle_execution
        settlement = create_task(
            orchestrator._settle_engine_call_failure(
                engine_agent.value,
                execution.value,
                primary_failure,
            )
        )
        await gather(drain_started.wait(), settlement_started.wait())

        self.assertTrue(settlement.cancel())
        release_task = create_task(self._release(release))
        await gather(settlement, release_task)

        self.assertFalse(settlement.cancelled())
        self.assertEqual(getattr(primary_failure, "__notes__", ()), ())

    async def test_provider_handoff_failure_collects_boundary_errors(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        response = _response()
        execution = _execution(status=AgentExecutionStatus.RUNNING)
        execution.settle_provider_exit.side_effect = RuntimeError(
            "execution settlement failed"
        )
        engine_agent = _engine_agent()
        engine_agent.acknowledge_provider_handoff.side_effect = RuntimeError(
            "engine acknowledgement failed"
        )
        engine_agent.drain_pending_provider_cleanups.side_effect = (
            RuntimeError("engine drain failed")
        )
        owner = _pending_cleanup(
            response=response,
            execution=execution,
            engine_agent=engine_agent,
        )
        orchestrator._pending_provider_cleanups[id(response.value)] = owner
        primary_failure = ValueError("handoff failed")

        with patch.object(
            owner,
            "converge",
            autospec=True,
            side_effect=RuntimeError("owner convergence failed"),
        ) as converge:
            await orchestrator._settle_provider_handoff_failure(
                engine_agent.value,
                execution.value,
                response.value,
                owner,
                primary_failure,
            )

        self.assertTrue(owner.cleanup_required)
        self.assertFalse(owner.cancelled)
        converge.assert_awaited_once_with()
        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            [
                (
                    "post-provider cleanup failure: RuntimeError: "
                    "execution settlement failed"
                ),
                (
                    "post-provider cleanup failure: RuntimeError: "
                    "engine acknowledgement failed"
                ),
                (
                    "post-provider cleanup failure: RuntimeError: "
                    "owner convergence failed"
                ),
                (
                    "post-provider cleanup failure: RuntimeError: "
                    "engine drain failed"
                ),
            ],
        )

    async def test_execution_provider_exit_attaches_settlement_failure(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        settlement_failure = RuntimeError("provider exit settlement failed")
        execution = _execution(status=AgentExecutionStatus.RUNNING)
        execution.settle_provider_exit.return_value = (settlement_failure,)
        primary_failure = ValueError("provider call failed")

        await orchestrator._settle_execution_provider_exit(
            execution.value,
            primary_failure,
            cancelled=False,
        )

        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            [
                "post-provider cleanup failure: RuntimeError: "
                "provider exit settlement failed"
            ],
        )

    async def test_execution_provider_exit_handles_cancelled_cleanup_task(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        primary_failure = RuntimeError("provider call interrupted")
        execution = _execution(status=AgentExecutionStatus.RUNNING)
        execution.settle_provider_exit.side_effect = CancelledError

        await orchestrator._settle_execution_provider_exit(
            execution.value,
            primary_failure,
            cancelled=True,
        )

        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            ["post-provider cleanup failure: CancelledError: "],
        )

    async def test_unowned_provider_settlement_attaches_cleanup_results(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        response = _response()
        cleanup_failure = RuntimeError("unowned cleanup failed")
        primary_failure = ValueError("wrapper construction failed")

        with patch.object(
            orchestrator,
            "_cleanup_unowned_provider_response",
            autospec=True,
            return_value=(cleanup_failure,),
        ) as cleanup:
            await orchestrator._settle_unowned_provider_response(
                response.value,
                _execution(status=AgentExecutionStatus.RUNNING).value,
                primary_failure,
                cancelled=False,
            )

        cleanup.assert_awaited_once()
        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            [
                "post-provider cleanup failure: RuntimeError: "
                "unowned cleanup failed"
            ],
        )

    async def test_unowned_cleanup_retains_execution_settlement_error(
        self,
    ) -> None:
        execution = _execution(status=AgentExecutionStatus.RUNNING)
        settlement_failure = RuntimeError("execution terminalization failed")
        execution.settle_provider_exit.side_effect = settlement_failure

        failures = await Orchestrator._cleanup_unowned_provider_response(
            _response().value,
            execution.value,
            cancelled=False,
        )

        self.assertEqual(failures, (settlement_failure,))

    @staticmethod
    async def _release(event: Event) -> None:
        """Release blocked cleanup collaborators on the next loop turn."""
        event.set()


class OrchestratorExitGapCoverageTest(IsolatedAsyncioTestCase):
    """Exercise synchronization and exit aggregation failures."""

    async def test_terminal_snapshot_preserves_provider_cleanup_failure(
        self,
    ) -> None:
        orchestrator = _orchestrator()
        cleanup_failure = RuntimeError("pending provider cleanup failed")

        with (
            patch.object(
                orchestrator,
                "_sync_pending_provider_cleanups",
                autospec=True,
                side_effect=cleanup_failure,
            ),
            patch.object(
                orchestrator,
                "_sync_response_collection",
                autospec=True,
                return_value=(),
            ) as sync_collection,
        ):
            with self.assertRaises(RuntimeError) as raised:
                await orchestrator._sync_terminal_responses_and_snapshot()

        self.assertIs(raised.exception, cleanup_failure)
        sync_collection.assert_awaited_once_with((), owned_only=True)

    async def test_close_response_collection_returns_base_exception(
        self,
    ) -> None:
        close_failure = KeyboardInterrupt("response close interrupted")
        response = create_autospec(
            OrchestratorResponse,
            instance=True,
            spec_set=True,
        )
        response_close = response.aclose
        assert isinstance(response_close, AsyncMock)
        response_close.side_effect = close_failure

        failures = await Orchestrator._close_response_collection((response,))

        self.assertEqual(failures, (close_failure,))

    async def test_abandon_unclaimed_marks_owner_before_sync(self) -> None:
        orchestrator = _orchestrator()
        owner = _pending_cleanup()
        orchestrator._pending_provider_cleanups[id(owner.response)] = owner

        with patch.object(
            owner,
            "sync_messages",
            autospec=True,
            return_value=None,
        ) as sync_messages:
            await orchestrator._sync_pending_provider_cleanups(
                abandon_unclaimed=True
            )

        self.assertTrue(owner.cleanup_required)
        self.assertTrue(owner.cancelled)
        sync_messages.assert_awaited_once_with()

    async def test_pending_cleanup_surfaces_engine_drain_error(self) -> None:
        orchestrator = _orchestrator()
        drain_failure = RuntimeError("engine cleanup drain failed")
        engine_agent = _engine_agent()
        engine_agent.drain_pending_provider_cleanups.side_effect = (
            drain_failure
        )
        orchestrator._engine_agents["one"] = engine_agent.value

        with self.assertRaises(RuntimeError) as raised:
            await orchestrator._sync_pending_provider_cleanups()

        self.assertIs(raised.exception, drain_failure)

    async def test_exit_collects_collection_invocation_failures(self) -> None:
        orchestrator = _orchestrator()
        close_failure = RuntimeError("response collection close failed")
        sync_failure = ValueError("response collection sync failed")
        primary_failure = OSError("body failed")

        with (
            patch.object(
                orchestrator,
                "_sync_pending_provider_cleanups",
                autospec=True,
                return_value=None,
            ),
            patch.object(
                orchestrator,
                "_close_response_collection",
                autospec=True,
                side_effect=close_failure,
            ),
            patch.object(
                orchestrator,
                "_sync_response_collection",
                autospec=True,
                side_effect=sync_failure,
            ),
        ):
            result = await orchestrator.__aexit__(
                OSError,
                primary_failure,
                None,
            )

        self.assertFalse(result)
        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            [
                (
                    "post-provider cleanup failure: RuntimeError: "
                    "response collection close failed"
                ),
                (
                    "post-provider cleanup failure: ValueError: "
                    "response collection sync failed"
                ),
            ],
        )
