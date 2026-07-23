"""Regress cancellation convergence during direct response iteration."""

from asyncio import CancelledError, Event, Task, create_task, gather, wait_for
from unittest import IsolatedAsyncioTestCase

from execution_attached_boundaries_test import (
    _AnswerHandler,
    _BoundaryBroker,
    _Gate,
    _Harness,
    _install_result_append_gate,
    _ModelManager,
)

from avalan.agent.execution import AgentExecutionStatus
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.model.stream import (
    StreamItemKind,
    validate_canonical_stream_items,
)


async def _consume(response: OrchestratorResponse) -> None:
    """Drain one response through its direct async-iterator surface."""
    async for _ in response:
        pass


async def _pause_after_continuation_start(
    response: OrchestratorResponse,
    started: Event,
    proceed: Event,
) -> None:
    """Pause before requesting the item after continuation start."""
    async for item in response:
        if item.kind is StreamItemKind.MODEL_CONTINUATION_STARTED:
            started.set()
            await proceed.wait()


async def _cancelled_failure(task: Task[None]) -> BaseException | None:
    """Return the exact direct-consumer failure."""
    try:
        await task
    except BaseException as exc:
        return exc
    return None


class DirectIterationCancellationTest(IsolatedAsyncioTestCase):
    """Require attached direct iteration to converge on cancellation."""

    def _assert_converged(
        self,
        response: OrchestratorResponse,
        broker: _BoundaryBroker,
    ) -> None:
        execution = response.execution
        assert execution is not None
        self.assertIs(execution.status, AgentExecutionStatus.CANCELLED)
        self.assertIsNone(execution.snapshot.pending_request)
        self.assertTrue(execution.snapshot.cleanup_started)
        self.assertTrue(response._interaction_cleanup_complete)
        self.assertIsNone(response._pending_interaction_task)
        self.assertIsNone(response._pending_tool_batch_task)

        kinds = tuple(item.kind for item in response.canonical_items)
        self.assertEqual(
            kinds[-2:],
            (
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ),
        )
        self.assertEqual(kinds.count(StreamItemKind.STREAM_CANCELLED), 1)
        self.assertEqual(kinds.count(StreamItemKind.STREAM_CLOSED), 1)
        self.assertNotIn(StreamItemKind.STREAM_ERRORED, kinds)
        validate_canonical_stream_items(response.canonical_items)

        self.assertEqual(len(broker.cancel_scope_commands), 1)
        cleanup = broker.cancel_scope_commands[0]
        self.assertEqual(cleanup.scope.run_id, execution.origin.run_id)
        self.assertEqual(cleanup.scope.branch_id, execution.origin.branch_id)

    async def test_consumer_cancel_after_result_append_converges(self) -> None:
        gate = _Gate()
        broker = _BoundaryBroker()
        manager = _ModelManager()
        harness = _Harness(broker=broker, manager=manager)
        response = await harness.orchestrator(
            "direct-result-append",
            interaction_runtime=harness.runtime(
                prefix="direct-result-append",
                handler=_AnswerHandler(),
            ),
        )
        execution = response.execution
        assert execution is not None
        _install_result_append_gate(execution, gate)
        consumer = create_task(_consume(response))
        try:
            await wait_for(gate.entered.wait(), timeout=1)
            self.assertIs(execution.status, AgentExecutionStatus.RESUMING)
            consumer.cancel()
            failure = await wait_for(_cancelled_failure(consumer), timeout=1)

            self.assertIs(type(failure), CancelledError)
            self._assert_converged(response, broker)
        finally:
            if not consumer.done():
                consumer.cancel()
                await gather(consumer, return_exceptions=True)
            await harness.close()

    async def test_checker_after_continuation_start_converges(self) -> None:
        provider_gate = _Gate()
        broker = _BoundaryBroker()
        manager = _ModelManager(
            blocking_continuation_gate=provider_gate,
        )
        harness = _Harness(broker=broker, manager=manager)
        prompt = "direct-continuation-checker"
        response = await harness.orchestrator(
            prompt,
            interaction_runtime=harness.runtime(
                prefix=prompt,
                handler=_AnswerHandler(),
            ),
        )
        continuation_started = Event()
        proceed = Event()
        consumer = create_task(
            _pause_after_continuation_start(
                response,
                continuation_started,
                proceed,
            )
        )
        try:
            await wait_for(continuation_started.wait(), timeout=1)

            async def check_cancellation() -> None:
                raise CancelledError()

            response.set_cancellation_checker(check_cancellation)
            proceed.set()
            failure = await wait_for(_cancelled_failure(consumer), timeout=1)

            self.assertIs(type(failure), CancelledError)
            self._assert_converged(response, broker)
            source = manager.continuation_sources[prompt]
            self.assertEqual(
                (source.cancel_calls, source.aclose_calls),
                (1, 1),
            )
            self.assertFalse(provider_gate.entered.is_set())
        finally:
            if not consumer.done():
                consumer.cancel()
                await gather(consumer, return_exceptions=True)
            await harness.close()


if __name__ == "__main__":
    from unittest import main

    main()
