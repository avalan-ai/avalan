from datetime import UTC, datetime, timedelta
from typing import cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.event import Event, EventType
from avalan.task import (
    DirectTaskRunner,
    HmacProvider,
    PrivacyAction,
    PrivacySanitizer,
    SanitizedTaskEventDraft,
    TaskDefinition,
    TaskDirectTarget,
    TaskEventCategory,
    TaskEventPipeline,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskObservedEvent,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskRunState,
    TaskTargetContext,
)
from avalan.task.stores import InMemoryTaskStore


class SequenceClock:
    def __init__(self) -> None:
        self._next = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self._next
        self._next = self._next + timedelta(seconds=1)
        return value


class SequenceIds:
    def __init__(self) -> None:
        self._next = 1

    def __call__(self) -> str:
        value = f"id-{self._next}"
        self._next = self._next + 1
        return value


class StaticHmacProvider(HmacProvider):
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"test-secret",
        )


class EventEmittingTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        if context.event_listener is not None:
            result = context.event_listener(
                Event(
                    type=EventType.TOKEN_GENERATED,
                    payload={
                        "count": 1,
                        "token": "private-token",
                        "token_id": 441,
                    },
                )
            )
            if result is not None:
                await result
        return "done"


class ListenerRecordingTarget:
    def __init__(self) -> None:
        self.saw_listener = False

    async def __call__(self, context: TaskTargetContext) -> object:
        self.saw_listener = context.event_listener is not None
        return "done"


def definition(
    *,
    observability: TaskObservabilityPolicy | None = None,
    privacy: TaskPrivacyPolicy | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
        observability=observability or TaskObservabilityPolicy(),
        privacy=privacy or TaskPrivacyPolicy(),
    )


async def maybe_observe(
    events: list[TaskObservedEvent],
    event: TaskObservedEvent,
) -> None:
    events.append(event)


async def fail_observer(event: TaskObservedEvent) -> None:
    assert event is not None
    raise RuntimeError("private observer failure")


class TaskEventPipelineTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryTaskStore(
            clock=SequenceClock(),
            id_factory=SequenceIds(),
        )
        await self.store.register_definition(
            definition(),
            definition_hash="hash-a",
        )
        self.run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        self.attempt = await self.store.create_attempt(self.run.run_id)

    async def test_pipeline_fans_out_persisted_sanitized_events(self) -> None:
        metrics_events: list[TaskObservedEvent] = []
        trace_events: list[TaskObservedEvent] = []
        pipeline = TaskEventPipeline(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            sanitizer=PrivacySanitizer(),
            metrics_observer=lambda event: maybe_observe(
                metrics_events,
                event,
            ),
            trace_observer=lambda event: maybe_observe(trace_events, event),
        )

        await pipeline(
            Event(
                type=EventType.TOOL_RESULT,
                payload={
                    "name": "search",
                    "status": "ok",
                    "arguments": {"query": "private query"},
                    "result": "private result",
                },
            )
        )

        stored_events = await self.store.list_events(self.run.run_id)
        self.assertEqual(metrics_events, list(stored_events))
        self.assertEqual(trace_events, list(stored_events))
        self.assertEqual(stored_events[0].category, TaskEventCategory.TOOL)
        self.assertNotIn("private", str(stored_events))
        self.assertNotIn("private", str(metrics_events))
        self.assertNotIn("private", str(trace_events))

    async def test_pipeline_can_fan_out_without_persisting_events(
        self,
    ) -> None:
        metrics_events: list[TaskObservedEvent] = []
        pipeline = TaskEventPipeline(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            sanitizer=PrivacySanitizer(),
            capture_events=False,
            metrics_observer=lambda event: maybe_observe(
                metrics_events,
                event,
            ),
        )

        await pipeline(
            Event(
                type=EventType.MODEL_EXECUTE_AFTER,
                payload={"status": "ok", "output": "private output"},
            )
        )

        self.assertEqual(await self.store.list_events(self.run.run_id), ())
        self.assertIsInstance(metrics_events[0], SanitizedTaskEventDraft)
        self.assertEqual(metrics_events[0].event_type, "model_execute_after")
        self.assertNotIn("private output", str(metrics_events))

    async def test_observer_failures_do_not_interrupt_safe_fanout(
        self,
    ) -> None:
        trace_events: list[TaskObservedEvent] = []
        pipeline = TaskEventPipeline(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            sanitizer=PrivacySanitizer(),
            metrics_observer=fail_observer,
            trace_observer=lambda event: maybe_observe(trace_events, event),
        )

        await pipeline(
            Event(
                type=EventType.TOOL_RESULT,
                payload={
                    "arguments": {"query": "private query"},
                    "result": "private result",
                },
            )
        )

        stored_events = await self.store.list_events(self.run.run_id)
        self.assertEqual(trace_events, list(stored_events))
        self.assertEqual(stored_events[0].event_type, "tool_result")
        self.assertNotIn("private observer failure", str(stored_events))
        self.assertNotIn("private query", str(trace_events))

    async def test_event_store_failures_do_not_interrupt_safe_fanout(
        self,
    ) -> None:
        metrics_events: list[TaskObservedEvent] = []
        pipeline = TaskEventPipeline(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            sanitizer=PrivacySanitizer(),
            metrics_observer=lambda event: maybe_observe(
                metrics_events,
                event,
            ),
        )

        with patch.object(
            self.store,
            "append_event",
            side_effect=RuntimeError("private event store failure"),
        ):
            await pipeline(
                Event(
                    type=EventType.TOOL_RESULT,
                    payload={
                        "arguments": {"query": "private query"},
                        "result": "private result",
                    },
                )
            )

        self.assertEqual(await self.store.list_events(self.run.run_id), ())
        self.assertIsInstance(metrics_events[0], SanitizedTaskEventDraft)
        self.assertEqual(metrics_events[0].event_type, "tool_result")
        self.assertNotIn("private event store failure", str(metrics_events))
        self.assertNotIn("private query", str(metrics_events))

    async def test_sanitizer_failures_reduce_to_safe_events(self) -> None:
        observed: list[TaskObservedEvent] = []
        pipeline = TaskEventPipeline(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            sanitizer=PrivacySanitizer(
                TaskPrivacyPolicy(events=PrivacyAction.HASH)
            ),
            metrics_observer=lambda event: maybe_observe(observed, event),
        )

        await pipeline(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload={"token": "private-token"},
            )
        )

        events = await self.store.list_events(self.run.run_id)
        self.assertEqual(events[0].event_type, "event_sanitization_failed")
        self.assertEqual(events[0].category, TaskEventCategory.UNKNOWN)
        self.assertEqual(observed, list(events))
        self.assertNotIn("private-token", str(events))
        self.assertNotIn("HMAC", str(events))


class DirectRunnerEventPipelineTest(IsolatedAsyncioTestCase):
    async def test_runner_omits_pipeline_when_observability_is_disabled(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        target = ListenerRecordingTarget()
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "hash-runner-no-observability",
        )

        result = await runner.run(
            definition(observability=TaskObservabilityPolicy.noop()),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertFalse(target.saw_listener)
        self.assertEqual(await store.list_events(result.run.run_id), ())

    async def test_runner_attaches_pipeline_for_sanitized_metrics_observers(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        metrics_events: list[TaskObservedEvent] = []
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, EventEmittingTarget()),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "hash-runner-observability",
            metrics_event_observer=lambda event: maybe_observe(
                metrics_events,
                event,
            ),
        )

        result = await runner.run(
            definition(
                observability=TaskObservabilityPolicy(
                    capture_events=False,
                    metrics=True,
                    trace=False,
                )
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(await store.list_events(result.run.run_id), ())
        self.assertEqual(metrics_events[0].event_type, "token_generated")
        self.assertNotIn("private-token", str(metrics_events))
        self.assertNotIn("token_id", str(metrics_events))

    async def test_runner_success_survives_observer_failures(self) -> None:
        store = InMemoryTaskStore()
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, EventEmittingTarget()),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "hash-runner-observer-failure",
            metrics_event_observer=fail_observer,
        )

        result = await runner.run(
            definition(
                observability=TaskObservabilityPolicy(
                    capture_events=True,
                    metrics=True,
                    trace=False,
                )
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        stored_events = await store.list_events(result.run.run_id)
        self.assertEqual(len(stored_events), 1)
        self.assertEqual(stored_events[0].event_type, "token_generated")
        self.assertNotIn("private observer failure", str(stored_events))

    async def test_runner_success_survives_event_store_failures(self) -> None:
        store = InMemoryTaskStore()
        metrics_events: list[TaskObservedEvent] = []
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, EventEmittingTarget()),
            hmac_provider=StaticHmacProvider(),
            definition_hash=lambda task: "hash-runner-event-store-failure",
            metrics_event_observer=lambda event: maybe_observe(
                metrics_events,
                event,
            ),
        )

        with patch.object(
            store,
            "append_event",
            side_effect=RuntimeError("private event store failure"),
        ):
            result = await runner.run(
                definition(
                    observability=TaskObservabilityPolicy(
                        capture_events=True,
                        metrics=True,
                        trace=False,
                    )
                ),
                input_value="private prompt",
            )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(await store.list_events(result.run.run_id), ())
        self.assertIsInstance(metrics_events[0], SanitizedTaskEventDraft)
        self.assertEqual(metrics_events[0].event_type, "token_generated")
        self.assertNotIn("private event store failure", str(metrics_events))
        self.assertNotIn("private-token", str(metrics_events))


if __name__ == "__main__":
    main()
