from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

import avalan.task.usage as usage_module
from avalan.task import (
    TaskAttempt,
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskStoreNotFoundError,
    UsageRecord,
    UsageSource,
    UsageTotals,
    attach_response_usage_recorder,
    freeze_usage_metadata,
    freeze_usage_value,
    usage_totals_from_response,
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


class FakeConsumedResponse:
    input_token_count: int | None
    output_token_count: int | None

    def __init__(
        self,
        *,
        input_token_count: int | None = 3,
        output_token_count: int | None = 4,
    ) -> None:
        self.input_token_count = input_token_count
        self.output_token_count = output_token_count
        self._callbacks: list[Callable[[], Awaitable[None] | None]] = []

    def add_done_callback(
        self,
        callback: Callable[[], Awaitable[None] | None],
    ) -> None:
        self._callbacks.append(callback)

    async def consume(self) -> None:
        for callback in self._callbacks:
            result = callback()
            if isinstance(result, Awaitable):
                await result


class FakeProviderResponse(FakeConsumedResponse):
    cached_input_token_count = 1
    cache_creation_input_token_count = 2
    reasoning_token_count = 5
    total_token_count = 99


class HostileResponse:
    input_token_count = -1
    output_token_count = True
    cached_input_token_count = "raw"


def definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
    )


class UsageTotalsTest(TestCase):
    def test_response_usage_preserves_unavailable_counters_as_none(
        self,
    ) -> None:
        totals = usage_totals_from_response(
            FakeConsumedResponse(input_token_count=None, output_token_count=2)
        )

        self.assertIsNotNone(totals)
        assert totals is not None
        self.assertIsNone(totals.input_tokens)
        self.assertEqual(totals.output_tokens, 2)
        self.assertIsNone(totals.cached_input_tokens)
        self.assertIsNone(totals.cache_creation_input_tokens)
        self.assertIsNone(totals.reasoning_tokens)
        self.assertIsNone(totals.total_tokens)

    def test_response_usage_derives_basic_total_from_available_counts(
        self,
    ) -> None:
        totals = usage_totals_from_response(FakeConsumedResponse())

        self.assertEqual(
            totals,
            UsageTotals(
                input_tokens=3,
                output_tokens=4,
                total_tokens=7,
            ),
        )

    def test_provider_reported_fields_are_preserved(self) -> None:
        totals = usage_totals_from_response(FakeProviderResponse())

        self.assertEqual(
            totals,
            UsageTotals(
                input_tokens=3,
                cached_input_tokens=1,
                cache_creation_input_tokens=2,
                output_tokens=4,
                reasoning_tokens=5,
                total_tokens=99,
            ),
        )

    def test_invalid_response_counters_are_unavailable(self) -> None:
        self.assertIsNone(usage_totals_from_response(object()))
        self.assertIsNone(usage_totals_from_response(HostileResponse()))

    def test_usage_entities_are_frozen_and_reject_invalid_counters(
        self,
    ) -> None:
        record = UsageRecord(
            usage_id="usage-1",
            run_id="run-1",
            attempt_id=None,
            sequence=1,
            source=UsageSource.UNAVAILABLE,
            totals=UsageTotals(),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            metadata={"labels": ["safe"]},
        )

        self.assertFalse(record.totals.has_observations)
        self.assertIsInstance(record.metadata, MappingProxyType)
        self.assertEqual(record.metadata["labels"], ("safe",))
        with self.assertRaises(TypeError):
            cast(dict[str, object], record.metadata)["raw"] = "leak"
        with self.assertRaises(AssertionError):
            UsageTotals(input_tokens=-1)
        with self.assertRaises(AssertionError):
            UsageTotals(total_tokens=True)
        with self.assertRaises(AssertionError):
            UsageRecord(
                usage_id="usage-2",
                run_id="run-1",
                attempt_id=None,
                sequence=0,
                source=UsageSource.UNAVAILABLE,
                totals=UsageTotals(),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_usage_metadata_rejects_unsafe_values(self) -> None:
        empty = freeze_usage_metadata(None)
        frozen = freeze_usage_value({"counts": [1, 2]})
        finite_float = freeze_usage_value(1.5)

        self.assertEqual(empty, {})
        self.assertEqual(cast(Mapping[str, object], frozen)["counts"], (1, 2))
        self.assertEqual(finite_float, 1.5)
        with self.assertRaises(AssertionError):
            freeze_usage_value({"raw": object()})
        with self.assertRaises(AssertionError):
            freeze_usage_value(float("inf"))
        with self.assertRaises(AssertionError):
            freeze_usage_value({"": "empty"})

    def test_unknown_usage_counter_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            usage_module._counter_value(  # noqa: SLF001
                UsageTotals(),
                "unknown",
            )


class UsageStoreTest(IsolatedAsyncioTestCase):
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

    async def test_response_callback_records_usage_after_consumption(
        self,
    ) -> None:
        response = FakeConsumedResponse()
        attached = attach_response_usage_recorder(
            response,
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            metadata={"model": "test"},
        )

        self.assertTrue(attached)
        self.assertEqual(await self.store.list_usage(self.run.run_id), ())

        await response.consume()
        await response.consume()

        records = await self.store.list_usage(self.run.run_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sequence, 1)
        self.assertEqual(records[0].source, UsageSource.ESTIMATED)
        self.assertEqual(records[0].attempt_id, self.attempt.attempt_id)
        self.assertEqual(records[0].totals.input_tokens, 3)
        self.assertEqual(records[0].totals.output_tokens, 4)
        self.assertEqual(records[0].totals.total_tokens, 7)
        self.assertEqual(records[0].metadata["model"], "test")

    async def test_usage_records_are_filtered_and_aggregated(self) -> None:
        other_attempt = await self._failed_attempt_then_new_attempt()
        await self.store.append_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            source=UsageSource.EXACT,
            totals=UsageTotals(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
            ),
        )
        await self.store.append_usage(
            self.run.run_id,
            attempt_id=other_attempt.attempt_id,
            source=UsageSource.ESTIMATED,
            totals=UsageTotals(output_tokens=4),
        )

        first_attempt_records = await self.store.list_usage(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )
        totals = await self.store.usage_totals(self.run.run_id)

        self.assertEqual(len(first_attempt_records), 1)
        self.assertEqual(totals.input_tokens, 2)
        self.assertEqual(totals.output_tokens, 7)
        self.assertEqual(totals.total_tokens, 5)
        self.assertIsNone(totals.cached_input_tokens)

    async def test_callback_is_not_attached_without_response_support(
        self,
    ) -> None:
        self.assertFalse(
            attach_response_usage_recorder(
                object(),
                store=self.store,
                run_id=self.run.run_id,
            )
        )

    async def test_callback_with_unavailable_counters_records_nothing(
        self,
    ) -> None:
        response = FakeConsumedResponse(
            input_token_count=None,
            output_token_count=None,
        )

        self.assertTrue(
            attach_response_usage_recorder(
                response,
                store=self.store,
                run_id=self.run.run_id,
                attempt_id=self.attempt.attempt_id,
            )
        )
        await response.consume()

        self.assertEqual(await self.store.list_usage(self.run.run_id), ())

    async def test_usage_attempt_must_belong_to_run(self) -> None:
        other_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        other_attempt = await self.store.create_attempt(other_run.run_id)

        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.append_usage(
                self.run.run_id,
                attempt_id=other_attempt.attempt_id,
                source=UsageSource.EXACT,
                totals=UsageTotals(input_tokens=1),
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_usage(
                self.run.run_id,
                attempt_id=other_attempt.attempt_id,
            )

    async def _failed_attempt_then_new_attempt(self) -> TaskAttempt:
        await self.store.transition_attempt(
            self.attempt.attempt_id,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
        )
        await self.store.transition_attempt(
            self.attempt.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="failed",
        )
        return await self.store.create_attempt(self.run.run_id)


if __name__ == "__main__":
    main()
