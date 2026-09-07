"""Test management transaction semantics without claiming database proof."""

from .records_test import NOW, OWNER
from .registration_test import registration

from asyncio import gather
from dataclasses import replace
from datetime import timedelta
from unittest import IsolatedAsyncioTestCase

from pytest import raises

from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.records import TriggerSnapshot, TriggerStatus
from avalan.trigger.store import HistoryCursor, TriggerStore
from avalan.trigger.stores.memory import InMemoryTriggerStore


class MemoryTriggerStoreTestCase(IsolatedAsyncioTestCase):
    async def test_creation_is_atomic_and_cas_serializes_competitors(
        self,
    ) -> None:
        store: TriggerStore = InMemoryTriggerStore(OWNER, clock=lambda: NOW)
        assert await store.inspect("daily") is None
        outcomes = await gather(
            store.apply(registration(), expected_generation=None),
            store.apply(registration(), expected_generation=None),
            return_exceptions=True,
        )
        assert sum(isinstance(item, TriggerSnapshot) for item in outcomes) == 1
        assert sum(isinstance(item, TriggerError) for item in outcomes) == 1
        current = await store.inspect("daily")
        assert current is not None
        assert current.state.generation == 1
        assert len((await store.events("daily")).items) == 1
        races = await gather(
            store.set_enabled("daily", enabled=False, expected_generation=1),
            store.apply(
                replace(registration(), semantic_hash="2" * 64),
                expected_generation=1,
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(item, TriggerError) for item in races) == 1
        assert len((await store.events("daily")).items) == 2

    async def test_replacement_retains_immutable_revisions_and_bounded_history(
        self,
    ) -> None:
        now = NOW
        store = InMemoryTriggerStore(OWNER, clock=lambda: now)
        initial = await store.apply(registration(), expected_generation=None)
        now += timedelta(minutes=5)
        updated = await store.apply(
            replace(registration(), semantic_hash="2" * 64),
            expected_generation=1,
        )
        assert updated.state.trigger_id == initial.state.trigger_id
        assert await store.revision("daily", 1) == initial.definition
        assert await store.revision("daily", 2) == updated.definition
        assert await store.revision("daily", 3) is None
        spans = await store.coverage("daily")
        assert len(spans.items) == 1
        assert spans.items[0].first_at == initial.state.next_at
        first = await store.events("daily", limit=1)
        assert first.next_cursor == HistoryCursor(offset=1)
        second = await store.events("daily", cursor=first.next_cursor, limit=1)
        assert second.next_cursor is None
        assert len(second.items) == 1
        unchanged = await store.apply(
            replace(registration(), semantic_hash="2" * 64),
            expected_generation=2,
        )
        assert unchanged == updated
        assert len((await store.events("daily")).items) == 2

    async def test_failed_mutation_leaves_all_management_surfaces_unchanged(
        self,
    ) -> None:
        store = InMemoryTriggerStore(OWNER, clock=lambda: NOW)
        initial = await store.apply(registration(), expected_generation=None)
        with raises(TriggerError):
            await store.apply(
                replace(registration(), semantic_hash="invalid"),
                expected_generation=1,
            )
        assert await store.inspect("daily") == initial
        assert len((await store.events("daily")).items) == 1
        assert (await store.coverage("daily")).items == ()
        assert await store.revision("daily", 2) is None
        with raises(TriggerError):
            await store.set_enabled(
                "missing", enabled=True, expected_generation=1
            )
        with raises(TriggerError):
            await store.events("daily", limit=201)
        with raises(TriggerError):
            HistoryCursor(offset=True)

    async def test_discovery_obeys_due_retry_pause_and_fairness_order(
        self,
    ) -> None:
        now = NOW
        store = InMemoryTriggerStore(OWNER, clock=lambda: now)
        one = await store.apply(registration(), expected_generation=None)
        now += timedelta(seconds=1)
        two = await store.apply(
            replace(registration(), name="second"), expected_generation=None
        )
        assert await store.discover(decision_time=NOW) == ()
        assert await store.discover(
            decision_time=NOW + timedelta(minutes=2), limit=1
        ) == (one,)
        now += timedelta(minutes=2)
        failed = await store.fail(
            "daily",
            expected_generation=1,
            error_code=TriggerErrorCode.ADMISSION_RETRYABLE,
        )
        assert failed.state.next_at == one.state.next_at
        assert await store.discover(decision_time=now) == (two,)
        assert await store.discover(
            decision_time=now + timedelta(seconds=1)
        ) == (
            two,
            failed,
        )
        paused = await store.set_enabled(
            "second", enabled=False, expected_generation=1
        )
        assert paused.state.status is TriggerStatus.PAUSED
        assert await store.discover(
            decision_time=now + timedelta(seconds=1)
        ) == (failed,)
        first_page = await store.list(limit=1)
        assert first_page.items == (failed,)
        assert (await store.list(cursor=first_page.next_cursor)).items == (
            paused,
        )
        assert (await store.list(cursor=HistoryCursor(offset=100))).items == ()

    async def test_default_clock_and_history_are_scoped_to_the_store(
        self,
    ) -> None:
        store = InMemoryTriggerStore(OWNER)
        current = await store.apply(registration(), expected_generation=None)
        assert current.state.next_at is not None
        assert current.state.next_at > NOW
        assert (await store.coverage("daily")).items == ()
