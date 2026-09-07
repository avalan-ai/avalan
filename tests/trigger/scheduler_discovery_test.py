"""Check authoritative wake times and bounded discovery arguments."""

from .preparation_e2e_test import file_task
from .preparation_fault_test import configuration, services
from .records_test import NOW
from .scheduler_fault_test import scheduling

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase

from pytest import raises

from avalan.trigger.apply import TriggerRegistrationService
from avalan.trigger.definition import IntervalTrigger
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.scheduler import TriggerScheduler
from avalan.trigger.scheduler_types import (
    TriggerProcessResult,
    TriggerSchedulerSettings,
)
from avalan.trigger.store import TriggerDiscoveryCursor


class TriggerSchedulerDiscoveryTest(IsolatedAsyncioTestCase):
    async def test_wake_uses_effective_cursor_and_caps_polling(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("input")
            preparation = await services(root)
            scheduler = TriggerScheduler(
                preparation,
                settings=TriggerSchedulerSettings(poll_interval_seconds=60),
            )
            assert await scheduler._wait_seconds(TriggerProcessResult()) == 60
            prepared = await preparation.prepare_registration(
                replace(
                    configuration(),
                    schedule=IntervalTrigger(
                        every_seconds=60,
                        start_at=NOW + timedelta(microseconds=250000),
                    ),
                ),
                file_task(),
            )
            registered = await TriggerRegistrationService(preparation).apply(
                prepared, expected_generation=None
            )
            assert registered.snapshot is not None
            assert (
                await scheduler._wait_seconds(await scheduler.process_once())
                == 0.25
            )
            failed = await scheduler.store.fail(
                "daily",
                expected_generation=registered.snapshot.state.generation,
                error_code=TriggerErrorCode.ADMISSION_RETRYABLE,
                base_seconds=5,
            )
            assert await scheduler.store.next_eligible_at() == NOW + timedelta(
                seconds=5
            )
            assert await scheduler._wait_seconds(TriggerProcessResult()) == 5
            await scheduler.store.set_enabled(
                "daily",
                enabled=False,
                expected_generation=failed.state.generation,
            )
            assert await scheduler.store.next_eligible_at() is None
            assert await scheduler._wait_seconds(TriggerProcessResult()) == 60
            assert (await scheduler.shutdown()).settled

    async def test_cursor_rejects_invalid_boundaries(self) -> None:
        with raises(TriggerError):
            TriggerDiscoveryCursor(
                last_processed_at=NOW, trigger_id="", round_started_at=NOW
            )
        with raises(TriggerError):
            TriggerDiscoveryCursor(
                last_processed_at=NOW.replace(tzinfo=None),
                trigger_id="id",
                round_started_at=NOW,
            )

    async def test_due_remainder_repolls_but_conflicts_back_off(self) -> None:
        async with scheduling() as scheduler:
            assert (
                await scheduler._wait_seconds(
                    TriggerProcessResult(remaining_work=True)
                )
                == 0
            )
            assert await scheduler._wait_seconds(TriggerProcessResult()) == 1
            assert (
                await scheduler._wait_seconds(
                    TriggerProcessResult(conflicts=1, remaining_work=True)
                )
                == 1
            )
