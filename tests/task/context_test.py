from unittest import IsolatedAsyncioTestCase, main

from avalan.task.context import TaskUsageObservationTracker


class TaskUsageObservationTrackerTest(IsolatedAsyncioTestCase):
    async def test_observed_tracks_successful_observations(self) -> None:
        calls: list[object] = []

        async def observer(response: object) -> None:
            calls.append(response)

        tracker = TaskUsageObservationTracker(
            observer,
            has_observations=lambda response: response == "usage",
        )

        self.assertFalse(tracker.observed)
        await tracker.observe("empty")
        self.assertFalse(tracker.observed)
        await tracker.observe("usage")
        self.assertTrue(tracker.observed)
        await tracker.observe("empty")
        self.assertTrue(tracker.observed)
        self.assertEqual(calls, ["empty", "usage", "empty"])

    async def test_observe_without_observer_leaves_state_unset(self) -> None:
        tracker = TaskUsageObservationTracker(
            None,
            has_observations=lambda _: True,
        )

        await tracker.observe("usage")

        self.assertFalse(tracker.observed)

    def test_constructor_rejects_invalid_callbacks(self) -> None:
        with self.assertRaises(AssertionError):
            TaskUsageObservationTracker(
                "not-callable",  # type: ignore[arg-type]
                has_observations=lambda _: True,
            )
        with self.assertRaises(AssertionError):
            TaskUsageObservationTracker(
                None,
                has_observations=False,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    main()
