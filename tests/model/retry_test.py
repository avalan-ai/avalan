from typing import cast
from unittest import TestCase, main

from avalan.model.retry import ProviderRetryBudget, ProviderRetryController


class ProviderRetryBudgetTest(TestCase):
    def test_budget_reserves_exactly_its_maximum_retries(self) -> None:
        budget = ProviderRetryBudget(2)

        self.assertEqual(budget.maximum_retries, 2)
        self.assertTrue(budget.can_retry())
        self.assertEqual(budget.take_retry_attempt(), 0)
        self.assertTrue(budget.can_retry())
        self.assertEqual(budget.take_retry_attempt(), 1)
        self.assertFalse(budget.can_retry())
        self.assertEqual(budget.retries_used, 2)
        with self.assertRaises(AssertionError):
            budget.take_retry_attempt()

    def test_budget_rejects_invalid_maximum_retries(self) -> None:
        for value in (-1, False, 1.5):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                ProviderRetryBudget(cast(int, value))


class ProviderRetryControllerTest(TestCase):
    def test_controller_caps_bounded_exponential_delay(self) -> None:
        controller = ProviderRetryController(
            5,
            initial_delay_seconds=1,
            maximum_delay_seconds=8,
        )

        self.assertEqual(
            [controller.take_retry_delay_seconds() for _ in range(5)],
            [1, 2, 4, 8, 8],
        )
        self.assertFalse(controller.can_retry())

    def test_controller_preserves_first_delay_above_configured_cap(
        self,
    ) -> None:
        controller = ProviderRetryController(
            2,
            initial_delay_seconds=10,
            maximum_delay_seconds=8,
        )

        self.assertEqual(
            [controller.take_retry_delay_seconds() for _ in range(2)],
            [10, 10],
        )

    def test_controller_saturates_huge_attempt_without_exponentiation(
        self,
    ) -> None:
        controller = ProviderRetryController(
            1,
            initial_delay_seconds=0.25,
            maximum_delay_seconds=0.75,
        )

        assert controller.delay_seconds_for_retry_attempt(1) == 0.5
        assert controller.delay_seconds_for_retry_attempt(2) == 0.75
        assert controller.delay_seconds_for_retry_attempt(10**100) == 0.75

    def test_controller_caps_fractional_delay_before_next_power_of_two(
        self,
    ) -> None:
        controller = ProviderRetryController(
            1,
            initial_delay_seconds=0.75,
            maximum_delay_seconds=1,
        )

        assert controller.delay_seconds_for_retry_attempt(0) == 0.75
        assert controller.delay_seconds_for_retry_attempt(1) == 1

    def test_controller_zero_delay_remains_zero_at_huge_attempt(self) -> None:
        controller = ProviderRetryController(
            1,
            initial_delay_seconds=0,
            maximum_delay_seconds=0,
        )

        assert controller.delay_seconds_for_retry_attempt(10**100) == 0

    def test_shared_controller_coordinates_opening_and_stream_retries(
        self,
    ) -> None:
        controller = ProviderRetryController(2, initial_delay_seconds=0)

        opening_delay = controller.take_retry_delay_seconds()
        stream_delay = controller.take_retry_delay_seconds()

        self.assertEqual((opening_delay, stream_delay), (0, 0))
        self.assertEqual(controller.budget.retries_used, 2)
        self.assertFalse(controller.can_retry())

    def test_controller_rejects_invalid_delay_configuration(self) -> None:
        for value in (-1, False, float("nan"), float("inf"), float("-inf")):
            with (
                self.subTest(initial_delay_seconds=value),
                self.assertRaises(AssertionError),
            ):
                ProviderRetryController(
                    1,
                    initial_delay_seconds=value,
                )
            with (
                self.subTest(maximum_delay_seconds=value),
                self.assertRaises(AssertionError),
            ):
                ProviderRetryController(
                    1,
                    maximum_delay_seconds=value,
                )

    def test_controller_rejects_delay_integers_outside_float_range(
        self,
    ) -> None:
        huge_delay = 10**400

        with self.assertRaisesRegex(
            AssertionError,
            "initial_delay_seconds must be finite",
        ):
            ProviderRetryController(
                1,
                initial_delay_seconds=huge_delay,
            )
        with self.assertRaisesRegex(
            AssertionError,
            "maximum_delay_seconds must be finite",
        ):
            ProviderRetryController(
                1,
                maximum_delay_seconds=huge_delay,
            )


if __name__ == "__main__":
    main()
