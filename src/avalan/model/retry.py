"""Provide provider-neutral bounded retry coordination."""

from ..types import assert_non_negative_int, assert_non_negative_number

from math import frexp, isfinite, ldexp


class ProviderRetryBudget:
    """Track the retries available to one provider operation."""

    _maximum_retries: int
    _retries_used: int

    def __init__(self, maximum_retries: int) -> None:
        assert_non_negative_int(maximum_retries, "maximum_retries")
        self._maximum_retries = maximum_retries
        self._retries_used = 0

    @property
    def maximum_retries(self) -> int:
        """Return the maximum number of retries for this operation."""
        return self._maximum_retries

    @property
    def retries_used(self) -> int:
        """Return the number of retries already reserved."""
        return self._retries_used

    def can_retry(self) -> bool:
        """Return whether the operation has another retry available."""
        return self._retries_used < self._maximum_retries

    def take_retry_attempt(self) -> int:
        """Reserve and return the zero-based attempt for one retry."""
        assert self.can_retry(), "provider retry budget is exhausted"
        attempt = self._retries_used
        self._retries_used += 1
        return attempt


class ProviderRetryController:
    """Coordinate bounded provider retries without choosing retry policy."""

    _budget: ProviderRetryBudget
    _initial_delay_seconds: float
    _maximum_delay_seconds: float

    def __init__(
        self,
        maximum_retries: int,
        *,
        initial_delay_seconds: int | float = 0,
        maximum_delay_seconds: int | float = 8,
    ) -> None:
        initial_delay = self._finite_delay_seconds(
            initial_delay_seconds,
            "initial_delay_seconds",
        )
        maximum_delay = self._finite_delay_seconds(
            maximum_delay_seconds,
            "maximum_delay_seconds",
        )
        self._budget = ProviderRetryBudget(maximum_retries)
        self._initial_delay_seconds = initial_delay
        self._maximum_delay_seconds = max(
            initial_delay,
            maximum_delay,
        )

    @staticmethod
    def _finite_delay_seconds(value: int | float, label: str) -> float:
        """Return one non-negative finite delay normalized to a float."""
        assert_non_negative_number(value, label)
        try:
            delay = float(value)
        except OverflowError:
            raise AssertionError(f"{label} must be finite") from None
        assert isfinite(delay), f"{label} must be finite"
        return delay

    @property
    def budget(self) -> ProviderRetryBudget:
        """Return the shared retry budget."""
        return self._budget

    def can_retry(self) -> bool:
        """Return whether the shared retry budget has another retry."""
        return self._budget.can_retry()

    def take_retry_delay_seconds(self) -> float:
        """Reserve one retry and return its bounded exponential delay."""
        return self.delay_seconds_for_retry_attempt(
            self._budget.take_retry_attempt()
        )

    def delay_seconds_for_retry_attempt(self, attempt: int) -> float:
        """Return the bounded delay for one zero-based retry attempt."""
        assert_non_negative_int(attempt, "attempt")
        if self._initial_delay_seconds == 0:
            return 0.0
        if self._initial_delay_seconds >= self._maximum_delay_seconds:
            return self._maximum_delay_seconds
        (
            initial_significand,
            initial_exponent,
        ) = frexp(self._initial_delay_seconds)
        (
            maximum_significand,
            maximum_exponent,
        ) = frexp(self._maximum_delay_seconds)
        saturation_attempt = maximum_exponent - initial_exponent
        if initial_significand < maximum_significand:
            saturation_attempt += 1
        if attempt >= saturation_attempt:
            return self._maximum_delay_seconds
        return ldexp(self._initial_delay_seconds, attempt)
