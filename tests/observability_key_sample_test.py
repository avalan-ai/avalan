from typing import Any, cast
from unittest import TestCase, main

from avalan.observability import observability_key_sample


class ObservabilityKeySampleTestCase(TestCase):
    def test_observability_key_sample_sorts_and_bounds_keys(self) -> None:
        keys, truncated = observability_key_sample(
            {"c": 3, "a": 1, "d": 4, "b": 2},
            limit=3,
        )

        self.assertEqual(keys, ["a", "b", "c"])
        self.assertTrue(truncated)

    def test_observability_key_sample_keeps_complete_key_set(self) -> None:
        keys, truncated = observability_key_sample({"b": 2, "a": 1})

        self.assertEqual(keys, ["a", "b"])
        self.assertFalse(truncated)

    def test_observability_key_sample_bounds_key_length(self) -> None:
        keys, truncated = observability_key_sample(
            {"a" * 10: 1},
            key_length_limit=6,
        )

        self.assertEqual(keys, ["aaa..."])
        self.assertTrue(truncated)

    def test_observability_key_sample_bounds_very_short_key_length(
        self,
    ) -> None:
        keys, truncated = observability_key_sample(
            {"a" * 10: 1},
            key_length_limit=3,
        )

        self.assertEqual(keys, ["aaa"])
        self.assertTrue(truncated)

    def test_observability_key_sample_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            observability_key_sample(cast(Any, []))
        with self.assertRaises(AssertionError):
            observability_key_sample({}, limit=0)
        with self.assertRaises(AssertionError):
            observability_key_sample({}, key_length_limit=0)


if __name__ == "__main__":
    main()
