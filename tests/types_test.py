from datetime import UTC, datetime
from unittest import TestCase, main

from avalan.types import (
    assert_counter,
    assert_int,
    assert_non_empty_string,
    assert_non_negative_int,
    assert_non_negative_number,
    assert_optional_non_negative_int,
    assert_optional_non_negative_number,
    assert_optional_positive_int,
    assert_optional_positive_number,
    assert_positive_int,
    assert_positive_number,
    assert_string_tuple,
    coerce_datetime,
)


class TypesTest(TestCase):
    def test_assert_non_empty_string_accepts_text(self) -> None:
        assert_non_empty_string("value", "field")

    def test_assert_non_empty_string_rejects_blank_text(self) -> None:
        with self.assertRaises(AssertionError):
            assert_non_empty_string(" ", "field")

    def test_assert_non_empty_string_rejects_non_string(self) -> None:
        with self.assertRaises(AssertionError):
            assert_non_empty_string(1, "field")

    def test_assert_int_accepts_int(self) -> None:
        assert_int(1, "field")

    def test_assert_int_rejects_bool(self) -> None:
        with self.assertRaises(AssertionError):
            assert_int(True, "field")

    def test_assert_positive_int_accepts_positive_int(self) -> None:
        assert_positive_int(1, "field")

    def test_assert_positive_int_rejects_zero(self) -> None:
        with self.assertRaises(AssertionError):
            assert_positive_int(0, "field")

    def test_assert_positive_number_accepts_float(self) -> None:
        assert_positive_number(1.5, "field")

    def test_assert_positive_number_rejects_bool(self) -> None:
        with self.assertRaises(AssertionError):
            assert_positive_number(True, "field")

    def test_assert_positive_number_rejects_zero(self) -> None:
        with self.assertRaises(AssertionError):
            assert_positive_number(0, "field")

    def test_assert_optional_positive_int_accepts_none(self) -> None:
        assert_optional_positive_int(None, "field")

    def test_assert_optional_positive_int_accepts_positive_int(self) -> None:
        assert_optional_positive_int(1, "field")

    def test_assert_optional_positive_number_accepts_none(self) -> None:
        assert_optional_positive_number(None, "field")

    def test_assert_optional_positive_number_accepts_float(self) -> None:
        assert_optional_positive_number(1.5, "field")

    def test_assert_non_negative_int_accepts_zero(self) -> None:
        assert_non_negative_int(0, "field")

    def test_assert_non_negative_int_rejects_negative_int(self) -> None:
        with self.assertRaises(AssertionError):
            assert_non_negative_int(-1, "field")

    def test_assert_non_negative_number_accepts_zero_float(self) -> None:
        assert_non_negative_number(0.0, "field")

    def test_assert_non_negative_number_rejects_negative_float(self) -> None:
        with self.assertRaises(AssertionError):
            assert_non_negative_number(-1.5, "field")

    def test_assert_optional_non_negative_int_accepts_none(self) -> None:
        assert_optional_non_negative_int(None, "field")

    def test_assert_optional_non_negative_int_accepts_zero(self) -> None:
        assert_optional_non_negative_int(0, "field")

    def test_assert_optional_non_negative_number_accepts_none(self) -> None:
        assert_optional_non_negative_number(None, "field")

    def test_assert_optional_non_negative_number_accepts_zero_float(
        self,
    ) -> None:
        assert_optional_non_negative_number(0.0, "field")

    def test_assert_counter_accepts_none(self) -> None:
        assert_counter(None, "field")

    def test_assert_string_tuple_accepts_strings(self) -> None:
        assert_string_tuple(("one", "two"), "field")

    def test_assert_string_tuple_rejects_non_tuple(self) -> None:
        with self.assertRaises(AssertionError):
            assert_string_tuple(["one"], "field")

    def test_assert_string_tuple_rejects_blank_string(self) -> None:
        with self.assertRaises(AssertionError):
            assert_string_tuple((" ",), "field")

    def test_coerce_datetime_accepts_datetime(self) -> None:
        value = datetime(2026, 1, 2, tzinfo=UTC)
        self.assertIs(coerce_datetime(value), value)

    def test_coerce_datetime_accepts_z_suffix(self) -> None:
        self.assertEqual(
            coerce_datetime("2026-01-02T03:04:05Z"),
            datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        )

    def test_coerce_datetime_rejects_non_string(self) -> None:
        with self.assertRaises(AssertionError):
            coerce_datetime(123)

    def test_coerce_datetime_rejects_invalid_string(self) -> None:
        with self.assertRaises(AssertionError):
            coerce_datetime("not a date")


if __name__ == "__main__":
    main()
