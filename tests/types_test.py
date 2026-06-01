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


if __name__ == "__main__":
    main()
