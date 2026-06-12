from datetime import UTC, datetime
from unittest import TestCase, main

from avalan.types import (
    assert_absolute_path,
    assert_absolute_path_mapping,
    assert_absolute_path_sequence,
    assert_bool,
    assert_counter,
    assert_env_name,
    assert_int,
    assert_known_string,
    assert_known_string_sequence,
    assert_media_type,
    assert_non_empty_string,
    assert_non_empty_string_sequence,
    assert_non_negative_int,
    assert_non_negative_number,
    assert_optional_bounded_number,
    assert_optional_non_negative_int,
    assert_optional_non_negative_number,
    assert_optional_positive_int,
    assert_optional_positive_number,
    assert_positive_int,
    assert_positive_number,
    assert_safe_suffix,
    assert_safe_suffix_sequence,
    assert_sha256_hex,
    assert_string_mapping,
    assert_string_tuple,
    assert_suffix_media_type_mapping,
    coerce_datetime,
)


class TypesTest(TestCase):
    def test_assert_bool_accepts_bool(self) -> None:
        assert_bool(True, "field")
        assert_bool(False, "field")

    def test_assert_bool_rejects_int(self) -> None:
        with self.assertRaises(AssertionError):
            assert_bool(1, "field")

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

    def test_assert_non_empty_string_sequence_accepts_list(self) -> None:
        assert_non_empty_string_sequence(["one", "two"], "field")

    def test_assert_non_empty_string_sequence_rejects_scalar(self) -> None:
        with self.assertRaises(AssertionError):
            assert_non_empty_string_sequence("one", "field")

    def test_assert_non_empty_string_sequence_rejects_empty(self) -> None:
        with self.assertRaises(AssertionError):
            assert_non_empty_string_sequence([], "field")

    def test_assert_non_empty_string_sequence_rejects_blank_item(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            assert_non_empty_string_sequence([""], "field")

    def test_assert_string_mapping_accepts_strings(self) -> None:
        assert_string_mapping({"KEY": "value"}, "field")

    def test_assert_string_mapping_rejects_non_mapping(self) -> None:
        with self.assertRaises(AssertionError):
            assert_string_mapping([], "field")

    def test_assert_string_mapping_rejects_blank_key_or_value(self) -> None:
        with self.assertRaises(AssertionError):
            assert_string_mapping({"": "value"}, "field")
        with self.assertRaises(AssertionError):
            assert_string_mapping({"KEY": ""}, "field")

    def test_assert_known_string_accepts_allowlisted_value(self) -> None:
        assert_known_string("txt", "field", ("txt", "json"))

    def test_assert_known_string_rejects_unknown_or_bad_allowlist(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            assert_known_string("xml", "field", ("txt", "json"))
        with self.assertRaises(AssertionError):
            assert_known_string("txt", "field", ())
        with self.assertRaises(AssertionError):
            assert_known_string("txt", "field", ("",))

    def test_assert_known_string_sequence_accepts_allowlisted_values(
        self,
    ) -> None:
        assert_known_string_sequence(("txt", "json"), "field", ("txt", "json"))

    def test_assert_known_string_sequence_rejects_invalid_values(
        self,
    ) -> None:
        invalid_values = (
            "txt",
            (),
            ("",),
            ("xml",),
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(AssertionError):
                    assert_known_string_sequence(
                        value,
                        "field",
                        ("txt", "json"),
                    )
        with self.assertRaises(AssertionError):
            assert_known_string_sequence(("txt",), "field", ())

    def test_assert_optional_bounded_number_accepts_none(self) -> None:
        assert_optional_bounded_number(
            None,
            "field",
            min_value=1,
            max_value=2,
        )

    def test_assert_optional_bounded_number_accepts_boundaries(self) -> None:
        assert_optional_bounded_number(1, "field", min_value=1)
        assert_optional_bounded_number(2, "field", max_value=2)

    def test_assert_optional_bounded_number_rejects_bool(self) -> None:
        with self.assertRaises(AssertionError):
            assert_optional_bounded_number(True, "field")

    def test_assert_optional_bounded_number_rejects_exclusive_bounds(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            assert_optional_bounded_number(
                1,
                "field",
                min_value=1,
                min_inclusive=False,
            )
        with self.assertRaises(AssertionError):
            assert_optional_bounded_number(
                2,
                "field",
                max_value=2,
                max_inclusive=False,
            )

    def test_assert_optional_bounded_number_rejects_invalid_bound(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            assert_optional_bounded_number(1, "field", min_value=True)

    def test_assert_env_name_accepts_identifier(self) -> None:
        assert_env_name("_NAME1", "field")

    def test_assert_env_name_rejects_invalid_values(self) -> None:
        for value in ("", "1NAME", "A-B", "A.B", "A B"):
            with self.subTest(value=value):
                with self.assertRaises(AssertionError):
                    assert_env_name(value, "field")

    def test_assert_safe_suffix_accepts_file_suffixes(self) -> None:
        assert_safe_suffix(".png", "field")
        assert_safe_suffix("txt", "field")

    def test_assert_safe_suffix_rejects_path_like_values(self) -> None:
        for value in ("", ".", "..", "../png", "dir/png", "dir\\png", "x\x00"):
            with self.subTest(value=value):
                with self.assertRaises(AssertionError):
                    assert_safe_suffix(value, "field")

    def test_assert_safe_suffix_sequence_accepts_suffixes(self) -> None:
        assert_safe_suffix_sequence((".png", ".txt"), "field")

    def test_assert_safe_suffix_sequence_rejects_invalid_sequences(
        self,
    ) -> None:
        for value in (".png", (), ("../png",)):
            with self.subTest(value=value):
                with self.assertRaises(AssertionError):
                    assert_safe_suffix_sequence(value, "field")

    def test_assert_media_type_accepts_type_and_subtype(self) -> None:
        assert_media_type("text/plain", "field")
        assert_media_type("application/vnd.test+json", "field")

    def test_assert_media_type_rejects_invalid_values(self) -> None:
        for value in (
            "",
            "text",
            "text/",
            "/plain",
            "text/plain; charset=utf-8",
        ):
            with self.subTest(value=value):
                with self.assertRaises(AssertionError):
                    assert_media_type(value, "field")

    def test_assert_suffix_media_type_mapping_accepts_mapping(self) -> None:
        assert_suffix_media_type_mapping({".png": "image/png"}, "field")

    def test_assert_suffix_media_type_mapping_rejects_bad_mapping(
        self,
    ) -> None:
        invalid_values = (
            [],
            {"../png": "image/png"},
            {".png": "image"},
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(AssertionError):
                    assert_suffix_media_type_mapping(value, "field")

    def test_assert_sha256_hex_accepts_lowercase_digest(self) -> None:
        assert_sha256_hex("a" * 64)

    def test_assert_sha256_hex_rejects_invalid_digests(self) -> None:
        for value in ("", "a" * 63, "A" * 64, "g" * 64):
            with self.subTest(value=value):
                with self.assertRaises(AssertionError):
                    assert_sha256_hex(value)

    def test_assert_absolute_path_accepts_absolute_string(self) -> None:
        assert_absolute_path("/tmp/value", "field")

    def test_assert_absolute_path_rejects_relative_and_nul(self) -> None:
        with self.assertRaises(AssertionError):
            assert_absolute_path("relative", "field")
        with self.assertRaises(AssertionError):
            assert_absolute_path("/tmp/\x00value", "field")

    def test_assert_absolute_path_sequence_accepts_paths(self) -> None:
        assert_absolute_path_sequence(("/tmp/one", "/tmp/two"), "field")

    def test_assert_absolute_path_sequence_rejects_scalar(self) -> None:
        with self.assertRaises(AssertionError):
            assert_absolute_path_sequence("/tmp/one", "field")

    def test_assert_absolute_path_mapping_accepts_paths(self) -> None:
        assert_absolute_path_mapping({"tool": "/tmp/tool"}, "field")

    def test_assert_absolute_path_mapping_rejects_bad_mapping(self) -> None:
        with self.assertRaises(AssertionError):
            assert_absolute_path_mapping([], "field")
        with self.assertRaises(AssertionError):
            assert_absolute_path_mapping({"tool": "relative"}, "field")

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
