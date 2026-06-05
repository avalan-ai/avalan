from unittest import TestCase, main

from avalan.model.provider import (
    ProviderFamily,
    provider_family_value,
    provider_options_from_uri_params,
    provider_string_option,
)


class ProviderTest(TestCase):
    def test_provider_family_value_accepts_enum(self) -> None:
        self.assertEqual(
            provider_family_value(ProviderFamily.OPENAI_COMPATIBLE),
            "openai_compatible",
        )

    def test_provider_family_value_accepts_custom_string(self) -> None:
        self.assertEqual(
            provider_family_value("private-provider"), "private-provider"
        )

    def test_provider_family_value_rejects_blank_string(self) -> None:
        with self.assertRaises(AssertionError):
            provider_family_value(" ")

    def test_provider_options_from_uri_params_keeps_legacy_key(self) -> None:
        self.assertEqual(
            provider_options_from_uri_params(
                {"azure_api_version": "2025-04-01-preview"}
            ),
            {"azure_api_version": "2025-04-01-preview"},
        )

    def test_provider_options_from_uri_params_strips_generic_prefix(
        self,
    ) -> None:
        self.assertEqual(
            provider_options_from_uri_params(
                {
                    "provider_timeout": 30,
                    "temperature": 1,
                }
            ),
            {"timeout": 30},
        )

    def test_provider_options_from_uri_params_rejects_empty_key(self) -> None:
        with self.assertRaises(AssertionError):
            provider_options_from_uri_params({"provider_": "value"})

    def test_provider_string_option_accepts_string(self) -> None:
        self.assertEqual(
            provider_string_option(
                {"api_version": "2026-01-01"}, "api_version"
            ),
            "2026-01-01",
        )

    def test_provider_string_option_returns_none_for_null_value(self) -> None:
        self.assertIsNone(
            provider_string_option({"api_version": None}, "api_version")
        )

    def test_provider_string_option_rejects_non_string(self) -> None:
        with self.assertRaises(AssertionError):
            provider_string_option({"api_version": 20260101}, "api_version")


if __name__ == "__main__":
    main()
