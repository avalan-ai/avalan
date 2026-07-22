from unittest import TestCase

from avalan.model import (
    DomainCapabilitySeed,
    ModelCapabilityCatalog,
    ModelCapabilityDescriptor,
)
from avalan.model.vendor import TextGenerationVendor


class VendorEncodeDecodeTestCase(TestCase):
    def test_encode_decode_roundtrip(self) -> None:
        original = "pkg.sub.tool"
        encoded = TextGenerationVendor.encode_tool_name(original)
        self.assertEqual(encoded, "avl_cGtnLnN1Yi50b29s")
        self.assertEqual(
            TextGenerationVendor.decode_tool_name(encoded), original
        )

    def test_encode_decode_noop(self) -> None:
        name = "plain"
        self.assertEqual(TextGenerationVendor.encode_tool_name(name), name)
        self.assertEqual(TextGenerationVendor.decode_tool_name(name), name)

    def test_decode_preserves_plain_double_underscore_name(self) -> None:
        self.assertEqual(
            TextGenerationVendor.decode_tool_name("pkg__tool"), "pkg__tool"
        )

    def test_encode_prefix_name_to_preserve_provenance(self) -> None:
        encoded = TextGenerationVendor.encode_tool_name("avl_plain")

        self.assertEqual(encoded, "avl_YXZsX3BsYWlu")
        self.assertEqual(
            TextGenerationVendor.decode_tool_name(encoded), "avl_plain"
        )

    def test_encode_rejects_empty_name(self) -> None:
        with self.assertRaises(AssertionError):
            TextGenerationVendor.encode_tool_name(" ")

    def test_decode_rejects_invalid_provider_name(self) -> None:
        with self.assertRaises(AssertionError):
            TextGenerationVendor.decode_tool_name("pkg.tool")

    def test_decode_rejects_malformed_encoded_name(self) -> None:
        with self.assertRaises(AssertionError):
            TextGenerationVendor.decode_tool_name("avl_notbase64")

    def test_decode_rejects_invalid_encoded_payload(self) -> None:
        with self.assertRaises(AssertionError):
            TextGenerationVendor.decode_tool_name("avl_A")

    def test_canonical_tool_name_preserves_raw_and_rejects_bad_encoded(
        self,
    ) -> None:
        self.assertEqual(
            TextGenerationVendor.canonical_tool_name("pkg.tool"),
            "pkg.tool",
        )
        with self.assertRaises(AssertionError):
            TextGenerationVendor.canonical_tool_name("avl_notbase64")

    def test_canonical_tool_name_uses_capability_projection(self) -> None:
        capability = ModelCapabilityCatalog.create(
            DomainCapabilitySeed(
                descriptors=(
                    ModelCapabilityDescriptor(
                        canonical_name="pkg.tool",
                        description="Invoke the tool.",
                        parameter_schema={"type": "object"},
                    ),
                )
            )
        )
        provider_name = TextGenerationVendor.provider_tool_name(
            "pkg.tool",
            capability=capability,
            provider_family="openai",
        )

        self.assertEqual(
            TextGenerationVendor.canonical_tool_name(
                provider_name,
                capability=capability,
                provider_family="openai",
            ),
            "pkg.tool",
        )
