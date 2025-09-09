from unittest import TestCase

from avalan.model.vendor import TextGenerationVendor


class VendorEncodeDecodeTestCase(TestCase):
    def test_encode_decode_roundtrip(self) -> None:
        original = "pkg.sub.tool"
        encoded = TextGenerationVendor.encode_tool_name(original)
        self.assertEqual(encoded, "pkg__sub__tool")
        self.assertEqual(
            TextGenerationVendor.decode_tool_name(encoded), original
        )

    def test_encode_decode_noop(self) -> None:
        name = "plain"
        self.assertEqual(TextGenerationVendor.encode_tool_name(name), name)
        self.assertEqual(TextGenerationVendor.decode_tool_name(name), name)
