import unittest
from unittest.mock import patch

from avalan.cli.__main__ import CLI


class DefaultAttentionTestCase(unittest.TestCase):
    def test_cuda_flash(self):
        with (
            patch("avalan.cli.__main__.is_available", return_value=True),
            patch(
                "avalan.cli.__main__.is_flash_attn_2_available",
                return_value=True,
            ),
            patch(
                "avalan.cli.__main__.is_torch_flex_attn_available",
                return_value=False,
            ),
        ):
            self.assertEqual(
                CLI._default_attention("cuda"), "flash_attention_2"
            )

    def test_cuda_flex(self):
        with (
            patch("avalan.cli.__main__.is_available", return_value=True),
            patch(
                "avalan.cli.__main__.is_flash_attn_2_available",
                return_value=False,
            ),
            patch(
                "avalan.cli.__main__.is_torch_flex_attn_available",
                return_value=True,
            ),
        ):
            self.assertEqual(CLI._default_attention("cuda"), "flex_attention")

    def test_mps_sdpa(self):
        with patch("torch.backends.mps.is_available", return_value=True):
            self.assertEqual(CLI._default_attention("mps"), "sdpa")

    def test_none_cuda_unavailable(self):
        with patch("avalan.cli.__main__.is_available", return_value=False):
            self.assertIsNone(CLI._default_attention("cuda"))

    def test_none_cpu(self):
        self.assertIsNone(CLI._default_attention("cpu"))

    def test_exception_ignored(self):
        with patch(
            "avalan.cli.__main__.is_available", side_effect=RuntimeError
        ):
            self.assertIsNone(CLI._default_attention("cuda"))
