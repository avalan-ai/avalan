import sys
from types import ModuleType
from unittest import TestCase
from unittest.mock import patch

from avalan.cli.__main__ import CLI


class CliAttentionDefaultTestCase(TestCase):
    def _create_fake_torch(self, mps_available: bool) -> dict[str, ModuleType]:
        mps_module = ModuleType("torch.backends.mps")
        mps_module.is_available = lambda: mps_available
        backends_module = ModuleType("torch.backends")
        backends_module.mps = mps_module
        torch_module = ModuleType("torch")
        torch_module.backends = backends_module
        return {
            "torch": torch_module,
            "torch.backends": backends_module,
            "torch.backends.mps": mps_module,
        }

    def _parse_attention(self, device: str, **patches: object) -> str | None:
        modules = self._create_fake_torch(patches.pop("mps_available", False))
        with (
            patch.dict(sys.modules, modules, clear=False),
            patch(
                "avalan.cli.__main__.is_available",
                return_value=patches.get("cuda_available", False),
            ),
            patch(
                "avalan.cli.__main__.is_flash_attn_2_available",
                return_value=patches.get("flash2", False),
            ),
            patch(
                "avalan.cli.__main__.is_torch_flex_attn_available",
                return_value=patches.get("flex", False),
            ),
        ):
            parser = CLI._create_parser(device, "/c", "/l", "en")
        args = parser.parse_args(["model", "run", "dummy"])
        return args.attention

    def test_default_mps(self):
        attention = self._parse_attention(
            "mps", mps_available=True, cuda_available=False
        )
        self.assertEqual(attention, "sdpa")

    def test_default_cuda(self):
        attention = self._parse_attention(
            "cuda",
            mps_available=False,
            cuda_available=True,
            flash2=True,
        )
        self.assertEqual(attention, "flash_attention_2")

    def test_default_cpu(self):
        attention = self._parse_attention(
            "cpu", mps_available=False, cuda_available=False
        )
        self.assertIsNone(attention)
