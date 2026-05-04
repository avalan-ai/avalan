from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from avalan.cli import __main__ as cli_main


class CudaAndTransformerUtilityTestCase(TestCase):
    def test_lightweight_default_device_prefers_cuda(self) -> None:
        with (
            patch("avalan.cli.__main__._is_cuda_available", return_value=True),
            patch("avalan.cli.__main__._is_mps_available", return_value=True),
        ):
            self.assertEqual(
                cli_main.TransformerModel.get_default_device(), "cuda"
            )

    def test_lightweight_default_device_uses_mps_when_cuda_missing(
        self,
    ) -> None:
        with (
            patch(
                "avalan.cli.__main__._is_cuda_available", return_value=False
            ),
            patch("avalan.cli.__main__._is_mps_available", return_value=True),
        ):
            self.assertEqual(
                cli_main.TransformerModel.get_default_device(), "mps"
            )

    def test_lightweight_default_device_falls_back_to_cpu(self) -> None:
        with (
            patch(
                "avalan.cli.__main__._is_cuda_available", return_value=False
            ),
            patch("avalan.cli.__main__._is_mps_available", return_value=False),
        ):
            self.assertEqual(
                cli_main.TransformerModel.get_default_device(), "cpu"
            )

    def test_cuda_helpers_when_torch_missing(self) -> None:
        with patch("avalan.cli.__main__._module_exists", return_value=False):
            self.assertFalse(cli_main._is_cuda_available())
            self.assertFalse(cli_main._is_mps_available())
            self.assertEqual(cli_main._cuda_device_count(), 1)
            cli_main._set_cuda_device(0)

    def test_cuda_helpers_when_torch_available(self) -> None:
        cuda_module = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 3,
            set_device=MagicMock(),
        )
        with (
            patch("avalan.cli.__main__._module_exists", return_value=True),
            patch(
                "avalan.cli.__main__.import_module", return_value=cuda_module
            ),
        ):
            self.assertTrue(cli_main.is_available())
            self.assertEqual(cli_main.device_count(), 3)
            cli_main.set_device(2)

        cuda_module.set_device.assert_called_once_with(2)

    def test_cuda_helpers_handle_missing_lazy_modules(self) -> None:
        with (
            patch("avalan.cli.__main__._module_exists", return_value=True),
            patch(
                "avalan.cli.__main__.import_module",
                side_effect=ModuleNotFoundError("torch.cuda"),
            ),
        ):
            self.assertFalse(cli_main._is_cuda_available())

        with (
            patch("avalan.cli.__main__._module_exists", return_value=True),
            patch(
                "avalan.cli.__main__.import_module",
                side_effect=ModuleNotFoundError("torch.backends.mps"),
            ),
        ):
            self.assertFalse(cli_main._is_mps_available())

    def test_destroy_process_group_paths(self) -> None:
        with patch("avalan.cli.__main__._module_exists", return_value=False):
            cli_main.destroy_process_group()

        dist_module = SimpleNamespace(destroy_process_group=MagicMock())
        with (
            patch("avalan.cli.__main__._module_exists", return_value=True),
            patch(
                "avalan.cli.__main__.import_module", return_value=dist_module
            ),
        ):
            cli_main.destroy_process_group()

        dist_module.destroy_process_group.assert_called_once_with()

    def test_transformers_utils_and_flash_attention_helpers(self) -> None:
        with patch("avalan.cli.__main__._module_exists", return_value=False):
            self.assertIsNone(cli_main._transformers_utils_module())
            self.assertFalse(cli_main.is_flash_attn_2_available())
            self.assertFalse(cli_main.is_torch_flex_attn_available())

        transformers_utils = SimpleNamespace(
            is_flash_attn_2_available=lambda: True
        )
        with (
            patch("avalan.cli.__main__._module_exists", return_value=True),
            patch(
                "avalan.cli.__main__.import_module",
                return_value=transformers_utils,
            ),
        ):
            self.assertIs(
                cli_main._transformers_utils_module(), transformers_utils
            )
            self.assertTrue(cli_main.is_flash_attn_2_available())
            self.assertFalse(cli_main.is_torch_flex_attn_available())
