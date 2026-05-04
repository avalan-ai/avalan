import subprocess
import sys
from pathlib import Path
from unittest import TestCase


class PlainInstallImportTestCase(TestCase):
    def test_version_does_not_import_torch(self) -> None:
        code = """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError(f"unexpected torch import: {fullname}")
        return None

sys.meta_path.insert(0, BlockTorch())
sys.path.insert(0, "src")

from avalan.cli.__main__ import main

sys.argv = ["avalan", "--version"]
main()
print("torch_loaded", "torch" in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("torch_loaded False", result.stdout)

    def test_openai_vendor_load_does_not_import_torch(self) -> None:
        code = """
import importlib.abc
import sys
import types
from contextlib import AsyncExitStack
from importlib.machinery import ModuleSpec
from logging import getLogger
from types import SimpleNamespace

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError(f"unexpected torch import: {fullname}")
        return None

sys.meta_path.insert(0, BlockTorch())
sys.path.insert(0, "src")

openai = types.ModuleType("openai")
openai.__spec__ = ModuleSpec("openai", loader=None)

class Omit:
    pass

class AsyncOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = SimpleNamespace(create=None)

openai.AsyncOpenAI = AsyncOpenAI
openai.Omit = Omit
sys.modules["openai"] = openai

from avalan.entities import Modality, TransformerEngineSettings
from avalan.model.manager import ModelManager
from avalan.model.modalities import ModalityRegistry

uri = ModelManager.parse_uri("ai://openai/gpt-test")
model = ModalityRegistry.load_engine(
    uri,
    TransformerEngineSettings(access_token="test"),
    Modality.TEXT_GENERATION,
    getLogger("test"),
    AsyncExitStack(),
)
print(type(model).__name__, "torch_loaded", "torch" in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OpenAIModel torch_loaded False", result.stdout)
