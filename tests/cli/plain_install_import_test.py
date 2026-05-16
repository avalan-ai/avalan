import subprocess
import sys
from pathlib import Path
from unittest import TestCase

_BLOCK_LOCAL_MODEL_DEPENDENCIES_CODE = """
import importlib.abc
import sys

_LOCAL_MODEL_MODULES = (
    "diffusers",
    "sentence_transformers",
    "torch",
    "transformers",
)

class BlockLocalModelDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefixes = tuple(f"{module}." for module in _LOCAL_MODEL_MODULES)
        if fullname in _LOCAL_MODEL_MODULES or fullname.startswith(prefixes):
            raise AssertionError(f"unexpected local model import: {fullname}")
        return None

def local_model_dependency_loaded():
    return any(name in sys.modules for name in _LOCAL_MODEL_MODULES)

sys.meta_path.insert(0, BlockLocalModelDependencies())
sys.path.insert(0, "src")
"""

_OPENAI_STUB_CODE = """
import types
from importlib.machinery import ModuleSpec
from types import SimpleNamespace

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
"""


class PlainInstallImportTestCase(TestCase):
    def _run_code(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
            text=True,
        )

    def test_version_does_not_import_local_model_dependencies(self) -> None:
        code = _BLOCK_LOCAL_MODEL_DEPENDENCIES_CODE + """
from avalan.cli.__main__ import main

sys.argv = ["avalan", "--version"]
main()
print("local_model_loaded", local_model_dependency_loaded())
"""
        result = self._run_code(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("local_model_loaded False", result.stdout)

    def test_openai_vendor_load_does_not_import_local_model_dependencies(
        self,
    ) -> None:
        code = _BLOCK_LOCAL_MODEL_DEPENDENCIES_CODE + _OPENAI_STUB_CODE + """
from contextlib import AsyncExitStack
from logging import getLogger

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
print(
    type(model).__name__,
    "local_model_loaded",
    local_model_dependency_loaded(),
)
"""
        result = self._run_code(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OpenAIModel local_model_loaded False", result.stdout)

    def test_openai_agent_load_does_not_import_local_model_dependencies(
        self,
    ) -> None:
        code = _BLOCK_LOCAL_MODEL_DEPENDENCIES_CODE + _OPENAI_STUB_CODE + """
import asyncio
from contextlib import AsyncExitStack
from logging import getLogger
from types import SimpleNamespace
from uuid import uuid4

from avalan.agent.loader import OrchestratorLoader
from avalan.entities import OrchestratorSettings

async def main():
    async with AsyncExitStack() as stack:
        settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"name": "agent", "role": "assistant"},
            uri="ai://openai/gpt-test",
            engine_config={"access_token": "test"},
            tools=[],
            call_options=None,
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=True,
            sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            log_events=True,
        )
        loader = OrchestratorLoader(
            hub=SimpleNamespace(),
            logger=getLogger("test"),
            participant_id=uuid4(),
            stack=stack,
        )
        orchestrator = await loader.from_settings(settings)
        orchestrator = await stack.enter_async_context(orchestrator)
        print(
            type(orchestrator.engine).__name__,
            "local_model_loaded",
            local_model_dependency_loaded(),
        )

asyncio.run(main())
"""
        result = self._run_code(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OpenAIModel local_model_loaded False", result.stdout)

    def test_agent_server_setup_does_not_import_local_model_dependencies(
        self,
    ) -> None:
        code = _BLOCK_LOCAL_MODEL_DEPENDENCIES_CODE + """
from logging import getLogger
from types import SimpleNamespace
from uuid import uuid4

from avalan.agent.loader import OrchestratorLoader
from avalan.entities import OrchestratorSettings
from avalan.server import agents_server

settings = OrchestratorSettings(
    agent_id=uuid4(),
    orchestrator_type=None,
    agent_config={"name": "agent", "role": "assistant"},
    uri="ai://openai/gpt-test",
    engine_config={"access_token": "test"},
    tools=[],
    call_options=None,
    template_vars=None,
    memory_permanent_message=None,
    permanent_memory=None,
    memory_recent=True,
    sentence_model_id=OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
    sentence_model_engine_config=None,
    sentence_model_max_tokens=500,
    sentence_model_overlap_size=125,
    sentence_model_window_size=250,
    json_config=None,
    log_events=True,
)
server = agents_server(
    hub=SimpleNamespace(),
    name="avalan",
    version="test",
    host="127.0.0.1",
    port=0,
    reload=False,
    specs_path=None,
    settings=settings,
    tool_settings=None,
    mcp_prefix="/mcp",
    openai_prefix="/v1",
    mcp_name="run",
    logger=getLogger("test"),
    protocols={
        "a2a": set(),
        "mcp": set(),
        "openai": {"completions", "responses"},
    },
)
print(
    type(server).__name__,
    "local_model_loaded",
    local_model_dependency_loaded(),
)
"""
        result = self._run_code(code)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Server local_model_loaded False", result.stdout)
