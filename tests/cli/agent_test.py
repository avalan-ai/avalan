import tomllib
import unittest
from argparse import Namespace
from asyncio import CancelledError
from base64 import b64encode
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from io import StringIO
from logging import getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch
from uuid import uuid4

from rich.syntax import Syntax

from avalan.agent import Specification
from avalan.agent.engine import EngineAgent
from avalan.cli.commands import agent as agent_cmds
from avalan.cli.display import CliStreamDisplayConfig
from avalan.container import (
    ContainerBackend,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    container_backend_capability_profile,
)
from avalan.entities import (
    EngineMessage,
    EngineUri,
    GenerationCacheStrategy,
    GenerationSettings,
    Message,
    MessageContentFile,
    MessageRole,
    Modality,
    Operation,
    OperationParameters,
    OperationTextParameters,
    OrchestratorSettings,
    ReasoningSettings,
    ReasoningSummaryMode,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallRecoveryFormat,
    ToolCallToken,
    ToolFormat,
    ToolNamePolicyMode,
)
from avalan.event import Event, EventType
from avalan.event.manager import EventManager, EventManagerMode
from avalan.memory import RecentMessageMemory
from avalan.memory.manager import MemoryManager
from avalan.memory.permanent import PermanentMessageMemory, VectorFunction
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.model.stream import StreamItemKind, StreamProviderEvent
from avalan.sandbox import BubblewrapSandboxBackend, SeatbeltSandboxBackend
from avalan.tool.browser import BrowserToolSettings
from avalan.tool.context import ToolSettingsContext
from avalan.tool.database import DatabaseToolSettings
from avalan.tool.graph_settings import GraphToolSettings
from avalan.tool.manager import ToolManager, ToolManagerSettings
from avalan.tool.parser import ToolCallParser
from avalan.tool.shell import ShellGitToolSettings, ShellToolSettings


def _apple_container_backend_class() -> type[ContainerFakeBackend]:
    def __init__(self: ContainerFakeBackend) -> None:
        ContainerFakeBackend.__init__(
            self,
            ContainerFakeBackendScript(
                capabilities=container_backend_capability_profile(
                    "apple-container-macos-linux"
                ).capabilities,
            ),
        )

    return cast(
        type[ContainerFakeBackend],
        type(
            "AppleContainerBackend",
            (ContainerFakeBackend,),
            {"__init__": __init__},
        ),
    )


class CliAgentMessageSearchTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = Namespace(
            specifications_file="spec.toml",
            id="aid",
            participant="pid",
            session="sid",
            function=VectorFunction.L2_DISTANCE,
            no_repl=False,
            quiet=False,
            skip_hub_access_check=False,
            limit=1,
            tool_events=2,
            tool=None,
            run_max_new_tokens=100,
            backend="transformers",
        )
        self.console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        self.console.status.return_value = status_cm
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme.icons = {"user_input": ">"}
        self.theme.get_spinner.return_value = "sp"
        self.theme.agent.return_value = "agent_panel"
        self.theme.search_message_matches.return_value = "matches_panel"
        self.hub = MagicMock()
        self.hub.can_access.return_value = True
        self.hub.model.side_effect = lambda m: f"mdl-{m}"
        self.logger = MagicMock()

    async def test_returns_when_no_input(self):
        with (
            patch.object(agent_cmds, "get_input", return_value=None) as gi,
            patch.object(
                agent_cmds.OrchestratorLoader, "from_file", new=AsyncMock()
            ) as lf,
        ):
            await agent_cmds.agent_message_search(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        gi.assert_called_once()
        lf.assert_not_called()
        self.console.print.assert_not_called()

    async def test_search_messages(self):
        orch = MagicMock()
        orch.engine_agent = True
        orch.engine = MagicMock(model_id="m")
        orch.model_ids = ["m"]
        orch.memory.search_messages = AsyncMock(return_value=["msg"])

        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=orch),
            ) as lf,
        ):
            await agent_cmds.agent_message_search(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        lf.assert_awaited_once()
        self.assertIs(
            lf.call_args.kwargs["event_manager_mode"],
            EventManagerMode.CLI,
        )
        dummy_stack.enter_async_context.assert_awaited_once_with(orch)
        orch.memory.search_messages.assert_awaited_once_with(
            search="hi",
            agent_id="aid",
            search_user_messages=False,
            session_id="sid",
            participant_id="pid",
            function=VectorFunction.L2_DISTANCE,
            limit=1,
        )
        self.console.print.assert_any_call("agent_panel")
        self.console.print.assert_any_call("matches_panel")

    async def test_search_messages_from_file_forwards_shell_settings(self):
        self.args.tool_shell_max_head_lines = 13
        self.args.tool_shell_allow_pipelines = True
        self.args.tool_shell_max_pipeline_stages = 3
        self.args.tool_shell_max_pipeline_bytes = 1024
        self.args.tool_shell_max_intermediate_bytes = 512
        orch = MagicMock()
        orch.engine_agent = True
        orch.engine = MagicMock(model_id="m")
        orch.model_ids = ["m"]
        orch.memory.search_messages = AsyncMock(return_value=["msg"])

        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=orch),
            ) as lf,
        ):
            await agent_cmds.agent_message_search(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        lf.assert_awaited_once()
        self.assertIs(
            lf.call_args.kwargs["event_manager_mode"],
            EventManagerMode.CLI,
        )
        tool_settings = lf.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertEqual(tool_settings.shell.max_head_lines, 13)
        self.assertTrue(tool_settings.shell.allow_pipelines)
        self.assertEqual(tool_settings.shell.max_pipeline_stages, 3)
        self.assertEqual(tool_settings.shell.max_pipeline_bytes, 1024)
        self.assertEqual(tool_settings.shell.max_intermediate_bytes, 512)

    async def test_search_messages_from_settings(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.name = None
        self.args.task = None
        self.args.instructions = None
        self.args.memory_recent = None
        self.args.memory_permanent_message = None
        self.args.memory_permanent = None
        self.args.memory_engine_model_id = (
            agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID
        )
        self.args.memory_engine_max_tokens = 500
        self.args.memory_engine_overlap = 125
        self.args.memory_engine_window = 250
        self.args.run_max_new_tokens = None
        self.args.run_skip_special_tokens = False
        self.args.tool_browser_engine = None
        self.args.tool_browser_debug = None
        self.args.tool_browser_search = None
        self.args.tool_browser_search_context = None
        self.args.tool = None

        orch = MagicMock()
        orch.engine_agent = True
        orch.engine = MagicMock(model_id="m")
        orch.model_ids = ["m"]
        orch.memory.search_messages = AsyncMock(return_value=["msg"])

        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=orch),
            ) as lfs,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ) as lf,
        ):
            await agent_cmds.agent_message_search(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        lfs.assert_awaited_once()
        self.assertIs(
            lfs.call_args.kwargs["event_manager_mode"],
            EventManagerMode.CLI,
        )
        settings = lfs.call_args.args[0]
        self.assertIsNone(settings.tools)
        lf.assert_not_called()
        dummy_stack.enter_async_context.assert_awaited_once_with(orch)
        orch.memory.search_messages.assert_awaited_once()


def _serve_from_settings_args(**overrides: object) -> Namespace:
    values = {
        "host": "0.0.0.0",
        "port": 80,
        "specifications_file": None,
        "openai_prefix": "oa",
        "mcp_name": "run",
        "mcp_description": None,
        "reload": False,
        "backend": "transformers",
        "engine_uri": "uri",
        "role": "assistant",
        "name": None,
        "task": None,
        "instructions": None,
        "memory_recent": None,
        "memory_permanent_message": None,
        "memory_permanent": None,
        "memory_engine_model_id": (
            agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID
        ),
        "memory_engine_max_tokens": 500,
        "memory_engine_overlap": 125,
        "memory_engine_window": 250,
        "run_max_new_tokens": None,
        "run_skip_special_tokens": False,
        "tool": None,
        "id": None,
        "participant": "pid",
        "cors_origin": None,
        "cors_origin_regex": None,
        "cors_method": None,
        "cors_header": None,
        "cors_credentials": False,
    }
    values.update(overrides)
    return Namespace(**values)


class CliAgentServeTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_agent_serve(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            specifications_file=None,
            openai_prefix="oa",
            mcp_prefix="mcp",
            a2a_prefix="a2a",
            mcp_name="run",
            mcp_description=None,
            reload=False,
            backend="transformers",
            id=None,
            participant="pid",
            cors_origin=None,
            cors_origin_regex=None,
            cors_method=None,
            cors_header=None,
            cors_credentials=False,
        )
        hub = MagicMock()
        logger = MagicMock()
        server = MagicMock()
        server.serve = AsyncMock()

        with NamedTemporaryFile("w") as spec:
            args.specifications_file = spec.name
            with patch.object(
                agent_cmds, "agents_server", return_value=server
            ) as asrv:
                await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")

            asrv.assert_called_once_with(
                hub=hub,
                name="name",
                version="1.0",
                mcp_prefix="mcp",
                openai_prefix="oa",
                a2a_prefix="a2a",
                mcp_name="run",
                mcp_description=None,
                a2a_tool_name="run",
                a2a_tool_description=None,
                specs_path=spec.name,
                settings=None,
                tool_settings=ToolSettingsContext(),
                tool_name_policy=None,
                host="0.0.0.0",
                port=80,
                reload=False,
                logger=logger,
                agent_id=None,
                participant_id="pid",
                allow_origins=None,
                allow_origin_regex=None,
                allow_methods=None,
                allow_headers=None,
                allow_credentials=False,
                protocols=None,
            )
        server.serve.assert_awaited_once()

    async def test_agent_serve_cors_args_forwarded(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            specifications_file=None,
            openai_prefix="oa",
            mcp_prefix="mcp",
            a2a_prefix="/a2a",
            mcp_name="run",
            mcp_description=None,
            reload=False,
            backend="transformers",
            id=None,
            participant="pid",
            cors_origin=["https://a"],
            cors_origin_regex="^https://.*$",
            cors_method=["GET"],
            cors_header=["X-Test"],
            cors_credentials=True,
        )
        hub = MagicMock()
        logger = MagicMock()
        server = MagicMock()
        server.serve = AsyncMock()

        with NamedTemporaryFile("w") as spec:
            args.specifications_file = spec.name
            with patch.object(
                agent_cmds, "agents_server", return_value=server
            ) as asrv:
                await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")

            asrv.assert_called_once_with(
                hub=hub,
                name="name",
                version="1.0",
                mcp_prefix="mcp",
                openai_prefix="oa",
                a2a_prefix="/a2a",
                mcp_name="run",
                mcp_description=None,
                a2a_tool_name="run",
                a2a_tool_description=None,
                specs_path=spec.name,
                settings=None,
                tool_settings=ToolSettingsContext(),
                tool_name_policy=None,
                host="0.0.0.0",
                port=80,
                reload=False,
                logger=logger,
                agent_id=None,
                participant_id="pid",
                allow_origins=["https://a"],
                allow_origin_regex="^https://.*$",
                allow_methods=["GET"],
                allow_headers=["X-Test"],
                allow_credentials=True,
                protocols=None,
            )
        server.serve.assert_awaited_once()

    async def test_agent_serve_from_settings(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            specifications_file=None,
            openai_prefix="oa",
            mcp_name="run",
            mcp_description=None,
            reload=False,
            backend="transformers",
            engine_uri="uri",
            role="assistant",
            name=None,
            task=None,
            instructions=None,
            memory_recent=None,
            memory_permanent_message=None,
            memory_permanent=None,
            memory_engine_model_id=agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            run_max_new_tokens=None,
            run_skip_special_tokens=False,
            tool=None,
            tool_browser_engine=None,
            tool_browser_debug=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            tools_confirm=False,
            id=None,
            participant="pid",
            cors_origin=None,
            cors_origin_regex=None,
            cors_method=None,
            cors_header=None,
            cors_credentials=False,
        )
        hub = MagicMock()
        logger = MagicMock()
        server = MagicMock()
        server.serve = AsyncMock()
        settings = MagicMock()
        browser_settings = MagicMock()

        with (
            patch.object(
                agent_cmds, "get_orchestrator_settings", return_value=settings
            ) as gos,
            patch.object(
                agent_cmds,
                "get_tool_settings",
                side_effect=[browser_settings, None, None, None],
            ) as gts,
            patch.object(
                agent_cmds, "agents_server", return_value=server
            ) as asrv,
        ):
            await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")

        gos.assert_called_once()
        self.assertIsNone(gos.call_args.kwargs["tools"])
        gts.assert_has_calls(
            [
                call(args, prefix="browser", settings_cls=BrowserToolSettings),
                call(
                    args, prefix="database", settings_cls=DatabaseToolSettings
                ),
                call(args, prefix="graph", settings_cls=GraphToolSettings),
                call(args, prefix="shell", settings_cls=ShellToolSettings),
            ]
        )
        asrv.assert_called_once_with(
            hub=hub,
            name="name",
            version="1.0",
            mcp_prefix="/mcp",
            openai_prefix="oa",
            a2a_prefix="/a2a",
            mcp_name="run",
            mcp_description=None,
            a2a_tool_name="run",
            a2a_tool_description=None,
            specs_path=None,
            settings=settings,
            tool_settings=ToolSettingsContext(
                browser=browser_settings, database=None, graph=None
            ),
            tool_name_policy=None,
            host="0.0.0.0",
            port=80,
            reload=False,
            logger=logger,
            agent_id=None,
            participant_id="pid",
            allow_origins=None,
            allow_origin_regex=None,
            allow_methods=None,
            allow_headers=None,
            allow_credentials=False,
            protocols=None,
        )
        server.serve.assert_awaited_once()

    async def test_agent_serve_from_file_forwards_shell_settings(self):
        args = _serve_from_settings_args(
            tool_shell_max_head_lines=9,
            tool_shell_allow_pipelines=True,
            tool_shell_max_pipeline_stages=3,
            tool_shell_max_pipeline_bytes=1024,
            tool_shell_max_intermediate_bytes=512,
        )
        hub = MagicMock()
        logger = MagicMock()
        server = MagicMock()
        server.serve = AsyncMock()

        with NamedTemporaryFile("w") as spec:
            args.specifications_file = spec.name
            args.engine_uri = None
            with patch.object(
                agent_cmds, "agents_server", return_value=server
            ) as asrv:
                await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")

        asrv.assert_called_once()
        self.assertEqual(asrv.call_args.kwargs["specs_path"], spec.name)
        self.assertIsNone(asrv.call_args.kwargs["settings"])
        tool_settings = asrv.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertEqual(tool_settings.shell.max_head_lines, 9)
        self.assertTrue(tool_settings.shell.allow_pipelines)
        self.assertEqual(tool_settings.shell.max_pipeline_stages, 3)
        self.assertEqual(tool_settings.shell.max_pipeline_bytes, 1024)
        self.assertEqual(tool_settings.shell.max_intermediate_bytes, 512)
        server.serve.assert_awaited_once()

    async def test_agent_serve_shell_settings_preserve_default_tools(self):
        args = _serve_from_settings_args(
            tool_shell_max_head_lines=9,
        )
        hub = MagicMock()
        logger = MagicMock()
        server = MagicMock()
        server.serve = AsyncMock()

        with patch.object(
            agent_cmds, "agents_server", return_value=server
        ) as asrv:
            await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")

        asrv.assert_called_once()
        settings = asrv.call_args.kwargs["settings"]
        self.assertIsNone(settings.tools)
        tool_settings = asrv.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertEqual(tool_settings.shell.max_head_lines, 9)
        self.assertIsNone(tool_settings.browser)
        self.assertIsNone(tool_settings.database)
        self.assertIsNone(tool_settings.graph)
        server.serve.assert_awaited_once()

    async def test_agent_serve_shell_explicit_tool_is_preserved(self):
        args = _serve_from_settings_args(
            tool=["shell.cat"],
            tool_shell_max_head_lines=9,
        )
        hub = MagicMock()
        logger = MagicMock()
        server = MagicMock()
        server.serve = AsyncMock()

        with patch.object(
            agent_cmds, "agents_server", return_value=server
        ) as asrv:
            await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")

        asrv.assert_called_once()
        settings = asrv.call_args.kwargs["settings"]
        self.assertEqual(settings.tools, ["shell.cat"])
        tool_settings = asrv.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertEqual(tool_settings.shell.max_head_lines, 9)
        server.serve.assert_awaited_once()

    def test_agent_tool_settings_accept_shell_git_read_only_profile(
        self,
    ) -> None:
        settings = agent_cmds._agent_tool_settings(
            Namespace(
                tool_shell_git_workspace_root="/workspace",
                tool_shell_git_cwd="repo",
                tool_shell_git_capabilities=["read"],
                tool_shell_git_allowed_commands=["status", "diff", "log"],
                tool_shell_git_max_log_count=12,
                tool_shell_git_max_diff_bytes=8192,
            )
        )

        self.assertIsInstance(settings.shell, ShellToolSettings)
        assert settings.shell is not None
        git_settings = settings.shell.git
        self.assertIsInstance(git_settings, ShellGitToolSettings)
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(git_settings.workspace_root, "/workspace")
        self.assertEqual(git_settings.cwd, "repo")
        self.assertEqual(git_settings.capabilities, ("read",))
        self.assertEqual(
            git_settings.allowed_commands,
            ("status", "diff", "log"),
        )
        self.assertEqual(git_settings.max_log_count, 12)
        self.assertEqual(git_settings.max_diff_bytes, 8192)
        self.assertEqual(
            settings.shell_explicit_fields,
            frozenset(
                {
                    "git.workspace_root",
                    "git.cwd",
                    "git.capabilities",
                    "git.allowed_commands",
                    "git.max_log_count",
                    "git.max_diff_bytes",
                }
            ),
        )

    def test_agent_tool_settings_routes_git_executable_path_mapping(
        self,
    ) -> None:
        settings = agent_cmds._agent_tool_settings(
            Namespace(
                tool_shell_executable_paths=[
                    ("git", "/usr/bin/git"),
                    ("rg", "/opt/homebrew/bin/rg"),
                ],
            )
        )

        self.assertIsInstance(settings.shell, ShellToolSettings)
        assert settings.shell is not None
        self.assertEqual(
            settings.shell.executable_paths,
            {"rg": "/opt/homebrew/bin/rg"},
        )
        git_settings = settings.shell.git
        self.assertIsInstance(git_settings, ShellGitToolSettings)
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(git_settings.executable_path, "/usr/bin/git")
        self.assertEqual(
            settings.shell_explicit_fields,
            frozenset({"executable_paths", "git.executable_path"}),
        )

    def test_agent_tool_settings_routes_only_git_executable_path_mapping(
        self,
    ) -> None:
        settings = agent_cmds._agent_tool_settings(
            Namespace(
                tool_shell_executable_paths=[
                    ("git", "/usr/bin/git"),
                ],
            )
        )

        self.assertIsInstance(settings.shell, ShellToolSettings)
        assert settings.shell is not None
        self.assertEqual(settings.shell.executable_paths, {})
        git_settings = settings.shell.git
        self.assertIsInstance(git_settings, ShellGitToolSettings)
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(git_settings.executable_path, "/usr/bin/git")
        self.assertEqual(
            settings.shell_explicit_fields,
            frozenset({"git.executable_path"}),
        )

    def test_agent_tool_settings_merges_matching_git_executable_path(
        self,
    ) -> None:
        settings = agent_cmds._agent_tool_settings(
            Namespace(
                tool_shell_executable_paths=[
                    ("git", "/usr/bin/git"),
                ],
                tool_shell_git_executable_path="/usr/bin/git",
                tool_shell_git_allowed_commands=["status"],
            )
        )

        self.assertIsInstance(settings.shell, ShellToolSettings)
        assert settings.shell is not None
        git_settings = settings.shell.git
        self.assertIsInstance(git_settings, ShellGitToolSettings)
        assert isinstance(git_settings, ShellGitToolSettings)
        self.assertEqual(git_settings.executable_path, "/usr/bin/git")
        self.assertEqual(git_settings.allowed_commands, ("status",))

    def test_agent_tool_settings_rejects_conflicting_git_executable_path(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            AssertionError,
            "git executable path settings conflict",
        ):
            agent_cmds._agent_tool_settings(
                Namespace(
                    tool_shell_executable_paths=[
                        ("git", "/usr/bin/git"),
                    ],
                    tool_shell_git_executable_path="/opt/homebrew/bin/git",
                )
            )

    def test_shell_executable_path_explicit_fields_ignores_invalid_shape(
        self,
    ) -> None:
        explicit_fields = (
            agent_cmds._tool_settings_explicit_fields_from_mapping(
                {"tool_shell_executable_paths": "git=/usr/bin/git"},
                prefix="shell",
                settings_cls=ShellToolSettings,
            )
        )

        self.assertEqual(explicit_fields, frozenset({"executable_paths"}))

    def test_shell_git_tool_settings_track_mapping_inputs(self) -> None:
        mapping = {
            "tool_shell_git_allowed_commands": ["status", "log"],
            "tool_shell_git_max_log_count": 12,
        }

        settings = agent_cmds.get_tool_settings(
            mapping,
            prefix="shell",
            settings_cls=ShellToolSettings,
        )
        explicit_fields = (
            agent_cmds._tool_settings_explicit_fields_from_mapping(
                mapping,
                prefix="shell",
                settings_cls=ShellToolSettings,
            )
        )

        self.assertIsInstance(settings, ShellToolSettings)
        assert isinstance(settings, ShellToolSettings)
        self.assertIsInstance(settings.git, ShellGitToolSettings)
        assert isinstance(settings.git, ShellGitToolSettings)
        self.assertEqual(settings.git.allowed_commands, ("status", "log"))
        self.assertEqual(settings.git.max_log_count, 12)
        self.assertEqual(
            explicit_fields,
            frozenset({"git.allowed_commands", "git.max_log_count"}),
        )

    async def test_agent_serve_needs_settings(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            specifications_file=None,
            openai_prefix="oa",
            mcp_prefix="mcp",
            a2a_prefix="/a2a",
            mcp_name="run",
            mcp_description=None,
            reload=False,
            backend="transformers",
            engine_uri=None,
            role=None,
            name=None,
            task=None,
            instructions=None,
            memory_recent=None,
            memory_permanent_message=None,
            memory_permanent=None,
            memory_engine_model_id=agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            run_max_new_tokens=None,
            run_skip_special_tokens=False,
            tool=None,
            tool_browser_engine=None,
            tool_browser_debug=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            tools_confirm=False,
            id=None,
            participant="pid",
            cors_origin=None,
            cors_origin_regex=None,
            cors_method=None,
            cors_header=None,
            cors_credentials=False,
        )
        hub = MagicMock()
        logger = MagicMock()

        with self.assertRaises(AssertionError):
            await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")


class CliAgentProxyTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_agent_proxy_defaults_and_forward(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            openai_prefix="oa",
            mcp_prefix="mcp",
            a2a_prefix="/a2a",
            mcp_name="run",
            mcp_description=None,
            reload=False,
            backend="transformers",
            engine_uri="uri",
            name=None,
            role=None,
            task=None,
            instructions=None,
            memory_recent=None,
            memory_permanent_message=None,
            memory_permanent=None,
            memory_engine_model_id=agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            run_max_new_tokens=None,
            run_skip_special_tokens=False,
            tool=None,
            tool_browser_engine=None,
            tool_browser_debug=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            tools_confirm=False,
            id=None,
            participant="pid",
            cors_origin=["https://a"],
            cors_origin_regex="^https://.*$",
            cors_method=["GET"],
            cors_header=["X-Test"],
            cors_credentials=True,
        )
        hub = MagicMock()
        logger = MagicMock()

        with patch.object(
            agent_cmds, "agent_serve", AsyncMock()
        ) as serve_mock:
            await agent_cmds.agent_proxy(args, hub, logger, "name", "1.0")

        serve_mock.assert_awaited_once_with(args, hub, logger, "name", "1.0")
        self.assertEqual(args.name, "Proxy")
        self.assertTrue(args.memory_recent)
        self.assertEqual(
            args.memory_permanent_message,
            "postgresql://avalan:password@localhost:5432/avalan",
        )
        self.assertIsNone(args.specifications_file)
        self.assertEqual(args.cors_origin, ["https://a"])
        self.assertEqual(args.cors_origin_regex, "^https://.*$")
        self.assertEqual(args.cors_method, ["GET"])
        self.assertEqual(args.cors_header, ["X-Test"])
        self.assertTrue(args.cors_credentials)

    async def test_agent_proxy_forwards_shell_settings(self):
        args = _serve_from_settings_args(
            name=None,
            memory_recent=None,
            memory_permanent_message=None,
            tool=["shell.cat"],
            tool_shell_max_stdout_bytes=2048,
            tool_shell_allow_pipelines=True,
            tool_shell_max_pipeline_stages=3,
            tool_shell_max_pipeline_bytes=1024,
            tool_shell_max_intermediate_bytes=512,
        )
        hub = MagicMock()
        logger = MagicMock()
        server = MagicMock()
        server.serve = AsyncMock()

        with patch.object(
            agent_cmds, "agents_server", return_value=server
        ) as asrv:
            await agent_cmds.agent_proxy(args, hub, logger, "name", "1.0")

        asrv.assert_called_once()
        self.assertEqual(args.name, "Proxy")
        self.assertTrue(args.memory_recent)
        self.assertEqual(
            args.memory_permanent_message,
            "postgresql://avalan:password@localhost:5432/avalan",
        )
        settings = asrv.call_args.kwargs["settings"]
        self.assertEqual(settings.tools, ["shell.cat"])
        tool_settings = asrv.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertEqual(tool_settings.shell.max_stdout_bytes, 2048)
        self.assertTrue(tool_settings.shell.allow_pipelines)
        self.assertEqual(tool_settings.shell.max_pipeline_stages, 3)
        self.assertEqual(tool_settings.shell.max_pipeline_bytes, 1024)
        self.assertEqual(tool_settings.shell.max_intermediate_bytes, 512)
        server.serve.assert_awaited_once()

    async def test_agent_proxy_requires_engine(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            openai_prefix="oa",
            mcp_prefix="mcp",
            a2a_prefix="/a2a",
            mcp_name="run",
            mcp_description=None,
            reload=False,
            backend="transformers",
            engine_uri=None,
            id=None,
            participant="pid",
            cors_origin=None,
            cors_origin_regex=None,
            cors_method=None,
            cors_header=None,
            cors_credentials=False,
        )
        hub = MagicMock()
        logger = MagicMock()

        with self.assertRaises(AssertionError):
            await agent_cmds.agent_proxy(args, hub, logger, "name", "1.0")


class CliAgentInitTestCase(unittest.IsolatedAsyncioTestCase):
    def _agent_init_args(self, **overrides: object) -> Namespace:
        values = {
            "name": "N",
            "role": "R",
            "task": "T",
            "instructions": None,
            "goal_instructions": "I",
            "memory_recent": True,
            "memory_permanent_message": "",
            "memory_permanent": None,
            "memory_engine_model_id": None,
            "memory_engine_max_tokens": 500,
            "memory_engine_overlap": 125,
            "memory_engine_window": 250,
            "engine_uri": "uri",
            "run_max_new_tokens": 10,
            "run_skip_special_tokens": True,
            "run_temperature": None,
            "run_top_k": None,
            "run_top_p": None,
            "run_use_cache": None,
            "run_cache_strategy": None,
            "tool": None,
            "backend": "transformers",
            "no_repl": False,
            "quiet": False,
            "tool_shell_execution_mode": None,
            "tool_container_backend": None,
            "tool_container_profile": None,
            "tool_container_image": None,
            "tool_container_workspace_root": None,
            "tool_container_pull_policy": None,
            "tool_container_platform": None,
            "tool_container_cpu_count": None,
            "tool_container_memory_bytes": None,
            "tool_container_pids": None,
            "tool_container_timeout_seconds": None,
            "tool_container_network_mode": None,
            "tool_container_review_mode": None,
            "tool_shell_container_profile": None,
            "tool_shell_container_required": None,
            "tool_sandbox_backend": None,
            "tool_sandbox_profile": None,
            "tool_sandbox_trusted_executables": None,
            "tool_sandbox_executable_search_roots": None,
            "tool_sandbox_read_roots": None,
            "tool_sandbox_write_roots": None,
            "tool_sandbox_deny_roots": None,
            "tool_sandbox_scratch_roots": None,
            "tool_sandbox_output_roots": None,
            "tool_sandbox_network_mode": None,
            "tool_sandbox_network_egress": None,
            "tool_sandbox_timeout_seconds": None,
            "tool_sandbox_pids": None,
            "tool_sandbox_max_stdout_bytes": None,
            "tool_sandbox_max_stderr_bytes": None,
            "tool_sandbox_max_artifact_bytes": None,
            "tool_sandbox_allow_artifacts": None,
            "tool_sandbox_child_processes": None,
            "tool_sandbox_inherited_fds": None,
            "tool_shell_sandbox_profile": None,
            "tool_shell_sandbox_required": None,
        }
        values.update(overrides)
        return Namespace(**values)

    async def test_agent_init(self):
        args = Namespace(
            name=None,
            role=None,
            task=None,
            instructions=None,
            memory_recent=None,
            memory_permanent_message=None,
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            engine_uri=None,
            run_max_new_tokens=None,
            run_skip_special_tokens=True,
            run_temperature=0.55,
            run_top_k=10,
            run_top_p=0.9,
            run_use_cache=False,
            run_cache_strategy="dynamic",
            tool=None,
            backend="transformers",
            no_repl=False,
            quiet=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template
        template.render.return_value = "rendered"

        with (
            patch.object(
                agent_cmds.Prompt, "ask", side_effect=["A", "", "uri"]
            ),
            patch.object(
                agent_cmds, "get_input", side_effect=["r", "t", "i"]
            ) as input_patch,
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_called_once()
        settings = template.render.call_args.kwargs["orchestrator"]
        self.assertTrue(settings.call_options["skip_special_tokens"])
        self.assertEqual(settings.call_options["max_new_tokens"], 1024)
        self.assertEqual(settings.call_options["temperature"], 0.55)
        self.assertEqual(settings.call_options["top_k"], 10)
        self.assertEqual(settings.call_options["top_p"], 0.9)
        self.assertFalse(settings.call_options["use_cache"])
        self.assertEqual(settings.call_options["cache_strategy"], "dynamic")
        self.assertEqual(settings.agent_config["goal_instructions"], "i")
        self.assertNotIn("instructions", settings.agent_config)
        self.assertEqual(
            input_patch.call_args_list[2].args[1],
            "Agent goal instructions ",
        )
        self.assertIsInstance(console.print.call_args.args[0], Syntax)
        self.assertIsInstance(console.print.call_args.args[0], Syntax)

    async def test_agent_init_output_without_tool_settings(self):
        args = Namespace(
            name="N",
            role="R",
            task="T",
            instructions=None,
            goal_instructions="I",
            memory_recent=True,
            memory_permanent_message="",
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            engine_uri="uri",
            run_max_new_tokens=10,
            run_skip_special_tokens=True,
            run_temperature=None,
            run_top_k=None,
            run_top_p=None,
            run_use_cache=None,
            run_cache_strategy=None,
            tool=None,
            tool_browser_engine=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
            backend="transformers",
            no_repl=False,
            quiet=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        @dataclass(frozen=True, kw_only=True, slots=True)
        class PatchedDatabaseToolSettings(DatabaseToolSettings):
            def items(self):
                return asdict(self).items()

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T", "I"]),
            patch.object(
                agent_cmds.Prompt, "ask", side_effect=["N", "", "uri"]
            ),
            patch.object(
                agent_cmds, "DatabaseToolSettings", PatchedDatabaseToolSettings
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertNotIn("[tool.browser.open]", output)
        self.assertNotIn("[tool.database]", output)
        self.assertNotIn("[tool.shell]", output)

    async def test_agent_init_tool_settings_output(self):
        args = Namespace(
            name="N",
            role="R",
            task="T",
            instructions=None,
            goal_instructions="I",
            memory_recent=True,
            memory_permanent_message="",
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            engine_uri="uri",
            run_max_new_tokens=10,
            run_skip_special_tokens=True,
            run_temperature=None,
            run_top_k=None,
            run_top_p=None,
            run_use_cache=None,
            run_cache_strategy=None,
            tool=None,
            tool_browser_engine="chromium",
            tool_browser_search=True,
            tool_browser_search_context=5,
            backend="transformers",
            no_repl=False,
            quiet=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        @dataclass(frozen=True, kw_only=True, slots=True)
        class PatchedDatabaseToolSettings(DatabaseToolSettings):
            def items(self):
                return asdict(self).items()

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T", "I"]),
            patch.object(
                agent_cmds.Prompt, "ask", side_effect=["N", "", "uri"]
            ),
            patch.object(
                agent_cmds, "DatabaseToolSettings", PatchedDatabaseToolSettings
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertIn("[tool.browser.open]", output)
        self.assertIn('engine = "chromium"', output)
        self.assertIn("search = true", output)

    async def test_agent_init_database_tool_settings_output(self):
        args = Namespace(
            name="N",
            role="R",
            task="T",
            instructions=None,
            goal_instructions="I",
            memory_recent=True,
            memory_permanent_message="",
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            engine_uri="uri",
            run_max_new_tokens=10,
            run_skip_special_tokens=True,
            run_temperature=None,
            run_top_k=None,
            run_top_p=None,
            run_use_cache=None,
            run_cache_strategy=None,
            tool=None,
            tool_database_dsn="sqlite:///db.sqlite",
            backend="transformers",
            no_repl=False,
            quiet=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        @dataclass(frozen=True, kw_only=True, slots=True)
        class PatchedDatabaseToolSettings(DatabaseToolSettings):
            def items(self):
                return asdict(self).items()

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T", "I"]),
            patch.object(
                agent_cmds.Prompt, "ask", side_effect=["N", "", "uri"]
            ),
            patch.object(
                agent_cmds, "DatabaseToolSettings", PatchedDatabaseToolSettings
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertIn("[tool.database]", output)
        self.assertIn('dsn = "sqlite:///db.sqlite"', output)

    async def test_agent_init_run_options_output(self):
        args = Namespace(
            name="N",
            role="R",
            task="T",
            instructions=None,
            goal_instructions="I",
            memory_recent=True,
            memory_permanent_message="",
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            engine_uri="uri",
            run_max_new_tokens=10,
            run_skip_special_tokens=False,
            run_temperature=0.75,
            run_top_k=13,
            run_top_p=0.82,
            run_use_cache=True,
            run_cache_strategy="hybrid",
            run_openai_max_retries=0,
            run_openai_response_failed_retries=0,
            run_openai_response_failed_retry_delay_seconds=0.5,
            run_openai_timeout_seconds=30,
            run_reasoning_effort="xhigh",
            run_reasoning_summary="detailed",
            run_chat_add_generation_prompt=False,
            run_chat_enable_thinking=True,
            tool=["math.calculator"],
            tool_format="json",
            tool_browser_engine=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
            backend="transformers",
            no_repl=False,
            quiet=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T", "I"]),
            patch.object(
                agent_cmds.Prompt, "ask", side_effect=["N", "", "uri"]
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertIn("temperature = 0.75", output)
        self.assertIn("top_k = 13", output)
        self.assertIn("top_p = 0.82", output)
        self.assertIn("use_cache = true", output)
        self.assertIn('cache_strategy = "hybrid"', output)
        self.assertIn("openai_max_retries = 0", output)
        self.assertIn("openai_response_failed_retries = 0", output)
        self.assertIn(
            "openai_response_failed_retry_delay_seconds = 0.5",
            output,
        )
        self.assertIn("openai_timeout_seconds = 30", output)
        self.assertIn("[run.chat]", output)
        self.assertIn("add_generation_prompt = false", output)
        self.assertIn("enable_thinking = true", output)
        self.assertIn("[run.reasoning]", output)
        self.assertIn('effort = "xhigh"', output)
        self.assertIn('summary = "detailed"', output)
        self.assertIn("[tool]", output)
        self.assertIn('format = "json"', output)
        self.assertIn('"math.calculator"', output)

    async def test_agent_init_tool_recovery_formats_output(self):
        args = Namespace(
            name="N",
            role="R",
            task="T",
            instructions=None,
            goal_instructions=None,
            memory_recent=False,
            memory_permanent_message=None,
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            engine_uri="uri",
            run_max_new_tokens=None,
            run_skip_special_tokens=False,
            run_temperature=None,
            run_top_k=None,
            run_top_p=None,
            run_use_cache=None,
            run_cache_strategy=None,
            run_reasoning_effort=None,
            run_chat_add_generation_prompt=None,
            run_chat_enable_thinking=None,
            tool=None,
            tool_format=None,
            tool_recovery_format=["fenced", "tool_call_block"],
            tool_browser_engine=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
            backend="transformers",
            no_repl=False,
            quiet=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T"]),
            patch.object(agent_cmds.Prompt, "ask", side_effect=["N", "uri"]),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertIn("[tool]", output)
        self.assertIn("recovery_formats = [", output)
        self.assertIn('"fenced"', output)
        self.assertIn('"tool_call_block"', output)

    def test_agent_tool_name_policy_from_cli_args(self):
        args = Namespace(
            tool_name_policy="mapped",
            tool_name_prefix="tool_",
            tool_name_replacement="__",
            tool_name_collapse_replacement=False,
            tool_name_map=["math.calculator=calc"],
        )

        policy = agent_cmds._agent_tool_name_policy(args)

        assert policy is not None
        self.assertIs(policy.mode, ToolNamePolicyMode.MAPPED)
        self.assertEqual(policy.prefix, "tool_")
        self.assertEqual(policy.replacement, "__")
        self.assertFalse(policy.collapse_replacement)
        self.assertEqual(policy.map, {"math.calculator": "calc"})

    def test_agent_tool_name_policy_maps_with_sanitized_fallback(self):
        args = Namespace(
            tool_name_policy="sanitized",
            tool_name_prefix=None,
            tool_name_replacement="_",
            tool_name_collapse_replacement=True,
            tool_name_map=[
                "shell.pdfinfo=pdfinfo",
                "shell.tesseract=tesseract",
            ],
        )

        policy = agent_cmds._agent_tool_name_policy(args)

        assert policy is not None
        self.assertIs(policy.mode, ToolNamePolicyMode.SANITIZED)
        self.assertEqual(policy.replacement, "_")
        self.assertTrue(policy.collapse_replacement)
        self.assertEqual(
            policy.map,
            {
                "shell.pdfinfo": "pdfinfo",
                "shell.tesseract": "tesseract",
            },
        )

    def test_agent_tool_name_policy_allows_empty_mapped_prefix(self):
        args = Namespace(
            tool_name_policy="mapped",
            tool_name_prefix="",
            tool_name_replacement=None,
            tool_name_collapse_replacement=None,
            tool_name_map=["shell.pdfinfo=pdfinfo"],
        )

        policy = agent_cmds._agent_tool_name_policy(args)

        assert policy is not None
        self.assertIs(policy.mode, ToolNamePolicyMode.MAPPED)
        self.assertEqual(policy.prefix, "")
        self.assertEqual(policy.map, {"shell.pdfinfo": "pdfinfo"})

    def test_agent_tool_name_policy_rejects_invalid_cli_values(self):
        cases = (
            Namespace(
                tool_name_policy="encoded",
                tool_name_prefix="",
                tool_name_replacement=None,
                tool_name_collapse_replacement=None,
                tool_name_map=None,
            ),
            Namespace(
                tool_name_policy="sanitized",
                tool_name_prefix=None,
                tool_name_replacement="",
                tool_name_collapse_replacement=None,
                tool_name_map=None,
            ),
            Namespace(
                tool_name_policy="mapped",
                tool_name_prefix=None,
                tool_name_replacement=None,
                tool_name_collapse_replacement=None,
                tool_name_map=["math.calculator"],
            ),
        )

        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(AssertionError):
                    agent_cmds._agent_tool_name_policy(args)

    async def test_agent_init_tool_name_policy_output(self):
        args = self._agent_init_args(
            memory_recent=False,
            tool=["math.calculator"],
            tool_name_policy="sanitized",
            tool_name_prefix="tool_",
            tool_name_replacement="_",
            tool_name_collapse_replacement=True,
            tool_name_map=["math.calculator=calc"],
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T"]),
            patch.object(agent_cmds.Prompt, "ask", side_effect=["N", "uri"]),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertIn("[tool]", output)
        self.assertIn('"math.calculator"', output)
        self.assertIn("[tool.name_policy]", output)
        self.assertIn('mode = "sanitized"', output)
        self.assertIn('prefix = "tool_"', output)
        self.assertIn('replacement = "_"', output)
        self.assertIn("collapse_replacement = true", output)
        self.assertIn("[tool.name_policy.map]", output)
        self.assertIn('"math.calculator" = "calc"', output)

    async def test_agent_init_shell_tool_settings_output(self):
        args = Namespace(
            name="N",
            role="R",
            task="T",
            instructions=None,
            goal_instructions="I",
            memory_recent=True,
            memory_permanent_message="",
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            engine_uri="uri",
            run_max_new_tokens=10,
            run_skip_special_tokens=True,
            run_temperature=None,
            run_top_k=None,
            run_top_p=None,
            run_use_cache=None,
            run_cache_strategy=None,
            tool=["shell.rg"],
            tool_shell_workspace_root="workspace",
            tool_shell_materialized_input_files_dir="agent-input-files",
            tool_shell_input_file_manifest_enabled=False,
            tool_shell_input_file_manifest_message=(
                'Use "attached" paths from C:\\docs'
            ),
            tool_shell_input_file_manifest_path_message="Pass them to tools.",
            tool_shell_max_stdout_bytes=4096,
            tool_shell_allow_media_tools=True,
            tool_shell_allow_pipelines=True,
            tool_shell_max_pipeline_stages=3,
            tool_shell_max_pipeline_bytes=1024,
            tool_shell_max_intermediate_bytes=512,
            tool_shell_allowed_commands=("rg", "cat"),
            tool_shell_executable_search_paths=["/usr/bin", "/bin"],
            tool_shell_executable_paths=[("rg", "/usr/bin/rg")],
            backend="transformers",
            no_repl=False,
            quiet=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T", "I"]),
            patch.object(
                agent_cmds.Prompt, "ask", side_effect=["N", "", "uri"]
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertIn("[tool.shell]", output)
        self.assertIn('workspace_root = "workspace"', output)
        self.assertIn(
            'materialized_input_files_dir = "agent-input-files"',
            output,
        )
        self.assertIn("input_file_manifest_enabled = false", output)
        self.assertIn(
            'input_file_manifest_message = "Use \\"attached\\" paths '
            'from C:\\\\docs"',
            output,
        )
        self.assertIn(
            'input_file_manifest_path_message = "Pass them to tools."',
            output,
        )
        self.assertIn("max_stdout_bytes = 4096", output)
        self.assertIn("allow_media_tools = true", output)
        self.assertIn("allow_pipelines = true", output)
        self.assertIn("max_pipeline_stages = 3", output)
        self.assertIn("max_pipeline_bytes = 1024", output)
        self.assertIn("max_intermediate_bytes = 512", output)
        self.assertIn("allowed_commands = [", output)
        self.assertIn("executable_search_paths = [", output)
        self.assertIn('executable_paths = { rg = "/usr/bin/rg" }', output)
        self.assertIn('"rg"', output)
        self.assertIn('"cat"', output)
        parsed = tomllib.loads(output)
        self.assertEqual(
            parsed["tool"]["shell"],
            {
                "workspace_root": "workspace",
                "materialized_input_files_dir": "agent-input-files",
                "input_file_manifest_enabled": False,
                "input_file_manifest_message": (
                    'Use "attached" paths from C:\\docs'
                ),
                "input_file_manifest_path_message": "Pass them to tools.",
                "max_stdout_bytes": 4096,
                "allow_media_tools": True,
                "allow_pipelines": True,
                "max_pipeline_stages": 3,
                "max_pipeline_bytes": 1024,
                "max_intermediate_bytes": 512,
                "allowed_commands": ["rg", "cat"],
                "executable_search_paths": ["/usr/bin", "/bin"],
                "executable_paths": {"rg": "/usr/bin/rg"},
            },
        )

    async def test_agent_init_shell_git_settings_output(self):
        args = self._agent_init_args(
            memory_recent=False,
            tool=["shell.git_status", "shell.git_log"],
            tool_shell_git_workspace_root="/workspace",
            tool_shell_git_cwd="repo",
            tool_shell_git_capabilities=["read"],
            tool_shell_git_allowed_commands=["status", "log"],
            tool_shell_git_default_timeout_seconds=5.0,
            tool_shell_git_max_timeout_seconds=20.0,
            tool_shell_git_max_log_count=12,
            tool_shell_git_redact_author_emails=True,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T"]),
            patch.object(agent_cmds.Prompt, "ask", side_effect=["N", "uri"]),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertIn("[tool.shell]", output)
        self.assertIn("[tool.shell.git]", output)
        self.assertNotIn("git = {", output)
        parsed = tomllib.loads(output)
        self.assertEqual(
            parsed["tool"]["shell"]["git"],
            {
                "workspace_root": "/workspace",
                "cwd": "repo",
                "allowed_commands": ["status", "log"],
                "default_timeout_seconds": 5.0,
                "max_timeout_seconds": 20.0,
                "max_log_count": 12,
                "redact_author_emails": True,
            },
        )

    def test_shell_tool_template_settings_filters_defaults(self):
        settings = ShellToolSettings(
            max_head_lines=7,
            allowed_commands=("rg",),
            executable_paths={"rg": "/usr/bin/rg"},
            executable_search_paths=("/usr/bin",),
        )

        rendered = agent_cmds._shell_tool_template_settings(settings)

        self.assertFalse(hasattr(settings, "items"))
        self.assertEqual(
            rendered,
            {
                "max_head_lines": 7,
                "allowed_commands": ("rg",),
                "executable_paths": {"rg": "/usr/bin/rg"},
                "executable_search_paths": ("/usr/bin",),
            },
        )

    def test_shell_tool_template_settings_renders_single_mode_field(self):
        settings = ShellToolSettings(
            backend="sandbox",
            execution_mode="sandbox",
        )

        rendered = agent_cmds._shell_tool_template_settings(settings)

        self.assertEqual(rendered, {"backend": "sandbox"})

    def test_shell_tool_template_settings_renders_git_settings(self):
        settings = ShellToolSettings(
            git=ShellGitToolSettings(
                workspace_root="/workspace",
                cwd="repo",
                allowed_commands=("status", "diff"),
                max_log_count=12,
                redact_author_emails=True,
            ),
        )

        rendered = agent_cmds._shell_tool_template_settings(settings)

        self.assertEqual(
            rendered,
            {
                "git": {
                    "workspace_root": "/workspace",
                    "cwd": "repo",
                    "allowed_commands": ("status", "diff"),
                    "max_log_count": 12,
                    "redact_author_emails": True,
                },
            },
        )

    def test_shell_tool_template_settings_rejects_non_simple_sequences(self):
        self.assertFalse(agent_cmds._is_simple_string_sequence("rg"))
        self.assertFalse(agent_cmds._is_simple_string_sequence(1))
        self.assertFalse(agent_cmds._is_simple_string_sequence(("rg", 1)))
        self.assertFalse(agent_cmds._is_simple_string_mapping(("rg",)))
        self.assertFalse(agent_cmds._is_simple_string_mapping({"rg": 1}))

    def test_toml_template_value_formats_supported_values(self):
        self.assertEqual(agent_cmds._toml_template_value(True), "true")
        self.assertEqual(agent_cmds._toml_template_value(False), "false")
        self.assertEqual(agent_cmds._toml_template_value(7), "7")
        self.assertEqual(
            agent_cmds._toml_template_value('Use "quoted" paths'),
            '"Use \\"quoted\\" paths"',
        )
        non_bmp_text = "Files " + chr(0x1F600)
        self.assertEqual(
            tomllib.loads(
                "value = "
                + agent_cmds._toml_template_value(non_bmp_text)
                + "\n"
            ),
            {"value": non_bmp_text},
        )
        self.assertEqual(
            agent_cmds._toml_template_value(("rg", "cat")),
            '["rg", "cat"]',
        )
        self.assertEqual(
            agent_cmds._toml_template_value({"rg": "/usr/bin/rg"}),
            '{ rg = "/usr/bin/rg" }',
        )

    def test_toml_template_value_rejects_unsupported_values(self):
        with self.assertRaises(AssertionError):
            agent_cmds._toml_template_value(object())

    def test_agent_tool_settings_builds_container_runtime_from_cli(self):
        args = Namespace(
            tool_shell_backend="container",
            tool_container_backend="docker",
            tool_container_profile="workspace-readonly",
            tool_container_image="ghcr.io/example/tools@sha256:" + "2" * 64,
            tool_container_workspace_root=".",
            tool_container_pull_policy="never",
            tool_container_platform="linux/amd64",
            tool_container_cpu_count=1,
            tool_container_memory_bytes=268435456,
            tool_container_pids=64,
            tool_container_timeout_seconds=30,
            tool_container_network_mode="none",
            tool_container_review_mode="deny",
            tool_shell_container_profile="workspace-readonly",
            tool_shell_container_required=True,
        )

        settings = agent_cmds._agent_tool_settings(args)

        self.assertIsInstance(settings.shell, ShellToolSettings)
        self.assertEqual(settings.shell.backend, "container")
        self.assertIsNotNone(settings.container)
        assert settings.container is not None
        self.assertTrue(settings.container.rootful_authorized)
        effective = settings.container.effective_settings
        self.assertIsNotNone(effective)
        assert effective is not None
        self.assertTrue(effective.required)
        self.assertEqual(effective.profile_name, "workspace-readonly")
        assert effective.profile is not None
        self.assertEqual(effective.profile.resources.cpu_count, 1)
        self.assertEqual(effective.profile.resources.memory_bytes, 268435456)

    def test_agent_tool_settings_records_explicit_shell_fields(self):
        args = Namespace(
            tool_shell_allow_media_tools=True,
            tool_shell_max_head_lines=ShellToolSettings().max_head_lines,
        )

        settings = agent_cmds._agent_tool_settings(args)

        self.assertIsInstance(settings.shell, ShellToolSettings)
        self.assertEqual(
            settings.shell_explicit_fields,
            frozenset({"allow_media_tools", "max_head_lines"}),
        )

    def test_tool_settings_explicit_fields_tracks_mapping_inputs(self):
        prefixed_fields = (
            agent_cmds._tool_settings_explicit_fields_from_mapping(
                {"tool_shell_allow_media_tools": True},
                prefix="shell",
                settings_cls=ShellToolSettings,
            )
        )
        direct_fields = agent_cmds._tool_settings_explicit_fields_from_mapping(
            {"allow_media_tools": True},
            prefix="shell",
            settings_cls=ShellToolSettings,
        )

        self.assertEqual(prefixed_fields, frozenset({"allow_media_tools"}))
        self.assertEqual(direct_fields, frozenset({"allow_media_tools"}))

    def test_agent_tool_settings_builds_sandbox_runtime_from_cli(self):
        args = Namespace(
            tool_shell_backend="sandbox",
            tool_sandbox_backend="seatbelt",
            tool_sandbox_profile="host-tools",
            tool_sandbox_trusted_executables=["/bin/cat"],
            tool_sandbox_executable_search_roots=["/bin"],
            tool_sandbox_read_roots=["/workspace"],
            tool_sandbox_write_roots=["/workspace/out"],
            tool_sandbox_deny_roots=["/etc/ssh"],
            tool_sandbox_scratch_roots=["/tmp/avalan"],
            tool_sandbox_output_roots=["/workspace/out"],
            tool_sandbox_network_mode="none",
            tool_sandbox_network_egress=None,
            tool_sandbox_timeout_seconds=30,
            tool_sandbox_pids=16,
            tool_sandbox_max_stdout_bytes=4096,
            tool_sandbox_max_stderr_bytes=2048,
            tool_sandbox_max_artifact_bytes=None,
            tool_sandbox_allow_artifacts=None,
            tool_sandbox_child_processes="deny",
            tool_sandbox_inherited_fds="stdio",
            tool_shell_sandbox_profile="host-tools",
            tool_shell_sandbox_required=True,
        )

        settings = agent_cmds._agent_tool_settings(args)

        self.assertIsInstance(settings.shell, ShellToolSettings)
        self.assertEqual(settings.shell.backend, "sandbox")
        self.assertIsNotNone(settings.isolation)
        assert settings.isolation is not None
        self.assertIsInstance(
            settings.isolation.sandbox_backend,
            SeatbeltSandboxBackend,
        )
        effective = settings.isolation.effective_settings
        self.assertEqual(effective.mode, "sandbox")
        assert effective.sandbox is not None
        self.assertTrue(effective.sandbox.required)
        self.assertEqual(effective.sandbox.profile_name, "host-tools")
        self.assertEqual(
            effective.sandbox.profile.trusted_executables,
            ("/bin/cat",),
        )
        self.assertEqual(effective.sandbox.profile.resources.pids, 16)

    def test_agent_tool_settings_wires_bubblewrap_sandbox_backend(self):
        args = self._agent_init_args(
            tool_shell_backend="sandbox",
            tool_sandbox_backend="bubblewrap",
            tool_sandbox_profile="host-tools",
            tool_sandbox_trusted_executables=["/bin/cat"],
        )

        settings = agent_cmds._agent_tool_settings(args)

        self.assertIsInstance(settings.shell, ShellToolSettings)
        self.assertEqual(settings.shell.backend, "sandbox")
        self.assertIsNotNone(settings.isolation)
        assert settings.isolation is not None
        self.assertIsInstance(
            settings.isolation.sandbox_backend,
            BubblewrapSandboxBackend,
        )

    def test_agent_sandbox_backend_from_args_returns_none_without_backend(
        self,
    ):
        args = self._agent_init_args()

        self.assertIsNone(agent_cmds._agent_sandbox_backend_from_args(args))

    def test_agent_tool_settings_wires_apple_container_backend(self):
        args = Namespace(
            tool_shell_backend="container",
            tool_container_backend="apple-container",
            tool_container_profile="workspace-readonly",
            tool_container_image="ghcr.io/example/tools@sha256:" + "2" * 64,
            tool_container_workspace_root=".",
            tool_container_pull_policy="never",
            tool_container_platform="linux/arm64",
            tool_container_cpu_count=1,
            tool_container_memory_bytes=268435456,
            tool_container_pids=64,
            tool_container_timeout_seconds=30,
            tool_container_network_mode="none",
            tool_container_review_mode="deny",
            tool_shell_container_profile="workspace-readonly",
            tool_shell_container_required=True,
        )
        module = SimpleNamespace(
            AppleContainerBackend=_apple_container_backend_class()
        )

        with patch.object(
            agent_cmds,
            "import_module",
            return_value=module,
        ):
            settings = agent_cmds._agent_tool_settings(args)

        self.assertIsNotNone(settings.container)
        assert settings.container is not None
        self.assertIsInstance(
            settings.container.backend,
            ContainerFakeBackend,
        )
        self.assertEqual(
            settings.container.opt_in_backends,
            (ContainerBackend.APPLE_CONTAINER,),
        )

    def test_agent_tool_settings_missing_docker_backend_fails_closed(self):
        args = Namespace(
            tool_shell_backend="container",
            tool_container_backend="docker",
            tool_container_profile="workspace-readonly",
            tool_container_image="ghcr.io/example/tools@sha256:" + "2" * 64,
            tool_container_workspace_root=".",
            tool_container_pull_policy="never",
            tool_container_platform="linux/amd64",
            tool_container_cpu_count=1,
            tool_container_memory_bytes=268435456,
            tool_container_pids=64,
            tool_container_timeout_seconds=30,
            tool_container_network_mode="none",
            tool_container_review_mode="deny",
            tool_shell_container_profile="workspace-readonly",
            tool_shell_container_required=True,
        )

        def missing_module(_module_name: str) -> object:
            raise ModuleNotFoundError(_module_name)

        with patch.object(
            agent_cmds,
            "import_module",
            side_effect=missing_module,
        ):
            settings = agent_cmds._agent_tool_settings(args)

        self.assertIsNotNone(settings.container)
        assert settings.container is not None
        self.assertIsNone(settings.container.backend)
        self.assertEqual(settings.container.opt_in_backends, ())

    def test_agent_container_backend_unknown_value_returns_none(self):
        args = Namespace(tool_container_backend="unknown")

        self.assertIsNone(agent_cmds._agent_container_backend_from_args(args))

    def test_agent_tool_settings_missing_apple_backend_fails_closed(self):
        args = Namespace(
            tool_shell_backend="container",
            tool_container_backend="apple-container",
            tool_container_profile="workspace-readonly",
            tool_container_image="ghcr.io/example/tools@sha256:" + "2" * 64,
            tool_container_workspace_root=".",
            tool_container_pull_policy="never",
            tool_container_platform="linux/arm64",
            tool_container_cpu_count=1,
            tool_container_memory_bytes=268435456,
            tool_container_pids=64,
            tool_container_timeout_seconds=30,
            tool_container_network_mode="none",
            tool_container_review_mode="deny",
            tool_shell_container_profile="workspace-readonly",
            tool_shell_container_required=True,
        )

        def missing_module(_module_name: str) -> object:
            raise ModuleNotFoundError(_module_name)

        with patch.object(
            agent_cmds,
            "import_module",
            side_effect=missing_module,
        ):
            settings = agent_cmds._agent_tool_settings(args)

        self.assertIsNotNone(settings.container)
        assert settings.container is not None
        self.assertIsNone(settings.container.backend)
        self.assertEqual(
            settings.container.opt_in_backends,
            (ContainerBackend.APPLE_CONTAINER,),
        )

    def test_agent_tool_settings_rejects_shell_container_without_backend(
        self,
    ):
        args = Namespace(
            tool_shell_backend="local",
            tool_container_backend="docker",
            tool_container_profile="workspace-readonly",
            tool_container_image="ghcr.io/example/tools@sha256:" + "2" * 64,
            tool_container_workspace_root=".",
            tool_container_pull_policy="never",
            tool_container_platform="linux/amd64",
            tool_container_cpu_count=1,
            tool_container_memory_bytes=268435456,
            tool_container_pids=64,
            tool_container_timeout_seconds=30,
            tool_container_network_mode="none",
            tool_container_review_mode="deny",
            tool_shell_container_profile="workspace-readonly",
            tool_shell_container_required=True,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "tool.container requires tool.shell backend container",
        ):
            agent_cmds._agent_tool_settings(args)

    def test_agent_tool_settings_rejects_mixed_sandbox_container_policy(
        self,
    ):
        args = Namespace(
            tool_shell_backend="sandbox",
            tool_container_backend="docker",
            tool_container_profile="workspace-readonly",
            tool_container_image="ghcr.io/example/tools@sha256:" + "2" * 64,
            tool_container_workspace_root=".",
            tool_container_pull_policy="never",
            tool_container_platform="linux/amd64",
            tool_container_cpu_count=1,
            tool_container_memory_bytes=268435456,
            tool_container_pids=64,
            tool_container_timeout_seconds=30,
            tool_container_network_mode="none",
            tool_container_review_mode="deny",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "tool.container requires tool.shell backend container",
        ):
            agent_cmds._agent_tool_settings(args)

    def test_agent_tool_settings_rejects_container_policy_without_shell_mode(
        self,
    ):
        args = Namespace(
            tool_container_backend="docker",
            tool_container_profile="workspace-readonly",
            tool_container_image="ghcr.io/example/tools@sha256:" + "2" * 64,
            tool_container_workspace_root=".",
            tool_container_pull_policy="never",
            tool_container_platform="linux/amd64",
            tool_container_cpu_count=1,
            tool_container_memory_bytes=268435456,
            tool_container_pids=64,
            tool_container_timeout_seconds=30,
            tool_container_network_mode="none",
            tool_container_review_mode="deny",
        )

        with self.assertRaisesRegex(
            AssertionError,
            "tool.container requires tool.shell backend container",
        ):
            agent_cmds._agent_tool_settings(args)

    def test_agent_tool_settings_rejects_sandbox_policy_without_shell_mode(
        self,
    ):
        args = Namespace(
            tool_sandbox_backend="seatbelt",
            tool_sandbox_profile="host-tools",
            tool_sandbox_trusted_executables=["/bin/cat"],
            tool_sandbox_executable_search_roots=["/bin"],
            tool_sandbox_read_roots=["/workspace"],
            tool_sandbox_write_roots=None,
            tool_sandbox_deny_roots=None,
            tool_sandbox_scratch_roots=None,
            tool_sandbox_output_roots=None,
            tool_sandbox_network_mode="none",
            tool_sandbox_network_egress=None,
            tool_sandbox_timeout_seconds=None,
            tool_sandbox_pids=None,
            tool_sandbox_max_stdout_bytes=None,
            tool_sandbox_max_stderr_bytes=None,
            tool_sandbox_max_artifact_bytes=None,
            tool_sandbox_allow_artifacts=None,
            tool_sandbox_child_processes=None,
            tool_sandbox_inherited_fds=None,
        )

        with self.assertRaisesRegex(
            AssertionError,
            "tool.sandbox requires tool.shell backend sandbox",
        ):
            agent_cmds._agent_tool_settings(args)

    async def test_agent_init_container_settings_output(self):
        image = "ghcr.io/example/tools@sha256:" + "3" * 64
        args = self._agent_init_args(
            goal_instructions=None,
            tool=["shell.cat"],
            tool_shell_backend="container",
            tool_container_backend="docker",
            tool_container_profile="workspace-readonly",
            tool_container_image=image,
            tool_container_workspace_root=".",
            tool_container_pull_policy="never",
            tool_container_platform="linux/amd64",
            tool_container_cpu_count=1,
            tool_container_memory_bytes=None,
            tool_container_pids=None,
            tool_container_timeout_seconds=None,
            tool_container_network_mode="none",
            tool_container_review_mode="deny",
            tool_shell_container_profile="workspace-readonly",
            tool_shell_container_required=True,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T"]),
            patch.object(agent_cmds.Prompt, "ask", side_effect=["N", "uri"]),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        parsed = tomllib.loads(output)
        self.assertEqual(parsed["tool"]["shell"]["backend"], "container")
        self.assertNotIn("execution_mode", parsed["tool"]["shell"])
        self.assertEqual(parsed["tool"]["container"]["backend"], "docker")
        self.assertEqual(
            parsed["tool"]["container"]["profiles"]["workspace-readonly"][
                "image"
            ],
            image,
        )
        self.assertEqual(
            parsed["tool"]["container"]["profiles"]["workspace-readonly"][
                "resources"
            ]["cpu_count"],
            1,
        )
        self.assertNotIn(
            "workspace_root",
            parsed["tool"]["container"]["profiles"]["workspace-readonly"],
        )
        self.assertNotIn(
            "pull_policy",
            parsed["tool"]["container"]["profiles"]["workspace-readonly"],
        )
        self.assertNotIn(
            "network",
            parsed["tool"]["container"]["profiles"]["workspace-readonly"],
        )
        self.assertEqual(
            parsed["tool"]["shell"]["container"],
            {"profile": "workspace-readonly", "required": True},
        )

    async def test_agent_init_container_settings_accepts_execution_mode_alias(
        self,
    ):
        image = "ghcr.io/example/tools@sha256:" + "3" * 64
        args = self._agent_init_args(
            goal_instructions=None,
            tool=["shell.cat"],
            tool_shell_execution_mode="container",
            tool_container_backend="docker",
            tool_container_profile="workspace-readonly",
            tool_container_image=image,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T"]),
            patch.object(agent_cmds.Prompt, "ask", side_effect=["N", "uri"]),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        parsed = tomllib.loads(output)
        self.assertEqual(parsed["tool"]["shell"]["backend"], "container")
        self.assertNotIn("execution_mode", parsed["tool"]["shell"])
        self.assertEqual(parsed["tool"]["container"]["backend"], "docker")
        self.assertEqual(
            parsed["tool"]["container"]["profiles"]["workspace-readonly"][
                "image"
            ],
            image,
        )

    async def test_agent_init_sandbox_settings_output(self):
        args = self._agent_init_args(
            goal_instructions=None,
            tool=["shell.cat"],
            tool_shell_backend="sandbox",
            tool_sandbox_backend="seatbelt",
            tool_sandbox_profile="host-tools",
            tool_sandbox_trusted_executables=["/bin/cat"],
            tool_sandbox_executable_search_roots=["/bin"],
            tool_sandbox_read_roots=["/workspace"],
            tool_sandbox_write_roots=None,
            tool_sandbox_deny_roots=["/etc/ssh"],
            tool_sandbox_scratch_roots=["/tmp/avalan"],
            tool_sandbox_output_roots=None,
            tool_sandbox_network_mode="none",
            tool_sandbox_network_egress=None,
            tool_sandbox_timeout_seconds=30,
            tool_sandbox_pids=16,
            tool_sandbox_max_stdout_bytes=4096,
            tool_sandbox_max_stderr_bytes=None,
            tool_sandbox_max_artifact_bytes=None,
            tool_sandbox_allow_artifacts=None,
            tool_sandbox_child_processes="deny",
            tool_sandbox_inherited_fds="stdio",
            tool_shell_sandbox_profile="host-tools",
            tool_shell_sandbox_required=True,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T"]),
            patch.object(agent_cmds.Prompt, "ask", side_effect=["N", "uri"]),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        parsed = tomllib.loads(output)
        self.assertEqual(parsed["tool"]["shell"]["backend"], "sandbox")
        self.assertNotIn("execution_mode", parsed["tool"]["shell"])
        self.assertEqual(parsed["tool"]["sandbox"]["backend"], "seatbelt")
        self.assertEqual(
            parsed["tool"]["sandbox"]["profiles"]["host-tools"][
                "trusted_executables"
            ],
            ["/bin/cat"],
        )
        self.assertEqual(
            parsed["tool"]["sandbox"]["profiles"]["host-tools"]["resources"][
                "pids"
            ],
            16,
        )
        self.assertNotIn(
            "network",
            parsed["tool"]["sandbox"]["profiles"]["host-tools"],
        )
        self.assertNotIn(
            "child_processes",
            parsed["tool"]["sandbox"]["profiles"]["host-tools"],
        )
        self.assertNotIn(
            "inherited_fds",
            parsed["tool"]["sandbox"]["profiles"]["host-tools"],
        )
        self.assertEqual(
            parsed["tool"]["shell"]["sandbox"],
            {"profile": "host-tools", "required": True},
        )

    async def test_agent_init_sandbox_settings_accepts_execution_mode_alias(
        self,
    ):
        args = self._agent_init_args(
            goal_instructions=None,
            tool=["shell.cat"],
            tool_shell_execution_mode="sandbox",
            tool_sandbox_backend="seatbelt",
            tool_sandbox_profile="host-tools",
            tool_sandbox_trusted_executables=["/bin/cat"],
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T"]),
            patch.object(agent_cmds.Prompt, "ask", side_effect=["N", "uri"]),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        parsed = tomllib.loads(output)
        self.assertEqual(parsed["tool"]["shell"]["backend"], "sandbox")
        self.assertNotIn("execution_mode", parsed["tool"]["shell"])
        self.assertEqual(parsed["tool"]["sandbox"]["backend"], "seatbelt")
        self.assertEqual(
            parsed["tool"]["sandbox"]["default_profile"],
            "host-tools",
        )
        self.assertEqual(
            parsed["tool"]["sandbox"]["profiles"]["host-tools"][
                "trusted_executables"
            ],
            ["/bin/cat"],
        )

    async def test_agent_init_sandbox_settings_renders_non_default_policies(
        self,
    ):
        args = self._agent_init_args(
            goal_instructions=None,
            tool=["shell.cat"],
            tool_shell_backend="sandbox",
            tool_sandbox_backend="seatbelt",
            tool_sandbox_profile="host-tools",
            tool_sandbox_trusted_executables=["/bin/cat"],
            tool_sandbox_network_mode="allowlist",
            tool_sandbox_network_egress=["example.test"],
            tool_sandbox_max_stderr_bytes=8192,
            tool_sandbox_max_artifact_bytes=4096,
            tool_sandbox_allow_artifacts=True,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T"]),
            patch.object(agent_cmds.Prompt, "ask", side_effect=["N", "uri"]),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        parsed = tomllib.loads(output)
        profile = parsed["tool"]["sandbox"]["profiles"]["host-tools"]
        self.assertEqual(
            profile["network"],
            {
                "mode": "allowlist",
                "egress_allowlist": ["example.test"],
            },
        )
        self.assertEqual(
            profile["output"],
            {
                "max_stderr_bytes": 8192,
                "max_artifact_bytes": 4096,
                "allow_artifacts": True,
            },
        )

    async def test_agent_init_sandbox_policy_validates_before_render(self):
        args = self._agent_init_args(
            tool_shell_backend="sandbox",
            tool_sandbox_backend="seatbelt",
            tool_sandbox_profile="host-tools",
            tool_sandbox_trusted_executables=["/bin/cat"],
            tool_sandbox_network_egress=["example.test"],
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "network none cannot define egress",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()

    async def test_agent_init_container_policy_requires_shell_backend(self):
        args = self._agent_init_args(
            tool_container_backend="docker",
            tool_container_profile="workspace-readonly",
            tool_container_image="ghcr.io/example/tools@sha256:" + "3" * 64,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "tool.container requires tool.shell backend container",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()

    async def test_agent_init_sandbox_policy_requires_shell_backend(self):
        args = self._agent_init_args(
            tool_sandbox_backend="seatbelt",
            tool_sandbox_profile="host-tools",
            tool_sandbox_trusted_executables=["/bin/cat"],
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "tool.sandbox requires tool.shell backend sandbox",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()

    async def test_agent_init_container_backend_requires_image(self):
        args = self._agent_init_args(
            tool_shell_backend="container",
            tool_container_backend="docker",
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "container image is required",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()

    async def test_agent_init_shell_container_profile_requires_backend(self):
        args = self._agent_init_args(
            tool_shell_container_profile="workspace-readonly",
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "tool.shell.container requires tool.shell backend container",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()

    async def test_agent_init_shell_sandbox_profile_requires_backend(self):
        args = self._agent_init_args(
            tool_shell_sandbox_profile="host-tools",
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "tool.shell.sandbox requires tool.shell backend sandbox",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()

    async def test_agent_init_shell_container_requires_image(self):
        args = self._agent_init_args(
            tool_shell_backend="container",
            tool_container_backend="docker",
            tool_shell_container_required=True,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "container image is required",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()

    async def test_agent_init_container_profile_requires_backend(self):
        args = self._agent_init_args(
            tool_shell_backend="container",
            tool_container_profile="workspace-readonly",
            tool_container_cpu_count=1,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "container backend is required",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()

    async def test_agent_init_rejects_unsupported_container_backend(self):
        args = self._agent_init_args(
            tool_shell_backend="container",
            tool_container_backend="none",
            tool_container_profile="ignored",
            tool_container_cpu_count=1,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        env = MagicMock()
        template = MagicMock()
        env.get_template.return_value = template

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
            self.assertRaisesRegex(
                AssertionError,
                "container backend is unsupported",
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_not_called()


class CliAgentRunTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = Namespace(
            specifications_file="spec.toml",
            use_sync_generator=False,
            display_tokens=0,
            stats=False,
            id="aid",
            participant="pid",
            session="sid",
            no_session=False,
            skip_load_recent_messages=False,
            load_recent_messages_limit=1,
            no_repl=False,
            quiet=False,
            skip_hub_access_check=False,
            conversation=False,
            watch=False,
            tty=None,
            tool_events=2,
            tool=None,
            tool_format=None,
            tool_choice=None,
            run_max_new_tokens=100,
            run_skip_special_tokens=False,
            engine_uri=None,
            name=None,
            role=None,
            task=None,
            instructions=None,
            memory_recent=None,
            memory_permanent_message=None,
            memory_permanent=None,
            memory_engine_model_id=agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            tool_browser_engine=None,
            tool_browser_debug=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            tools_confirm=False,
            backend="transformers",
        )
        self.console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        self.console.status.return_value = status_cm
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme.icons = {"user_input": ">", "agent_output": "<"}
        self.theme.get_spinner.return_value = "sp"
        self.theme.agent.return_value = "agent_panel"
        self.theme.recent_messages.return_value = "recent_panel"
        self.hub = MagicMock()
        self.hub.can_access.return_value = True
        self.hub.model.side_effect = lambda m: f"mdl-{m}"
        self.logger = MagicMock()

        self.orch = AsyncMock()
        self.orch.engine_agent = True
        self.orch.engine = MagicMock(model_id="m")
        self.orch.model_ids = ["m"]
        self.orch.event_manager.add_listener = MagicMock()
        self.orch.memory = MagicMock()
        self.orch.memory.has_recent_message = False
        self.orch.memory.has_permanent_message = False
        self.orch.memory.recent_message = MagicMock(
            is_empty=True, size=0, data=[]
        )
        self.orch.memory.continue_session = AsyncMock()
        self.orch.memory.start_session = AsyncMock()

        self.dummy_stack = AsyncMock()
        self.dummy_stack.__aenter__.return_value = self.dummy_stack
        self.dummy_stack.__aexit__.return_value = False
        self.dummy_stack.enter_async_context = AsyncMock(
            return_value=self.orch
        )

    def _callback_stack(self):
        class CallbackStack:
            def __init__(self, orchestrator):
                self.callbacks = []
                self.orchestrator = orchestrator
                self.enter_async_context = AsyncMock(return_value=orchestrator)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_value, traceback):
                _ = (exc_type, exc_value, traceback)
                for callback, args, kwargs in reversed(self.callbacks):
                    callback(*args, **kwargs)
                return False

            def callback(self, callback, *args, **kwargs):
                self.callbacks.append((callback, args, kwargs))
                return callback

        return CallbackStack(self.orch)

    async def test_returns_when_no_input(self):
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.assert_not_awaited()
        self.orch.memory.continue_session.assert_awaited_once_with(
            session_id="sid",
            load_recent_messages=True,
            load_recent_messages_limit=1,
        )
        self.console.print.assert_any_call("agent_panel")
        self.assertEqual(len(self.console.print.call_args_list), 1)

    async def test_run_accepts_native_pdf_input_without_prompt(self):
        class DummyOrchestratorResponse:
            pass

        with NamedTemporaryFile(suffix=".pdf") as tmp:
            tmp.write(b"%PDF-1.7")
            tmp.flush()
            self.args.input_file = [tmp.name]
            self.orch.return_value = DummyOrchestratorResponse()

            with (
                patch.object(agent_cmds, "get_input", return_value=None),
                patch.object(
                    agent_cmds,
                    "AsyncExitStack",
                    return_value=self.dummy_stack,
                ),
                patch.object(
                    agent_cmds.OrchestratorLoader,
                    "from_file",
                    new=AsyncMock(return_value=self.orch),
                ),
                patch.object(
                    agent_cmds, "token_generation", new_callable=AsyncMock
                ) as tg,
                patch.object(
                    agent_cmds,
                    "OrchestratorResponse",
                    DummyOrchestratorResponse,
                ),
            ):
                await agent_cmds.agent_run(
                    self.args,
                    self.console,
                    self.theme,
                    self.hub,
                    self.logger,
                    1,
                )

        self.orch.assert_awaited_once()
        request_input = self.orch.await_args.args[0]
        self.assertIsInstance(request_input, Message)
        self.assertEqual(request_input.role, MessageRole.USER)
        assert isinstance(request_input.content, list)
        self.assertEqual(
            request_input.content,
            [
                MessageContentFile(
                    type="file",
                    file={
                        "file_data": b64encode(b"%PDF-1.7").decode("ascii"),
                        "filename": Path(tmp.name).name,
                        "local_path": str(Path(tmp.name).resolve()),
                        "mime_type": "application/pdf",
                    },
                )
            ],
        )
        tg.assert_awaited_once()
        self.assertEqual(tg.await_args.kwargs["input_string"], "")

    async def test_run_rejects_missing_native_input_file(self):
        self.args.input_file = ["/tmp/avalan-missing-agent-input.pdf"]

        with (
            patch.object(agent_cmds, "get_input", return_value="Summarize"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg,
        ):
            with self.assertRaisesRegex(
                AssertionError, "Input file not found"
            ):
                await agent_cmds.agent_run(
                    self.args,
                    self.console,
                    self.theme,
                    self.hub,
                    self.logger,
                    1,
                )

        self.orch.assert_not_awaited()
        tg.assert_not_awaited()

    async def test_run_from_file_forwards_shell_settings(self):
        self.args.tool_shell_max_stdout_bytes = 2048
        self.args.tool_shell_allow_pipelines = True
        self.args.tool_shell_max_pipeline_stages = 3
        self.args.tool_shell_max_pipeline_bytes = 1024
        self.args.tool_shell_max_intermediate_bytes = 512

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ) as lf,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        lf.assert_awaited_once()
        self.assertIs(
            lf.call_args.kwargs["event_manager_mode"],
            EventManagerMode.CLI,
        )
        tool_settings = lf.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertEqual(tool_settings.shell.max_stdout_bytes, 2048)
        self.assertTrue(tool_settings.shell.allow_pipelines)
        self.assertEqual(tool_settings.shell.max_pipeline_stages, 3)
        self.assertEqual(tool_settings.shell.max_pipeline_bytes, 1024)
        self.assertEqual(tool_settings.shell.max_intermediate_bytes, 512)

    async def test_no_session_option_skips_session(self):
        self.args.no_session = True
        self.args.session = uuid4()
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.memory.continue_session.assert_not_awaited()
        self.orch.memory.start_session.assert_not_awaited()

    async def test_start_session_when_no_session_id(self):
        self.args.session = None
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.memory.start_session.assert_awaited_once_with()
        self.orch.memory.continue_session.assert_not_awaited()

    async def test_continue_session_skip_load_recent_messages(self):
        self.args.skip_load_recent_messages = True
        self.args.load_recent_messages_limit = 5
        session_id = uuid4()
        self.args.session = session_id

        permanent = AsyncMock(spec=PermanentMessageMemory)
        permanent.get_recent_messages = AsyncMock()
        recent = RecentMessageMemory()
        manager = MemoryManager(
            agent_id=uuid4(),
            participant_id=uuid4(),
            permanent_message_memory=permanent,
            recent_message_memory=recent,
            text_partitioner=AsyncMock(),
            logger=self.logger,
        )
        manager.continue_session = AsyncMock(wraps=manager.continue_session)
        self.orch.memory = manager

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        manager.continue_session.assert_awaited_once_with(
            session_id=session_id,
            load_recent_messages=False,
            load_recent_messages_limit=5,
        )
        permanent.get_recent_messages.assert_not_awaited()
        self.assertEqual(manager.recent_messages, [])

    async def test_continue_session_with_limit_loads_recent_messages(self):
        self.args.load_recent_messages_limit = 3
        session_id = uuid4()
        self.args.session = session_id

        for existing in (False, True):
            permanent = AsyncMock(spec=PermanentMessageMemory)
            recent = RecentMessageMemory()
            manager = MemoryManager(
                agent_id=uuid4(),
                participant_id=uuid4(),
                permanent_message_memory=permanent,
                recent_message_memory=recent,
                text_partitioner=AsyncMock(),
                logger=self.logger,
            )
            manager.continue_session = AsyncMock(
                wraps=manager.continue_session
            )
            if existing:
                msg = EngineMessage(
                    agent_id=manager._agent_id,
                    model_id="m",
                    message=Message(role=MessageRole.USER, content="x"),
                )
                messages = [msg]
            else:
                messages = []
            permanent.get_recent_messages = AsyncMock(return_value=messages)
            self.orch.memory = manager
            self.console.print.reset_mock()

            with (
                patch.object(agent_cmds, "get_input", return_value=None),
                patch.object(
                    agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
                ),
                patch.object(
                    agent_cmds.OrchestratorLoader,
                    "from_file",
                    new=AsyncMock(return_value=self.orch),
                ),
                patch.object(
                    agent_cmds, "token_generation", new_callable=AsyncMock
                ),
            ):
                await agent_cmds.agent_run(
                    self.args,
                    self.console,
                    self.theme,
                    self.hub,
                    self.logger,
                    1,
                )

            manager.continue_session.assert_awaited_once_with(
                session_id=session_id,
                load_recent_messages=True,
                load_recent_messages_limit=3,
            )
            permanent.get_recent_messages.assert_awaited_once_with(
                participant_id=manager.participant_id,
                session_id=session_id,
                limit=3,
            )
            self.assertEqual(manager.recent_messages, messages)
            calls = [c.args[0] for c in self.console.print.call_args_list]
            self.assertIn("agent_panel", calls)
            if messages:
                self.assertIn("recent_panel", calls)
            else:
                self.assertNotIn("recent_panel", calls)

    async def test_continue_session_without_limit_loads_all_messages(self):
        self.args.load_recent_messages_limit = None
        session_id = uuid4()
        self.args.session = session_id

        for existing in (False, True):
            permanent = AsyncMock(spec=PermanentMessageMemory)
            recent = RecentMessageMemory()
            manager = MemoryManager(
                agent_id=uuid4(),
                participant_id=uuid4(),
                permanent_message_memory=permanent,
                recent_message_memory=recent,
                text_partitioner=AsyncMock(),
                logger=self.logger,
            )
            manager.continue_session = AsyncMock(
                wraps=manager.continue_session
            )
            if existing:
                msg = EngineMessage(
                    agent_id=manager._agent_id,
                    model_id="m",
                    message=Message(role=MessageRole.USER, content="x"),
                )
                messages = [msg]
            else:
                messages = []
            permanent.get_recent_messages = AsyncMock(return_value=messages)
            self.orch.memory = manager
            self.console.print.reset_mock()

            with (
                patch.object(agent_cmds, "get_input", return_value=None),
                patch.object(
                    agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
                ),
                patch.object(
                    agent_cmds.OrchestratorLoader,
                    "from_file",
                    new=AsyncMock(return_value=self.orch),
                ),
                patch.object(
                    agent_cmds, "token_generation", new_callable=AsyncMock
                ),
            ):
                await agent_cmds.agent_run(
                    self.args,
                    self.console,
                    self.theme,
                    self.hub,
                    self.logger,
                    1,
                )

            manager.continue_session.assert_awaited_once_with(
                session_id=session_id,
                load_recent_messages=True,
                load_recent_messages_limit=None,
            )
            permanent.get_recent_messages.assert_awaited_once_with(
                participant_id=manager.participant_id,
                session_id=session_id,
                limit=None,
            )
            self.assertEqual(manager.recent_messages, messages)
            calls = [c.args[0] for c in self.console.print.call_args_list]
            self.assertIn("agent_panel", calls)
            if messages:
                self.assertIn("recent_panel", calls)
            else:
                self.assertNotIn("recent_panel", calls)

    async def test_run_with_text_response(self):
        class DummyResponse:
            def __aiter__(self_inner):
                async def gen():
                    yield "t"

                return gen()

        class DummyOrchestratorResponse:
            def __aiter__(self_inner):
                async def gen():
                    yield DummyResponse()

                return gen()

        with (
            patch("avalan.cli.has_input", return_value=True),
            patch("avalan.cli.stdin", StringIO("hi\n")),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds,
                "OrchestratorResponse",
                DummyOrchestratorResponse,
            ),
        ):
            self.orch.return_value = DummyOrchestratorResponse()
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.assert_awaited_once_with(
            "hi", use_async_generator=True, tool_confirm=None
        )
        tg_patch.assert_awaited_once()
        tg_kwargs = tg_patch.await_args.kwargs
        display_config = tg_kwargs["display_config"]
        self.assertIsInstance(display_config, CliStreamDisplayConfig)
        self.assertFalse(display_config.show_stats)
        self.assertFalse(tg_kwargs["with_stats"])
        self.assertEqual(tg_kwargs["tool_events_limit"], 2)
        self.orch.memory.continue_session.assert_awaited()
        self.console.print.assert_any_call("agent_panel")
        self.console.print.assert_any_call("< ", end="")

    async def test_run_non_interactive_uses_stderr_diagnostics(self):
        self.console.is_terminal = False
        self.args.stats = True
        self.args.display_tools = True
        self.args.display_events = True
        self.args.record = True
        self.orch.memory.has_recent_message = True
        self.orch.memory.recent_message.is_empty = False
        self.orch.memory.recent_message.data = ["m"]

        class DummyOrchestratorResponse:
            pass

        with (
            patch.object(
                agent_cmds, "get_input", return_value="hi"
            ) as get_input_patch,
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds,
                "OrchestratorResponse",
                DummyOrchestratorResponse,
            ),
        ):
            self.orch.return_value = DummyOrchestratorResponse()
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        tg_patch.assert_awaited_once()
        display_config = tg_patch.await_args.kwargs["display_config"]
        self.assertTrue(display_config.answer_stdout_only)
        self.assertEqual(display_config.diagnostic_channel, "stderr")
        self.assertTrue(display_config.show_stats)
        self.assertTrue(display_config.show_tools)
        self.assertTrue(display_config.show_events)
        get_input_patch.assert_called_once()
        self.assertTrue(get_input_patch.call_args.kwargs["is_quiet"])
        self.assertTrue(get_input_patch.call_args.kwargs["echo_stdin"])
        self.console.status.assert_not_called()
        self.theme.agent.assert_not_called()
        self.theme.recent_messages.assert_not_called()
        self.console.print.assert_not_called()

    async def test_run_with_tool_events(self):
        class DummyOrchestratorResponse:
            def __aiter__(self_inner):
                async def gen():
                    yield agent_cmds.Event(
                        type=agent_cmds.EventType.TOOL_EXECUTE,
                        payload={"call": SimpleNamespace(name="calc")},
                    )
                    yield agent_cmds.Event(
                        type=agent_cmds.EventType.TOOL_RESULT,
                        payload={
                            "result": SimpleNamespace(
                                name="calc",
                                result="2",
                            )
                        },
                    )

                return gen()

        status_loading = MagicMock()
        status_loading.__enter__.return_value = None
        status_loading.__exit__.return_value = False
        status_tool = MagicMock()
        status_tool.__enter__.return_value = None
        status_tool.__exit__.return_value = False

        self.console.status.side_effect = [status_loading, status_tool]

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds,
                "OrchestratorResponse",
                DummyOrchestratorResponse,
            ),
        ):
            self.orch.return_value = DummyOrchestratorResponse()
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.assert_awaited_once_with(
            "hi", use_async_generator=True, tool_confirm=None
        )
        tg_patch.assert_awaited_once()

        self.assertEqual(len(self.console.status.call_args_list), 1)
        self.console.status.assert_called_with(
            "Loading agent...",
            spinner=self.theme.get_spinner.return_value,
            refresh_per_second=1,
        )

    async def test_run_display_tools_without_stats_passes_display_config(self):
        self.args.display_tools = True
        self.args.stats = False

        class DummyOrchestratorResponse:
            pass

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds,
                "OrchestratorResponse",
                DummyOrchestratorResponse,
            ),
        ):
            self.orch.return_value = DummyOrchestratorResponse()
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        display_config = tg_patch.await_args.kwargs["display_config"]
        self.assertTrue(display_config.show_tools)
        self.assertFalse(display_config.show_stats)
        self.assertFalse(tg_patch.await_args.kwargs["with_stats"])

    async def test_run_display_events_without_stats_passes_display_config(
        self,
    ) -> None:
        self.args.display_events = True
        self.args.stats = False

        class DummyOrchestratorResponse:
            pass

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds,
                "OrchestratorResponse",
                DummyOrchestratorResponse,
            ),
        ):
            self.orch.return_value = DummyOrchestratorResponse()
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        display_config = tg_patch.await_args.kwargs["display_config"]
        self.assertTrue(display_config.show_events)
        self.assertFalse(display_config.show_stats)
        self.assertFalse(tg_patch.await_args.kwargs["with_stats"])

    async def test_run_from_settings(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.run_skip_special_tokens = True
        self.args.tool_recovery_format = [
            ToolCallRecoveryFormat.FENCED.value,
            ToolCallRecoveryFormat.TOOL_CALL_BLOCK.value,
        ]

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ) as ff_patch,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        settings = fs_patch.call_args.args[0]
        self.assertIsNone(settings.tools)
        self.assertTrue(settings.call_options["skip_special_tokens"])
        self.assertIs(
            fs_patch.call_args.kwargs["event_manager_mode"],
            EventManagerMode.CLI,
        )
        tool_settings = fs_patch.call_args.kwargs["tool_settings"]
        self.assertIsNone(tool_settings.browser)
        self.assertIsNone(tool_settings.database)
        self.assertIsNone(tool_settings.graph)
        self.assertEqual(
            fs_patch.call_args.kwargs["tool_recovery_formats"],
            [
                ToolCallRecoveryFormat.FENCED,
                ToolCallRecoveryFormat.TOOL_CALL_BLOCK,
            ],
        )
        ff_patch.assert_not_called()

    async def test_run_from_settings_shell_settings_preserve_default_tools(
        self,
    ):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.tool_shell_max_stdout_bytes = 2048
        self.args.tool_shell_allow_pipelines = True
        self.args.tool_shell_max_pipeline_stages = 3
        self.args.tool_shell_max_pipeline_bytes = 1024
        self.args.tool_shell_max_intermediate_bytes = 512

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ) as ff_patch,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        settings = fs_patch.call_args.args[0]
        self.assertIsNone(settings.tools)
        self.assertIs(
            fs_patch.call_args.kwargs["event_manager_mode"],
            EventManagerMode.CLI,
        )
        tool_settings = fs_patch.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertEqual(tool_settings.shell.max_stdout_bytes, 2048)
        self.assertTrue(tool_settings.shell.allow_pipelines)
        self.assertEqual(tool_settings.shell.max_pipeline_stages, 3)
        self.assertEqual(tool_settings.shell.max_pipeline_bytes, 1024)
        self.assertEqual(tool_settings.shell.max_intermediate_bytes, 512)
        self.assertIsNone(tool_settings.browser)
        self.assertIsNone(tool_settings.database)
        self.assertIsNone(tool_settings.graph)
        ff_patch.assert_not_called()

    async def test_run_from_settings_shell_explicit_tool_is_preserved(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.tool = ["shell.cat"]
        self.args.tool_shell_max_stdout_bytes = 2048

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ) as ff_patch,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        settings = fs_patch.call_args.args[0]
        self.assertEqual(settings.tools, ["shell.cat"])
        tool_settings = fs_patch.call_args.kwargs["tool_settings"]
        self.assertIsInstance(tool_settings.shell, ShellToolSettings)
        self.assertEqual(tool_settings.shell.max_stdout_bytes, 2048)
        ff_patch.assert_not_called()

    async def test_run_sets_hidden_states(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.output_hidden_states = True

        orch_settings = OrchestratorSettings(
            agent_id=uuid4(),
            orchestrator_type=None,
            agent_config={"role": "assistant"},
            uri="engine",
            engine_config={"output_hidden_states": True},
            call_options={},
            template_vars=None,
            memory_permanent_message=None,
            permanent_memory=None,
            memory_recent=True,
            sentence_model_id=agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            sentence_model_engine_config=None,
            sentence_model_max_tokens=500,
            sentence_model_overlap_size=125,
            sentence_model_window_size=250,
            json_config=None,
            tools=[],
            log_events=True,
        )

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader, "from_file", new=AsyncMock()
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
            patch.object(
                agent_cmds,
                "get_orchestrator_settings",
                return_value=orch_settings,
            ) as gos_patch,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        gos_patch.assert_called_once()
        settings = fs_patch.call_args.args[0]
        self.assertTrue(settings.engine_config["output_hidden_states"])

    async def test_run_with_browser_settings(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.tool_browser_engine = "chromium"
        self.args.tool_browser_debug = True
        self.args.tool_browser_search = True
        self.args.tool_browser_search_context = 5

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        tool_settings = fs_patch.call_args.kwargs["tool_settings"]
        bs = tool_settings.browser
        self.assertEqual(bs.engine, "chromium")
        self.assertTrue(bs.debug)
        self.assertTrue(bs.search)
        self.assertEqual(bs.search_context, 5)
        dbs = tool_settings.database
        self.assertIsNone(dbs)
        self.assertIsNone(tool_settings.graph)

    async def test_run_with_graph_settings(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.tool_graph_file = "/tmp/chart.png"

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        tool_settings = fs_patch.call_args.kwargs["tool_settings"]
        graph = tool_settings.graph
        self.assertIsInstance(graph, GraphToolSettings)
        self.assertEqual(graph.file, "/tmp/chart.png")

    async def test_run_start_session_and_print_recent(self):
        self.args.session = None
        self.orch.memory.has_recent_message = True
        self.orch.memory.recent_message.is_empty = False
        self.orch.memory.recent_message.data = ["m"]
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )
        self.orch.memory.start_session.assert_awaited_once()
        self.console.print.assert_any_call("agent_panel")
        self.console.print.assert_any_call("recent_panel")

    async def test_run_start_session_without_recent_messages(self):
        self.args.session = None
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.assert_not_awaited()
        self.orch.memory.start_session.assert_awaited_once()
        self.orch.memory.continue_session.assert_not_awaited()
        self.console.print.assert_called_once_with("agent_panel")

    async def test_run_quiet_prints_output(self):
        self.args.quiet = True
        self.orch.memory.has_recent_message = True
        self.orch.memory.recent_message.is_empty = False
        self.orch.memory.recent_message.data = ["m"]

        class DummyOrchestratorResponse:
            pass

        output = DummyOrchestratorResponse()
        output.to_str = AsyncMock(return_value="out")
        self.orch.return_value = output
        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )
        tg_patch.assert_awaited_once()
        output.to_str.assert_not_called()
        self.theme.agent.assert_not_called()
        self.theme.recent_messages.assert_not_called()
        self.console.print.assert_not_called()

    async def test_run_conversation_prints_blank(self):
        self.args.conversation = True
        self.args.display_reasoning = True
        self.orch.return_value = MagicMock(
            spec=agent_cmds.OrchestratorResponse
        )
        self.orch.return_value.to_str = AsyncMock(return_value="x")
        with (
            patch.object(agent_cmds, "get_input", side_effect=["hi", None]),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ) as from_file,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )
        self.console.print.assert_any_call("")
        self.assertNotIn(
            "call_options_override",
            from_file.call_args.kwargs,
        )

    async def test_run_watch_reloads_when_file_changes(self):
        self.args.conversation = True
        self.args.watch = True
        self.args.run_reasoning_summary = "concise"
        second_orch = AsyncMock()
        second_orch.engine_agent = True
        second_orch.engine = MagicMock(model_id="m")
        second_orch.model_ids = ["m"]
        second_orch.event_manager.add_listener = MagicMock()
        second_orch.memory = MagicMock()
        second_orch.memory.continue_session = AsyncMock()
        second_orch.memory.start_session = AsyncMock()
        second_orch.memory.has_recent_message = False
        second_orch.memory.has_permanent_message = False
        second_orch.memory.recent_message = MagicMock(is_empty=True, size=0)
        self.orch.return_value = MagicMock(
            spec=agent_cmds.OrchestratorResponse
        )
        second_orch.return_value = MagicMock(
            spec=agent_cmds.OrchestratorResponse
        )
        self.dummy_stack.enter_async_context.side_effect = lambda o: o

        with (
            patch.object(agent_cmds, "get_input", side_effect=["hi", None]),
            patch.object(agent_cmds, "has_input", return_value=False),
            patch.object(agent_cmds, "getmtime", side_effect=[1, 1, 2, 2]),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(side_effect=[self.orch, second_orch]),
            ) as ff,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.assertEqual(ff.await_count, 2)
        for load_call in ff.await_args_list:
            self.assertEqual(
                load_call.kwargs["call_options_override"],
                {
                    "reasoning": {
                        "summary": ReasoningSummaryMode.CONCISE,
                    }
                },
            )

    async def test_run_watch_start_session_after_reload(self):
        self.args.conversation = True
        self.args.watch = True
        self.args.session = None
        second_orch = AsyncMock()
        second_orch.engine_agent = True
        second_orch.engine = MagicMock(model_id="m")
        second_orch.model_ids = ["m"]
        second_orch.event_manager.add_listener = MagicMock()
        second_orch.memory = MagicMock()
        second_orch.memory.continue_session = AsyncMock()
        second_orch.memory.start_session = AsyncMock()
        second_orch.memory.has_recent_message = False
        second_orch.memory.has_permanent_message = False
        second_orch.memory.recent_message = MagicMock(is_empty=True, size=0)
        self.orch.return_value = MagicMock(
            spec=agent_cmds.OrchestratorResponse
        )
        second_orch.return_value = MagicMock(
            spec=agent_cmds.OrchestratorResponse
        )
        self.dummy_stack.enter_async_context.side_effect = lambda o: o

        with (
            patch.object(agent_cmds, "get_input", side_effect=["hi", None]),
            patch.object(agent_cmds, "has_input", return_value=False),
            patch.object(agent_cmds, "getmtime", side_effect=[1, 1, 2, 2]),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(side_effect=[self.orch, second_orch]),
            ) as ff,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.assertEqual(ff.await_count, 2)
        self.orch.memory.start_session.assert_awaited_once()
        second_orch.memory.start_session.assert_awaited_once()
        tg.assert_awaited_once()

    async def test_event_listener_counts_events(self):
        captured = {}

        def add_listener(fn):
            captured["fn"] = fn

        self.orch.event_manager.add_listener.side_effect = add_listener
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )
        ev = Event(type=EventType.START, payload={})
        self.assertIn("fn", captured)
        stats = captured["fn"].__closure__[0].cell_contents
        self.assertEqual(stats.total_triggers, 0)
        await captured["fn"](ev)
        self.assertEqual(stats.total_triggers, 1)

    async def test_event_listener_counts_duplicate_events(self):
        captured = {}

        def add_listener(fn):
            captured["fn"] = fn

        self.orch.event_manager.add_listener.side_effect = add_listener
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )
        ev = Event(type=EventType.START, payload={})
        fn = captured["fn"]
        self.assertEqual(fn.__closure__[0].cell_contents.total_triggers, 0)
        await fn(ev)
        await fn(ev)
        stats = fn.__closure__[0].cell_contents
        self.assertEqual(stats.total_triggers, 2)
        self.assertEqual(stats.triggers[EventType.START], 2)

    async def test_event_listener_uses_bounded_ui_policy_when_available(
        self,
    ):
        captured = {}

        def add_ui_listener(fn):
            captured["fn"] = fn

        self.orch.event_manager.add_ui_listener = MagicMock(
            side_effect=add_ui_listener
        )
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.event_manager.add_ui_listener.assert_called_once()
        self.orch.event_manager.add_listener.assert_not_called()
        self.assertIn("fn", captured)

    async def test_event_listener_removed_on_stack_exit(self):
        captured = {}
        stack = self._callback_stack()
        self.orch.event_manager.remove_listener = MagicMock()

        def add_listener(fn):
            captured["fn"] = fn

        self.orch.event_manager.add_listener.side_effect = add_listener
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(agent_cmds, "AsyncExitStack", return_value=stack),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.assertIn("fn", captured)
        self.orch.event_manager.remove_listener.assert_called_once_with(
            captured["fn"]
        )

    async def test_ui_event_listener_removed_on_stack_exit(self):
        captured = {}
        stack = self._callback_stack()
        self.orch.event_manager.remove_listener = MagicMock()

        def add_ui_listener(fn):
            captured["fn"] = fn

        self.orch.event_manager.add_ui_listener = MagicMock(
            side_effect=add_ui_listener
        )
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(agent_cmds, "AsyncExitStack", return_value=stack),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.assertIn("fn", captured)
        self.orch.event_manager.remove_listener.assert_called_once_with(
            captured["fn"]
        )

    async def test_run_tools_confirm_calls_callback(self):
        self.args.tools_confirm = True
        self.args.tty = "/tmp/tty"
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})

        class DummyOrchestratorResponse:
            pass

        async def orch_call(*args, tool_confirm=None, **kwargs):
            self.assertIsNotNone(tool_confirm)
            self.callback_result = await tool_confirm(call_obj)
            return DummyOrchestratorResponse()

        self.orch.side_effect = orch_call

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
            patch.object(
                agent_cmds.CliStreamCoordinator,
                "confirm_tool_call",
                new=AsyncMock(return_value="y"),
            ) as ctc,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.assert_awaited_once_with(
            "hi", use_async_generator=True, tool_confirm=unittest.mock.ANY
        )
        tg.assert_awaited_once()
        ctc.assert_awaited_once_with(call_obj, tty_path="/tmp/tty")
        self.assertEqual(self.callback_result, "y")

    async def test_run_tools_confirm_fallback_rejects_live_kwarg(self):
        self.args.tools_confirm = True
        self.args.tty = "/tmp/tty"
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})

        class DummyOrchestratorResponse:
            pass

        async def orch_call(
            *_args: object,
            tool_confirm: Callable[[ToolCall], Awaitable[str]] | None = None,
            **_kwargs: object,
        ) -> DummyOrchestratorResponse:
            self.assertIsNotNone(tool_confirm)
            callback = cast(Callable[[ToolCall], Awaitable[str]], tool_confirm)
            with self.assertRaises(TypeError):
                await cast(Any, callback)(call_obj, live=MagicMock())
            self.callback_result = await callback(call_obj)
            return DummyOrchestratorResponse()

        async def strict_confirm(
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            self.assertIs(call, call_obj)
            self.assertEqual(tty_path, "/tmp/tty")
            return "y"

        self.orch.side_effect = orch_call

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
            patch.object(
                agent_cmds.CliStreamCoordinator,
                "confirm_tool_call",
                new=AsyncMock(side_effect=strict_confirm),
            ) as ctc,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        ctc.assert_awaited_once_with(call_obj, tty_path="/tmp/tty")
        tg.assert_awaited_once()
        self.assertEqual(self.callback_result, "y")

    async def test_run_tools_confirm_stdin_not_tty(self):
        self.args.tools_confirm = True
        self.args.tty = "/tmp/tty"
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})

        class DummyOrchestratorResponse:
            pass

        async def orch_call(*args, tool_confirm=None, **kwargs):
            self.assertIsNotNone(tool_confirm)
            self.callback_result = await tool_confirm(call_obj)
            return DummyOrchestratorResponse()

        self.orch.side_effect = orch_call

        fake_tty = MagicMock()
        ctx = MagicMock()
        ctx.__enter__.return_value = fake_tty
        ctx.__exit__.return_value = False
        stdin_mock = MagicMock(isatty=MagicMock(return_value=False))

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
            patch("avalan.cli.stdin", stdin_mock),
            patch("avalan.cli.open", return_value=ctx) as open_patch,
            patch("avalan.cli.Prompt.ask", return_value="y") as ask,
            patch("avalan.cli.stream_coordinator.Live") as live_patch,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        open_patch.assert_called_once_with("/tmp/tty")
        ask.assert_called_once_with(
            "Execute tool call?",
            choices=["y", "a", "n"],
            default="n",
            show_choices=True,
            show_default=True,
            console=self.console,
            stream=fake_tty,
        )
        self.orch.assert_awaited_once_with(
            "hi", use_async_generator=True, tool_confirm=unittest.mock.ANY
        )
        tg.assert_awaited_once()
        self.assertEqual(self.callback_result, "y")
        live_patch.assert_not_called()

    async def test_run_tools_confirm_with_display_tools(self):
        self.args.tools_confirm = True
        self.args.display_tools = True
        self.args.tty = "/tmp/tty"
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})

        class DummyOrchestratorResponse:
            pass

        async def orch_call(*args, tool_confirm=None, **kwargs):
            self.assertIsNotNone(tool_confirm)
            self.callback_result = await tool_confirm(call_obj)
            return DummyOrchestratorResponse()

        self.orch.side_effect = orch_call

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
            patch.object(
                agent_cmds.CliStreamCoordinator,
                "confirm_tool_call",
                new=AsyncMock(return_value="y"),
            ) as ctc,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        ctc.assert_awaited_once_with(call_obj, tty_path="/tmp/tty")
        tg.assert_awaited_once()
        self.assertEqual(self.callback_result, "y")

    async def test_run_tools_confirm_uses_active_stream_coordinator(self):
        self.args.tools_confirm = True
        self.args.display_tools = True
        self.args.tty = "/tmp/tty"
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})
        callbacks: list[Callable[[ToolCall], Awaitable[str]]] = []
        results: list[str] = []

        class DummyOrchestratorResponse:
            pass

        async def orch_call(
            *_args: object,
            tool_confirm: Callable[[ToolCall], Awaitable[str]] | None = None,
            **_kwargs: object,
        ) -> DummyOrchestratorResponse:
            self.assertIsNotNone(tool_confirm)
            callbacks.append(tool_confirm)
            return DummyOrchestratorResponse()

        confirm_tool_call = AsyncMock(return_value="a")
        active_coordinator = SimpleNamespace(
            confirm_tool_call=confirm_tool_call
        )

        async def fake_token_generation(
            **kwargs: object,
        ) -> None:
            coordinator_container = kwargs["coordinator_container"]
            assert isinstance(coordinator_container, dict)
            coordinator_container["coordinator"] = active_coordinator
            callback = callbacks[0]
            results.append(await callback(call_obj))
            coordinator_container["coordinator"] = None

        self.orch.side_effect = orch_call

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds,
                "token_generation",
                new=AsyncMock(side_effect=fake_token_generation),
            ) as tg,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
            patch.object(agent_cmds, "CliStreamCoordinator") as fallback,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        tg.assert_awaited_once()
        self.assertEqual(results, ["a"])
        confirm_tool_call.assert_awaited_once_with(
            call_obj,
            tty_path="/tmp/tty",
        )
        fallback.assert_not_called()
        coordinator_container = cast(
            dict[str, object | None],
            tg.await_args.kwargs["coordinator_container"],
        )
        self.assertIsNone(coordinator_container["coordinator"])

    async def test_run_tools_confirm_active_rejects_live_kwarg(self):
        self.args.tools_confirm = True
        self.args.display_tools = True
        self.args.tty = "/tmp/tty"
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})
        callbacks: list[Callable[[ToolCall], Awaitable[str]]] = []
        results: list[str] = []

        class DummyOrchestratorResponse:
            pass

        async def orch_call(
            *_args: object,
            tool_confirm: Callable[[ToolCall], Awaitable[str]] | None = None,
            **_kwargs: object,
        ) -> DummyOrchestratorResponse:
            self.assertIsNotNone(tool_confirm)
            callbacks.append(
                cast(Callable[[ToolCall], Awaitable[str]], tool_confirm)
            )
            return DummyOrchestratorResponse()

        async def strict_confirm(
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            self.assertIs(call, call_obj)
            self.assertEqual(tty_path, "/tmp/tty")
            return "a"

        confirm_tool_call = AsyncMock(side_effect=strict_confirm)
        active_coordinator = SimpleNamespace(
            confirm_tool_call=confirm_tool_call
        )

        async def fake_token_generation(
            **kwargs: object,
        ) -> None:
            coordinator_container = kwargs["coordinator_container"]
            assert isinstance(coordinator_container, dict)
            coordinator_container["coordinator"] = active_coordinator
            callback = callbacks[0]
            with self.assertRaises(TypeError):
                await cast(Any, callback)(call_obj, live=MagicMock())
            results.append(await callback(call_obj))
            coordinator_container["coordinator"] = None

        self.orch.side_effect = orch_call

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds,
                "token_generation",
                new=AsyncMock(side_effect=fake_token_generation),
            ) as tg,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
            patch.object(agent_cmds, "CliStreamCoordinator") as fallback,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        tg.assert_awaited_once()
        self.assertEqual(results, ["a"])
        confirm_tool_call.assert_awaited_once_with(
            call_obj,
            tty_path="/tmp/tty",
        )
        fallback.assert_not_called()

    async def test_run_tools_confirm_active_prompt_error_propagates(self):
        self.args.tools_confirm = True
        self.args.display_tools = True
        self.args.tty = "/tmp/tty"
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})
        callbacks: list[Callable[[ToolCall], Awaitable[str]]] = []

        class DummyOrchestratorResponse:
            pass

        async def orch_call(
            *_args: object,
            tool_confirm: Callable[[ToolCall], Awaitable[str]] | None = None,
            **_kwargs: object,
        ) -> DummyOrchestratorResponse:
            self.assertIsNotNone(tool_confirm)
            callbacks.append(
                cast(Callable[[ToolCall], Awaitable[str]], tool_confirm)
            )
            return DummyOrchestratorResponse()

        confirm_tool_call = AsyncMock(
            side_effect=RuntimeError("prompt failed")
        )
        active_coordinator = SimpleNamespace(
            confirm_tool_call=confirm_tool_call
        )

        async def fake_token_generation(
            **kwargs: object,
        ) -> None:
            coordinator_container = kwargs["coordinator_container"]
            assert isinstance(coordinator_container, dict)
            coordinator_container["coordinator"] = active_coordinator
            try:
                await callbacks[0](call_obj)
            finally:
                coordinator_container["coordinator"] = None

        self.orch.side_effect = orch_call

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds,
                "token_generation",
                new=AsyncMock(side_effect=fake_token_generation),
            ) as tg,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
            patch.object(agent_cmds, "CliStreamCoordinator") as fallback,
        ):
            with self.assertRaisesRegex(RuntimeError, "prompt failed"):
                await agent_cmds.agent_run(
                    self.args,
                    self.console,
                    self.theme,
                    self.hub,
                    self.logger,
                    1,
                )

        tg.assert_awaited_once()
        confirm_tool_call.assert_awaited_once_with(
            call_obj,
            tty_path="/tmp/tty",
        )
        fallback.assert_not_called()

    async def test_run_tools_confirm_active_prompt_cancel_propagates(self):
        self.args.tools_confirm = True
        self.args.display_tools = True
        self.args.tty = "/tmp/tty"
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})
        callbacks: list[Callable[[ToolCall], Awaitable[str]]] = []

        class DummyOrchestratorResponse:
            pass

        async def orch_call(
            *_args: object,
            tool_confirm: Callable[[ToolCall], Awaitable[str]] | None = None,
            **_kwargs: object,
        ) -> DummyOrchestratorResponse:
            self.assertIsNotNone(tool_confirm)
            callbacks.append(
                cast(Callable[[ToolCall], Awaitable[str]], tool_confirm)
            )
            return DummyOrchestratorResponse()

        confirm_tool_call = AsyncMock(side_effect=CancelledError)
        active_coordinator = SimpleNamespace(
            confirm_tool_call=confirm_tool_call
        )

        async def fake_token_generation(
            **kwargs: object,
        ) -> None:
            coordinator_container = kwargs["coordinator_container"]
            assert isinstance(coordinator_container, dict)
            coordinator_container["coordinator"] = active_coordinator
            try:
                await callbacks[0](call_obj)
            finally:
                coordinator_container["coordinator"] = None

        self.orch.side_effect = orch_call

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds,
                "token_generation",
                new=AsyncMock(side_effect=fake_token_generation),
            ) as tg,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
            patch.object(agent_cmds, "CliStreamCoordinator") as fallback,
        ):
            with self.assertRaises(CancelledError):
                await agent_cmds.agent_run(
                    self.args,
                    self.console,
                    self.theme,
                    self.hub,
                    self.logger,
                    1,
                )

        tg.assert_awaited_once()
        confirm_tool_call.assert_awaited_once_with(
            call_obj,
            tty_path="/tmp/tty",
        )
        fallback.assert_not_called()

    async def test_run_passes_tool_manager_to_model_call(self):
        self.args.tool = ["math"]

        tool_manager = MagicMock(spec=ToolManager)
        self.orch.tool = tool_manager

        class DummyEngine:
            model_id = "m"
            model_type = "t"
            last_tool = None

            async def __call__(self, *_a, tool=None, **_k):
                DummyEngine.last_tool = tool
                return "out"

            def input_token_count(self, *_a, **_k):
                return 0

        class DummyModelManager:
            def __init__(self) -> None:
                self.passed_tool = None

            async def __call__(self, task: ModelCall):
                self.passed_tool = task.tool
                return await task.model(None, tool=task.tool)

        engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )

        model_manager = DummyModelManager()
        memory = MagicMock(spec=MemoryManager)
        memory.has_permanent_message = False
        memory.has_recent_message = False
        memory.recent_message = Message(role=MessageRole.USER, content="hi")
        memory.recent_messages = []
        memory.append_message = AsyncMock()
        memory.participant_id = uuid4()
        memory.permanent_message = None

        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        class DummyAgent(EngineAgent):
            def _prepare_call(self, context: ModelCallContext):
                return {}

            async def _run(self, context: ModelCallContext, input, **_kwargs):
                operation = Operation(
                    generation_settings=GenerationSettings(),
                    input=input,
                    modality=Modality.TEXT_GENERATION,
                    parameters=OperationParameters(
                        text=OperationTextParameters()
                    ),
                    requires_input=True,
                )
                return await self._model_manager(
                    ModelCall(
                        engine_uri=engine_uri,
                        model=engine,
                        operation=operation,
                        tool=self._tool,
                        context=context,
                    )
                )

        engine = DummyEngine()
        agent = DummyAgent(
            engine,
            memory,
            tool_manager,
            event_manager,
            model_manager,
            engine_uri,
        )

        async def orch_call(*_a, **_k):
            context = ModelCallContext(
                specification=Specification(role="assistant", goal=None),
                input=Message(role=MessageRole.USER, content="hi"),
            )
            await agent(context)
            return DummyOrchestratorResponse()

        class DummyOrchestratorResponse:
            def __aiter__(self_inner):
                async def gen():
                    yield "x"

                return gen()

        self.orch.side_effect = orch_call
        self.orch.engine = engine

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.assertIs(model_manager.passed_tool, tool_manager)
        self.assertIs(DummyEngine.last_tool, tool_manager)

    async def test_tool_format_sets_tool_manager_format(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.tool_format = ToolFormat.REACT.value

        async def from_settings_side_effect(
            _settings,
            *,
            tool_settings=None,
            tool_format=None,
            tool_recovery_formats=None,
            event_manager_mode=None,
        ):
            self.assertIs(event_manager_mode, EventManagerMode.CLI)
            self.orch.tool = ToolManager.create_instance(
                available_toolsets=[],
                enable_tools=None,
                settings=ToolManagerSettings(tool_format=tool_format),
            )
            return self.orch

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(side_effect=from_settings_side_effect),
            ),
            patch.object(
                agent_cmds.OrchestratorLoader, "from_file", new=AsyncMock()
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.assertEqual(
            self.orch.tool._settings.tool_format, ToolFormat.REACT
        )

    def test_agent_tool_format_defaults_react_for_local_tool_backends(self):
        for backend in ("ds4", "mlx"):
            with self.subTest(backend=backend):
                self.args.backend = backend
                self.args.tool = ["math.calculator"]
                self.args.tool_format = None

                self.assertIs(
                    agent_cmds._agent_tool_format(self.args),
                    ToolFormat.REACT,
                )

    def test_agent_tool_format_keeps_explicit_and_non_tool_defaults(self):
        self.args.backend = "ds4"
        self.args.tool = ["math.calculator"]
        self.args.tool_format = ToolFormat.JSON.value

        self.assertIs(
            agent_cmds._agent_tool_format(self.args), ToolFormat.JSON
        )

        self.args.tool_format = None
        self.args.tool = None

        self.assertIsNone(agent_cmds._agent_tool_format(self.args))

        self.args.backend = "transformers"
        self.args.tool = ["math.calculator"]

        self.assertIsNone(agent_cmds._agent_tool_format(self.args))

    async def test_run_engine_uri_only_generates_id(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = None
        self.args.id = None

        uid = uuid4()

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ) as ff_patch,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
            patch("avalan.cli.commands.agent.uuid4", return_value=uid),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        ff_patch.assert_not_called()
        settings = fs_patch.call_args.args[0]
        self.assertEqual(settings.agent_id, uid)
        self.assertEqual(settings.uri, "engine")

    async def test_run_engine_uri_with_id(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = None
        self.args.id = "custom"

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ) as ff_patch,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        ff_patch.assert_not_called()
        settings = fs_patch.call_args.args[0]
        self.assertEqual(settings.agent_id, "custom")

    async def test_run_engine_uri_with_name(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.name = "Agent"

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ) as ff_patch,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        ff_patch.assert_not_called()
        settings = fs_patch.call_args.args[0]
        self.assertEqual(settings.agent_config["name"], "Agent")

    async def test_run_engine_uri_with_generation_settings(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.run_temperature = 0.5
        self.args.run_top_p = 0.9
        self.args.run_top_k = 5
        self.args.run_max_new_tokens = 42
        self.args.run_openai_max_retries = 0
        self.args.run_openai_response_failed_retries = 0
        self.args.run_openai_response_failed_retry_delay_seconds = 0.5
        self.args.run_openai_timeout_seconds = 30
        self.args.tool_choice = "mcp.call"

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        settings = fs_patch.call_args.args[0]
        self.assertEqual(settings.call_options["temperature"], 0.5)
        self.assertEqual(settings.call_options["top_p"], 0.9)
        self.assertEqual(settings.call_options["top_k"], 5)
        self.assertEqual(settings.call_options["max_new_tokens"], 42)
        self.assertEqual(settings.call_options["openai_max_retries"], 0)
        self.assertEqual(
            settings.call_options["openai_response_failed_retries"],
            0,
        )
        self.assertEqual(
            settings.call_options[
                "openai_response_failed_retry_delay_seconds"
            ],
            0.5,
        )
        self.assertEqual(settings.call_options["openai_timeout_seconds"], 30)
        self.assertEqual(settings.call_options["tool_choice"], "mcp.call")

    async def test_run_engine_uri_use_cache_cli(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.run_use_cache = False

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        fs_patch.assert_awaited_once()
        settings = fs_patch.call_args.args[0]
        self.assertFalse(settings.call_options["use_cache"])

    async def test_run_engine_uri_cache_strategy_cli(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        for strat in [None] + [s.value for s in GenerationCacheStrategy]:
            self.args.run_cache_strategy = strat
            with (
                patch.object(agent_cmds, "get_input", return_value=None),
                patch.object(
                    agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
                ),
                patch.object(
                    agent_cmds.OrchestratorLoader,
                    "from_settings",
                    new=AsyncMock(return_value=self.orch),
                ) as fs_patch,
                patch.object(
                    agent_cmds.OrchestratorLoader,
                    "from_file",
                    new=AsyncMock(),
                ),
                patch.object(
                    agent_cmds, "token_generation", new_callable=AsyncMock
                ),
            ):
                await agent_cmds.agent_run(
                    self.args,
                    self.console,
                    self.theme,
                    self.hub,
                    self.logger,
                    1,
                )
            settings = fs_patch.call_args.args[0]
            if strat is None:
                self.assertNotIn("cache_strategy", settings.call_options)
            else:
                self.assertEqual(
                    settings.call_options["cache_strategy"], strat
                )

    async def test_run_engine_uri_reasoning_effort_cli(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.run_reasoning_effort = "xhigh"

        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=self.orch),
            ) as fs_patch,
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args,
                self.console,
                self.theme,
                self.hub,
                self.logger,
                1,
            )

        settings = fs_patch.call_args.args[0]
        self.assertEqual(settings.call_options["reasoning"]["effort"], "xhigh")

    async def test_run_spec_use_cache(self):
        captured = {}

        async def orch_call(input, **engine_args):
            captured["args"] = {
                **getattr(self.orch, "_call_options", {}),
                **engine_args,
            }
            return MagicMock(spec=agent_cmds.OrchestratorResponse)

        self.orch.side_effect = orch_call

        for use_cache in (None, True, False):
            with NamedTemporaryFile("w+", suffix=".toml") as spec:
                spec.write(
                    "[engine]\nuri='engine'\n[agent]\nrole='assistant'\n"
                )
                spec.write("[run]\n")
                if use_cache is not None:
                    spec.write(f"use_cache = {str(use_cache).lower()}\n")
                spec.flush()
                self.args.specifications_file = spec.name
                self.args.engine_uri = None
                self.args.quiet = True

                async def from_file(
                    self_loader,
                    path,
                    agent_id,
                    disable_memory=None,
                    **kwargs,
                ):
                    import tomllib

                    with open(path, "rb") as f:
                        data = tomllib.load(f)
                    self.orch._call_options = data.get("run", {})
                    return self.orch

                with (
                    patch.object(agent_cmds, "get_input", return_value="hi"),
                    patch.object(
                        agent_cmds,
                        "AsyncExitStack",
                        return_value=self.dummy_stack,
                    ),
                    patch.object(
                        agent_cmds.OrchestratorLoader,
                        "from_file",
                        new=from_file,
                    ),
                    patch.object(
                        agent_cmds, "token_generation", new_callable=AsyncMock
                    ),
                ):
                    await agent_cmds.agent_run(
                        self.args,
                        self.console,
                        self.theme,
                        self.hub,
                        self.logger,
                        1,
                    )

            engine_args = captured["args"]
            if use_cache is None:
                self.assertNotIn("use_cache", engine_args)
            else:
                self.assertEqual(engine_args["use_cache"], use_cache)

    async def test_run_spec_cache_strategy(self):
        captured = {}

        async def orch_call(input, **engine_args):
            captured["args"] = {
                **getattr(self.orch, "_call_options", {}),
                **engine_args,
            }
            return MagicMock(spec=agent_cmds.OrchestratorResponse)

        self.orch.side_effect = orch_call

        strategies = [None] + [s.value for s in GenerationCacheStrategy]
        for strat in strategies:
            with NamedTemporaryFile("w+", suffix=".toml") as spec:
                spec.write(
                    "[engine]\nuri='engine'\n[agent]\nrole='assistant'\n"
                )
                spec.write("[run]\n")
                if strat is not None:
                    spec.write(f"cache_strategy = '{strat}'\n")
                spec.flush()
                self.args.specifications_file = spec.name
                self.args.engine_uri = None
                self.args.quiet = True

                async def from_file(
                    self_loader,
                    path,
                    agent_id,
                    disable_memory=None,
                    **kwargs,
                ):
                    import tomllib

                    with open(path, "rb") as f:
                        data = tomllib.load(f)
                    self.orch._call_options = data.get("run", {})
                    return self.orch

                with (
                    patch.object(agent_cmds, "get_input", return_value="hi"),
                    patch.object(
                        agent_cmds,
                        "AsyncExitStack",
                        return_value=self.dummy_stack,
                    ),
                    patch.object(
                        agent_cmds.OrchestratorLoader,
                        "from_file",
                        new=from_file,
                    ),
                    patch.object(
                        agent_cmds, "token_generation", new_callable=AsyncMock
                    ),
                ):
                    await agent_cmds.agent_run(
                        self.args,
                        self.console,
                        self.theme,
                        self.hub,
                        self.logger,
                        1,
                    )

            engine_args = captured["args"]
            if strat is None:
                self.assertNotIn("cache_strategy", engine_args)
            else:
                self.assertEqual(engine_args["cache_strategy"], strat)


class CliAgentInitNoRoleTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_agent_init_accepts_no_role(self):
        args = Namespace(
            name="A",
            role=None,
            task=None,
            instructions=None,
            memory_recent=None,
            memory_permanent_message=None,
            memory_permanent=None,
            memory_engine_model_id=None,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            engine_uri="uri",
            run_max_new_tokens=10,
            run_skip_special_tokens=True,
            tool=None,
            backend="transformers",
            no_repl=False,
            quiet=False,
        )
        console = MagicMock()
        theme = MagicMock()
        theme._ = lambda s: s
        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=False),
            patch.object(agent_cmds, "get_input", return_value=""),
            patch.object(agent_cmds.Prompt, "ask", return_value="A"),
        ):
            await agent_cmds.agent_init(args, console, theme)
        console.print.assert_called_once()


class CliAgentMixedTokensTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = Namespace(
            specifications_file="spec.toml",
            use_sync_generator=False,
            display_tokens=0,
            stats=False,
            id="aid",
            participant="pid",
            session="sid",
            no_session=False,
            skip_load_recent_messages=False,
            load_recent_messages_limit=1,
            no_repl=False,
            quiet=False,
            skip_hub_access_check=False,
            conversation=False,
            watch=False,
            tty=None,
            tool_events=2,
            tool=None,
            run_max_new_tokens=100,
            run_skip_special_tokens=False,
            engine_uri=None,
            name=None,
            role=None,
            task=None,
            instructions=None,
            memory_recent=None,
            memory_permanent_message=None,
            memory_permanent=None,
            memory_engine_model_id=agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID,
            memory_engine_max_tokens=500,
            memory_engine_overlap=125,
            memory_engine_window=250,
            tool_browser_engine=None,
            tool_browser_debug=None,
            tool_browser_search=None,
            tool_browser_search_context=None,
            display_events=False,
            display_tools=False,
            display_tools_events=2,
            tools_confirm=False,
        )
        self.console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        self.console.status.return_value = status_cm
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme.icons = {"user_input": ">", "agent_output": "<"}
        self.theme.get_spinner.return_value = "sp"
        self.theme.agent.return_value = "agent_panel"
        self.theme.recent_messages.return_value = "recent_panel"
        self.hub = MagicMock()
        self.hub.can_access.return_value = True
        self.hub.model.side_effect = lambda m: f"mdl-{m}"
        self.logger = MagicMock()

        self.orch = AsyncMock()
        self.orch.engine_agent = True
        self.orch.engine = MagicMock(model_id="m")
        self.orch.model_ids = ["m"]
        self.orch.event_manager.add_listener = MagicMock()
        self.orch.memory = MagicMock()
        self.orch.memory.has_recent_message = False
        self.orch.memory.has_permanent_message = False
        self.orch.memory.recent_message = MagicMock(
            is_empty=True, size=0, data=[]
        )
        self.orch.memory.continue_session = AsyncMock()
        self.orch.memory.start_session = AsyncMock()

        self.dummy_stack = AsyncMock()
        self.dummy_stack.__aenter__.return_value = self.dummy_stack
        self.dummy_stack.__aexit__.return_value = False
        self.dummy_stack.enter_async_context = AsyncMock(
            return_value=self.orch
        )

    async def test_agent_run_mixed_tokens(self):
        async def complex_generator():
            rp = ReasoningParser(
                reasoning_settings=ReasoningSettings(),
                logger=getLogger(),
            )
            tm = MagicMock()
            tm.is_potential_tool_call.return_value = True
            tm.get_calls.return_value = None
            base_parser = ToolCallParser()
            tm.tool_call_status.side_effect = base_parser.tool_call_status
            tp = ToolCallResponseParser(tm, None)
            sequence = [
                "X",
                "<think>",
                "ra",
                "rb",
                "</think>",
                "Y",
                "<tool_call>",
                "foo",
                "bar",
                "</tool_call>",
                "Z",
            ]
            for s in sequence:
                items = await rp.push(s)
                for item in items:
                    parsed = (
                        await tp.push(item)
                        if isinstance(item, str)
                        else [item]
                    )
                    for p in parsed:
                        assert isinstance(p, StreamProviderEvent)
                        yield p

        class DummyOrchestratorResponse:
            input_token_count = 1

            def __aiter__(self_inner):
                return complex_generator()

        self.orch.return_value = DummyOrchestratorResponse()

        with (
            patch.object(agent_cmds, "get_input", return_value="hi"),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds, "OrchestratorResponse", DummyOrchestratorResponse
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        tg_patch.assert_awaited_once()
        resp_obj = tg_patch.await_args.kwargs["response"]
        tokens = []
        async for t in resp_obj:
            tokens.append(t)

        self.assertFalse(any(isinstance(t, ReasoningToken) for t in tokens))
        reasoning_deltas = [
            t
            for t in tokens
            if (
                isinstance(t, StreamProviderEvent)
                and t.kind is StreamItemKind.REASONING_DELTA
            )
        ]
        self.assertEqual(
            [event.text_delta for event in reasoning_deltas],
            ["<think>", "ra", "rb", "</think>"],
        )
        self.assertTrue(
            any(
                isinstance(t, StreamProviderEvent)
                and t.kind is StreamItemKind.REASONING_DONE
                for t in tokens
            )
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, ToolCallToken)]),
            0,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, TokenDetail)]),
            0,
        )
        self.assertEqual(len([t for t in tokens if type(t) is Token]), 0)
        self.assertEqual(len([t for t in tokens if isinstance(t, str)]), 0)
        answer_events = [
            t
            for t in tokens
            if (
                isinstance(t, StreamProviderEvent)
                and t.kind is StreamItemKind.ANSWER_DELTA
            )
        ]
        self.assertEqual(
            [event.text_delta for event in answer_events],
            ["X", "Y", "Z"],
        )
