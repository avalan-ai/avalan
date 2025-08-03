import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from argparse import Namespace
from uuid import uuid4
from tempfile import NamedTemporaryFile

from rich.syntax import Syntax

from avalan.entities import (
    ToolCall,
    Message,
    MessageRole,
    GenerationSettings,
    EngineUri,
    Modality,
    Operation,
    OperationParameters,
    OperationTextParameters,
)
from avalan.agent import Specification
from avalan.agent.engine import EngineAgent
from avalan.memory.manager import MemoryManager
from avalan.event.manager import EventManager
from avalan.tool.manager import ToolManager
from avalan.cli.commands import agent as agent_cmds
from avalan.event import Event, EventType
from avalan.memory.permanent import VectorFunction
from avalan.model.response.text import TextGenerationResponse
from avalan.model.response.parsers.reasoning import ReasoningParser
from logging import getLogger
from avalan.entities import OrchestratorSettings, ReasoningSettings
from avalan.model.response.parsers.tool import ToolCallParser
from avalan.entities import ReasoningToken, Token, TokenDetail, ToolCallToken
from avalan.tool.browser import BrowserToolSettings


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
        lf.assert_not_called()
        dummy_stack.enter_async_context.assert_awaited_once_with(orch)
        orch.memory.search_messages.assert_awaited_once()


class CliAgentServeTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_agent_serve(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            specifications_file=None,
            prefix_openai="oa",
            prefix_mcp="mcp",
            reload=False,
            backend="transformers",
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
            prefix_openai="oa",
            prefix_mcp="mcp",
            specs_path=spec.name,
            settings=None,
            browser_settings=None,
            host="0.0.0.0",
            port=80,
            reload=False,
            logger=logger,
        )
        server.serve.assert_awaited_once()

    async def test_agent_serve_from_settings(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            specifications_file=None,
            prefix_openai="oa",
            prefix_mcp="mcp",
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
                agent_cmds, "get_tool_settings", return_value=browser_settings
            ) as gts,
            patch.object(
                agent_cmds, "agents_server", return_value=server
            ) as asrv,
        ):
            await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")

        gos.assert_called_once()
        gts.assert_called_once_with(
            args, prefix="browser", settings_cls=BrowserToolSettings
        )
        asrv.assert_called_once_with(
            hub=hub,
            name="name",
            version="1.0",
            prefix_openai="oa",
            prefix_mcp="mcp",
            specs_path=None,
            settings=settings,
            browser_settings=browser_settings,
            host="0.0.0.0",
            port=80,
            reload=False,
            logger=logger,
        )
        server.serve.assert_awaited_once()

    async def test_agent_serve_needs_settings(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            specifications_file=None,
            prefix_openai="oa",
            prefix_mcp="mcp",
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
            prefix_openai="oa",
            prefix_mcp="mcp",
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

    async def test_agent_proxy_requires_engine(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            prefix_openai="oa",
            prefix_mcp="mcp",
            reload=False,
            backend="transformers",
            engine_uri=None,
        )
        hub = MagicMock()
        logger = MagicMock()

        with self.assertRaises(AssertionError):
            await agent_cmds.agent_proxy(args, hub, logger, "name", "1.0")


class CliAgentInitTestCase(unittest.IsolatedAsyncioTestCase):
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
            patch.object(agent_cmds, "get_input", side_effect=["r", "t", "i"]),
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "Environment", return_value=env),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_called_once()
        settings = template.render.call_args.kwargs["orchestrator"]
        self.assertTrue(settings.call_options["skip_special_tokens"])
        self.assertEqual(settings.call_options["max_new_tokens"], 1024)
        self.assertIsInstance(console.print.call_args.args[0], Syntax)
        self.assertIsInstance(console.print.call_args.args[0], Syntax)

    async def test_agent_init_output_without_tool_settings(self):
        args = Namespace(
            name="N",
            role="R",
            task="T",
            instructions="I",
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

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T", "I"]),
            patch.object(
                agent_cmds.Prompt, "ask", side_effect=["N", "", "uri"]
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertNotIn("[tool.browser.open]", output)

    async def test_agent_init_tool_settings_output(self):
        args = Namespace(
            name="N",
            role="R",
            task="T",
            instructions="I",
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

        with (
            patch.object(agent_cmds.Confirm, "ask", return_value=True),
            patch.object(agent_cmds, "get_input", side_effect=["R", "T", "I"]),
            patch.object(
                agent_cmds.Prompt, "ask", side_effect=["N", "", "uri"]
            ),
        ):
            await agent_cmds.agent_init(args, console, theme)

        output = console.print.call_args.args[0].code
        self.assertIn("[tool.browser.open]", output)
        self.assertIn('engine = "chromium"', output)
        self.assertIn("search = true", output)


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
        self.orch.memory.continue_session.assert_awaited()
        self.console.print.assert_any_call("agent_panel")
        self.console.print.assert_any_call("< ", end="")

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

    async def test_run_from_settings(self):
        self.args.specifications_file = None
        self.args.engine_uri = "engine"
        self.args.role = "assistant"
        self.args.run_skip_special_tokens = True

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
        self.assertTrue(settings.call_options["skip_special_tokens"])
        browser_settings = fs_patch.call_args.kwargs["browser_settings"]
        self.assertIsNone(browser_settings)
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
        bs = fs_patch.call_args.kwargs["browser_settings"]
        self.assertEqual(bs.engine, "chromium")
        self.assertTrue(bs.debug)
        self.assertTrue(bs.search)
        self.assertEqual(bs.search_context, 5)

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
        output = AsyncMock()
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
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )
        tg_patch.assert_not_called()
        output.to_str.assert_awaited_once()
        self.console.print.assert_any_call("out")

    async def test_run_conversation_prints_blank(self):
        self.args.conversation = True
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
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ),
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )
        self.console.print.assert_any_call("")

    async def test_run_watch_reloads_when_file_changes(self):
        self.args.conversation = True
        self.args.watch = True
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

    async def test_run_tools_confirm_calls_callback(self):
        self.args.tools_confirm = True
        self.orch.tool = MagicMock(is_empty=False)
        call_obj = ToolCall(id=uuid4(), name="calc", arguments={"a": 1})

        class DummyOrchestratorResponse:
            pass

        async def orch_call(*args, tool_confirm=None, **kwargs):
            self.assertIsNotNone(tool_confirm)
            self.callback_result = tool_confirm(call_obj)
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
                agent_cmds, "confirm_tool_call", return_value="y"
            ) as ctc,
        ):
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.assert_awaited_once_with(
            "hi", use_async_generator=True, tool_confirm=unittest.mock.ANY
        )
        tg.assert_awaited_once()
        ctc.assert_called_once_with(self.console, call_obj)
        self.assertEqual(self.callback_result, "y")

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

            async def __call__(self, engine_uri, model, operation, tool):
                self.passed_tool = tool
                return await model(None, tool=tool)

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
            def _prepare_call(self, specification, input, **kwargs):
                return {}

            async def _run(self, input, **_kwargs):
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
                    engine_uri,
                    engine,
                    operation,
                    self._tool,
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
            await agent(
                Specification(role="assistant", goal=None),
                Message(role=MessageRole.USER, content="hi"),
            )
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
                reasoning_settings=ReasoningSettings(), logger=getLogger()
            )
            tm = MagicMock()
            tm.is_potential_tool_call.return_value = True
            tm.get_calls.return_value = None
            tp = ToolCallParser(tm, None)
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
                        if isinstance(p, str):
                            if p == "</think>":
                                yield TokenDetail(
                                    id=3, token=p, probability=0.5
                                )
                            elif p in {"X", "Y"}:
                                yield Token(id=1, token=p)
                            elif p == "<think>" or p == "Z":
                                yield p
                        elif isinstance(p, ToolCallToken):
                            if p.token == "</tool_call>":
                                yield TokenDetail(
                                    id=4, token=p.token, probability=0.5
                                )
                            else:
                                yield p
                        else:
                            yield p

        class DummyOrchestratorResponse:
            input_token_count = 1

            def __init__(self):
                settings = GenerationSettings()
                self._resp = TextGenerationResponse(
                    lambda **_: complex_generator(),
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )

            def __aiter__(self_inner):
                return self_inner._resp.__aiter__()

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

        self.assertEqual(
            len([t for t in tokens if isinstance(t, ReasoningToken)]),
            4,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, ToolCallToken)]),
            3,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, TokenDetail)]),
            1,
        )
        self.assertGreaterEqual(
            len([t for t in tokens if type(t) is Token]),
            2,
        )
        self.assertEqual(len([t for t in tokens if isinstance(t, str)]), 1)
