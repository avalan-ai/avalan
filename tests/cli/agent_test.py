import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from argparse import Namespace
from rich.syntax import Syntax

from avalan.cli.commands import agent as agent_cmds


class CliAgentMessageSearchTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.args = Namespace(
            specifications_file="spec.toml",
            id="aid",
            participant="pid",
            session="sid",
            no_repl=False,
            quiet=False,
            skip_hub_access_check=False,
            limit=1,
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
                agent_cmds.OrchestrationLoader, "from_file", new=AsyncMock()
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
                agent_cmds.OrchestrationLoader,
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
            session_id="sid",
            participant_id="pid",
            function=agent_cmds.VectorFunction.L2_DISTANCE,
            limit=1,
        )
        self.console.print.assert_any_call("agent_panel")
        self.console.print.assert_any_call("matches_panel")


class CliAgentServeTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_agent_serve(self):
        args = Namespace(
            host="0.0.0.0",
            port=80,
            specifications_file="spec.toml",
            prefix_openai="oa",
            prefix_mcp="mcp",
            reload=False,
        )
        hub = MagicMock()
        logger = MagicMock()
        orch = MagicMock()
        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)
        server = MagicMock()
        server.serve = AsyncMock()

        with (
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestrationLoader,
                "from_file",
                new=AsyncMock(return_value=orch),
            ) as lf,
            patch.object(
                agent_cmds, "agents_server", return_value=server
            ) as asrv,
        ):
            await agent_cmds.agent_serve(args, hub, logger, "name", "1.0")

        lf.assert_awaited_once()
        asrv.assert_called_once()
        server.serve.assert_awaited_once()


class CliAgentInitTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_agent_init(self):
        args = Namespace(
            name=None,
            role=None,
            task=None,
            instructions=None,
            memory_recent=None,
            memory_permanent=None,
            memory_engine_model_id=None,
            engine_uri=None,
            use_cache=None,
            skip_special_tokens=False,
            max_new_tokens=None,
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
            patch.object(agent_cmds.Confirm, "ask", side_effect=[True, False]),
            patch.object(agent_cmds, "Environment", return_value=env),
        ):
            await agent_cmds.agent_init(args, console, theme)

        template.render.assert_called_once()
        self.assertIsInstance(console.print.call_args.args[0], Syntax)


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
            tty=None,
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
        self.dummy_stack.enter_async_context = AsyncMock(return_value=self.orch)

    async def test_returns_when_no_input(self):
        with (
            patch.object(agent_cmds, "get_input", return_value=None),
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=self.dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestrationLoader,
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
                agent_cmds.OrchestrationLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds,
                "OrchestratorExecutionResponse",
                DummyOrchestratorResponse,
            ),
        ):
            self.orch.return_value = DummyOrchestratorResponse()
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.assert_awaited_once_with("hi", use_async_generator=True)
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
                agent_cmds.OrchestrationLoader,
                "from_file",
                new=AsyncMock(return_value=self.orch),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
            patch.object(
                agent_cmds,
                "OrchestratorExecutionResponse",
                DummyOrchestratorResponse,
            ),
        ):
            self.orch.return_value = DummyOrchestratorResponse()
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )

        self.orch.assert_awaited_once_with("hi", use_async_generator=True)
        tg_patch.assert_awaited_once()

        self.assertEqual(len(self.console.status.call_args_list), 1)
        self.console.status.assert_called_with(
            "Loading agent...",
            spinner=self.theme.get_spinner.return_value,
            refresh_per_second=1,
        )


if __name__ == "__main__":
    unittest.main()
