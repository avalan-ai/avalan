import unittest
from argparse import Namespace
from io import StringIO
from logging import getLogger
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from avalan.agent import (
    AgentOperation,
    EngineEnvironment,
    Specification,
)
from avalan.agent.engine import EngineAgent
from avalan.cli.commands import agent as agent_cmds
from avalan.cli.theme_registry import DEFAULT_THEME_NAME, create_theme
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
    Modality,
    OperationParameters,
    OperationTextParameters,
    ToolCall,
)
from avalan.entities import (
    Operation as EntitiesOperation,
)
from avalan.event.manager import EventManager
from avalan.model.call import ModelCall, ModelCallContext
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.tool.manager import ToolManager, ToolManagerSettings
from avalan.tool.math import MathToolSet

def _canonical_answer_response(*text_deltas: str) -> TextGenerationResponse:
    async def gen():
        sequence = 0
        yield CanonicalStreamItem(
            stream_session_id="cli-math-stream",
            run_id="cli-math-run",
            turn_id="cli-math-turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        sequence += 1
        for text_delta in text_deltas:
            yield CanonicalStreamItem(
                stream_session_id="cli-math-stream",
                run_id="cli-math-run",
                turn_id="cli-math-turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text_delta,
            )
            sequence += 1
        if text_deltas:
            yield CanonicalStreamItem(
                stream_session_id="cli-math-stream",
                run_id="cli-math-run",
                turn_id="cli-math-turn",
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            sequence += 1
        yield CanonicalStreamItem(
            stream_session_id="cli-math-stream",
            run_id="cli-math-run",
            turn_id="cli-math-turn",
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )

    return TextGenerationResponse(
        lambda: gen(), logger=getLogger(), use_async_generator=True
    )


def _canonical_tool_call_response() -> TextGenerationResponse:
    call = ToolCall(
        id="cli_math_tool_call_1",
        name="math.calculator",
        arguments={"expression": "(4 + 6) * 5 / 2"},
    )

    async def gen():
        yield CanonicalStreamItem(
            stream_session_id="cli-math-tool-stream",
            run_id="cli-math-tool-run",
            turn_id="cli-math-tool-turn",
            sequence=0,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        yield CanonicalStreamItem(
            stream_session_id="cli-math-tool-stream",
            run_id="cli-math-tool-run",
            turn_id="cli-math-tool-turn",
            sequence=1,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id=call.id),
            text_delta='{"expression": "(4 + 6) * 5 / 2"}',
        )
        yield CanonicalStreamItem(
            stream_session_id="cli-math-tool-stream",
            run_id="cli-math-tool-run",
            turn_id="cli-math-tool-turn",
            sequence=2,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id=call.id),
            data={"name": call.name, "arguments": call.arguments},
        )
        yield CanonicalStreamItem(
            stream_session_id="cli-math-tool-stream",
            run_id="cli-math-tool-run",
            turn_id="cli-math-tool-turn",
            sequence=3,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id=call.id),
        )
        yield CanonicalStreamItem(
            stream_session_id="cli-math-tool-stream",
            run_id="cli-math-tool-run",
            turn_id="cli-math-tool-turn",
            sequence=4,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield CanonicalStreamItem(
            stream_session_id="cli-math-tool-stream",
            run_id="cli-math-tool-run",
            turn_id="cli-math-tool-turn",
            sequence=5,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )

    return TextGenerationResponse(
        lambda: gen(),
        logger=getLogger(),
        use_async_generator=True,
    )


class DummyEngine:
    model_id = "m"
    model_type = "t"
    last_tool = None
    tokenizer = MagicMock()

    async def __call__(self, input, *, tool=None):
        DummyEngine.last_tool = tool
        if isinstance(input, Message):
            return _canonical_tool_call_response()
        else:
            result = (
                input[-1].content if isinstance(input, list) else str(input)
            )
            return _canonical_answer_response(f"The result is {result}.")

    def input_token_count(self, *_a, **_k):
        return 0


class DummyDs4Engine(DummyEngine):
    model_id = "./ds4flash.gguf"
    model_type = "ds4"
    tokenizer = None

    async def __call__(self, input, *, tool=None):
        DummyEngine.last_tool = tool
        if isinstance(input, Message):
            return _canonical_tool_call_response()
        return await super().__call__(input, tool=tool)


class DummyModelManager:
    def __init__(self) -> None:
        self.passed_tool = None

    async def __call__(self, task: ModelCall):
        self.passed_tool = task.tool
        return await task.model(task.operation.input, tool=task.tool)


class DummyAgent(EngineAgent):
    def _prepare_call(self, context: ModelCallContext):
        return {}

    async def _run(self, context: ModelCallContext, input, **_kwargs):
        operation = EntitiesOperation(
            generation_settings=GenerationSettings(),
            input=input,
            modality=Modality.TEXT_GENERATION,
            parameters=OperationParameters(text=OperationTextParameters()),
            requires_input=True,
        )
        self._last_operation = operation
        return await self._model_manager(
            ModelCall(
                engine_uri=self._engine_uri,
                model=self._model,
                operation=operation,
                tool=self._tool,
                context=context,
            )
        )


class DummyOrchestrator:
    def __init__(
        self,
        *,
        engine=None,
        engine_uri: EngineUri | None = None,
    ):
        self.id = "dummy-orchestrator"
        self.name = "Dummy"
        self.event_manager = EventManager()
        self.memory = MagicMock()
        self.memory.has_recent_message = False
        self.memory.has_permanent_message = False
        self.memory.recent_message = MagicMock(is_empty=True, size=0, data=[])
        self.memory.start_session = AsyncMock()
        self.memory.continue_session = AsyncMock()

        self.tool = ToolManager.create_instance(
            available_toolsets=[MathToolSet(namespace="math")],
            enable_tools=["math.calculator"],
            settings=ToolManagerSettings(),
        )
        self.engine_uri = engine_uri or EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.engine = engine or DummyEngine()
        self.model_manager = DummyModelManager()
        self.agent = DummyAgent(
            self.engine,
            self.memory,
            self.tool,
            self.event_manager,
            self.model_manager,
            self.engine_uri,
        )
        self.engine_agent = self.agent
        self.model_ids = ["m"]
        spec = Specification(role="assistant", goal=None)
        env = EngineEnvironment(engine_uri=self.engine_uri, settings=None)
        self._operation = AgentOperation(specification=spec, environment=env)

    async def __aenter__(self):
        await self.tool.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.tool.__aexit__(exc_type, exc, tb)
        return False

    async def __call__(self, input_str, **_kwargs):
        message = Message(role=MessageRole.USER, content=input_str)
        context = ModelCallContext(
            specification=self._operation.specification,
            input=message,
        )
        response = await self.agent(context)
        return agent_cmds.OrchestratorResponse(
            message,
            response,
            self.agent,
            self._operation,
            {},
            context,
            event_manager=self.event_manager,
            tool=self.tool,
        )


def make_args() -> Namespace:
    return Namespace(
        specifications_file=None,
        use_sync_generator=False,
        display_tokens=0,
        stats=True,
        id="aid",
        participant="pid",
        session="sid",
        no_session=False,
        skip_load_recent_messages=False,
        load_recent_messages_limit=1,
        no_repl=False,
        quiet=False,
        record=False,
        skip_hub_access_check=True,
        conversation=False,
        watch=False,
        tty=None,
        tool_events=2,
        tool=["math.calculator"],
        tool_format=None,
        run_max_new_tokens=1024,
        run_skip_special_tokens=False,
        engine_uri="NousResearch/Hermes-3-Llama-3.1-8B",
        name="Tool",
        role=(
            "You are a helpful assistant named Tool, that can resolve user"
            " requests using tools."
        ),
        task=None,
        instructions=None,
        memory_recent=True,
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
        display_events=True,
        display_tools=True,
        display_tools_events=2,
        tools_confirm=False,
    )


class AgentRunMathToolTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_cli_run_math_tool(self):
        args = make_args()
        console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm

        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">", "agent_output": "<"}
        theme.get_spinner.return_value = "sp"
        theme.agent.return_value = "agent_panel"
        theme.recent_messages.return_value = "recent_panel"
        hub = MagicMock()
        logger = MagicMock()

        orch = DummyOrchestrator()
        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)

        with (
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=orch),
            ),
            patch.object(
                agent_cmds.OrchestratorLoader, "from_file", new=AsyncMock()
            ),
            patch.object(
                agent_cmds,
                "get_input",
                return_value=(
                    "What is (4 + 6) and then that result times 5, divided"
                    " by 2?"
                ),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await agent_cmds.agent_run(args, console, theme, hub, logger, 1)

        tg_patch.assert_awaited_once()
        resp = tg_patch.await_args.kwargs["response"]
        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertTrue(
            any(
                t.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
                and t.data
                and t.data.get("result") == "25"
                for t in tokens
            )
        )
        self.assertTrue(
            any(
                t.kind is StreamItemKind.ANSWER_DELTA
                and t.text_delta
                and "25" in t.text_delta
                for t in tokens
            )
        )

    async def test_cli_run_math_tool_with_default_and_explicit_fancy_theme(
        self,
    ) -> None:
        rendered: dict[str, str] = {}
        for label, theme_name in {
            "omitted": DEFAULT_THEME_NAME,
            "explicit": "fancy",
        }.items():
            with self.subTest(label=label):
                args = make_args()
                args.display_tokens = 0
                console = MagicMock()
                console.width = 120
                console.is_terminal = True
                status_cm = MagicMock()
                status_cm.__enter__.return_value = None
                status_cm.__exit__.return_value = False
                console.status.return_value = status_cm
                live = MagicMock()
                live_cm = MagicMock()
                live_cm.__enter__.return_value = live
                live_cm.__exit__.return_value = False
                theme = create_theme(
                    theme_name,
                    lambda message: message,
                    lambda singular, plural, count: (
                        singular if count == 1 else plural
                    ),
                )
                hub = MagicMock()
                logger = MagicMock()

                orch = DummyOrchestrator()
                dummy_stack = AsyncMock()
                dummy_stack.__aenter__.return_value = dummy_stack
                dummy_stack.__aexit__.return_value = False
                dummy_stack.enter_async_context = AsyncMock(return_value=orch)

                with (
                    patch.object(
                        agent_cmds, "AsyncExitStack", return_value=dummy_stack
                    ),
                    patch.object(
                        agent_cmds.OrchestratorLoader,
                        "from_settings",
                        new=AsyncMock(return_value=orch),
                    ),
                    patch.object(
                        agent_cmds.OrchestratorLoader,
                        "from_file",
                        new=AsyncMock(),
                    ),
                    patch.object(
                        agent_cmds,
                        "get_input",
                        return_value=(
                            "What is (4 + 6) and then that result times 5,"
                            " divided by 2?"
                        ),
                    ),
                    patch(
                        "avalan.cli.stream_coordinator.Live",
                        return_value=live_cm,
                    ),
                ):
                    await agent_cmds.agent_run(
                        args, console, theme, hub, logger, 1
                    )

                render_console = Console(
                    file=StringIO(),
                    record=True,
                    width=160,
                )
                for update_call in live.update.call_args_list:
                    render_console.print(update_call.args[0])
                rendered[label] = render_console.export_text()

                self.assertIn("25", rendered[label])
                self.assertIn("Tool calls", rendered[label])
                self.assertIn("tool", rendered[label].lower())

        self.assertEqual(
            "Tool calls" in rendered["omitted"],
            "Tool calls" in rendered["explicit"],
        )
        self.assertEqual(
            "25" in rendered["omitted"],
            "25" in rendered["explicit"],
        )

    async def test_cli_run_math_tool_with_basic_theme_outputs_final_result(
        self,
    ) -> None:
        args = make_args()
        args.stats = False
        args.display_events = False
        args.display_tools = False
        args.display_tools_events = 0
        args.display_tokens = 0
        console = MagicMock()
        console.width = 120
        console.is_terminal = True
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm
        theme = create_theme(
            "basic",
            lambda message: message,
            lambda singular, plural, count: singular if count == 1 else plural,
        )
        hub = MagicMock()
        logger = MagicMock()

        orch = DummyOrchestrator()
        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)

        with (
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=orch),
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ),
            patch.object(
                agent_cmds,
                "get_input",
                return_value=(
                    "What is (4 + 6) and then that result times 5, divided"
                    " by 2?"
                ),
            ),
            patch("avalan.cli.stream_coordinator.Live") as live_patch,
        ):
            await agent_cmds.agent_run(args, console, theme, hub, logger, 1)

        live_patch.assert_not_called()
        answer_stdout = "".join(
            str(args[0])
            for args, kwargs in console.print.call_args_list
            if kwargs.get("end") == ""
        )
        self.assertRegex(answer_stdout, r'The result is "?25"?\.')
        self.assertNotIn("<tool_call", answer_stdout)
        self.assertNotIn("arguments", answer_stdout)
        self.assertNotIn("Tool calls", answer_stdout)

    async def test_cli_run_math_tool_display_tools_without_stats(self):
        args = make_args()
        args.stats = False
        args.display_events = False
        args.display_tools = True
        console = MagicMock()
        console.width = 80
        console.is_terminal = True
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm
        live = MagicMock()
        live_cm = MagicMock()
        live_cm.__enter__.return_value = live
        live_cm.__exit__.return_value = False

        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">", "agent_output": "<"}
        theme.get_spinner.return_value = "sp"
        theme.agent.return_value = "agent_panel"
        theme.recent_messages.return_value = "recent_panel"

        def events_side_effect(*_args, **kwargs):
            return "tool-panel" if kwargs["include_tools"] else None

        theme.events.side_effect = events_side_effect
        theme.token_frames.return_value = ()
        hub = MagicMock()
        logger = MagicMock()

        orch = DummyOrchestrator()
        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)

        with (
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=orch),
            ),
            patch.object(
                agent_cmds.OrchestratorLoader, "from_file", new=AsyncMock()
            ),
            patch.object(
                agent_cmds,
                "get_input",
                return_value=(
                    "What is (4 + 6) and then that result times 5, divided"
                    " by 2?"
                ),
            ),
            patch("avalan.cli.stream_coordinator.Live", return_value=live_cm),
        ):
            await agent_cmds.agent_run(args, console, theme, hub, logger, 1)

        theme.events.assert_not_called()
        theme.tokens.assert_not_called()
        live.update.assert_called()
        self.assertTrue(
            any(
                "tool" in str(call.args[0])
                for call in live.update.call_args_list
            )
        )
        theme.token_frames.assert_not_called()
        console.print.assert_any_call("< ", end="")

    async def test_cli_conversation_with_piped_input_exits_without_tty(self):
        args = make_args()
        args.conversation = True
        args.tty = "/dev/missing"
        console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm

        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">", "agent_output": "<"}
        theme.get_spinner.return_value = "sp"
        theme.agent.return_value = "agent_panel"
        theme.recent_messages.return_value = "recent_panel"
        hub = MagicMock()
        logger = MagicMock()
        stdin_mock = MagicMock()
        stdin_mock.read.return_value = (
            "What is (4 + 6) and then that result times 5, divided by 2?"
        )

        orch = DummyOrchestrator()
        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)

        with (
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=orch),
            ),
            patch.object(
                agent_cmds.OrchestratorLoader, "from_file", new=AsyncMock()
            ),
            patch("avalan.cli.has_input", side_effect=[True, True]),
            patch("avalan.cli.stdin", stdin_mock),
            patch(
                "avalan.cli.open", side_effect=OSError("no tty")
            ) as open_patch,
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await agent_cmds.agent_run(args, console, theme, hub, logger, 1)

        tg_patch.assert_awaited_once()
        open_patch.assert_called_once_with("/dev/missing")

    async def test_cli_run_math_tool_with_ds4_backend_uri(self):
        args = make_args()
        args.backend = "ds4"
        args.engine_uri = "ai://local/./ds4flash.gguf?backend=ds4&ds4_ctx=4096"
        console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        console.status.return_value = status_cm

        theme = MagicMock()
        theme._ = lambda s: s
        theme.icons = {"user_input": ">", "agent_output": "<"}
        theme.get_spinner.return_value = "sp"
        theme.agent.return_value = "agent_panel"
        theme.recent_messages.return_value = "recent_panel"
        hub = MagicMock()
        hub.can_access.side_effect = AssertionError(
            "DS4 local model paths must not query Hugging Face access."
        )
        hub.model.side_effect = AssertionError(
            "DS4 local model paths must not query Hugging Face metadata."
        )
        logger = MagicMock()

        orch = DummyOrchestrator(
            engine=DummyDs4Engine(),
            engine_uri=EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor="local",
                model_id="./ds4flash.gguf",
                params={"ds4_ctx": "4096"},
            ),
        )
        dummy_stack = AsyncMock()
        dummy_stack.__aenter__.return_value = dummy_stack
        dummy_stack.__aexit__.return_value = False
        dummy_stack.enter_async_context = AsyncMock(return_value=orch)

        async def from_settings(settings, **_kwargs):
            self.assertEqual(settings.uri, args.engine_uri)
            self.assertEqual(settings.engine_config, {"backend": "ds4"})
            self.assertEqual(settings.tools, ["math.calculator"])
            return orch

        with (
            patch.object(
                agent_cmds, "AsyncExitStack", return_value=dummy_stack
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(side_effect=from_settings),
            ) as from_settings_patch,
            patch.object(
                agent_cmds.OrchestratorLoader, "from_file", new=AsyncMock()
            ),
            patch.object(
                agent_cmds,
                "get_input",
                return_value=(
                    "What is (4 + 6) and then that result times 5, divided"
                    " by 2?"
                ),
            ),
            patch.object(
                agent_cmds, "token_generation", new_callable=AsyncMock
            ) as tg_patch,
        ):
            await agent_cmds.agent_run(args, console, theme, hub, logger, 1)

        from_settings_patch.assert_awaited_once()
        hub.can_access.assert_not_called()
        hub.model.assert_not_called()
        theme.agent.assert_called_once()
        self.assertEqual(theme.agent.call_args.kwargs["models"], ["m"])
        tg_patch.assert_awaited_once()
        resp = tg_patch.await_args.kwargs["response"]
        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertIs(DummyEngine.last_tool, orch.tool)
        self.assertTrue(
            any(
                t.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
                and t.data
                and t.data.get("name") == "math.calculator"
                and t.data.get("result") == "25"
                for t in tokens
            )
        )
        self.assertTrue(
            any(
                t.kind is StreamItemKind.ANSWER_DELTA
                and t.text_delta
                and "25" in t.text_delta
                for t in tokens
            )
        )


if __name__ == "__main__":
    unittest.main()
