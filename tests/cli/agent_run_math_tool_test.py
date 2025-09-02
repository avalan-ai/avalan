import unittest
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.cli.commands import agent as agent_cmds
from avalan.agent import (
    Specification,
    EngineEnvironment,
    AgentOperation,
)
from avalan.agent.engine import EngineAgent
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
    Modality,
    Operation as EntitiesOperation,
    OperationParameters,
    OperationTextParameters,
)
from avalan.event import Event, EventType
from avalan.event.manager import EventManager
from avalan.model.response.text import TextGenerationResponse
from logging import getLogger
from avalan.tool.math import MathToolSet
from avalan.tool.manager import ToolManager, ToolManagerSettings


class DummyEngine:
    model_id = "m"
    model_type = "t"
    last_tool = None
    tokenizer = MagicMock()

    async def __call__(self, input, *, tool=None):
        DummyEngine.last_tool = tool
        if isinstance(input, Message):

            async def gen():
                yield "<tool_call>"
                yield (
                    '{"name": "math.calculator", "arguments": {"expression":'
                    ' "(4 + 6) * 5 / 2"}}'
                )
                yield "</tool_call>"

            return TextGenerationResponse(
                lambda: gen(), logger=getLogger(), use_async_generator=True
            )
        else:
            result = (
                input[-1].content if isinstance(input, list) else str(input)
            )

            async def gen():
                yield f"The result is {result}."

            return TextGenerationResponse(
                lambda: gen(), logger=getLogger(), use_async_generator=True
            )

    def input_token_count(self, *_a, **_k):
        return 0


class DummyModelManager:
    def __init__(self) -> None:
        self.passed_tool = None

    async def __call__(self, engine_uri, model, operation, tool):
        self.passed_tool = tool
        return await model(operation.input, tool=tool)


class DummyAgent(EngineAgent):
    def _prepare_call(self, specification, input, **kwargs):
        return {}

    async def _run(self, input, **_kwargs):
        operation = EntitiesOperation(
            generation_settings=GenerationSettings(),
            input=input,
            modality=Modality.TEXT_GENERATION,
            parameters=OperationParameters(text=OperationTextParameters()),
            requires_input=True,
        )
        self._last_operation = operation
        return await self._model_manager(
            self._engine_uri,
            self._model,
            operation,
            self._tool,
        )


class DummyOrchestrator:
    def __init__(self):
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
        self.engine_uri = EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        )
        self.engine = DummyEngine()
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
        response = await self.agent(
            self._operation.specification,
            message,
        )
        return agent_cmds.OrchestratorResponse(
            message,
            response,
            self.agent,
            self._operation,
            {},
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
                isinstance(t, Event)
                and t.type == EventType.TOOL_RESULT
                and t.payload["result"].result == "25"
                for t in tokens
            )
        )
        self.assertTrue(any(isinstance(t, str) and "25" in t for t in tokens))


if __name__ == "__main__":
    unittest.main()
