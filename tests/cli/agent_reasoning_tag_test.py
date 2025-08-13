from avalan.cli.commands import agent as agent_cmds
from avalan.entities import GenerationSettings, ReasoningSettings, ReasoningTag
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.text import TextGenerationResponse
from argparse import Namespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from logging import getLogger


class CliAgentReasoningTagTestCase(IsolatedAsyncioTestCase):
    def setUp(self) -> None:  # type: ignore[override]
        self.args = Namespace(
            specifications_file="spec.toml",
            id="aid",
            participant="pid",
            session="sid",
            no_session=False,
            no_repl=False,
            quiet=False,
            skip_hub_access_check=False,
            tool_events=0,
            tool=None,
            run_max_new_tokens=100,
            backend="transformers",
            memory_recent=None,
            memory_permanent_message=None,
            memory_permanent=None,
            tty="/dev/tty",
            display_tokens=0,
            display_events=False,
            display_tools=False,
            display_tools_events=0,
            tools_confirm=False,
            use_sync_generator=False,
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
            stats=False,
            conversation=False,
            watch=False,
            skip_load_recent_messages=True,
            load_recent_messages_limit=None,
            start_thinking=False,
            chat_disable_thinking=False,
            reasoning_tag=None,
        )
        self.console = MagicMock()
        status_cm = MagicMock()
        status_cm.__enter__.return_value = None
        status_cm.__exit__.return_value = False
        self.console.status.return_value = status_cm
        self.theme = MagicMock()
        self.theme._ = lambda s: s
        self.theme.icons = {"user_input": ">", "agent_output": "<"}
        self.theme.agent.return_value = "agent_panel"
        self.hub = MagicMock()
        self.hub.can_access.return_value = True
        self.hub.model.side_effect = lambda m: f"mdl-{m}"
        self.logger = MagicMock()
        self.orch = AsyncMock()
        self.orch.engine_agent = True
        self.orch.engine = MagicMock(model_id="m")
        self.orch.model_ids = ["m"]
        self.orch.event_manager.add_listener = MagicMock()
        memory = MagicMock()
        memory.continue_session = AsyncMock()
        self.orch.memory = memory
        self.dummy_stack = AsyncMock()
        self.dummy_stack.__aenter__.return_value = self.dummy_stack
        self.dummy_stack.__aexit__.return_value = False
        self.dummy_stack.enter_async_context = AsyncMock(
            return_value=self.orch
        )

    async def _run(self, tag: ReasoningTag) -> tuple[str, str]:
        self.args.reasoning_tag = tag.value

        async def gen():
            yield "t"

        class DummyOrchestratorResponse:
            input_token_count = 1

            def __init__(self) -> None:
                settings = GenerationSettings(
                    reasoning=ReasoningSettings(tag=tag)
                )
                self._resp = TextGenerationResponse(
                    lambda **_: gen(),
                    logger=getLogger(),
                    use_async_generator=True,
                    generation_settings=settings,
                    settings=settings,
                )

            def __aiter__(self):
                return self._resp.__aiter__()

        recorded: dict[str, str] = {}
        orig_init = ReasoningParser.__init__

        def rec_init(
            self,
            *,
            reasoning_settings,
            logger,
            bos_token=None,
            start_tag=None,
            end_tag=None,
            prefixes=None,
            max_thinking_turns=1,
        ) -> None:
            orig_init(
                self,
                reasoning_settings=reasoning_settings,
                logger=logger,
                bos_token=bos_token,
                start_tag=start_tag,
                end_tag=end_tag,
                prefixes=prefixes,
                max_thinking_turns=max_thinking_turns,
            )
            recorded["start_tag"] = self._start_tag
            recorded["end_tag"] = self._end_tag

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
            patch.object(ReasoningParser, "__init__", rec_init),
        ):
            self.orch.side_effect = (
                lambda *a, **kw: DummyOrchestratorResponse()
            )
            await agent_cmds.agent_run(
                self.args, self.console, self.theme, self.hub, self.logger, 1
            )
        return recorded["start_tag"], recorded["end_tag"]

    async def test_think_tag(self):
        start, end = await self._run(ReasoningTag.THINK)
        self.assertEqual(start, "<think>")
        self.assertEqual(end, "</think>")

    async def test_channel_tag(self):
        start, end = await self._run(ReasoningTag.CHANNEL)
        self.assertEqual(start, "<|channel|>analysis<|message|>")
        self.assertEqual(end, "<|end|>")
