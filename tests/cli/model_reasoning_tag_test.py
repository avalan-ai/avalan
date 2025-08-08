from avalan.cli.commands import model as model_cmds
from avalan.entities import Modality, ReasoningTag
from avalan.model.manager import ModelManager as RealModelManager
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.text import TextGenerationResponse
from types import SimpleNamespace
from argparse import Namespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from logging import getLogger


class CliModelReasoningTagTestCase(IsolatedAsyncioTestCase):
    async def _run(self, tag: ReasoningTag) -> tuple[str, str]:
        args = Namespace(
            skip_display_reasoning_time=False,
            model="id",
            device="cpu",
            max_new_tokens=2,
            quiet=True,
            skip_hub_access_check=False,
            no_repl=False,
            do_sample=True,
            enable_gradient_calculation=True,
            min_p=0.1,
            repetition_penalty=1.0,
            temperature=0.5,
            top_k=5,
            top_p=0.9,
            use_cache=False,
            stop_on_keyword=None,
            system=None,
            skip_special_tokens=False,
            display_tokens=0,
            tool_events=0,
            display_events=False,
            display_tools=False,
            display_tools_events=0,
            start_thinking=False,
            chat_disable_thinking=False,
            reasoning_tag=tag.value,
        )
        console = MagicMock()
        theme = MagicMock()
        hub = MagicMock()
        logger = MagicMock()
        engine_uri = SimpleNamespace(model_id="id", is_local=True)
        load_cm = MagicMock()
        load_cm.__enter__.return_value = MagicMock()
        load_cm.__exit__.return_value = False
        manager = RealModelManager(hub, logger)
        manager.parse_uri = MagicMock(return_value=engine_uri)
        manager.load = MagicMock(return_value=load_cm)

        async def manager_call(
            self, _engine_uri, _model, operation, tool=None
        ):
            async def gen():
                yield "t"

            return TextGenerationResponse(
                lambda **_: gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=operation.generation_settings,
                settings=operation.generation_settings,
            )

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
            patch.object(model_cmds, "ModelManager", return_value=manager),
            patch.object(
                model_cmds.ModelManager,
                "get_operation_from_arguments",
                side_effect=RealModelManager.get_operation_from_arguments,
            ),
            patch.object(RealModelManager, "__call__", manager_call),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ),
            patch.object(model_cmds, "get_input", return_value="hi"),
            patch.object(
                model_cmds, "token_generation", new_callable=AsyncMock
            ),
            patch.object(ReasoningParser, "__init__", rec_init),
        ):
            await model_cmds.model_run(args, console, theme, hub, 5, logger)
        return recorded["start_tag"], recorded["end_tag"]

    async def test_think_tag(self):
        start, end = await self._run(ReasoningTag.THINK)
        self.assertEqual(start, "<think>")
        self.assertEqual(end, "</think>")

    async def test_channel_tag(self):
        start, end = await self._run(ReasoningTag.CHANNEL)
        self.assertEqual(start, "<|channel|>analysis<|message|>")
        self.assertEqual(
            end,
            "<|end|><|start|>assistant<|channel|>final<|message|>",
        )
