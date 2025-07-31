import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

import avalan.cli.commands.model as model_cmds


class _Resp:
    input_token_count = 1
    can_think = True
    is_thinking = False

    def __init__(self) -> None:
        self.thinking = None

    def set_thinking(self, value: bool) -> None:
        self.thinking = value

    def __aiter__(self):
        async def gen():
            yield model_cmds.Token(id=1, token="A")

        return gen()


class TokenStreamStartThinkingTestCase(IsolatedAsyncioTestCase):
    async def test_start_thinking(self):
        args = Namespace(
            skip_display_reasoning_time=False,
            display_time_to_n_token=None,
            display_pause=0,
            start_thinking=True,
            display_probabilities=False,
            display_probabilities_maximum=0.0,
            display_probabilities_sample_minimum=0.0,
            record=False,
        )
        console = MagicMock()
        console.width = 80
        live = MagicMock()
        logger = MagicMock()
        stop_signal = asyncio.Event()
        group = SimpleNamespace(renderables=[None])

        async def fake_frames(*_, **__):
            yield (None, "frame")

        theme = MagicMock()
        theme.tokens = MagicMock(side_effect=fake_frames)

        lm = SimpleNamespace(
            model_id="m", tokenizer_config=None, input_token_count=lambda s: 1
        )

        resp = _Resp()

        with patch("avalan.cli.commands.model.sleep", new=AsyncMock()):
            await model_cmds._token_stream(
                live=live,
                group=group,
                tokens_group_index=0,
                args=args,
                console=console,
                theme=theme,
                logger=logger,
                orchestrator=None,
                event_stats=None,
                lm=lm,
                input_string="hi",
                response=resp,
                display_tokens=1,
                dtokens_pick=1,
                refresh_per_second=1,
                stop_signal=stop_signal,
                tool_events_limit=None,
                with_stats=True,
            )

        self.assertTrue(stop_signal.is_set())
        self.assertTrue(resp.thinking)
