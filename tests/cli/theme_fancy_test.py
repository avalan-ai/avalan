from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch, MagicMock
from rich.spinner import Spinner

from avalan.cli.theme.fancy import FancyTheme
from avalan.event import Event, EventType


class FancyThemeTokensTestCase(IsolatedAsyncioTestCase):
    async def test_tool_running_spinner_text(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        spinner = Spinner("dots", text="[cyan]run[/cyan]", style="cyan")
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            gen = theme.tokens(
                model_id="m",
                added_tokens=None,
                special_tokens=None,
                display_token_size=None,
                display_probabilities=False,
                pick=0,
                focus_on_token_when=None,
                text_tokens=["a"],
                tokens=None,
                input_token_count=0,
                total_tokens=0,
                tool_events=[],
                tool_event_calls=[
                    Event(type=EventType.TOOL_PROCESS, payload=[])
                ],
                tool_event_results=[],
                tool_running_spinner=spinner,
                ttft=0.0,
                ttnt=0.0,
                ellapsed=1.0,
                console_width=80,
                logger=MagicMock(),
            )
            frame = await gen.__anext__()
        self.assertTrue(
            any(
                getattr(r, "renderable", None) is spinner
                for r in frame[1].renderables
            )
        )
        self.assertIn("second", spinner.text)
