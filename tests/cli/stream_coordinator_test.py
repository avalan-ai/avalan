from asyncio import CancelledError
from datetime import datetime, timezone
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

from rich.console import Group

from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.stream_coordinator import (
    CliStreamCoordinator,
    _default_live_factory,
    stream_recording_filename,
)
from avalan.cli.stream_presenter import (
    CliStreamAnswerTextChunk,
    CliStreamRenderableFrame,
)
from avalan.entities import ToolCall


class _FakeLive:
    def __init__(self, events: list[tuple[str, object]] | None = None) -> None:
        self.auto_refresh = True
        self.entered = 0
        self.exited = 0
        self.refreshed = 0
        self.updates: list[object] = []
        self._events = events

    def __enter__(self) -> "_FakeLive":
        self.entered += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> bool:
        self.exited += 1
        if self._events is not None:
            self._events.append(("exit", exc_value))
        return False

    def refresh(self) -> None:
        self.refreshed += 1

    def update(self, renderable: object) -> None:
        self.updates.append(renderable)
        if self._events is not None:
            self._events.append(("update", renderable))


def _display_config(
    *,
    quiet: bool = False,
    stats: bool = True,
    display_tools: bool = False,
    display_events: bool = False,
    record: bool = False,
    interactive: bool = True,
) -> CliStreamDisplayConfig:
    return CliStreamDisplayConfig(
        quiet=quiet,
        stats=stats,
        display_tools=display_tools,
        display_events=display_events,
        display_tools_events=2,
        record=record,
        interactive=interactive,
        refresh_per_second=5,
        answer_height=12,
        answer_height_expand=False,
        display_tokens=0,
        display_pause=0,
        display_probabilities=False,
        display_probabilities_maximum=0.8,
        display_probabilities_sample_minimum=0.1,
        display_time_to_n_token=None,
        display_reasoning_time=True,
    )


class CliStreamCoordinatorTestCase(IsolatedAsyncioTestCase):
    async def test_live_roles_share_one_owner(self) -> None:
        fake_live = _FakeLive()
        live_factory = MagicMock(return_value=fake_live)
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(display_tools=True, display_events=True),
            live_factory=live_factory,
        )

        async with coordinator:
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="token", role="stream")
            )
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="event", role="events")
            )
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="tool", role="tools")
            )
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="stats", role="stats")
            )

        live_factory.assert_called_once()
        self.assertEqual(fake_live.entered, 1)
        self.assertEqual(fake_live.exited, 1)
        self.assertIsInstance(fake_live.updates[-1], Group)
        self.assertEqual(
            list(cast(Group, fake_live.updates[-1]).renderables),
            ["event", "tool", "stats", "token"],
        )

    async def test_answer_only_prints_without_live(self) -> None:
        console = MagicMock()
        live_factory = MagicMock()
        coordinator = CliStreamCoordinator(
            console,
            _display_config(quiet=True, stats=False, record=True),
            live_factory=live_factory,
        )

        async with coordinator:
            await coordinator.handle_item(CliStreamAnswerTextChunk(text="a"))
            await coordinator.print_answer_chunk(
                CliStreamAnswerTextChunk(text="b")
            )
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="ignored")
            )

        console.print.assert_has_calls([call("a", end=""), call("b", end="")])
        live_factory.assert_not_called()
        console.save_svg.assert_not_called()

    async def test_recording_saves_after_owner_render(self) -> None:
        events: list[tuple[str, object]] = []
        fake_live = _FakeLive(events)
        console = MagicMock()
        console.save_svg.side_effect = lambda filename, clear: events.append(
            (
                "save",
                (filename, clear),
            )
        )
        coordinator = CliStreamCoordinator(
            console,
            _display_config(record=True),
            live_factory=MagicMock(return_value=fake_live),
            record_filename_factory=lambda: "frame.svg",
        )

        async with coordinator:
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="frame")
            )

        self.assertEqual(
            events[:2],
            [("update", "frame"), ("save", ("frame.svg", True))],
        )
        console.save_svg.assert_called_once_with("frame.svg", clear=True)

    async def test_recorder_error_closes_owner(self) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        console.save_svg.side_effect = RuntimeError("record failed")
        coordinator = CliStreamCoordinator(
            console,
            _display_config(record=True),
            live_factory=MagicMock(return_value=fake_live),
        )

        with self.assertRaisesRegex(RuntimeError, "record failed"):
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="frame")
            )

        self.assertEqual(fake_live.exited, 1)
        with self.assertRaises(AssertionError):
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="later")
            )

    async def test_resume_recording_error_closes_owner(self) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        console.save_svg.side_effect = RuntimeError("resume record failed")
        coordinator = CliStreamCoordinator(
            console,
            _display_config(record=True),
            live_factory=MagicMock(return_value=fake_live),
        )

        await coordinator.pause()
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="queued")
        )
        with self.assertRaisesRegex(RuntimeError, "resume record failed"):
            await coordinator.resume()

        self.assertEqual(fake_live.exited, 1)

    async def test_pause_queues_latest_frame_until_resume(self) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        async with coordinator:
            await coordinator.pause()
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="first")
            )
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="latest")
            )
            await coordinator.flush()
            self.assertEqual(fake_live.updates, [])
            await coordinator.resume()
            await coordinator.resume()

        self.assertEqual(fake_live.updates, ["latest"])

    async def test_nested_pause_resumes_after_outer_prompt(self) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        async with coordinator:
            await coordinator.pause()
            await coordinator.pause()
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="queued")
            )
            await coordinator.resume()
            self.assertEqual(fake_live.updates, [])
            await coordinator.resume()

        self.assertEqual(fake_live.updates, ["queued"])

    async def test_paused_context_closes_on_prompt_error(self) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="x")
        )
        with self.assertRaisesRegex(RuntimeError, "prompt failed"):
            async with coordinator.paused():
                raise RuntimeError("prompt failed")

        self.assertEqual(fake_live.exited, 1)

    async def test_confirm_tool_call_pauses_live_refresh(self) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="x")
        )
        seen_paused: list[bool] = []

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call)
            seen_paused.append(not fake_live.auto_refresh)
            self.assertEqual(tty_path, "/tmp/tty")
            return "y"

        result = await coordinator.confirm_tool_call(
            ToolCall(id=uuid4(), name="calc", arguments={"x": 1}),
            tty_path="/tmp/tty",
            prompt=prompt,
        )

        self.assertEqual(result, "y")
        self.assertEqual(seen_paused, [True])
        self.assertTrue(fake_live.auto_refresh)
        self.assertEqual(fake_live.refreshed, 2)
        await coordinator.aclose()

    async def test_cancelled_error_closes_live_without_final_flush(
        self,
    ) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        with self.assertRaises(CancelledError):
            async with coordinator:
                await coordinator.handle_item(
                    CliStreamRenderableFrame(renderable="frame")
                )
                raise CancelledError()

        self.assertEqual(fake_live.updates, ["frame"])
        self.assertEqual(fake_live.exited, 1)

    async def test_keyboard_interrupt_closes_live_without_final_flush(
        self,
    ) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        with self.assertRaises(KeyboardInterrupt):
            async with coordinator:
                await coordinator.handle_item(
                    CliStreamRenderableFrame(renderable="frame")
                )
                raise KeyboardInterrupt()

        self.assertEqual(fake_live.exited, 1)

    async def test_unknown_presenter_item_closes_owner(self) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="x")
        )
        with self.assertRaises(AssertionError):
            await coordinator.handle_item(cast(Any, object()))

        self.assertEqual(fake_live.exited, 1)

    async def test_close_is_idempotent(self) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="x")
        )
        await coordinator.aclose()
        await coordinator.aclose()

        self.assertEqual(fake_live.exited, 1)

    async def test_flush_without_pending_frame_does_not_start_live(
        self,
    ) -> None:
        live_factory = MagicMock()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=live_factory,
        )

        await coordinator.flush()

        live_factory.assert_not_called()


class CliStreamCoordinatorHelperTestCase(TestCase):
    def test_default_live_factory_delegates_to_rich_live(self) -> None:
        console = MagicMock()
        live = MagicMock()
        with patch(
            "avalan.cli.stream_coordinator.Live", return_value=live
        ) as live_type:
            result = _default_live_factory(
                "frame",
                console=console,
                refresh_per_second=3,
                screen=True,
            )

        self.assertIs(result, live)
        live_type.assert_called_once_with(
            "frame",
            console=console,
            refresh_per_second=3,
            screen=True,
        )

    def test_stream_recording_filename_uses_utc_timestamp(self) -> None:
        dt_value = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
        with patch("avalan.cli.stream_coordinator.datetime") as dt_patch:
            dt_patch.now.return_value = dt_value
            filename = stream_recording_filename()

        self.assertEqual(filename, "avalan-screenshot-20240102030405-123.svg")
