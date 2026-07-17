from asyncio import CancelledError, create_task, sleep
from asyncio import Event as AsyncEvent
from datetime import datetime, timezone
from io import StringIO
from threading import Event
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

from rich.console import Console, Group
from rich.spinner import Spinner

from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.stream_coordinator import (
    CliStreamCoordinator,
    _default_live_factory,
    _TailOverflowLive,
    _TailOverflowRenderable,
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
        self.exit_auto_refreshes: list[bool] = []
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
        self.exit_auto_refreshes.append(self.auto_refresh)
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
    display_reasoning: bool = False,
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
        display_reasoning=display_reasoning,
    )


class CliStreamCoordinatorTestCase(IsolatedAsyncioTestCase):
    async def _wait_until_set(self, event: Event) -> None:
        for _ in range(100):
            if event.is_set():
                return
            await sleep(0.01)
        self.fail("timed out waiting for prompt")

    async def _assert_confirm_choice_flushes_latest_frame(
        self,
        choice: str,
    ) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        coordinator = CliStreamCoordinator(
            console,
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )
        started = Event()
        release = Event()
        seen_auto_refresh: list[bool] = []

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = console
            self.assertEqual(call.name, "calc")
            self.assertEqual(tty_path, "/tmp/tty")
            seen_auto_refresh.append(fake_live.auto_refresh)
            started.set()
            self.assertTrue(release.wait(2.0))
            return choice

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="initial")
        )
        task = create_task(
            coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={"x": 1}),
                tty_path="/tmp/tty",
                prompt=prompt,
            )
        )
        await self._wait_until_set(started)

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="queued")
        )
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="latest")
        )
        await coordinator.flush()

        self.assertEqual(fake_live.updates, ["initial"])
        console.save_svg.assert_not_called()

        release.set()
        self.assertEqual(await task, choice)

        self.assertEqual(fake_live.updates, ["initial", "latest"])
        self.assertEqual(seen_auto_refresh, [False])
        self.assertTrue(fake_live.auto_refresh)
        self.assertEqual(fake_live.refreshed, 2)
        await coordinator.aclose()

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
                CliStreamRenderableFrame(
                    renderable="reasoning",
                    role="reasoning",
                )
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
            ["event", "reasoning", "tool", "stats", "token"],
        )

    async def test_gate_coalesces_mixed_roles_and_records_flushes(
        self,
    ) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        now = 0.0

        def clock() -> float:
            return now

        coordinator = CliStreamCoordinator(
            console,
            _display_config(
                display_tools=True,
                display_events=True,
                record=True,
            ),
            live_factory=MagicMock(return_value=fake_live),
            record_filename_factory=lambda: "frame.svg",
            clock=clock,
        )

        async with coordinator:
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="stream-1", role="stream")
            )
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="event-1", role="events")
            )
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="stats-1", role="stats")
            )

            self.assertEqual(fake_live.updates, ["stream-1"])
            console.save_svg.assert_called_once_with("frame.svg", clear=True)

            now = 0.1
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="tool-1", role="tools")
            )
            self.assertEqual(fake_live.updates, ["stream-1"])

            now = 0.2
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="stream-2", role="stream")
            )

            self.assertEqual(len(fake_live.updates), 2)
            self.assertIsInstance(fake_live.updates[-1], Group)
            self.assertEqual(
                list(cast(Group, fake_live.updates[-1]).renderables),
                ["event-1", "tool-1", "stats-1", "stream-2"],
            )
            self.assertEqual(console.save_svg.call_count, 2)

            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="event-2", role="events")
            )
            self.assertEqual(len(fake_live.updates), 2)

            await coordinator.flush()

        self.assertEqual(len(fake_live.updates), 3)
        self.assertEqual(console.save_svg.call_count, 3)
        self.assertIsInstance(fake_live.updates[-1], Group)
        self.assertEqual(
            list(cast(Group, fake_live.updates[-1]).renderables),
            ["event-2", "tool-1", "stats-1", "stream-2"],
        )

    async def test_close_force_flushes_pending_frame_inside_gate(self) -> None:
        fake_live = _FakeLive()
        now = 0.0

        def clock() -> float:
            return now

        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
            clock=clock,
        )

        async with coordinator:
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="first")
            )
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="final")
            )

        self.assertEqual(fake_live.updates, ["first", "final"])

    async def test_close_without_flush_drops_pending_frame_inside_gate(
        self,
    ) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        now = 0.0

        def clock() -> float:
            return now

        coordinator = CliStreamCoordinator(
            console,
            _display_config(record=True),
            live_factory=MagicMock(return_value=fake_live),
            record_filename_factory=lambda: "frame.svg",
            clock=clock,
        )

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="first")
        )
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="pending")
        )
        await coordinator.aclose(flush=False)

        self.assertEqual(fake_live.updates, ["first"])
        console.save_svg.assert_called_once_with("frame.svg", clear=True)

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

    async def test_answer_chunk_closes_active_live_before_printing(
        self,
    ) -> None:
        events: list[tuple[str, object]] = []
        fake_live = _FakeLive(events)
        console = MagicMock()
        console.print.side_effect = lambda text, end="": events.append(
            (
                "print",
                (text, end),
            )
        )
        coordinator = CliStreamCoordinator(
            console,
            _display_config(display_tools=True),
            live_factory=MagicMock(return_value=fake_live),
        )

        async with coordinator:
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="tool", role="tools")
            )
            await coordinator.handle_item(CliStreamAnswerTextChunk(text="a"))

        self.assertEqual(
            events,
            [
                ("update", "tool"),
                ("exit", None),
                ("print", ("a", "")),
            ],
        )

    async def test_stderr_diagnostics_use_separate_console_without_live(
        self,
    ) -> None:
        console = MagicMock()
        diagnostic_console = MagicMock()
        live_factory = MagicMock()
        coordinator = CliStreamCoordinator(
            console,
            _display_config(
                display_events=True,
                display_tools=True,
                interactive=False,
                record=True,
                stats=True,
            ),
            diagnostic_console=diagnostic_console,
            live_factory=live_factory,
        )

        async with coordinator:
            await coordinator.handle_item(CliStreamAnswerTextChunk(text="a"))
            await coordinator.handle_item(
                CliStreamRenderableFrame(
                    renderable="event start",
                    role="events",
                )
            )
            await coordinator.handle_item(
                CliStreamRenderableFrame(
                    renderable="event start",
                    role="events",
                )
            )
            await coordinator.handle_item(
                CliStreamRenderableFrame(
                    renderable="tool calc completed",
                    role="tools",
                )
            )

        console.print.assert_called_once_with("a", end="")
        diagnostic_console.print.assert_has_calls(
            [
                call("event start"),
                call("tool calc completed"),
            ]
        )
        self.assertEqual(diagnostic_console.print.call_count, 2)
        live_factory.assert_not_called()
        console.save_svg.assert_not_called()

    async def test_stderr_diagnostics_deduplicate_group_renderables(
        self,
    ) -> None:
        console = MagicMock()
        diagnostic_console = MagicMock()
        coordinator = CliStreamCoordinator(
            console,
            _display_config(
                display_tools=True,
                interactive=False,
            ),
            diagnostic_console=diagnostic_console,
            live_factory=MagicMock(),
        )

        async with coordinator:
            await coordinator.handle_item(
                CliStreamRenderableFrame(
                    renderable=Group(
                        Spinner("dots", text="tool calc running"),
                        "tool calc completed",
                    ),
                    role="tools",
                )
            )
            await coordinator.handle_item(
                CliStreamRenderableFrame(
                    renderable=Group(
                        Spinner("dots", text="tool calc running"),
                        "tool calc completed",
                    ),
                    role="tools",
                )
            )

        diagnostic_console.print.assert_called_once()
        console.print.assert_not_called()

    async def test_stderr_reasoning_append_preserves_repeated_suffixes(
        self,
    ) -> None:
        diagnostic_console = MagicMock()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(
                stats=False,
                display_reasoning=True,
                interactive=False,
            ),
            diagnostic_console=diagnostic_console,
            live_factory=MagicMock(),
        )

        async with coordinator:
            for suffix in ("Reasoning summary:\nx", "x", "\n"):
                await coordinator.handle_item(
                    CliStreamRenderableFrame(
                        renderable=suffix,
                        role="reasoning",
                        stderr_append=True,
                    )
                )

        diagnostic_console.print.assert_has_calls(
            [
                call("Reasoning summary:\nx", end=""),
                call("x", end=""),
                call("\n", end=""),
            ]
        )
        self.assertEqual(diagnostic_console.print.call_count, 3)

    async def test_stderr_empty_frame_clears_deduplication_key(self) -> None:
        console = MagicMock()
        diagnostic_console = MagicMock()
        coordinator = CliStreamCoordinator(
            console,
            _display_config(
                display_tools=True,
                interactive=False,
            ),
            diagnostic_console=diagnostic_console,
            live_factory=MagicMock(),
        )

        async with coordinator:
            await coordinator.handle_item(
                CliStreamRenderableFrame(
                    renderable="tool calc completed",
                    role="tools",
                )
            )
            await coordinator.handle_item(
                CliStreamRenderableFrame(renderable="", role="tools")
            )
            await coordinator.handle_item(
                CliStreamRenderableFrame(
                    renderable="tool calc completed",
                    role="tools",
                )
            )

        diagnostic_console.print.assert_has_calls(
            [
                call("tool calc completed"),
                call("tool calc completed"),
            ]
        )
        self.assertEqual(diagnostic_console.print.call_count, 2)
        console.print.assert_not_called()

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

    async def test_answer_chunks_do_not_rerecord_closed_reasoning_role(
        self,
    ) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        coordinator = CliStreamCoordinator(
            console,
            _display_config(
                stats=False,
                display_reasoning=True,
                record=True,
            ),
            live_factory=MagicMock(return_value=fake_live),
            record_filename_factory=lambda: "reasoning.svg",
        )

        async with coordinator:
            await coordinator.handle_item(
                CliStreamRenderableFrame(
                    renderable="Reasoning: plan",
                    role="reasoning",
                )
            )
            await coordinator.handle_item(CliStreamAnswerTextChunk(text="a"))
            await coordinator.handle_item(CliStreamAnswerTextChunk(text="b"))

        self.assertEqual(fake_live.updates, ["Reasoning: plan"])
        self.assertEqual(fake_live.entered, 1)
        self.assertEqual(fake_live.exited, 1)
        console.save_svg.assert_called_once_with(
            "reasoning.svg",
            clear=True,
        )
        console.print.assert_has_calls(
            [
                call("a", end=""),
                call("b", end=""),
            ]
        )

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

    async def test_recording_waits_for_successful_resume(self) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        live_factory = MagicMock(return_value=fake_live)
        coordinator = CliStreamCoordinator(
            console,
            _display_config(record=True),
            live_factory=live_factory,
            record_filename_factory=lambda: "frame.svg",
        )

        await coordinator.pause()
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="first")
        )
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="latest")
        )

        self.assertEqual(fake_live.updates, [])
        live_factory.assert_not_called()
        console.save_svg.assert_not_called()

        await coordinator.resume()

        self.assertEqual(fake_live.updates, ["latest"])
        live_factory.assert_called_once()
        console.save_svg.assert_called_once_with("frame.svg", clear=True)
        await coordinator.aclose()
        console.save_svg.assert_called_once_with("frame.svg", clear=True)

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

    async def test_paused_context_success_resumes_and_flushes(self) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        async with coordinator.paused():
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="queued")
            )
            await coordinator.flush()
            self.assertEqual(fake_live.updates, [])

        self.assertEqual(fake_live.updates, ["queued"])
        await coordinator.aclose()

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

    async def test_confirm_tool_call_accept_flushes_latest_frame(self) -> None:
        await self._assert_confirm_choice_flushes_latest_frame("y")

    async def test_confirm_tool_call_decline_flushes_latest_frame(
        self,
    ) -> None:
        await self._assert_confirm_choice_flushes_latest_frame("n")

    async def test_confirm_tool_call_accept_all_flushes_latest_frame(
        self,
    ) -> None:
        await self._assert_confirm_choice_flushes_latest_frame("a")

    async def test_confirm_tool_call_before_live_does_not_create_owner(
        self,
    ) -> None:
        live_factory = MagicMock()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=live_factory,
        )

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            return "y"

        result = await coordinator.confirm_tool_call(
            ToolCall(id=uuid4(), name="calc", arguments={}),
            prompt=prompt,
        )

        self.assertEqual(result, "y")
        live_factory.assert_not_called()

    async def test_confirm_tool_call_respects_active_manual_pause(
        self,
    ) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )
        started = Event()
        release = Event()

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            started.set()
            self.assertTrue(release.wait(2.0))
            return "y"

        await coordinator.pause()
        task = create_task(
            coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={}),
                prompt=prompt,
            )
        )
        await self._wait_until_set(started)
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="queued")
        )

        release.set()
        self.assertEqual(await task, "y")
        self.assertEqual(fake_live.updates, [])

        await coordinator.resume()

        self.assertEqual(fake_live.updates, ["queued"])
        await coordinator.aclose()

    async def test_confirm_tool_call_records_once_after_prompt_resume(
        self,
    ) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        live_factory = MagicMock(return_value=fake_live)
        coordinator = CliStreamCoordinator(
            console,
            _display_config(record=True),
            live_factory=live_factory,
            record_filename_factory=lambda: "prompt.svg",
        )
        started = Event()
        release = Event()

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            started.set()
            self.assertTrue(release.wait(2.0))
            return "y"

        task = create_task(
            coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={}),
                prompt=prompt,
            )
        )
        await self._wait_until_set(started)
        live_factory.assert_not_called()

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="first")
        )
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="latest")
        )

        live_factory.assert_not_called()
        console.save_svg.assert_not_called()

        release.set()
        self.assertEqual(await task, "y")

        live_factory.assert_called_once()
        self.assertEqual(fake_live.updates, ["latest"])
        console.save_svg.assert_called_once_with("prompt.svg", clear=True)
        await coordinator.aclose()

    async def test_confirm_tool_call_closed_during_prompt_does_not_resume(
        self,
    ) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )
        started = Event()
        release = Event()

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            started.set()
            self.assertTrue(release.wait(2.0))
            return "y"

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="initial")
        )
        task = create_task(
            coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={}),
                prompt=prompt,
            )
        )
        await self._wait_until_set(started)
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="queued")
        )
        await coordinator.aclose(flush=False)

        release.set()

        self.assertEqual(await task, "y")
        self.assertEqual(fake_live.updates, ["initial"])
        self.assertEqual(fake_live.exited, 1)

    async def test_confirm_tool_call_prompt_error_closes_without_flush(
        self,
    ) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )
        started = Event()
        release = Event()

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            started.set()
            self.assertTrue(release.wait(2.0))
            raise RuntimeError("prompt failed")

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="initial")
        )
        task = create_task(
            coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={}),
                prompt=prompt,
            )
        )
        await self._wait_until_set(started)
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="queued")
        )

        self.assertEqual(fake_live.updates, ["initial"])

        release.set()
        with self.assertRaisesRegex(RuntimeError, "prompt failed"):
            await task

        self.assertEqual(fake_live.updates, ["initial"])
        self.assertEqual(fake_live.exited, 1)
        self.assertTrue(fake_live.auto_refresh)
        self.assertEqual(fake_live.exit_auto_refreshes, [True])

    async def test_confirm_tool_call_prompt_error_restores_disabled_refresh(
        self,
    ) -> None:
        fake_live = _FakeLive()
        fake_live.auto_refresh = False
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            raise RuntimeError("prompt failed")

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="initial")
        )
        with self.assertRaisesRegex(RuntimeError, "prompt failed"):
            await coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={}),
                prompt=prompt,
            )

        self.assertFalse(fake_live.auto_refresh)
        self.assertEqual(fake_live.exit_auto_refreshes, [False])
        with self.assertRaises(AssertionError):
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="later")
            )

    async def test_confirm_tool_call_resume_error_closes_owner(self) -> None:
        fake_live = _FakeLive()
        console = MagicMock()
        console.save_svg.side_effect = RuntimeError("record failed")
        coordinator = CliStreamCoordinator(
            console,
            _display_config(record=True),
            live_factory=MagicMock(return_value=fake_live),
        )
        started = Event()
        release = Event()

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            started.set()
            self.assertTrue(release.wait(2.0))
            return "y"

        task = create_task(
            coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={}),
                prompt=prompt,
            )
        )
        await self._wait_until_set(started)
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="queued")
        )

        release.set()
        with self.assertRaisesRegex(RuntimeError, "record failed"):
            await task

        self.assertEqual(fake_live.updates, ["queued"])
        self.assertEqual(fake_live.exited, 1)
        with self.assertRaises(AssertionError):
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="later")
            )

    async def test_confirm_tool_call_cancellation_closes_without_flush(
        self,
    ) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )
        started = Event()
        release = Event()

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            started.set()
            release.wait(2.0)
            return "y"

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="initial")
        )
        task = create_task(
            coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={}),
                prompt=prompt,
            )
        )
        await self._wait_until_set(started)
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="queued")
        )
        self.assertEqual(fake_live.updates, ["initial"])

        task.cancel()
        try:
            with self.assertRaises(CancelledError):
                await task
        finally:
            release.set()

        self.assertEqual(fake_live.updates, ["initial"])
        self.assertEqual(fake_live.exited, 1)
        self.assertTrue(fake_live.auto_refresh)
        self.assertEqual(fake_live.exit_auto_refreshes, [True])
        with self.assertRaises(AssertionError):
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="later")
            )

    async def test_confirm_tool_call_cancelled_during_prompt_resume_closes(
        self,
    ) -> None:
        fake_live = _FakeLive()
        resume_started = AsyncEvent()
        resume_release = AsyncEvent()

        class BlockingResumeCoordinator(CliStreamCoordinator):
            async def _resume_prompt(self) -> None:
                resume_started.set()
                await resume_release.wait()

        coordinator = BlockingResumeCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )
        started = Event()
        release = Event()

        def prompt(
            console: object,
            call: ToolCall,
            *,
            tty_path: str,
        ) -> str:
            _ = (console, call, tty_path)
            started.set()
            self.assertTrue(release.wait(2.0))
            return "y"

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="initial")
        )
        task = create_task(
            coordinator.confirm_tool_call(
                ToolCall(id=uuid4(), name="calc", arguments={}),
                prompt=prompt,
            )
        )
        await self._wait_until_set(started)
        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="queued")
        )

        release.set()
        await resume_started.wait()
        task.cancel()

        try:
            with self.assertRaises(CancelledError):
                await task
        finally:
            resume_release.set()

        self.assertEqual(fake_live.updates, ["initial"])
        self.assertEqual(fake_live.exited, 1)
        self.assertTrue(fake_live.auto_refresh)
        self.assertEqual(fake_live.exit_auto_refreshes, [True])
        with self.assertRaises(AssertionError):
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable="later")
            )

    async def test_consecutive_tool_prompts_do_not_leak_state(self) -> None:
        fake_live = _FakeLive()
        coordinator = CliStreamCoordinator(
            MagicMock(),
            _display_config(),
            live_factory=MagicMock(return_value=fake_live),
        )
        seen_auto_refresh: list[bool] = []

        async def confirm_with_queued_frame(
            choice: str,
            renderable: str,
        ) -> str:
            started = Event()
            release = Event()

            def prompt(
                console: object,
                call: ToolCall,
                *,
                tty_path: str,
            ) -> str:
                _ = (console, call, tty_path)
                seen_auto_refresh.append(fake_live.auto_refresh)
                started.set()
                self.assertTrue(release.wait(2.0))
                return choice

            task = create_task(
                coordinator.confirm_tool_call(
                    ToolCall(id=uuid4(), name="calc", arguments={}),
                    prompt=prompt,
                )
            )
            await self._wait_until_set(started)
            await coordinator.render_frame(
                CliStreamRenderableFrame(renderable=renderable)
            )
            release.set()
            return await task

        await coordinator.render_frame(
            CliStreamRenderableFrame(renderable="initial")
        )

        self.assertEqual(
            await confirm_with_queued_frame("y", "first"),
            "y",
        )
        self.assertEqual(fake_live.updates, ["initial", "first"])
        self.assertTrue(fake_live.auto_refresh)

        self.assertEqual(
            await confirm_with_queued_frame("n", "second"),
            "n",
        )
        await coordinator.flush()

        self.assertEqual(fake_live.updates, ["initial", "first", "second"])
        self.assertEqual(seen_auto_refresh, [False, False])
        self.assertTrue(fake_live.auto_refresh)
        self.assertEqual(fake_live.refreshed, 4)
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
    def test_tail_overflow_renderable_keeps_latest_rows(self) -> None:
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=False,
            width=20,
            height=3,
        )

        console.print(
            _TailOverflowRenderable(
                Group("line 0", "line 1", "line 2", "line 3")
            )
        )

        self.assertEqual(
            [line.strip() for line in output.getvalue().splitlines()],
            ["...", "line 2", "line 3"],
        )

    def test_tail_overflow_renderable_uses_latest_row_at_height_one(
        self,
    ) -> None:
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=False,
            width=20,
            height=1,
        )

        console.print(_TailOverflowRenderable(Group("old", "latest")))

        self.assertEqual(output.getvalue().strip(), "latest")

    def test_tail_overflow_renderable_preserves_short_output(self) -> None:
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=False,
            width=20,
            height=3,
        )

        console.print(_TailOverflowRenderable(Group("first", "second")))

        self.assertEqual(
            output.getvalue().splitlines(),
            ["first", "second"],
        )
        self.assertEqual(
            str(_TailOverflowRenderable(Group("first", "second"))),
            "first\nsecond",
        )
        self.assertEqual(
            _TailOverflowRenderable(Group("first", "second")).renderables,
            ["first", "second"],
        )

    def test_tail_overflow_renderable_restores_complete_output(self) -> None:
        output = StringIO()
        console = Console(
            file=output,
            force_terminal=False,
            width=20,
            height=2,
        )
        renderable = _TailOverflowRenderable(
            Group("line 0", "line 1", "line 2")
        )

        renderable.show_all()
        console.print(renderable)

        self.assertEqual(
            output.getvalue().splitlines(),
            ["line 0", "line 1", "line 2"],
        )

    def test_tail_overflow_live_delegates_and_restores_full_render(
        self,
    ) -> None:
        delegate = _FakeLive()
        full_render = Group("old", "latest")
        live = _TailOverflowLive(delegate, None)

        self.assertIs(live.__enter__(), live)
        live.auto_refresh = False
        live.update(full_render)
        live.refresh()
        result = live.__exit__(None, None, None)

        self.assertFalse(result)
        self.assertFalse(live.auto_refresh)
        self.assertEqual(delegate.entered, 1)
        self.assertEqual(delegate.exited, 1)
        self.assertEqual(delegate.refreshed, 1)
        self.assertIsInstance(
            delegate.updates[0],
            _TailOverflowRenderable,
        )
        self.assertEqual(len(delegate.updates), 1)
        output = StringIO()
        Console(
            file=output,
            force_terminal=False,
            width=20,
            height=1,
        ).print(delegate.updates[0])
        self.assertEqual(output.getvalue().splitlines(), ["old", "latest"])

    def test_tail_overflow_live_without_render_exits_cleanly(self) -> None:
        delegate = _FakeLive()
        live = _TailOverflowLive(delegate, None)

        result = live.__exit__(None, None, None)

        self.assertFalse(result)
        self.assertEqual(delegate.updates, [])
        self.assertEqual(delegate.exited, 1)

    def test_default_live_factory_wraps_rich_live(self) -> None:
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

        self.assertIsInstance(result, _TailOverflowLive)
        initial_renderable = live_type.call_args.args[0]
        self.assertIsInstance(initial_renderable, _TailOverflowRenderable)
        self.assertEqual(initial_renderable.renderable, "frame")
        live_type.assert_called_once_with(
            initial_renderable,
            console=console,
            refresh_per_second=3,
            screen=True,
        )

    def test_default_live_factory_preserves_empty_initial_render(self) -> None:
        console = MagicMock()
        live = MagicMock()
        with patch(
            "avalan.cli.stream_coordinator.Live", return_value=live
        ) as live_type:
            result = _default_live_factory(
                None,
                console=console,
                refresh_per_second=3,
                screen=False,
            )

        self.assertIsInstance(result, _TailOverflowLive)
        live_type.assert_called_once_with(
            None,
            console=console,
            refresh_per_second=3,
            screen=False,
        )

    def test_stream_recording_filename_uses_utc_timestamp(self) -> None:
        dt_value = datetime(2024, 1, 2, 3, 4, 5, 123456, tzinfo=timezone.utc)
        with patch("avalan.cli.stream_coordinator.datetime") as dt_patch:
            dt_patch.now.return_value = dt_value
            filename = stream_recording_filename()

        self.assertEqual(filename, "avalan-screenshot-20240102030405-123.svg")
