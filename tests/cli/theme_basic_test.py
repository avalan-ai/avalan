import subprocess
import sys
import unittest
from argparse import Namespace
from dataclasses import replace
from datetime import datetime
from io import StringIO
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from rich.console import Console
from rich.spinner import Spinner

from avalan.cli.commands import cache as cache_cmds
from avalan.cli.commands import model as model_cmds
from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.display_snapshot import (
    CliProjectionMetadataSummarySnapshot,
    CliStreamSnapshot,
    CliStreamSnapshotBuilder,
    CliToolExecutionSummarySnapshot,
)
from avalan.cli.download import tqdm_rich_progress
from avalan.cli.theme import Theme
from avalan.cli.theme.basic import (
    BasicStreamPresenter,
    BasicTheme,
    _basic_active_model_renderable,
    _basic_active_tool_line,
    _basic_active_tool_renderable,
    _basic_completed_model_line,
    _basic_completed_tool_line,
    _basic_has_executed_tool_frame,
    _basic_json_tool_answer,
    _basic_open_harmony_pattern,
    _basic_tool_elapsed_text,
    _basic_tool_result_summary,
    _basic_tool_status_outcome_markup,
    _basic_tool_subject_markup,
    _basic_visible_answer_text,
    _BasicActiveModelSpinner,
    _BasicAnswerPresenter,
    _BasicToolLineEntry,
)
from avalan.cli.theme.fancy import FancyTheme
from avalan.cli.theme.stream_presenter import (
    CliStreamAnswerTextChunk,
    CliStreamPresenterContext,
    CliStreamPresenterRequest,
    CliStreamRenderableFrame,
)
from avalan.cli.theme.tool_projection import projection_terminal_markup
from avalan.entities import (
    Model,
    TokenizerConfig,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
)
from avalan.event import Event, EventType
from avalan.model.stream import StreamTerminalOutcome
from avalan.tool.display import (
    ToolDisplayDetail,
    ToolDisplayPreview,
    ToolDisplayProjection,
)


def _gettext(message: str) -> str:
    return f"translated:{message}"


def _ngettext(singular: str, plural: str, n: int) -> str:
    return singular if n == 1 else plural


def _model() -> Model:
    now = datetime(2024, 1, 1)
    return Model(
        id="model-id",
        parameters=None,
        parameter_types=None,
        inference=None,
        library_name=None,
        license=None,
        pipeline_tag=None,
        tags=[],
        architectures=None,
        model_type=None,
        auto_model=None,
        processor=None,
        gated=False,
        private=False,
        disabled=False,
        last_downloads=0,
        downloads=0,
        likes=0,
        ranking=None,
        author="author",
        created_at=now,
        updated_at=now,
    )


def _stream_config(**overrides: object) -> CliStreamDisplayConfig:
    values = {
        "quiet": False,
        "stats": False,
        "display_tools": False,
        "display_events": False,
        "display_tools_events": 2,
        "record": False,
        "interactive": True,
        "refresh_per_second": 10,
        "answer_height": 12,
        "answer_height_expand": False,
        "display_tokens": 0,
        "display_pause": 0,
        "display_probabilities": False,
        "display_probabilities_maximum": 0.8,
        "display_probabilities_sample_minimum": 0.1,
        "display_time_to_n_token": None,
        "display_reasoning_time": False,
    }
    values.update(overrides)
    return CliStreamDisplayConfig(**values)


def _stream_context() -> CliStreamPresenterContext:
    return CliStreamPresenterContext(model_id="model", console_width=80)


def _stream_request(
    config: CliStreamDisplayConfig,
    snapshot: CliStreamSnapshot,
    *,
    mode: str = "live",
) -> CliStreamPresenterRequest:
    return CliStreamPresenterRequest(
        snapshot=snapshot,
        display_config=config,
        context=_stream_context(),
        mode=mode,  # type: ignore[arg-type]
    )


async def _collect_stream_items(
    presenter: BasicStreamPresenter,
    request: CliStreamPresenterRequest,
) -> list[object]:
    return [item async for item in presenter.present(request)]


def _answer_chunks(items: list[object]) -> list[str]:
    return [
        item.text
        for item in items
        if isinstance(item, CliStreamAnswerTextChunk)
    ]


def _frames(items: list[object]) -> list[CliStreamRenderableFrame]:
    return [
        item for item in items if isinstance(item, CliStreamRenderableFrame)
    ]


def _render_text(renderable: object) -> str:
    output = StringIO()
    Console(file=output, force_terminal=False, width=120).print(renderable)
    return output.getvalue()


def _visible_text(items: list[object]) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, CliStreamAnswerTextChunk):
            parts.append(item.text)
        elif isinstance(item, CliStreamRenderableFrame):
            parts.append(_render_text(item.renderable))
    return "".join(parts)


class BasicStreamPresenterTestCase(unittest.IsolatedAsyncioTestCase):
    async def test_default_streaming_is_answer_only_with_final_newline(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_reasoning_text("private reasoning")
        builder.append_tool_call_request_text('{"name": "calc"}')
        builder.append_answer_text("Answer is 25.")
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        repeated = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(items), ["Answer is 25.", "\n"])
        self.assertEqual(_frames(items), [])
        self.assertEqual(_answer_chunks(repeated), [])

    async def test_completed_empty_answer_does_not_emit_newline(self) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(items), [])
        self.assertEqual(_frames(items), [])

    async def test_completed_empty_answer_warning_survives_zero_tool_limit(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=True,
            display_tools_events=0,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_result_summary(
            tool_call_id="call-1",
            name="shell.cat",
            status="result",
            result="output",
            arguments_count=0,
        )
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(items), [])
        frames = _frames(items)
        self.assertEqual([frame.role for frame in frames], ["tools"])
        tool_text = _render_text(frames[0].renderable)
        self.assertIn("No final answer emitted.", tool_text)
        self.assertNotIn("shell.cat", tool_text)

    async def test_errored_terminal_survives_zero_tool_limit(self) -> None:
        config = _stream_config(
            display_tools=True,
            display_tools_events=0,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_result_summary(
            tool_call_id="call-1",
            name="shell.cat",
            status="result",
            result="output",
            arguments_count=0,
        )
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.ERRORED,
            error={"error": {"message": "response failed"}},
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(items), [])
        frames = _frames(items)
        self.assertEqual([frame.role for frame in frames], ["tools"])
        tool_text = _render_text(frames[0].renderable)
        self.assertIn("Model stream error: ", tool_text)
        self.assertIn("response failed", tool_text)
        self.assertNotIn("No final answer emitted.", tool_text)
        self.assertNotIn("shell.cat", tool_text)

    async def test_completed_answer_ending_newline_is_not_duplicated(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("Answer is 25.\n")
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(items), ["Answer is 25.\n"])
        self.assertEqual(_frames(items), [])

    async def test_answer_mode_suppresses_frames_and_reset_replays(
        self,
    ) -> None:
        config = _stream_config(
            stats=True,
            display_tools=True,
            display_events=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("Answer.")
        builder.add_active_tool(tool_call_id="call", name="calc")
        builder.add_event_summary(event_type=EventType.START)
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        snapshot = builder.snapshot()
        presenter = BasicStreamPresenter(getLogger(__name__))

        first = await _collect_stream_items(
            presenter,
            _stream_request(config, snapshot, mode="answer"),
        )
        presenter.reset()
        second = await _collect_stream_items(
            presenter,
            _stream_request(config, snapshot, mode="answer"),
        )

        self.assertEqual(_answer_chunks(first), ["Answer.", "\n"])
        self.assertEqual(_frames(first), [])
        self.assertEqual(_answer_chunks(second), ["Answer.", "\n"])
        self.assertEqual(_frames(second), [])

    async def test_theme_stream_presenter_prefixes_visible_answer_only(
        self,
    ) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        config = _stream_config(display_tools=True)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(tool_call_id="call", name="calc")
        presenter = theme.stream_presenter(
            getLogger(__name__),
            answer_prefix=":robot: ",
        )

        tool_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.append_answer_text("Answer.")
        answer_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(tool_items), [])
        self.assertEqual(
            _answer_chunks(answer_items),
            [":robot: ", "Answer."],
        )

    async def test_theme_stream_presenter_separates_answer_after_executed_tool(
        self,
    ) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        config = _stream_config(display_tools=True)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(tool_call_id="call", name="calc")
        builder.complete_tool(
            tool_call_id="call",
            name="calc",
            elapsed_seconds=5.004,
        )
        builder.add_tool_result_summary(
            tool_call_id="call",
            name="calc",
            status="result",
            result=25,
            arguments_count=1,
            elapsed_seconds=5.004,
        )
        presenter = theme.stream_presenter(
            getLogger(__name__),
            answer_prefix=":robot: ",
        )

        tool_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.append_answer_text("Answer.")
        answer_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertIn("Executed tool calc · 5s: 25", _visible_text(tool_items))
        self.assertEqual(
            _answer_chunks(answer_items),
            ["\n", ":robot: ", "Answer."],
        )

    async def test_theme_stream_presenter_does_not_clear_absent_tool_frame(
        self,
    ) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        config = _stream_config(display_tools=True)
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("Answer.")
        presenter = theme.stream_presenter(
            getLogger(__name__),
            answer_prefix=":robot: ",
        )

        answer_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_frames(answer_items), [])
        self.assertEqual(
            _answer_chunks(answer_items),
            [":robot: ", "Answer."],
        )

    async def test_requested_tools_and_events_render_compact_frames(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=True,
            display_events=True,
            display_tools_events=8,
            interactive=False,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("working")
        builder.add_active_tool(
            tool_call_id="active-call",
            name="search",
            arguments={"query": "weather"},
        )
        builder.add_active_tool(tool_call_id="done-call", name="calc")
        builder.complete_tool(tool_call_id="done-call", name="calc")
        builder.add_tool_result_summary(
            tool_call_id="done-call",
            name="calc",
            status="result",
            result=25,
            arguments_count=1,
        )
        builder.add_tool_diagnostic(
            ToolCallDiagnostic(
                id="diag",
                call_id="done-call",
                requested_name="calc",
                code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
                stage=ToolCallDiagnosticStage.RESOLVE,
                message="Unknown tool.",
            )
        )
        builder.add_tool_event(
            Event(type=EventType.TOOL_PROCESS, payload={"name": "calc"}),
            tool_call_id="done-call",
            name="calc",
        )
        builder.add_event_summary(
            event_type=EventType.START,
            payload={"node": "math"},
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        frames = _frames(items)
        self.assertEqual(_answer_chunks(items), ["working"])
        self.assertEqual([frame.role for frame in frames], ["tools", "events"])
        tool_text = _render_text(frames[0].renderable)
        event_text = str(frames[1].renderable)
        for fragment in (
            "Starting tool search",
            "Executed tool calc: 25",
            "✅",
            "tool calc diagnostic tool.unknown",
            "tool event tool_process: calc",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, tool_text)
        self.assertLess(
            tool_text.index("Executed tool calc: 25"),
            tool_text.index("Starting tool search"),
        )
        self.assertIn("event start", event_text)
        self.assertNotIn("tool_process", event_text)

    async def test_model_continuation_progress_renders_in_tool_frame(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_model_continuation(
            model_continuation_id="continuation-1",
            started_at=1.0,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )
            tool_text = _render_text(_frames(items)[0].renderable)

        self.assertIn("Thinking for 1s...", tool_text)
        self.assertNotIn("tool event model_continuation", tool_text)

        with patch("avalan.cli.theme.basic.perf_counter", return_value=3.0):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )
            repeated_active_text = _render_text(_frames(items)[0].renderable)

        self.assertIn("Thinking for 2s...", repeated_active_text)

        builder.finish_model_continuation(
            model_continuation_id="continuation-1",
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        frames = _frames(items)
        self.assertEqual([frame.role for frame in frames], ["tools"])
        self.assertIn("Thought for 1s.", _render_text(frames[0].renderable))

        builder.add_active_model_continuation(
            model_continuation_id="continuation-1",
            started_at=3.0,
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=4.0):
            await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )
        builder.finish_model_continuation(
            model_continuation_id="continuation-1",
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=4.0):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        self.assertEqual(_frames(items)[0].renderable, "")

    async def test_model_continuation_progress_is_cleared_before_answer(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_model_continuation(
            model_continuation_id="continuation-1",
            started_at=1.0,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        builder.finish_model_continuation(
            model_continuation_id="continuation-1",
        )
        builder.append_answer_text("Done.")
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        self.assertIsInstance(items[0], CliStreamRenderableFrame)
        self.assertEqual(
            cast(CliStreamRenderableFrame, items[0]).renderable,
            "",
        )
        self.assertEqual(_answer_chunks(items), ["Done.", "\n"])
        self.assertNotIn("Thinking for", _visible_text(items))
        self.assertNotIn("Thought for", _visible_text(items))

    async def test_json_answer_keeps_tool_history_without_model_progress(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.complete_tool(
            tool_call_id="call",
            name="database.query",
            elapsed_seconds=1.2,
        )
        builder.add_tool_result_summary(
            tool_call_id="call",
            name="database.query",
            status="result",
            result={"rows": 8},
            arguments_count=1,
            elapsed_seconds=1.2,
        )
        builder.add_active_model_continuation(
            model_continuation_id="continuation-1",
            started_at=1.0,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        builder.finish_model_continuation(
            model_continuation_id="continuation-1",
        )
        builder.append_answer_text('{"items":[{"amount":134}]}')
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        frames = _frames(items)
        self.assertEqual([frame.role for frame in frames], ["tools"])
        tool_text = _render_text(frames[0].renderable)
        answer_text = "".join(_answer_chunks(items))
        self.assertIn("Executed tool database.query", tool_text)
        self.assertNotIn("Thinking for", tool_text)
        self.assertNotIn("Thought for", tool_text)
        self.assertIn('{\n  "items": [', answer_text)
        self.assertIn('"amount": 134', answer_text)

    async def test_pdf_tooling_lines_do_not_interleave_with_json_answer(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=12)
        builder = CliStreamSnapshotBuilder(config)
        presenter = BasicStreamPresenter(getLogger(__name__))

        builder.add_tool_result_summary(
            tool_call_id="pdfinfo",
            name="pdfinfo",
            status="result",
            result="PDF metadata",
            arguments_count=1,
            elapsed_seconds=0.06,
        )
        first_tools = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        builder.append_answer_text('{"items":[{"matching_signals":["docket')
        partial_answer = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        builder.add_tool_result_summary(
            tool_call_id="pdftotext",
            name="pdftotext",
            status="result",
            result="OCR text",
            arguments_count=1,
            elapsed_seconds=0.05,
        )
        builder.add_tool_result_summary(
            tool_call_id="pdftoppm",
            name="pdftoppm",
            status="result",
            result="GENERATED_PREFIX",
            arguments_count=1,
            elapsed_seconds=1.5,
        )
        builder.add_tool_result_summary(
            tool_call_id="tesseract",
            name="tesseract",
            status="result",
            result="OCR output",
            arguments_count=1,
            elapsed_seconds=0.48,
        )
        more_tools = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        builder.append_answer_text('","payor","amount"]}]}')
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        final_answer = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        tool_output = _visible_text(
            [
                *first_tools,
                *partial_answer,
                *more_tools,
            ]
        )
        answer_output = "".join(_answer_chunks(final_answer))
        combined_output = tool_output + answer_output
        self.assertEqual(_answer_chunks(partial_answer), [])
        self.assertEqual(_answer_chunks(more_tools), [])
        self.assertIn("Executed tool pdfinfo", tool_output)
        self.assertIn("Executed tool pdftotext", tool_output)
        self.assertIn("Executed tool pdftoppm", tool_output)
        self.assertIn("Executed tool tesseract", tool_output)
        self.assertNotIn('"docket', tool_output)
        self.assertNotIn("docket✅", combined_output)
        self.assertIn('{\n  "items": [', answer_output)
        self.assertIn('"docket"', answer_output)
        self.assertIn('"payor"', answer_output)
        self.assertIn('"amount"', answer_output)

    async def test_basic_keeps_event_only_tool_execution_side_events(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_event(
            Event(type=EventType.TOOL_MODEL_RUN, payload={"channel": "x"}),
            tool_call_id="call",
            name="calc",
        )
        builder.add_tool_event(
            Event(type=EventType.TOOL_EXECUTE, payload={"channel": "x"}),
            tool_call_id="call",
            name="calc",
        )
        builder.add_tool_event(
            Event(type=EventType.TOOL_RESULT, payload={"channel": "x"}),
            tool_call_id="call",
            name="calc",
        )
        builder.add_tool_event(
            Event(
                type=EventType.TOOL_MODEL_RESPONSE,
                payload={"channel": "x"},
            ),
            tool_call_id="call",
            name="calc",
        )
        builder.add_tool_event(
            Event(type=EventType.TOOL_DIAGNOSTIC, payload={"channel": "x"}),
            tool_call_id="call",
            name="calc",
        )
        builder.add_tool_event(
            Event(type=EventType.TOOL_PROCESS, payload={"name": "calc"}),
            tool_call_id="call",
            name="calc",
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        output = _visible_text(items)
        self.assertNotIn("tool_model_run", output)
        self.assertNotIn("tool_model_response", output)
        self.assertNotIn("tool_diagnostic", output)
        self.assertIn("tool event tool_execute: calc", output)
        self.assertIn("tool event tool_result: calc", output)
        self.assertIn("tool event tool_process: calc", output)

    async def test_basic_suppresses_canonical_duplicate_tool_side_events(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_result_summary(
            tool_call_id="call",
            name="calc",
            status="result",
            result=25,
            arguments_count=1,
        )
        builder.add_tool_event(
            Event(type=EventType.TOOL_EXECUTE, payload={"channel": "x"}),
            tool_call_id="call",
            name="calc",
        )
        builder.add_tool_event(
            Event(type=EventType.TOOL_RESULT, payload={"channel": "x"}),
            tool_call_id="call",
            name="calc",
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        output = _visible_text(items)
        self.assertIn("Executed tool calc: 25", output)
        self.assertNotIn("tool_execute", output)
        self.assertNotIn("tool_result", output)

    async def test_delayed_calculator_progress_renders_lifecycle(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(
            tool_call_id="calculator-call",
            name="math.calculator",
            arguments={"expression": "(4 + 6) * 5 / 2"},
            started_at=10.0,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        start_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.update_active_tool(
            tool_call_id="calculator-call",
            name="math.calculator",
            updated_at=12.5,
        )
        running_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.complete_tool(
            tool_call_id="calculator-call",
            name="math.calculator",
            elapsed_seconds=2.5,
        )
        builder.add_tool_result_summary(
            tool_call_id="calculator-call",
            name="math.calculator",
            status="result",
            result={
                "arguments": {"expression": "(4 + 6) * 5 / 2"},
                "name": "math.calculator",
                "result": "25",
            },
            arguments_count=1,
            elapsed_seconds=2.5,
        )
        completed_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        start_frames = _frames(start_items)
        running_frames = _frames(running_items)
        completed_frames = _frames(completed_items)
        self.assertEqual([frame.role for frame in start_frames], ["tools"])
        self.assertEqual([frame.role for frame in running_frames], ["tools"])
        self.assertEqual([frame.role for frame in completed_frames], ["tools"])
        self.assertIsInstance(start_frames[0].renderable, Spinner)
        self.assertEqual(
            cast(Spinner, start_frames[0].renderable).name,
            "point",
        )
        self.assertEqual(
            cast(Spinner, start_frames[0].renderable).style,
            "cyan",
        )
        self.assertEqual(
            cast(Spinner, running_frames[0].renderable).name,
            "point",
        )
        self.assertEqual(
            cast(Spinner, running_frames[0].renderable).style,
            "cyan",
        )

        start_text = _render_text(start_frames[0].renderable)
        running_text = _render_text(running_frames[0].renderable)
        completed_text = _render_text(completed_frames[0].renderable)

        self.assertIn("Starting tool math.calculator", start_text)
        self.assertNotIn("expression", start_text)
        self.assertIn("Running tool math.calculator for 2.5s", running_text)
        self.assertNotEqual(start_text, running_text)
        self.assertIn(
            "Executed tool math.calculator · 2.5s: 25",
            completed_text,
        )
        self.assertIn(": 25", completed_text)
        self.assertNotIn("milliseconds", completed_text)
        self.assertNotIn("microseconds", completed_text)
        self.assertNotIn("arguments", completed_text)
        self.assertIn("✅", completed_text)

    async def test_completed_tool_without_elapsed_keeps_compact_text(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(tool_call_id="call", name="math.calculator")
        builder.complete_tool(tool_call_id="call", name="math.calculator")
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        tool_text = _render_text(_frames(items)[0].renderable)
        self.assertIn("Executed tool math.calculator", tool_text)
        self.assertNotIn(":", tool_text)
        self.assertNotIn("unknown", tool_text)

    def test_tool_subject_styles_name_args_and_elapsed_separately(
        self,
    ) -> None:
        self.assertEqual(_basic_tool_subject_markup(""), "")
        self.assertEqual(
            _basic_tool_subject_markup("rg --no-heading needle"),
            "[bold]rg[/bold] [dim]--no-heading needle[/dim]",
        )
        active_line = _basic_active_tool_line(
            "rg --no-heading needle",
            started_at=None,
            updated_at=None,
        )
        completed_line = _basic_completed_tool_line(
            "rg --no-heading needle",
            "completed",
            0.133,
        )

        self.assertIn(
            "[bold]rg[/bold] [dim]--no-heading needle[/dim]",
            active_line,
        )
        self.assertIn(
            "[bold]rg[/bold] [dim]--no-heading needle[/dim]",
            completed_line,
        )
        self.assertIn("[bold]· 133ms[/bold]", completed_line)
        self.assertIn(
            ": error",
            _basic_completed_tool_line("calc", "error", None),
        )
        self.assertIsNone(_basic_tool_status_outcome_markup(None, None))
        self.assertEqual(
            _basic_tool_status_outcome_markup(
                "non_executed",
                "tool_call.repeated",
            ),
            "non_executed tool_call.repeated",
        )

    def test_database_projection_lines_use_human_phrases(self) -> None:
        tables_projection = ToolDisplayProjection(
            action="list",
            label="database.tables",
            target="tables",
            scope="database",
            details=(
                ToolDisplayDetail(label="operation", value="tables"),
                ToolDisplayDetail(label="database", value="analytics"),
            ),
        )
        completed_tables_projection = ToolDisplayProjection(
            action="list",
            label="database.tables",
            target="tables",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="tables"),
                ToolDisplayDetail(label="database", value="analytics"),
                ToolDisplayDetail(label="tables", value=3),
            ),
        )
        inspect_projection = ToolDisplayProjection(
            action="inspect",
            label="database.inspect",
            target="users, orders",
            scope="database",
            details=(
                ToolDisplayDetail(label="operation", value="inspect"),
                ToolDisplayDetail(label="database", value="claims"),
            ),
        )
        unknown_inspect_projection = ToolDisplayProjection(
            action="inspect",
            label="database.inspect",
            target="tables",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="inspect"),
                ToolDisplayDetail(label="database", value="claims"),
            ),
        )
        run_projection = ToolDisplayProjection(
            action="query",
            label="database.run",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="run"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(label="sql", value="SELECT * FROM users"),
            ),
            metrics={"rows": 2},
            preview=ToolDisplayPreview(
                label="sql",
                content="SELECT * FROM users",
            ),
        )
        long_select = (
            "SELECT field_1, field_2, field_3, field_4, field_5, "
            "field_6 FROM synced_claims WHERE organization_id = 42"
        )
        long_select_projection = ToolDisplayProjection(
            action="query",
            label="database.run",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="run"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(label="sql", value=long_select),
            ),
            metrics={"rows": 5},
        )
        empty_select_projection = ToolDisplayProjection(
            action="query",
            label="database.run",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="run"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(
                    label="sql",
                    value=(
                        "SELECT id FROM batches WHERE tag = 'reese-20260618'"
                    ),
                ),
            ),
            metrics={"rows": 0},
        )
        select_without_count_projection = ToolDisplayProjection(
            action="query",
            label="database.run",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="run"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(label="sql", value="SELECT id FROM batches"),
            ),
        )
        plan_projection = ToolDisplayProjection(
            action="explain",
            label="database.plan",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="plan"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(label="sql_command", value="select"),
            ),
        )
        inspect_detail_projection = ToolDisplayProjection(
            action="inspect",
            label="database.inspect",
            target="tables",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="inspect"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(label="tables", value="users, orders"),
            ),
            metrics={"tables": 2},
        )
        update_projection = ToolDisplayProjection(
            action="query",
            label="database.run",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="run"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(
                    label="sql", value="UPDATE users SET active = TRUE"
                ),
            ),
            metrics={"rows": 0},
        )
        float_rows_projection = ToolDisplayProjection(
            action="query",
            label="database.run",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="run"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(
                    label="sql", value="UPDATE users SET active = TRUE"
                ),
            ),
            metrics={"rows": 2.0},
        )
        bool_rows_projection = ToolDisplayProjection(
            action="query",
            label="database.run",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="run"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(
                    label="sql", value="UPDATE users SET active = TRUE"
                ),
            ),
            metrics={"rows": True},
        )

        active_tables = _basic_active_tool_line(
            "database.tables",
            started_at=None,
            updated_at=None,
            display_projection=tables_projection,
        )
        completed_tables = _basic_completed_tool_line(
            "database.tables",
            "completed",
            None,
            display_projection=completed_tables_projection,
        )
        running_inspect = _basic_active_tool_line(
            "database.inspect",
            started_at=1.0,
            updated_at=3.0,
            display_projection=inspect_projection,
        )
        completed_unknown_inspect = _basic_completed_tool_line(
            "database.inspect",
            "completed",
            None,
            display_projection=unknown_inspect_projection,
        )
        completed_run = _basic_completed_tool_line(
            "database.run",
            "completed",
            0.019,
            display_projection=run_projection,
        )
        completed_long_select = _basic_completed_tool_line(
            "database.run",
            "completed",
            None,
            display_projection=long_select_projection,
        )
        completed_empty_select = _basic_completed_tool_line(
            "database.run",
            "completed",
            None,
            display_projection=empty_select_projection,
        )
        completed_select_without_count = _basic_completed_tool_line(
            "database.run",
            "completed",
            None,
            display_projection=select_without_count_projection,
        )
        completed_plan = _basic_completed_tool_line(
            "database.plan",
            "completed",
            None,
            display_projection=plan_projection,
        )
        completed_inspect_detail = _basic_completed_tool_line(
            "database.inspect",
            "completed",
            None,
            display_projection=inspect_detail_projection,
        )
        completed_update = _basic_completed_tool_line(
            "database.run",
            "completed",
            None,
            display_projection=update_projection,
        )
        completed_float_rows = _basic_completed_tool_line(
            "database.run",
            "completed",
            None,
            display_projection=float_rows_projection,
        )
        completed_bool_rows = _basic_completed_tool_line(
            "database.run",
            "completed",
            None,
            display_projection=bool_rows_projection,
        )

        self.assertIn(
            "[bold]Listing[/bold] [dim]tables[/dim] "
            "[dim]in database[/dim] analytics...",
            active_tables,
        )
        self.assertIn(
            "[bold]Listed 3 tables[/bold] [dim]in database[/dim] analytics",
            completed_tables,
        )
        self.assertIn(
            "[bold]Inspecting 2 tables[/bold]: users, orders "
            "[dim]from database[/dim] claims for 2s.",
            running_inspect,
        )
        self.assertIn(
            "[bold]Inspected tables[/bold] [dim]from database[/dim] claims",
            completed_unknown_inspect,
        )
        self.assertIn(
            "[bold]Executed query[/bold] SELECT * FROM users "
            "[dim]in database[/dim] claims: [bold]2 rows[/bold] found. "
            "[bold]· 19ms[/bold]",
            completed_run,
        )
        self.assertIn(
            "[bold]Executed query[/bold] SELECT field_1", completed_long_select
        )
        self.assertIn(
            " ... [dim]in database[/dim] claims", completed_long_select
        )
        self.assertIn("[bold]5 rows[/bold] found.", completed_long_select)
        self.assertIn(
            "[bold]Executed query[/bold] SELECT id FROM batches "
            "WHERE tag = 'reese-20260618' "
            "[dim]in database[/dim] claims: no results.",
            completed_empty_select,
        )
        self.assertNotIn("[bold]no results[/bold]", completed_empty_select)
        self.assertIn(
            "[bold]Executed query[/bold] SELECT id FROM batches "
            "[dim]in database[/dim] claims",
            completed_select_without_count,
        )
        self.assertNotIn("found.", completed_select_without_count)
        self.assertIn(
            "[bold]Explained[/bold] [bold]SELECT[/bold] [dim]statement[/dim] "
            "[dim]in database[/dim] claims",
            completed_plan,
        )
        self.assertIn(
            "[bold]Inspected 2 tables[/bold]: users, orders "
            "[dim]from database[/dim] claims",
            completed_inspect_detail,
        )
        self.assertIn(
            "[bold]Ran SQL[/bold] [dim]statement[/dim] "
            "UPDATE users SET active = TRUE [dim]in database[/dim] claims: "
            "[bold]0 rows[/bold]",
            completed_update,
        )
        self.assertIn("[bold]2 rows[/bold]", completed_float_rows)
        self.assertNotIn("True rows", completed_bool_rows)

    def test_database_projection_lines_cover_fallback_edges(self) -> None:
        tables_projection = ToolDisplayProjection(
            action="list",
            label="database.tables",
            target="tables",
            scope="database",
            details=(ToolDisplayDetail(label="operation", value="tables"),),
        )
        identity_projection = ToolDisplayProjection(
            action="list",
            label="database.tables",
            target="tables",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="tables"),
                ToolDisplayDetail(label="database", value=" \n "),
                ToolDisplayDetail(label="db_name", value="analytics"),
            ),
        )
        rows_projection = ToolDisplayProjection(
            action="query",
            label="database.run",
            target="SQL statement",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="run"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(label="sql", value="UPDATE users SET x = 1"),
                ToolDisplayDetail(label="rows", value="two"),
                ToolDisplayDetail(label="rows", value=2),
            ),
        )
        subject_projection = ToolDisplayProjection(
            action="inspect",
            scope="workspace",
        )

        active_without_started_at = _basic_active_tool_line(
            "database.tables",
            started_at=None,
            updated_at=3.0,
            display_projection=tables_projection,
        )
        with patch(
            "avalan.cli.theme.basic._basic_tool_elapsed_text",
            return_value=None,
        ):
            active_without_elapsed_text = _basic_active_tool_line(
                "database.tables",
                started_at=1.0,
                updated_at=3.0,
                display_projection=tables_projection,
            )
            active_tool_without_elapsed_text = _basic_active_tool_line(
                "calc",
                started_at=1.0,
                updated_at=3.0,
            )
        completed_identity = _basic_completed_tool_line(
            "database.tables",
            "completed",
            None,
            display_projection=identity_projection,
        )
        completed_rows = _basic_completed_tool_line(
            "database.run",
            "completed",
            None,
            display_projection=rows_projection,
        )
        active_subject = _basic_active_tool_line(
            "workspace.inspect",
            started_at=None,
            updated_at=None,
            display_projection=subject_projection,
        )

        self.assertIn(
            "Listing tables in database.",
            _render_text(active_without_started_at),
        )
        self.assertIn(
            "Listing tables in database.",
            _render_text(active_without_elapsed_text),
        )
        self.assertIn(
            "Running tool calc.",
            _render_text(active_tool_without_elapsed_text),
        )
        self.assertIn("analytics", _render_text(completed_identity))
        self.assertIn("2 rows", _render_text(completed_rows))
        self.assertIn(
            "Starting tool inspect in workspace.",
            _render_text(active_subject),
        )

    def test_zero_count_database_inspect_does_not_show_table_list(
        self,
    ) -> None:
        projection = ToolDisplayProjection(
            action="inspect",
            label="database.inspect",
            target="public.jobs",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(label="operation", value="inspect"),
                ToolDisplayDetail(label="database", value="claims"),
                ToolDisplayDetail(label="tables", value="public.jobs"),
            ),
            metrics={"tables": 0},
        )

        line = _basic_completed_tool_line(
            "database.inspect",
            "completed",
            None,
            display_projection=projection,
        )

        self.assertIn(
            "[bold]Inspected 0 tables[/bold] [dim]from database[/dim] claims",
            line,
        )
        self.assertNotIn("public.jobs", line)

    def test_database_projection_lines_cover_operation_variants(self) -> None:
        cases = (
            ("database.count", "users", "Counted rows in table users"),
            ("database.keys", "users", "Inspected keys for table users"),
            (
                "database.relationships",
                "users",
                "Inspected relationships for table users",
            ),
            ("database.plan", "SQL statement", "Explained SQL statement"),
            ("database.sample", "users", "Sampled rows from table users"),
            ("database.size", "users", "Measured table users"),
            ("database.tasks", "tasks", "Listed tasks"),
            ("database.locks", "locks", "Inspected locks"),
        )

        for label, target, expected in cases:
            with self.subTest(label=label):
                projection = ToolDisplayProjection(
                    action=label.rsplit(".", 1)[1],
                    label=label,
                    target=target,
                    scope="database",
                    status="completed",
                )
                line = _render_text(
                    _basic_completed_tool_line(
                        label,
                        "completed",
                        None,
                        display_projection=projection,
                    )
                )

                self.assertIn(expected, line)

        task_projection = ToolDisplayProjection(
            action="list",
            target="tasks",
            scope="database",
            status="completed",
        )
        redacted_sql_projection = ToolDisplayProjection(
            action="query",
            target="SQL statement",
            scope="database",
            details=(ToolDisplayDetail(label="sql", value=None),),
        )
        redacted_database_projection = ToolDisplayProjection(
            action="list",
            target="tables",
            scope="database",
            status="completed",
            details=(
                ToolDisplayDetail(
                    label="database",
                    value="private-db",
                    redacted=True,
                ),
            ),
        )
        unknown_projection = ToolDisplayProjection(
            action="inspect",
            target="things",
            scope="database",
            details=(ToolDisplayDetail(label="operation", value="unknown"),),
        )

        self.assertIn(
            "Listed tasks",
            _render_text(
                _basic_completed_tool_line(
                    "database.tasks",
                    "completed",
                    None,
                    display_projection=task_projection,
                )
            ),
        )
        redacted_database_text = _render_text(
            _basic_completed_tool_line(
                "database.tables",
                "completed",
                None,
                display_projection=redacted_database_projection,
            )
        )
        self.assertIn("Listed tables in database", redacted_database_text)
        self.assertNotIn("private-db", redacted_database_text)
        self.assertIn(
            "Running SQL statement",
            _render_text(
                _basic_active_tool_line(
                    "database.run",
                    started_at=1.0,
                    updated_at=1.5,
                    display_projection=redacted_sql_projection,
                )
            ),
        )
        self.assertIn(
            "Executed tool inspect things in database",
            _render_text(
                _basic_completed_tool_line(
                    "database.unknown",
                    "completed",
                    None,
                    display_projection=unknown_projection,
                )
            ),
        )

    async def test_projected_completed_tool_without_result_uses_projection(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=True,
            display_tools_events=8,
            interactive=False,
        )
        projection = ToolDisplayProjection(
            action="cancel",
            target="[task-1]",
            scope="database",
            status="completed",
            outcome="cancel_requested",
            summary="Cancellation [yellow]requested[/yellow].",
            details=(ToolDisplayDetail(label="task_id", value="[task-1]"),),
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(
            tool_call_id="database-call",
            name="database.kill",
            arguments={"task_id": "raw-json"},
        )
        builder.complete_tool(
            tool_call_id="database-call",
            name="database.kill",
            display_projection=projection,
            elapsed_seconds=0.5,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        text = _render_text(_frames(items)[0].renderable)
        compact_text = " ".join(text.split())
        self.assertIn(
            "Cancelled task [task-1] in database · 500ms",
            compact_text,
        )
        self.assertNotIn("Executed tool", compact_text)
        self.assertNotIn("details:", compact_text)
        self.assertNotIn("raw-json", text)

    async def test_projected_shell_success_omits_run_default_scope_and_summary(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=True,
            display_tools_events=8,
            interactive=False,
        )
        projection = ToolDisplayProjection(
            action="run",
            target="rg --no-heading needle",
            scope=".",
            status="completed",
            outcome="completed",
            summary="rg completed.",
            details=(ToolDisplayDetail(label="exit code", value=0),),
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(
            tool_call_id="shell-call",
            name="shell.run",
            display_projection=projection,
        )
        builder.complete_tool(
            tool_call_id="shell-call",
            name="shell.run",
            display_projection=projection,
            elapsed_seconds=0.133,
        )
        builder.add_tool_result_summary(
            tool_call_id="shell-call",
            name="shell.run",
            status="result",
            result={"status": "completed"},
            arguments_count=1,
            display_projection=projection,
            elapsed_seconds=0.133,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        text = _render_text(_frames(items)[0].renderable)
        self.assertIn("Executed tool rg --no-heading needle · 133ms", text)
        self.assertNotIn("Executed tool run", text)
        self.assertNotIn(" in .", text)
        self.assertNotIn("rg completed", text)
        self.assertNotIn("details:", text)

    async def test_projected_shell_non_success_keeps_status_summary(self):
        config = _stream_config(
            display_tools=True,
            display_tools_events=8,
            interactive=False,
        )
        projection = ToolDisplayProjection(
            action="run",
            target="pdftotext",
            scope=".",
            status="policy_denied",
            outcome="policy_denied",
            summary="pdftotext was denied by policy.",
            details=(ToolDisplayDetail(label="exit code", value=None),),
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(
            tool_call_id="shell-call",
            name="shell.run",
            display_projection=projection,
        )
        builder.complete_tool(
            tool_call_id="shell-call",
            name="shell.run",
            status="policy_denied",
            display_projection=projection,
            elapsed_seconds=0.001,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        text = _render_text(_frames(items)[0].renderable)
        self.assertIn("Executed tool pdftotext · 1ms: policy_denied", text)
        self.assertIn("pdftotext was denied by policy", text)
        self.assertNotIn("details:", text)
        self.assertNotIn("exit code=none", text)

    async def test_projected_tool_lines_prefer_display_projection(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=True,
            display_tools_events=8,
            interactive=False,
        )
        builder = CliStreamSnapshotBuilder(config)
        active_projection = ToolDisplayProjection(
            action="search",
            target="[TODO]",
            scope="src/[avalan]",
            summary="Search [red]source[/red].",
        )
        builder.add_active_tool(
            tool_call_id="shell-call",
            name="shell.run",
            arguments={"query": "raw-json"},
            display_projection=active_projection,
            started_at=1.0,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        active_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        active_text = _render_text(_frames(active_items)[0].renderable)
        self.assertIn(
            "Starting tool search [TODO] in src/[avalan]",
            active_text,
        )
        self.assertNotIn("raw-json", active_text)
        self.assertNotIn('"query"', active_text)

        terminal_projection = ToolDisplayProjection(
            action="inspect",
            target="[users]",
            scope="database",
            status="completed",
            outcome="rows",
            summary="Returned [green]2 rows[/green].",
            details=(
                ToolDisplayDetail(label="[table]", value="[users]"),
                ToolDisplayDetail(label="rows", value=2),
            ),
        )
        builder.complete_tool(
            tool_call_id="shell-call",
            name="database.query",
            display_projection=terminal_projection,
            elapsed_seconds=1.25,
        )
        builder.add_tool_result_summary(
            tool_call_id="shell-call",
            name="database.query",
            status="result",
            result={"raw": "json"},
            arguments_count=1,
            display_projection=terminal_projection,
            elapsed_seconds=1.25,
        )

        completed_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        completed_text = _render_text(_frames(completed_items)[0].renderable)
        completed_compact = " ".join(completed_text.split())
        self.assertIn(
            "Inspected 1 table: [users] from database · 1.2s",
            completed_compact,
        )
        self.assertNotIn("Executed tool", completed_compact)
        self.assertNotIn("details:", completed_compact)
        self.assertNotIn('"raw"', completed_text)
        self.assertNotIn('"json"', completed_text)

    async def test_projected_tool_lines_bound_long_values(self) -> None:
        config = _stream_config(
            display_tools=True,
            display_tools_events=8,
            interactive=False,
        )
        long_value = "x" * 500
        projection = ToolDisplayProjection(
            action="inspect",
            target=long_value,
            scope="database",
            status="completed",
            outcome="rows",
            summary=long_value,
            details=(
                ToolDisplayDetail(label="path", value=long_value),
                ToolDisplayDetail(label="rows", value=25),
                ToolDisplayDetail(label="statement", value=long_value),
                ToolDisplayDetail(label="extra", value=long_value),
            ),
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_result_summary(
            tool_call_id="database-call",
            name="database.query",
            status="result",
            result={"rows": ["raw"]},
            arguments_count=1,
            display_projection=projection,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        text = _render_text(_frames(items)[0].renderable)
        compact_text = " ".join(text.split())
        self.assertIn("Inspected 1 table: " + ("x" * 20), compact_text)
        self.assertIn("...", text)
        self.assertNotIn('"rows": ["raw"]', text)

    async def test_failed_calculator_error_is_hidden_until_tools_display(
        self,
    ) -> None:
        source_config = _stream_config(display_tools=True)
        builder = CliStreamSnapshotBuilder(source_config)
        builder.add_active_tool(
            tool_call_id="calculator-call",
            name="math.calculator",
            arguments={
                "expression": "1 / 0",
                "raw": '<tool_call>{"arguments": "hidden"}</tool_call>',
            },
        )
        builder.complete_tool(
            tool_call_id="calculator-call",
            name="math.calculator",
            status="error",
        )
        builder.add_tool_result_summary(
            tool_call_id="calculator-call",
            name="math.calculator",
            status="error",
            result="ZeroDivisionError: division by zero",
            arguments_count=1,
        )
        snapshot = builder.snapshot()

        default_items = await _collect_stream_items(
            BasicStreamPresenter(getLogger(__name__)),
            _stream_request(_stream_config(), snapshot),
        )
        tools_items = await _collect_stream_items(
            BasicStreamPresenter(getLogger(__name__)),
            _stream_request(
                _stream_config(display_tools=True, display_tools_events=8),
                snapshot,
            ),
        )

        self.assertEqual(_answer_chunks(default_items), [])
        self.assertEqual(_frames(default_items), [])
        tools_frames = _frames(tools_items)
        self.assertEqual([frame.role for frame in tools_frames], ["tools"])
        tool_text = _render_text(tools_frames[0].renderable)
        self.assertIn("Executed tool math.calculator", tool_text)
        self.assertIn("❌", tool_text)
        self.assertIn("ZeroDivisionError: division by zero", tool_text)
        for fragment in (
            "Traceback",
            "<tool_call>",
            "arguments",
            "expression",
            "hidden",
        ):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, tool_text)

    async def test_requested_single_surfaces_render_only_that_surface(
        self,
    ) -> None:
        source_config = _stream_config(
            stats=True,
            display_tools=True,
            display_events=True,
            display_tools_events=8,
        )
        builder = CliStreamSnapshotBuilder(source_config)
        builder.append_answer_text("done")
        builder.add_active_tool(tool_call_id="call", name="calc")
        builder.complete_tool(tool_call_id="call", name="calc")
        builder.add_tool_result_summary(
            tool_call_id="call",
            name="calc",
            status="result",
            result=25,
            arguments_count=1,
        )
        builder.add_event_summary(event_type=EventType.START)
        builder.update_token_counts(
            input_tokens=2,
            cached_input_tokens=1,
            output_tokens=3,
            reasoning_usage_tokens=4,
            total_tokens=10,
        )
        builder.update_timing(elapsed_seconds=0.5)
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        snapshot = builder.snapshot()

        cases = [
            (
                "tools",
                _stream_config(
                    display_tools=True,
                    display_tools_events=8,
                    interactive=False,
                ),
                ["tools"],
                "Executed tool calc: 25",
                ("event start", "tokens "),
            ),
            (
                "events",
                _stream_config(display_events=True, interactive=False),
                ["events"],
                "event start",
                ("tool calc", "tokens "),
            ),
            (
                "stats",
                _stream_config(stats=True, interactive=False),
                ["stats"],
                "tokens in=2, cached=1, out=3, reasoning=4, total=10",
                ("tool calc", "event start"),
            ),
        ]

        for label, config, roles, expected, unexpected in cases:
            with self.subTest(label=label):
                presenter = BasicStreamPresenter(getLogger(__name__))

                items = await _collect_stream_items(
                    presenter,
                    _stream_request(config, snapshot),
                )

                frames = _frames(items)
                output = _visible_text(items)
                self.assertEqual([frame.role for frame in frames], roles)
                self.assertIn("done", output)
                self.assertIn(expected, output)
                for fragment in unexpected:
                    self.assertNotIn(fragment, output)

    async def test_stats_render_only_for_completed_terminal_snapshot(
        self,
    ) -> None:
        config = _stream_config(stats=True)
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("done")
        builder.update_token_counts(
            input_tokens=2,
            output_tokens=3,
            total_tokens=5,
        )
        builder.update_timing(elapsed_seconds=1.25)
        builder.add_usage_summary(
            {"input_tokens": 2, "output_tokens": 3},
            kind="stream.completed",
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        running = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        completed = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(running), ["done"])
        self.assertEqual(_frames(running), [])
        completed_frames = _frames(completed)
        self.assertEqual(_answer_chunks(completed), ["\n"])
        self.assertEqual([frame.role for frame in completed_frames], ["stats"])
        stats_text = str(completed_frames[0].renderable)
        self.assertIn("tokens in=2, out=3, total=5", stats_text)
        self.assertIn("elapsed 1.25s", stats_text)
        self.assertIn("usage stream.completed", stats_text)

    async def test_live_active_tool_spinner_starts_then_ages_to_running(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        now = 100.0

        def clock() -> float:
            return now

        builder.add_active_tool(
            tool_call_id="active-call",
            name="calc",
            started_at=99.0,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        with patch("avalan.cli.theme.basic.perf_counter", clock):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )
            frames = _frames(items)
            self.assertEqual([frame.role for frame in frames], ["tools"])
            self.assertIsInstance(frames[0].renderable, Spinner)
            first_text = _render_text(frames[0].renderable)
            now = 101.25
            second_text = _render_text(frames[0].renderable)
            now = 3700.25
            stale_text = _render_text(frames[0].renderable)

        self.assertIn("Starting tool calc", first_text)
        self.assertIn(
            "Running tool calc for",
            second_text,
        )
        self.assertIn("Starting tool calc", stale_text)

    async def test_live_diagnostics_pause_after_visible_answer_starts(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=True,
            display_events=True,
            display_tools_events=8,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(tool_call_id="active-call", name="calc")
        builder.add_event_summary(event_type=EventType.START)
        presenter = BasicStreamPresenter(getLogger(__name__))

        diagnostics = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.append_answer_text("Answer.")
        answer = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(
            [frame.role for frame in _frames(diagnostics)],
            ["tools", "events"],
        )
        self.assertEqual(_answer_chunks(answer), ["Answer."])
        self.assertEqual(
            [(frame.role, frame.renderable) for frame in _frames(answer)],
            [("tools", "")],
        )

    async def test_live_tool_history_is_not_repeated_during_answer(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_result_summary(
            tool_call_id="pdfinfo-call",
            name="pdfinfo",
            status="result",
            result="PDF metadata",
            arguments_count=1,
            elapsed_seconds=0.065,
        )
        builder.add_active_model_continuation(
            model_continuation_id="continuation-1",
            started_at=1.0,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            diagnostics = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        builder.append_answer_text("I")
        first_answer = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.append_answer_text(" matched and read the skill.")
        second_answer = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        diagnostics_text = _visible_text(diagnostics)
        first_answer_text = _visible_text(first_answer)
        second_answer_text = _visible_text(second_answer)
        self.assertIn("Executed tool pdfinfo", diagnostics_text)
        self.assertEqual(_answer_chunks(first_answer), ["I"])
        self.assertEqual(
            _answer_chunks(second_answer),
            [" matched and read the skill."],
        )
        self.assertIn("Executed tool pdfinfo", first_answer_text)
        self.assertNotIn("Executed tool pdfinfo", second_answer_text)
        self.assertEqual(_frames(second_answer), [])

    def test_live_tool_history_frame_handles_no_entries_or_notices(
        self,
    ) -> None:
        presenter = BasicStreamPresenter(getLogger(__name__))

        empty_frame = presenter._live_tool_history_frame(())
        notice_frame = presenter._live_tool_history_frame(
            (_BasicToolLineEntry(key="terminal:notice", line="Notice."),)
        )

        self.assertIsNone(empty_frame)
        self.assertEqual(notice_frame, "Notice.")
        self.assertFalse(presenter._executed_tool_frame_seen)

    async def test_display_tools_events_zero_keeps_active_and_clears_stale(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=0)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(
            tool_call_id="active-call",
            name="calc",
            arguments={"x": 1},
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        first = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.complete_tool(tool_call_id="active-call", name="calc")
        second = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        first_frames = _frames(first)
        second_frames = _frames(second)
        self.assertEqual(_answer_chunks(first), [])
        self.assertEqual([frame.role for frame in first_frames], ["tools"])
        self.assertIsInstance(first_frames[0].renderable, Spinner)
        first_text = _render_text(first_frames[0].renderable)
        self.assertIn("Starting tool calc", first_text)
        self.assertNotIn("completed", first_text)
        self.assertEqual(_answer_chunks(second), [])
        self.assertEqual([frame.role for frame in second_frames], ["tools"])
        self.assertEqual(second_frames[0].renderable, "")

    async def test_non_live_active_tools_remain_line_oriented(self) -> None:
        config = _stream_config(
            display_tools=True,
            interactive=False,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(
            tool_call_id="active-call",
            name="calc",
            arguments={"x": 1},
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        frames = _frames(items)
        self.assertEqual([frame.role for frame in frames], ["tools"])
        self.assertIsInstance(frames[0].renderable, str)
        self.assertIn(
            "Starting tool [bold]calc[/bold]", str(frames[0].renderable)
        )

    def test_active_model_and_tool_non_spinner_renderables(self) -> None:
        model_renderable = _basic_active_model_renderable(
            started_at=1.0,
            updated_at=2.0,
            spinner=False,
        )
        tool_renderable = _basic_active_tool_renderable(
            "calc",
            started_at=1.0,
            updated_at=2.0,
            spinner=False,
        )
        spinner = _BasicActiveModelSpinner(
            started_at=1.0,
            updated_at=2.0,
        )
        unknown_start_spinner = _BasicActiveModelSpinner(
            started_at=None,
            updated_at=None,
        )
        completed_model_line = _basic_completed_model_line(
            started_at=None,
            updated_at=None,
        )

        self.assertIn("Thinking for 1s...", str(model_renderable))
        self.assertIn(
            "Running tool [bold]calc[/bold] for 1s.",
            str(tool_renderable),
        )
        self.assertEqual(spinner._current_updated_at(), 2.0)
        self.assertIsNone(unknown_start_spinner._current_updated_at())
        self.assertEqual(completed_model_line, "[cyan]Thought.[/cyan]")

    async def test_stderr_tool_history_emits_only_new_lines(self) -> None:
        config = _stream_config(
            display_tools=True,
            display_tools_events=None,
            interactive=False,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(tool_call_id="call-1", name="calc")
        builder.complete_tool(tool_call_id="call-1", name="calc")
        builder.add_tool_result_summary(
            tool_call_id="call-1",
            name="calc",
            status="result",
            result=25,
            arguments_count=1,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        first_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        first_text = _render_text(_frames(first_items)[0].renderable)
        self.assertIn("Executed tool calc: 25", first_text)

        builder.add_active_tool(tool_call_id="call-2", name="database.run")
        builder.complete_tool(tool_call_id="call-2", name="database.run")
        builder.add_tool_result_summary(
            tool_call_id="call-2",
            name="database.run",
            status="result",
            result=[],
            arguments_count=1,
        )
        second_items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        second_text = _render_text(_frames(second_items)[0].renderable)
        self.assertIn("Executed tool database.run: []", second_text)
        self.assertNotIn("Executed tool calc", second_text)

    async def test_stderr_model_completion_keys_are_stable(self) -> None:
        config = _stream_config(
            display_tools=True,
            display_tools_events=None,
            interactive=False,
        )
        builder = CliStreamSnapshotBuilder(config)
        presenter = BasicStreamPresenter(getLogger(__name__))

        builder.add_active_model_continuation(
            model_continuation_id="continuation-1",
            started_at=1.0,
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )
        builder.finish_model_continuation(
            model_continuation_id="continuation-1",
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=2.0):
            first_completed = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        builder.add_active_model_continuation(
            model_continuation_id="continuation-2",
            started_at=4.0,
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=5.0):
            await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )
        builder.finish_model_continuation(
            model_continuation_id="continuation-2",
        )
        with patch("avalan.cli.theme.basic.perf_counter", return_value=6.0):
            second_completed = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        self.assertIn(
            "Thought for 1s.",
            _render_text(_frames(first_completed)[0].renderable),
        )
        self.assertIn(
            "Thought for 2s.",
            _render_text(_frames(second_completed)[0].renderable),
        )

    def test_projection_terminal_markup_renders_status_and_scalars(
        self,
    ) -> None:
        text = projection_terminal_markup(
            ToolDisplayProjection(
                action="inspect",
                status="completed",
                details=(
                    ToolDisplayDetail(label="cached", value=True),
                    ToolDisplayDetail(label="available", value=False),
                    ToolDisplayDetail(label="owner", value=None),
                ),
            )
        )
        mismatch_text = projection_terminal_markup(
            ToolDisplayProjection(
                action="inspect",
                status="completed",
                outcome="rows",
            ),
            fallback_status="fallback",
        )

        self.assertIn("completed", text)
        self.assertIn("cached=true", text)
        self.assertIn("available=false", text)
        self.assertIn("owner=none", text)
        self.assertIn("completed rows", mismatch_text)

    async def test_completed_empty_response_reports_no_final_answer(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=True,
            interactive=False,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(items), [])
        frames = _frames(items)
        self.assertEqual([frame.role for frame in frames], ["tools"])
        self.assertTrue(str(frames[0].renderable).startswith("\n"))
        rendered = _render_text(frames[0].renderable)
        self.assertIn("⚠️ No final answer emitted.", rendered)

    async def test_compact_formatting_handles_empty_and_truncated_summaries(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=4)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_event(Event(type=EventType.TOOL_PROCESS))
        snapshot = replace(
            builder.snapshot(),
            active_tools=(
                CliToolExecutionSummarySnapshot(
                    tool_call_id="long-call",
                    name="x" * 200,
                ),
            ),
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, snapshot),
        )

        frames = _frames(items)
        self.assertEqual([frame.role for frame in frames], ["tools"])
        tool_text = _render_text(frames[0].renderable)
        self.assertIn("x" * 157 + "...", tool_text.replace("\n", ""))
        self.assertIn("tool event tool_process", tool_text)
        self.assertNotIn("tool event tool_process:", tool_text)

    def test_compact_tool_elapsed_ranges_and_result_fallbacks(self) -> None:
        self.assertEqual(_basic_tool_elapsed_text(None), None)
        self.assertEqual(_basic_tool_elapsed_text(-1.0), "1ms")
        self.assertEqual(_basic_tool_elapsed_text(0.25), "250ms")
        self.assertEqual(_basic_tool_elapsed_text(12.4), "12s")
        self.assertEqual(_basic_tool_elapsed_text(75.0), "1m 15s")
        self.assertEqual(_basic_tool_elapsed_text(3720.0), "1h 02m")
        self.assertEqual(_basic_tool_result_summary("not json"), "not json")
        self.assertEqual(
            _basic_tool_result_summary('{"value": 25}'),
            '{"value": 25}',
        )

    def test_active_tool_edge_lines_keep_progress_style(self) -> None:
        self.assertIn(
            "[cyan]Starting tool [bold]calc[/bold].",
            _basic_active_tool_line(
                "calc",
                started_at=1.0,
                updated_at=1.5,
            ),
        )
        self.assertIn(
            "[cyan]Running tool [bold]calc[/bold].",
            _basic_active_tool_line(
                "calc",
                started_at=None,
                updated_at=2.0,
            ),
        )

    def test_executed_tool_frame_detection_respects_display_config(
        self,
    ) -> None:
        config = _stream_config(display_tools=False)
        builder = CliStreamSnapshotBuilder(config)

        self.assertFalse(
            _basic_has_executed_tool_frame(
                _stream_request(config, builder.snapshot())
            )
        )

    async def test_protocol_text_is_not_rendered_as_answer_or_diagnostics(
        self,
    ) -> None:
        config = _stream_config(
            stats=True,
            display_tools=True,
            display_events=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("Thought: secret\n")
        builder.append_answer_text(
            '<tool_call>{"name": "calc", "arguments": {}}</tool_call>'
        )
        builder.append_answer_text(
            "<|start|>assistant<|channel|>analysis<|message|>hidden<|end|>"
        )
        builder.append_answer_text('{"tool": "calc", "arguments": {"x": 1}}\n')
        builder.append_answer_text(
            "<DSML\uff5ctool_calls>"
            '<DSML\uff5cinvoke name="math.calculator">'
            "</DSML\uff5cinvoke>"
            "</DSML\uff5ctool_calls>"
        )
        builder.append_answer_text("25")
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        snapshot = replace(
            builder.snapshot(),
            projection_metadata_summaries=(
                CliProjectionMetadataSummarySnapshot(
                    sequence=1,
                    kind="chunk",
                    data_summary='{"protocol": "hidden"}',
                ),
            ),
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, snapshot),
        )

        self.assertEqual(_answer_chunks(items), ["25", "\n"])
        output = "\n".join(
            (
                _render_text(item.renderable)
                if isinstance(item, CliStreamRenderableFrame)
                else item.text
            )
            for item in items
            if isinstance(
                item,
                (CliStreamRenderableFrame, CliStreamAnswerTextChunk),
            )
        )
        self.assertIn("25", output)
        for fragment in (
            "ReACT",
            "Thought",
            "<tool_call>",
            '{"name": "calc"}',
            "hidden",
            "tool",
            "arguments",
            "DSML",
            "protocol",
        ):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, output)

    async def test_ordinary_json_answer_text_is_pretty_printed(self) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text('{"answer": 25}')
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(
            _answer_chunks(items),
            ['{\n  "answer": 25\n}', "\n"],
        )

    async def test_invalid_json_answer_text_is_not_pretty_printed(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text('{"answer":')
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(items), ['{"answer":', "\n"])

    async def test_incremental_protocol_prefix_is_withheld_and_suppressed(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        presenter = BasicStreamPresenter(getLogger(__name__))

        builder.append_answer_text("<")
        first = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.append_answer_text(
            'tool_call>{"name": "calc", "arguments": {}}</tool_call>'
        )
        second = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        completed = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(first), [])
        self.assertEqual(_answer_chunks(second), [])
        self.assertEqual(_answer_chunks(completed), [])

    async def test_incremental_protocol_prefix_can_resolve_as_answer(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        presenter = BasicStreamPresenter(getLogger(__name__))

        builder.append_answer_text("1 <")
        first = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        builder.append_answer_text(" 2")
        second = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertEqual(_answer_chunks(first), ["1 "])
        self.assertEqual(_answer_chunks(second), ["< 2"])

    async def test_answer_presenter_rejects_invalid_snapshot_sequence(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("long")
        presenter = _BasicAnswerPresenter()

        with self.assertRaisesRegex(
            AssertionError,
            "answer presenter requires answer mode",
        ):
            [
                item
                async for item in presenter.present(
                    _stream_request(config, builder.snapshot(), mode="live")
                )
            ]

        first = [
            item
            async for item in presenter.present(
                _stream_request(config, builder.snapshot(), mode="answer")
            )
        ]
        shorter = CliStreamSnapshotBuilder(config)
        shorter.append_answer_text("lo")

        with self.assertRaisesRegex(
            AssertionError,
            "answer snapshots must grow monotonically",
        ):
            [
                item
                async for item in presenter.present(
                    _stream_request(config, shorter.snapshot(), mode="answer")
                )
            ]

        self.assertEqual(_answer_chunks(first), ["long"])

    async def test_partial_json_tool_line_can_be_replaced_by_answer(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        presenter = _BasicAnswerPresenter()

        builder.append_answer_text('{"name": "calc"')
        first = [
            item
            async for item in presenter.present(
                _stream_request(config, builder.snapshot(), mode="answer")
            )
        ]
        builder.append_answer_text(', "arguments": {}}\nFinal')
        second = [
            item
            async for item in presenter.present(
                _stream_request(config, builder.snapshot(), mode="answer")
            )
        ]

        self.assertEqual(_answer_chunks(first), [])
        self.assertEqual(_answer_chunks(second), ["Final"])

    async def test_visible_answer_can_replace_previously_emitted_tail(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        presenter = _BasicAnswerPresenter()

        builder.append_answer_text('Visible\n{"name": "calc"')
        first = [
            item
            async for item in presenter.present(
                _stream_request(config, builder.snapshot(), mode="answer")
            )
        ]
        builder.append_answer_text(', "arguments": {}}\nFinal')
        second = [
            item
            async for item in presenter.present(
                _stream_request(config, builder.snapshot(), mode="answer")
            )
        ]

        self.assertEqual(_answer_chunks(first), ['Visible\n{"name": "calc"'])
        self.assertEqual(_answer_chunks(second), ["Visible\nFinal"])

    async def test_visible_answer_can_be_replaced_by_empty_protocol(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        presenter = _BasicAnswerPresenter()

        builder.append_answer_text('{"x": 1')
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        first = [
            item
            async for item in presenter.present(
                _stream_request(config, builder.snapshot(), mode="answer")
            )
        ]
        builder.append_answer_text(', "name": "calc", "arguments": {}}')
        second = [
            item
            async for item in presenter.present(
                _stream_request(config, builder.snapshot(), mode="answer")
            )
        ]

        self.assertEqual(_answer_chunks(first), ['{"x": 1'])
        self.assertEqual(_answer_chunks(second), [])

    def test_basic_filter_helper_edge_cases(self) -> None:
        self.assertEqual(
            _basic_visible_answer_text("", terminal_completed=True),
            "",
        )
        self.assertEqual(
            _basic_visible_answer_text(
                '{"tool": "calc", "arguments": {}}',
                terminal_completed=True,
            ),
            "",
        )
        self.assertEqual(
            _basic_visible_answer_text(
                '{\n  "tool": "calc",\n  "arguments": {}\n}',
                terminal_completed=True,
            ),
            "",
        )
        self.assertEqual(
            _basic_visible_answer_text(
                "answer\n   ",
                terminal_completed=False,
            ),
            "answer\n   ",
        )
        self.assertEqual(
            _basic_visible_answer_text("Thought", terminal_completed=False),
            "",
        )
        self.assertEqual(
            _basic_open_harmony_pattern("<|channel|>analysis pending").sub(
                "",
                "<|channel|>analysis pending",
            ),
            "",
        )
        self.assertFalse(_basic_json_tool_answer("not-json{"))
        self.assertTrue(_basic_json_tool_answer('{"tool_calls": ['))
        self.assertFalse(_basic_json_tool_answer('{"arguments": {'))
        self.assertTrue(
            _basic_json_tool_answer('{"name": "calc", "arguments": {')
        )

    async def test_quiet_suppresses_all_non_answer_noise(self) -> None:
        source_config = _stream_config(
            stats=True,
            display_tools=True,
            display_events=True,
            display_tools_events=8,
        )
        builder = CliStreamSnapshotBuilder(source_config)
        builder.append_answer_text("Thought: hidden\n")
        builder.append_answer_text(
            '<tool_call>{"name": "calc", "arguments": {}}</tool_call>'
        )
        builder.append_answer_text(
            "<|channel|>analysis<|message|>hidden<|end|>"
        )
        builder.append_answer_text("Clean answer.")
        builder.add_active_tool(tool_call_id="call", name="calc")
        builder.add_event_summary(event_type=EventType.START)
        builder.update_token_counts(input_tokens=1, output_tokens=2)
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        quiet_config = _stream_config(
            quiet=True,
            stats=True,
            display_tools=True,
            display_events=True,
            display_tools_events=8,
        )
        presenter = BasicStreamPresenter(getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(quiet_config, builder.snapshot()),
        )

        output = _visible_text(items)
        self.assertEqual(_frames(items), [])
        self.assertIn("Clean answer.", output)
        for fragment in (
            "Thought",
            "<tool_call",
            "arguments",
            "<|channel|>",
            "hidden",
            "tool calc",
            "event start",
            "tokens ",
            "Spinner",
            "[",
        ):
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, output)

    async def test_protocol_internals_are_suppressed_across_basic_modes(
        self,
    ) -> None:
        protocol_text = (
            "Thought: use a tool\n"
            "Action: math.calculator\n"
            'Action Input: {"expression": "(4 + 6) * 5 / 2"}\n'
            "Observation: 25\n"
            "```tool_call\n"
            '{"name": "math.calculator", "arguments": {"expression": "x"}}\n'
            "```\n"
            '<function_call>{"name": "math.calculator"}</function_call>'
            "<DSML:tool_calls>"
            '<DSML:invoke name="math.calculator"></DSML:invoke>'
            "</DSML:tool_calls>"
            "<|start|>assistant<|channel|>analysis<|message|>"
            "internal plan<|end|>"
            '{"tool_calls": [{"name": "math.calculator"}]}\n'
            '{"type": "tool_call", "name": "math.calculator", '
            '"arguments": {}}\n'
            "Final result: 25"
        )
        cases = [
            ("default", _stream_config()),
            (
                "quiet",
                _stream_config(
                    quiet=True,
                    stats=True,
                    display_tools=True,
                    display_events=True,
                ),
            ),
            (
                "non-interactive",
                _stream_config(
                    interactive=False,
                    stats=True,
                    display_tools=True,
                    display_events=True,
                ),
            ),
            ("display-tools", _stream_config(display_tools=True)),
        ]

        for label, config in cases:
            with self.subTest(label=label):
                builder = CliStreamSnapshotBuilder(config)
                builder.append_answer_text(protocol_text)
                builder.set_terminal(
                    completed=True,
                    outcome=StreamTerminalOutcome.COMPLETED,
                )
                presenter = BasicStreamPresenter(getLogger(__name__))

                items = await _collect_stream_items(
                    presenter,
                    _stream_request(config, builder.snapshot()),
                )

                answer_text = "".join(_answer_chunks(items))
                self.assertIn("Final result: 25", answer_text)
                for fragment in (
                    "Thought:",
                    "Action:",
                    "Action Input:",
                    "Observation:",
                    "```tool_call",
                    "<function_call",
                    "DSML",
                    "<|channel|>",
                    "internal plan",
                    "tool_calls",
                    '"arguments"',
                ):
                    self.assertNotIn(fragment, answer_text)

    async def test_malformed_tool_diagnostics_are_concise_and_requested_only(
        self,
    ) -> None:
        config = _stream_config(display_tools=True, display_tools_events=4)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_diagnostic(
            ToolCallDiagnostic(
                id="diag",
                call_id="call",
                requested_name="broken_tool",
                code=ToolCallDiagnosticCode.MALFORMED_CALL,
                stage=ToolCallDiagnosticStage.PARSE,
                message="Malformed tool call " + "x" * 220,
                details={
                    "raw": "<tool_call>raw-secret</tool_call>",
                    "trace": "hidden traceback",
                },
            )
        )
        snapshot = builder.snapshot()

        hidden = await _collect_stream_items(
            BasicStreamPresenter(getLogger(__name__)),
            _stream_request(_stream_config(), snapshot),
        )
        shown = await _collect_stream_items(
            BasicStreamPresenter(getLogger(__name__)),
            _stream_request(config, snapshot),
        )

        self.assertEqual(_frames(hidden), [])
        frames = _frames(shown)
        self.assertEqual([frame.role for frame in frames], ["tools"])
        tool_text = _render_text(frames[0].renderable)
        self.assertIn(
            "tool broken_tool diagnostic tool_call.malformed:",
            tool_text,
        )
        self.assertIn("Malformed tool call", tool_text)
        self.assertIn("...", tool_text)
        self.assertLess(len(tool_text), 260)
        self.assertNotIn("raw-secret", tool_text)
        self.assertNotIn("hidden traceback", tool_text)
        self.assertNotIn("parse", tool_text)


class BasicThemeTestCase(unittest.TestCase):
    def test_importing_basic_theme_does_not_import_fancy(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys\n"
                    "import avalan.cli.theme.basic\n"
                    "print('avalan.cli.theme.fancy' in sys.modules)"
                ),
            ],
            capture_output=True,
            check=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False")

    def test_basic_theme_instantiates_as_concrete_theme(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)

        self.assertIsInstance(theme, Theme)
        self.assertNotIsInstance(theme, FancyTheme)
        self.assertEqual(theme._("message"), "translated:message")
        self.assertEqual(theme._n("one", "many", 2), "many")

    def test_basic_theme_returns_basic_stream_presenter(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)

        presenter = theme.stream_presenter(getLogger(__name__))

        self.assertIsInstance(presenter, BasicStreamPresenter)

    def test_basic_theme_inherits_common_defaults(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)

        self.assertEqual(theme.icons["user_input"], ":speaking_head:")
        self.assertEqual(theme.icons["agent_output"], ":robot:")
        self.assertTrue(theme.default_display_tools)
        self.assertTrue(theme.prefix_stream_answers)
        self.assertEqual(
            theme.ask_access_token(),
            "translated:Enter your Huggingface access token",
        )
        self.assertEqual(theme.events([]), None)
        self.assertEqual(
            theme.flow_run_progress(
                "flowchart LR\n",
                node_states={},
                active_nodes=(),
                message="Flow run started.",
                console_width=80,
            ),
            "Flow run started.\n\nflowchart LR\n",
        )
        self.assertIsInstance(theme.download_progress(), tuple)

    def test_basic_agent_header_embeds_model_information(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        agent = SimpleNamespace(
            id="agent-id",
            name="Tool",
            memory=SimpleNamespace(
                has_recent_message=True,
                has_permanent_message=False,
            ),
        )

        welcome = _render_text(
            theme.welcome(
                "https://avalan.ai",
                "avalan",
                "1.5.0",
                "MIT",
                None,
            )
        )
        output = _render_text(
            theme.agent(cast(Any, agent), models=[_model(), "gpt-5-mini"])
        )

        self.assertEqual(welcome, "")
        self.assertFalse(output.endswith("\n\n"))
        self.assertIn("⭕ avalan 1.5.0 - MIT License", output)
        self.assertIn("✨", output)
        self.assertIn("Tool", output)
        self.assertIn("model-id", output)
        self.assertIn("gpt-5-mini", output)
        self.assertIn("translated:short-term message", output)
        self.assertNotIn("translated:Models", output)
        self.assertNotIn("translated:Memory", output)

        stateless_output = _render_text(
            theme.agent(
                cast(Any, SimpleNamespace(id="agent-id")),
                models=[],
            )
        )
        self.assertIn("translated:stateless", stateless_output)
        self.assertIn("translated:none", stateless_output)

    def test_basic_theme_inherits_low_clutter_domain_outputs(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        tool_call = ToolCall(
            id="call-1",
            name="calc[red]",
            arguments={"password": "hidden", "x": 1},
        )

        events = theme.events(
            [Event(type=EventType.TOOL_EXECUTE, payload={"call": tool_call})]
        )
        tokenizer = theme.tokenizer_config(
            TokenizerConfig(
                name_or_path="tok[red]",
                tokens=None,
                special_tokens=None,
                tokenizer_model_max_length=128,
            )
        )
        model = theme.model(_model(), can_access=True, summary=True)

        assert events is not None
        self.assertIn(r"calc\[red]", events)
        self.assertIn("<redacted>", events)
        self.assertNotIn("hidden", events)
        self.assertIn(r"translated:Tokenizer: tok\[red]", tokenizer)
        self.assertIn("translated:Fast: translated:no", tokenizer)
        self.assertIn("translated:Model: model-id", model)
        self.assertIn("translated:Access: translated:yes", model)
        self.assertNotIn("translated:Downloads", model)

    def test_cache_list_command_renders_with_basic_theme(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        console = MagicMock()
        hub = MagicMock()
        hub.cache_dir = "/cache"
        hub.cache_scan.return_value = []

        cache_cmds.cache_list(
            Namespace(model=None, summary=False),
            console,
            theme,
            hub,
        )

        console.print.assert_called_once_with(
            "translated:Cache: /cache\nModels: translated:none"
        )

    def test_cache_download_command_uses_basic_progress_template(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        console = MagicMock()
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = _model()
        hub.download.return_value = "/models/model-id"

        cache_cmds.cache_download(
            Namespace(
                model="model-id",
                skip_hub_access_check=False,
                workers=3,
                local_dir="/models",
                local_dir_symlinks=True,
            ),
            console,
            theme,
            hub,
        )

        hub.download.assert_called_once()
        download_kwargs = hub.download.call_args.kwargs
        self.assertTrue(
            issubclass(download_kwargs["tqdm_class"], tqdm_rich_progress)
        )
        self.assertEqual(download_kwargs["workers"], 3)
        self.assertEqual(download_kwargs["local_dir"], "/models")
        self.assertTrue(download_kwargs["local_dir_use_symlinks"])
        self.assertEqual(console.print.call_count, 3)

    def test_model_display_command_renders_with_basic_theme(self) -> None:
        theme = BasicTheme(_gettext, _ngettext)
        console = MagicMock()
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.return_value = _model()
        manager = MagicMock()
        manager.__enter__.return_value = manager
        manager.__exit__.return_value = False
        manager.parse_uri.return_value = SimpleNamespace(is_local=False)
        loaded_model = SimpleNamespace(
            config=SimpleNamespace(model_type="text-generation"),
            tokenizer_config=None,
        )

        with patch.object(model_cmds, "ModelManager", return_value=manager):
            model_cmds.model_display(
                Namespace(
                    model="model-id",
                    skip_hub_access_check=False,
                    summary=False,
                    load=False,
                ),
                console,
                theme,
                hub,
                getLogger(__name__),
                model=loaded_model,
            )

        hub.can_access.assert_called_once_with("model-id")
        hub.model.assert_called_once_with("model-id")
        self.assertEqual(console.print.call_count, 2)
