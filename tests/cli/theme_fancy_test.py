import unittest
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from io import StringIO
from logging import getLogger
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch
from uuid import UUID

import numpy as np
from numpy.linalg import norm
from rich import box
from rich.console import Console
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.display_snapshot import (
    CliDisplayTokenSnapshot,
    CliProjectionMetadataSummarySnapshot,
    CliStreamSnapshot,
    CliStreamSnapshotBuilder,
)
from avalan.cli.theme import (
    TokenRenderDisplayToken,
    TokenRenderDisplayTokenCandidate,
    TokenRenderState,
)
from avalan.cli.theme import (
    fancy as fancy_theme_module,
)
from avalan.cli.theme.fancy import FancyStreamPresenter, FancyTheme
from avalan.cli.theme.stream_presenter import (
    CliStreamAnswerTextChunk,
    CliStreamPresenterContext,
    CliStreamPresenterRequest,
    CliStreamRenderableFrame,
)
from avalan.entities import (
    EngineMessage,
    EngineMessageScored,
    HubCache,
    HubCacheDeletion,
    HubCacheFile,
    ImageEntity,
    Message,
    ModelConfig,
    SearchMatch,
    SentenceTransformerModelConfig,
    Similarity,
    Token,
    TokenDetail,
    TokenizerConfig,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    User,
)
from avalan.event import (
    Event,
    EventObservabilityPayload,
    EventStats,
    EventType,
)
from avalan.memory.partitioner.text import TextPartition
from avalan.memory.permanent import PermanentMemoryPartition
from avalan.model.stream import (
    StreamChannel,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
)
from avalan.tool.display import ToolDisplayDetail, ToolDisplayProjection


@dataclass(frozen=True, kw_only=True, slots=True)
class _DisplayToken(Token):
    tokens: list[Token] | None = None


def _render_display_token(
    token: Token, sequence: int = 1
) -> TokenRenderDisplayToken:
    candidates = tuple(
        TokenRenderDisplayTokenCandidate(
            token=candidate.token,
            id=candidate.id if isinstance(candidate.id, int) else None,
            probability=candidate.probability,
        )
        for candidate in getattr(token, "tokens", []) or []
    )
    probability_distribution = getattr(token, "probability_distribution", None)
    return TokenRenderDisplayToken(
        sequence=sequence,
        kind=StreamItemKind.ANSWER_DELTA,
        channel=StreamChannel.ANSWER,
        token=token.token,
        id=token.id if isinstance(token.id, int) else None,
        probability=token.probability,
        step=getattr(token, "step", None),
        probability_distribution=(
            str(probability_distribution)
            if probability_distribution is not None
            else None
        ),
        tokens=candidates,
    )


def _token_state(
    *,
    model_id: str = "m",
    added_tokens: list[str] | None = None,
    special_tokens: list[str] | None = None,
    display_token_size: int | None = None,
    display_probabilities: bool = False,
    pick: int = 0,
    focus_on_token_when: (
        Callable[[TokenRenderDisplayToken], bool] | None
    ) = None,
    thinking_text_tokens: list[str] | None = None,
    tool_text_tokens: list[str] | None = None,
    answer_text_tokens: list[str] | None = None,
    tokens: list[Token] | None = None,
    input_token_count: int = 0,
    total_tokens: int = 0,
    tool_running_spinner: Spinner | None = None,
    ttft: float | None = None,
    ttnt: float | None = None,
    ttsr: float | None = None,
    elapsed: float = 0.0,
    event_stats: EventStats | None = None,
    tool_token_count: int = 0,
    display_reasoning: bool | None = None,
    display_tools: bool | None = None,
) -> TokenRenderState:
    reasoning_tokens = tuple(thinking_text_tokens or [])
    tool_tokens = tuple(tool_text_tokens or [])
    display_tokens = tuple(
        _render_display_token(token, sequence)
        for sequence, token in enumerate(tokens or [], start=1)
    )
    return TokenRenderState(
        model_id=model_id,
        added_tokens=tuple(added_tokens) if added_tokens else None,
        special_tokens=tuple(special_tokens) if special_tokens else None,
        display_token_size=display_token_size,
        display_probabilities=display_probabilities,
        pick=pick,
        focus_on_token_when=focus_on_token_when,
        reasoning_text_tokens=reasoning_tokens,
        tool_text_tokens=tool_tokens,
        answer_text_tokens=tuple(answer_text_tokens or []),
        display_tokens=display_tokens,
        display_reasoning=(
            bool(reasoning_tokens)
            if display_reasoning is None
            else display_reasoning
        ),
        display_tools=(
            bool(tool_tokens) if display_tools is None else display_tools
        ),
        input_token_count=input_token_count,
        total_tokens=total_tokens,
        tool_token_count=tool_token_count,
        tool_running=tool_running_spinner is not None,
        tool_running_spinner=tool_running_spinner,
        ttft=ttft,
        ttnt=ttnt,
        ttsr=ttsr,
        elapsed=elapsed,
        event_stats=event_stats,
    )


def _stream_config(**overrides: object) -> CliStreamDisplayConfig:
    values = {
        "quiet": False,
        "stats": True,
        "display_tools": True,
        "display_events": True,
        "display_tools_events": 2,
        "record": False,
        "interactive": True,
        "refresh_per_second": 10,
        "answer_height": 12,
        "answer_height_expand": False,
        "display_tokens": 4,
        "display_pause": 0,
        "display_probabilities": True,
        "display_probabilities_maximum": 0.8,
        "display_probabilities_sample_minimum": 0.1,
        "display_time_to_n_token": 2,
        "display_reasoning_time": True,
    }
    values.update(overrides)
    return CliStreamDisplayConfig(**values)


def _stream_context(**overrides: object) -> CliStreamPresenterContext:
    values = {
        "model_id": "model",
        "console_width": 96,
        "input_token_count": 3,
        "tokenizer_tokens": ("added",),
        "tokenizer_special_tokens": ("<eos>",),
        "token_probability_pick": 3,
        "start_thinking": False,
    }
    values.update(overrides)
    return CliStreamPresenterContext(**values)


def _stream_request(
    config: CliStreamDisplayConfig,
    snapshot: CliStreamSnapshot,
    *,
    context: CliStreamPresenterContext | None = None,
    mode: str = "live",
) -> CliStreamPresenterRequest:
    return CliStreamPresenterRequest(
        snapshot=snapshot,
        display_config=config,
        context=_stream_context() if context is None else context,
        mode=mode,  # type: ignore[arg-type]
    )


async def _collect_stream_items(
    presenter: FancyStreamPresenter,
    request: CliStreamPresenterRequest,
) -> list[object]:
    return [item async for item in presenter.present(request)]


def _render_visible_text(*renderables: object) -> str:
    console = Console(file=StringIO(), record=True, width=240)
    for renderable in renderables:
        console.print(renderable)
    return console.export_text()


def _frame_text(frame: CliStreamRenderableFrame) -> str:
    return _render_visible_text(frame.renderable)


class FancyThemeFlowProgressTestCase(unittest.TestCase):
    def test_flow_run_progress_message_reports_active_node(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)

        message = theme.flow_run_progress_message(
            "flow_node_started",
            node="analyze_pov_1",
            attempt=2,
        )

        self.assertIn("Running", message)
        self.assertIn("analyze_pov_1", message)
        self.assertIn("(attempt 2)", message)

    def test_flow_run_progress_message_omits_first_attempt(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)

        message = theme.flow_run_progress_message(
            "flow_node_started",
            node="analyze_pov_1",
            attempt=1,
        )

        self.assertIn("Running", message)
        self.assertIn("analyze_pov_1", message)
        self.assertNotIn("attempt", message)

    def test_flow_run_progress_message_covers_remaining_events(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        cases = {
            "flow_node_retrying": (
                "Retrying [cyan]node_a[/cyan] after attempt 2."
            ),
            "flow_node_completed": "Finished [cyan]node_a[/cyan].",
            "flow_node_failed": "[cyan]node_a[/cyan] failed.",
            "flow_node_skipped": "Skipped [cyan]node_a[/cyan].",
            "flow_node_paused": "Paused [cyan]node_a[/cyan].",
            "flow_node_resumed": "Resumed [cyan]node_a[/cyan].",
            "flow_node_cancelled": "Cancelled [cyan]node_a[/cyan].",
        }

        for event_type, expected in cases.items():
            with self.subTest(event_type=event_type):
                self.assertEqual(
                    theme.flow_run_progress_message(
                        event_type,
                        node="node_a",
                        attempt=2,
                    ),
                    expected,
                )

        self.assertEqual(
            theme.flow_run_progress_message("flow_cancelled"),
            "Flow run cancelled.",
        )
        self.assertEqual(
            theme.flow_run_progress_message("flow_started"),
            "Flow run started.",
        )
        self.assertEqual(
            theme.flow_run_progress_message("flow_completed"),
            "Flow run completed.",
        )
        self.assertEqual(
            theme.flow_run_progress_message("unknown"),
            "Flow run is active.",
        )

    def test_flow_run_progress_inverts_running_node_class(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        captured: dict[str, str] = {}

        def fake_render(source: str, console_width: int) -> Text:
            captured["source"] = source
            captured["width"] = str(console_width)
            return Text("diagram")

        with patch(
            "avalan.cli.theme.fancy._flow_run_mermaid_renderable",
            fake_render,
        ):
            theme.flow_run_progress(
                "flowchart LR\n  analyze_pov_1[analyze_pov_1]\n",
                node_states={"analyze_pov_1": "pending"},
                active_nodes=("analyze_pov_1",),
                message="Running node [cyan]analyze_pov_1[/cyan].",
                console_width=120,
            )

        self.assertIn(
            "classDef avalanRunning fill:#ecfeff,stroke:#0f172a",
            captured["source"],
        )
        self.assertIn(
            "class analyze_pov_1 avalanRunning",
            captured["source"],
        )

    def test_flow_run_progress_renders_sidebar_stats(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        console = Console(file=StringIO(), record=True, width=160)
        renderable = theme.flow_run_progress(
            "flowchart LR\n  analyze_pov_1[analyze_pov_1]\n",
            node_states={
                "analyze_pov_1": "succeeded",
                "write_spec_request": "failed",
            },
            active_nodes=(),
            message="Finished [cyan]analyze_pov_1[/cyan].",
            console_width=160,
            flow_stats={
                "__total__": {
                    "elapsed_ms": 1500,
                    "executed_nodes": 2,
                    "succeeded_nodes": 1,
                    "failed_nodes": 1,
                    "average_node_ms": 750,
                    "input_tokens": 1234,
                    "cached_input_tokens": 99,
                    "output_tokens": 56,
                    "reasoning_tokens": 7,
                    "tools_executed": 3,
                },
                "analyze_pov_1": {
                    "elapsed_ms": 1000,
                    "input_tokens": 12,
                    "cached_input_tokens": 4,
                    "output_tokens": 34,
                    "reasoning_tokens": 5,
                    "tools_executed": 2,
                },
            },
        )

        console.print(renderable)
        output = console.export_text()

        self.assertIn("Stats", output)
        self.assertIn("Nodes", output)
        self.assertIn("time", output)
        self.assertIn("nodes", output)
        self.assertIn("cached", output)
        self.assertIn("tool", output)
        self.assertIn("1.5s", output)
        self.assertIn("1,234", output)
        self.assertIn("out 56 🔥", output)
        self.assertNotIn("nodes 🔗", output)
        self.assertNotIn("ok ✅", output)
        self.assertNotIn("out 🚀", output)
        self.assertRegex(output, "📨\\s+12")
        self.assertRegex(output, "💾\\s+33%")
        self.assertRegex(output, "💬\\s+34")
        self.assertRegex(output, "🧠\\s+5")
        self.assertRegex(output, "🛠\\s+2")
        self.assertNotRegex(output, "[\\u2800-\\u28ff]")

    def test_flow_run_stats_header_expands_columns(self):
        console = Console(file=StringIO(), record=True, width=100)
        console.print(
            fancy_theme_module._flow_run_stats_header(
                {
                    "__total__": {
                        "elapsed_ms": 1500,
                        "executed_nodes": 7,
                        "succeeded_nodes": 7,
                        "failed_nodes": 0,
                        "average_node_ms": 3100,
                        "input_tokens": 2092,
                        "cached_input_tokens": 1536,
                        "output_tokens": 94,
                        "reasoning_tokens": 0,
                        "tools_executed": 0,
                    }
                }
            )
        )

        stat_line = next(
            line
            for line in console.export_text().splitlines()
            if "nodes 7" in line
        )
        self.assertGreaterEqual(stat_line.index("tool"), 75)

    def test_flow_run_progress_sanitizes_negative_stats(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        console = Console(file=StringIO(), record=True, width=120)
        renderable = theme.flow_run_progress(
            "flowchart LR\n  analyze_pov_1[analyze_pov_1]\n",
            node_states={"analyze_pov_1": "running"},
            active_nodes=("analyze_pov_1",),
            message="Running [cyan]analyze_pov_1[/cyan].",
            console_width=120,
            flow_stats={
                "__total__": {
                    "elapsed_ms": -1,
                    "executed_nodes": True,
                    "succeeded_nodes": -2,
                    "failed_nodes": -3,
                    "average_node_ms": -4,
                    "input_tokens": -5,
                    "cached_input_tokens": -9,
                    "output_tokens": -6,
                    "reasoning_tokens": -7,
                    "tools_executed": -8,
                }
            },
        )

        console.print(renderable)
        output = console.export_text()

        self.assertIn("0ms", output)
        self.assertNotIn("-5", output)

    def test_flow_run_styled_mermaid_skips_unsafe_node(self):
        source = fancy_theme_module._flow_run_styled_mermaid_source(
            "flowchart LR\n  bad-node[bad-node]\n",
            node_states={
                "bad-node": "running",
                "safe_node": "unknown",
            },
            active_nodes=(),
        )

        self.assertNotIn("class bad-node", source)
        self.assertNotIn("class safe_node", source)

    def test_flow_run_helper_branches(self):
        self.assertEqual(
            fancy_theme_module._flow_run_node_class("skipped"),
            "avalanSkipped",
        )
        self.assertEqual(
            fancy_theme_module._flow_run_node_class("paused"),
            "avalanPaused",
        )
        self.assertIsNone(fancy_theme_module._flow_run_node_class("pending"))
        self.assertIsNone(fancy_theme_module._flow_run_status_table({}))
        self.assertEqual(
            fancy_theme_module._flow_run_format_duration(61_000),
            "1m01s",
        )
        self.assertEqual(
            fancy_theme_module._flow_run_format_duration(3_660_000),
            "1h01m",
        )
        self.assertEqual(
            fancy_theme_module._flow_run_state_display("skipped")[0],
            "-",
        )
        self.assertEqual(
            fancy_theme_module._flow_run_state_display("paused")[0],
            "=",
        )

    def test_flow_run_mermaid_renderable_falls_back_without_termaid(self):
        with patch(
            "avalan.cli.theme.fancy.import_module",
            side_effect=ImportError,
        ):
            renderable = fancy_theme_module._flow_run_mermaid_renderable(
                "flowchart LR\n",
                80,
            )

        self.assertIsInstance(renderable, Text)

    def test_flow_run_mermaid_renderable_falls_back_without_renderer(self):
        module = SimpleNamespace(render_rich=None)
        with patch(
            "avalan.cli.theme.fancy.import_module",
            return_value=module,
        ):
            renderable = fancy_theme_module._flow_run_mermaid_renderable(
                "flowchart LR\n",
                80,
            )

        self.assertIsInstance(renderable, Text)

    def test_flow_run_mermaid_renderable_reports_renderer_error(self):
        def render_rich(*_args, **_kwargs):
            raise RuntimeError("private renderer detail")

        module = SimpleNamespace(render_rich=render_rich)
        with patch(
            "avalan.cli.theme.fancy.import_module",
            return_value=module,
        ):
            renderable = fancy_theme_module._flow_run_mermaid_renderable(
                "flowchart LR\n",
                80,
            )

        console = Console(file=StringIO(), record=True, width=120)
        console.print(renderable)
        output = console.export_text()

        self.assertIn("Diagram renderer unavailable", output)
        self.assertIn("RuntimeError", output)


class FancyStreamPresenterTestCase(IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    async def test_presenter_renders_snapshot_native_rich_surfaces(
        self,
    ) -> None:
        config = _stream_config(
            display_tools_events=None,
            display_reasoning=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_reasoning_text("think before acting\n")
        builder.append_tool_call_request_text('{"name": "calc"}')
        builder.append_answer_text("Answer is 25.")
        builder.update_token_counts(input_tokens=7, output_tokens=8)
        builder.update_timing(
            started_at=1.0,
            updated_at=3.0,
            first_token_seconds=0.2,
            reasoning_seconds=0.4,
            time_to_n_token_seconds=0.6,
            elapsed_seconds=2.0,
        )
        builder.add_active_tool(
            tool_call_id="active-call",
            name="calc",
            arguments={"expr": "(4 + 6) * 5 / 2"},
        )
        builder.add_event_summary(
            event_type=EventType.FLOW_NODE_STARTED,
            payload={"node": "math"},
            elapsed=1.2,
        )
        builder.add_usage_summary(
            {"input_tokens": 7, "output_tokens": 8},
            kind="stream.completed",
        )
        builder.add_display_token(
            TokenDetail(
                id=1,
                token="25",
                probability=0.3,
                step=4,
                probability_distribution="softmax",
                tokens=[
                    Token(id=1, token="25", probability=0.3),
                    Token(id=2, token="24", probability=0.7),
                    Token(id=3, token="26", probability=0.1),
                ],
            )
        )
        builder.add_display_token(Token(id=4, token="low", probability=0.4))
        builder.add_display_token(Token(id=5, token="added", probability=0.9))
        builder.add_display_token(Token(id=6, token="plain", probability=0.9))
        snapshot = replace(
            builder.snapshot(),
            projection_metadata_summaries=(
                CliProjectionMetadataSummarySnapshot(
                    sequence=11,
                    kind="chunk",
                    data_summary="projection-data",
                ),
            ),
        )
        event_stats = EventStats()
        event_stats.total_triggers = 3
        event_stats.triggers = {
            EventType.TOOL_EXECUTE: 1,
            EventType.TOOL_RESULT: 1,
        }
        presenter = FancyStreamPresenter(
            self.theme,
            getLogger(__name__),
            event_stats=event_stats,
        )

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, snapshot),
        )

        frames = [
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
        ]
        self.assertEqual(
            [frame.role for frame in frames],
            ["reasoning", "tools", "events", "stats", "stream"],
        )
        current_token = frames[-1].current_token
        self.assertIsNotNone(current_token)
        assert current_token is not None
        self.assertEqual(current_token.token_id, 1)
        self.assertEqual(current_token.text, "25")
        self.assertEqual(current_token.step, 4)
        self.assertEqual(current_token.probability_distribution, "softmax")
        self.assertEqual(current_token.candidates[0].text, "25")
        self.assertEqual(current_token.candidates[1].probability, 0.7)
        output = _render_visible_text(*(frame.renderable for frame in frames))
        for fragment in (
            "Tool calls",
            "Executing tool calc",
            "active-c",
            "Events",
            "flow_node_started",
            "math",
            "Stats",
            "usage stream.completed",
            "projection chunk",
            "projection-data",
            "model Reasoning",
            "think before acting",
            "Tool call requests",
            '{"name": "calc"}',
            "Answer is 25.",
            "token distribution",
            "25",
            "70%",
            "3 events",
            "1 tool call",
            "1 result",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, output)

    async def test_reasoning_gate_prevents_hidden_panel_construction(
        self,
    ) -> None:
        config = _stream_config(display_reasoning=False)
        builder = CliStreamSnapshotBuilder(config)
        builder.append_reasoning_text("private reasoning")
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        with patch(
            "avalan.cli.theme.fancy.reasoning_display_blocks",
            side_effect=AssertionError("hidden reasoning was materialized"),
        ):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot()),
            )

        self.assertNotIn(
            "reasoning",
            [
                item.role
                for item in items
                if isinstance(item, CliStreamRenderableFrame)
            ],
        )

    async def test_reasoning_panels_label_mixed_representations(self) -> None:
        config = _stream_config(
            stats=False,
            display_tools=False,
            display_events=False,
            display_tokens=0,
            display_reasoning=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_reasoning_text("native plan")
        builder.append_reasoning_text(
            "summary plan",
            representation=StreamReasoningRepresentation.SUMMARY,
            segment_instance_ordinal=1,
            follows_completion=True,
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        reasoning_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "reasoning"
        )
        output = _frame_text(reasoning_frame)
        self.assertIn("model Reasoning", output)
        self.assertIn("native plan", output)
        self.assertIn("model Reasoning summary", output)
        self.assertIn("summary plan", output)

    async def test_blank_reasoning_block_creates_no_panel(self) -> None:
        config = _stream_config(
            stats=False,
            display_tools=False,
            display_events=False,
            display_tokens=0,
            display_reasoning=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_reasoning_text(" \n ")
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertNotIn(
            "reasoning",
            [
                item.role
                for item in items
                if isinstance(item, CliStreamRenderableFrame)
            ],
        )

    async def test_noninteractive_stats_reasoning_keeps_json_answer_chunk(
        self,
    ) -> None:
        config = _stream_config(
            interactive=False,
            display_tools=False,
            display_events=False,
            display_tokens=0,
            display_reasoning=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_reasoning_text(
            "summary",
            representation=StreamReasoningRepresentation.SUMMARY,
        )
        builder.append_answer_text('{"ok":true}')
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        self.assertIn(
            "reasoning",
            [
                item.role
                for item in items
                if isinstance(item, CliStreamRenderableFrame)
            ],
        )
        self.assertEqual(
            [
                item.text
                for item in items
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ['{\n  "ok": true\n}'],
        )

    async def test_tool_panel_prefers_display_projection_fields(self) -> None:
        config = _stream_config(display_tools_events=8, display_events=False)
        builder = CliStreamSnapshotBuilder(config)
        active_projection = ToolDisplayProjection(
            action="search",
            target="[TODO]",
            scope="src/[avalan]",
            summary="Search [red]source[/red].",
        )
        builder.add_active_tool(
            tool_call_id="active-call",
            name="shell.run",
            arguments={"query": "raw-json"},
            display_projection=active_projection,
        )
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
        builder.add_tool_result_summary(
            tool_call_id="done-call",
            name="database.query",
            status="result",
            result={"raw": "json"},
            arguments_count=1,
            display_projection=terminal_projection,
            elapsed_seconds=0.5,
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        tools_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "tools"
        )
        output = _frame_text(tools_frame)
        self.assertIn("Executing tool search [TODO] in src/[avalan]", output)
        self.assertIn("call #active-c", output)
        self.assertIn("Executed tool inspect [users] in database", output)
        self.assertIn("call #done-cal", output)
        self.assertIn("with status completed and outcome rows", output)
        self.assertIn("Returned [green]2 rows[/green].", output)
        self.assertIn("Details [table]=[users], rows=2", output)
        self.assertNotIn("raw-json", output)
        self.assertNotIn('"query"', output)
        self.assertNotIn('"raw"', output)
        self.assertNotIn('"json"', output)

    async def test_tool_panel_completed_without_result_uses_projection(
        self,
    ) -> None:
        config = _stream_config(display_tools_events=8, display_events=False)
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
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        tools_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "tools"
        )
        output = _frame_text(tools_frame)
        self.assertIn("Completed tool cancel [task-1] in database", output)
        self.assertIn("call #database", output)
        self.assertIn(
            "with status completed and outcome cancel_requested",
            output,
        )
        self.assertIn("Cancellation [yellow]requested[/yellow].", output)
        self.assertIn("Details task_id=[task-1]", output)
        self.assertNotIn("raw-json", output)

    async def test_tool_panel_bounds_long_projection_values(self) -> None:
        config = _stream_config(display_tools_events=8, display_events=False)
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
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        tools_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "tools"
        )
        output = _frame_text(tools_frame)
        self.assertIn("inspect " + ("x" * 20), output)
        self.assertIn("...", output)
        self.assertNotIn('"rows": ["raw"]', output)

    async def test_progress_title_pluralization_singular(self) -> None:
        config = _stream_config(display_tools=False, display_events=False)
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("a", tokens=1)
        builder.update_token_counts(input_tokens=1, output_tokens=1)
        event_stats = EventStats()
        event_stats.triggers = {
            EventType.TOOL_EXECUTE: 1,
            EventType.TOOL_RESULT: 1,
        }
        event_stats.total_triggers = 1
        presenter = FancyStreamPresenter(
            self.theme,
            getLogger(__name__),
            event_stats=event_stats,
        )

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        output = _frame_text(stream_frame)
        self.assertIn("1 token in", output)
        self.assertIn("1 token out", output)
        self.assertIn("1 event", output)
        self.assertIn("1 tool call", output)
        self.assertIn("1 result", output)

    async def test_progress_title_pluralization_plural(self) -> None:
        config = _stream_config(display_tools=False, display_events=False)
        builder = CliStreamSnapshotBuilder(config)
        builder.append_tool_call_request_text("tool", tokens=3)
        builder.append_answer_text("answer", tokens=2)
        builder.update_token_counts(input_tokens=2, output_tokens=2)
        event_stats = EventStats()
        event_stats.triggers = {
            EventType.TOOL_EXECUTE: 2,
            EventType.TOOL_RESULT: 2,
        }
        event_stats.total_triggers = 2
        presenter = FancyStreamPresenter(
            self.theme,
            getLogger(__name__),
            event_stats=event_stats,
        )

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        output = _frame_text(stream_frame)
        self.assertIn("2 tokens in", output)
        self.assertIn("2 tokens out", output)
        self.assertIn("3 tool tokens", output)
        self.assertIn("2 events", output)
        self.assertIn("2 tool calls", output)
        self.assertIn("2 results", output)

    async def test_progress_panel_without_text_output(self) -> None:
        config = _stream_config(display_tools=False, display_events=False)
        builder = CliStreamSnapshotBuilder(config)
        builder.update_token_counts(input_tokens=4, output_tokens=0)
        event_stats = EventStats()
        event_stats.triggers = {}
        event_stats.total_triggers = 3
        presenter = FancyStreamPresenter(
            self.theme,
            getLogger(__name__),
            event_stats=event_stats,
        )

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        frames = [
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
        ]
        self.assertEqual([frame.role for frame in frames], ["stream"])
        output = _frame_text(frames[0])
        self.assertIn("Token stats", output)
        self.assertIn("4 tokens in", output)
        self.assertIn("0 tokens out", output)
        self.assertIn("3 events", output)

    async def test_answer_wraps_long_lines(self) -> None:
        config = _stream_config(
            display_tools=False,
            display_events=False,
            display_tokens=0,
            answer_height_expand=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("Word " * 20 + "\n" + "Another word " * 20)
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(
                config,
                builder.snapshot(),
                context=_stream_context(console_width=40),
            ),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        output = _frame_text(stream_frame)
        self.assertIn("Word Word", output)
        self.assertIn("Another word", output)
        self.assertGreaterEqual(output.count("\n"), 1)

    def test_limited_wrapped_output_tails_before_wrapping(self) -> None:
        text = "01234567" * 50
        expected = "\n".join(FancyTheme._wrap_lines([text], 8)[-2:]).rstrip()

        output = fancy_theme_module._fancy_wrapped_output(
            text,
            max_width=8,
            height=3,
            padding=1,
            limit_height=True,
            visible_line_count=2,
        )

        self.assertEqual(output, expected)

    def test_limited_wrapped_output_tails_unaligned_unbroken_text(
        self,
    ) -> None:
        text = ("01234567" * 50) + "01"
        expected = "\n".join(FancyTheme._wrap_lines([text], 8)[-2:]).rstrip()

        output = fancy_theme_module._fancy_wrapped_output(
            text,
            max_width=8,
            height=3,
            padding=1,
            limit_height=True,
            visible_line_count=2,
        )

        self.assertEqual(output, expected)

    def test_limited_wrapped_output_tails_long_word_before_whitespace(
        self,
    ) -> None:
        text = ("x" * 18) + " "
        expected = "\n".join(FancyTheme._wrap_lines([text], 8)[-1:]).rstrip()

        output = fancy_theme_module._fancy_wrapped_output(
            text,
            max_width=8,
            height=2,
            padding=1,
            limit_height=True,
            visible_line_count=1,
        )

        self.assertEqual(output, expected)

    def test_limited_wrapped_output_tails_trailing_blank_lines(
        self,
    ) -> None:
        text = ("abcd " * 20) + "\n "
        expected = "\n".join(FancyTheme._wrap_lines([text], 4)[-3:]).rstrip()

        output = fancy_theme_module._fancy_wrapped_output(
            text,
            max_width=4,
            height=4,
            padding=1,
            limit_height=True,
            visible_line_count=3,
        )

        self.assertEqual(output, expected)

    def test_limited_wrapped_output_tails_collapsed_whitespace_context(
        self,
    ) -> None:
        text = "word" + (" " * 5) + "tail"
        expected = "\n".join(FancyTheme._wrap_lines([text], 4)[-2:]).rstrip()

        output = fancy_theme_module._fancy_wrapped_output(
            text,
            max_width=4,
            height=3,
            padding=1,
            limit_height=True,
            visible_line_count=2,
        )

        self.assertEqual(output, expected)

    def test_limited_wrapped_output_respects_skip_blank_lines_tail(
        self,
    ) -> None:
        text = "aa\n "
        expected = "\n".join(
            FancyTheme._wrap_lines([text], 1, skip_blank_lines=True)[-2:]
        ).rstrip()

        output = fancy_theme_module._fancy_wrapped_output(
            text,
            max_width=1,
            height=3,
            padding=1,
            limit_height=True,
            visible_line_count=2,
            skip_blank_lines=True,
        )

        self.assertEqual(output, expected)

    def test_limited_wrapped_output_preserves_indented_tail_parity(
        self,
    ) -> None:
        text = "\n  " + ("x" * 20) + "a     \t  b  tailaword"
        expected = "\n".join(
            FancyTheme._wrap_lines([text], 5, skip_blank_lines=True)[-5:]
        ).rstrip()

        output = fancy_theme_module._fancy_wrapped_output(
            text,
            max_width=5,
            height=6,
            padding=1,
            limit_height=True,
            visible_line_count=5,
            skip_blank_lines=True,
        )

        self.assertEqual(output, expected)

    def test_unlimited_wrapped_output_ignores_visible_line_count(self) -> None:
        text = "01234567" * 4
        expected = "\n".join(FancyTheme._wrap_lines([text], 8)).rstrip()

        output = fancy_theme_module._fancy_wrapped_output(
            text,
            max_width=8,
            height=2,
            padding=1,
            limit_height=False,
            visible_line_count=1,
        )

        self.assertEqual(output, expected)
        self.assertGreater(len((output or "").splitlines()), 1)

    async def test_answer_mode_skips_stream_panels(self) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("answer")
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        with patch.object(
            presenter,
            "_stream_renderable",
            side_effect=AssertionError("stream panels should not render"),
        ):
            items = await _collect_stream_items(
                presenter,
                _stream_request(config, builder.snapshot(), mode="answer"),
            )

        self.assertEqual(
            [
                item.text
                for item in items
                if isinstance(
                    item,
                    CliStreamAnswerTextChunk,
                )
            ],
            ["answer"],
        )

    async def test_answer_mode_pretty_prints_terminal_json(self) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text('{"items":[{"amount":134}]}')
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot(), mode="answer"),
        )

        self.assertEqual(
            [
                item.text
                for item in items
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ['{\n  "items": [\n    {\n      "amount": 134\n    }\n  ]\n}'],
        )

    async def test_answer_mode_buffers_partial_json_until_terminal(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text('{"items":[{"amount":')
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        first = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot(), mode="answer"),
        )
        builder.append_answer_text("134}]}")
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        second = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot(), mode="answer"),
        )

        self.assertEqual(first, [])
        self.assertEqual(
            [
                item.text
                for item in second
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ['{\n  "items": [\n    {\n      "amount": 134\n    }\n  ]\n}'],
        )

    async def test_stats_panel_pretty_prints_terminal_json(self) -> None:
        config = _stream_config(
            display_tools=False,
            display_events=False,
            answer_height_expand=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text('{"items":[{"amount":134}]}')
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        output = _frame_text(stream_frame)
        self.assertIn("{", output)
        self.assertIn('"items": [', output)
        self.assertIn('"amount": 134', output)
        self.assertNotIn('{"items":[{"amount":134}]}', output)

    def test_theme_factory_returns_fancy_stream_presenter(self) -> None:
        presenter = self.theme.stream_presenter(getLogger(__name__))

        self.assertIsInstance(presenter, FancyStreamPresenter)

    async def test_tool_history_renders_completed_result_diagnostic_and_event(
        self,
    ) -> None:
        config = _stream_config(display_events=False, display_tools_events=8)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(
            tool_call_id="done-call",
            name="calc",
            arguments={"x": 1},
        )
        builder.complete_tool(
            tool_call_id="done-call",
            name="calc",
            elapsed_seconds=0.5,
        )
        builder.add_tool_result_summary(
            tool_call_id="done-call",
            name="calc",
            status="result",
            result=25,
            arguments_count=1,
            elapsed_seconds=0.5,
        )
        builder.add_tool_diagnostic(
            ToolCallDiagnostic(
                id="diag-details",
                call_id="done-call",
                requested_name="calc",
                canonical_name="math.calculator",
                code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
                stage=ToolCallDiagnosticStage.VALIDATE,
                message="Bad arguments.",
                retryable=True,
                details={"field": "expr"},
            )
        )
        builder.add_tool_diagnostic(
            ToolCallDiagnostic(
                id="diag-plain",
                requested_name="calc",
                code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
                stage=ToolCallDiagnosticStage.RESOLVE,
                message="Unknown tool.",
            )
        )
        builder.add_tool_event(
            Event(
                type=EventType.TOOL_MODEL_RUN,
                payload={"model_id": "react"},
                elapsed=0.2,
            ),
            tool_call_id="done-call",
            name="calc",
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        tools_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "tools"
        )
        output = _frame_text(tools_frame)
        for fragment in (
            "Completed tool calc",
            "Executed tool calc",
            'Got result "25"',
            "Tool diagnostic tool_call.arguments_invalid",
            "Bad arguments",
            "Retryable",
            "Tool diagnostic tool.unknown",
            "Unknown tool",
            "Tool event tool_model_run",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, output)

    async def test_display_tools_events_zero_keeps_active_and_clears_stale(
        self,
    ) -> None:
        config = _stream_config(
            stats=False,
            display_events=False,
            display_tools_events=0,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("working")
        builder.add_active_tool(
            tool_call_id="active-call",
            name="calc",
            arguments={"x": 1},
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        first = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        no_tool_builder = CliStreamSnapshotBuilder(config)
        no_tool_builder.append_answer_text("working")
        no_tool_snapshot = no_tool_builder.snapshot()
        second = await _collect_stream_items(
            presenter,
            _stream_request(config, no_tool_snapshot),
        )

        first_frames = [
            item
            for item in first
            if isinstance(item, CliStreamRenderableFrame)
        ]
        self.assertEqual([frame.role for frame in first_frames], ["tools"])
        self.assertIn("Executing tool calc", _frame_text(first_frames[0]))
        self.assertNotIn("Completed tool", _frame_text(first_frames[0]))
        second_frames = [
            item
            for item in second
            if isinstance(item, CliStreamRenderableFrame)
        ]
        self.assertEqual([frame.role for frame in second_frames], ["tools"])
        self.assertEqual(second_frames[0].renderable, "")
        self.assertEqual(
            [
                item.text
                for item in (*first, *second)
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["working"],
        )

    async def test_events_and_stats_frames_clear_stale_surfaces(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=False,
            display_events=True,
            display_tokens=0,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("answer")
        builder.add_event_summary(
            event_type=EventType.FLOW_NODE_STARTED,
            payload={"node": "math"},
        )
        builder.add_usage_summary(
            {"input_tokens": 1, "output_tokens": 2},
            kind="stream.completed",
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        first = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )
        second_builder = CliStreamSnapshotBuilder(config)
        second_builder.append_answer_text("answer")
        second = await _collect_stream_items(
            presenter,
            _stream_request(config, second_builder.snapshot()),
        )

        first_frames = [
            item
            for item in first
            if isinstance(item, CliStreamRenderableFrame)
        ]
        self.assertEqual(
            [frame.role for frame in first_frames],
            ["events", "stats", "stream"],
        )
        self.assertIn("flow_node_started", _frame_text(first_frames[0]))
        self.assertIn("usage stream.completed", _frame_text(first_frames[1]))

        second_frames = [
            item
            for item in second
            if isinstance(item, CliStreamRenderableFrame)
        ]
        self.assertEqual(
            [frame.role for frame in second_frames],
            ["events", "stats", "stream"],
        )
        self.assertEqual(second_frames[0].renderable, "")
        self.assertEqual(second_frames[1].renderable, "")
        second_output = _render_visible_text(
            *(frame.renderable for frame in second_frames)
        )
        self.assertNotIn("flow_node_started", second_output)
        self.assertNotIn("usage stream.completed", second_output)

    async def test_combined_tools_and_events_does_not_duplicate_tool_events(
        self,
    ) -> None:
        config = _stream_config(stats=False, display_tools_events=4)
        builder = CliStreamSnapshotBuilder(config)
        builder.add_tool_event(
            Event(type=EventType.TOOL_MODEL_RUN, payload={"model": "react"}),
            tool_call_id="tool-call",
            name="calc",
        )
        builder.add_event_summary(
            event_type=EventType.FLOW_NODE_COMPLETED,
            payload={"node": "math"},
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        tools_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "tools"
        )
        events_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "events"
        )
        self.assertIn("tool_model_run", _frame_text(tools_frame))
        event_output = _frame_text(events_frame)
        self.assertIn("flow_node_completed", event_output)
        self.assertNotIn("tool_model_run", event_output)

    async def test_answer_height_and_expand_are_preserved(self) -> None:
        answer = "\n".join(f"line {index}" for index in range(6))
        limited_config = _stream_config(
            display_tools=False,
            display_events=False,
            display_tokens=0,
            answer_height=3,
        )
        limited_builder = CliStreamSnapshotBuilder(limited_config)
        limited_builder.append_answer_text(answer)
        expanded_config = _stream_config(
            display_tools=False,
            display_events=False,
            display_tokens=0,
            answer_height=3,
            answer_height_expand=True,
        )
        expanded_builder = CliStreamSnapshotBuilder(expanded_config)
        expanded_builder.append_answer_text(answer)
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        limited_items = await _collect_stream_items(
            presenter,
            _stream_request(limited_config, limited_builder.snapshot()),
        )
        expanded_items = await _collect_stream_items(
            presenter,
            _stream_request(expanded_config, expanded_builder.snapshot()),
        )

        limited_frame = next(
            item
            for item in limited_items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        expanded_frame = next(
            item
            for item in expanded_items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        limited_panel = limited_frame.renderable.renderables[0]
        expanded_panel = expanded_frame.renderable.renderables[0]
        self.assertEqual(limited_panel.height, 5)
        self.assertIsNone(expanded_panel.height)
        self.assertNotIn("line 0", _frame_text(limited_frame))
        self.assertIn("line 0", _frame_text(expanded_frame))

    async def test_reasoning_height_truncates_to_legacy_content_budget(
        self,
    ) -> None:
        config = _stream_config(display_reasoning=True)
        builder = CliStreamSnapshotBuilder(config)
        builder.append_reasoning_text(
            "\n".join(f"reasoning-line-{index}" for index in range(5))
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        reasoning_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "reasoning"
        )
        output = _frame_text(reasoning_frame)
        self.assertNotIn("reasoning-line-0", output)
        for index in range(1, 5):
            with self.subTest(index=index):
                self.assertIn(f"reasoning-line-{index}", output)

    async def test_tool_request_height_truncates_to_legacy_content_budget(
        self,
    ) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.append_tool_call_request_text("drop0 keep1 keep2 keep3 keep4")
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(
                config,
                builder.snapshot(),
                context=_stream_context(console_width=9),
            ),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        output = _frame_text(stream_frame)
        self.assertNotIn("drop0", output)
        for index in range(1, 5):
            with self.subTest(index=index):
                self.assertIn(f"keep{index}", output)

    async def test_probability_disabled_shows_tokens_without_current_token(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=False,
            display_events=False,
            display_probabilities=False,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("token output")
        builder.add_display_token(Token(id=1, token="<eos>", probability=0.1))
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        self.assertIsNone(stream_frame.current_token)
        output = _frame_text(stream_frame)
        self.assertIn("token distribution", output)
        self.assertIn("<eos>", output)

    async def test_probability_maximum_zero_shows_tokens_without_panel(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=False,
            display_events=False,
            display_probabilities=True,
            display_probabilities_maximum=0.0,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("token output")
        builder.add_display_token(
            TokenDetail(
                id=1,
                token="base",
                probability=0.2,
                tokens=[
                    Token(id=1, token="base", probability=0.2),
                    Token(id=2, token="alternate", probability=0.7),
                ],
            )
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        self.assertIsNone(stream_frame.current_token)
        output = _frame_text(stream_frame)
        self.assertIn("token distribution", output)
        self.assertIn("base", output)
        self.assertNotIn("#1", output)
        self.assertNotIn("alternate", output)
        self.assertNotIn("70%", output)

    async def test_probability_pick_zero_shows_tokens_without_panel(
        self,
    ) -> None:
        config = _stream_config(
            display_tools=False,
            display_events=False,
            display_probabilities=True,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("token output")
        builder.add_display_token(
            TokenDetail(
                id=1,
                token="base",
                probability=0.2,
                tokens=[
                    Token(id=1, token="base", probability=0.2),
                    Token(id=2, token="alternate", probability=0.7),
                ],
            )
        )
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(
                config,
                builder.snapshot(),
                context=_stream_context(token_probability_pick=0),
            ),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        self.assertIsNone(stream_frame.current_token)
        output = _frame_text(stream_frame)
        self.assertIn("token distribution", output)
        self.assertIn("base", output)
        self.assertNotIn("#1", output)
        self.assertNotIn("alternate", output)
        self.assertNotIn("70%", output)

    async def test_low_probability_token_without_candidates_is_current(
        self,
    ) -> None:
        config = _stream_config(display_tools=False, display_events=False)
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("token output")
        builder.add_display_token(Token(id=7, token="solo", probability=0.2))
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        current_token = stream_frame.current_token
        self.assertIsNotNone(current_token)
        assert current_token is not None
        self.assertEqual(current_token.token_id, 7)
        self.assertEqual(current_token.text, "solo")
        output = _frame_text(stream_frame)
        self.assertIn("token distribution", output)
        self.assertIn("#7", output)
        self.assertIn("solo", output)
        self.assertNotIn("20%", output)

    async def test_event_stat_title_suppresses_missing_and_zero_counts(
        self,
    ) -> None:
        config = _stream_config(display_tools=False, display_events=False)
        builder = CliStreamSnapshotBuilder(config)
        event_stats = EventStats()
        event_stats.total_triggers = 0
        event_stats.triggers = {EventType.TOOL_EXECUTE: 0}
        presenter = FancyStreamPresenter(
            self.theme,
            getLogger(__name__),
            event_stats=event_stats,
        )

        items = await _collect_stream_items(
            presenter,
            _stream_request(config, builder.snapshot()),
        )

        stream_frame = next(
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "stream"
        )
        output = _frame_text(stream_frame)
        self.assertIn("0 events", output)
        self.assertNotIn("tool call", output)
        self.assertNotIn("result", output)

    async def test_empty_events_do_not_emit_event_frame(self) -> None:
        config = _stream_config(stats=False, display_tools=False)
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))

        items = await _collect_stream_items(
            presenter,
            _stream_request(
                config, CliStreamSnapshotBuilder(config).snapshot()
            ),
        )

        self.assertEqual(
            [
                item.role
                for item in items
                if isinstance(item, CliStreamRenderableFrame)
            ],
            [],
        )

    def test_probability_helpers_handle_empty_candidates(self) -> None:
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))
        current_token = CliDisplayTokenSnapshot(
            sequence=1,
            token_id=7,
            text="solo",
            probability=0.2,
        )

        renderables = presenter._probability_renderables(current_token, 2)
        empty_chart = fancy_theme_module._fancy_probability_chart([], 2)

        output = _render_visible_text(*renderables, empty_chart)
        self.assertIn("solo", output)
        self.assertIn("#7", output)
        self.assertIn("1 2", output)

    async def test_answer_mode_reset_replays_answer(self) -> None:
        config = _stream_config(stats=False)
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("hello")
        presenter = FancyStreamPresenter(self.theme, getLogger(__name__))
        request = _stream_request(config, builder.snapshot(), mode="answer")

        first = await _collect_stream_items(presenter, request)
        second = await _collect_stream_items(presenter, request)
        presenter.reset()
        third = await _collect_stream_items(presenter, request)

        self.assertEqual(
            [
                item.text
                for item in (*first, *second, *third)
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["hello", "hello"],
        )

    def test_fancy_elapsed_seconds_fallbacks(self) -> None:
        config = _stream_config()
        builder = CliStreamSnapshotBuilder(config)
        builder.update_timing(started_at=5.0, updated_at=7.5)
        started_updated = builder.snapshot()
        empty = CliStreamSnapshotBuilder(config).snapshot()

        self.assertEqual(
            fancy_theme_module._fancy_elapsed_seconds(started_updated),
            2.5,
        )
        self.assertEqual(fancy_theme_module._fancy_elapsed_seconds(empty), 0.0)


class FancyThemeTokensTestCase(IsolatedAsyncioTestCase):
    async def test_tool_running_spinner_text(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        spinner = Spinner("dots", text="[cyan]run[/cyan]", style="cyan")
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = theme.token_frames(
                _token_state(
                    answer_text_tokens=["a"],
                    tool_running_spinner=spinner,
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=80,
                logger=MagicMock(),
            )
            frame = frames[0]
        self.assertTrue(
            any(
                getattr(r, "renderable", None) is spinner
                for r in frame[1].renderables
            )
        )

    async def test_tool_text_tokens_panel(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = theme.token_frames(
                _token_state(
                    tool_text_tokens=["tool"],
                    answer_text_tokens=["answer"],
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=80,
                logger=MagicMock(),
            )
            _, frame = frames[0]
        self.assertEqual(len(frame.renderables), 2)
        self.assertIn("tool", str(frame.renderables[0].renderable.renderable))

    async def test_progress_title_pluralization_singular(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        es = EventStats()
        es.triggers = {
            EventType.TOOL_EXECUTE: 1,
            EventType.TOOL_RESULT: 1,
        }
        es.total_triggers = 1
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = theme.token_frames(
                _token_state(
                    answer_text_tokens=["a"],
                    input_token_count=1,
                    total_tokens=1,
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                    event_stats=es,
                ),
                console_width=80,
                logger=MagicMock(),
            )
            _, frame = frames[0]
        subtitle = frame.renderables[0].subtitle
        self.assertIn("1 token in", subtitle)
        self.assertIn("1 token out", subtitle)
        self.assertIn("1 event", subtitle)
        self.assertIn("1 tool call", subtitle)
        self.assertIn("1 result", subtitle)

    async def test_progress_title_pluralization_plural(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        es = EventStats()
        es.triggers = {
            EventType.TOOL_EXECUTE: 2,
            EventType.TOOL_RESULT: 2,
        }
        es.total_triggers = 2
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = theme.token_frames(
                _token_state(
                    answer_text_tokens=["a"],
                    input_token_count=2,
                    total_tokens=2,
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                    event_stats=es,
                    tool_token_count=3,
                ),
                console_width=80,
                logger=MagicMock(),
            )
            _, frame = frames[0]
        subtitle = frame.renderables[0].subtitle
        self.assertIn("2 tokens in", subtitle)
        self.assertIn("2 tokens out", subtitle)
        self.assertIn("3 tool tokens", subtitle)
        self.assertIn("2 events", subtitle)
        self.assertIn("2 tool calls", subtitle)
        self.assertIn("2 results", subtitle)

    async def test_progress_panel_without_text_output(self):
        theme = FancyTheme(lambda s: s, lambda s, p, n: s if n == 1 else p)
        es = EventStats()
        es.triggers = {}
        es.total_triggers = 3
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = theme.token_frames(
                _token_state(
                    input_token_count=4,
                    total_tokens=0,
                    ttft=None,
                    ttnt=None,
                    ttsr=None,
                    elapsed=2.0,
                    event_stats=es,
                ),
                console_width=80,
                logger=MagicMock(),
            )
            _, frame = frames[0]

        self.assertEqual(len(frame.renderables), 1)
        panel = frame.renderables[0]
        self.assertIn("Token stats", str(panel.title))
        self.assertIn("4 tokens in", str(panel.renderable))
        self.assertIn("0 tokens out", str(panel.renderable))
        self.assertIn("3 events", str(panel.renderable))


class FancyThemeTestCase(IsolatedAsyncioTestCase):
    def setUp(self):
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_quantity_data(self):
        self.assertEqual(self.theme.quantity_data, ["likes"])

    def test_agent(self):
        memory = SimpleNamespace(
            has_recent_message=True,
            has_permanent_message=True,
            permanent_message=SimpleNamespace(
                session_id=UUID("11111111-1111-1111-1111-111111111111"),
                has_session=True,
            ),
        )
        agent = SimpleNamespace(id=UUID(int=0), name="agent", memory=memory)
        model = SimpleNamespace(
            id="m",
            parameters=1,
            parameter_types=["p"],
            inference=None,
            library_name=None,
            pipeline_tag=None,
            tags=[],
            architectures=None,
            model_type=None,
            license=None,
            gated=False,
            private=False,
            disabled=False,
            downloads=1,
            likes=2,
            ranking=1,
            author="a",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        panel = self.theme.agent(agent, models=[model], can_access=True)
        self.assertTrue(panel.title)

    def test_ask_methods(self):
        self.assertEqual(
            self.theme.ask_access_token(),
            "Enter your Huggingface access token",
        )
        self.assertEqual(
            self.theme.ask_delete_paths(), "Delete selected paths?"
        )
        self.assertEqual(
            self.theme.ask_login_to_hub(), "Login to huggingface?"
        )
        self.assertEqual(
            self.theme.ask_secret_password("k"), "Enter secret for k"
        )
        self.assertEqual(
            self.theme.ask_override_secret("k"), "Secret k exists, override?"
        )

    def test_cache_methods(self):
        deletion = HubCacheDeletion(
            model_id="m",
            revisions=["r"],
            deletable_size_on_disk=1,
            deletable_blobs=["b"],
            deletable_refs=[],
            deletable_repos=[],
            deletable_snapshots=[],
        )
        result = self.theme.cache_delete(deletion)
        self.assertTrue(
            hasattr(result, "renderables") or hasattr(result, "text")
        )

        cache = HubCache(
            model_id="m",
            path="/p",
            size_on_disk=1,
            revisions=["r"],
            files={},
            total_files=0,
            total_revisions=1,
        )
        table = self.theme.cache_list("/c", [cache], show_summary=True)
        self.assertEqual(len(table.rows), 1)

    def test_download_methods(self):
        self.theme.download_progress()
        self.theme.download_start("m")
        self.theme.download_finished("m", "/path")
        self.theme.download_access_denied("m", "url")

    def test_memory_embeddings(self):
        data = np.arange(6, dtype=float)
        grp = self.theme.memory_embeddings(
            "text",
            data,
            total_tokens=1,
            minv=float(data.min()),
            maxv=float(data.max()),
            meanv=float(data.mean()),
            stdv=float(data.std()),
            normv=float(norm(data)),
        )
        self.assertTrue(grp.renderables)

        grp = self.theme.memory_embeddings(
            "text",
            data,
            total_tokens=1,
            minv=float(data.min()),
            maxv=float(data.max()),
            meanv=float(data.mean()),
            stdv=float(data.std()),
            normv=float(norm(data)),
            embedding_peek=2,
            show_stats=False,
        )
        self.assertTrue(grp.renderables)

    def test_memory_embeddings_comparison(self):
        sim = Similarity(
            cosine_distance=0.0,
            inner_product=0.0,
            l1_distance=0.0,
            l2_distance=0.0,
            pearson=0.0,
        )
        res = self.theme.memory_embeddings_comparison({"t": sim}, "t")
        self.assertEqual(len(res.renderable.rows), 1)

    def test_memory_embeddings_search(self):
        match = SearchMatch(query="q", match="m", l2_distance=0.1)
        res = self.theme.memory_embeddings_search([match])
        self.assertEqual(len(res.renderable.rows), 1)

    def test_memory_partitions(self):
        part = TextPartition(
            data="t", total_tokens=1, embeddings=np.array([1])
        )
        group = self.theme.memory_partitions([part] * 3, display_partitions=2)
        self.assertEqual(len(group.renderables), 3)

    def test_model(self):
        model = SimpleNamespace(
            id="m",
            parameters=1,
            parameter_types=["p"],
            inference=None,
            library_name=None,
            pipeline_tag=None,
            tags=[],
            architectures=None,
            model_type=None,
            license=None,
            gated=False,
            private=False,
            disabled=False,
            downloads=1,
            likes=2,
            ranking=1,
            author="a",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        panel = self.theme.model(model, can_access=True)
        self.assertIn("m", panel.title)

    def test_model_display(self):
        cfg = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=None,
            bos_token=None,
            decoder_start_token_id=None,
            eos_token_id=None,
            eos_token=None,
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type=None,
            max_position_embeddings=10,
            model_type="t",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=None,
            pad_token=None,
            prefix=None,
            sep_token_id=None,
            sep_token=None,
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        tok_cfg = TokenizerConfig(
            name_or_path="t",
            tokens=["a"],
            special_tokens=["b"],
            tokenizer_model_max_length=10,
            fast=True,
        )
        group = self.theme.model_display(cfg, tok_cfg)
        self.assertEqual(len(group.renderables), 2)

    def test_recent_messages(self):
        msg = EngineMessage(
            agent_id=UUID(int=0),
            model_id="m",
            message=Message(role="user", content="hi"),
        )
        agent = SimpleNamespace(name="n")
        group = self.theme.recent_messages(str(UUID(int=1)), agent, [msg])
        self.assertEqual(len(group.renderables), 1)

    def test_saved_tokenizer_files(self):
        pad = self.theme.saved_tokenizer_files("/d", 2)
        self.assertIn("2 tokenizer files", pad.renderable)

    def test_saved_tokenizer_files_with_total_only(self):
        pad = self.theme.saved_tokenizer_files(1)
        self.assertIn("1 tokenizer file", pad.renderable)

    def test_events_log_tool_branches(self):
        tool_call = SimpleNamespace(
            id=UUID(int=2), name="calc", arguments={"x": 1}
        )
        tool_result = SimpleNamespace(call=tool_call, result={"ok": True})
        diagnostic = ToolCallDiagnostic(
            id="diag-1",
            call_id="call-1",
            requested_name="weather",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )
        events = [
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": tool_call},
            ),
            Event(
                type=EventType.TOOL_MODEL_RUN,
                payload={"model_id": "react", "messages": ["a", "b"]},
            ),
            Event(
                type=EventType.TOOL_MODEL_RESPONSE,
                payload={"model_id": "react"},
            ),
            Event(
                type=EventType.TOOL_PROCESS,
                payload=[SimpleNamespace(name="weather")],
            ),
            Event(
                type=EventType.TOOL_RESULT,
                payload={"result": tool_result},
                elapsed=0.01,
            ),
            Event(
                type=EventType.TOOL_DIAGNOSTIC,
                payload={"diagnostic": diagnostic},
            ),
        ]

        log = self.theme._events_log(
            events,
            events_limit=None,
            include_tokens=False,
            include_tool_detect=False,
            include_tools=True,
            include_non_tools=False,
        )

        assert log is not None
        self.assertEqual(len(log), 6)
        self.assertIn("tool.unknown", log[-1])
        self.assertIn("Unknown tool.", log[-1])

    def test_events_log_tool_result_diagnostic_and_error(self):
        call = ToolCall(id="call-2", name="calc", arguments={"x": 1})
        diagnostic = ToolCallDiagnostic(
            id="diag-2",
            call_id=call.id,
            requested_name="calc",
            code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
            stage=ToolCallDiagnosticStage.VALIDATE,
            message="Invalid arguments.",
        )
        error = ToolCallError(
            id="error-1",
            call=call,
            name="calc",
            arguments={"x": 1},
            error=RuntimeError("secret"),
            message="Tool failed.",
        )
        events = [
            Event(
                type=EventType.TOOL_RESULT,
                payload={"call": call, "result": diagnostic},
            ),
            Event(
                type=EventType.TOOL_RESULT,
                payload={"result": error},
                elapsed=0.01,
            ),
            Event(
                type=EventType.TOOL_DIAGNOSTIC,
                payload={"diagnostics": ["bad"]},
            ),
        ]

        log = self.theme._events_log(
            events,
            events_limit=None,
            include_tokens=False,
            include_tool_detect=False,
            include_tools=True,
            include_non_tools=False,
        )

        assert log is not None
        self.assertIn("tool_call.arguments_invalid", log[0])
        self.assertIn("Tool call failed.", log[1])
        self.assertNotIn("secret", log[1])
        self.assertNotIn("Tool failed.", log[1])
        self.assertIn("bad", log[2])

    def test_events_log_canonical_tool_payloads(self):
        def event(
            event_type: EventType,
            kind: str,
            correlation: dict[str, str],
            *,
            summary: dict[str, object] | None = None,
            usage: dict[str, object] | None = None,
            terminal_outcome: str | None = None,
            derived: bool | None = None,
        ) -> Event:
            data: dict[str, object] = {
                "stream_session_id": "stream-1",
                "run_id": "run-1",
                "turn_id": "turn-1",
                "sequence": 1,
                "kind": kind,
                "channel": "tool.execution",
                "visibility": "public",
                "correlation": correlation,
                "data": {"secret": "raw legacy payload"},
            }
            if summary is not None:
                data["summary"] = summary
            if usage is not None:
                data["usage"] = usage
            if terminal_outcome is not None:
                data["terminal_outcome"] = terminal_outcome
            if derived is not None:
                data["derived"] = derived
            return Event.from_observability_payload(
                type=event_type,
                observability_payload=EventObservabilityPayload.canonical_stream(
                    data
                ),
            )

        events = [
            event(
                EventType.TOOL_EXECUTE,
                "tool.execution.started",
                {"tool_call_id": "call-1"},
                summary={"data_keys": ["name", "arguments"]},
                derived=True,
            ),
            event(
                EventType.TOOL_MODEL_RUN,
                "model.continuation.started",
                {"model_continuation_id": "cont-1"},
            ),
            event(
                EventType.TOOL_MODEL_RESPONSE,
                "model.continuation.completed",
                {"model_continuation_id": "cont-1"},
            ),
            event(
                EventType.TOOL_RESULT,
                "tool.execution.completed",
                {"tool_call_id": "call-1"},
                usage={"output_tokens": 3},
                terminal_outcome="completed",
            ),
            event(
                EventType.TOOL_DIAGNOSTIC,
                "tool.execution.error",
                {"tool_call_id": "call-2"},
            ),
            event(
                EventType.TOOL_PROGRESS,
                "tool.execution.progress",
                {"tool_call_id": "call-1"},
            ),
        ]

        log = self.theme._events_log(
            events,
            events_limit=None,
            include_tokens=False,
            include_tool_detect=False,
            include_tools=True,
            include_non_tools=False,
        )

        assert log is not None
        self.assertEqual(len(log), 6)
        self.assertIn("Canonical event", log[0])
        self.assertIn("kind=", log[0])
        self.assertIn("channel=", log[0])
        self.assertIn("tool.execution.started", log[0])
        self.assertIn("correlation=", log[0])
        self.assertIn("tool_call_id", log[0])
        self.assertIn("summary=", log[0])
        self.assertIn("derived=", log[0])
        self.assertIn("true", log[0])
        self.assertIn("model.continuation.completed", log[2])
        self.assertIn("model_continuation_id", log[2])
        self.assertIn("usage=", log[3])
        self.assertIn("terminal_outcome=", log[3])
        self.assertIn("completed", log[3])
        self.assertIn("tool.execution.progress", log[5])
        self.assertNotIn("raw legacy payload", "\n".join(log))

    def test_events_log_canonical_payloads_route_by_projection(self):
        def event(kind: str, channel: str) -> Event:
            return Event.from_observability_payload(
                type=EventType.TOKEN_GENERATED,
                observability_payload=EventObservabilityPayload.canonical_stream(
                    {
                        "stream_session_id": "stream-1",
                        "run_id": "run-1",
                        "turn_id": "turn-1",
                        "sequence": 1,
                        "kind": kind,
                        "channel": channel,
                        "visibility": "public",
                    }
                ),
            )

        tool_log = self.theme._events_log(
            [
                event("answer.delta", "answer"),
                event("tool_call.ready", "tool_call"),
            ],
            events_limit=None,
            include_tokens=False,
            include_tool_detect=False,
            include_tools=True,
            include_non_tools=False,
        )
        event_log = self.theme._events_log(
            [
                event("stream.diagnostic", "control"),
                event("tool_execution.started", "tool_execution"),
            ],
            events_limit=None,
            include_tokens=True,
            include_tool_detect=False,
            include_tools=False,
            include_non_tools=True,
        )

        self.assertEqual(len(tool_log or []), 1)
        self.assertIn("tool_call.ready", tool_log[0] if tool_log else "")
        self.assertEqual(len(event_log or []), 1)
        self.assertIn("stream.diagnostic", event_log[0] if event_log else "")

    async def test_events_render_canonical_observability_panels(self) -> None:
        def event(event_type: EventType, kind: str, channel: str) -> Event:
            return Event.from_observability_payload(
                type=event_type,
                observability_payload=EventObservabilityPayload.canonical_stream(
                    {
                        "stream_session_id": "stream-1",
                        "run_id": "run-1",
                        "turn_id": "turn-1",
                        "sequence": 1,
                        "kind": kind,
                        "channel": channel,
                        "visibility": "diagnostic",
                    }
                ),
            )

        events = [
            event(EventType.TOOL_DETECT, "stream.diagnostic", "control"),
            event(EventType.START, "tool.call.ready", "control"),
        ]

        diagnostics_panel = self.theme.events(
            events,
            events_limit=None,
            include_tokens=False,
            include_tool_detect=False,
            include_tools=False,
            include_non_tools=True,
        )
        tools_panel = self.theme.events(
            events,
            events_limit=None,
            include_tokens=False,
            include_tool_detect=False,
            include_tools=True,
            include_non_tools=False,
            tool_view=True,
        )

        assert diagnostics_panel is not None
        assert tools_panel is not None
        diagnostics_console = Console(file=StringIO(), record=True, width=120)
        tools_console = Console(file=StringIO(), record=True, width=120)
        diagnostics_console.print(diagnostics_panel)
        tools_console.print(tools_panel)
        diagnostics_output = diagnostics_console.export_text()
        tools_output = tools_console.export_text()

        self.assertIn("Events", diagnostics_output)
        self.assertIn("stream.diagnostic", diagnostics_output)
        self.assertNotIn("tool.call.ready", diagnostics_output)
        self.assertIn("Tool calls", tools_output)
        self.assertIn("tool.call.ready", tools_output)
        self.assertNotIn("stream.diagnostic", tools_output)

    def test_canonical_event_payload_helpers_cover_fallbacks(self):
        self.assertIsNone(self.theme._canonical_event_payload("bad"))
        self.assertIsNone(
            self.theme._canonical_event_payload(
                {
                    "stream_session_id": "stream-1",
                    "run_id": "run-1",
                    "turn_id": "turn-1",
                    "kind": "tool.execution.started",
                    "channel": 1,
                }
            )
        )
        self.assertEqual(
            self.theme._canonical_event_correlation_detail({}),
            "",
        )
        self.assertEqual(
            self.theme._canonical_event_correlation_detail(
                {"correlation": {"artifact_id": "artifact-1"}}
            ),
            "",
        )
        self.assertEqual(
            self.theme._canonical_event_correlation_detail(
                {"correlation": {"tool_call_id": "call-123456"}}
            ),
            " for call #call-123",
        )
        self.assertIn("42", self.theme._canonical_event_value(42))

    def test_tool_diagnostic_from_payload_variants(self):
        diagnostic = ToolCallDiagnostic(
            id="diag-payload",
            call_id="call-payload",
            requested_name="calc",
            code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
            stage=ToolCallDiagnosticStage.RESOLVE,
            message="Unknown tool.",
        )

        self.assertIsNone(self.theme._tool_diagnostic_from_payload("bad"))
        self.assertIs(
            self.theme._tool_diagnostic_from_payload({"result": diagnostic}),
            diagnostic,
        )
        self.assertIsNone(self.theme._tool_diagnostic_from_payload({}))

    def test_search_message_matches(self):
        msg = EngineMessageScored(
            agent_id=UUID(int=0),
            model_id="m",
            message=Message(role="user", content="hi"),
            score=0.1,
        )
        agent = SimpleNamespace(name="n")
        group = self.theme.search_message_matches(
            str(UUID(int=1)), agent, [msg]
        )
        self.assertEqual(len(group.renderables), 1)

    def test_memory_search_matches(self):
        partition = PermanentMemoryPartition(
            participant_id=UUID(int=1),
            memory_id=UUID(int=0),
            partition=1,
            data="d",
            embedding=np.array([0.1]),
            created_at=datetime.now(),
        )
        group = self.theme.memory_search_matches(
            UUID(int=1), "ns", [partition]
        )
        self.assertEqual(len(group.renderables), 1)

    def test_tokenizer_config(self):
        cfg = TokenizerConfig(
            name_or_path="t",
            tokens=["a"],
            special_tokens=["b"],
            tokenizer_model_max_length=10,
            fast=True,
        )
        panel = self.theme.tokenizer_config(cfg)
        self.assertTrue(panel.renderable.rows)

    def test_tokenizer_tokens(self):
        t1 = Token(id=1, token="a")
        t2 = Token(id=2, token="b")
        panel = self.theme.tokenizer_tokens(
            [t1, t2], ["a"], ["s"], current_dtoken=t1
        )
        self.assertEqual(len(panel.renderable.renderable.renderables), 2)

    async def test_tokens_thinking(self):
        t = Token(id=1, token="a")
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = self.theme.token_frames(
                _token_state(
                    display_token_size=1,
                    focus_on_token_when=lambda x: True,
                    thinking_text_tokens=["x\n"],
                    answer_text_tokens=["y"],
                    tokens=[t],
                    total_tokens=1,
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
                maximum_frames=1,
            )
            frame = frames[0]
        self.assertTrue(frame[1].renderables)

    async def test_tokens_multiple_frames(self):
        alt1 = Token(id=2, token="b", probability=0.6)
        alt2 = Token(id=3, token="c", probability=0.4)
        dtoken = TokenDetail(
            id=1, token="a", probability=0.8, tokens=[alt1, alt2]
        )

        with (
            patch(
                "avalan.cli.theme.fancy._lf",
                lambda i: list(filter(None, i or [])),
            ),
            patch(
                "avalan.cli.theme.fancy._j",
                lambda sep, items: sep.join(str(x) for x in items if x),
            ),
        ):
            frames = self.theme.token_frames(
                _token_state(
                    display_token_size=1,
                    display_probabilities=True,
                    pick=2,
                    focus_on_token_when=lambda x: True,
                    thinking_text_tokens=["x\n"],
                    answer_text_tokens=["y"],
                    tokens=[dtoken],
                    total_tokens=1,
                    ttft=0.1,
                    ttnt=0.1,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
                maximum_frames=2,
            )
            self.assertEqual(len(frames), 2)
            frame1, frame2 = frames

        self.assertTrue(frame1[1].renderables)
        self.assertTrue(frame2[1].renderables)

    async def test_tokens_probability_uses_display_token_metadata(self):
        alt = Token(id=2, token="b", probability=0.6)
        dtoken = _DisplayToken(
            id=1,
            token="a",
            probability=0.8,
            tokens=[alt],
        )

        with (
            patch(
                "avalan.cli.theme.fancy._lf",
                lambda i: list(filter(None, i or [])),
            ),
            patch(
                "avalan.cli.theme.fancy._j",
                lambda sep, items: sep.join(str(x) for x in items if x),
            ),
        ):
            frames = self.theme.token_frames(
                _token_state(
                    display_token_size=1,
                    display_probabilities=True,
                    pick=1,
                    focus_on_token_when=lambda x: True,
                    answer_text_tokens=["y"],
                    tokens=[dtoken],
                    total_tokens=1,
                    ttft=0.1,
                    ttnt=0.1,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
                maximum_frames=1,
            )
            token, frame = frames[0]

        self.assertIsNotNone(token)
        assert token is not None
        self.assertEqual(token.token, dtoken.token)
        self.assertTrue(frame.renderables)

    async def test_tokens_probability_ignores_plain_display_token(self):
        token = Token(id=1, token="a", probability=0.8)

        with (
            patch(
                "avalan.cli.theme.fancy._lf",
                lambda i: list(filter(None, i or [])),
            ),
            patch(
                "avalan.cli.theme.fancy._j",
                lambda sep, items: sep.join(str(x) for x in items if x),
            ),
        ):
            frames = self.theme.token_frames(
                _token_state(
                    display_token_size=1,
                    display_probabilities=True,
                    pick=1,
                    focus_on_token_when=lambda x: True,
                    answer_text_tokens=["y"],
                    tokens=[token],
                    total_tokens=1,
                    ttft=0.1,
                    ttnt=0.1,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
                maximum_frames=1,
            )
            current_token, frame = frames[0]

        self.assertIsNone(current_token)
        self.assertTrue(frame.renderables)

    async def test_tokens_early_return(self):
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = self.theme.token_frames(
                _token_state(
                    answer_text_tokens=["x\n"],
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
            )
            self.assertEqual(len(frames), 1)
            token, frame = frames[0]

        self.assertIsNone(token)
        self.assertTrue(frame.renderables)

    async def test_tokens_pick_first_full_batch(self):
        alt = Token(id=2, token="b", probability=0.5)
        dtoken = TokenDetail(id=1, token="a", probability=0.6, tokens=[alt])
        with (
            patch(
                "avalan.cli.theme.fancy._lf",
                lambda i: list(filter(None, i or [])),
            ),
            patch(
                "avalan.cli.theme.fancy._j",
                lambda sep, items: sep.join(str(x) for x in items if x),
            ),
        ):
            frames = self.theme.token_frames(
                _token_state(
                    display_token_size=1,
                    display_probabilities=True,
                    pick=1,
                    focus_on_token_when=lambda x: True,
                    answer_text_tokens=["x\n"],
                    tokens=[dtoken],
                    total_tokens=1,
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
                maximum_frames=1,
            )
            frame = frames[0]
        self.assertTrue(frame[1].renderables)

    async def test_tokens_wrap_long_lines(self):
        long1 = "Word " * 20
        long2 = "Another word " * 20
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = self.theme.token_frames(
                _token_state(
                    answer_text_tokens=[f"{long1}\n", long2],
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
            )
            _, frame = frames[0]

        self.assertEqual(len(frame.renderables), 1)
        text = frame.renderables[0].renderable.renderable
        self.assertIn("Word Word", text)
        self.assertIn("Another word", text)
        self.assertGreaterEqual(text.count("\n"), 1)

    async def test_tokens_thinking_wrap_long_lines(self):
        think_line = "Reasoning " * 10
        answer_line = "Answer " * 10
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = self.theme.token_frames(
                _token_state(
                    thinking_text_tokens=[
                        f"{think_line}\n",
                        f"{think_line}\n",
                    ],
                    answer_text_tokens=[f"{answer_line}\n", answer_line],
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
            )
            _, frame = frames[0]

        self.assertEqual(len(frame.renderables), 2)
        think_text = frame.renderables[0].renderable.renderable
        answer_text = frame.renderables[1].renderable.renderable
        self.assertIn("Reasoning", think_text)
        self.assertIn("Answer", answer_text)
        self.assertGreaterEqual(answer_text.count("\n"), 1)

    async def test_tokens_thinking_uses_full_height(self):
        lines = [f"line{i}\n" for i in range(4)]
        with patch(
            "avalan.cli.theme.fancy._lf", lambda i: list(filter(None, i or []))
        ):
            frames = self.theme.token_frames(
                _token_state(
                    thinking_text_tokens=lines,
                    ttft=0.0,
                    ttnt=0.0,
                    ttsr=0.0,
                    elapsed=1.0,
                ),
                console_width=40,
                logger=MagicMock(),
            )
            _, frame = frames[0]

        self.assertEqual(len(frame.renderables), 1)
        think_text = frame.renderables[0].renderable.renderable
        self.assertIn("line0", think_text)
        self.assertGreaterEqual(think_text.count("\n"), 3)


class FancyThemeAdditionalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_bye(self):
        self.assertEqual(self.theme.bye(), ":vulcan_salute: bye :)")

    def test_action(self):
        panel = self.theme.action(
            "task",
            "desc",
            "author",
            "m",
            "lib",
            highlight=True,
            finished=True,
        )
        self.assertEqual(panel.box, box.DOUBLE)
        self.assertIn(
            "[green]desc[/green]", panel.renderable.renderable.renderables[0]
        )

    def test_cache_delete_deleted_true(self):
        deletion = HubCacheDeletion(
            model_id="m",
            revisions=["r"],
            deletable_size_on_disk=1,
            deletable_blobs=["b"],
            deletable_refs=[],
            deletable_repos=[],
            deletable_snapshots=[],
        )
        result = self.theme.cache_delete(deletion, deleted=True)
        self.assertTrue(result.renderables)

    def test_cache_list_display_models(self):
        cache = HubCache(
            model_id="m",
            path="/p",
            size_on_disk=1,
            revisions=["r"],
            files={"r": []},
            total_files=0,
            total_revisions=1,
        )
        group = self.theme.cache_list(
            "/c",
            [cache],
            display_models=["m"],
            show_summary=False,
        )
        self.assertEqual(len(group.renderables), 1)
        self.assertEqual(group.renderables[0].renderable.title, cache.model_id)

    def test_logging_in(self):
        self.assertEqual(
            self.theme.logging_in("hf"),
            "Logging in to hf...",
        )

    def test_memory_embeddings_orientation(self):
        data = np.arange(6, dtype=float)
        group = self.theme.memory_embeddings(
            "text",
            data,
            total_tokens=1,
            minv=float(data.min()),
            maxv=float(data.max()),
            meanv=float(data.mean()),
            stdv=float(data.std()),
            normv=float(norm(data)),
            embedding_peek=2,
            horizontal=False,
            show_stats=False,
        )
        table = group.renderables[0].renderable
        self.assertEqual(len(table.columns), 2)

        group = self.theme.memory_embeddings(
            "text",
            data,
            total_tokens=1,
            minv=float(data.min()),
            maxv=float(data.max()),
            meanv=float(data.mean()),
            stdv=float(data.std()),
            normv=float(norm(data)),
            embedding_peek=2,
            horizontal=True,
            show_stats=False,
        )
        table = group.renderables[0].renderable
        self.assertEqual(len(table.columns), 5)

    def test_model_display_sentence_transformer(self):
        cfg = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=None,
            bos_token=None,
            decoder_start_token_id=None,
            eos_token_id=None,
            eos_token=None,
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type=None,
            max_position_embeddings=10,
            model_type="t",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=None,
            pad_token=None,
            prefix=None,
            sep_token_id=None,
            sep_token=None,
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        st_cfg = SentenceTransformerModelConfig(
            backend="torch",
            similarity_function="cosine",
            truncate_dimension=None,
            transformer_model_config=cfg,
        )
        tok_cfg = TokenizerConfig(
            name_or_path="t",
            tokens=["a"],
            special_tokens=["b"],
            tokenizer_model_max_length=10,
            fast=True,
        )
        group = self.theme.model_display(st_cfg, tok_cfg)
        self.assertEqual(len(group.renderables), 2)

    def test_welcome(self):
        user = User(name="u", access_token_name="tok")
        pad = self.theme.welcome("http://u", "avalan", "1.0", "MIT", user)
        text = str(pad.renderable.renderable)
        self.assertIn("avalan", text)
        self.assertIn("1.0", text)
        self.assertIn("tok", text)


class FancyThemeMoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_cache_delete_none(self):
        result = self.theme.cache_delete(None)
        self.assertIsInstance(result, Text)
        self.assertIn("Nothing found", result.plain)

    def test_cache_list_multiple_revision_files(self):
        now = datetime.now()
        file1 = HubCacheFile(
            name="f1",
            path="/f1",
            size_on_disk=1,
            last_accessed=now,
            last_modified=now,
        )
        file2 = HubCacheFile(
            name="f2",
            path="/f2",
            size_on_disk=1,
            last_accessed=now,
            last_modified=now,
        )
        cache = HubCache(
            model_id="m",
            path="/p",
            size_on_disk=2,
            revisions=["r1", "r2"],
            files={"r1": [file1, file2], "r2": []},
            total_files=2,
            total_revisions=2,
        )
        group = self.theme.cache_list(
            "/c",
            [cache],
            display_models=["m"],
            show_summary=False,
        )
        table = group.renderables[0].renderable
        self.assertEqual(table.row_count, 2)
        self.assertIn("[bright_black]", table.columns[0]._cells[1])

    def test_memory_partitions_many(self):
        part = TextPartition(
            data="t", total_tokens=1, embeddings=np.array([1])
        )
        group = self.theme.memory_partitions([part] * 5, display_partitions=3)
        self.assertEqual(len(group.renderables), 4)

    def test_sentence_transformer_model_config(self):
        cfg = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=1,
            bos_token="<s>",
            decoder_start_token_id=None,
            eos_token_id=2,
            eos_token="</s>",
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type="ce",
            max_position_embeddings=10,
            model_type="t",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=0,
            pad_token="<pad>",
            prefix="pre",
            sep_token_id=3,
            sep_token="<sep>",
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        st_cfg = SentenceTransformerModelConfig(
            backend="torch",
            similarity_function="cosine",
            truncate_dimension=128,
            transformer_model_config=cfg,
        )
        align = self.theme._sentence_transformer_model_config(
            st_cfg, is_runnable=True, summary=False
        )
        table = align.renderable
        headers = table.columns[0]._cells
        self.assertIn("Truncate dimension", headers)

    def test_display_image_entities(self):
        align = self.theme.display_image_entities(
            [
                ImageEntity(label="cat", score=0.5, box=[0.0, 1.0, 2.0, 3.0]),
                ImageEntity(label="dog", score=0.9, box=None),
            ],
            sort=True,
        )
        table = align.renderable
        self.assertEqual(table.row_count, 2)
        # dog should be first due to higher score
        self.assertEqual(table.columns[0]._cells[0], "dog")
        self.assertEqual(table.columns[0]._cells[1], "cat")
        self.assertEqual(table.columns[1]._cells[0], "[score]0.90[/score]")
        self.assertEqual(table.columns[1]._cells[1], "[score]0.50[/score]")

    def test_display_image_entity(self):
        align = self.theme.display_image_entity(ImageEntity(label="cat"))
        table = align.renderable
        self.assertEqual(table.row_count, 1)
        self.assertEqual(table.columns[0]._cells[0], "cat")

    def test_display_image_labels(self):
        align = self.theme.display_image_labels(["cat", "dog"])
        table = align.renderable
        self.assertEqual(table.row_count, 2)
        self.assertEqual(table.columns[0]._cells[0], "cat")
        self.assertEqual(table.columns[0]._cells[1], "dog")

    def test_display_audio_labels(self):
        align = self.theme.display_audio_labels({"dog": 0.9, "cat": 0.5})
        table = align.renderable
        self.assertEqual(table.row_count, 2)
        self.assertEqual(table.columns[0]._cells[0], "dog")
        self.assertEqual(table.columns[0]._cells[1], "cat")
        self.assertEqual(table.columns[1]._cells[0], "[score]0.90[/score]")
        self.assertEqual(table.columns[1]._cells[1], "[score]0.50[/score]")

    def test_display_token_labels(self):
        align = self.theme.display_token_labels([{"tok": "LBL"}])
        table = align.renderable
        self.assertEqual(table.row_count, 1)
        self.assertEqual(table.columns[0]._cells[0], "tok")
        self.assertEqual(table.columns[1]._cells[0], "LBL")

    def test_fill_model_config_table(self):
        cfg = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=1,
            bos_token="<s>",
            decoder_start_token_id=None,
            eos_token_id=2,
            eos_token="</s>",
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type="ce",
            max_position_embeddings=10,
            model_type="t",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=0,
            pad_token="<pad>",
            prefix="pre",
            sep_token_id=3,
            sep_token="<sep>",
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        table = Table(show_lines=True)
        table.add_column()
        table.add_column()
        filled = self.theme._fill_model_config_table(
            cfg, table, is_runnable=True, summary=False
        )
        cells = filled.columns[0]._cells
        self.assertIn("Runs on this instance", cells)
        self.assertIn("Architectures", cells)
        self.assertIn("Start of stream token", cells)

    def test_fill_model_config_table_pad_token(self):
        cfg = ModelConfig(
            architectures=["a"],
            attribute_map={},
            bos_token_id=1,
            bos_token="<s>",
            decoder_start_token_id=None,
            eos_token_id=2,
            eos_token="</s>",
            finetuning_task=None,
            hidden_size=1,
            hidden_sizes=None,
            keys_to_ignore_at_inference=[],
            loss_type="ce",
            max_position_embeddings=10,
            model_type="t",
            num_attention_heads=1,
            num_hidden_layers=1,
            num_labels=1,
            output_attentions=False,
            output_hidden_states=False,
            pad_token_id=4,
            pad_token="<pad>",
            prefix=None,
            sep_token_id=None,
            sep_token=None,
            state_size=1,
            task_specific_params=None,
            torch_dtype=float,
            vocab_size=10,
            tokenizer_class=None,
        )
        table = Table(show_lines=True)
        table.add_column()
        table.add_column()
        filled = self.theme._fill_model_config_table(
            cfg, table, is_runnable=True, summary=False
        )
        cells = filled.columns[0]._cells
        self.assertIn("Padding token", cells)

    def test_tokenizer_config_tokens(self):
        cfg = TokenizerConfig(
            name_or_path="t",
            tokens=["a"],
            special_tokens=["b"],
            tokenizer_model_max_length=10,
            fast=True,
        )
        panel = self.theme.tokenizer_config(cfg)
        headers = panel.renderable.columns[0]._cells
        self.assertIn("Added tokens", headers)

    def test_parameter_count_none(self):
        self.assertEqual(self.theme._parameter_count(None), "N/A")

    def test_symmetric_indices(self):
        self.assertEqual(
            FancyTheme._symmetric_indices([0.1, 0.5, 0.2, 0.4]),
            [2, 0, 1, 3],
        )

    def test_percentage(self):
        self.assertEqual(FancyTheme._percentage(0.5), "50%")
        self.assertEqual(FancyTheme._percentage(0.123), "12.3%")
        self.assertEqual(FancyTheme._percentage(1), "100%")


class FancyThemeWrapLinesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_parameter_count_values(self):
        self.assertEqual(self.theme._parameter_count(10_000), "10.0 thousand")
        self.assertEqual(self.theme._parameter_count(2_000_000_000), "2.0B")

    def test_wrap_lines_blank_lines(self):
        result = FancyTheme._wrap_lines(["a\n\nb"], width=10)
        self.assertEqual(result, ["a", "", "b"])

    def test_wrap_lines_skip_blank_lines(self):
        result = FancyTheme._wrap_lines(
            ["a\n\nb"], width=10, skip_blank_lines=True
        )
        self.assertEqual(result, ["a", "b"])


class FancyThemeEventsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.theme = FancyTheme(
            lambda s: s, lambda s, p, n: s if n == 1 else p
        )

    def test_no_events(self):
        self.assertIsNone(self.theme.events([]))
        self.assertIsNone(self.theme.events([], events_limit=0))

    def test_single_event(self):
        event = Event(type=EventType.START)
        panel = self.theme.events([event])
        self.assertEqual(panel.height, 4)
        self.assertIn("<start>", str(panel.renderable))

        panel = self.theme.events([event], events_limit=1)
        self.assertEqual(panel.height, 3)
        self.assertIn("<start>", str(panel.renderable))

    def test_multiple_events_with_limit(self):
        e1 = Event(type=EventType.START)
        e2 = Event(type=EventType.END)

        panel = self.theme.events([e1, e2])
        text = str(panel.renderable)
        self.assertEqual(panel.height, 4)
        self.assertIn("<start>", text)
        self.assertIn("<end>", text)

        panel = self.theme.events([e1, e2], events_limit=1)
        text = str(panel.renderable)
        self.assertEqual(panel.height, 3)
        self.assertNotIn("<start>", text)
        self.assertIn("<end>", text)

        panel = self.theme.events([e1, e2], events_limit=2)
        text = str(panel.renderable)
        self.assertEqual(panel.height, 4)
        self.assertIn("<start>", text)
        self.assertIn("<end>", text)
