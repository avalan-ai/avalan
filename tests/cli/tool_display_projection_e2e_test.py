from collections.abc import AsyncIterator, Mapping
from io import StringIO
from logging import getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase

from rich.console import Console

from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.display_reducer import CliStreamSnapshotReducer
from avalan.cli.display_snapshot import CliStreamSnapshot
from avalan.cli.theme.basic import BasicStreamPresenter
from avalan.cli.theme.fancy import FancyStreamPresenter, FancyTheme
from avalan.cli.theme.stream_presenter import (
    CliStreamAnswerTextChunk,
    CliStreamPresenter,
    CliStreamPresenterContext,
    CliStreamPresenterRequest,
    CliStreamRenderableFrame,
)
from avalan.entities import (
    ToolCall,
    ToolCallResult,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamVisibility,
    iter_stream_consumer_projections,
    stream_channel_for_kind,
)
from avalan.tool.builtin_display import project_calculator_tool_display
from avalan.tool.display import (
    TOOL_DISPLAY_PROJECTION_METADATA_KEY,
    ToolDisplayDetail,
    ToolDisplayProjection,
)
from avalan.tool.shell.display import project_shell_command_request
from avalan.tool.shell.entities import PathOperand, ShellCommandRequest

_RAW_ARGUMENT_SECRET = "RAW_ARGUMENT_SECRET"
_RAW_RESULT_SECRET = "RAW_RESULT_SECRET"
_DENIED_SHELL_PATH = "/private/fixture/denied.txt"
_DB_DSN = "postgresql+asyncpg://db_user:db-password@example.test/app"
_PROVIDER_SENTINEL = "PROVIDER_PAYLOAD_SENTINEL"
_PRIVATE_DATA_SENTINEL = "PRIVATE_DATA_SENTINEL"


def _config(**overrides: object) -> CliStreamDisplayConfig:
    values: dict[str, object] = {
        "quiet": False,
        "stats": False,
        "display_tools": True,
        "display_events": False,
        "display_tools_events": 8,
        "record": False,
        "interactive": False,
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
    return CliStreamDisplayConfig(**cast(Any, values))


def _gettext(message: str) -> str:
    return message


def _ngettext(singular: str, plural: str, count: int) -> str:
    return singular if count == 1 else plural


class _CounterClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class _CanonicalSource:
    def __init__(self, *items: CanonicalStreamItem) -> None:
        self._items = items

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[CanonicalStreamItem]:
        for item in self._items:
            yield item


class ToolDisplayProjectionE2ETestCase(IsolatedAsyncioTestCase):
    async def test_basic_renders_projected_tool_metadata_cleanly(
        self,
    ) -> None:
        snapshot = await _snapshot_from_items(
            _config(),
            *_projected_shell_items(),
        )

        output = await _render_theme("basic", snapshot, _config())

        self.assertIn("Executed tool search needle in workspace", output)
        self.assertIn("Found matches.", output)
        self.assertIn("details: matches=3", output)
        _assert_not_promoted(self, output)

    async def test_fancy_renders_projected_tool_metadata_cleanly(
        self,
    ) -> None:
        snapshot = await _snapshot_from_items(
            _config(),
            *_projected_shell_items(),
        )

        output = await _render_theme("fancy", snapshot, _config())

        self.assertIn("Executed tool search needle in workspace", output)
        self.assertIn("with status completed and outcome matches", output)
        self.assertIn("Found matches.", output)
        self.assertIn("Details matches=3", output)
        _assert_not_promoted(self, output)

    async def test_database_projection_renders_local_result_data_safely(
        self,
    ) -> None:
        snapshot = await _snapshot_from_items(
            _config(),
            *_projected_database_items(),
        )

        for theme_name in ("basic", "fancy"):
            with self.subTest(theme=theme_name):
                output = await _render_theme(theme_name, snapshot, _config())

                self.assertIn("inspect locks in database", output)
                self.assertIn("blocking", output)
                self.assertIn("Found blocking database locks.", output)
                self.assertNotIn("db-password", output)
                self.assertNotIn("db_user", output)
                self.assertNotIn(_DB_DSN, output)
                self.assertNotIn("lock-query-secret", output)
                self.assertNotIn(_RAW_RESULT_SECRET, output)
                self.assertNotIn(_PROVIDER_SENTINEL, output)

    async def test_builtin_projection_renders_through_canonical_metadata(
        self,
    ) -> None:
        snapshot = await _snapshot_from_items(
            _config(),
            *_projected_calculator_items(),
        )

        for theme_name in ("basic", "fancy"):
            with self.subTest(theme=theme_name):
                output = await _render_theme(theme_name, snapshot, _config())

                self.assertIn("calculate 2 + 2 in math", output)
                self.assertIn("Calculation completed.", output)
                self.assertIn("4", output)
                self.assertNotIn(_RAW_ARGUMENT_SECRET, output)
                self.assertNotIn(_RAW_RESULT_SECRET, output)

    async def test_shell_projection_redacts_denied_paths_before_display(
        self,
    ) -> None:
        snapshot = await _snapshot_from_items(
            _config(),
            *_projected_denied_shell_path_items(),
        )

        for theme_name in ("basic", "fancy"):
            with self.subTest(theme=theme_name):
                output = await _render_theme(theme_name, snapshot, _config())

                self.assertIn("read [redacted]", output)
                self.assertIn("Read a file.", output)
                self.assertNotIn(_DENIED_SHELL_PATH, output)
                self.assertNotIn("/private/fixture", output)

    async def test_missing_projection_fallback_still_renders_result(
        self,
    ) -> None:
        snapshot = await _snapshot_from_items(
            _config(),
            *_fallback_items(metadata=None),
        )

        for theme_name in ("basic", "fancy"):
            with self.subTest(theme=theme_name):
                output = await _render_theme(theme_name, snapshot, _config())

                self.assertIn("plain.lookup", output)
                self.assertIn("fallback-result-sentinel", output)

    async def test_malformed_projection_metadata_does_not_crash_rendering(
        self,
    ) -> None:
        snapshot = await _snapshot_from_items(
            _config(),
            *_fallback_items(
                metadata={
                    TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
                        "action": 42,
                        "summary": "MALFORMED_PRIVATE_SENTINEL",
                    }
                }
            ),
        )

        for theme_name in ("basic", "fancy"):
            with self.subTest(theme=theme_name):
                output = await _render_theme(theme_name, snapshot, _config())

                self.assertIn("plain.lookup", output)
                self.assertIn("fallback-result-sentinel", output)
                self.assertNotIn("MALFORMED_PRIVATE_SENTINEL", output)


def _projected_shell_items() -> tuple[CanonicalStreamItem, ...]:
    start_projection = ToolDisplayProjection(
        action="search",
        target="needle",
        scope="workspace",
        summary="Search source files.",
        details=(ToolDisplayDetail(label="matches", value=3),),
    )
    terminal_projection = ToolDisplayProjection(
        action="search",
        target="needle",
        scope="workspace",
        status="completed",
        outcome="matches",
        summary="Found matches.",
        details=(ToolDisplayDetail(label="matches", value=3),),
    )
    return _tool_trace_items(
        tool_call_id="shell-call",
        tool_name="shell.rg",
        argument_delta=(
            '{"pattern": "needle", "api_key": "'
            + _RAW_ARGUMENT_SECRET
            + '", "path": "'
            + _DENIED_SHELL_PATH
            + '"}'
        ),
        start_metadata=start_projection.to_metadata(),
        terminal_metadata=terminal_projection.to_metadata(),
        terminal_data={
            "name": "shell.rg",
            "result": {
                "raw": _RAW_RESULT_SECRET,
                "private": _PRIVATE_DATA_SENTINEL,
            },
        },
        terminal_visibility=StreamVisibility.PRIVATE,
    )


def _projected_database_items() -> tuple[CanonicalStreamItem, ...]:
    start_projection = ToolDisplayProjection(
        action="inspect",
        label="database.locks",
        target="locks",
        scope="database",
        summary="List database locks.",
        details=(
            ToolDisplayDetail(label="operation", value="locks"),
            ToolDisplayDetail(label="dialect", value="postgresql"),
            ToolDisplayDetail(label="read_only", value=True),
        ),
    )
    terminal_projection = ToolDisplayProjection(
        action="inspect",
        label="database.locks",
        target="locks",
        scope="database",
        summary="Found blocking database locks.",
        status="completed",
        outcome="blocking",
        severity="warning",
        details=(
            ToolDisplayDetail(label="operation", value="locks"),
            ToolDisplayDetail(label="locks", value=1),
            ToolDisplayDetail(label="blocking_locks", value=1),
        ),
        metrics={"locks": 1, "blocking_locks": 1},
    )
    return _tool_trace_items(
        tool_call_id="db-locks",
        tool_name="database.locks",
        argument_delta='{"dsn": "' + _DB_DSN + '"}',
        start_metadata=start_projection.to_metadata(),
        terminal_metadata=terminal_projection.to_metadata(),
        terminal_data={
            "name": "database.locks",
            "result": {
                "dsn": _DB_DSN,
                "query": "UPDATE users SET token = 'lock-query-secret'",
                "raw": _RAW_RESULT_SECRET,
            },
        },
        terminal_visibility=StreamVisibility.REDACTED,
    )


def _projected_calculator_items() -> tuple[CanonicalStreamItem, ...]:
    call = ToolCall(
        id="calculator-call",
        name="math.calculator",
        arguments={"expression": "2 + 2"},
    )
    start_projection = project_calculator_tool_display(call=call)
    outcome = ToolCallResult(
        id="calculator-result",
        name=call.name,
        arguments=call.arguments,
        call=call,
        result="4",
    )
    terminal_projection = project_calculator_tool_display(
        call=call,
        outcome=outcome,
    )
    return _tool_trace_items(
        tool_call_id="calculator-call",
        tool_name="math.calculator",
        argument_delta=(
            '{"expression": "2 + 2", "api_key": "'
            + _RAW_ARGUMENT_SECRET
            + '"}'
        ),
        start_metadata=start_projection.to_metadata(),
        terminal_metadata=terminal_projection.to_metadata(),
        terminal_data={
            "name": "math.calculator",
            "result": _RAW_RESULT_SECRET,
        },
    )


def _projected_denied_shell_path_items() -> tuple[CanonicalStreamItem, ...]:
    request = ShellCommandRequest(
        tool_name="shell.cat",
        command="cat",
        options={},
        paths=(
            PathOperand(
                name="path",
                path=_DENIED_SHELL_PATH,
                kind="text_file",
                access="read",
            ),
        ),
        cwd=None,
    )
    projection = project_shell_command_request(request)
    terminal_projection = ToolDisplayProjection(
        action=projection.action,
        target=projection.target,
        scope=projection.scope,
        summary=projection.summary,
        status="completed",
        outcome="result",
        details=projection.details,
        redacted=projection.redacted,
    )
    return _tool_trace_items(
        tool_call_id="shell-denied",
        tool_name="shell.cat",
        argument_delta='{"path": "' + _DENIED_SHELL_PATH + '"}',
        start_metadata=projection.to_metadata(),
        terminal_metadata=terminal_projection.to_metadata(),
        terminal_data={"name": "shell.cat", "result": "done"},
    )


def _fallback_items(
    *,
    metadata: Mapping[str, object] | None,
) -> tuple[CanonicalStreamItem, ...]:
    return _tool_trace_items(
        tool_call_id="fallback-call",
        tool_name="plain.lookup",
        argument_delta='{"query": "fallback"}',
        start_metadata=metadata,
        terminal_metadata=metadata,
        terminal_data={
            "name": "plain.lookup",
            "result": "fallback-result-sentinel",
        },
    )


def _tool_trace_items(
    *,
    tool_call_id: str,
    tool_name: str,
    argument_delta: str,
    start_metadata: Mapping[str, object] | None,
    terminal_metadata: Mapping[str, object] | None,
    terminal_data: object,
    terminal_visibility: StreamVisibility = StreamVisibility.PUBLIC,
) -> tuple[CanonicalStreamItem, ...]:
    return (
        _item(StreamItemKind.STREAM_STARTED, 0),
        _item(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            1,
            tool_call_id=tool_call_id,
            text_delta=argument_delta,
            data={"name": tool_name, "arguments": argument_delta},
        ),
        _item(
            StreamItemKind.TOOL_CALL_READY,
            2,
            tool_call_id=tool_call_id,
            data={"name": tool_name},
        ),
        _item(
            StreamItemKind.TOOL_CALL_DONE,
            3,
            tool_call_id=tool_call_id,
        ),
        _item(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            4,
            tool_call_id=tool_call_id,
            data={"name": tool_name},
            metadata=start_metadata,
            provider_payload={"raw": _PROVIDER_SENTINEL},
        ),
        _item(
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            5,
            tool_call_id=tool_call_id,
            data=terminal_data,
            metadata=terminal_metadata,
            provider_payload={"raw": _PROVIDER_SENTINEL},
            visibility=terminal_visibility,
        ),
        _item(StreamItemKind.ANSWER_DELTA, 6, text_delta="done"),
        _item(StreamItemKind.ANSWER_DONE, 7),
        _item(
            StreamItemKind.STREAM_COMPLETED,
            8,
            usage={"output_tokens": 1},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    )


def _item(
    kind: StreamItemKind,
    sequence: int,
    *,
    tool_call_id: str | None = None,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    metadata: Mapping[str, object] | None = None,
    provider_payload: object | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
    terminal_outcome: StreamTerminalOutcome | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="session",
        run_id="run",
        turn_id="turn",
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
        text_delta=text_delta,
        data=cast(Any, data),
        usage=cast(Any, usage),
        metadata=cast(dict[str, Any], dict(metadata or {})),
        provider_payload=cast(Any, provider_payload),
        visibility=visibility,
        terminal_outcome=terminal_outcome,
    )


async def _snapshot_from_items(
    config: CliStreamDisplayConfig,
    *items: CanonicalStreamItem,
) -> CliStreamSnapshot:
    reducer = CliStreamSnapshotReducer(config, clock=_CounterClock())
    async for projection in iter_stream_consumer_projections(
        _CanonicalSource(*items)
    ):
        reducer.reduce_projection(projection)
    return reducer.snapshot()


async def _render_theme(
    theme_name: str,
    snapshot: CliStreamSnapshot,
    config: CliStreamDisplayConfig,
) -> str:
    presenter: CliStreamPresenter
    if theme_name == "basic":
        presenter = BasicStreamPresenter(getLogger(__name__))
    elif theme_name == "fancy":
        presenter = FancyStreamPresenter(
            FancyTheme(_gettext, _ngettext),
            getLogger(__name__),
        )
    else:
        raise AssertionError("unsupported theme")
    request = CliStreamPresenterRequest(
        snapshot=snapshot,
        display_config=config,
        context=CliStreamPresenterContext(
            model_id="model",
            console_width=160,
        ),
        mode="live",
    )
    parts: list[str] = []
    async for item in presenter.present(request):
        if isinstance(item, CliStreamAnswerTextChunk):
            parts.append(item.text)
        elif isinstance(item, CliStreamRenderableFrame):
            parts.append(_render_text(item.renderable))
    return "".join(parts)


def _render_text(renderable: object) -> str:
    output = StringIO()
    Console(file=output, force_terminal=False, width=180).print(renderable)
    return output.getvalue()


def _assert_not_promoted(
    test_case: IsolatedAsyncioTestCase,
    output: str,
) -> None:
    for fragment in (
        _RAW_ARGUMENT_SECRET,
        _RAW_RESULT_SECRET,
        _DENIED_SHELL_PATH,
        _PROVIDER_SENTINEL,
        _PRIVATE_DATA_SENTINEL,
        '"api_key"',
        '"path"',
        '"pattern"',
        '"raw"',
        "provider_payload",
    ):
        with test_case.subTest(fragment=fragment):
            test_case.assertNotIn(fragment, output)
