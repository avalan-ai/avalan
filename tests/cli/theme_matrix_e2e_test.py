from argparse import Namespace
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from logging import getLogger
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from avalan.cli.commands import agent as agent_cmds
from avalan.cli.commands import model as model_cmds
from avalan.cli.display import (
    CliStreamDisplayConfig,
    cli_stream_display_config,
)
from avalan.cli.theme import Theme
from avalan.cli.theme_registry import create_theme
from avalan.entities import Modality, Model
from avalan.event import Event
from avalan.model.manager import ModelManager as RealModelManager
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    project_canonical_stream_item,
    stream_channel_for_kind,
)


@dataclass(frozen=True, slots=True)
class MatrixFlagCase:
    label: str
    overrides: dict[str, object]


class MatrixProjectionResponse:
    input_token_count = 1
    can_think = False
    is_thinking = False

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def set_thinking(self, value: bool) -> None:
        self.is_thinking = value

    def consumer_projections(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
    ) -> AsyncIterator[StreamConsumerProjection]:
        async def gen() -> AsyncIterator[StreamConsumerProjection]:
            for item in self._items(stream_session_id, run_id, turn_id):
                yield project_canonical_stream_item(item)

        return gen()

    def _items(
        self,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
    ) -> tuple[CanonicalStreamItem, ...]:
        tool_correlation = StreamItemCorrelation(tool_call_id="matrix-tool")
        return (
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                0,
                StreamItemKind.STREAM_STARTED,
            ),
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                1,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                correlation=tool_correlation,
                data={"name": "matrix.lookup"},
            ),
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                2,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                correlation=tool_correlation,
                data={"result": "matrix-result-sentinel"},
            ),
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                3,
                StreamItemKind.FLOW_EVENT,
                data={"stage": "matrix"},
            ),
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                4,
                StreamItemKind.USAGE_UPDATE,
                usage={"input_tokens": 1, "output_tokens": 2},
            ),
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                5,
                StreamItemKind.ANSWER_DELTA,
                text_delta=self._answer,
            ),
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                6,
                StreamItemKind.ANSWER_DONE,
            ),
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                7,
                StreamItemKind.STREAM_COMPLETED,
                usage={
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )


class MatrixModel:
    model_id = "matrix-model"
    tokenizer_config = None
    config = SimpleNamespace()

    def input_token_count(self, *_args: object, **_kwargs: object) -> int:
        return 1


class MatrixModelManager:
    engine_uri = SimpleNamespace(
        model_id="matrix-model",
        is_local=True,
        params={},
    )

    def __init__(self, hub: object, logger: object) -> None:
        self.hub = hub
        self.logger = logger
        self.model = MatrixModel()

    def __enter__(self) -> "MatrixModelManager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        _ = exc_type, exc_value, traceback
        return False

    @staticmethod
    def get_operation_from_arguments(
        modality: Modality,
        args: Namespace,
        input_string: object | None,
    ) -> object:
        return RealModelManager.get_operation_from_arguments(
            modality,
            args,
            input_string,
        )

    def parse_uri(self, model: str) -> object:
        assert model == "matrix-model"
        return self.engine_uri

    def load(self, **_settings: object) -> object:
        manager = self

        class LoadContext:
            def __enter__(self) -> MatrixModel:
                return manager.model

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_value: BaseException | None,
                traceback: object,
            ) -> bool:
                _ = exc_type, exc_value, traceback
                return False

        return LoadContext()

    async def __call__(self, task: object) -> MatrixProjectionResponse:
        _ = task
        return MatrixProjectionResponse("model matrix answer")


class MatrixEventManager:
    def __init__(self) -> None:
        self.listeners: list[object] = []

    def add_listener(self, listener: object) -> None:
        self.listeners.append(listener)

    def remove_listener(self, listener: object) -> None:
        self.listeners.remove(listener)

    async def listen(
        self,
        *,
        stop_signal: object,
    ) -> AsyncIterator[Event]:
        _ = stop_signal
        if False:
            yield Event()


class MatrixOrchestrator:
    id = "matrix-agent"
    name = "Matrix Agent"
    model_ids = ["matrix-agent-model"]

    def __init__(self) -> None:
        self.event_manager = MatrixEventManager()
        self.memory = SimpleNamespace(
            has_recent_message=False,
            has_permanent_message=False,
            recent_message=SimpleNamespace(is_empty=True, size=0, data=[]),
            permanent_message=None,
            start_session=AsyncMock(),
            continue_session=AsyncMock(),
        )
        self.engine = SimpleNamespace(
            model_id="matrix-agent-model",
            model_type="text",
            tokenizer_config=None,
            input_token_count=lambda *_args, **_kwargs: 1,
        )
        self.engine_agent = SimpleNamespace(
            engine_uri=SimpleNamespace(params={}),
        )
        self.tool = SimpleNamespace(is_empty=True)
        self.calls: list[str] = []

    async def __aenter__(self) -> "MatrixOrchestrator":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        _ = exc_type, exc_value, traceback
        return False

    async def __call__(self, input_string: str, **_kwargs: object) -> object:
        self.calls.append(input_string)
        return MatrixProjectionResponse("agent matrix answer")


def _canonical_item(
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    sequence: int,
    kind: StreamItemKind,
    *,
    correlation: StreamItemCorrelation | None = None,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        data=data,  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        terminal_outcome=terminal_outcome,
    )


def _theme(name: str) -> Theme:
    return create_theme(
        name,
        lambda message: message,
        lambda singular, plural, count: singular if count == 1 else plural,
    )


def _model_summary(model_id: str) -> Model:
    now = datetime(2024, 1, 1)
    return Model(
        id=model_id,
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


def _flag_cases() -> tuple[MatrixFlagCase, ...]:
    return (
        MatrixFlagCase("default", {}),
        MatrixFlagCase("quiet", {"quiet": True}),
        MatrixFlagCase("stats", {"stats": True}),
        MatrixFlagCase("display-tools", {"display_tools": True}),
        MatrixFlagCase("display-events", {"display_events": True}),
        MatrixFlagCase(
            "display-tools-events",
            {"display_tools": True, "display_events": True},
        ),
        MatrixFlagCase(
            "display-tools-events-0",
            {
                "display_tools": True,
                "display_events": True,
                "display_tools_events": 0,
            },
        ),
        MatrixFlagCase("record", {"record": True, "stats": True}),
    )


def _base_stream_args(**overrides: object) -> dict[str, object]:
    values = {
        "quiet": False,
        "stats": False,
        "display_tools": False,
        "display_events": False,
        "display_tools_events": 2,
        "record": False,
        "display_tokens": 0,
        "display_pause": 0,
        "display_probabilities": False,
        "display_probabilities_maximum": 0.8,
        "display_probabilities_sample_minimum": 0.1,
        "display_time_to_n_token": None,
        "display_answer_height": 12,
        "display_answer_height_expand": False,
        "skip_display_reasoning_time": False,
        "start_thinking": False,
    }
    values.update(overrides)
    return values


def _model_args(case: MatrixFlagCase) -> Namespace:
    values = _base_stream_args(
        model="matrix-model",
        device="cpu",
        max_new_tokens=4,
        skip_hub_access_check=True,
        no_repl=True,
        do_sample=False,
        enable_gradient_calculation=False,
        min_p=None,
        repetition_penalty=1.0,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        use_cache=True,
        stop_on_keyword=None,
        system=None,
        instructions=None,
        developer=None,
        skip_special_tokens=False,
        input_file=None,
        backend="transformers",
        disable_loading_progress_bar=True,
        loader_class="auto",
        low_cpu_mem_usage=False,
        revision=None,
        special_token=None,
        token=None,
        tokenizer=None,
        output_hidden_states=False,
        cache_strategy=None,
        chat_disable_thinking=True,
        no_reasoning=True,
    )
    values.update(case.overrides)
    return Namespace(**values)


def _agent_args(case: MatrixFlagCase) -> Namespace:
    values = _base_stream_args(
        specifications_file=None,
        use_sync_generator=False,
        id="aid",
        participant="pid",
        session="sid",
        no_session=False,
        skip_load_recent_messages=False,
        load_recent_messages_limit=1,
        no_repl=True,
        skip_hub_access_check=True,
        conversation=False,
        watch=False,
        tty=None,
        tool_events=2,
        tool=None,
        tool_format=None,
        tools=None,
        run_max_new_tokens=4,
        run_skip_special_tokens=False,
        run_temperature=None,
        run_top_k=None,
        run_top_p=None,
        run_use_cache=None,
        run_cache_strategy=None,
        engine_uri="matrix-agent-model",
        name="Matrix",
        role="assistant",
        task=None,
        instructions=None,
        memory_recent=True,
        memory_permanent_message=None,
        memory_permanent=None,
        memory_engine_model_id=(
            agent_cmds.OrchestratorLoader.DEFAULT_SENTENCE_MODEL_ID
        ),
        memory_engine_max_tokens=500,
        memory_engine_overlap=125,
        memory_engine_window=250,
        tool_browser_engine=None,
        tool_browser_debug=None,
        tool_browser_search=None,
        tool_browser_search_context=None,
        tools_confirm=False,
        backend="transformers",
    )
    values.update(case.overrides)
    return Namespace(**values)


def _console(*, interactive: bool) -> MagicMock:
    console = MagicMock()
    console.width = 100
    console.is_terminal = interactive
    status = MagicMock()
    status.__enter__.return_value = None
    status.__exit__.return_value = False
    console.status.return_value = status
    return console


def _printed_text(console: MagicMock) -> str:
    return "".join(
        _render_text(arg)
        for print_call in console.print.call_args_list
        for arg in print_call.args
    )


def _render_text(renderable: object) -> str:
    if isinstance(renderable, str):
        return renderable
    output = StringIO()
    Console(file=output, force_terminal=False, width=140).print(renderable)
    return output.getvalue()


def _rendered_mock_text(mock: MagicMock) -> str:
    return "".join(
        _render_text(call_args.args[0])
        for call_args in mock.call_args_list
        if call_args.args
    )


class ThemeMatrixE2ETestCase(IsolatedAsyncioTestCase):
    async def test_model_run_theme_display_matrix(self) -> None:
        for theme_name in ("basic", "fancy"):
            for interactive in (True, False):
                for case in _flag_cases():
                    with self.subTest(
                        theme=theme_name,
                        interactive=interactive,
                        flags=case.label,
                    ):
                        args = _model_args(case)
                        display_config = cli_stream_display_config(
                            args,
                            refresh_per_second=2,
                            interactive=interactive,
                        )
                        console = _console(interactive=interactive)
                        hub = MagicMock()
                        hub.can_access.return_value = True
                        hub.model.side_effect = _model_summary
                        await self._run_model_case(
                            args,
                            console,
                            _theme(theme_name),
                            hub,
                            display_config,
                        )

    async def test_agent_run_theme_display_matrix(self) -> None:
        for theme_name in ("basic", "fancy"):
            for interactive in (True, False):
                for case in _flag_cases():
                    with self.subTest(
                        theme=theme_name,
                        interactive=interactive,
                        flags=case.label,
                    ):
                        args = _agent_args(case)
                        display_config = cli_stream_display_config(
                            args,
                            refresh_per_second=2,
                            interactive=interactive,
                        )
                        console = _console(interactive=interactive)
                        hub = MagicMock()
                        hub.can_access.return_value = True
                        hub.model.side_effect = _model_summary
                        await self._run_agent_case(
                            args,
                            console,
                            _theme(theme_name),
                            hub,
                            display_config,
                        )

    async def _run_model_case(
        self,
        args: Namespace,
        console: MagicMock,
        theme: Theme,
        hub: MagicMock,
        display_config: CliStreamDisplayConfig,
    ) -> None:
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        diagnostic_console = MagicMock()
        with (
            patch.object(model_cmds, "ModelManager", MatrixModelManager),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": MatrixModelManager.engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ),
            patch.object(model_cmds, "get_input", return_value="prompt"),
            patch("avalan.cli.stream_coordinator.Live", return_value=live),
            patch(
                "avalan.cli.stream_coordinator.Console",
                return_value=diagnostic_console,
            ) as diagnostic_console_cls,
        ):
            await model_cmds.model_run(
                args,
                console,
                theme,
                hub,
                2,
                getLogger(__name__),
            )

        self._assert_display_semantics(
            console,
            live,
            diagnostic_console,
            diagnostic_console_cls,
            display_config,
            supports_stderr_diagnostics=bool(
                getattr(
                    theme.stream_presenter(getLogger(__name__)),
                    "supports_stderr_diagnostics",
                    False,
                )
            ),
            answer="model matrix answer",
        )

    async def _run_agent_case(
        self,
        args: Namespace,
        console: MagicMock,
        theme: Theme,
        hub: MagicMock,
        display_config: CliStreamDisplayConfig,
    ) -> None:
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        diagnostic_console = MagicMock()
        orchestrator = MatrixOrchestrator()
        with (
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_settings",
                new=AsyncMock(return_value=orchestrator),
            ),
            patch.object(
                agent_cmds.OrchestratorLoader,
                "from_file",
                new=AsyncMock(),
            ),
            patch.object(agent_cmds, "get_input", return_value="prompt"),
            patch.object(
                agent_cmds, "OrchestratorResponse", MatrixProjectionResponse
            ),
            patch("avalan.cli.stream_coordinator.Live", return_value=live),
            patch(
                "avalan.cli.stream_coordinator.Console",
                return_value=diagnostic_console,
            ) as diagnostic_console_cls,
        ):
            await agent_cmds.agent_run(
                args,
                console,
                theme,
                hub,
                getLogger(__name__),
                2,
            )

        self.assertEqual(orchestrator.calls, ["prompt"])
        self._assert_display_semantics(
            console,
            live,
            diagnostic_console,
            diagnostic_console_cls,
            display_config,
            supports_stderr_diagnostics=bool(
                getattr(
                    theme.stream_presenter(getLogger(__name__)),
                    "supports_stderr_diagnostics",
                    False,
                )
            ),
            answer="agent matrix answer",
        )

    def _assert_display_semantics(
        self,
        console: MagicMock,
        live: MagicMock,
        diagnostic_console: MagicMock,
        diagnostic_console_cls: MagicMock,
        display_config: CliStreamDisplayConfig,
        *,
        supports_stderr_diagnostics: bool,
        answer: str,
    ) -> None:
        output = "".join(
            (
                _printed_text(console),
                _rendered_mock_text(live.update),
                _rendered_mock_text(diagnostic_console.print),
            )
        )
        self.assertIn(answer, output)
        self.assertEqual(live.update.called, display_config.live_enabled)
        self.assertEqual(
            diagnostic_console_cls.called,
            display_config.diagnostic_channel == "stderr"
            and supports_stderr_diagnostics,
        )
        self._assert_flag_sentinels(
            output,
            display_config,
            supports_stderr_diagnostics=supports_stderr_diagnostics,
        )
        if display_config.record_enabled:
            console.save_svg.assert_called()
        else:
            console.save_svg.assert_not_called()
        if display_config.quiet:
            self.assertFalse(display_config.show_stats)
            self.assertFalse(display_config.show_tools)
            self.assertFalse(display_config.show_events)

    def _assert_flag_sentinels(
        self,
        output: str,
        display_config: CliStreamDisplayConfig,
        *,
        supports_stderr_diagnostics: bool,
    ) -> None:
        diagnostics_visible = display_config.live_enabled or (
            display_config.diagnostic_channel == "stderr"
            and supports_stderr_diagnostics
        )
        if display_config.show_tools and diagnostics_visible:
            self.assertIn("matrix.lookup", output)
            if display_config.display_tools_events == 0:
                self.assertNotIn("matrix-result-sentinel", output)
            else:
                self.assertIn("matrix-result-sentinel", output)
        elif not display_config.show_stats:
            self.assertNotIn("matrix.lookup", output)

        if display_config.show_events and diagnostics_visible:
            self.assertIn("stage", output)
        elif not display_config.show_stats:
            self.assertNotIn("stage", output)

        if display_config.show_stats and diagnostics_visible:
            self.assertIn("usage", output)
        else:
            self.assertNotIn("usage", output)
