from argparse import Namespace
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from itertools import combinations, product
from json import loads
from logging import getLogger
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from avalan.cli.commands import agent as agent_cmds
from avalan.cli.commands import model as model_cmds
from avalan.cli.display import (
    CliStreamDisplayConfig,
    cli_stream_display_config,
)
from avalan.cli.display_reducer import CliStreamSnapshotReducer
from avalan.cli.stream_presenter import (
    CliStreamPresenterContext,
    CliStreamPresenterItem,
    CliStreamPresenterRequest,
    CliStreamRenderableFrame,
)
from avalan.cli.theme import Theme
from avalan.cli.theme_registry import create_theme
from avalan.entities import Modality, Model
from avalan.event import Event
from avalan.model.manager import ModelManager as RealModelManager
from avalan.model.stream import (
    REASONING_SEGMENT_BOUNDARY_METADATA_KEY,
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamVisibility,
    project_canonical_stream_item,
    stream_channel_for_kind,
)


@dataclass(frozen=True, slots=True)
class MatrixFlagCase:
    label: str
    overrides: dict[str, object]


@dataclass(frozen=True, slots=True)
class ReasoningMatrixCase:
    id: str
    theme: str
    terminal: str
    representation: str
    display: str
    statistics: str
    quiet: str
    recording: str
    answer: str
    coverage: str
    case_name: str | None
    entrypoint: str | None

    @property
    def interactive(self) -> bool:
        return self.terminal == "interactive"

    @property
    def displays_reasoning(self) -> bool:
        return self.display == "display-reasoning"

    @property
    def shows_reasoning(self) -> bool:
        return self.displays_reasoning and self.quiet == "off"

    @property
    def has_reasoning(self) -> bool:
        return self.representation != "empty"


@dataclass(frozen=True, slots=True)
class MatrixCommandCapture:
    stdout: str
    stderr: str
    live: str
    live_updates: int
    recordings: int
    diagnostic_console_created: bool


_REASONING_MATRIX_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cli"
    / "reasoning_display_pairwise_matrix.json"
)
_REASONING_MATRIX_FACTORS: dict[str, tuple[str, ...]] = {
    "theme": ("basic", "fancy"),
    "terminal": ("interactive", "non-interactive"),
    "representation": ("native", "summary", "mixed", "empty"),
    "display": ("off", "display-reasoning"),
    "statistics": ("off", "on"),
    "quiet": ("off", "on"),
    "recording": ("off", "on"),
    "answer": ("text", "strict-json"),
}
_REASONING_MATRIX_E2E_CASES: dict[str, tuple[str, str]] = {
    "r00": ("model-basic-nontty-native-hidden-text", "model"),
    "r17": ("agent-basic-nontty-summary-stderr-text", "agent"),
    "r19": ("model-basic-nontty-empty-no-frame-text", "model"),
    "r24": ("agent-basic-tty-native-quiet-json", "agent"),
    "r28": ("model-fancy-tty-native-display-only-json", "model"),
    "r29": ("agent-fancy-tty-summary-display-only-json", "agent"),
    "r30": ("model-fancy-tty-mixed-display-only-json", "model"),
    "r40": ("agent-basic-tty-native-stats-only-text", "agent"),
    "r53": ("model-fancy-nontty-summary-stderr-stats-json", "model"),
    "r55": ("agent-fancy-nontty-empty-no-frame-json", "agent"),
    "r57": ("model-basic-tty-summary-recorded-text", "model"),
    "r58": ("agent-basic-tty-mixed-recorded-text", "agent"),
}
_NATIVE_REASONING_SENTINEL = "native-reasoning-sentinel"
_SUMMARY_REASONING_SENTINEL = "summary-reasoning-sentinel"
_REASONING_SENTINELS = (
    _NATIVE_REASONING_SENTINEL,
    _SUMMARY_REASONING_SENTINEL,
)


def _reasoning_matrix_document() -> dict[str, object]:
    return cast(
        dict[str, object],
        loads(_REASONING_MATRIX_PATH.read_text(encoding="utf-8")),
    )


def _reasoning_matrix_cases() -> tuple[ReasoningMatrixCase, ...]:
    raw_rows = _reasoning_matrix_document()["rows"]
    assert isinstance(raw_rows, list)
    cases: list[ReasoningMatrixCase] = []
    for raw_row in raw_rows:
        assert isinstance(raw_row, dict)
        row = cast(dict[str, object], raw_row)
        cases.append(
            ReasoningMatrixCase(
                id=cast(str, row["id"]),
                theme=cast(str, row["theme"]),
                terminal=cast(str, row["terminal"]),
                representation=cast(str, row["representation"]),
                display=cast(str, row["display"]),
                statistics=cast(str, row["statistics"]),
                quiet=cast(str, row["quiet"]),
                recording=cast(str, row["recording"]),
                answer=cast(str, row["answer"]),
                coverage=cast(str, row["coverage"]),
                case_name=cast(str | None, row["case_name"]),
                entrypoint=cast(str | None, row["entrypoint"]),
            )
        )
    return tuple(cases)


def _expected_reasoning_matrix_values(index: int) -> dict[str, str]:
    representations = ("native", "summary", "mixed", "empty")
    x2 = (index >> 2) & 1
    x3 = (index >> 3) & 1
    x4 = (index >> 4) & 1
    x5 = (index >> 5) & 1
    return {
        "theme": "fancy" if x2 else "basic",
        "terminal": "interactive" if x3 else "non-interactive",
        "representation": representations[index & 3],
        "display": "display-reasoning" if x4 else "off",
        "statistics": "on" if x5 else "off",
        "quiet": "on" if x2 ^ x3 ^ x5 else "off",
        "recording": "on" if x3 ^ x4 ^ x5 else "off",
        "answer": "strict-json" if x3 ^ x5 else "text",
    }


class MatrixProjectionResponse:
    input_token_count = 1
    can_think = False
    is_thinking = False

    def __init__(self, answer: str, *, representation: str = "empty") -> None:
        assert representation in _REASONING_MATRIX_FACTORS["representation"]
        self._answer = answer
        self._representation = representation

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
        items = [
            _canonical_item(
                stream_session_id,
                run_id,
                turn_id,
                0,
                StreamItemKind.STREAM_STARTED,
            )
        ]
        for representation, text in _matrix_reasoning_parts(
            self._representation
        ):
            ordinal = len(
                [
                    item
                    for item in items
                    if item.kind is StreamItemKind.REASONING_DELTA
                ]
            )
            items.append(
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items),
                    StreamItemKind.REASONING_DELTA,
                    correlation=StreamItemCorrelation(
                        protocol_item_id=f"matrix-reasoning-{ordinal}",
                        provider_output_index=ordinal,
                        provider_summary_index=(
                            ordinal
                            if representation
                            is StreamReasoningRepresentation.SUMMARY
                            else None
                        ),
                    ),
                    text_delta=text,
                    visibility=StreamVisibility.PRIVATE,
                    reasoning_representation=representation,
                    segment_instance_ordinal=ordinal,
                    metadata=(
                        {REASONING_SEGMENT_BOUNDARY_METADATA_KEY: "completed"}
                        if ordinal
                        else None
                    ),
                )
            )
        if self._representation != "empty":
            items.append(
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items),
                    StreamItemKind.REASONING_DONE,
                )
            )

        tool_correlation = StreamItemCorrelation(tool_call_id="matrix-tool")
        items.extend(
            (
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items),
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    correlation=tool_correlation,
                    data={"name": "matrix.lookup"},
                ),
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items) + 1,
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    correlation=tool_correlation,
                    data={"result": "matrix-result-sentinel"},
                ),
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items) + 2,
                    StreamItemKind.FLOW_EVENT,
                    data={"stage": "matrix"},
                ),
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items) + 3,
                    StreamItemKind.USAGE_UPDATE,
                    usage={"input_tokens": 1, "output_tokens": 2},
                ),
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items) + 4,
                    StreamItemKind.ANSWER_DELTA,
                    text_delta=self._answer,
                ),
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items) + 5,
                    StreamItemKind.ANSWER_DONE,
                ),
                _canonical_item(
                    stream_session_id,
                    run_id,
                    turn_id,
                    len(items) + 6,
                    StreamItemKind.STREAM_COMPLETED,
                    usage={
                        "input_tokens": 1,
                        "output_tokens": 2,
                        "total_tokens": 3,
                    },
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            )
        )
        return tuple(items)


def _matrix_reasoning_parts(
    representation: str,
) -> tuple[tuple[StreamReasoningRepresentation, str], ...]:
    if representation == "native":
        return (
            (
                StreamReasoningRepresentation.NATIVE_TEXT,
                _NATIVE_REASONING_SENTINEL,
            ),
        )
    if representation == "summary":
        return (
            (
                StreamReasoningRepresentation.SUMMARY,
                _SUMMARY_REASONING_SENTINEL,
            ),
        )
    if representation == "mixed":
        return (
            (
                StreamReasoningRepresentation.NATIVE_TEXT,
                _NATIVE_REASONING_SENTINEL,
            ),
            (
                StreamReasoningRepresentation.SUMMARY,
                _SUMMARY_REASONING_SENTINEL,
            ),
        )
    assert representation == "empty"
    return ()


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

    def __init__(
        self,
        *,
        answer: str = "agent matrix answer",
        representation: str = "empty",
        structured_output: bool = False,
    ) -> None:
        self._answer = answer
        self._representation = representation
        self._call_options = (
            {"response_format": {"type": "json_schema"}}
            if structured_output
            else None
        )
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
        return MatrixProjectionResponse(
            self._answer,
            representation=self._representation,
        )


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
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
    reasoning_representation: StreamReasoningRepresentation | None = None,
    segment_instance_ordinal: int | None = None,
    metadata: dict[str, object] | None = None,
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
        visibility=visibility,
        reasoning_representation=reasoning_representation,
        segment_instance_ordinal=segment_instance_ordinal,
        metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
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
        "display_reasoning": False,
        "start_thinking": False,
    }
    values.update(overrides)
    return values


def _reasoning_case_display_config(
    case: ReasoningMatrixCase,
) -> CliStreamDisplayConfig:
    return cli_stream_display_config(
        Namespace(
            **_base_stream_args(
                quiet=case.quiet == "on",
                stats=case.statistics == "on",
                record=case.recording == "on",
                display_reasoning=case.displays_reasoning,
            )
        ),
        refresh_per_second=1000,
        interactive=case.interactive,
    )


def _reasoning_case_answer(
    case: ReasoningMatrixCase,
    label: str,
) -> str:
    if case.answer == "strict-json":
        return '{"answer":"' + label + '"}'
    return label


def _rendered_presenter_text(
    items: list[CliStreamPresenterItem],
    *,
    role: str,
) -> str:
    return "".join(
        _render_text(item.renderable)
        for item in items
        if isinstance(item, CliStreamRenderableFrame)
        and item.role == role
        and item.renderable != ""
    )


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


class ReasoningDisplayPairwiseMatrixTestCase(TestCase):
    def test_reasoning_display_matrix_is_exact_and_pairwise_complete(
        self,
    ) -> None:
        document = _reasoning_matrix_document()
        self.assertEqual(
            set(document),
            {
                "schema_version",
                "matrix_id",
                "factors",
                "critical_e2e_ids",
                "rows",
            },
        )
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(
            document["matrix_id"],
            "phase6-reasoning-cli-pairwise-v1",
        )

        factors = document["factors"]
        self.assertIsInstance(factors, dict)
        assert isinstance(factors, dict)
        self.assertEqual(
            factors,
            {
                name: list(values)
                for name, values in _REASONING_MATRIX_FACTORS.items()
            },
        )

        raw_rows = document["rows"]
        self.assertIsInstance(raw_rows, list)
        assert isinstance(raw_rows, list)
        self.assertEqual(len(raw_rows), 64)
        rows: list[dict[str, object]] = []
        expected_row_keys = {
            "id",
            *_REASONING_MATRIX_FACTORS,
            "coverage",
            "case_name",
            "entrypoint",
        }
        for index, raw_row in enumerate(raw_rows):
            with self.subTest(row=index):
                self.assertIsInstance(raw_row, dict)
                assert isinstance(raw_row, dict)
                row = cast(dict[str, object], raw_row)
                self.assertEqual(set(row), expected_row_keys)
                row_id = f"r{index:02d}"
                self.assertEqual(row["id"], row_id)
                expected_values = _expected_reasoning_matrix_values(index)
                self.assertEqual(
                    {
                        factor: row[factor]
                        for factor in _REASONING_MATRIX_FACTORS
                    },
                    expected_values,
                )
                for factor, values in _REASONING_MATRIX_FACTORS.items():
                    self.assertIn(row[factor], values)

                e2e_case = _REASONING_MATRIX_E2E_CASES.get(row_id)
                self.assertEqual(
                    row["coverage"],
                    "e2e" if e2e_case is not None else "integration",
                )
                self.assertEqual(
                    row["case_name"],
                    None if e2e_case is None else e2e_case[0],
                )
                self.assertEqual(
                    row["entrypoint"],
                    None if e2e_case is None else e2e_case[1],
                )
                rows.append(row)

        expected_ids = tuple(f"r{index:02d}" for index in range(64))
        self.assertEqual(tuple(row["id"] for row in rows), expected_ids)
        self.assertEqual(len({cast(str, row["id"]) for row in rows}), 64)
        self.assertEqual(
            len(
                {
                    tuple(row[factor] for factor in _REASONING_MATRIX_FACTORS)
                    for row in rows
                }
            ),
            64,
        )

        factor_names = tuple(_REASONING_MATRIX_FACTORS)
        for left, right in combinations(factor_names, 2):
            with self.subTest(left=left, right=right):
                expected_pairs = set(
                    product(
                        _REASONING_MATRIX_FACTORS[left],
                        _REASONING_MATRIX_FACTORS[right],
                    )
                )
                actual_pairs = {
                    (cast(str, row[left]), cast(str, row[right]))
                    for row in rows
                }
                self.assertEqual(actual_pairs, expected_pairs)

        raw_critical_ids = document["critical_e2e_ids"]
        self.assertIsInstance(raw_critical_ids, list)
        assert isinstance(raw_critical_ids, list)
        critical_ids = tuple(cast(str, value) for value in raw_critical_ids)
        self.assertEqual(critical_ids, tuple(_REASONING_MATRIX_E2E_CASES))
        e2e_rows = [row for row in rows if row["coverage"] == "e2e"]
        integration_rows = [
            row for row in rows if row["coverage"] == "integration"
        ]
        self.assertEqual(len(e2e_rows), 12)
        self.assertEqual(len(integration_rows), 52)
        self.assertEqual(
            len({cast(str, row["case_name"]) for row in e2e_rows}),
            12,
        )
        self.assertEqual(
            [row["entrypoint"] for row in e2e_rows].count("model"),
            6,
        )
        self.assertEqual(
            [row["entrypoint"] for row in e2e_rows].count("agent"),
            6,
        )
        self.assertFalse(
            any("applicable" in key for row in rows for key in row)
        )


class ReasoningDisplayIntegrationMatrixTestCase(IsolatedAsyncioTestCase):
    async def test_reasoning_reducer_presenter_integration_rows(self) -> None:
        cases = tuple(
            case
            for case in _reasoning_matrix_cases()
            if case.coverage == "integration"
        )
        self.assertEqual(len(cases), 52)

        for case in cases:
            with self.subTest(row=case.id):
                await self._assert_integration_case(case)

    async def _assert_integration_case(
        self,
        case: ReasoningMatrixCase,
    ) -> None:
        config = _reasoning_case_display_config(case)
        response = MatrixProjectionResponse(
            _reasoning_case_answer(case, "integration matrix answer"),
            representation=case.representation,
        )
        reducer = CliStreamSnapshotReducer(config)
        presenter = _theme(case.theme).stream_presenter(getLogger(__name__))
        context = CliStreamPresenterContext(
            model_id="matrix-model",
            console_width=100,
            input_token_count=1,
        )
        presented: list[CliStreamPresenterItem] = []

        async def consume() -> None:
            for canonical_item in response._items(
                "matrix-integration-stream",
                "matrix-integration-run",
                "matrix-integration-turn",
            ):
                projection = project_canonical_stream_item(canonical_item)
                if not reducer.apply_projection(projection):
                    continue
                request = CliStreamPresenterRequest(
                    snapshot=reducer.snapshot(),
                    display_config=config,
                    context=context,
                    mode=(
                        "live"
                        if config.diagnostic_channel != "none"
                        else "answer"
                    ),
                )
                presented.extend(
                    [item async for item in presenter.present(request)]
                )

        visible_reasoning = case.shows_reasoning and case.has_reasoning
        if case.theme == "fancy" and not visible_reasoning:
            with patch.object(
                presenter,
                "_reasoning_panel",
                side_effect=AssertionError(
                    "hidden Fancy reasoning must not build panel content"
                ),
            ):
                await consume()
        else:
            await consume()

        snapshot = reducer.snapshot()
        self.assertEqual(config.show_reasoning, case.shows_reasoning)
        reasoning_frames = [
            item
            for item in presented
            if isinstance(item, CliStreamRenderableFrame)
            and item.role == "reasoning"
            and item.renderable != ""
        ]
        reasoning_output = _rendered_presenter_text(
            presented,
            role="reasoning",
        )

        if not visible_reasoning:
            self.assertEqual(reasoning_frames, [])
            self.assertEqual(snapshot.reasoning_text, "")
            self.assertEqual(snapshot.reasoning_segments, ())
            self.assertEqual(
                snapshot.build_stats.retained_reasoning_segments,
                0,
            )
            self.assertEqual(
                snapshot.build_stats.retained_reasoning_characters,
                0,
            )
            self.assertEqual(
                snapshot.build_stats.retained_reasoning_utf8_bytes,
                0,
            )
            self.assertEqual(
                snapshot.build_stats.reasoning_materializations,
                0,
            )
            for sentinel in _REASONING_SENTINELS:
                self.assertNotIn(sentinel, reasoning_output)
            return

        self.assertTrue(reasoning_frames)
        expected_parts = _matrix_reasoning_parts(case.representation)
        self.assertEqual(
            tuple(
                segment.representation
                for segment in snapshot.reasoning_segments
            ),
            tuple(representation for representation, _ in expected_parts),
        )
        for representation, sentinel in expected_parts:
            self.assertIn(sentinel, snapshot.reasoning_text)
            self.assertIn(sentinel, reasoning_output)
            label = (
                "Reasoning"
                if representation is StreamReasoningRepresentation.NATIVE_TEXT
                else "Reasoning summary"
            )
            if case.theme == "basic" and case.interactive:
                self.assertIn("💭", reasoning_output)
                self.assertNotIn(label, reasoning_output)
            else:
                self.assertIn(label, reasoning_output)

        if case.representation == "native":
            self.assertNotIn(_SUMMARY_REASONING_SENTINEL, reasoning_output)
            self.assertNotIn("Reasoning summary", reasoning_output)
        elif case.representation == "summary":
            self.assertNotIn(_NATIVE_REASONING_SENTINEL, reasoning_output)
        else:
            self.assertLess(
                reasoning_output.index(_NATIVE_REASONING_SENTINEL),
                reasoning_output.index(_SUMMARY_REASONING_SENTINEL),
            )
            if case.theme != "basic" or not case.interactive:
                self.assertLess(
                    reasoning_output.index("Reasoning"),
                    reasoning_output.index("Reasoning summary"),
                )

        if case.theme == "basic" and not case.interactive:
            for _, sentinel in expected_parts:
                self.assertEqual(reasoning_output.count(sentinel), 1)


class ThemeMatrixE2ETestCase(IsolatedAsyncioTestCase):
    async def test_reasoning_critical_public_boundary_rows(self) -> None:
        cases = tuple(
            case
            for case in _reasoning_matrix_cases()
            if case.coverage == "e2e"
        )
        self.assertEqual(len(cases), 12)

        for case in cases:
            with self.subTest(row=case.id, name=case.case_name):
                await self._assert_reasoning_public_boundary_case(case)

    async def test_agent_structured_json_suppresses_theme_prefixes(
        self,
    ) -> None:
        for theme_name in ("basic", "fancy"):
            for terminal in ("interactive", "non-interactive"):
                case = ReasoningMatrixCase(
                    id=f"structured-{theme_name}-{terminal}",
                    theme=theme_name,
                    terminal=terminal,
                    representation="summary",
                    display="display-reasoning",
                    statistics="off",
                    quiet="off",
                    recording="off",
                    answer="strict-json",
                    coverage="direct",
                    case_name="agent-structured-json-no-prefix",
                    entrypoint="agent",
                )
                with self.subTest(theme=theme_name, terminal=terminal):
                    await self._assert_reasoning_public_boundary_case(case)

    async def _assert_reasoning_public_boundary_case(
        self,
        case: ReasoningMatrixCase,
    ) -> None:
        assert case.case_name is not None
        assert case.entrypoint is not None
        args_case = MatrixFlagCase(
            case.case_name,
            {
                "quiet": case.quiet == "on",
                "stats": case.statistics == "on",
                "record": case.recording == "on",
                "display_reasoning": case.displays_reasoning,
            },
        )
        display_config = _reasoning_case_display_config(case)
        console = _console(interactive=case.interactive)
        hub = MagicMock()
        hub.can_access.return_value = True
        hub.model.side_effect = _model_summary
        answer_label = f"{case.entrypoint} matrix answer"
        answer = _reasoning_case_answer(case, answer_label)

        if case.entrypoint == "model":
            capture = await self._run_reasoning_model_case(
                _model_args(args_case),
                console,
                _theme(case.theme),
                hub,
                answer=answer,
                representation=case.representation,
            )
        else:
            capture = await self._run_reasoning_agent_case(
                _agent_args(args_case),
                console,
                _theme(case.theme),
                hub,
                answer=answer,
                representation=case.representation,
                structured_output=case.answer == "strict-json",
            )

        for sentinel in _REASONING_SENTINELS:
            self.assertNotIn(sentinel, capture.stdout)
        if case.answer == "strict-json":
            self.assertEqual(loads(capture.stdout), {"answer": answer_label})
            self.assertNotIn("\x1b", capture.stdout)
        else:
            expected_prefix = (
                ":robot: "
                if case.entrypoint == "agent"
                and case.theme == "basic"
                and case.interactive
                else ""
            )
            self.assertEqual(capture.stdout, expected_prefix + answer + "\n")

        visible_reasoning = case.shows_reasoning and case.has_reasoning
        reasoning_surface = (
            capture.live if case.interactive else capture.stderr
        )
        other_surface = capture.stderr if case.interactive else capture.live
        expected_parts = _matrix_reasoning_parts(case.representation)
        if visible_reasoning:
            self.assertTrue(reasoning_surface)
            for representation, sentinel in expected_parts:
                self.assertIn(sentinel, reasoning_surface)
                label = (
                    "Reasoning"
                    if representation
                    is StreamReasoningRepresentation.NATIVE_TEXT
                    else "Reasoning summary"
                )
                if case.theme == "basic" and case.interactive:
                    self.assertIn("💭", reasoning_surface)
                    self.assertNotIn(label, reasoning_surface)
                else:
                    self.assertIn(label, reasoning_surface)
            if case.representation == "mixed":
                self.assertLess(
                    reasoning_surface.index(_NATIVE_REASONING_SENTINEL),
                    reasoning_surface.index(_SUMMARY_REASONING_SENTINEL),
                )
                if case.theme != "basic" or not case.interactive:
                    self.assertLess(
                        reasoning_surface.index("Reasoning"),
                        reasoning_surface.index("Reasoning summary"),
                    )
            if case.theme == "basic" and not case.interactive:
                for _, sentinel in expected_parts:
                    self.assertEqual(reasoning_surface.count(sentinel), 1)
        else:
            for sentinel in _REASONING_SENTINELS:
                self.assertNotIn(sentinel, reasoning_surface)
        for sentinel in _REASONING_SENTINELS:
            self.assertNotIn(sentinel, other_surface)

        diagnostics_have_content = visible_reasoning or case.statistics == "on"
        self.assertEqual(
            capture.live_updates > 0,
            display_config.live_enabled and diagnostics_have_content,
        )
        self.assertEqual(
            capture.diagnostic_console_created,
            display_config.diagnostic_channel == "stderr"
            and diagnostics_have_content,
        )
        self.assertEqual(
            capture.recordings > 0,
            display_config.record_enabled,
        )
        if display_config.record_enabled:
            for _, sentinel in expected_parts:
                self.assertIn(sentinel, capture.live)

    async def _run_reasoning_model_case(
        self,
        args: Namespace,
        console: MagicMock,
        theme: Theme,
        hub: MagicMock,
        *,
        answer: str,
        representation: str,
    ) -> MatrixCommandCapture:
        class CaseModelManager(MatrixModelManager):
            async def __call__(
                self,
                task: object,
            ) -> MatrixProjectionResponse:
                _ = task
                return MatrixProjectionResponse(
                    answer,
                    representation=representation,
                )

        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        diagnostic_console = MagicMock()

        def input_after_header(*_args: object, **_kwargs: object) -> str:
            console.print.reset_mock()
            return "prompt"

        with (
            patch.object(model_cmds, "ModelManager", CaseModelManager),
            patch.object(
                model_cmds,
                "get_model_settings",
                return_value={
                    "engine_uri": MatrixModelManager.engine_uri,
                    "modality": Modality.TEXT_GENERATION,
                },
            ),
            patch.object(
                model_cmds, "get_input", side_effect=input_after_header
            ),
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
                1000,
                getLogger(__name__),
            )

        return MatrixCommandCapture(
            stdout=_printed_text(console),
            stderr=_rendered_mock_text(diagnostic_console.print),
            live=_rendered_mock_text(live.update),
            live_updates=live.update.call_count,
            recordings=console.save_svg.call_count,
            diagnostic_console_created=diagnostic_console_cls.called,
        )

    async def _run_reasoning_agent_case(
        self,
        args: Namespace,
        console: MagicMock,
        theme: Theme,
        hub: MagicMock,
        *,
        answer: str,
        representation: str,
        structured_output: bool,
    ) -> MatrixCommandCapture:
        live = MagicMock()
        live.__enter__.return_value = live
        live.__exit__.return_value = False
        diagnostic_console = MagicMock()
        orchestrator = MatrixOrchestrator(
            answer=answer,
            representation=representation,
            structured_output=structured_output,
        )

        def input_after_header(*_args: object, **_kwargs: object) -> str:
            console.print.reset_mock()
            return "prompt"

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
            patch.object(
                agent_cmds, "get_input", side_effect=input_after_header
            ),
            patch.object(
                agent_cmds,
                "OrchestratorResponse",
                MatrixProjectionResponse,
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
                1000,
            )

        self.assertEqual(orchestrator.calls, ["prompt"])
        return MatrixCommandCapture(
            stdout=_printed_text(console),
            stderr=_rendered_mock_text(diagnostic_console.print),
            live=_rendered_mock_text(live.update),
            live_updates=live.update.call_count,
            recordings=console.save_svg.call_count,
            diagnostic_console_created=diagnostic_console_cls.called,
        )

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
