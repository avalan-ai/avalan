from collections.abc import AsyncIterator
from dataclasses import FrozenInstanceError, replace
from logging import getLogger
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.display_reducer import (
    CliStreamSnapshotReducer,
    iter_cli_canonical_stream_snapshots,
)
from avalan.cli.display_snapshot import (
    CliDisplayTokenCandidateSnapshot,
    CliDisplayTokenSnapshot,
    CliStreamSnapshot,
    CliStreamSnapshotBuilder,
)
from avalan.cli.stream_presenter import (
    CliStreamAnswerPresenter,
    CliStreamAnswerTextChunk,
    CliStreamPresenterContext,
    CliStreamPresenterRequest,
    CliStreamRenderableFrame,
    LegacyThemeStreamPresenter,
)
from avalan.cli.theme.fancy import FancyTheme
from avalan.entities import Token, TokenDetail
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    canonical_item_from_consumer_projection,
)


def _config(**overrides: object) -> CliStreamDisplayConfig:
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
        "display_tokens": 2,
        "display_pause": 0,
        "display_probabilities": True,
        "display_probabilities_maximum": 0.8,
        "display_probabilities_sample_minimum": 0.1,
        "display_time_to_n_token": 2,
        "display_reasoning_time": True,
    }
    values.update(overrides)
    return CliStreamDisplayConfig(**values)


def _context(**overrides: object) -> CliStreamPresenterContext:
    values = {
        "model_id": "model",
        "console_width": 80,
        "input_token_count": 3,
        "tokenizer_tokens": ("a", "b"),
        "tokenizer_special_tokens": ("<eos>",),
        "token_probability_pick": 2,
        "start_thinking": True,
    }
    values.update(overrides)
    return CliStreamPresenterContext(**values)


def _snapshot(
    config: CliStreamDisplayConfig,
    *,
    answer: str = "",
    reasoning: str = "",
    tool_call_request: str = "",
    display_token: Token | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> CliStreamSnapshot:
    builder = CliStreamSnapshotBuilder(config)
    builder.append_answer_text(answer, tokens=1 if answer else 0)
    builder.append_reasoning_text(reasoning, tokens=1 if reasoning else 0)
    builder.append_tool_call_request_text(
        tool_call_request, tokens=1 if tool_call_request else 0
    )
    builder.update_token_counts(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    builder.update_timing(
        started_at=1.0,
        updated_at=2.0,
        first_token_seconds=0.2,
        reasoning_seconds=0.4,
        time_to_n_token_seconds=0.6,
        elapsed_seconds=1.5,
    )
    if display_token is not None:
        builder.add_display_token(display_token, sequence=7)
    return builder.snapshot()


def _request(
    config: CliStreamDisplayConfig,
    snapshot: CliStreamSnapshot,
    *,
    mode: str = "live",
    context: CliStreamPresenterContext | None = None,
) -> CliStreamPresenterRequest:
    return CliStreamPresenterRequest(
        snapshot=snapshot,
        display_config=config,
        context=_context() if context is None else context,
        mode=mode,  # type: ignore[arg-type]
    )


async def _collect(
    presenter: object,
    request: CliStreamPresenterRequest,
) -> list[object]:
    return [item async for item in presenter.present(request)]  # type: ignore[attr-defined]


async def _frame_stream(
    *frames: tuple[Token | None, str],
) -> AsyncIterator[tuple[Token | None, str]]:
    for frame in frames:
        yield frame


async def _canonical_items(
    *projections: StreamConsumerProjection,
) -> AsyncIterator[CanonicalStreamItem]:
    for projection in projections:
        yield canonical_item_from_consumer_projection(projection)


class RecordingTheme:
    def __init__(
        self,
        *frames: tuple[Token | None, str],
        awaitable: bool = False,
    ) -> None:
        self.awaitable = awaitable
        self.calls: list[dict[str, object]] = []
        self.frames = frames

    def tokens(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        stream = _frame_stream(*self.frames)
        if self.awaitable:
            return self._await_stream(stream)
        return stream

    async def _await_stream(
        self,
        stream: AsyncIterator[tuple[Token | None, str]],
    ) -> AsyncIterator[tuple[Token | None, str]]:
        return stream


class ClosableFrameStream:
    def __init__(self) -> None:
        self.closed = False
        self.yielded = False

    def __aiter__(self) -> "ClosableFrameStream":
        return self

    async def __anext__(self) -> tuple[Token | None, str]:
        if self.yielded:
            raise StopAsyncIteration
        self.yielded = True
        return Token(id=1, token="x", probability=0.7), "frame"

    async def aclose(self) -> None:
        self.closed = True


class FailingClosableFrameStream:
    def __init__(self) -> None:
        self.closed = False
        self.yielded = False

    def __aiter__(self) -> "FailingClosableFrameStream":
        return self

    async def __anext__(self) -> tuple[Token | None, str]:
        if self.yielded:
            raise RuntimeError("theme stream failed")
        self.yielded = True
        return Token(id=1, token="x", probability=0.7), "frame"

    async def aclose(self) -> None:
        self.closed = True


class ClosableTheme:
    def __init__(self, stream: ClosableFrameStream) -> None:
        self.stream = stream

    def tokens(self, **_: object) -> ClosableFrameStream:
        return self.stream


class FailingClosableTheme:
    def __init__(self, stream: FailingClosableFrameStream) -> None:
        self.stream = stream

    def tokens(self, **_: object) -> FailingClosableFrameStream:
        return self.stream


class InvalidTheme:
    pass


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self.calls = 0

    def __call__(self) -> float:
        value = self._values[self.calls]
        self.calls += 1
        return value


class StreamPresenterContractTestCase(TestCase):
    def test_contract_dataclasses_validate_and_freeze(self) -> None:
        config = _config()
        snapshot = _snapshot(config, answer="hi")
        context = _context()
        request = _request(config, snapshot, context=context)
        display_token = CliDisplayTokenSnapshot(
            sequence=1,
            token_id=2,
            text="x",
        )
        frame = CliStreamRenderableFrame(
            renderable="frame",
            role="stream",
            current_token=display_token,
        )
        chunk = CliStreamAnswerTextChunk(text="hi")

        self.assertEqual(request.mode, "live")
        self.assertEqual(frame.current_token, display_token)
        self.assertEqual(chunk.role, "answer")
        with self.assertRaises(FrozenInstanceError):
            context.model_id = "other"  # type: ignore[misc]
        with self.assertRaises(AssertionError):
            CliStreamPresenterContext(model_id="", console_width=80)
        with self.assertRaises(AssertionError):
            CliStreamPresenterContext(
                model_id="model",
                console_width=80,
                tokenizer_tokens="abc",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            CliStreamPresenterContext(
                model_id="model",
                console_width=80,
                tokenizer_special_tokens="abc",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            CliStreamPresenterRequest(
                snapshot=snapshot,
                display_config=config,
                context=context,
                mode="other",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            CliStreamRenderableFrame(
                renderable="frame",
                role="invalid",  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            CliStreamRenderableFrame(
                renderable="frame",
                current_token=object(),  # type: ignore[arg-type]
            )
        with self.assertRaises(AssertionError):
            CliStreamAnswerTextChunk(text="")
        with self.assertRaises(AssertionError):
            CliStreamAnswerTextChunk(text="x", role="stream")

    def test_legacy_adapter_requires_theme_token_factory(self) -> None:
        with self.assertRaises(AssertionError):
            LegacyThemeStreamPresenter(InvalidTheme(), getLogger(__name__))

    def test_context_rejects_list_and_dict_token_collections(self) -> None:
        invalid_values: tuple[object, ...] = (["a"], {"a": "b"})

        for invalid_value in invalid_values:
            with self.subTest(
                field="tokenizer_tokens",
                value_type=type(invalid_value).__name__,
            ):
                with self.assertRaises(AssertionError):
                    CliStreamPresenterContext(
                        model_id="model",
                        console_width=80,
                        tokenizer_tokens=invalid_value,  # type: ignore[arg-type]
                    )
            with self.subTest(
                field="tokenizer_special_tokens",
                value_type=type(invalid_value).__name__,
            ):
                with self.assertRaises(AssertionError):
                    CliStreamPresenterContext(
                        model_id="model",
                        console_width=80,
                        tokenizer_special_tokens=invalid_value,  # type: ignore[arg-type]
                    )


class StreamAnswerPresenterTestCase(IsolatedAsyncioTestCase):
    async def test_answer_presenter_emits_only_monotonic_suffixes(
        self,
    ) -> None:
        config = _config()
        presenter = CliStreamAnswerPresenter()

        items = await _collect(
            presenter,
            _request(config, _snapshot(config), mode="answer"),
        )
        self.assertEqual(items, [])

        first = await _collect(
            presenter,
            _request(config, _snapshot(config, answer="He"), mode="answer"),
        )
        second = await _collect(
            presenter,
            _request(config, _snapshot(config, answer="Hello"), mode="answer"),
        )
        repeated = await _collect(
            presenter,
            _request(config, _snapshot(config, answer="Hello"), mode="answer"),
        )

        self.assertEqual(
            [
                item.text
                for item in first
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["He"],
        )
        self.assertEqual(
            [
                item.text
                for item in second
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["llo"],
        )
        self.assertEqual(repeated, [])

        with self.assertRaisesRegex(AssertionError, "monotonically"):
            await _collect(
                presenter,
                _request(
                    config, _snapshot(config, answer="Hi"), mode="answer"
                ),
            )

        presenter.reset()
        reset_items = await _collect(
            presenter,
            _request(config, _snapshot(config, answer="Hi"), mode="answer"),
        )
        self.assertEqual(
            [
                item.text
                for item in reset_items
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["Hi"],
        )

    async def test_answer_presenter_rejects_live_mode(self) -> None:
        config = _config()
        presenter = CliStreamAnswerPresenter()

        with self.assertRaisesRegex(AssertionError, "answer mode"):
            await _collect(
                presenter,
                _request(config, _snapshot(config, answer="hi")),
            )

    async def test_answer_presenter_ignores_reasoning_and_tool_only_snapshots(
        self,
    ) -> None:
        config = _config()
        presenter = CliStreamAnswerPresenter()

        items = await _collect(
            presenter,
            _request(
                config,
                _snapshot(
                    config,
                    reasoning="think",
                    tool_call_request='{"x": 1}',
                ),
                mode="answer",
            ),
        )

        self.assertEqual(items, [])


class LegacyThemeStreamPresenterTestCase(IsolatedAsyncioTestCase):
    async def test_answer_mode_and_quiet_live_mode_never_call_theme(
        self,
    ) -> None:
        config = _config()
        quiet_config = _config(quiet=True, interactive=True, stats=False)
        theme = RecordingTheme((None, "unused"))
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        answer_items = await _collect(
            presenter,
            _request(config, _snapshot(config, answer="hi"), mode="answer"),
        )
        quiet_items = await _collect(
            presenter,
            _request(
                quiet_config,
                _snapshot(quiet_config, answer="hi there"),
                mode="live",
            ),
        )

        self.assertEqual(
            [
                item.text
                for item in answer_items
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["hi"],
        )
        self.assertEqual(
            [
                item.text
                for item in quiet_items
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            [" there"],
        )
        self.assertEqual(theme.calls, [])

        presenter.reset()
        reset_items = await _collect(
            presenter,
            _request(
                quiet_config,
                _snapshot(quiet_config, answer="hi"),
                mode="live",
            ),
        )
        self.assertEqual(
            [
                item.text
                for item in reset_items
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["hi"],
        )

    async def test_tools_events_only_live_mode_emits_diagnostics_and_answer(
        self,
    ) -> None:
        config = _config(
            stats=False,
            display_tools=True,
            display_events=True,
            display_tokens=0,
            display_probabilities=False,
        )
        builder = CliStreamSnapshotBuilder(config)
        builder.append_answer_text("plain")
        builder.add_active_tool(
            tool_call_id="call-1",
            name="calc",
            arguments={"x": 1},
        )
        builder.add_event_summary(
            event_type="flow_node_started",
            payload={"node": "math"},
        )
        theme = RecordingTheme((Token(id=9, token="unused"), "unused"))
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        items = await _collect(
            presenter,
            _request(config, builder.snapshot(), mode="live"),
        )

        self.assertEqual(theme.calls, [])
        self.assertEqual(
            [
                item.role
                for item in items
                if isinstance(item, CliStreamRenderableFrame)
            ],
            ["tools", "events"],
        )
        self.assertEqual(
            [
                item.text
                for item in items
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["plain"],
        )

    async def test_live_adapter_maps_snapshot_and_config_to_theme_call(
        self,
    ) -> None:
        config = _config(answer_height_expand=True)
        display_token = TokenDetail(
            id=1,
            token="answer",
            probability=0.9,
            step=4,
            probability_distribution="softmax",
            tokens=[Token(id=2, token="alt", probability=0.2)],
        )
        output_token = TokenDetail(
            id=3,
            token="out",
            probability=0.7,
            step=5,
            probability_distribution="sparsemax",
            tokens=[Token(id=4, token="candidate", probability=0.4)],
        )
        object_id_token = Token(
            id=object(),  # type: ignore[arg-type]
            token="object-id",
            probability=None,
        )
        snapshot = _snapshot(
            config,
            answer="Hello",
            reasoning="think",
            tool_call_request='{"x": 1}',
            display_token=display_token,
            input_tokens=9,
        )
        theme = RecordingTheme(
            (None, "empty-frame"),
            (output_token, "frame"),
            (object_id_token, "object-id-frame"),
            awaitable=True,
        )
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        items = await _collect(
            presenter,
            _request(config, snapshot),
        )

        self.assertEqual(len(items), 3)
        empty_frame = items[0]
        self.assertIsInstance(empty_frame, CliStreamRenderableFrame)
        self.assertIsNone(empty_frame.current_token)
        frame = items[1]
        self.assertIsInstance(frame, CliStreamRenderableFrame)
        self.assertEqual(frame.renderable, "frame")
        self.assertEqual(frame.role, "stream")
        self.assertEqual(frame.current_token.text, "out")  # type: ignore[union-attr]
        self.assertEqual(frame.current_token.candidates[0].text, "candidate")  # type: ignore[union-attr]
        object_id_frame = items[2]
        self.assertIsInstance(object_id_frame, CliStreamRenderableFrame)
        self.assertIsInstance(object_id_frame.current_token.token_id, str)  # type: ignore[union-attr]

        call = theme.calls[0]
        self.assertEqual(call["model_id"], "model")
        self.assertEqual(call["added_tokens"], ["a", "b"])
        self.assertEqual(call["special_tokens"], ["<eos>"])
        self.assertEqual(call["display_token_size"], 2)
        self.assertIs(call["display_probabilities"], True)
        self.assertEqual(call["pick"], 2)
        self.assertEqual(call["thinking_text_tokens"], ["think"])
        self.assertEqual(call["tool_text_tokens"], ['{"x": 1}'])
        self.assertEqual(call["answer_text_tokens"], ["Hello"])
        self.assertEqual(call["input_token_count"], 9)
        self.assertEqual(call["total_tokens"], 2)
        self.assertEqual(call["tool_events"], [])
        self.assertEqual(call["tool_event_calls"], [])
        self.assertEqual(call["tool_event_results"], [])
        self.assertIsNone(call["tool_running_spinner"])
        self.assertEqual(call["ttft"], 0.2)
        self.assertEqual(call["ttnt"], 0.6)
        self.assertEqual(call["ttsr"], 0.4)
        self.assertEqual(call["elapsed"], 1.5)
        self.assertEqual(call["console_width"], 80)
        self.assertEqual(call["tool_token_count"], 1)
        self.assertEqual(call["maximum_frames"], 1)
        self.assertIsNone(call["logits_count"])
        self.assertEqual(call["tool_events_limit"], 2)
        self.assertEqual(call["height"], 12)
        self.assertIs(call["limit_answer_height"], False)
        self.assertIs(call["start_thinking"], True)

        tokens = call["tokens"]
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(tokens[0], TokenDetail)
        self.assertEqual(tokens[0].token, "answer")
        self.assertEqual(tokens[0].tokens[0].token, "alt")

        predicate = call["focus_on_token_when"]
        self.assertTrue(predicate(Token(id=9, token="low", probability=0.1)))
        self.assertFalse(predicate(Token(id=9, token="high", probability=0.9)))
        self.assertTrue(
            predicate(
                TokenDetail(
                    id=9,
                    token="sampled",
                    probability=0.9,
                    tokens=[Token(id=10, token="candidate", probability=0.2)],
                )
            )
        )
        self.assertFalse(
            predicate(
                TokenDetail(
                    id=9,
                    token="sampled",
                    probability=0.9,
                    tokens=[Token(id=9, token="same", probability=0.2)],
                )
            )
        )

    async def test_live_adapter_hides_disabled_timing_fields(self) -> None:
        config = _config(
            display_reasoning_time=False,
            display_time_to_n_token=None,
        )
        theme = RecordingTheme()
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        await _collect(
            presenter,
            _request(
                config,
                _snapshot(config, answer="Hello", reasoning="think"),
            ),
        )

        self.assertIsNone(theme.calls[0]["ttnt"])
        self.assertIsNone(theme.calls[0]["ttsr"])

    async def test_live_adapter_preserves_snapshot_string_token_ids(
        self,
    ) -> None:
        config = _config()
        token_snapshot = CliDisplayTokenSnapshot(
            sequence=1,
            token_id="main-token",
            text="answer",
            probability=0.9,
            step=1,
            probability_distribution="softmax",
            candidates=(
                CliDisplayTokenCandidateSnapshot(
                    token_id="alt-token",
                    text="candidate",
                    probability=0.2,
                ),
            ),
        )
        snapshot = replace(
            _snapshot(config),
            display_tokens=(token_snapshot,),
        )
        theme = RecordingTheme()
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        await _collect(presenter, _request(config, snapshot))

        tokens = theme.calls[0]["tokens"]
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(tokens[0], TokenDetail)
        self.assertEqual(tokens[0].id, "main-token")
        self.assertEqual(tokens[0].tokens[0].id, "alt-token")

    async def test_live_adapter_uses_output_usage_for_total_tokens(
        self,
    ) -> None:
        config = _config()
        theme = RecordingTheme()
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        items = await _collect(
            presenter,
            _request(
                config,
                _snapshot(
                    config,
                    answer="hello",
                    reasoning="think",
                    tool_call_request='{"x": 1}',
                    output_tokens=8,
                ),
            ),
        )

        self.assertEqual(items, [])
        self.assertEqual(theme.calls[0]["total_tokens"], 8)
        self.assertEqual(theme.calls[0]["tool_token_count"], 1)

    async def test_live_adapter_disables_probabilities_when_pick_is_zero(
        self,
    ) -> None:
        config = _config()
        theme = RecordingTheme()
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        await _collect(
            presenter,
            _request(
                config,
                _snapshot(
                    config,
                    display_token=Token(id=1, token="x", probability=0.1),
                ),
                context=_context(token_probability_pick=0),
            ),
        )

        self.assertIs(theme.calls[0]["display_probabilities"], False)
        self.assertIsNone(theme.calls[0]["focus_on_token_when"])

    async def test_live_adapter_emits_snapshot_diagnostic_frames(self) -> None:
        config = _config()
        builder = CliStreamSnapshotBuilder(config)
        builder.add_active_tool(
            tool_call_id="call-1",
            name="calc",
            arguments={"x": 1},
            sequence=1,
        )
        builder.add_tool_result_summary(
            tool_call_id="call-1",
            name="calc",
            status="result",
            result={"value": 2},
            arguments_count=1,
            sequence=2,
        )
        builder.add_event_summary(
            event_type="flow_node_started",
            payload={"node": "math"},
            sequence=3,
        )
        builder.add_usage_summary(
            {"input_tokens": 1, "output_tokens": 2},
            sequence=4,
            kind="usage.delta",
        )
        builder.add_projection_summary(
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=5,
                kind=StreamItemKind.FLOW_EVENT,
                channel=StreamChannel.FLOW,
                correlation=StreamItemCorrelation(),
                data={"provider": "fixture"},
                metadata={"trace": "visible"},
            )
        )
        snapshot = builder.snapshot()
        presenter = LegacyThemeStreamPresenter(
            RecordingTheme(), getLogger(__name__)
        )

        items = await _collect(presenter, _request(config, snapshot))
        frames = [
            item
            for item in items
            if isinstance(item, CliStreamRenderableFrame)
        ]

        self.assertEqual(
            [frame.role for frame in frames],
            ["tools", "events", "stats"],
        )
        self.assertIn("active tool calc", str(frames[0].renderable))
        self.assertIn("event flow_node_started", str(frames[1].renderable))
        self.assertIn("usage usage.delta", str(frames[2].renderable))
        self.assertIn("projection flow.event", str(frames[2].renderable))

    async def test_live_adapter_renders_real_fancy_theme_frame(self) -> None:
        config = _config()
        theme = FancyTheme(lambda message: message, lambda s, p, n: s)
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        items = await _collect(
            presenter,
            _request(
                config,
                _snapshot(
                    config,
                    answer="hello",
                    display_token=Token(id=1, token="hello"),
                    input_tokens=1,
                ),
            ),
        )

        self.assertTrue(
            any(
                isinstance(item, CliStreamRenderableFrame)
                and item.role == "stream"
                for item in items
            )
        )

    async def test_live_adapter_uses_context_input_count_when_missing(
        self,
    ) -> None:
        config = _config(display_probabilities=False)
        theme = RecordingTheme()
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        items = await _collect(
            presenter,
            _request(
                config,
                _snapshot(config, display_token=Token(id=5, token="x")),
                context=_context(
                    input_token_count=11,
                    tokenizer_tokens=None,
                    tokenizer_special_tokens=None,
                ),
            ),
        )

        self.assertEqual(items, [])
        call = theme.calls[0]
        self.assertEqual(call["input_token_count"], 11)
        self.assertIsNone(call["added_tokens"])
        self.assertIsNone(call["special_tokens"])
        self.assertIsNone(call["focus_on_token_when"])
        self.assertEqual(call["tokens"][0].id, 5)

    async def test_live_adapter_rejects_invalid_theme_stream_result(
        self,
    ) -> None:
        config = _config()
        theme = RecordingTheme()
        theme.tokens = lambda **_: object()  # type: ignore[method-assign]
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        with self.assertRaises(AssertionError):
            await _collect(
                presenter,
                _request(config, _snapshot(config, answer="hi")),
            )

    async def test_live_adapter_closes_theme_frame_stream(self) -> None:
        config = _config()
        stream = ClosableFrameStream()
        presenter = LegacyThemeStreamPresenter(
            ClosableTheme(stream), getLogger(__name__)
        )
        iterator = presenter.present(
            _request(config, _snapshot(config, answer="hi"))
        )

        item = await anext(iterator)
        await iterator.aclose()

        self.assertIsInstance(item, CliStreamRenderableFrame)
        self.assertTrue(stream.closed)

    async def test_live_adapter_closes_theme_frame_stream_after_error(
        self,
    ) -> None:
        config = _config()
        stream = FailingClosableFrameStream()
        presenter = LegacyThemeStreamPresenter(
            FailingClosableTheme(stream), getLogger(__name__)
        )

        with self.assertRaisesRegex(RuntimeError, "theme stream failed"):
            await _collect(
                presenter,
                _request(config, _snapshot(config, answer="hi")),
            )

        self.assertTrue(stream.yielded)
        self.assertTrue(stream.closed)

    async def test_reducer_snapshot_feeds_answer_presenter(self) -> None:
        config = _config()
        projections = (
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
                correlation=StreamItemCorrelation(),
            ),
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
                text_delta="OK",
            ),
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
            ),
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                correlation=StreamItemCorrelation(),
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
                usage={"total_tokens": 2},
            ),
        )
        snapshots = [
            snapshot
            async for snapshot in iter_cli_canonical_stream_snapshots(
                _canonical_items(*projections),
                config,
            )
        ]
        presenter = CliStreamAnswerPresenter()

        items = await _collect(
            presenter,
            _request(config, snapshots[-1], mode="answer"),
        )

        self.assertEqual(
            [
                item.text
                for item in items
                if isinstance(item, CliStreamAnswerTextChunk)
            ],
            ["OK"],
        )

    async def test_reducer_in_progress_snapshot_feeds_elapsed_to_adapter(
        self,
    ) -> None:
        config = _config()
        projections = (
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
                correlation=StreamItemCorrelation(),
            ),
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                correlation=StreamItemCorrelation(),
                text_delta="OK",
            ),
        )
        reducer = CliStreamSnapshotReducer(config, clock=FakeClock(10.0, 12.5))
        reducer.reduce_projection(projections[0])
        snapshot = reducer.reduce_projection(projections[1])
        theme = RecordingTheme()
        presenter = LegacyThemeStreamPresenter(theme, getLogger(__name__))

        await _collect(presenter, _request(config, snapshot))

        self.assertIsNone(snapshot.timing.elapsed_seconds)
        self.assertEqual(theme.calls[0]["elapsed"], 2.5)
