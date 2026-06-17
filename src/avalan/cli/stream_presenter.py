"""Define CLI stream presenter contracts."""

from ..event import EventStats
from ..model.stream import StreamChannel, StreamItemKind
from .display import CliStreamDisplayConfig
from .display_snapshot import (
    CliDisplayTokenCandidateSnapshot,
    CliDisplayTokenSnapshot,
    CliProjectionMetadataSummarySnapshot,
    CliStreamSnapshot,
)

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from logging import Logger
from typing import Any, Literal, Protocol, TypeAlias, cast

from rich.console import RenderableType

StreamPresenterMode = Literal["live", "answer"]
StreamFrameRole = Literal["stream", "events", "tools", "stats", "answer"]
TokenFrameStream: TypeAlias = AsyncIterator[
    tuple["_ThemeTokenRenderDisplayToken | None", RenderableType]
]
TokenFrameStreamFactory: TypeAlias = Callable[
    ..., object
]


@dataclass(frozen=True, kw_only=True, slots=True)
class _ThemeTokenRenderDisplayTokenCandidate:
    token: str
    id: int | str | None = None
    probability: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class _ThemeTokenRenderDisplayToken:
    sequence: int
    kind: StreamItemKind
    channel: StreamChannel
    token: str
    id: int | str | None = None
    probability: float | None = None
    step: int | None = None
    probability_distribution: str | None = None
    tokens: tuple[_ThemeTokenRenderDisplayTokenCandidate, ...] = ()


@dataclass(frozen=True, kw_only=True, slots=True)
class _ThemeTokenRenderState:
    model_id: str
    projection_sequence: int | None = None
    projection_kind: StreamItemKind | None = None
    projection_channel: StreamChannel | None = None
    added_tokens: tuple[str, ...] | None = None
    special_tokens: tuple[str, ...] | None = None
    display_token_size: int | None = None
    display_probabilities: bool = False
    pick: int = 0
    focus_on_token_when: Callable[
        [_ThemeTokenRenderDisplayToken], bool
    ] | None = None
    reasoning_text_tokens: tuple[str, ...] = ()
    tool_text_tokens: tuple[str, ...] = ()
    answer_text_tokens: tuple[str, ...] = ()
    display_tokens: tuple[_ThemeTokenRenderDisplayToken, ...] = ()
    display_reasoning: bool = False
    display_tools: bool = False
    input_token_count: int = 0
    total_tokens: int = 0
    tool_token_count: int = 0
    tool_running: bool = False
    tool_running_spinner: object | None = None
    ttft: float | None = None
    ttnt: float | None = None
    ttsr: float | None = None
    elapsed: float = 0.0
    event_stats: EventStats | None = None
    start_thinking: bool = False


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamPresenterContext:
    """Represent non-snapshot state needed by stream presenters."""

    model_id: str
    console_width: int
    input_token_count: int = 0
    tokenizer_tokens: tuple[str, ...] | None = None
    tokenizer_special_tokens: tuple[str, ...] | None = None
    token_probability_pick: int = 0
    start_thinking: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.model_id, str) and self.model_id.strip()
        assert isinstance(self.console_width, int)
        assert self.console_width > 0
        assert isinstance(self.input_token_count, int)
        assert self.input_token_count >= 0
        assert self.tokenizer_tokens is None or (
            isinstance(self.tokenizer_tokens, tuple)
            and all(isinstance(token, str) for token in self.tokenizer_tokens)
        )
        assert self.tokenizer_special_tokens is None or (
            isinstance(self.tokenizer_special_tokens, tuple)
            and all(
                isinstance(token, str)
                for token in self.tokenizer_special_tokens
            )
        )
        assert isinstance(self.token_probability_pick, int)
        assert self.token_probability_pick >= 0
        assert isinstance(self.start_thinking, bool)


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamPresenterRequest:
    """Represent one immutable presenter input."""

    snapshot: CliStreamSnapshot
    display_config: CliStreamDisplayConfig
    context: CliStreamPresenterContext
    mode: StreamPresenterMode

    def __post_init__(self) -> None:
        assert isinstance(self.snapshot, CliStreamSnapshot)
        assert isinstance(self.display_config, CliStreamDisplayConfig)
        assert isinstance(self.context, CliStreamPresenterContext)
        assert self.mode in ("live", "answer")


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamRenderableFrame:
    """Represent one renderable live frame."""

    renderable: RenderableType
    role: StreamFrameRole = "stream"
    current_token: CliDisplayTokenSnapshot | None = None

    def __post_init__(self) -> None:
        assert self.role in ("stream", "events", "tools", "stats", "answer")
        assert self.current_token is None or isinstance(
            self.current_token, CliDisplayTokenSnapshot
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamAnswerTextChunk:
    """Represent one plain answer text chunk."""

    text: str
    role: StreamFrameRole = "answer"

    def __post_init__(self) -> None:
        assert isinstance(self.text, str)
        assert self.text
        assert self.role == "answer"


CliStreamPresenterItem: TypeAlias = (
    CliStreamRenderableFrame | CliStreamAnswerTextChunk
)


class CliStreamPresenter(Protocol):
    """Present immutable stream snapshots for one runner mode."""

    def present(
        self,
        request: CliStreamPresenterRequest,
    ) -> AsyncIterator[CliStreamPresenterItem]: ...


class CliStreamAnswerPresenter:
    """Emit monotonic answer suffixes from stream snapshots."""

    def __init__(self) -> None:
        self._emitted_answer_text = ""

    def reset(self) -> None:
        """Forget previously emitted answer text."""
        self._emitted_answer_text = ""

    async def present(
        self,
        request: CliStreamPresenterRequest,
    ) -> AsyncIterator[CliStreamPresenterItem]:
        """Yield unseen answer text from one snapshot."""
        assert isinstance(request, CliStreamPresenterRequest)
        if request.mode != "answer":
            raise AssertionError("answer presenter requires answer mode")
        if request.snapshot.answer_text == self._emitted_answer_text:
            return
        if not request.snapshot.answer_text.startswith(
            self._emitted_answer_text
        ):
            raise AssertionError("answer snapshots must grow monotonically")

        text = request.snapshot.answer_text[len(self._emitted_answer_text) :]
        self._emitted_answer_text = request.snapshot.answer_text
        yield CliStreamAnswerTextChunk(text=text)


class CliStreamSnapshotPresenter:
    """Present snapshots without theme token-frame animation."""

    def __init__(
        self,
        logger: Logger,
        *,
        event_stats: EventStats | None = None,
    ) -> None:
        assert isinstance(logger, Logger)
        assert event_stats is None or isinstance(event_stats, EventStats)
        self._answer_presenter = CliStreamAnswerPresenter()
        self._event_stats = event_stats
        self._logger = logger

    def reset(self) -> None:
        """Forget answer text emitted by this presenter."""
        self._answer_presenter.reset()

    async def present(
        self,
        request: CliStreamPresenterRequest,
    ) -> AsyncIterator[CliStreamPresenterItem]:
        """Yield renderable frames or plain answer chunks."""
        assert isinstance(request, CliStreamPresenterRequest)
        _ = self._event_stats, self._logger
        if request.mode == "answer":
            async for chunk in self._answer_presenter.present(
                _answer_request(request)
            ):
                yield chunk
            return

        for frame in _snapshot_diagnostic_frames(request):
            yield frame
        if not request.display_config.show_stats:
            async for chunk in self._answer_presenter.present(
                _answer_request(request)
            ):
                yield chunk
            return
        if request.snapshot.answer_text:
            yield CliStreamRenderableFrame(
                renderable=request.snapshot.answer_text,
                role="answer",
            )


class LegacyThemeStreamPresenter:
    """Adapt immutable snapshots to a theme token-frame renderer."""

    def __init__(
        self,
        theme: object,
        logger: Logger,
        *,
        event_stats: EventStats | None = None,
    ) -> None:
        token_frame_stream_factory = getattr(theme, "token_frames", None)
        assert callable(token_frame_stream_factory)
        assert isinstance(logger, Logger)
        assert event_stats is None or isinstance(event_stats, EventStats)
        self._answer_presenter = CliStreamAnswerPresenter()
        self._event_stats = event_stats
        self._logger = logger
        self._token_frame_stream_factory = cast(
            TokenFrameStreamFactory, token_frame_stream_factory
        )

    def reset(self) -> None:
        """Forget answer text emitted by this presenter."""
        self._answer_presenter.reset()

    async def present(
        self,
        request: CliStreamPresenterRequest,
    ) -> AsyncIterator[CliStreamPresenterItem]:
        """Yield renderable frames or plain answer chunks."""
        assert isinstance(request, CliStreamPresenterRequest)
        if request.mode == "answer":
            async for chunk in self._answer_presenter.present(
                _answer_request(request)
            ):
                yield chunk
            return

        for frame in _snapshot_diagnostic_frames(request):
            yield frame

        if not request.display_config.show_stats:
            async for chunk in self._answer_presenter.present(
                _answer_request(request)
            ):
                yield chunk
            return

        token_frame_stream = await self._theme_token_frame_stream(request)
        try:
            async for token, renderable in token_frame_stream:
                yield CliStreamRenderableFrame(
                    renderable=renderable,
                    current_token=_display_token_from_render_token(token),
                )
        finally:
            await _close_token_frame_stream(token_frame_stream)

    async def _theme_token_frame_stream(
        self,
        request: CliStreamPresenterRequest,
    ) -> TokenFrameStream:
        context = request.context
        snapshot = request.snapshot
        display_config = request.display_config
        display_probabilities = (
            display_config.show_probabilities
            and context.token_probability_pick > 0
        )
        state = _theme_token_render_state(
            request,
            display_probabilities=display_probabilities,
            focus_on_token_when=_focus_on_display_token(
                _last_display_token(snapshot),
                display_config,
                display_probabilities=display_probabilities,
            ),
            event_stats=self._event_stats,
        )
        return await _coerce_token_frame_stream(
            self._token_frame_stream_factory(
                state,
                console_width=context.console_width,
                logger=self._logger,
                maximum_frames=1,
                logits_count=None,
                tool_events_limit=display_config.display_tools_events,
                height=display_config.answer_height,
                limit_answer_height=not display_config.answer_height_expand,
                start_thinking=context.start_thinking,
            )
        )


def _snapshot_diagnostic_frames(
    request: CliStreamPresenterRequest,
) -> tuple[CliStreamRenderableFrame, ...]:
    snapshot = request.snapshot
    display_config = request.display_config
    role_texts: tuple[tuple[StreamFrameRole, str], ...] = (
        (
            "tools",
            _tool_summary(snapshot) if display_config.show_tools else "",
        ),
        (
            "events",
            _event_summary(snapshot) if display_config.show_events else "",
        ),
        (
            "stats",
            _stats_summary(snapshot) if display_config.show_stats else "",
        ),
    )
    return tuple(
        CliStreamRenderableFrame(renderable=text, role=role)
        for role, text in role_texts
        if text
    )


def _tool_summary(snapshot: CliStreamSnapshot) -> str:
    return "\n".join(
        (
            *(
                f"active tool {tool.name}: "
                f"{tool.arguments_summary or tool.tool_call_id}"
                for tool in snapshot.active_tools
            ),
            *(
                f"completed tool {tool.name}: {tool.status}"
                for tool in snapshot.completed_tools
            ),
            *(
                f"tool {result.status} {result.name}: {result.result_summary}"
                for result in snapshot.tool_results
            ),
            *(
                f"tool diagnostic {diagnostic.code}: {diagnostic.message}"
                for diagnostic in snapshot.tool_diagnostics
            ),
            *(
                f"tool event {event.event_type}: "
                f"{event.name or event.tool_call_id or event.payload_summary}"
                for event in snapshot.tool_events
            ),
        )
    )


def _event_summary(snapshot: CliStreamSnapshot) -> str:
    return "\n".join(
        f"event {event.event_type}: "
        f"{event.payload_summary or event.observability_summary or ''}"
        for event in snapshot.events
    )


def _stats_summary(snapshot: CliStreamSnapshot) -> str:
    return "\n".join(
        (
            *(
                f"usage {usage.kind or 'usage'}: {usage.usage_summary}"
                for usage in snapshot.usage_summaries
            ),
            *(
                f"projection {projection.kind or 'projection'}: "
                f"{_projection_summary_text(projection)}"
                for projection in snapshot.projection_metadata_summaries
            ),
        )
    )


def _projection_summary_text(
    projection: CliProjectionMetadataSummarySnapshot,
) -> str:
    return projection.data_summary or projection.metadata_summary or ""


def _output_token_count(snapshot: CliStreamSnapshot) -> int:
    return (
        snapshot.token_counts.output_tokens
        if snapshot.token_counts.output_tokens is not None
        else (
            snapshot.token_counts.answer_tokens
            + snapshot.token_counts.reasoning_tokens
        )
    )


def _snapshot_elapsed_seconds(snapshot: CliStreamSnapshot) -> float:
    if snapshot.timing.elapsed_seconds is not None:
        return snapshot.timing.elapsed_seconds
    if (
        snapshot.timing.started_at is not None
        and snapshot.timing.updated_at is not None
    ):
        return max(
            0.0, snapshot.timing.updated_at - snapshot.timing.started_at
        )
    return 0.0


def _answer_request(
    request: CliStreamPresenterRequest,
) -> CliStreamPresenterRequest:
    return CliStreamPresenterRequest(
        snapshot=request.snapshot,
        display_config=request.display_config,
        context=request.context,
        mode="answer",
    )


def _theme_token_render_state(
    request: CliStreamPresenterRequest,
    *,
    display_probabilities: bool,
    focus_on_token_when: Callable[
        [_ThemeTokenRenderDisplayToken], bool
    ] | None,
    event_stats: EventStats | None,
) -> _ThemeTokenRenderState:
    snapshot = request.snapshot
    context = request.context
    display_config = request.display_config
    return _ThemeTokenRenderState(
        model_id=context.model_id,
        added_tokens=context.tokenizer_tokens,
        special_tokens=context.tokenizer_special_tokens,
        display_token_size=display_config.display_tokens or None,
        display_probabilities=display_probabilities,
        pick=context.token_probability_pick,
        focus_on_token_when=focus_on_token_when,
        reasoning_text_tokens=(
            (snapshot.reasoning_text,) if snapshot.reasoning_text else ()
        ),
        tool_text_tokens=(
            (snapshot.tool_call_request_text,)
            if snapshot.tool_call_request_text
            else ()
        ),
        answer_text_tokens=(
            (snapshot.answer_text,) if snapshot.answer_text else ()
        ),
        display_tokens=tuple(
            _render_token_from_display_snapshot(display_token)
            for display_token in snapshot.display_tokens
        ),
        display_reasoning=bool(snapshot.reasoning_text),
        display_tools=display_config.show_tools,
        input_token_count=(
            snapshot.token_counts.input_tokens
            if snapshot.token_counts.input_tokens is not None
            else context.input_token_count
        ),
        total_tokens=_output_token_count(snapshot),
        tool_token_count=snapshot.token_counts.tool_call_tokens,
        ttft=snapshot.timing.first_token_seconds,
        ttnt=(
            snapshot.timing.time_to_n_token_seconds
            if display_config.display_time_to_n_token is not None
            else None
        ),
        ttsr=(
            snapshot.timing.reasoning_seconds
            if display_config.display_reasoning_time
            else None
        ),
        elapsed=_snapshot_elapsed_seconds(snapshot),
        event_stats=event_stats,
        start_thinking=context.start_thinking,
    )


async def _coerce_token_frame_stream(
    result: object,
) -> TokenFrameStream:
    if isawaitable(result):
        result = await result
    if isinstance(result, AsyncIterable):
        return cast(TokenFrameStream, result.__aiter__())
    if isinstance(result, tuple) or isinstance(result, list):
        return _iter_token_frame_stream(result)
    if hasattr(result, "__iter__"):
        return _iter_token_frame_stream(tuple(cast(Any, result)))
    raise AssertionError("theme token_frames must return frame iterable")


async def _iter_token_frame_stream(
    frames: tuple[object, ...] | list[object],
) -> TokenFrameStream:
    for frame in frames:
        assert isinstance(frame, tuple) and len(frame) == 2
        current_token, renderable = frame
        assert current_token is None or isinstance(
            current_token, _ThemeTokenRenderDisplayToken
        )
        yield cast(
            tuple[_ThemeTokenRenderDisplayToken | None, RenderableType],
            frame,
        )


async def _close_token_frame_stream(stream: TokenFrameStream) -> None:
    aclose = getattr(stream, "aclose", None)
    if callable(aclose):
        result = aclose()
        if isawaitable(result):
            await result


def _last_display_token(
    snapshot: CliStreamSnapshot,
) -> CliDisplayTokenSnapshot | None:
    return snapshot.display_tokens[-1] if snapshot.display_tokens else None


def _focus_on_display_token(
    current: CliDisplayTokenSnapshot | None,
    config: CliStreamDisplayConfig,
    *,
    display_probabilities: bool,
) -> Callable[[_ThemeTokenRenderDisplayToken], bool] | None:
    if (
        current is None
        or not display_probabilities
        or config.display_probabilities_maximum <= 0
    ):
        return None

    def token_is_focused(token: _ThemeTokenRenderDisplayToken) -> bool:
        if (
            token.probability is not None
            and token.probability < config.display_probabilities_maximum
        ):
            return True
        if not token.tokens:
            return False
        for candidate in token.tokens:
            if (
                candidate.id != token.id
                and candidate.probability is not None
                and candidate.probability
                >= config.display_probabilities_sample_minimum
            ):
                return True
        return False

    return token_is_focused


def _render_token_from_display_snapshot(
    token: CliDisplayTokenSnapshot,
) -> _ThemeTokenRenderDisplayToken:
    return _ThemeTokenRenderDisplayToken(
        sequence=token.sequence if token.sequence is not None else 0,
        kind=StreamItemKind.ANSWER_DELTA,
        channel=StreamChannel.ANSWER,
        token=token.text,
        id=_token_id(token.token_id),
        probability=token.probability,
        step=token.step,
        probability_distribution=token.probability_distribution,
        tokens=tuple(
            _ThemeTokenRenderDisplayTokenCandidate(
                id=_token_id(candidate.token_id),
                token=candidate.text,
                probability=candidate.probability,
            )
            for candidate in token.candidates
        )
    )


def _display_token_from_render_token(
    token: _ThemeTokenRenderDisplayToken | None,
) -> CliDisplayTokenSnapshot | None:
    if token is None:
        return None
    candidates = tuple(
        CliDisplayTokenCandidateSnapshot(
            token_id=_display_token_id(candidate.id),
            text=candidate.token,
            probability=candidate.probability,
        )
        for candidate in token.tokens
    )
    return CliDisplayTokenSnapshot(
        sequence=token.sequence,
        token_id=_display_token_id(token.id),
        text=token.token,
        probability=token.probability,
        step=token.step,
        probability_distribution=token.probability_distribution,
        candidates=candidates,
    )


def _token_id(token_id: int | str | None) -> int | str | None:
    return (
        token_id
        if isinstance(token_id, int | str) and not isinstance(token_id, bool)
        else None
    )


def _display_token_id(token_id: object) -> int | str | None:
    if token_id is None or isinstance(token_id, int | str):
        return token_id
    return str(token_id)
