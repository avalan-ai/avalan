"""Define CLI stream presenter contracts."""

from ..entities import ProbabilityDistribution, Token, TokenDetail
from .display import CliStreamDisplayConfig
from .display_snapshot import (
    CliDisplayTokenCandidateSnapshot,
    CliDisplayTokenSnapshot,
    CliProjectionMetadataSummarySnapshot,
    CliStreamSnapshot,
)

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from logging import Logger
from typing import Any, Literal, Protocol, TypeAlias, cast

from rich.console import RenderableType

StreamPresenterMode = Literal["live", "answer"]
StreamFrameRole = Literal["stream", "events", "tools", "stats", "answer"]
TokenFrameStream: TypeAlias = AsyncIterator[
    tuple[Token | None, RenderableType]
]
TokenFrameStreamFactory: TypeAlias = Callable[
    ..., TokenFrameStream | Awaitable[TokenFrameStream]
]

_PROBABILITY_DISTRIBUTIONS: tuple[ProbabilityDistribution, ...] = (
    "entmax",
    "gumbel_softmax",
    "log_softmax",
    "sparsemax",
    "softmax",
)


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


class LegacyThemeStreamPresenter:
    """Adapt immutable snapshots to the temporary legacy theme stream API."""

    def __init__(self, theme: object, logger: Logger) -> None:
        token_frame_stream_factory = getattr(theme, "tokens", None)
        assert callable(token_frame_stream_factory)
        assert isinstance(logger, Logger)
        self._answer_presenter = CliStreamAnswerPresenter()
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
                    current_token=_display_token_from_token(token),
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
        return await _coerce_token_frame_stream(
            self._token_frame_stream_factory(
                model_id=context.model_id,
                added_tokens=(
                    list(context.tokenizer_tokens)
                    if context.tokenizer_tokens is not None
                    else None
                ),
                special_tokens=(
                    list(context.tokenizer_special_tokens)
                    if context.tokenizer_special_tokens is not None
                    else None
                ),
                display_token_size=display_config.display_tokens,
                display_probabilities=display_probabilities,
                pick=context.token_probability_pick,
                focus_on_token_when=_focus_on_display_token(
                    _last_display_token(snapshot),
                    display_config,
                    display_probabilities=display_probabilities,
                ),
                thinking_text_tokens=(
                    [snapshot.reasoning_text]
                    if snapshot.reasoning_text
                    else []
                ),
                tool_text_tokens=(
                    [snapshot.tool_call_request_text]
                    if snapshot.tool_call_request_text
                    else []
                ),
                answer_text_tokens=(
                    [snapshot.answer_text] if snapshot.answer_text else []
                ),
                tokens=[
                    _token_from_display_snapshot(display_token)
                    for display_token in snapshot.display_tokens
                ]
                or None,
                input_token_count=(
                    snapshot.token_counts.input_tokens
                    if snapshot.token_counts.input_tokens is not None
                    else context.input_token_count
                ),
                total_tokens=_legacy_output_token_count(snapshot),
                tool_events=[],
                tool_event_calls=[],
                tool_event_results=[],
                tool_running_spinner=None,
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
                console_width=context.console_width,
                logger=self._logger,
                event_stats=None,
                tool_token_count=snapshot.token_counts.tool_call_tokens,
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


def _legacy_output_token_count(snapshot: CliStreamSnapshot) -> int:
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


async def _coerce_token_frame_stream(
    result: TokenFrameStream | Awaitable[TokenFrameStream],
) -> TokenFrameStream:
    if isawaitable(result):
        result = await result
    assert hasattr(result, "__aiter__")
    return result


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
) -> Callable[[Token], bool] | None:
    if (
        current is None
        or not display_probabilities
        or config.display_probabilities_maximum <= 0
    ):
        return None

    def token_is_focused(token: Token) -> bool:
        if (
            token.probability is not None
            and token.probability < config.display_probabilities_maximum
        ):
            return True
        if not isinstance(token, TokenDetail) or not token.tokens:
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


def _token_from_display_snapshot(
    token: CliDisplayTokenSnapshot,
) -> Token:
    candidates = [
        Token(
            id=cast(Any, _token_id(candidate.token_id)),
            token=candidate.text,
            probability=candidate.probability,
        )
        for candidate in token.candidates
    ]
    probability_distribution = _probability_distribution(
        token.probability_distribution
    )
    if candidates or token.step is not None or probability_distribution:
        return TokenDetail(
            id=cast(Any, _token_id(token.token_id)),
            token=token.text,
            probability=token.probability,
            step=token.step,
            probability_distribution=probability_distribution,
            tokens=candidates or None,
        )
    return Token(
        id=cast(Any, _token_id(token.token_id)),
        token=token.text,
        probability=token.probability,
    )


def _display_token_from_token(
    token: Token | None,
) -> CliDisplayTokenSnapshot | None:
    if token is None:
        return None
    candidates = (
        tuple(
            CliDisplayTokenCandidateSnapshot(
                token_id=_display_token_id(candidate.id),
                text=candidate.token,
                probability=candidate.probability,
            )
            for candidate in token.tokens
        )
        if isinstance(token, TokenDetail) and token.tokens
        else ()
    )
    return CliDisplayTokenSnapshot(
        sequence=None,
        token_id=_display_token_id(token.id),
        text=token.token,
        probability=token.probability,
        step=token.step if isinstance(token, TokenDetail) else None,
        probability_distribution=(
            token.probability_distribution
            if isinstance(token, TokenDetail)
            else None
        ),
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


def _probability_distribution(
    value: str | None,
) -> ProbabilityDistribution | None:
    if value in _PROBABILITY_DISTRIBUTIONS:
        return value
    return None
