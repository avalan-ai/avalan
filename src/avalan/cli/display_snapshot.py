"""Represent immutable CLI stream display snapshots."""

from ..entities import ToolCallDiagnostic, ToolCallError
from ..model.stream import (
    StreamConsumerProjection,
    stream_projection_text_delta,
)
from ..tool.display import ToolDisplayProjection
from .display import CliStreamDisplayConfig, DiagnosticChannel
from .display_safety import (
    MAX_SUMMARY_CHARS,
    REDACTED,
    contains_sensitive_marker,
    event_type_value,
    safe_summary,
    safe_text,
    safe_tool_call_request_text,
    truncate_text,
    value_from,
)

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Generic, Literal, TypeVar

DEFAULT_EVENT_HISTORY_LIMIT = 4
DEFAULT_PROJECTION_SUMMARY_LIMIT = 4
# A config value of display_tools_events=None means "no visible display
# limit"; snapshots still use this hard cap so retained CLI history is bounded.
DEFAULT_UNLIMITED_TOOL_HISTORY_LIMIT = 256
_DISPLAY_TOKEN_METADATA_KEYS = frozenset(
    (
        "token_id",
        "probability",
        "step",
        "probability_distribution",
        "tokens",
    )
)

HistoryItem = TypeVar("HistoryItem")
CoalescedValue = TypeVar("CoalescedValue")
ToolStatus = Literal["active", "completed", "error", "cancelled"]
ToolResultStatus = Literal["result", "error"]
ModelContinuationStatus = Literal["active"]


class CliAppendOnlyTextBuffer:
    """Collect text chunks and materialize them only on demand."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._character_count = 0
        self._materialization_count = 0

    @property
    def chunk_count(self) -> int:
        """Return how many chunks were appended."""
        return len(self._chunks)

    @property
    def character_count(self) -> int:
        """Return how many characters were appended."""
        return self._character_count

    @property
    def materialization_count(self) -> int:
        """Return how many times the full text was joined."""
        return self._materialization_count

    def append(self, text: str) -> None:
        """Append one text chunk."""
        assert isinstance(text, str)
        if not text:
            return
        self._chunks.append(text)
        self._character_count += len(text)

    def materialize(self) -> str:
        """Return the joined text."""
        self._materialization_count += 1
        return "".join(self._chunks)


class CliBoundedTextBuffer:
    """Collect recent text while keeping display memory bounded."""

    def __init__(self, limit: int = MAX_SUMMARY_CHARS) -> None:
        assert isinstance(limit, int)
        assert limit >= 0
        self._limit = limit
        self._text = ""
        self._character_count = 0
        self._chunk_count = 0
        self._dropped_count = 0
        self._materialization_count = 0
        self._scan_tail = ""
        self._sensitive_seen = False

    @property
    def chunk_count(self) -> int:
        """Return how many chunks were appended."""
        return self._chunk_count

    @property
    def character_count(self) -> int:
        """Return how many characters were appended."""
        return self._character_count

    @property
    def materialization_count(self) -> int:
        """Return how many times retained text was materialized."""
        return self._materialization_count

    def append(self, text: str) -> None:
        """Append one text chunk while retaining a bounded tail."""
        assert isinstance(text, str)
        if not text:
            return
        self._chunk_count += 1
        self._character_count += len(text)
        scan_text = self._scan_tail + text
        self._sensitive_seen = (
            self._sensitive_seen or contains_sensitive_marker(scan_text)
        )
        self._scan_tail = scan_text[-MAX_SUMMARY_CHARS:]

        if self._limit == 0:
            self._dropped_count += len(text)
            self._text = ""
            return

        retained = self._text + text
        if len(retained) <= self._limit:
            self._text = retained
            return

        dropped = len(retained) - self._limit
        self._dropped_count += dropped
        self._text = truncate_text(retained[-self._limit :], self._limit)

    def materialize(self) -> str:
        """Return the retained display text."""
        self._materialization_count += 1
        if self._sensitive_seen and self._dropped_count:
            return REDACTED
        return self._text


class CliBoundedHistoryBuffer(Generic[HistoryItem]):
    """Collect recent immutable history entries up to a fixed limit."""

    def __init__(self, limit: int) -> None:
        assert isinstance(limit, int)
        assert limit >= 0
        self._limit = limit
        self._items: deque[HistoryItem] = deque()
        self._dropped_count = 0
        self._materialization_count = 0

    @property
    def dropped_count(self) -> int:
        """Return how many entries were dropped by the bound."""
        return self._dropped_count

    @property
    def item_count(self) -> int:
        """Return how many entries are retained."""
        return len(self._items)

    @property
    def limit(self) -> int:
        """Return the hard retention limit."""
        return self._limit

    @property
    def materialization_count(self) -> int:
        """Return how many immutable tuple views were built."""
        return self._materialization_count

    def append(self, item: HistoryItem) -> None:
        """Append one immutable history item."""
        if self._limit == 0:
            self._dropped_count += 1
            return
        if len(self._items) == self._limit:
            self._items.popleft()
            self._dropped_count += 1
        self._items.append(item)

    def remove_matching(
        self,
        predicate: Callable[[HistoryItem], bool],
    ) -> int:
        """Remove retained items matching predicate."""
        assert callable(predicate)
        kept: deque[HistoryItem] = deque()
        removed = 0
        for item in self._items:
            if predicate(item):
                removed += 1
            else:
                kept.append(item)
        self._items = kept
        return removed

    def snapshot(self) -> tuple[HistoryItem, ...]:
        """Return an immutable view of retained history."""
        self._materialization_count += 1
        return tuple(self._items)


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamDisplayFlagsSnapshot:
    quiet: bool
    stats: bool
    display_tools: bool
    display_events: bool
    display_tools_events: int | None
    record: bool
    interactive: bool
    diagnostic_channel: DiagnosticChannel
    show_stats: bool
    show_tools: bool
    show_events: bool
    show_token_details: bool
    show_probabilities: bool
    show_timing: bool
    live_enabled: bool
    record_enabled: bool
    answer_stdout_only: bool
    refresh_per_second: int
    answer_height: int
    answer_height_expand: bool
    display_tokens: int
    display_pause: int
    display_probabilities: bool
    display_probabilities_maximum: float
    display_probabilities_sample_minimum: float
    display_time_to_n_token: int | None
    display_reasoning_time: bool


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamRetentionSnapshot:
    show_tools: bool
    show_events: bool
    show_stats: bool
    active_tools_retained: bool
    visible_tool_history_limit: int | None
    internal_tool_history_limit: int
    event_history_limit: int
    display_token_history_limit: int
    usage_summary_history_limit: int
    projection_metadata_history_limit: int


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamTokenCountsSnapshot:
    answer_tokens: int = 0
    reasoning_tokens: int = 0
    tool_call_tokens: int = 0
    display_tokens: int = 0
    total_tokens: int = 0
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_usage_tokens: int | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamTimingSnapshot:
    started_at: float | None = None
    updated_at: float | None = None
    finished_at: float | None = None
    elapsed_seconds: float | None = None
    first_token_seconds: float | None = None
    reasoning_seconds: float | None = None
    time_to_n_token_seconds: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamTerminalSnapshot:
    completed: bool = False
    outcome: str | None = None
    sequence: int | None = None
    error_summary: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliDisplayTokenCandidateSnapshot:
    token_id: int | str | None
    text: str
    probability: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliDisplayTokenSnapshot:
    sequence: int | None
    token_id: int | str | None
    text: str
    probability: float | None = None
    step: int | None = None
    probability_distribution: str | None = None
    candidates: tuple[CliDisplayTokenCandidateSnapshot, ...] = ()


@dataclass(frozen=True, kw_only=True, slots=True)
class CliToolExecutionSummarySnapshot:
    tool_call_id: str
    name: str
    arguments_summary: str | None = None
    display_projection: ToolDisplayProjection | None = None
    provider_name: str | None = None
    sequence: int | None = None
    status: ToolStatus = "active"
    started_at: float | None = None
    updated_at: float | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliToolResultSummarySnapshot:
    tool_call_id: str
    name: str
    status: ToolResultStatus
    result_summary: str
    arguments_count: int
    display_projection: ToolDisplayProjection | None = None
    sequence: int | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliToolDiagnosticSummarySnapshot:
    diagnostic_id: str
    tool_call_id: str | None
    requested_name: str | None
    canonical_name: str | None
    status: str
    code: str
    stage: str
    message: str
    details_summary: str | None = None
    retryable: bool = False
    sequence: int | None = None
    duration_ms: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliToolEventSummarySnapshot:
    event_type: str
    tool_call_id: str | None = None
    name: str | None = None
    payload_summary: str | None = None
    observability_summary: str | None = None
    sequence: int | None = None
    started: float | None = None
    finished: float | None = None
    elapsed: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliModelContinuationSnapshot:
    model_continuation_id: str
    status: ModelContinuationStatus = "active"
    sequence: int | None = None
    started_at: float | None = None
    updated_at: float | None = None
    elapsed_seconds: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliEventSummarySnapshot:
    event_type: str
    payload_summary: str | None = None
    observability_summary: str | None = None
    sequence: int | None = None
    started: float | None = None
    finished: float | None = None
    elapsed: float | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliUsageSummarySnapshot:
    sequence: int | None
    kind: str | None
    usage_summary: str
    provider_family: str | None = None
    provider_event_type: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliProjectionMetadataSummarySnapshot:
    sequence: int | None
    kind: str | None
    data_summary: str | None = None
    metadata_summary: str | None = None
    provider_family: str | None = None
    provider_event_type: str | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamBuildStatsSnapshot:
    snapshots_built: int
    answer_chunks: int
    reasoning_chunks: int
    tool_call_request_chunks: int
    answer_characters: int
    reasoning_characters: int
    tool_call_request_characters: int
    text_materializations: int
    history_materializations: int
    active_tools: int
    retained_completed_tools: int
    retained_tool_results: int
    retained_tool_diagnostics: int
    retained_tool_events: int
    retained_events: int
    retained_display_tokens: int
    retained_usage_summaries: int
    retained_projection_metadata_summaries: int
    dropped_completed_tools: int
    dropped_tool_results: int
    dropped_tool_diagnostics: int
    dropped_tool_events: int
    dropped_events: int
    dropped_display_tokens: int
    dropped_usage_summaries: int
    dropped_projection_metadata_summaries: int


@dataclass(frozen=True, kw_only=True, slots=True)
class CliStreamSnapshot:
    display: CliStreamDisplayFlagsSnapshot
    retention: CliStreamRetentionSnapshot
    token_counts: CliStreamTokenCountsSnapshot
    timing: CliStreamTimingSnapshot
    terminal: CliStreamTerminalSnapshot
    answer_text: str
    reasoning_text: str
    tool_call_request_text: str
    active_tools: tuple[CliToolExecutionSummarySnapshot, ...]
    completed_tools: tuple[CliToolExecutionSummarySnapshot, ...]
    tool_results: tuple[CliToolResultSummarySnapshot, ...]
    tool_diagnostics: tuple[CliToolDiagnosticSummarySnapshot, ...]
    tool_events: tuple[CliToolEventSummarySnapshot, ...]
    active_model_continuations: tuple[CliModelContinuationSnapshot, ...]
    events: tuple[CliEventSummarySnapshot, ...]
    display_tokens: tuple[CliDisplayTokenSnapshot, ...]
    usage_summaries: tuple[CliUsageSummarySnapshot, ...]
    projection_metadata_summaries: tuple[
        CliProjectionMetadataSummarySnapshot, ...
    ]
    build_stats: CliStreamBuildStatsSnapshot


def display_flags_from_config(
    config: CliStreamDisplayConfig,
) -> CliStreamDisplayFlagsSnapshot:
    """Return immutable display flags derived from config."""
    assert isinstance(config, CliStreamDisplayConfig)
    return CliStreamDisplayFlagsSnapshot(
        quiet=config.quiet,
        stats=config.stats,
        display_tools=config.display_tools,
        display_events=config.display_events,
        display_tools_events=config.display_tools_events,
        record=config.record,
        interactive=config.interactive,
        diagnostic_channel=config.diagnostic_channel,
        show_stats=config.show_stats,
        show_tools=config.show_tools,
        show_events=config.show_events,
        show_token_details=config.show_token_details,
        show_probabilities=config.show_probabilities,
        show_timing=config.show_timing,
        live_enabled=config.live_enabled,
        record_enabled=config.record_enabled,
        answer_stdout_only=config.answer_stdout_only,
        refresh_per_second=config.refresh_per_second,
        answer_height=config.answer_height,
        answer_height_expand=config.answer_height_expand,
        display_tokens=config.display_tokens,
        display_pause=config.display_pause,
        display_probabilities=config.display_probabilities,
        display_probabilities_maximum=config.display_probabilities_maximum,
        display_probabilities_sample_minimum=(
            config.display_probabilities_sample_minimum
        ),
        display_time_to_n_token=config.display_time_to_n_token,
        display_reasoning_time=config.display_reasoning_time,
    )


def retention_from_config(
    config: CliStreamDisplayConfig,
    *,
    event_history_limit: int = DEFAULT_EVENT_HISTORY_LIMIT,
    unlimited_tool_history_limit: int = DEFAULT_UNLIMITED_TOOL_HISTORY_LIMIT,
    projection_summary_limit: int = DEFAULT_PROJECTION_SUMMARY_LIMIT,
) -> CliStreamRetentionSnapshot:
    """Return bounded snapshot retention derived from config."""
    assert isinstance(config, CliStreamDisplayConfig)
    assert isinstance(event_history_limit, int)
    assert event_history_limit >= 0
    assert isinstance(unlimited_tool_history_limit, int)
    assert unlimited_tool_history_limit >= 0
    assert isinstance(projection_summary_limit, int)
    assert projection_summary_limit >= 0

    show_tools = config.show_tools
    visible_tool_limit = config.display_tools_events if show_tools else 0
    if not show_tools:
        internal_tool_limit = 0
    elif config.display_tools_events is None:
        internal_tool_limit = unlimited_tool_history_limit
    else:
        internal_tool_limit = config.display_tools_events

    show_stats = config.show_stats
    return CliStreamRetentionSnapshot(
        show_tools=show_tools,
        show_events=config.show_events,
        show_stats=show_stats,
        active_tools_retained=show_tools,
        visible_tool_history_limit=visible_tool_limit,
        internal_tool_history_limit=internal_tool_limit,
        event_history_limit=event_history_limit if config.show_events else 0,
        display_token_history_limit=(
            config.display_tokens if config.show_token_details else 0
        ),
        usage_summary_history_limit=(
            projection_summary_limit if show_stats else 0
        ),
        projection_metadata_history_limit=(
            projection_summary_limit if show_stats else 0
        ),
    )


def _has_token_display_fields(value: object) -> bool:
    return (
        hasattr(value, "id")
        and hasattr(value, "token")
        and hasattr(value, "probability")
    )


def _source_token_candidate_snapshots(
    value: object,
) -> tuple[CliDisplayTokenCandidateSnapshot, ...]:
    if not isinstance(value, tuple | list):
        return ()
    candidates: list[CliDisplayTokenCandidateSnapshot] = []
    for candidate in value:
        if not _has_token_display_fields(candidate):
            continue
        candidates.append(
            CliDisplayTokenCandidateSnapshot(
                token_id=_safe_id(getattr(candidate, "id")),
                text=safe_text(getattr(candidate, "token")),
                probability=_safe_probability(
                    getattr(candidate, "probability")
                ),
            )
        )
    return tuple(candidates)


def display_token_snapshot(
    token: object,
    *,
    sequence: int | None = None,
    include_candidates: bool = False,
) -> CliDisplayTokenSnapshot:
    """Return immutable display metadata for a token."""
    assert _has_token_display_fields(token)
    candidates: tuple[CliDisplayTokenCandidateSnapshot, ...] = ()
    if include_candidates:
        candidates = _source_token_candidate_snapshots(
            getattr(token, "tokens", ())
        )
    probability_distribution = _primitive_value(
        getattr(token, "probability_distribution", None)
    )
    step = _metadata_int(getattr(token, "step", None))
    return CliDisplayTokenSnapshot(
        sequence=sequence,
        token_id=_safe_id(getattr(token, "id")),
        text=safe_text(getattr(token, "token")),
        probability=_safe_probability(getattr(token, "probability")),
        step=step,
        probability_distribution=probability_distribution,
        candidates=candidates,
    )


def display_token_snapshot_from_projection(
    projection: StreamConsumerProjection,
) -> CliDisplayTokenSnapshot | None:
    """Return display token snapshot from a canonical projection."""
    assert isinstance(projection, StreamConsumerProjection)
    token_text = stream_projection_text_delta(projection)
    if token_text is None:
        return None
    metadata = projection.metadata
    if not any(key in metadata for key in _DISPLAY_TOKEN_METADATA_KEYS):
        return None
    token_id = _metadata_int(metadata.get("token_id"))
    probability = _metadata_probability(metadata.get("probability"))
    step = _metadata_int(metadata.get("step"))
    probability_distribution = _metadata_string(
        metadata.get("probability_distribution")
    )
    candidates = _metadata_token_candidate_snapshots(metadata.get("tokens"))
    return CliDisplayTokenSnapshot(
        sequence=projection.sequence,
        token_id=_safe_id(token_id),
        text=safe_text(token_text),
        probability=_safe_probability(probability),
        step=step,
        probability_distribution=_primitive_value(probability_distribution),
        candidates=tuple(candidates),
    )


class CliStreamSnapshotBuilder:
    """Build immutable CLI stream snapshots from append-only state."""

    def __init__(
        self,
        config: CliStreamDisplayConfig,
        *,
        event_history_limit: int = DEFAULT_EVENT_HISTORY_LIMIT,
        unlimited_tool_history_limit: int = (
            DEFAULT_UNLIMITED_TOOL_HISTORY_LIMIT
        ),
        projection_summary_limit: int = DEFAULT_PROJECTION_SUMMARY_LIMIT,
    ) -> None:
        assert isinstance(config, CliStreamDisplayConfig)
        self.display = display_flags_from_config(config)
        self.retention = retention_from_config(
            config,
            event_history_limit=event_history_limit,
            unlimited_tool_history_limit=unlimited_tool_history_limit,
            projection_summary_limit=projection_summary_limit,
        )
        self._answer_text = CliAppendOnlyTextBuffer()
        self._reasoning_text = CliAppendOnlyTextBuffer()
        self._tool_call_request_text = CliBoundedTextBuffer()
        self._active_tools: dict[str, CliToolExecutionSummarySnapshot] = {}
        tool_limit = self.retention.internal_tool_history_limit
        self._completed_tools = CliBoundedHistoryBuffer[
            CliToolExecutionSummarySnapshot
        ](tool_limit)
        self._tool_results = CliBoundedHistoryBuffer[
            CliToolResultSummarySnapshot
        ](tool_limit)
        self._tool_diagnostics = CliBoundedHistoryBuffer[
            CliToolDiagnosticSummarySnapshot
        ](tool_limit)
        self._tool_events = CliBoundedHistoryBuffer[
            CliToolEventSummarySnapshot
        ](tool_limit)
        self._active_model_continuations: dict[
            str, CliModelContinuationSnapshot
        ] = {}
        self._events = CliBoundedHistoryBuffer[CliEventSummarySnapshot](
            self.retention.event_history_limit
        )
        self._display_tokens = CliBoundedHistoryBuffer[
            CliDisplayTokenSnapshot
        ](self.retention.display_token_history_limit)
        self._usage_summaries = CliBoundedHistoryBuffer[
            CliUsageSummarySnapshot
        ](self.retention.usage_summary_history_limit)
        self._projection_metadata_summaries = CliBoundedHistoryBuffer[
            CliProjectionMetadataSummarySnapshot
        ](self.retention.projection_metadata_history_limit)
        self._token_counts = CliStreamTokenCountsSnapshot()
        self._timing = CliStreamTimingSnapshot()
        self._terminal = CliStreamTerminalSnapshot()
        self._snapshots_built = 0

    def append_answer_text(self, text: str, *, tokens: int = 1) -> None:
        """Append answer text and update answer token counts."""
        assert isinstance(tokens, int)
        assert tokens >= 0
        self._answer_text.append(text)
        self._token_counts = replace(
            self._token_counts,
            answer_tokens=self._token_counts.answer_tokens + tokens,
            total_tokens=self._token_counts.total_tokens + tokens,
        )

    def append_reasoning_text(self, text: str, *, tokens: int = 1) -> None:
        """Append reasoning text and update reasoning token counts."""
        assert isinstance(tokens, int)
        assert tokens >= 0
        self._reasoning_text.append(text)
        self._token_counts = replace(
            self._token_counts,
            reasoning_tokens=self._token_counts.reasoning_tokens + tokens,
            total_tokens=self._token_counts.total_tokens + tokens,
        )

    def append_tool_call_request_text(
        self, text: str, *, tokens: int = 1
    ) -> None:
        """Append streamed tool-call request text."""
        assert isinstance(tokens, int)
        assert tokens >= 0
        self._tool_call_request_text.append(text)
        self._token_counts = replace(
            self._token_counts,
            tool_call_tokens=self._token_counts.tool_call_tokens + tokens,
            total_tokens=self._token_counts.total_tokens + tokens,
        )

    def update_token_counts(
        self,
        *,
        input_tokens: int | None = None,
        cached_input_tokens: int | None = None,
        output_tokens: int | None = None,
        reasoning_usage_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Update usage token counts from summarized usage data."""
        self._token_counts = replace(
            self._token_counts,
            input_tokens=_coalesce(
                input_tokens, self._token_counts.input_tokens
            ),
            cached_input_tokens=_coalesce(
                cached_input_tokens,
                self._token_counts.cached_input_tokens,
            ),
            output_tokens=_coalesce(
                output_tokens,
                self._token_counts.output_tokens,
            ),
            reasoning_usage_tokens=_coalesce(
                reasoning_usage_tokens,
                self._token_counts.reasoning_usage_tokens,
            ),
            total_tokens=(
                self._token_counts.total_tokens
                if total_tokens is None
                else total_tokens
            ),
        )

    def update_timing(
        self,
        *,
        started_at: float | None = None,
        updated_at: float | None = None,
        finished_at: float | None = None,
        elapsed_seconds: float | None = None,
        first_token_seconds: float | None = None,
        reasoning_seconds: float | None = None,
        time_to_n_token_seconds: float | None = None,
    ) -> None:
        """Update primitive timing statistics."""
        self._timing = replace(
            self._timing,
            started_at=_coalesce(started_at, self._timing.started_at),
            updated_at=_coalesce(updated_at, self._timing.updated_at),
            finished_at=_coalesce(finished_at, self._timing.finished_at),
            elapsed_seconds=_coalesce(
                elapsed_seconds, self._timing.elapsed_seconds
            ),
            first_token_seconds=_coalesce(
                first_token_seconds, self._timing.first_token_seconds
            ),
            reasoning_seconds=_coalesce(
                reasoning_seconds, self._timing.reasoning_seconds
            ),
            time_to_n_token_seconds=_coalesce(
                time_to_n_token_seconds,
                self._timing.time_to_n_token_seconds,
            ),
        )

    def set_terminal(
        self,
        *,
        completed: bool,
        outcome: object | None = None,
        sequence: int | None = None,
        error: object | None = None,
    ) -> None:
        """Set terminal stream state."""
        assert isinstance(completed, bool)
        self._terminal = CliStreamTerminalSnapshot(
            completed=completed,
            outcome=_primitive_value(outcome),
            sequence=sequence,
            error_summary=None if error is None else safe_summary(error),
        )

    def add_display_token(
        self,
        token: object,
        *,
        sequence: int | None = None,
    ) -> None:
        """Add one optional display token metadata entry."""
        if self.retention.display_token_history_limit == 0:
            return
        self._display_tokens.append(
            display_token_snapshot(
                token,
                sequence=sequence,
                include_candidates=self.display.show_probabilities,
            )
        )
        self._token_counts = replace(
            self._token_counts,
            display_tokens=self._token_counts.display_tokens + 1,
        )

    def add_display_token_from_projection(
        self,
        projection: StreamConsumerProjection,
    ) -> None:
        """Add display token metadata from a canonical projection."""
        snapshot = display_token_snapshot_from_projection(projection)
        if snapshot is None:
            return
        if self.retention.display_token_history_limit == 0:
            return
        self._display_tokens.append(snapshot)
        self._token_counts = replace(
            self._token_counts,
            display_tokens=self._token_counts.display_tokens + 1,
        )

    def add_active_tool(
        self,
        *,
        tool_call_id: object,
        name: object,
        arguments: object | None = None,
        display_projection: ToolDisplayProjection | None = None,
        provider_name: object | None = None,
        sequence: int | None = None,
        started_at: float | None = None,
    ) -> None:
        """Add or replace an active tool execution summary."""
        if not self.display.show_tools:
            return
        if display_projection is not None:
            assert isinstance(display_projection, ToolDisplayProjection)
        tool_id = _safe_string_id(tool_call_id)
        self._active_tools[tool_id] = CliToolExecutionSummarySnapshot(
            tool_call_id=tool_id,
            name=safe_text(name),
            arguments_summary=(
                None if arguments is None else safe_summary(arguments)
            ),
            display_projection=display_projection,
            provider_name=(
                None if provider_name is None else safe_text(provider_name)
            ),
            sequence=sequence,
            status="active",
            started_at=started_at,
        )

    def update_active_tool(
        self,
        *,
        tool_call_id: object,
        name: object | None = None,
        arguments: object | None = None,
        display_projection: ToolDisplayProjection | None = None,
        provider_name: object | None = None,
        sequence: int | None = None,
        updated_at: float | None = None,
    ) -> None:
        """Update retained active tool execution details."""
        if not self.display.show_tools:
            return
        if display_projection is not None:
            assert isinstance(display_projection, ToolDisplayProjection)
        tool_id = _safe_string_id(tool_call_id)
        active = self._active_tools.get(tool_id)
        if active is None:
            return
        self._active_tools[tool_id] = replace(
            active,
            name=active.name if name is None else safe_text(name),
            arguments_summary=(
                active.arguments_summary
                if arguments is None
                else safe_summary(arguments)
            ),
            display_projection=(
                active.display_projection
                if display_projection is None
                else display_projection
            ),
            provider_name=(
                active.provider_name
                if provider_name is None
                else safe_text(provider_name)
            ),
            sequence=active.sequence if sequence is None else sequence,
            updated_at=active.updated_at if updated_at is None else updated_at,
        )

    def complete_tool(
        self,
        *,
        tool_call_id: object,
        status: ToolStatus = "completed",
        name: object | None = None,
        display_projection: ToolDisplayProjection | None = None,
        elapsed_seconds: float | None = None,
        sequence: int | None = None,
    ) -> None:
        """Move an active tool to completed history when retained."""
        if not self.display.show_tools:
            return
        if display_projection is not None:
            assert isinstance(display_projection, ToolDisplayProjection)
        tool_id = _safe_string_id(tool_call_id)
        tool_name = safe_text("tool" if name is None else name)
        active = self._active_tools.pop(tool_id, None)
        summary = CliToolExecutionSummarySnapshot(
            tool_call_id=tool_id,
            name=active.name if active and name is None else tool_name,
            arguments_summary=(
                None if active is None else active.arguments_summary
            ),
            display_projection=(
                display_projection
                if display_projection is not None
                else (None if active is None else active.display_projection)
            ),
            provider_name=None if active is None else active.provider_name,
            sequence=sequence if sequence is not None else _sequence(active),
            status=status,
            started_at=None if active is None else active.started_at,
            updated_at=None if active is None else active.updated_at,
            elapsed_seconds=elapsed_seconds,
        )
        self._completed_tools.append(summary)

    def add_tool_result(
        self,
        result: object,
        *,
        sequence: int | None = None,
        elapsed_seconds: float | None = None,
        display_projection: ToolDisplayProjection | None = None,
    ) -> None:
        """Add a safe tool result or error summary."""
        if self.retention.internal_tool_history_limit == 0:
            return
        if display_projection is not None:
            assert isinstance(display_projection, ToolDisplayProjection)
        call = value_from(result, "call") or result
        arguments = value_from(call, "arguments")
        result_value = (
            value_from(result, "message")
            if isinstance(result, ToolCallError)
            else value_from(result, "result")
        )
        self._tool_results.append(
            CliToolResultSummarySnapshot(
                tool_call_id=_safe_string_id(
                    value_from(call, "id") or value_from(result, "id")
                ),
                name=_tool_name(call),
                status=(
                    "error" if isinstance(result, ToolCallError) else "result"
                ),
                result_summary=safe_summary(result_value),
                arguments_count=_payload_size(arguments),
                display_projection=display_projection,
                sequence=sequence,
                elapsed_seconds=elapsed_seconds,
            )
        )

    def add_tool_result_summary(
        self,
        *,
        tool_call_id: object,
        name: object,
        status: ToolResultStatus,
        result: object,
        arguments_count: int,
        display_projection: ToolDisplayProjection | None = None,
        sequence: int | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        """Add a safe tool result summary from canonical stream data."""
        assert status in ("result", "error")
        assert isinstance(arguments_count, int)
        assert arguments_count >= 0
        if self.retention.internal_tool_history_limit == 0:
            return
        if display_projection is not None:
            assert isinstance(display_projection, ToolDisplayProjection)
        self._tool_results.append(
            CliToolResultSummarySnapshot(
                tool_call_id=_safe_string_id(tool_call_id),
                name=safe_text(name),
                status=status,
                result_summary=safe_summary(result),
                arguments_count=arguments_count,
                display_projection=display_projection,
                sequence=sequence,
                elapsed_seconds=elapsed_seconds,
            )
        )

    def add_tool_diagnostic(
        self,
        diagnostic: ToolCallDiagnostic,
        *,
        sequence: int | None = None,
    ) -> None:
        """Add a safe tool diagnostic summary."""
        assert isinstance(diagnostic, ToolCallDiagnostic)
        if self.retention.internal_tool_history_limit == 0:
            return
        self._tool_diagnostics.append(
            CliToolDiagnosticSummarySnapshot(
                diagnostic_id=_safe_string_id(diagnostic.id),
                tool_call_id=(
                    None
                    if diagnostic.call_id is None
                    else _safe_string_id(diagnostic.call_id)
                ),
                requested_name=(
                    None
                    if diagnostic.requested_name is None
                    else safe_text(diagnostic.requested_name)
                ),
                canonical_name=(
                    None
                    if diagnostic.canonical_name is None
                    else safe_text(diagnostic.canonical_name)
                ),
                status=diagnostic.status.value,
                code=diagnostic.code.value,
                stage=diagnostic.stage.value,
                message=safe_text(diagnostic.message),
                details_summary=(
                    safe_summary(diagnostic.details)
                    if diagnostic.details
                    else None
                ),
                retryable=diagnostic.retryable,
                sequence=sequence,
                duration_ms=(
                    None
                    if diagnostic.duration_ms is None
                    else float(diagnostic.duration_ms)
                ),
            )
        )

    def add_tool_event(
        self,
        event: object,
        *,
        tool_call_id: object | None = None,
        name: object | None = None,
        sequence: int | None = None,
    ) -> None:
        """Add a safe CLI-only tool event summary."""
        assert _has_event_display_fields(event)
        event_type = event_type_value(getattr(event, "type"))
        if not self.display.show_tools or not event_type.startswith("tool_"):
            return
        payload = getattr(event, "payload", None)
        self._tool_events.append(
            CliToolEventSummarySnapshot(
                event_type=event_type,
                tool_call_id=(
                    None
                    if tool_call_id is None
                    else _safe_string_id(tool_call_id)
                ),
                name=None if name is None else safe_text(name),
                payload_summary=(
                    None if payload is None else safe_summary(payload)
                ),
                observability_summary=_event_observability_summary(event),
                sequence=sequence,
                started=getattr(event, "started"),
                finished=getattr(event, "finished"),
                elapsed=getattr(event, "elapsed"),
            )
        )

    def add_tool_event_summary(
        self,
        *,
        event_type: object,
        tool_call_id: object | None = None,
        name: object | None = None,
        payload: object | None = None,
        observability: object | None = None,
        sequence: int | None = None,
        started: float | None = None,
        finished: float | None = None,
        elapsed: float | None = None,
    ) -> None:
        """Add a safe CLI-only tool event summary."""
        if not self.display.show_tools:
            return
        self._tool_events.append(
            CliToolEventSummarySnapshot(
                event_type=event_type_value(event_type),
                tool_call_id=(
                    None
                    if tool_call_id is None
                    else _safe_string_id(tool_call_id)
                ),
                name=None if name is None else safe_text(name),
                payload_summary=(
                    None if payload is None else safe_summary(payload)
                ),
                observability_summary=(
                    None
                    if observability is None
                    else safe_summary(observability)
                ),
                sequence=sequence,
                started=started,
                finished=finished,
                elapsed=elapsed,
            )
        )

    def add_active_model_continuation(
        self,
        *,
        model_continuation_id: object,
        sequence: int | None = None,
        started_at: float | None = None,
    ) -> None:
        """Add or replace an active model continuation summary."""
        if not self.display.show_tools:
            return
        continuation_id = _safe_string_id(model_continuation_id)
        self._active_model_continuations[continuation_id] = (
            CliModelContinuationSnapshot(
                model_continuation_id=continuation_id,
                sequence=sequence,
                started_at=started_at,
            )
        )

    def finish_model_continuation(
        self,
        *,
        model_continuation_id: object,
        updated_at: float | None = None,
    ) -> None:
        """Remove an active model continuation summary."""
        _ = updated_at
        continuation_id = _safe_string_id(model_continuation_id)
        self._active_model_continuations.pop(continuation_id, None)

    def remove_tool_events_for_tool_call(
        self,
        tool_call_id: object,
    ) -> int:
        """Remove CLI-only tool events superseded by canonical data."""
        tool_id = _safe_string_id(tool_call_id)
        return self._tool_events.remove_matching(
            lambda event: event.tool_call_id == tool_id
        )

    def add_event(
        self,
        event: object,
        *,
        sequence: int | None = None,
    ) -> None:
        """Add a safe non-tool event summary when event display is enabled."""
        assert _has_event_display_fields(event)
        event_type = event_type_value(getattr(event, "type"))
        if not self.display.show_events or event_type.startswith("tool_"):
            return
        payload = getattr(event, "payload", None)
        self._events.append(
            CliEventSummarySnapshot(
                event_type=event_type,
                payload_summary=(
                    None if payload is None else safe_summary(payload)
                ),
                observability_summary=_event_observability_summary(event),
                sequence=sequence,
                started=getattr(event, "started"),
                finished=getattr(event, "finished"),
                elapsed=getattr(event, "elapsed"),
            )
        )

    def add_event_summary(
        self,
        *,
        event_type: object,
        payload: object | None = None,
        observability: object | None = None,
        sequence: int | None = None,
        started: float | None = None,
        finished: float | None = None,
        elapsed: float | None = None,
    ) -> None:
        """Add a safe non-tool event summary without retaining payloads."""
        event_type_text = event_type_value(event_type)
        if not self.display.show_events or event_type_text.startswith("tool_"):
            return
        self._events.append(
            CliEventSummarySnapshot(
                event_type=event_type_text,
                payload_summary=(
                    None if payload is None else safe_summary(payload)
                ),
                observability_summary=(
                    None
                    if observability is None
                    else safe_summary(observability)
                ),
                sequence=sequence,
                started=started,
                finished=finished,
                elapsed=elapsed,
            )
        )

    def add_usage_summary(
        self,
        usage: object,
        *,
        sequence: int | None = None,
        kind: object | None = None,
        provider_family: object | None = None,
        provider_event_type: object | None = None,
    ) -> None:
        """Add a bounded safe usage summary."""
        if self.retention.usage_summary_history_limit == 0:
            return
        self._usage_summaries.append(
            CliUsageSummarySnapshot(
                sequence=sequence,
                kind=_primitive_value(kind),
                usage_summary=safe_summary(usage),
                provider_family=_primitive_value(provider_family),
                provider_event_type=_primitive_value(provider_event_type),
            )
        )

    def add_projection_summary(
        self,
        projection: StreamConsumerProjection,
    ) -> None:
        """Add safe summaries from a canonical consumer projection."""
        assert isinstance(projection, StreamConsumerProjection)
        if projection.usage is not None:
            self.add_usage_summary(
                projection.usage,
                sequence=projection.sequence,
                kind=projection.kind,
                provider_family=projection.provider_family,
                provider_event_type=projection.provider_event_type,
            )
        if self.retention.projection_metadata_history_limit == 0:
            return
        if projection.data is None and not projection.metadata:
            return
        self._projection_metadata_summaries.append(
            CliProjectionMetadataSummarySnapshot(
                sequence=projection.sequence,
                kind=_primitive_value(projection.kind),
                data_summary=(
                    None
                    if projection.data is None
                    else safe_summary(projection.data)
                ),
                metadata_summary=(
                    safe_summary(projection.metadata)
                    if projection.metadata
                    else None
                ),
                provider_family=_primitive_value(projection.provider_family),
                provider_event_type=_primitive_value(
                    projection.provider_event_type
                ),
            )
        )

    def snapshot(self) -> CliStreamSnapshot:
        """Materialize an immutable snapshot of current display state."""
        self._snapshots_built += 1
        answer_text = self._answer_text.materialize()
        reasoning_text = self._reasoning_text.materialize()
        tool_call_request_text = self._tool_call_request_text.materialize()
        completed_tools = self._completed_tools.snapshot()
        tool_results = self._tool_results.snapshot()
        tool_diagnostics = self._tool_diagnostics.snapshot()
        tool_events = self._tool_events.snapshot()
        active_model_continuations = tuple(
            self._active_model_continuations.values()
        )
        events = self._events.snapshot()
        display_tokens = self._display_tokens.snapshot()
        usage_summaries = self._usage_summaries.snapshot()
        projection_metadata = self._projection_metadata_summaries.snapshot()
        build_stats = self._build_stats(
            completed_tools=completed_tools,
            tool_results=tool_results,
            tool_diagnostics=tool_diagnostics,
            tool_events=tool_events,
            events=events,
            display_tokens=display_tokens,
            usage_summaries=usage_summaries,
            projection_metadata=projection_metadata,
        )
        return CliStreamSnapshot(
            display=self.display,
            retention=self.retention,
            token_counts=self._token_counts,
            timing=self._timing,
            terminal=self._terminal,
            answer_text=answer_text,
            reasoning_text=reasoning_text,
            tool_call_request_text=safe_tool_call_request_text(
                tool_call_request_text
            ),
            active_tools=tuple(self._active_tools.values()),
            completed_tools=completed_tools,
            tool_results=tool_results,
            tool_diagnostics=tool_diagnostics,
            tool_events=tool_events,
            active_model_continuations=active_model_continuations,
            events=events,
            display_tokens=display_tokens,
            usage_summaries=usage_summaries,
            projection_metadata_summaries=projection_metadata,
            build_stats=build_stats,
        )

    def _build_stats(
        self,
        *,
        completed_tools: tuple[CliToolExecutionSummarySnapshot, ...],
        tool_results: tuple[CliToolResultSummarySnapshot, ...],
        tool_diagnostics: tuple[CliToolDiagnosticSummarySnapshot, ...],
        tool_events: tuple[CliToolEventSummarySnapshot, ...],
        events: tuple[CliEventSummarySnapshot, ...],
        display_tokens: tuple[CliDisplayTokenSnapshot, ...],
        usage_summaries: tuple[CliUsageSummarySnapshot, ...],
        projection_metadata: tuple[CliProjectionMetadataSummarySnapshot, ...],
    ) -> CliStreamBuildStatsSnapshot:
        text_materializations = (
            self._answer_text.materialization_count
            + self._reasoning_text.materialization_count
            + self._tool_call_request_text.materialization_count
        )
        history_materializations = (
            self._completed_tools.materialization_count
            + self._tool_results.materialization_count
            + self._tool_diagnostics.materialization_count
            + self._tool_events.materialization_count
            + self._events.materialization_count
            + self._display_tokens.materialization_count
            + self._usage_summaries.materialization_count
            + self._projection_metadata_summaries.materialization_count
        )
        return CliStreamBuildStatsSnapshot(
            snapshots_built=self._snapshots_built,
            answer_chunks=self._answer_text.chunk_count,
            reasoning_chunks=self._reasoning_text.chunk_count,
            tool_call_request_chunks=self._tool_call_request_text.chunk_count,
            answer_characters=self._answer_text.character_count,
            reasoning_characters=self._reasoning_text.character_count,
            tool_call_request_characters=(
                self._tool_call_request_text.character_count
            ),
            text_materializations=text_materializations,
            history_materializations=history_materializations,
            active_tools=len(self._active_tools),
            retained_completed_tools=len(completed_tools),
            retained_tool_results=len(tool_results),
            retained_tool_diagnostics=len(tool_diagnostics),
            retained_tool_events=len(tool_events),
            retained_events=len(events),
            retained_display_tokens=len(display_tokens),
            retained_usage_summaries=len(usage_summaries),
            retained_projection_metadata_summaries=len(projection_metadata),
            dropped_completed_tools=self._completed_tools.dropped_count,
            dropped_tool_results=self._tool_results.dropped_count,
            dropped_tool_diagnostics=self._tool_diagnostics.dropped_count,
            dropped_tool_events=self._tool_events.dropped_count,
            dropped_events=self._events.dropped_count,
            dropped_display_tokens=self._display_tokens.dropped_count,
            dropped_usage_summaries=self._usage_summaries.dropped_count,
            dropped_projection_metadata_summaries=(
                self._projection_metadata_summaries.dropped_count
            ),
        )


def _payload_size(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping | list | tuple | set | frozenset):
        return len(value)
    return 1


def _coalesce(
    value: CoalescedValue | None,
    fallback: CoalescedValue | None,
) -> CoalescedValue | None:
    return fallback if value is None else value


def _primitive_value(value: object | None) -> str | None:
    if value is None:
        return None
    primitive = getattr(value, "value", value)
    return safe_text(primitive)


def _safe_id(value: object | None) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return safe_text(value)


def _safe_string_id(value: object | None) -> str:
    safe_id = _safe_id(value)
    return "?" if safe_id is None else str(safe_id)


def _safe_probability(value: object | None) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _has_event_display_fields(value: object) -> bool:
    return (
        hasattr(value, "type")
        and hasattr(value, "payload")
        and hasattr(value, "observability")
        and hasattr(value, "started")
        and hasattr(value, "finished")
        and hasattr(value, "elapsed")
    )


def _event_observability_summary(value: object) -> str:
    observability = getattr(value, "observability")
    to_dict = getattr(observability, "to_dict", None)
    return safe_summary(to_dict() if callable(to_dict) else observability)


def _metadata_int(value: object | None) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _metadata_probability(value: object | None) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _metadata_string(value: object | None) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _metadata_token_candidate_snapshots(
    value: object | None,
) -> list[CliDisplayTokenCandidateSnapshot]:
    if not isinstance(value, list | tuple):
        return []
    candidates: list[CliDisplayTokenCandidateSnapshot] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        token = item.get("token")
        if not isinstance(token, str):
            continue
        candidates.append(
            CliDisplayTokenCandidateSnapshot(
                token_id=_safe_id(_metadata_int(item.get("token_id"))),
                text=safe_text(token),
                probability=_metadata_probability(item.get("probability")),
            )
        )
    return candidates


def _tool_name(value: object) -> str:
    name = (
        value_from(value, "name")
        or value_from(value, "canonical_name")
        or value_from(value, "requested_name")
        or "tool"
    )
    return safe_text(name)


def _sequence(
    summary: CliToolExecutionSummarySnapshot | None,
) -> int | None:
    return None if summary is None else summary.sequence
