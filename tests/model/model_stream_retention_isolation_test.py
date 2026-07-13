from collections.abc import Callable
from unittest import TestCase
from unittest.mock import patch

from avalan.model.stream import (
    REASONING_SEGMENT_BOUNDARY_METADATA_KEY,
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamReasoningSegmentStatus,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamVisibility,
    stream_channel_for_kind,
)
from avalan.types import LooseJsonValue


def _item(
    kind: StreamItemKind,
    sequence: int,
    *,
    stream_session_id: str = "stream-1",
    run_id: str = "run-1",
    turn_id: str = "turn-1",
    data: LooseJsonValue | None = None,
    text_delta: str | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
    correlation: StreamItemCorrelation | None = None,
    reasoning_representation: StreamReasoningRepresentation | None = None,
    segment_instance_ordinal: int | None = None,
    metadata: dict[str, LooseJsonValue] | None = None,
    usage: LooseJsonValue | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        text_delta=text_delta,
        data=data,
        usage=usage,
        terminal_outcome=terminal_outcome,
        visibility=visibility,
        correlation=correlation or StreamItemCorrelation(),
        reasoning_representation=reasoning_representation,
        segment_instance_ordinal=segment_instance_ordinal,
        metadata=metadata or {},
    )


def _reasoning_item(
    sequence: int,
    text: str,
    *,
    ordinal: int,
    representation: StreamReasoningRepresentation = (
        StreamReasoningRepresentation.SUMMARY
    ),
    provider_item_id: str | None = "reasoning-1",
    output_index: int | None = 0,
    summary_index: int | None = 0,
    continuation_id: str | None = None,
    follows_completion: bool = False,
) -> CanonicalStreamItem:
    return _item(
        StreamItemKind.REASONING_DELTA,
        sequence,
        text_delta=text,
        visibility=StreamVisibility.PRIVATE,
        correlation=StreamItemCorrelation(
            protocol_item_id=provider_item_id,
            provider_output_index=output_index,
            provider_summary_index=summary_index,
            model_continuation_id=continuation_id,
        ),
        reasoning_representation=representation,
        segment_instance_ordinal=ordinal,
        metadata=(
            {REASONING_SEGMENT_BOUNDARY_METADATA_KEY: "completed"}
            if follows_completion
            else {}
        ),
    )


def _terminal_item(
    sequence: int,
    outcome: StreamTerminalOutcome = StreamTerminalOutcome.COMPLETED,
) -> CanonicalStreamItem:
    kind = {
        StreamTerminalOutcome.COMPLETED: StreamItemKind.STREAM_COMPLETED,
        StreamTerminalOutcome.ERRORED: StreamItemKind.STREAM_ERRORED,
        StreamTerminalOutcome.CANCELLED: StreamItemKind.STREAM_CANCELLED,
    }[outcome]
    return _item(
        kind,
        sequence,
        terminal_outcome=outcome,
        usage={} if outcome is StreamTerminalOutcome.COMPLETED else None,
    )


def _reasoning_accumulator(
    reasoning_items: tuple[CanonicalStreamItem, ...],
    *,
    retention_policy: StreamRetentionPolicy | None = None,
    outcome: StreamTerminalOutcome = StreamTerminalOutcome.COMPLETED,
) -> CanonicalStreamAccumulator:
    accumulator = CanonicalStreamAccumulator(retention_policy=retention_policy)
    accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
    for item in reasoning_items:
        accumulator.add(item)
    next_sequence = len(reasoning_items) + 1
    accumulator.add(_item(StreamItemKind.REASONING_DONE, next_sequence))
    accumulator.add(_terminal_item(next_sequence + 1, outcome))
    accumulator.validate_complete()
    return accumulator


class StreamRetentionIsolationTestCase(TestCase):
    def test_responses_reasoning_item_retention_defaults(self) -> None:
        policy = StreamRetentionPolicy()

        self.assertEqual(policy.responses_reasoning_item_segment_limit, 1024)
        self.assertEqual(
            policy.responses_reasoning_item_character_limit,
            262144,
        )
        self.assertEqual(
            policy.responses_reasoning_item_text_byte_limit,
            1048576,
        )

    def test_responses_reasoning_item_retention_accepts_custom_limits(
        self,
    ) -> None:
        policy = StreamRetentionPolicy(
            responses_reasoning_item_segment_limit=1,
            responses_reasoning_item_character_limit=2,
            responses_reasoning_item_text_byte_limit=3,
        )

        self.assertEqual(policy.responses_reasoning_item_segment_limit, 1)
        self.assertEqual(policy.responses_reasoning_item_character_limit, 2)
        self.assertEqual(policy.responses_reasoning_item_text_byte_limit, 3)

    def test_responses_reasoning_item_retention_rejects_invalid_limits(
        self,
    ) -> None:
        field_names = (
            "responses_reasoning_item_segment_limit",
            "responses_reasoning_item_character_limit",
            "responses_reasoning_item_text_byte_limit",
        )
        for field_name in field_names:
            for invalid_value in (-1, True):
                with self.subTest(
                    field_name=field_name,
                    invalid_value=invalid_value,
                ):
                    with self.assertRaises(AssertionError):
                        StreamRetentionPolicy(
                            **{field_name: invalid_value}  # type: ignore[arg-type]
                        )

    def test_cli_reasoning_retention_limits_are_independent_and_validated(
        self,
    ) -> None:
        policy = StreamRetentionPolicy(
            reasoning_segment_limit=1,
            reasoning_character_limit=2,
            reasoning_text_byte_limit=3,
            cli_reasoning_segment_limit=4,
            cli_reasoning_character_limit=5,
            cli_reasoning_text_byte_limit=6,
        )

        self.assertEqual(policy.reasoning_segment_limit, 1)
        self.assertEqual(policy.reasoning_character_limit, 2)
        self.assertEqual(policy.reasoning_text_byte_limit, 3)
        self.assertEqual(policy.cli_reasoning_segment_limit, 4)
        self.assertEqual(policy.cli_reasoning_character_limit, 5)
        self.assertEqual(policy.cli_reasoning_text_byte_limit, 6)

        invalid_factories: tuple[Callable[[], StreamRetentionPolicy], ...] = (
            lambda: StreamRetentionPolicy(cli_reasoning_segment_limit=-1),
            lambda: StreamRetentionPolicy(cli_reasoning_character_limit=-1),
            lambda: StreamRetentionPolicy(cli_reasoning_text_byte_limit=-1),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(AssertionError):
                    factory()

    def test_accumulator_instances_do_not_share_retained_histories(
        self,
    ) -> None:
        first_policy = StreamRetentionPolicy(
            accumulator_item_limit=2,
            replay_history_item_limit=1,
            flow_history_item_limit=1,
            metrics_history_item_limit=1,
        )
        first = CanonicalStreamAccumulator(retention_policy=first_policy)
        second = CanonicalStreamAccumulator()
        first_items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.FLOW_EVENT,
                1,
                data={"node": "first"},
            ),
            _item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                2,
                text_delta="first diagnostic",
                visibility=StreamVisibility.DIAGNOSTIC,
            ),
            _item(
                StreamItemKind.STREAM_ERRORED,
                3,
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        )

        first.add_many(first_items)

        self.assertEqual(first.items, first_items[-2:])
        self.assertEqual(first.flow_items, (first_items[1],))
        self.assertEqual(first.diagnostics, (first_items[2],))
        self.assertEqual(first.control_items, (first_items[3],))
        self.assertEqual(second.items, ())
        self.assertEqual(second.flow_items, ())
        self.assertEqual(second.diagnostics, ())
        self.assertEqual(second.control_items, ())

        second_items = (
            _item(
                StreamItemKind.STREAM_STARTED,
                0,
                stream_session_id="stream-2",
                run_id="run-2",
                turn_id="turn-2",
            ),
            _item(
                StreamItemKind.STREAM_ERRORED,
                1,
                stream_session_id="stream-2",
                run_id="run-2",
                turn_id="turn-2",
                terminal_outcome=StreamTerminalOutcome.ERRORED,
            ),
        )
        second.add_many(second_items)

        self.assertEqual(first.items, first_items[-2:])
        self.assertEqual(first.flow_items, (first_items[1],))
        self.assertEqual(first.diagnostics, (first_items[2],))
        self.assertEqual(second.items, second_items)
        self.assertEqual(second.control_items, second_items)
        self.assertIs(
            first.terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )
        self.assertIs(
            second.terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )


def test_summary_parts_have_readable_boundaries() -> None:
    first = _reasoning_item(1, "first", ordinal=0, summary_index=0)
    second = _reasoning_item(
        2,
        "second",
        ordinal=1,
        summary_index=1,
        follows_completion=True,
    )
    accumulator = _reasoning_accumulator((first, second))

    assert (
        "".join(item.text_delta or "" for item in (first, second))
        == "firstsecond"
    )
    assert accumulator.reasoning_text == "first\n\nsecond"
    assert [segment.text for segment in accumulator.reasoning_segments] == [
        "first",
        "second",
    ]


def test_flat_reasoning_preserves_part_order() -> None:
    accumulator = _reasoning_accumulator(
        (
            _reasoning_item(1, "two", ordinal=0, summary_index=2),
            _reasoning_item(
                2,
                "seven",
                ordinal=1,
                summary_index=7,
                follows_completion=True,
            ),
            _reasoning_item(
                3,
                "zero",
                ordinal=2,
                summary_index=0,
                follows_completion=True,
            ),
        )
    )

    assert accumulator.reasoning_text == "two\n\nseven\n\nzero"
    assert [
        segment.summary_index for segment in accumulator.reasoning_segments
    ] == [
        2,
        7,
        0,
    ]


def test_structured_reasoning_preserves_representation() -> None:
    accumulator = _reasoning_accumulator(
        (
            _reasoning_item(
                1,
                "native",
                ordinal=0,
                representation=StreamReasoningRepresentation.NATIVE_TEXT,
                provider_item_id=None,
                output_index=None,
                summary_index=None,
                continuation_id="continuation-1",
            ),
            _reasoning_item(
                2,
                "summary",
                ordinal=1,
                provider_item_id="reasoning-2",
                output_index=3,
                summary_index=5,
                continuation_id="continuation-1",
                follows_completion=True,
            ),
        )
    )

    native, summary = accumulator.reasoning_segments
    assert native.representation is StreamReasoningRepresentation.NATIVE_TEXT
    assert native.provider_item_id is None
    assert native.output_index is None
    assert native.summary_index is None
    assert native.continuation_id == "continuation-1"
    assert summary.representation is StreamReasoningRepresentation.SUMMARY
    assert summary.provider_item_id == "reasoning-2"
    assert summary.output_index == 3
    assert summary.summary_index == 5
    assert summary.continuation_id == "continuation-1"
    assert [
        segment.segment_instance_ordinal for segment in (native, summary)
    ] == [
        0,
        1,
    ]


def test_summary_obeys_reasoning_retention() -> None:
    policy = StreamRetentionPolicy(
        reasoning_segment_limit=2,
        reasoning_character_limit=100,
        reasoning_text_byte_limit=100,
    )
    accumulator = _reasoning_accumulator(
        (
            _reasoning_item(1, "a", ordinal=0, summary_index=0),
            _reasoning_item(
                2,
                "b",
                ordinal=1,
                summary_index=1,
                follows_completion=True,
            ),
            _reasoning_item(
                3,
                "c",
                ordinal=2,
                summary_index=2,
                follows_completion=True,
            ),
        ),
        retention_policy=policy,
    )

    assert accumulator.reasoning_text == "b\n\nc"
    assert [segment.text for segment in accumulator.reasoning_segments] == [
        "b",
        "c",
    ]
    assert accumulator.reasoning_truncation.dropped_segments == 1
    assert accumulator.reasoning_truncation.dropped_characters == 3
    assert accumulator.reasoning_truncation.dropped_utf8_bytes == 3
    assert not accumulator.reasoning_truncation.leading_segment_partial


def test_multipart_summary_is_ordered_and_readable() -> None:
    accumulator = _reasoning_accumulator(
        (
            _reasoning_item(1, "first\n", ordinal=0, summary_index=4),
            _reasoning_item(
                2,
                "\nsecond",
                ordinal=1,
                summary_index=1,
                follows_completion=True,
            ),
            _reasoning_item(
                3,
                " \n third",
                ordinal=2,
                summary_index=9,
                follows_completion=True,
            ),
        )
    )

    assert accumulator.reasoning_text == "first\n\nsecond\n \n third"
    assert [
        segment.summary_index for segment in accumulator.reasoning_segments
    ] == [
        4,
        1,
        9,
    ]


def test_reasoning_merges_only_contiguous_exact_segment_identity() -> None:
    accumulator = _reasoning_accumulator(
        (
            _reasoning_item(1, "a", ordinal=0, summary_index=0),
            _reasoning_item(2, "b", ordinal=0, summary_index=0),
            _reasoning_item(
                3,
                "c",
                ordinal=1,
                summary_index=0,
                follows_completion=True,
            ),
            _reasoning_item(
                4,
                "d",
                ordinal=2,
                summary_index=1,
                follows_completion=True,
            ),
        )
    )

    assert [segment.text for segment in accumulator.reasoning_segments] == [
        "ab",
        "c",
        "d",
    ]
    assert accumulator.reasoning_text == "ab\n\nc\n\nd"


def test_reasoning_terminal_finalizes_status_without_boundary_leak() -> None:
    items = (
        _reasoning_item(1, "first", ordinal=0, summary_index=0),
        _reasoning_item(
            2,
            "last",
            ordinal=1,
            summary_index=1,
            follows_completion=True,
        ),
    )
    completed = _reasoning_accumulator(items)
    errored = _reasoning_accumulator(
        items, outcome=StreamTerminalOutcome.ERRORED
    )
    cancelled = _reasoning_accumulator(
        items, outcome=StreamTerminalOutcome.CANCELLED
    )

    assert all(segment.completed for segment in completed.reasoning_segments)
    assert all(
        segment.status is StreamReasoningSegmentStatus.COMPLETED
        for segment in completed.reasoning_segments
    )
    for accumulator, outcome in (
        (errored, StreamTerminalOutcome.ERRORED),
        (cancelled, StreamTerminalOutcome.CANCELLED),
    ):
        assert all(
            not segment.completed for segment in accumulator.reasoning_segments
        )
        assert all(
            segment.status is StreamReasoningSegmentStatus.INCOMPLETE
            for segment in accumulator.reasoning_segments
        )
        assert all(
            segment.terminal_outcome is outcome
            for segment in accumulator.reasoning_segments
        )


def test_reasoning_segment_is_in_progress_before_terminal() -> None:
    accumulator = CanonicalStreamAccumulator()
    accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
    accumulator.add(_reasoning_item(1, "working", ordinal=0))

    segment = accumulator.reasoning_segments[0]

    assert not segment.completed
    assert segment.status is StreamReasoningSegmentStatus.IN_PROGRESS
    assert segment.terminal_outcome is None


def test_zero_segment_retention_discards_active_reasoning() -> None:
    accumulator = CanonicalStreamAccumulator(
        retention_policy=StreamRetentionPolicy(reasoning_segment_limit=0)
    )
    accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
    accumulator.add(_reasoning_item(1, "private", ordinal=0))

    assert accumulator.reasoning_text == ""
    assert accumulator.reasoning_segments == ()
    assert accumulator.reasoning_truncation.dropped_characters == 7

    accumulator.add(_item(StreamItemKind.REASONING_DONE, 2))
    accumulator.add(_terminal_item(3))
    accumulator.validate_complete()

    assert accumulator.reasoning_segments == ()
    assert accumulator.reasoning_truncation.dropped_segments == 1


def test_empty_trimmed_segment_removes_preceding_separator() -> None:
    accumulator = CanonicalStreamAccumulator()
    accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
    accumulator.add(_reasoning_item(1, "first", ordinal=0))
    accumulator.add(
        _reasoning_item(
            2,
            "last",
            ordinal=1,
            summary_index=1,
            follows_completion=True,
        )
    )
    owner = accumulator._reasoning
    active = owner._active
    assert active is not None

    owner._trim_all_active_text(active)
    owner._close_active()

    assert accumulator.reasoning_text == "first"
    assert accumulator.reasoning_truncation.dropped_segments == 1
    assert accumulator.reasoning_truncation.dropped_characters == 6
    assert accumulator.reasoning_truncation.dropped_utf8_bytes == 6


def test_closed_segment_is_removed_when_tightened_limit_trims_all() -> None:
    accumulator = CanonicalStreamAccumulator()
    accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
    accumulator.add(_reasoning_item(1, "private", ordinal=0))
    accumulator.add(_item(StreamItemKind.REASONING_DONE, 2))
    owner = accumulator._reasoning

    owner._character_limit = 0
    owner._enforce_limits()

    assert accumulator.reasoning_text == ""
    assert accumulator.reasoning_segments == ()
    assert accumulator.reasoning_truncation.dropped_segments == 1
    assert accumulator.reasoning_truncation.dropped_characters == 7


def test_reasoning_retention_stops_when_no_trim_progress_is_possible() -> None:
    accumulator = CanonicalStreamAccumulator()
    accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
    accumulator.add(_reasoning_item(1, "private", ordinal=0))
    owner = accumulator._reasoning

    with patch.object(owner, "_over_text_limit", return_value=True) as over:
        owner._enforce_limits()

    over.assert_called_once_with()
    assert accumulator.reasoning_text == "private"


def test_reasoning_character_limit_retains_utf8_safe_suffix() -> None:
    accumulator = _reasoning_accumulator(
        (_reasoning_item(1, "abcdef", ordinal=0),),
        retention_policy=StreamRetentionPolicy(
            reasoning_segment_limit=10,
            reasoning_character_limit=4,
            reasoning_text_byte_limit=100,
        ),
    )

    assert accumulator.reasoning_text == "cdef"
    assert accumulator.reasoning_segments[0].text == "cdef"
    assert accumulator.reasoning_truncation.dropped_segments == 0
    assert accumulator.reasoning_truncation.dropped_characters == 2
    assert accumulator.reasoning_truncation.dropped_utf8_bytes == 2
    assert accumulator.reasoning_truncation.leading_segment_partial


def test_reasoning_utf8_byte_limit_never_splits_character() -> None:
    accumulator = _reasoning_accumulator(
        (_reasoning_item(1, "aé🙂", ordinal=0),),
        retention_policy=StreamRetentionPolicy(
            reasoning_segment_limit=10,
            reasoning_character_limit=100,
            reasoning_text_byte_limit=5,
        ),
    )

    assert accumulator.reasoning_text == "🙂"
    assert accumulator.retained_reasoning_characters == 1
    assert accumulator.retained_reasoning_utf8_bytes == 4
    assert accumulator.reasoning_truncation.dropped_characters == 2
    assert accumulator.reasoning_truncation.dropped_utf8_bytes == 3
    assert accumulator.reasoning_truncation.leading_segment_partial


def test_reasoning_separator_is_charged_before_following_segment() -> None:
    accumulator = _reasoning_accumulator(
        (
            _reasoning_item(1, "old", ordinal=0, summary_index=0),
            _reasoning_item(
                2,
                "new",
                ordinal=1,
                summary_index=1,
                follows_completion=True,
            ),
        ),
        retention_policy=StreamRetentionPolicy(
            reasoning_segment_limit=10,
            reasoning_character_limit=5,
            reasoning_text_byte_limit=100,
        ),
    )

    assert accumulator.reasoning_text == "new"
    assert accumulator.reasoning_truncation.dropped_segments == 1
    assert accumulator.reasoning_truncation.dropped_characters == 5
    assert accumulator.reasoning_truncation.dropped_utf8_bytes == 5
    assert not accumulator.reasoning_truncation.leading_segment_partial


def test_reasoning_history_is_independent_from_canonical_item_retention() -> (
    None
):
    policy = StreamRetentionPolicy(
        accumulator_item_limit=2,
        reasoning_segment_limit=10,
        reasoning_character_limit=100,
        reasoning_text_byte_limit=100,
    )
    accumulator = _reasoning_accumulator(
        (
            _reasoning_item(1, "a", ordinal=0),
            _reasoning_item(2, "b", ordinal=0),
            _reasoning_item(3, "c", ordinal=0),
        ),
        retention_policy=policy,
    )

    assert len(accumulator.items) == 2
    assert accumulator.reasoning_text == "abc"
    assert accumulator.reasoning_segments[0].text == "abc"


def test_reasoning_accumulator_instances_isolate_concurrent_sentinels() -> (
    None
):
    accumulators = [CanonicalStreamAccumulator() for _ in range(16)]
    for accumulator in accumulators:
        accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
    for index, accumulator in enumerate(accumulators):
        accumulator.add(_reasoning_item(1, f"<sentinel-{index}>", ordinal=0))
    for accumulator in accumulators:
        accumulator.add(_item(StreamItemKind.REASONING_DONE, 2))
        accumulator.add(_terminal_item(3))

    for index, accumulator in enumerate(accumulators):
        assert accumulator.reasoning_text == f"<sentinel-{index}>"
        assert all(
            f"<sentinel-{other}>" not in accumulator.reasoning_text
            for other in range(len(accumulators))
            if other != index
        )


def test_reasoning_flat_view_has_no_second_retained_history() -> None:
    accumulator = _reasoning_accumulator(
        (_reasoning_item(1, "private", ordinal=0),)
    )

    assert not hasattr(accumulator, "_reasoning_text")
    assert accumulator.reasoning_segments[0].text == "private"
    assert accumulator.reasoning_text == "private"
    assert accumulator.reasoning_text == "private"


def test_reasoning_character_retention_limit_minus_equal_plus_one() -> None:
    expected = (
        ("abc", "abc", 0, False),
        ("abcd", "abcd", 0, False),
        ("abcde", "bcde", 1, True),
    )
    for source, retained, dropped, partial in expected:
        accumulator = _reasoning_accumulator(
            (_reasoning_item(1, source, ordinal=0),),
            retention_policy=StreamRetentionPolicy(
                reasoning_segment_limit=10,
                reasoning_character_limit=4,
                reasoning_text_byte_limit=100,
            ),
        )

        assert accumulator.reasoning_text == retained
        assert accumulator.retained_reasoning_characters == len(retained)
        assert accumulator.reasoning_truncation.dropped_characters == dropped
        assert (
            accumulator.reasoning_truncation.leading_segment_partial is partial
        )


def test_reasoning_utf8_retention_limit_minus_equal_plus_one() -> None:
    expected = (
        ("aé", "aé", 0, 0, False),
        ("éé", "éé", 0, 0, False),
        ("aéé", "éé", 1, 1, True),
    )
    for (
        source,
        retained,
        dropped_characters,
        dropped_bytes,
        partial,
    ) in expected:
        accumulator = _reasoning_accumulator(
            (_reasoning_item(1, source, ordinal=0),),
            retention_policy=StreamRetentionPolicy(
                reasoning_segment_limit=10,
                reasoning_character_limit=100,
                reasoning_text_byte_limit=4,
            ),
        )

        assert accumulator.reasoning_text == retained
        assert accumulator.retained_reasoning_utf8_bytes == len(
            retained.encode("utf-8")
        )
        assert (
            accumulator.reasoning_truncation.dropped_characters
            == dropped_characters
        )
        assert (
            accumulator.reasoning_truncation.dropped_utf8_bytes
            == dropped_bytes
        )
        assert (
            accumulator.reasoning_truncation.leading_segment_partial is partial
        )


def test_reasoning_segment_retention_limit_minus_equal_plus_one() -> None:
    for count, retained, dropped_segments in (
        (1, "a", 0),
        (2, "a\n\nb", 0),
        (3, "b\n\nc", 1),
    ):
        items = tuple(
            _reasoning_item(
                sequence,
                chr(ord("a") + sequence - 1),
                ordinal=sequence - 1,
                summary_index=sequence - 1,
                follows_completion=sequence > 1,
            )
            for sequence in range(1, count + 1)
        )
        accumulator = _reasoning_accumulator(
            items,
            retention_policy=StreamRetentionPolicy(
                reasoning_segment_limit=2,
                reasoning_character_limit=100,
                reasoning_text_byte_limit=100,
            ),
        )

        assert accumulator.reasoning_text == retained
        assert len(accumulator.reasoning_segments) == min(count, 2)
        assert (
            accumulator.reasoning_truncation.dropped_segments
            == dropped_segments
        )


def test_reasoning_separator_budget_limit_minus_equal_plus_one() -> None:
    items = (
        _reasoning_item(1, "a", ordinal=0, summary_index=0),
        _reasoning_item(
            2,
            "b",
            ordinal=1,
            summary_index=1,
            follows_completion=True,
        ),
    )
    for limit, retained, dropped_segments, dropped_characters in (
        (3, "b", 1, 3),
        (4, "a\n\nb", 0, 0),
        (5, "a\n\nb", 0, 0),
    ):
        accumulator = _reasoning_accumulator(
            items,
            retention_policy=StreamRetentionPolicy(
                reasoning_segment_limit=10,
                reasoning_character_limit=limit,
                reasoning_text_byte_limit=limit,
            ),
        )

        assert "".join(item.text_delta or "" for item in items) == "ab"
        assert accumulator.reasoning_text == retained
        assert (
            accumulator.reasoning_truncation.dropped_segments
            == dropped_segments
        )
        assert (
            accumulator.reasoning_truncation.dropped_characters
            == dropped_characters
        )
        assert (
            accumulator.reasoning_truncation.dropped_utf8_bytes
            == dropped_characters
        )


def test_canonical_item_retention_limit_minus_equal_plus_one() -> None:
    items = (
        _item(StreamItemKind.STREAM_STARTED, 0),
        _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="answer"),
        _item(StreamItemKind.ANSWER_DONE, 2),
    )
    for count in (1, 2, 3):
        accumulator = CanonicalStreamAccumulator(
            retention_policy=StreamRetentionPolicy(accumulator_item_limit=2)
        )

        accumulator.add_many(items[:count])

        assert accumulator.items == items[max(0, count - 2) : count]


def test_large_unicode_reasoning_owner_work_and_storage_are_bounded() -> None:
    delta_count = 8192
    character_limit = 1024
    delta = "🙂"
    accumulator = CanonicalStreamAccumulator(
        retention_policy=StreamRetentionPolicy(
            accumulator_item_limit=64,
            reasoning_segment_limit=4,
            reasoning_character_limit=character_limit,
            reasoning_text_byte_limit=character_limit * 4,
        )
    )
    accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
    source_reads = 0
    for sequence in range(1, delta_count + 1):
        source_reads += 1
        accumulator.add(
            _reasoning_item(sequence, delta, ordinal=0, summary_index=0)
        )

    owner = accumulator._reasoning
    assert source_reads == delta_count
    assert len(owner._segments) == 1
    retained = owner._segments[0]
    assert retained.characters == character_limit
    assert retained.utf8_bytes == character_limit * 4
    assert len(retained.chunks) == character_limit
    assert retained._materialized_text is None

    first_flat_view = accumulator.reasoning_text
    second_flat_view = accumulator.reasoning_text

    assert first_flat_view == delta * character_limit
    assert second_flat_view == first_flat_view
    assert len(retained.chunks) == 1
    assert retained._materialized_text == first_flat_view
    assert accumulator.retained_reasoning_characters == character_limit
    assert accumulator.retained_reasoning_utf8_bytes == character_limit * 4
    assert (
        accumulator.reasoning_truncation.dropped_characters
        == delta_count - character_limit
    )
    assert (
        accumulator.reasoning_truncation.dropped_utf8_bytes
        == (delta_count - character_limit) * 4
    )
    assert accumulator.reasoning_truncation.dropped_segments == 0
    assert accumulator.reasoning_truncation.leading_segment_partial
    assert not hasattr(accumulator, "_reasoning_text")

    accumulator.add(_item(StreamItemKind.REASONING_DONE, delta_count + 1))
    accumulator.add(_terminal_item(delta_count + 2))
    accumulator.validate_complete()
    assert accumulator.reasoning_segments[0].completed
