from unittest import TestCase

from avalan.model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamItemKind,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamVisibility,
    stream_channel_for_kind,
)


def _item(
    kind: StreamItemKind,
    sequence: int,
    *,
    stream_session_id: str = "stream-1",
    run_id: str = "run-1",
    turn_id: str = "turn-1",
    data: dict[str, str] | None = None,
    text_delta: str | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
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
        terminal_outcome=terminal_outcome,
        visibility=visibility,
    )


class StreamRetentionIsolationTestCase(TestCase):
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
