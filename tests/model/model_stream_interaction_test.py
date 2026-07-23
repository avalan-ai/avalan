"""Exercise canonical stream suspension for structured interaction."""

from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast
from unittest import TestCase, main

from avalan.event import project_observer_id, validate_observer_id
from avalan.interaction import (
    AgentId,
    BranchId,
    ContinuationId,
    InputRequestId,
    InputValidationError,
    TaskId,
)
from avalan.model import stream as stream_module
from avalan.model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProjectionState,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamReasoningSegmentStatus,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    canonical_item_from_consumer_projection,
    project_canonical_stream_item,
    stream_observability_payload,
    validate_canonical_stream_items,
)
from avalan.types import LooseJsonValue

_OBSERVER_ID_SENTINEL = "PRIVATE_PROMPT_SENTINEL"
_OBSERVER_ID_ATTACKS = (
    f"{_OBSERVER_ID_SENTINEL}\nanswer",
    f"{_OBSERVER_ID_SENTINEL} answer",
    f"{_OBSERVER_ID_SENTINEL}\tanswer",
    f"{_OBSERVER_ID_SENTINEL}\x1banswer",
    f"{_OBSERVER_ID_SENTINEL}\u202eanswer",
    f"{_OBSERVER_ID_SENTINEL}\\nanswer",
    f"<{_OBSERVER_ID_SENTINEL}>answer",
)


def _correlation(
    *,
    request_id: str = "request-1",
    continuation_id: str = "continuation-1",
    agent_id: str = "agent-1",
    branch_id: str = "branch-1",
) -> StreamItemCorrelation:
    return StreamItemCorrelation(
        request_id=InputRequestId(request_id),
        continuation_id=ContinuationId(continuation_id),
        task_id=TaskId("task-1"),
        agent_id=AgentId(agent_id),
        branch_id=BranchId(branch_id),
        parent_branch_id=BranchId("parent-branch-1"),
    )


def _item(
    kind: StreamItemKind,
    sequence: int,
    *,
    correlation: StreamItemCorrelation | None = None,
    text_delta: str | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    reasoning_representation: StreamReasoningRepresentation | None = None,
    segment_instance_ordinal: int | None = None,
    usage: LooseJsonValue | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=sequence,
        kind=kind,
        channel=(
            StreamChannel.INTERACTION
            if kind.value.startswith("interaction.")
            else (
                StreamChannel.CONTROL
                if kind.value.startswith("stream.")
                else StreamChannel.REASONING
            )
        ),
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        usage=usage,
        terminal_outcome=terminal_outcome,
        visibility=(
            StreamVisibility.PRIVATE
            if kind is StreamItemKind.REASONING_DELTA
            else StreamVisibility.PUBLIC
        ),
        reasoning_representation=reasoning_representation,
        segment_instance_ordinal=segment_instance_ordinal,
    )


class ModelStreamInteractionTestCase(TestCase):
    def _assert_rejected_by_all_state_paths(
        self,
        items: tuple[CanonicalStreamItem, ...],
        message: str,
    ) -> tuple[str, str, str]:
        with self.assertRaisesRegex(
            StreamValidationError,
            message,
        ) as validation_error:
            validate_canonical_stream_items(items)
        with self.assertRaisesRegex(
            StreamValidationError,
            message,
        ) as accumulator_error:
            CanonicalStreamAccumulator().add_many(items)
        projection = StreamProjectionState(
            stream_session_id="stream-1",
            run_id="run-1",
            turn_id="turn-1",
        )
        with self.assertRaisesRegex(
            StreamValidationError,
            message,
        ) as projection_error:
            for item in items:
                projection.project(
                    item,
                    item.sequence,
                    unsupported_message="unsupported",
                )
        return (
            str(validation_error.exception),
            str(accumulator_error.exception),
            str(projection_error.exception),
        )

    def test_every_interaction_stream_id_rejects_observer_smuggling(
        self,
    ) -> None:
        base = _item(
            StreamItemKind.INTERACTION_PENDING,
            1,
            correlation=_correlation(),
        )
        item_fields = ("stream_session_id", "run_id", "turn_id")
        correlation_fields = (
            "request_id",
            "continuation_id",
            "task_id",
            "agent_id",
            "branch_id",
            "parent_branch_id",
        )
        for field_name in (*item_fields, *correlation_fields):
            for attack in _OBSERVER_ID_ATTACKS:
                with self.subTest(field_name=field_name, attack=attack):
                    try:
                        if field_name in item_fields:
                            canonical = replace(
                                base,
                                **cast(Any, {field_name: attack}),
                            )
                        else:
                            canonical = replace(
                                base,
                                correlation=replace(
                                    base.correlation,
                                    **cast(Any, {field_name: attack}),
                                ),
                            )
                    except InputValidationError as raised:
                        self.assertNotIn(
                            _OBSERVER_ID_SENTINEL,
                            str(raised),
                        )
                    else:
                        trace = canonical.to_trace_dict()
                        observability = stream_observability_payload(canonical)
                        self.assertNotIn(
                            _OBSERVER_ID_SENTINEL,
                            repr((trace, observability)),
                        )
                        if field_name in item_fields:
                            validate_observer_id(
                                trace[field_name],
                                field_name,
                            )
                        else:
                            correlation = cast(
                                dict[str, object],
                                trace["correlation"],
                            )
                            validate_observer_id(
                                correlation[field_name],
                                field_name,
                            )

    def test_pending_interaction_rejects_false_stream_completion(
        self,
    ) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.INTERACTION_PENDING,
                1,
                correlation=_correlation(),
            ),
            _item(
                StreamItemKind.STREAM_COMPLETED,
                2,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
                usage={},
            ),
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "unresolved interaction cannot use this stream terminal outcome",
        ):
            validate_canonical_stream_items(items)
        self._assert_rejected_by_all_state_paths(
            items,
            "unresolved interaction cannot use this stream terminal outcome",
        )

    def test_valid_opaque_ids_project_to_content_safe_trace_tokens(
        self,
    ) -> None:
        raw_ids = {
            "stream_session_id": "stream café",
            "run_id": "run 東京",
            "turn_id": "turn ü",
            "task_id": "task mañana",
            "request_id": "request α",
            "continuation_id": "continuation β",
            "agent_id": "agent γ",
            "branch_id": "branch δ",
            "parent_branch_id": "parent ε",
        }
        correlation = StreamItemCorrelation(
            task_id=raw_ids["task_id"],
            request_id=InputRequestId(raw_ids["request_id"]),
            continuation_id=ContinuationId(raw_ids["continuation_id"]),
            agent_id=AgentId(raw_ids["agent_id"]),
            branch_id=BranchId(raw_ids["branch_id"]),
            parent_branch_id=BranchId(raw_ids["parent_branch_id"]),
        )
        item = CanonicalStreamItem(
            stream_session_id=raw_ids["stream_session_id"],
            run_id=raw_ids["run_id"],
            turn_id=raw_ids["turn_id"],
            sequence=1,
            kind=StreamItemKind.INTERACTION_PENDING,
            channel=StreamChannel.INTERACTION,
            correlation=correlation,
        )

        projection = project_canonical_stream_item(item)
        trace = item.to_trace_dict()
        observability = stream_observability_payload(item)

        self.assertEqual(
            projection.correlation.request_id,
            raw_ids["request_id"],
        )
        for raw_id in raw_ids.values():
            self.assertNotIn(raw_id, repr((trace, observability)))
        for field_name in ("stream_session_id", "run_id", "turn_id"):
            self.assertEqual(
                trace[field_name],
                project_observer_id(raw_ids[field_name], field_name),
            )
        trace_correlation = cast(
            dict[str, object],
            trace["correlation"],
        )
        for field_name in (
            "task_id",
            "request_id",
            "continuation_id",
            "agent_id",
            "branch_id",
            "parent_branch_id",
        ):
            self.assertEqual(
                trace_correlation[field_name],
                project_observer_id(raw_ids[field_name], field_name),
            )
        self.assertEqual(
            StreamItemCorrelation(task_id="task café").to_trace_dict(),
            {"task_id": "task café"},
        )

    def test_multiple_pending_requests_reject_input_required_cardinality(
        self,
    ) -> None:
        first = _correlation(request_id="request-1")
        second = _correlation(request_id="request-2")
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.INTERACTION_PENDING,
                1,
                correlation=first,
            ),
            _item(
                StreamItemKind.INTERACTION_PENDING,
                2,
                correlation=second,
            ),
            _item(
                StreamItemKind.STREAM_INPUT_REQUIRED,
                3,
                correlation=second,
                terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
            ),
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "multiple interaction requests in one stream",
        ):
            validate_canonical_stream_items(items)
        self._assert_rejected_by_all_state_paths(
            items,
            "multiple interaction requests in one stream",
        )

    def test_terminal_validator_rejects_input_required_without_one_pending(
        self,
    ) -> None:
        item = _item(
            StreamItemKind.STREAM_INPUT_REQUIRED,
            2,
            correlation=_correlation(),
            terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "input_required requires exactly one pending interaction",
        ):
            stream_module._validate_interaction_terminal_outcome(item, {})

    def test_interaction_terminal_outcomes_require_prior_resolution(
        self,
    ) -> None:
        for kind, outcome, usage in (
            (
                StreamItemKind.STREAM_COMPLETED,
                StreamTerminalOutcome.COMPLETED,
                {},
            ),
            (
                StreamItemKind.STREAM_CANCELLED,
                StreamTerminalOutcome.CANCELLED,
                None,
            ),
            (
                StreamItemKind.STREAM_ERRORED,
                StreamTerminalOutcome.ERRORED,
                None,
            ),
        ):
            items = (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.INTERACTION_PENDING,
                    1,
                    correlation=_correlation(),
                ),
                _item(
                    kind,
                    2,
                    terminal_outcome=outcome,
                    usage=usage,
                ),
            )
            with self.subTest(kind=kind):
                self._assert_rejected_by_all_state_paths(
                    items,
                    "unresolved interaction cannot use this stream"
                    " terminal outcome",
                )

    def test_resolved_interactions_allow_adjacent_stream_terminals(
        self,
    ) -> None:
        correlation = _correlation()
        cases: tuple[
            tuple[
                StreamItemKind,
                StreamItemKind,
                StreamTerminalOutcome,
                LooseJsonValue | None,
            ],
            ...,
        ] = (
            (
                StreamItemKind.INTERACTION_ANSWERED,
                StreamItemKind.STREAM_COMPLETED,
                StreamTerminalOutcome.COMPLETED,
                {},
            ),
            (
                StreamItemKind.INTERACTION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamTerminalOutcome.CANCELLED,
                None,
            ),
            (
                StreamItemKind.INTERACTION_UNAVAILABLE,
                StreamItemKind.STREAM_ERRORED,
                StreamTerminalOutcome.ERRORED,
                None,
            ),
        )
        for interaction_kind, stream_kind, outcome, usage in cases:
            items = (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.INTERACTION_PENDING,
                    1,
                    correlation=correlation,
                ),
                _item(interaction_kind, 2, correlation=correlation),
                _item(
                    stream_kind,
                    3,
                    terminal_outcome=outcome,
                    usage=usage,
                ),
            )
            with self.subTest(interaction_kind=interaction_kind):
                self.assertEqual(validate_canonical_stream_items(items), items)
                accumulator = CanonicalStreamAccumulator().add_many(items)
                self.assertEqual(accumulator.validate_complete(), items)
                projection = StreamProjectionState(
                    stream_session_id="stream-1",
                    run_id="run-1",
                    turn_id="turn-1",
                )
                for item in items:
                    projection.project(
                        item,
                        item.sequence,
                        unsupported_message="unsupported",
                    )
                projection.validate_complete()

    def test_sequential_resolved_interactions_share_one_stream(self) -> None:
        first = _correlation(request_id="request-1")
        second = _correlation(
            request_id="request-2",
            continuation_id="continuation-2",
        )
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.INTERACTION_CREATED,
                1,
                correlation=first,
            ),
            _item(
                StreamItemKind.INTERACTION_PENDING,
                2,
                correlation=first,
            ),
            _item(
                StreamItemKind.INTERACTION_ANSWERED,
                3,
                correlation=first,
            ),
            _item(
                StreamItemKind.INTERACTION_CREATED,
                4,
                correlation=second,
            ),
            _item(
                StreamItemKind.INTERACTION_PENDING,
                5,
                correlation=second,
            ),
            _item(
                StreamItemKind.INTERACTION_DECLINED,
                6,
                correlation=second,
            ),
            _item(
                StreamItemKind.STREAM_COMPLETED,
                7,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
                usage={},
            ),
        )

        self.assertEqual(validate_canonical_stream_items(items), items)
        accumulator = CanonicalStreamAccumulator().add_many(items)
        self.assertEqual(accumulator.validate_complete(), items)
        projection = StreamProjectionState(
            stream_session_id="stream-1",
            run_id="run-1",
            turn_id="turn-1",
        )
        for item in items:
            projection.project(
                item,
                item.sequence,
                unsupported_message="unsupported",
            )
        projection.validate_complete()

    def test_overlapping_interactions_remain_invalid(self) -> None:
        first = _correlation(request_id="request-1")
        second = _correlation(
            request_id="request-2",
            continuation_id="continuation-2",
        )
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.INTERACTION_PENDING,
                1,
                correlation=first,
            ),
            _item(
                StreamItemKind.INTERACTION_CREATED,
                2,
                correlation=second,
            ),
        )

        rejection_messages = self._assert_rejected_by_all_state_paths(
            items,
            "multiple interaction requests in one stream",
        )
        self.assertEqual(
            tuple(
                "multiple interaction requests in one stream" in detail
                for detail in rejection_messages
            ),
            (True, True, True),
        )

    def test_interaction_stream_projects_correlated_input_required(
        self,
    ) -> None:
        correlation = _correlation()
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.INTERACTION_CREATED,
                1,
                correlation=correlation,
            ),
            _item(
                StreamItemKind.INTERACTION_PENDING,
                2,
                correlation=correlation,
            ),
            _item(
                StreamItemKind.STREAM_INPUT_REQUIRED,
                3,
                correlation=correlation,
                terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
            ),
            _item(StreamItemKind.STREAM_CLOSED, 4),
        )

        self.assertEqual(validate_canonical_stream_items(items), items)
        accumulator = CanonicalStreamAccumulator().add_many(items)
        self.assertEqual(accumulator.validate_complete(), items)
        self.assertIs(
            accumulator.terminal_outcome,
            StreamTerminalOutcome.INPUT_REQUIRED,
        )
        self.assertEqual(
            correlation.to_trace_dict(),
            {
                "task_id": project_observer_id("task-1", "task_id"),
                "request_id": project_observer_id(
                    "request-1",
                    "request_id",
                ),
                "continuation_id": project_observer_id(
                    "continuation-1",
                    "continuation_id",
                ),
                "agent_id": project_observer_id("agent-1", "agent_id"),
                "branch_id": project_observer_id(
                    "branch-1",
                    "branch_id",
                ),
                "parent_branch_id": project_observer_id(
                    "parent-branch-1",
                    "parent_branch_id",
                ),
            },
        )
        terminal_projection = project_canonical_stream_item(items[3])
        self.assertEqual(
            canonical_item_from_consumer_projection(terminal_projection),
            items[3],
        )
        self.assertEqual(
            stream_observability_payload(items[2]),
            {
                "request_id": project_observer_id(
                    "request-1",
                    "request_id",
                ),
                "run_id": project_observer_id("run-1", "run_id"),
                "turn_id": project_observer_id("turn-1", "turn_id"),
                "task_id": project_observer_id("task-1", "task_id"),
                "agent_id": project_observer_id("agent-1", "agent_id"),
                "branch_id": project_observer_id(
                    "branch-1",
                    "branch_id",
                ),
                "state": "pending",
            },
        )
        self.assertEqual(
            stream_observability_payload(items[3]),
            {
                "request_id": project_observer_id(
                    "request-1",
                    "request_id",
                ),
                "run_id": project_observer_id("run-1", "run_id"),
                "turn_id": project_observer_id("turn-1", "turn_id"),
                "task_id": project_observer_id("task-1", "task_id"),
                "agent_id": project_observer_id("agent-1", "agent_id"),
                "branch_id": project_observer_id(
                    "branch-1",
                    "branch_id",
                ),
                "state": "pending",
            },
        )

    def test_terminal_interaction_observability_has_resolution_category(
        self,
    ) -> None:
        item = _item(
            StreamItemKind.INTERACTION_DECLINED,
            1,
            correlation=_correlation(),
        )

        self.assertEqual(
            stream_observability_payload(item)["resolution_category"],
            "declined",
        )
        self.assertEqual(
            validate_canonical_stream_items(
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    item,
                    _item(
                        StreamItemKind.STREAM_CANCELLED,
                        2,
                        terminal_outcome=StreamTerminalOutcome.CANCELLED,
                    ),
                )
            )[1],
            item,
        )

    def test_input_required_keeps_retained_reasoning_incomplete(self) -> None:
        correlation = _correlation()
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.REASONING_DELTA,
                1,
                text_delta="partial thought",
                reasoning_representation=(
                    StreamReasoningRepresentation.NATIVE_TEXT
                ),
                segment_instance_ordinal=0,
            ),
            _item(StreamItemKind.REASONING_DONE, 2),
            _item(
                StreamItemKind.INTERACTION_PENDING,
                3,
                correlation=correlation,
            ),
            _item(
                StreamItemKind.STREAM_INPUT_REQUIRED,
                4,
                correlation=correlation,
                terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
            ),
        )

        accumulator = CanonicalStreamAccumulator().add_many(items)

        self.assertEqual(accumulator.validate_complete(), items)
        self.assertEqual(len(accumulator.reasoning_segments), 1)
        segment = accumulator.reasoning_segments[0]
        self.assertIs(segment.status, StreamReasoningSegmentStatus.INCOMPLETE)
        self.assertFalse(segment.completed)
        self.assertIs(
            segment.terminal_outcome,
            StreamTerminalOutcome.INPUT_REQUIRED,
        )

    def test_interaction_stream_rejects_content_and_provider_authorship(
        self,
    ) -> None:
        correlation = _correlation()
        base = _item(
            StreamItemKind.INTERACTION_PENDING,
            1,
            correlation=correlation,
        )
        invalid_items: tuple[Callable[[], CanonicalStreamItem], ...] = (
            lambda: replace(base, data={"prompt": "private"}),
            lambda: replace(base, usage={"input_tokens": 1}),
            lambda: replace(base, metadata={"answer": "private"}),
            lambda: replace(base, provider_payload={"raw": "private"}),
            lambda: replace(base, provider_family="provider"),
            lambda: replace(base, provider_event_type="event"),
        )
        for build_item in invalid_items:
            with (
                self.subTest(build_item=build_item),
                self.assertRaises(AssertionError),
            ):
                build_item()

        prohibited_correlations = (
            replace(
                correlation,
                provider_request_id="provider-request",
            ),
            replace(
                correlation,
                artifact_id="artifact",
            ),
        )
        for prohibited in prohibited_correlations:
            with self.assertRaises(AssertionError):
                replace(base, correlation=prohibited)

        with self.assertRaises(AssertionError):
            StreamProviderEvent(
                kind=StreamItemKind.INTERACTION_PENDING,
                correlation=correlation,
            )
        projection = StreamConsumerProjection.from_item(base)
        with self.assertRaises(AssertionError):
            replace(
                projection,
                metadata={"prompt": "private"},
            )

    def test_interaction_metadata_is_defensively_frozen(self) -> None:
        source_metadata: dict[str, LooseJsonValue] = {}
        nested: dict[str, LooseJsonValue] = {}
        item = replace(
            _item(
                StreamItemKind.INTERACTION_PENDING,
                1,
                correlation=_correlation(),
            ),
            metadata=source_metadata,
        )
        projection_source: dict[str, LooseJsonValue] = {}
        projection = replace(
            project_canonical_stream_item(item),
            metadata=projection_source,
        )

        source_metadata["prompt"] = nested
        nested["answer"] = _OBSERVER_ID_SENTINEL
        projection_source["answer"] = _OBSERVER_ID_SENTINEL

        self.assertIsInstance(item.metadata, MappingProxyType)
        self.assertIsInstance(projection.metadata, MappingProxyType)
        for metadata in (item.metadata, projection.metadata):
            with self.assertRaises(TypeError):
                cast(dict[str, LooseJsonValue], metadata)[
                    "prompt"
                ] = _OBSERVER_ID_SENTINEL

        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            item,
            _item(
                StreamItemKind.STREAM_INPUT_REQUIRED,
                2,
                correlation=item.correlation,
                terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
            ),
        )
        self.assertEqual(validate_canonical_stream_items(items), items)
        projected = project_canonical_stream_item(item)
        outward = (
            item,
            item.to_trace_dict(),
            stream_observability_payload(item),
            projection,
            projected,
            canonical_item_from_consumer_projection(projection),
        )
        self.assertNotIn(_OBSERVER_ID_SENTINEL, repr(outward))

    def test_tampered_interaction_metadata_fails_closed_on_export(
        self,
    ) -> None:
        item = _item(
            StreamItemKind.INTERACTION_PENDING,
            1,
            correlation=_correlation(),
        )
        object.__setattr__(
            item,
            "metadata",
            {"prompt": {"answer": _OBSERVER_ID_SENTINEL}},
        )

        with self.assertRaises(AssertionError):
            item.to_trace_dict()
        with self.assertRaises(AssertionError):
            stream_observability_payload(item)
        with self.assertRaises(AssertionError):
            project_canonical_stream_item(item)
        with self.assertRaises(AssertionError):
            validate_canonical_stream_items((item,))

    def test_interaction_stream_rejects_missing_and_mixed_correlation(
        self,
    ) -> None:
        base = _correlation()
        required_fields = (
            "request_id",
            "continuation_id",
            "agent_id",
            "branch_id",
        )
        for field_name in required_fields:
            with (
                self.subTest(field_name=field_name),
                self.assertRaises(AssertionError),
            ):
                _item(
                    StreamItemKind.INTERACTION_PENDING,
                    1,
                    correlation=(
                        replace(base, request_id=None)
                        if field_name == "request_id"
                        else (
                            replace(base, continuation_id=None)
                            if field_name == "continuation_id"
                            else (
                                replace(base, agent_id=None)
                                if field_name == "agent_id"
                                else replace(base, branch_id=None)
                            )
                        )
                    ),
                )

        changed = _correlation(continuation_id="continuation-2")
        with self.assertRaisesRegex(
            StreamValidationError,
            "correlation changed before input_required",
        ):
            validate_canonical_stream_items(
                (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _item(
                        StreamItemKind.INTERACTION_PENDING,
                        1,
                        correlation=base,
                    ),
                    _item(
                        StreamItemKind.STREAM_INPUT_REQUIRED,
                        2,
                        correlation=changed,
                        terminal_outcome=(
                            StreamTerminalOutcome.INPUT_REQUIRED
                        ),
                    ),
                )
            )

    def test_interaction_stream_rejects_illegal_lifecycle_order(self) -> None:
        correlation = _correlation()
        cases = (
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_INPUT_REQUIRED,
                    1,
                    correlation=correlation,
                    terminal_outcome=StreamTerminalOutcome.INPUT_REQUIRED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.INTERACTION_CREATED,
                    1,
                    correlation=correlation,
                ),
                _item(
                    StreamItemKind.INTERACTION_ANSWERED,
                    2,
                    correlation=correlation,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.INTERACTION_PENDING,
                    1,
                    correlation=correlation,
                ),
                _item(
                    StreamItemKind.INTERACTION_PENDING,
                    2,
                    correlation=correlation,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.INTERACTION_PENDING,
                    1,
                    correlation=correlation,
                ),
                _item(
                    StreamItemKind.INTERACTION_ANSWERED,
                    2,
                    correlation=correlation,
                ),
                _item(
                    StreamItemKind.INTERACTION_DECLINED,
                    3,
                    correlation=correlation,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.INTERACTION_CREATED,
                    1,
                    correlation=correlation,
                ),
                _item(
                    StreamItemKind.INTERACTION_PENDING,
                    2,
                    correlation=_correlation(branch_id="branch-2"),
                ),
            ),
        )
        self.assertEqual(len(cases), 5)
        for items in cases:
            with (
                self.subTest(items=items),
                self.assertRaises(StreamValidationError),
            ):
                validate_canonical_stream_items(items)


if __name__ == "__main__":
    main()
