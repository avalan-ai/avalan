"""Exercise interaction observer privacy boundaries."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase, main

from avalan.event import (
    Event,
    EventObservabilityPayload,
    EventType,
    InteractionLifecyclePayload,
)
from avalan.event.manager import EventHistoryConfig, EventManager
from avalan.interaction import (
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputErrorCode,
    InputRequestId,
    ModelCallId,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    RunId,
    StreamSessionId,
    TaskId,
    TextAnswer,
    TextQuestion,
    TurnId,
    create_input_request,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    stream_observability_payload,
)
from avalan.task import (
    PrivacySanitizer,
    TaskEventCategory,
    sanitize_raw_task_event,
    sanitize_raw_task_event_closed,
    task_event_category,
)

_SENTINEL = "PRIVATE_INTERACTION_CONTENT_SENTINEL"
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
_NOW = datetime(2026, 7, 20, 18, 0, tzinfo=UTC)


def _private_domain_objects() -> tuple[object, ...]:
    origin = ExecutionOrigin(
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        task_id=TaskId("task-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        model_call_id=ModelCallId("model-call-1"),
        stream_session_id=StreamSessionId("stream-1"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://safe",
            agent_definition_revision="r1",
            operation_id="operation-1",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
    )
    request = create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=origin,
        mode=RequirementMode.REQUIRED,
        reason=_SENTINEL,
        questions=(
            TextQuestion(
                question_id=QuestionId("question-1"),
                prompt=_SENTINEL,
                required=True,
                header=_SENTINEL,
                help_text=_SENTINEL,
                default_value=_SENTINEL,
            ),
        ),
        created_at=_NOW,
    )
    resolution = AnsweredResolution(
        request_id=request.request_id,
        provenance=AnswerProvenance.HUMAN,
        resolved_at=_NOW + timedelta(seconds=1),
        answers=(
            TextAnswer(
                question_id=QuestionId("question-1"),
                provenance=AnswerProvenance.HUMAN,
                value=_SENTINEL,
            ),
        ),
    )
    return request, resolution, origin, RuntimeError(_SENTINEL)


def _safe_payload() -> InteractionLifecyclePayload:
    return InteractionLifecyclePayload.from_canonical_ids(
        request_id=InputRequestId("request-1"),
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        task_id=TaskId("task-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        state=RequestState.ANSWERED,
        resolution_category=ResolutionStatus.ANSWERED,
        surface="task.events",
        wait_duration_ms=1000,
        validation_code=InputErrorCode.INVALID_TYPE,
        duplicate=False,
        stale=False,
        provenance_category=AnswerProvenance.HUMAN,
    )


class InteractionEventSanitizationTestCase(TestCase):
    def test_canonical_ids_never_enter_task_observer_storage(self) -> None:
        raw_ids = (
            "request α",
            "run 東京",
            "turn ü",
            "task mañana",
            "agent γ",
            "branch δ",
        )
        payload = InteractionLifecyclePayload.from_canonical_ids(
            request_id=InputRequestId(raw_ids[0]),
            run_id=RunId(raw_ids[1]),
            turn_id=TurnId(raw_ids[2]),
            task_id=TaskId(raw_ids[3]),
            agent_id=AgentId(raw_ids[4]),
            branch_id=BranchId(raw_ids[5]),
            state=RequestState.PENDING,
        )
        event = Event.from_interaction_lifecycle(payload)

        draft = sanitize_raw_task_event(event, PrivacySanitizer())
        history = event.for_history()

        for raw_id in raw_ids:
            self.assertNotIn(raw_id, repr((event, history, draft)))

    def test_every_restored_id_rejects_observer_smuggling(self) -> None:
        base = _safe_payload().to_dict()
        for field_name in (
            "request_id",
            "run_id",
            "turn_id",
            "task_id",
            "agent_id",
            "branch_id",
        ):
            for attack in _OBSERVER_ID_ATTACKS:
                with self.subTest(field_name=field_name, attack=attack):
                    observability = EventObservabilityPayload.canonical_stream(
                        cast(Any, {**base, field_name: attack})
                    )
                    raw_event = SimpleNamespace(
                        type=EventType.INTERACTION_LIFECYCLE,
                        payload={},
                        observability_payload=observability,
                        started=None,
                        finished=None,
                        elapsed=None,
                    )

                    draft = sanitize_raw_task_event(
                        raw_event,
                        PrivacySanitizer(),
                    )
                    closed = sanitize_raw_task_event_closed(
                        raw_event,
                        PrivacySanitizer(),
                    )

                    for sanitized in (draft, closed):
                        self.assertNotIn("canonical_stream", sanitized.payload)
                        self.assertNotIn(
                            "interaction_lifecycle", sanitized.payload
                        )
                        self.assertNotIn(
                            _OBSERVER_ID_SENTINEL,
                            repr(sanitized),
                        )

    def test_observer_projection_excludes_full_domain_objects(self) -> None:
        event = Event.from_interaction_lifecycle(_safe_payload())
        request, resolution, origin, error = _private_domain_objects()
        malicious = replace(
            event,
            payload={
                "request": request,
                "resolution": resolution,
                "origin": origin,
                "exception": error,
                "prompt": _SENTINEL,
                "answer": _SENTINEL,
            },
        )

        draft = sanitize_raw_task_event(malicious, PrivacySanitizer())
        historical = malicious.for_history()

        self.assertIs(draft.category, TaskEventCategory.INTERACTION)
        self.assertEqual(
            draft.payload["interaction_lifecycle"],
            _safe_payload().to_dict(),
        )
        self.assertNotIn(_SENTINEL, repr(draft.payload))
        self.assertNotIn(_SENTINEL, repr(historical))
        self.assertIsNone(historical.payload)

    def test_canonical_stream_observer_projection_is_allowlist_only(
        self,
    ) -> None:
        item = CanonicalStreamItem(
            stream_session_id="stream-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=2,
            kind=StreamItemKind.INTERACTION_PENDING,
            channel=StreamChannel.INTERACTION,
            correlation=StreamItemCorrelation(
                request_id=InputRequestId("request-1"),
                continuation_id=ContinuationId("continuation-1"),
                task_id="task-1",
                agent_id=AgentId("agent-1"),
                branch_id=BranchId("branch-1"),
                parent_branch_id=BranchId("parent-1"),
            ),
        )
        projected = stream_observability_payload(item)
        event = Event.from_observability_payload(
            type=EventType.INTERACTION_LIFECYCLE,
            observability_payload=EventObservabilityPayload.canonical_stream(
                projected
            ),
        )

        draft = sanitize_raw_task_event(event, PrivacySanitizer())

        self.assertEqual(
            draft.payload["interaction_lifecycle"],
            projected,
        )
        self.assertNotIn("continuation_id", repr(draft.payload))
        self.assertNotIn("parent_branch_id", repr(draft.payload))

    def test_forged_stream_observability_cannot_smuggle_content(
        self,
    ) -> None:
        manager = EventManager(history_config=EventHistoryConfig(max_events=2))
        valid = _safe_payload().to_dict()
        valid["state"] = "pending"
        valid.pop("resolution_category")
        valid.pop("provenance_category")

        with self.assertRaises(AssertionError) as raised:
            Event.from_observability_payload(
                type=EventType.INTERACTION_LIFECYCLE,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        cast(
                            Any,
                            {
                                **valid,
                                "surface": "bad/surface",
                                "wait_duration_ms": -1,
                                "validation_code": "unknown",
                                "duplicate": 1,
                                "stale": "false",
                                "reason": _SENTINEL,
                                "prompt": _SENTINEL,
                                "help": _SENTINEL,
                                "choice_label": _SENTINEL,
                                "choice_description": _SENTINEL,
                                "default_value": _SENTINEL,
                                "answer": _SENTINEL,
                                "request": _private_domain_objects()[0],
                                "resolution": _private_domain_objects()[1],
                                "origin": _private_domain_objects()[2],
                                "exception": _private_domain_objects()[3],
                            },
                        )
                    )
                ),
            )

        self.assertEqual(len(manager.history), 0)
        self.assertNotIn(_SENTINEL, str(raised.exception))

    def test_invalid_interaction_state_fails_closed(self) -> None:
        invalid = _safe_payload().to_dict()
        invalid["state"] = "unknown"
        raw_event = SimpleNamespace(
            type=EventType.INTERACTION_LIFECYCLE,
            payload={},
            observability_payload=EventObservabilityPayload.canonical_stream(
                cast(Any, invalid)
            ),
            started=None,
            finished=None,
            elapsed=None,
        )

        for sanitized in (
            sanitize_raw_task_event(raw_event, PrivacySanitizer()),
            sanitize_raw_task_event_closed(raw_event, PrivacySanitizer()),
        ):
            self.assertNotIn("canonical_stream", sanitized.payload)
            self.assertNotIn("interaction_lifecycle", sanitized.payload)

    def test_noninteraction_events_do_not_restore_interaction_fields(
        self,
    ) -> None:
        observability = EventObservabilityPayload.canonical_stream(
            {
                "request_id": "request-1",
                "state": "pending",
            }
        )
        token = Event.from_observability_payload(
            type=EventType.TOKEN_GENERATED,
            observability_payload=observability,
        )
        legacy = Event(type=EventType.INTERACTION_LIFECYCLE)
        temporary = Event.from_observability_payload(
            type=EventType.INTERACTION_LIFECYCLE,
            observability_payload=EventObservabilityPayload.temporary_legacy(
                {"request_id": "request-1", "state": "pending"},
                owner="interaction-boundary",
                removal_condition="canonical projection available",
            ),
        )

        token_draft = sanitize_raw_task_event(token, PrivacySanitizer())
        legacy_draft = sanitize_raw_task_event(legacy, PrivacySanitizer())
        temporary_draft = sanitize_raw_task_event(
            temporary,
            PrivacySanitizer(),
        )

        self.assertNotIn("interaction_lifecycle", token_draft.payload)
        self.assertNotIn("interaction_lifecycle", legacy_draft.payload)
        self.assertNotIn("interaction_lifecycle", temporary_draft.payload)
        self.assertIs(
            task_event_category("interaction_lifecycle"),
            TaskEventCategory.INTERACTION,
        )


if __name__ == "__main__":
    main()
