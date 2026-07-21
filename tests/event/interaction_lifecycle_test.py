"""Exercise content-safe interaction lifecycle events."""

from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.event import (
    Event,
    EventObservabilityPayload,
    EventPayloadKind,
    EventType,
    InteractionLifecyclePayload,
    ObserverId,
    project_observer_id,
    validate_observer_id,
)
from avalan.event.manager import EventHistoryConfig, EventManager
from avalan.interaction import (
    AgentId,
    AnswerProvenance,
    BranchId,
    ContinuationId,
    InputErrorCode,
    InputRequestId,
    RequestState,
    ResolutionStatus,
    RunId,
    TaskId,
    TurnId,
)
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
)

_OBSERVER_ID_SENTINEL = "PRIVATE_PROMPT_SENTINEL"
_ENUM_SENTINEL = "PRIVATE_STATE_SENTINEL"
_OBSERVER_ID_ATTACKS = (
    f"{_OBSERVER_ID_SENTINEL}\nanswer",
    f"{_OBSERVER_ID_SENTINEL} answer",
    f"{_OBSERVER_ID_SENTINEL}\tanswer",
    f"{_OBSERVER_ID_SENTINEL}\x1banswer",
    f"{_OBSERVER_ID_SENTINEL}\u202eanswer",
    f"{_OBSERVER_ID_SENTINEL}\\nanswer",
    f"<{_OBSERVER_ID_SENTINEL}>answer",
)


def _payload(
    *,
    state: RequestState = RequestState.ANSWERED,
    resolution_category: ResolutionStatus | None = ResolutionStatus.ANSWERED,
    provenance_category: AnswerProvenance | None = AnswerProvenance.HUMAN,
) -> InteractionLifecyclePayload:
    return InteractionLifecyclePayload.from_canonical_ids(
        request_id=InputRequestId("request-1"),
        run_id=RunId("run-1"),
        turn_id=TurnId("turn-1"),
        task_id=TaskId("task-1"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        state=state,
        resolution_category=resolution_category,
        surface="sdk.attached",
        wait_duration_ms=25,
        validation_code=InputErrorCode.STALE_REVISION,
        duplicate=False,
        stale=True,
        provenance_category=provenance_category,
    )


class InteractionLifecycleTestCase(IsolatedAsyncioTestCase):
    def test_observer_id_contract_is_bounded_and_content_safe(self) -> None:
        observer_id: ObserverId = project_observer_id(
            "request-1",
            "request_id",
        )

        self.assertEqual(
            validate_observer_id(observer_id, "request_id"),
            observer_id,
        )
        self.assertTrue(observer_id.startswith("oid_"))
        self.assertEqual(len(observer_id), 68)
        self.assertNotEqual(
            observer_id,
            project_observer_id("request-1", "run_id"),
        )
        self.assertEqual(
            project_observer_id("cafe\u0301", "request_id"),
            project_observer_id("café", "request_id"),
        )
        with self.assertRaises(AssertionError):
            validate_observer_id("request-1", "request_id")

    def test_canonical_unicode_ids_are_projected_before_storage(self) -> None:
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

        for raw_id in raw_ids:
            self.assertNotIn(
                raw_id, repr((payload, event, event.for_history()))
            )
        for field_name in (
            "request_id",
            "run_id",
            "turn_id",
            "task_id",
            "agent_id",
            "branch_id",
        ):
            validate_observer_id(payload.to_dict()[field_name], field_name)

    def test_every_lifecycle_id_rejects_observer_smuggling(self) -> None:
        base = _payload()
        valid = base.to_dict()
        field_names = (
            "request_id",
            "run_id",
            "turn_id",
            "task_id",
            "agent_id",
            "branch_id",
        )
        for field_name in field_names:
            for attack in _OBSERVER_ID_ATTACKS:
                with self.subTest(field_name=field_name, attack=attack):
                    with self.assertRaises(AssertionError) as payload_error:
                        replace(
                            base,
                            **{field_name: cast(Any, attack)},
                        )
                    self.assertNotIn(
                        _OBSERVER_ID_SENTINEL,
                        str(payload_error.exception),
                    )

                    invalid_data = {**valid, field_name: attack}
                    with self.assertRaises(AssertionError) as direct_error:
                        EventObservabilityPayload(
                            kind=EventPayloadKind.INTERACTION_LIFECYCLE,
                            data=cast(Any, invalid_data),
                        )
                    self.assertNotIn(
                        _OBSERVER_ID_SENTINEL,
                        str(direct_error.exception),
                    )

                    observability = EventObservabilityPayload.canonical_stream(
                        cast(Any, invalid_data)
                    )
                    with self.assertRaises(AssertionError) as event_error:
                        Event.from_observability_payload(
                            type=EventType.INTERACTION_LIFECYCLE,
                            observability_payload=observability,
                        )
                    self.assertNotIn(
                        _OBSERVER_ID_SENTINEL,
                        str(event_error.exception),
                    )

    def test_raw_lifecycle_discriminators_fail_without_echo(self) -> None:
        valid = _payload().to_dict()
        for field_name, sentinel in (
            ("state", _ENUM_SENTINEL),
            ("resolution_category", _ENUM_SENTINEL),
            ("provenance_category", _ENUM_SENTINEL),
            ("validation_code", _ENUM_SENTINEL),
            ("surface", f"{_ENUM_SENTINEL}/surface"),
        ):
            invalid_data = {**valid, field_name: sentinel}
            with self.subTest(field_name=field_name, kind="direct"):
                with self.assertRaises(AssertionError) as direct_error:
                    EventObservabilityPayload(
                        kind=EventPayloadKind.INTERACTION_LIFECYCLE,
                        data=cast(Any, invalid_data),
                    )
                self.assertNotIn(
                    sentinel,
                    str(direct_error.exception),
                )
            with self.subTest(field_name=field_name, kind="canonical"):
                observability = EventObservabilityPayload.canonical_stream(
                    cast(Any, invalid_data)
                )
                with self.assertRaises(AssertionError) as canonical_error:
                    Event.from_observability_payload(
                        type=EventType.INTERACTION_LIFECYCLE,
                        observability_payload=observability,
                    )
                self.assertNotIn(
                    sentinel,
                    str(canonical_error.exception),
                )

    async def test_forged_canonical_history_is_fresh_frozen_and_allowlisted(
        self,
    ) -> None:
        nested = {"answer": _OBSERVER_ID_SENTINEL}
        source = {
            **_payload().to_dict(),
            "prompt": nested,
            "answer": _OBSERVER_ID_SENTINEL,
        }
        original_observability = EventObservabilityPayload.canonical_stream(
            cast(Any, source)
        )
        event = Event.from_observability_payload(
            type=EventType.INTERACTION_LIFECYCLE,
            observability_payload=original_observability,
        )

        source["request_id"] = _OBSERVER_ID_SENTINEL
        nested["answer"] = f"changed-{_OBSERVER_ID_SENTINEL}"
        history = event.for_history()
        manager = EventManager(history_config=EventHistoryConfig(max_events=2))
        await manager.trigger(event)

        self.assertIsNot(event.observability, original_observability)
        self.assertIsInstance(event.observability.data, MappingProxyType)
        self.assertIsInstance(event.payload, MappingProxyType)
        self.assertIsNot(history, event)
        self.assertIsNone(history.payload)
        self.assertEqual(len(manager.history), 1)
        self.assertIsNone(manager.history[0].payload)
        for outward in (
            event,
            event.observability,
            history,
            manager.history,
        ):
            self.assertNotIn(_OBSERVER_ID_SENTINEL, repr(outward))
        with self.assertRaises(TypeError):
            cast(dict[str, object], event.observability.data)[
                "prompt"
            ] = _OBSERVER_ID_SENTINEL
        with self.assertRaises(TypeError):
            cast(dict[str, object], event.payload)[
                "answer"
            ] = _OBSERVER_ID_SENTINEL

    async def test_typed_lifecycle_event_is_immutable_and_publishable(
        self,
    ) -> None:
        payload = _payload()
        event = Event.from_interaction_lifecycle(
            payload,
            started=1.0,
            finished=1.025,
            elapsed=0.025,
        )

        self.assertIs(event.type, EventType.INTERACTION_LIFECYCLE)
        self.assertIs(
            event.observability.kind,
            EventPayloadKind.INTERACTION_LIFECYCLE,
        )
        self.assertIsInstance(event.observability.data, MappingProxyType)
        lifecycle_payload = event.interaction_lifecycle_payload
        self.assertIsInstance(
            lifecycle_payload,
            InteractionLifecyclePayload,
        )
        self.assertEqual(lifecycle_payload, payload)
        self.assertEqual(
            event.observability.to_dict(),
            {
                "kind": "interaction_lifecycle",
                "data": payload.to_dict(),
            },
        )
        with self.assertRaises(TypeError):
            cast(dict[str, object], event.observability.data)[
                "prompt"
            ] = "private"

        manager = EventManager(history_config=EventHistoryConfig(max_events=2))
        await manager.trigger_interaction_lifecycle(payload)

        self.assertEqual(len(manager.history), 1)
        self.assertIs(
            manager.history[0].type,
            EventType.INTERACTION_LIFECYCLE,
        )
        self.assertIsNone(manager.history[0].payload)
        self.assertIsInstance(
            manager.history[0].interaction_lifecycle_payload,
            InteractionLifecyclePayload,
        )
        self.assertEqual(
            manager.history[0].interaction_lifecycle_payload,
            payload,
        )

    def test_typed_lifecycle_accessor_is_absent_for_legacy_events(
        self,
    ) -> None:
        event = Event(type=EventType.START, payload=_payload())
        unstructured_lifecycle = Event(
            type=EventType.INTERACTION_LIFECYCLE,
            payload=_payload(),
        )
        legacy_observability = EventObservabilityPayload.temporary_legacy(
            {"event_type": EventType.INTERACTION_LIFECYCLE.value},
            owner="test",
            removal_condition="test-only negative lifecycle payload",
        )
        legacy_lifecycle = Event.from_observability_payload(
            type=EventType.INTERACTION_LIFECYCLE,
            observability_payload=legacy_observability,
        )

        self.assertIsNone(event.interaction_lifecycle_payload)
        self.assertIsNone(unstructured_lifecycle.payload)
        self.assertIsNone(unstructured_lifecycle.interaction_lifecycle_payload)
        self.assertIsNone(legacy_lifecycle.payload)
        self.assertIsNone(legacy_lifecycle.interaction_lifecycle_payload)

    def test_observer_hydration_does_not_hash_tokens_twice(self) -> None:
        projected = _payload().to_dict()
        observability = EventObservabilityPayload(
            kind=EventPayloadKind.INTERACTION_LIFECYCLE,
            data=projected,
        )

        event = Event.from_observability_payload(
            type=EventType.INTERACTION_LIFECYCLE,
            observability_payload=observability,
        )
        history = event.for_history()

        for field_name in (
            "request_id",
            "run_id",
            "turn_id",
            "task_id",
            "agent_id",
            "branch_id",
        ):
            self.assertEqual(
                event.observability.data[field_name],
                projected[field_name],
            )
            self.assertEqual(
                history.observability.data[field_name],
                projected[field_name],
            )

    async def test_interaction_stream_item_uses_interaction_event_type(
        self,
    ) -> None:
        manager = EventManager(history_config=EventHistoryConfig(max_events=2))
        item = CanonicalStreamItem(
            stream_session_id="stream-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=1,
            kind=StreamItemKind.INTERACTION_PENDING,
            channel=StreamChannel.INTERACTION,
            correlation=StreamItemCorrelation(
                request_id=InputRequestId("request-1"),
                continuation_id=ContinuationId("continuation-1"),
                task_id="task-1",
                agent_id=AgentId("agent-1"),
                branch_id=BranchId("branch-1"),
            ),
        )

        await manager.trigger_stream_item(item)

        self.assertEqual(len(manager.history), 1)
        event = manager.history[0]
        self.assertIs(event.type, EventType.INTERACTION_LIFECYCLE)
        self.assertIsNone(event.payload)
        self.assertIsNotNone(event.observability)
        assert event.observability is not None
        self.assertEqual(event.observability.data["state"], "pending")
        self.assertNotIn("continuation_id", event.observability.data)

    def test_lifecycle_payload_rejects_invalid_scalar_fields(self) -> None:
        base = _payload()
        invalid_payloads: tuple[
            Callable[[], InteractionLifecyclePayload], ...
        ] = (
            lambda: replace(base, request_id=InputRequestId("")),
            lambda: replace(base, task_id=TaskId("")),
            lambda: replace(base, state=cast(Any, "answered")),
            lambda: replace(base, validation_code=cast(Any, "bad")),
            lambda: replace(base, provenance_category=cast(Any, "human")),
            lambda: replace(base, surface=""),
            lambda: replace(base, surface=" leading"),
            lambda: replace(base, surface="bad/surface"),
            lambda: replace(base, surface="é"),
            lambda: replace(base, surface="a" * 257),
            lambda: replace(base, wait_duration_ms=-1),
            lambda: replace(base, duplicate=cast(Any, 1)),
            lambda: replace(base, stale=cast(Any, "yes")),
        )
        for build_payload in invalid_payloads:
            with (
                self.subTest(build_payload=build_payload),
                self.assertRaises((AssertionError, ValueError)),
            ):
                build_payload()

    def test_lifecycle_payload_rejects_inconsistent_resolution(self) -> None:
        base = _payload()
        invalid_payloads: tuple[
            Callable[[], InteractionLifecyclePayload], ...
        ] = (
            lambda: replace(base, resolution_category=None),
            lambda: replace(
                base,
                resolution_category=ResolutionStatus.DECLINED,
            ),
            lambda: replace(base, provenance_category=None),
            lambda: _payload(
                state=RequestState.PENDING,
                resolution_category=ResolutionStatus.ANSWERED,
                provenance_category=AnswerProvenance.HUMAN,
            ),
        )
        for build_payload in invalid_payloads:
            with (
                self.subTest(build_payload=build_payload),
                self.assertRaises(AssertionError),
            ):
                build_payload()

        pending = _payload(
            state=RequestState.PENDING,
            resolution_category=None,
            provenance_category=None,
        )
        self.assertEqual(pending.to_dict()["state"], "pending")
        self.assertNotIn("resolution_category", pending.to_dict())

    def test_direct_observability_construction_is_fail_closed(self) -> None:
        valid = _payload().to_dict()
        invalid_data: tuple[dict[str, Any], ...] = (
            {"request_id": "request"},
            {**valid, "request": object()},
            {**valid, "request_id": 1},
            {**valid, "state": "unknown"},
            {**valid, "resolution_category": 1},
            {**valid, "resolution_category": "declined"},
            {**valid, "provenance_category": 1},
            {**valid, "provenance_category": "unknown"},
            {**valid, "surface": "bad/surface"},
            {**valid, "wait_duration_ms": -1},
            {**valid, "validation_code": 1},
            {**valid, "validation_code": "unknown"},
            {**valid, "duplicate": 1},
            {**valid, "stale": "true"},
        )
        for data in invalid_data:
            with (
                self.subTest(data=data),
                self.assertRaises((AssertionError, ValueError)),
            ):
                EventObservabilityPayload(
                    kind=EventPayloadKind.INTERACTION_LIFECYCLE,
                    data=cast(Any, data),
                )

        with self.assertRaises(AssertionError):
            EventObservabilityPayload.interaction_lifecycle(
                cast(Any, object())
            )


if __name__ == "__main__":
    main()
