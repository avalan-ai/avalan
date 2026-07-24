from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.event import Event, EventObservabilityPayload, EventType
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamVisibility,
    stream_observability_payload,
)
from avalan.task import (
    PrivacyAction,
    PrivacySanitizer,
    RawTaskEventListener,
    SanitizedTaskEvent,
    SanitizedTaskUsageEvent,
    TaskDefinition,
    TaskEventCategory,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInteractionEventType,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskStoreNotFoundError,
    UsageProviderFamily,
    UsageSource,
    UsageTotals,
    freeze_task_event_value,
    sanitize_raw_task_event,
    sanitize_raw_task_event_closed,
    task_event_category,
    task_interaction_event_payload,
)
from avalan.task.event import (
    _is_safe_correlation_value,
    _restore_reasoning_observability,
)
from avalan.task.stores import InMemoryTaskStore


class SequenceClock:
    def __init__(self) -> None:
        self._next = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self._next
        self._next = self._next + timedelta(seconds=1)
        return value


class SequenceIds:
    def __init__(self) -> None:
        self._next = 1

    def __call__(self) -> str:
        value = f"id-{self._next}"
        self._next = self._next + 1
        return value


def definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
    )


class SanitizedTaskEventTest(TestCase):
    def test_raw_correlation_identifier_is_not_safe(self) -> None:
        self.assertFalse(_is_safe_correlation_value("raw-correlation"))

    def test_interaction_event_payload_tracks_resumed_segment(self) -> None:
        payload = task_interaction_event_payload(
            event_type=TaskInteractionEventType.INPUT_RESUMED,
            request_id="request-1",
            continuation_id="continuation-1",
            segment_id="segment-1",
            next_segment_id="segment-2",
        )

        self.assertEqual(
            payload,
            {
                "request_id": "request-1",
                "continuation_id": "continuation-1",
                "segment_id": "segment-1",
                "next_segment_id": "segment-2",
            },
        )
        with self.assertRaises(AssertionError):
            task_interaction_event_payload(
                event_type=TaskInteractionEventType.INPUT_REQUIRED,
                request_id="request-1",
                continuation_id="continuation-1",
                segment_id="segment-1",
                next_segment_id="segment-2",
            )

    def test_token_event_drops_raw_token_text_and_ids(self) -> None:
        draft = sanitize_raw_task_event(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload={
                    "count": 1,
                    "token": "secret token",
                    "token_id": 123,
                    "token_ids": [123],
                },
                elapsed=0.25,
            ),
            PrivacySanitizer(),
        )

        payload = cast(dict[str, object], draft.payload)
        self.assertEqual(draft.category, TaskEventCategory.TOKEN)
        self.assertEqual(payload["event_type"], "token_generated")
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["duration_ms"], 250.0)
        self.assertNotIn("secret token", str(payload))
        self.assertNotIn("token_id", payload)
        self.assertNotIn("token_ids", payload)

    def test_token_event_keeps_canonical_stream_observability(self) -> None:
        item = CanonicalStreamItem(
            stream_session_id="stream-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=7,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta="private reasoning",
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=(
                StreamReasoningRepresentation.NATIVE_TEXT
            ),
            segment_instance_ordinal=0,
        )
        draft = sanitize_raw_task_event(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload={
                    "canonical_stream": {"kind": "answer.delta"},
                    "token": "private answer",
                },
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        stream_observability_payload(item)
                    )
                ),
            ),
            PrivacySanitizer(),
        )

        payload = cast(dict[str, object], draft.payload)
        canonical_stream = cast(
            dict[str, object],
            payload["canonical_stream"],
        )
        self.assertEqual(
            canonical_stream["kind"],
            StreamItemKind.REASONING_DELTA.value,
        )
        self.assertEqual(canonical_stream["channel"], "reasoning")
        self.assertEqual(canonical_stream["sequence"], 7)
        self.assertEqual(
            canonical_stream["summary"],
            {
                "text_delta_length": len("private reasoning"),
                "reasoning_representation": "native_text",
                "segment_instance_ordinal": 0,
            },
        )
        self.assertNotIn("private reasoning", str(payload))

        correlation_id = "sha256:" + ("a" * 64)
        complete_observability = sanitize_raw_task_event(
            Event(
                type=EventType.TOKEN_GENERATED,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        {
                            "kind": "reasoning.delta",
                            "provider_family": "openai",
                            "provider_event_type": "response.reasoning.delta",
                            "terminal_outcome": "completed",
                            "correlation": {
                                "model_continuation_id": correlation_id,
                                "protocol_item_id": correlation_id,
                                "provider_output_index": 2,
                                "provider_summary_index": 3,
                            },
                            "summary": {
                                "text_delta_length": 17,
                                "reasoning_representation": "native_text",
                                "segment_instance_ordinal": 1,
                            },
                            "usage": {
                                "input_tokens": 5,
                                "output_tokens": 7,
                                "total_tokens": 12,
                            },
                        }
                    )
                ),
            ),
            PrivacySanitizer(),
        )
        complete_payload = cast(
            dict[str, object],
            complete_observability.payload,
        )
        complete_canonical = cast(
            dict[str, object],
            complete_payload["canonical_stream"],
        )
        self.assertEqual(complete_canonical["provider_family"], "openai")
        self.assertEqual(
            complete_canonical["provider_event_type"],
            "response.reasoning.delta",
        )
        self.assertEqual(complete_canonical["terminal_outcome"], "completed")
        self.assertEqual(
            complete_canonical["correlation"],
            {
                "model_continuation_id": correlation_id,
                "protocol_item_id": correlation_id,
                "provider_output_index": 2,
                "provider_summary_index": 3,
            },
        )
        self.assertEqual(
            complete_canonical["usage"],
            {
                "input_tokens": 5,
                "output_tokens": 7,
                "total_tokens": 12,
            },
        )

        _restore_reasoning_observability(
            Event(
                type=EventType.TOKEN_GENERATED,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        {"kind": "reasoning.delta"}
                    )
                ),
            ),
            {"canonical_stream": "not-a-mapping"},
        )

        invalid_summary = sanitize_raw_task_event(
            Event(
                type=EventType.TOKEN_GENERATED,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        {
                            "kind": "reasoning.delta",
                            "summary": "not structured observability",
                        }
                    )
                ),
            ),
            PrivacySanitizer(),
        )
        invalid_payload = cast(dict[str, object], invalid_summary.payload)
        invalid_canonical = cast(
            dict[str, object], invalid_payload["canonical_stream"]
        )
        self.assertNotIn("summary", invalid_canonical)

        sentinel = "PRIVATE_OBSERVABILITY_KEY_SENTINEL"
        malicious_source = {
            "stream_session_id": "stream-1",
            "run_id": "run-1",
            "turn_id": "turn-1",
            "sequence": 8,
            "kind": "reasoning.delta",
            "channel": "reasoning",
            "visibility": "private",
            "provider_family": sentinel,
            "provider_event_type": sentinel,
            "correlation": {sentinel: sentinel},
            "data": {sentinel: sentinel},
            "metadata": {sentinel: sentinel},
            "summary": {
                "text_delta_length": 17,
                "reasoning_representation": "native_text",
                "segment_instance_ordinal": 0,
                "has_provider_payload": True,
                sentinel: sentinel,
            },
        }
        malicious = sanitize_raw_task_event(
            Event(
                type=EventType.TOKEN_GENERATED,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        malicious_source
                    )
                ),
            ),
            PrivacySanitizer(),
        )
        malicious_payload = cast(dict[str, object], malicious.payload)
        malicious_canonical = cast(
            dict[str, object], malicious_payload["canonical_stream"]
        )
        self.assertEqual(
            malicious_canonical["summary"],
            {
                "text_delta_length": 17,
                "reasoning_representation": "native_text",
                "segment_instance_ordinal": 0,
            },
        )
        self.assertNotIn("has_provider_payload", repr(malicious_payload))
        self.assertNotIn(sentinel, repr(malicious_payload))

        invalid_field_values = sanitize_raw_task_event(
            Event(
                type=EventType.TOKEN_GENERATED,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        {
                            **malicious_source,
                            "summary": {
                                "text_delta_length": "/private/path",
                                "reasoning_representation": {
                                    sentinel: sentinel
                                },
                                "segment_instance_ordinal": True,
                                "has_provider_payload": sentinel,
                            },
                        }
                    )
                ),
            ),
            PrivacySanitizer(),
        )
        invalid_values_payload = cast(
            dict[str, object], invalid_field_values.payload
        )
        invalid_values_canonical = cast(
            dict[str, object], invalid_values_payload["canonical_stream"]
        )
        self.assertNotIn("summary", invalid_values_canonical)
        self.assertNotIn(sentinel, repr(invalid_values_payload))

    def test_tool_model_engine_and_memory_events_keep_safe_metadata(
        self,
    ) -> None:
        cases = (
            ("tool_result", TaskEventCategory.TOOL),
            ("model_execute_after", TaskEventCategory.MODEL),
            ("engine_run_before", TaskEventCategory.ENGINE),
            ("memory_append_after", TaskEventCategory.MEMORY),
        )

        for event_type, category in cases:
            with self.subTest(event_type=event_type):
                draft = sanitize_raw_task_event(
                    Event(
                        type=cast(EventType, event_type),
                        payload={
                            "name": "retriever",
                            "status": "ok",
                            "arguments": {"query": "private"},
                            "result": "private result",
                            "output": "private output",
                        },
                    ),
                    PrivacySanitizer(),
                )

                payload = cast(dict[str, object], draft.payload)
                self.assertEqual(draft.category, category)
                self.assertEqual(payload["name"], "retriever")
                self.assertEqual(payload["status"], "ok")
                self.assertNotIn("private", str(payload))

    def test_flow_manager_events_keep_safe_metadata(self) -> None:
        draft = sanitize_raw_task_event(
            Event(
                type=EventType.FLOW_MANAGER_CALL_AFTER,
                payload={
                    "name": "flow",
                    "status": "succeeded",
                    "flow": object(),
                    "result": "private output",
                },
                elapsed=0.5,
            ),
            PrivacySanitizer(),
        )

        payload = cast(dict[str, object], draft.payload)
        self.assertEqual(draft.category, TaskEventCategory.ENGINE)
        self.assertEqual(payload["event_type"], "flow_manager_call_after")
        self.assertEqual(payload["name"], "flow")
        self.assertEqual(payload["status"], "succeeded")
        self.assertEqual(payload["duration_ms"], 500.0)
        self.assertNotIn("flow", payload)
        self.assertNotIn("private output", str(payload))

    def test_flow_runtime_events_keep_safe_metadata(self) -> None:
        draft = sanitize_raw_task_event(
            Event(
                type=EventType.FLOW_NODE_FAILED,
                payload={
                    "flow_id": "onboarding@1#abc",
                    "flow_name": "onboarding",
                    "flow_version": "1",
                    "flow_revision": "abc",
                    "node": "classify",
                    "status": "failed",
                    "attempt": 1,
                    "attempts": 2,
                    "duration_ms": 25.0,
                    "edge_count": 2,
                    "edge_index": 1,
                    "edge_kind": "error",
                    "eligible": False,
                    "matched": False,
                    "node_count": 3,
                    "output_name": "answer",
                    "ready": False,
                    "retry_delay_seconds": 0.0,
                    "route_kind": "error",
                    "source": "classify",
                    "target": "fallback",
                    "diagnostic_codes": ("flow.execution.node_failed",),
                    "file_data": b"raw file bytes",
                    "prompt": "private prompt",
                    "provider_body": {"secret": "private provider body"},
                    "result": "private result",
                    "response_body": "private model output",
                    "token_text": "private token text",
                },
            ),
            PrivacySanitizer(),
        )

        payload = cast(dict[str, object], draft.payload)
        self.assertEqual(draft.category, TaskEventCategory.ENGINE)
        self.assertEqual(payload["event_type"], "flow_node_failed")
        self.assertEqual(payload["flow_id"], "onboarding@1#abc")
        self.assertEqual(payload["flow_name"], "onboarding")
        self.assertEqual(payload["flow_version"], "1")
        self.assertEqual(payload["flow_revision"], "abc")
        self.assertEqual(payload["node"], "classify")
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["attempt"], 1)
        self.assertEqual(payload["attempts"], 2)
        self.assertEqual(payload["duration_ms"], 25.0)
        self.assertEqual(payload["edge_count"], 2)
        self.assertEqual(payload["edge_index"], 1)
        self.assertEqual(payload["edge_kind"], "error")
        self.assertFalse(payload["eligible"])
        self.assertFalse(payload["matched"])
        self.assertEqual(payload["node_count"], 3)
        self.assertEqual(payload["output_name"], "answer")
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["retry_delay_seconds"], 0.0)
        self.assertEqual(payload["route_kind"], "error")
        self.assertEqual(payload["source"], "classify")
        self.assertEqual(payload["target"], "fallback")
        self.assertEqual(
            payload["diagnostic_codes"],
            ("flow.execution.node_failed",),
        )
        self.assertNotIn("file_data", payload)
        self.assertNotIn("prompt", payload)
        self.assertNotIn("provider_body", payload)
        self.assertNotIn("result", payload)
        self.assertNotIn("response_body", payload)
        self.assertNotIn("token_text", payload)
        self.assertNotIn("private", str(payload))

    def test_closed_sanitizer_reduces_event_sanitization_errors(
        self,
    ) -> None:
        draft = sanitize_raw_task_event_closed(
            Event(
                type=EventType.FLOW_STARTED,
                payload={"status": "started"},
            ),
            PrivacySanitizer(
                TaskPrivacyPolicy(events=PrivacyAction.HASH),
            ),
        )

        payload = cast(dict[str, object], draft.payload)
        self.assertEqual(draft.event_type, "event_sanitization_failed")
        self.assertEqual(draft.category, TaskEventCategory.UNKNOWN)
        self.assertEqual(payload["privacy"], "<redacted>")

    def test_unknown_events_reduce_to_safe_metadata_only(self) -> None:
        draft = sanitize_raw_task_event(
            Event(
                type=cast(EventType, "custom_event"),
                payload={
                    "status": "started",
                    "prompt": "raw prompt",
                    "nested": {
                        "status": "ok",
                        "error": "raw exception",
                    },
                },
            ),
            PrivacySanitizer(),
        )
        hostile = sanitize_raw_task_event(
            Event(
                type=cast(EventType, "secret /tmp/input.txt"),
                payload={"status": "done", "output": "raw output"},
            ),
            PrivacySanitizer(),
        )

        payload = cast(dict[str, object], draft.payload)
        hostile_payload = cast(dict[str, object], hostile.payload)
        self.assertEqual(draft.category, TaskEventCategory.UNKNOWN)
        self.assertEqual(payload["event_type"], "custom_event")
        self.assertEqual(payload["status"], "started")
        self.assertNotIn("nested", payload)
        self.assertNotIn("raw prompt", str(payload))
        self.assertNotIn("raw exception", str(payload))
        self.assertEqual(hostile.event_type, "unknown")
        self.assertEqual(hostile_payload["event_type"], "unknown")
        self.assertNotIn("raw output", str(hostile_payload))

    def test_model_execute_event_redacts_provider_instructions(self) -> None:
        draft = sanitize_raw_task_event(
            Event(
                type=EventType.MODEL_EXECUTE_BEFORE,
                payload={
                    "name": "model-a",
                    "status": "started",
                    "instructions": "private instructions",
                    "system_prompt": "private system",
                    "developer_prompt": "private developer",
                },
            ),
            PrivacySanitizer(),
        )

        payload = cast(dict[str, object], draft.payload)
        self.assertEqual(draft.category, TaskEventCategory.MODEL)
        self.assertEqual(payload["event_type"], "model_execute_before")
        self.assertEqual(payload["name"], "model-a")
        self.assertEqual(payload["status"], "started")
        self.assertNotIn("private instructions", str(payload))
        self.assertNotIn("private system", str(payload))
        self.assertNotIn("private developer", str(payload))

    def test_task_event_values_are_frozen_and_reject_unsafe_values(
        self,
    ) -> None:
        frozen = freeze_task_event_value({"counts": [1, 2]})
        event = SanitizedTaskEvent(
            event_id="event-1",
            run_id="run-1",
            sequence=1,
            event_type="tool_result",
            category=TaskEventCategory.TOOL,
            payload={"status": "ok"},
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )

        self.assertIsInstance(frozen, MappingProxyType)
        self.assertEqual(cast(dict[str, object], frozen)["counts"], (1, 2))
        with self.assertRaises(TypeError):
            cast(dict[str, object], event.payload)["raw"] = "leak"
        with self.assertRaises(AssertionError):
            freeze_task_event_value({"raw": object()})
        with self.assertRaises(AssertionError):
            freeze_task_event_value(float("inf"))
        with self.assertRaises(AssertionError):
            freeze_task_event_value({"": "empty"})
        with self.assertRaises(AssertionError):
            SanitizedTaskEvent(
                event_id="event-2",
                run_id="run-1",
                sequence=0,
                event_type="tool_result",
                category=TaskEventCategory.TOOL,
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
            )

    def test_event_category_mapping_rejects_invalid_event_types(self) -> None:
        self.assertEqual(
            task_event_category("start"),
            TaskEventCategory.ENGINE,
        )
        self.assertEqual(
            task_event_category("usage_observed"),
            TaskEventCategory.USAGE,
        )
        with self.assertRaises(AssertionError):
            task_event_category("../secret")
        with self.assertRaises(AssertionError):
            task_event_category("")

    def test_usage_event_payload_is_categorical_and_numeric_only(
        self,
    ) -> None:
        event = SanitizedTaskUsageEvent(
            run_id="run-private",
            attempt_id="attempt-private",
            source=UsageSource.EXACT,
            totals=UsageTotals(
                input_tokens=3,
                cached_input_tokens=0,
                cache_creation_input_tokens=2,
                output_tokens=5,
                reasoning_tokens=7,
                total_tokens=17,
            ),
            metadata={
                "provider_family": UsageProviderFamily.AZURE_OPENAI,
                "flow_node": "analyze_pov_1",
                "cache_key": "private-cache-key",
                "headers": {"status": "private-header"},
                "raw_model_id": "provider/model-private",
            },
        )

        payload = cast(dict[str, object], event.payload)
        self.assertEqual(event.category, TaskEventCategory.USAGE)
        self.assertEqual(payload["event_type"], "usage_observed")
        self.assertEqual(payload["source"], "exact")
        self.assertEqual(payload["provider_family"], "azure_openai")
        self.assertEqual(payload["flow_node"], "analyze_pov_1")
        self.assertEqual(payload["input_tokens"], 3)
        self.assertEqual(payload["cached_input_tokens"], 0)
        self.assertEqual(payload["cache_creation_input_tokens"], 2)
        self.assertEqual(payload["output_tokens"], 5)
        self.assertEqual(payload["reasoning_tokens"], 7)
        self.assertEqual(payload["total_tokens"], 17)
        self.assertNotIn("run-private", str(payload))
        self.assertNotIn("attempt-private", str(payload))
        self.assertNotIn("private-cache-key", str(payload))
        self.assertNotIn("private-header", str(payload))
        self.assertNotIn("provider/model-private", str(payload))

    def test_usage_event_rejects_empty_observations(self) -> None:
        with self.assertRaises(AssertionError):
            SanitizedTaskUsageEvent(
                run_id="run-1",
                source=UsageSource.UNAVAILABLE,
                totals=UsageTotals(),
            )

    def test_event_redaction_denylist_overrides_custom_allowlists(
        self,
    ) -> None:
        sanitizer = PrivacySanitizer(
            event_allowlists={
                "model_execute_after": (
                    "cache_key",
                    "prompt_cache_key",
                    "cache_handle",
                    "cached_prompt_prefix",
                    "provider_cache_handle",
                    "response_metadata",
                    "headers",
                    "file_data",
                    "image_url",
                    "instructions",
                    "raw_provider_body",
                    "status",
                )
            }
        )
        draft = sanitize_raw_task_event(
            Event(
                type=EventType.MODEL_EXECUTE_AFTER,
                payload={
                    "cache_key": "private-cache-key",
                    "prompt_cache_key": "private-prompt-cache-key",
                    "cache_handle": "private-cache-handle",
                    "cached_prompt_prefix": "private-prefix",
                    "provider_cache_handle": "private-provider-cache",
                    "response_metadata": {"status": "private-response"},
                    "headers": {"status": "private-header"},
                    "file_data": "private-file-bytes",
                    "image_url": "private-image-url",
                    "instructions": "private instructions",
                    "raw_provider_body": "private body",
                    "status": "ok",
                },
            ),
            sanitizer,
        )

        payload = cast(dict[str, object], draft.payload)
        self.assertEqual(payload["status"], "ok")
        self.assertNotIn("private", str(payload))
        self.assertNotIn("cache_key", payload)
        self.assertNotIn("headers", payload)

    def test_non_event_objects_reduce_to_unknown_empty_payload(self) -> None:
        draft = sanitize_raw_task_event(object(), PrivacySanitizer())
        payload_only = type(
            "PayloadOnly",
            (),
            {
                "type": EventType.START,
                "payload": {"status": "private"},
            },
        )()
        payload_only_draft = sanitize_raw_task_event(
            payload_only,
            PrivacySanitizer(),
        )
        none_payload = sanitize_raw_task_event(
            Event(type=EventType.END, payload=None),
            PrivacySanitizer(),
        )
        timed = sanitize_raw_task_event(
            Event(
                type=EventType.END,
                payload=[],
                started=1.0,
                finished=2.0,
            ),
            PrivacySanitizer(),
        )

        payload = cast(dict[str, object], draft.payload)
        payload_only_value = cast(
            dict[str, object], payload_only_draft.payload
        )
        none_value = cast(dict[str, object], none_payload.payload)
        timed_value = cast(dict[str, object], timed.payload)
        self.assertEqual(draft.event_type, "unknown")
        self.assertEqual(payload_only_draft.event_type, "unknown")
        self.assertEqual(payload, {"event_type": "unknown"})
        self.assertEqual(payload_only_value, {"event_type": "unknown"})
        self.assertEqual(none_value, {"event_type": "end"})
        self.assertEqual(timed_value["started_at"], 1.0)
        self.assertEqual(timed_value["finished_at"], 2.0)
        self.assertNotIn("duration_ms", timed_value)


class RawTaskEventListenerTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryTaskStore(
            clock=SequenceClock(),
            id_factory=SequenceIds(),
        )
        await self.store.register_definition(
            definition(),
            definition_hash="hash-a",
        )
        self.run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        self.attempt = await self.store.create_attempt(self.run.run_id)

    async def test_listener_sanitizes_before_persisting_ordered_events(
        self,
    ) -> None:
        listener = RawTaskEventListener(
            store=self.store,
            run_id=self.run.run_id,
            attempt_id=self.attempt.attempt_id,
            sanitizer=PrivacySanitizer(),
        )

        await listener(
            Event(
                type=EventType.TOOL_RESULT,
                payload={
                    "name": "search",
                    "status": "ok",
                    "arguments": {"query": "private"},
                    "result": "private result",
                },
            )
        )
        await listener(
            Event(
                type=EventType.MODEL_EXECUTE_AFTER,
                payload={
                    "name": "model-a",
                    "status": "done",
                    "output": "private output",
                },
            )
        )

        events = await self.store.list_events(self.run.run_id)
        attempt_events = await self.store.list_events(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )

        self.assertEqual([event.sequence for event in events], [1, 2])
        self.assertEqual(
            [event.event_id for event in events], ["id-3", "id-4"]
        )
        self.assertEqual(events, attempt_events)
        self.assertEqual(events[0].category, TaskEventCategory.TOOL)
        self.assertEqual(events[1].category, TaskEventCategory.MODEL)
        self.assertNotIn("private", str(events))

    async def test_run_level_events_are_sequenced_with_attempt_events(
        self,
    ) -> None:
        listener = RawTaskEventListener(
            store=self.store,
            run_id=self.run.run_id,
            sanitizer=PrivacySanitizer(),
        )

        await listener(Event(type=EventType.START, payload={"status": "ok"}))

        events = await self.store.list_events(self.run.run_id)
        attempt_events = await self.store.list_events(
            self.run.run_id,
            attempt_id=self.attempt.attempt_id,
        )

        self.assertEqual(events[0].sequence, 1)
        self.assertIsNone(events[0].attempt_id)
        self.assertEqual(attempt_events, ())

    async def test_store_rejects_events_for_attempts_from_other_runs(
        self,
    ) -> None:
        other_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )

        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.append_event(
                other_run.run_id,
                attempt_id=self.attempt.attempt_id,
                event_type="start",
                category=TaskEventCategory.ENGINE,
                payload={"event_type": "start"},
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_events(
                other_run.run_id,
                attempt_id=self.attempt.attempt_id,
            )


if __name__ == "__main__":
    main()
