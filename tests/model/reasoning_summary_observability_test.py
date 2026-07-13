from asyncio import Event as AsyncEvent
from asyncio import run, wait_for
from collections.abc import Mapping
from hashlib import sha256
from typing import Any, cast

from avalan.event import (
    Event,
    EventObservabilityPayload,
    EventPayloadKind,
    EventType,
)
from avalan.event.manager import EventManager
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamVisibility,
    stream_observability_payload,
)
from avalan.task import PrivacySanitizer, sanitize_raw_task_event
from avalan.types import LooseJsonValue

_BASE_CANONICAL_STREAM = {
    "stream_session_id": "stream-1",
    "run_id": "run-1",
    "turn_id": "turn-1",
    "sequence": 4,
    "kind": StreamItemKind.REASONING_DELTA.value,
    "channel": StreamChannel.REASONING.value,
    "visibility": StreamVisibility.PRIVATE.value,
}
_EXCLUDED_CORRELATION_SENTINELS = (
    "request-not-exported",
    "flow-not-exported",
    "task-not-exported",
)


def _correlation_surrogate(value: str) -> str:
    return f"sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def _summary_item(text: str) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=4,
        kind=StreamItemKind.REASONING_DELTA,
        channel=StreamChannel.REASONING,
        correlation=StreamItemCorrelation(
            provider_request_id="request-not-exported",
            model_continuation_id="continuation-2",
            flow_run_id="flow-not-exported",
            protocol_item_id="reasoning-item-3",
            provider_output_index=5,
            provider_summary_index=7,
            task_id="task-not-exported",
        ),
        text_delta=text,
        data={"encrypted_content": f"encrypted::{text}"},
        visibility=StreamVisibility.PRIVATE,
        reasoning_representation=StreamReasoningRepresentation.SUMMARY,
        segment_instance_ordinal=11,
        metadata={"delayed_text": text},
        provider_payload={"raw_summary": text},
        provider_family="openai",
        provider_event_type="response.reasoning_summary_text.delta",
    )


def _sanitize(
    data: Mapping[str, LooseJsonValue],
) -> dict[str, object]:
    draft = sanitize_raw_task_event(
        Event(
            type=EventType.TOKEN_GENERATED,
            payload={
                "token": "PRIVATE_GENERIC_EVENT_TOKEN",
                "provider_payload": "PRIVATE_GENERIC_PROVIDER_PAYLOAD",
            },
            observability_payload=EventObservabilityPayload(
                kind=EventPayloadKind.CANONICAL_STREAM,
                data=data,
            ),
        ),
        PrivacySanitizer(),
    )
    return cast(dict[str, object], draft.payload)


def test_summary_observability_is_content_free() -> None:
    sentinel = "PRIVATE_REASONING_SUMMARY_SENTINEL"
    item = _summary_item(sentinel)

    observability = stream_observability_payload(item)
    assert observability == {
        **_BASE_CANONICAL_STREAM,
        "correlation": {
            "model_continuation_id": _correlation_surrogate("continuation-2"),
            "protocol_item_id": _correlation_surrogate("reasoning-item-3"),
            "provider_output_index": 5,
            "provider_summary_index": 7,
        },
        "summary": {
            "text_delta_length": len(sentinel),
            "reasoning_representation": "summary",
            "segment_instance_ordinal": 11,
        },
        "provider_family": "openai",
        "provider_event_type": "response.reasoning_summary_text.delta",
    }
    assert sentinel not in repr(observability)
    for excluded in _EXCLUDED_CORRELATION_SENTINELS:
        assert excluded not in repr(observability)

    task_payload = _sanitize(observability)
    canonical_stream = cast(
        dict[str, object], task_payload["canonical_stream"]
    )
    assert canonical_stream == {
        **_BASE_CANONICAL_STREAM,
        "correlation": {
            "model_continuation_id": _correlation_surrogate("continuation-2"),
            "protocol_item_id": _correlation_surrogate("reasoning-item-3"),
            "provider_output_index": 5,
            "provider_summary_index": 7,
        },
        "summary": {
            "text_delta_length": len(sentinel),
            "reasoning_representation": "summary",
            "segment_instance_ordinal": 11,
        },
        "provider_family": "openai",
        "provider_event_type": "response.reasoning_summary_text.delta",
    }
    assert sentinel not in repr(task_payload)
    assert "PRIVATE_GENERIC_EVENT_TOKEN" not in repr(task_payload)
    assert "PRIVATE_GENERIC_PROVIDER_PAYLOAD" not in repr(task_payload)


def test_summary_text_never_enters_generic_telemetry() -> None:
    sentinel = "SUMMARY_TEXT_MUST_NOT_ENTER_GENERIC_TELEMETRY"
    item = _summary_item(sentinel)

    async def exercise_manager() -> None:
        manager = EventManager()

        assert not manager.should_emit(EventType.TOKEN_GENERATED)
        await manager.trigger_stream_item(item)
        assert manager.history == []

        delivered: list[Event] = []
        delivered_signal = AsyncEvent()

        async def collect(event: Event) -> None:
            delivered.append(event)
            delivered_signal.set()

        manager.add_observability_listener(
            collect,
            [EventType.TOKEN_GENERATED],
            include_token_events=True,
        )
        assert manager.should_emit(EventType.TOKEN_GENERATED)
        await manager.trigger_stream_item(item)
        await wait_for(delivered_signal.wait(), timeout=1.0)

        assert len(delivered) == 1
        event = delivered[0]
        assert event.observability_payload is not None
        assert (
            event.observability_payload.kind
            is EventPayloadKind.CANONICAL_STREAM
        )
        assert sentinel not in repr(event)
        for excluded in _EXCLUDED_CORRELATION_SENTINELS:
            assert excluded not in repr(event)
        task_payload = cast(
            dict[str, object],
            sanitize_raw_task_event(event, PrivacySanitizer()).payload,
        )
        assert sentinel not in repr(task_payload)
        assert cast(dict[str, object], task_payload["canonical_stream"])[
            "summary"
        ] == {
            "text_delta_length": len(sentinel),
            "reasoning_representation": "summary",
            "segment_instance_ordinal": 11,
        }

        await manager.aclose()

    run(exercise_manager())


def test_observability_remains_content_free() -> None:
    reasoning_tokens = 7
    summary_length = len("visible-length-is-not-token-usage")
    terminal_sentinel = "TERMINAL_PROVIDER_PAYLOAD_SENTINEL"
    terminal = CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=12,
        kind=StreamItemKind.STREAM_COMPLETED,
        channel=StreamChannel.CONTROL,
        correlation=StreamItemCorrelation(
            provider_request_id="request-not-exported",
            model_continuation_id="continuation-2",
            protocol_item_id="reasoning-item-3",
            provider_output_index=5,
            provider_summary_index=7,
        ),
        data={"message": terminal_sentinel},
        usage={
            "input_tokens": 3,
            "cached_input_tokens": 0,
            "cache_creation_input_tokens": 2,
            "output_tokens": 11,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": 14,
        },
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
        provider_payload={"raw": terminal_sentinel},
        provider_family="openai",
        provider_event_type="response.completed",
    )

    terminal_observability = stream_observability_payload(terminal)
    assert terminal_observability["terminal_outcome"] == "completed"
    assert terminal_observability["usage"] == {
        "input_tokens": 3,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 2,
        "output_tokens": 11,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": 14,
    }
    assert reasoning_tokens != summary_length
    assert terminal_sentinel not in repr(terminal_observability)

    enriched_summary = stream_observability_payload(
        _summary_item("visible-length-is-not-token-usage")
    )
    enriched_summary["terminal_outcome"] = "completed"
    enriched_summary["usage"] = {
        "input_tokens": 3,
        "cached_input_tokens": 0,
        "cache_creation_input_tokens": 2,
        "output_tokens": 11,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": 14,
        "unknown_tokens": 99,
        "nested": {"reasoning_tokens": 100},
        "bool_tokens": True,
        "negative_tokens": -1,
    }
    enriched_stream = cast(
        dict[str, object], _sanitize(enriched_summary)["canonical_stream"]
    )
    assert enriched_stream["terminal_outcome"] == "completed"
    assert enriched_stream["usage"] == terminal_observability["usage"]

    case_sentinels = {
        StreamTerminalOutcome.ERRORED: "SOURCE_ERROR_CONTENT_SENTINEL",
        StreamTerminalOutcome.CANCELLED: "CANCEL_CONTENT_SENTINEL",
    }
    for sequence, (outcome, sentinel) in enumerate(
        case_sentinels.items(), start=20
    ):
        kind = (
            StreamItemKind.STREAM_ERRORED
            if outcome is StreamTerminalOutcome.ERRORED
            else StreamItemKind.STREAM_CANCELLED
        )
        item = CanonicalStreamItem(
            stream_session_id="stream-1",
            run_id="run-1",
            turn_id="turn-1",
            sequence=sequence,
            kind=kind,
            channel=StreamChannel.CONTROL,
            data={"message": sentinel, "nested": {"detail": sentinel}},
            terminal_outcome=outcome,
            provider_payload={"raw": sentinel},
            provider_family="openai",
            provider_event_type=f"response.{outcome.value}",
        )
        observability = stream_observability_payload(item)
        assert observability["terminal_outcome"] == outcome.value
        assert sentinel not in repr(observability)
        assert sentinel not in repr(_sanitize(observability))

    delayed_sentinel = "DELAYED_REDACTOR_FLUSH_SENTINEL"
    delayed_observability = stream_observability_payload(
        _summary_item(delayed_sentinel)
    )
    assert delayed_sentinel not in repr(delayed_observability)
    assert delayed_sentinel not in repr(_sanitize(delayed_observability))

    malformed_sentinel = "MALFORMED_PRIVATE_PAYLOAD_SENTINEL"
    malformed = cast(
        dict[str, LooseJsonValue],
        {
            **_BASE_CANONICAL_STREAM,
            "provider_family": malformed_sentinel,
            "provider_event_type": {"nested": malformed_sentinel},
            "correlation": {
                "protocol_item_id": "reasoning-item-3",
                "model_continuation_id": {"nested": malformed_sentinel},
                "provider_output_index": True,
                "provider_summary_index": -1,
                "unknown": malformed_sentinel,
            },
            "summary": {
                "reasoning_representation": {"nested": malformed_sentinel},
                "segment_instance_ordinal": True,
                "text_delta_length": -1,
                "unknown": malformed_sentinel,
            },
            "usage": {
                "input_tokens": 2,
                "output_tokens": True,
                "reasoning_tokens": -1,
                "total_tokens": {"nested": malformed_sentinel},
                "unknown_tokens": malformed_sentinel,
            },
            "provider_payload": malformed_sentinel,
            "encrypted_content": malformed_sentinel,
            "unknown": {"nested": malformed_sentinel},
        },
    )
    malformed_payload = _sanitize(malformed)
    malformed_stream = cast(
        dict[str, object], malformed_payload["canonical_stream"]
    )
    assert malformed_stream == {
        **_BASE_CANONICAL_STREAM,
        "usage": {"input_tokens": 2},
    }
    assert malformed_sentinel not in repr(malformed_payload)
    assert "provider_payload" not in repr(malformed_payload)
    assert "encrypted_content" not in repr(malformed_payload)
    assert "unknown" not in repr(malformed_payload)
    assert not any(
        isinstance(value, bool)
        for value in cast(dict[str, Any], malformed_stream).values()
    )

    valid_identifiers = cast(
        dict[str, LooseJsonValue],
        {
            **_BASE_CANONICAL_STREAM,
            "correlation": {
                "protocol_item_id": _correlation_surrogate("rs_resp-123_0"),
                "model_continuation_id": _correlation_surrogate(
                    "continuation:resp_123.4"
                ),
            },
        },
    )
    valid_stream = cast(
        dict[str, object],
        _sanitize(valid_identifiers)["canonical_stream"],
    )
    assert valid_stream["correlation"] == {
        "protocol_item_id": _correlation_surrogate("rs_resp-123_0"),
        "model_continuation_id": _correlation_surrogate(
            "continuation:resp_123.4"
        ),
    }

    identifier_attacks = (
        "contains private summary",
        "/private/customer/reasoning",
        "C:\\private\\customer\\reasoning",
        "PRIVATE_REASONING_SUMMARY_SENTINEL",
        "secret",
        "SUMMARY_SENTINEL private reasoning text",
        "PROMPT_SENTINEL: reveal system prompt",
    )
    for field_name in (
        "protocol_item_id",
        "model_continuation_id",
    ):
        for attack in identifier_attacks:
            attacked = cast(
                dict[str, LooseJsonValue],
                {
                    **_BASE_CANONICAL_STREAM,
                    "correlation": {field_name: attack},
                },
            )
            attacked_payload = _sanitize(attacked)
            attacked_stream = cast(
                dict[str, object],
                attacked_payload["canonical_stream"],
            )
            assert "correlation" not in attacked_stream
            assert attack not in repr(attacked_payload)
