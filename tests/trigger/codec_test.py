"""Verify closed persistent envelopes and opaque trigger input."""

from .records_test import NOW, definition, event, occurrence, span, state

from dataclasses import replace
from json import dumps
from typing import cast

from pytest import mark, raises

from avalan.trigger import AtPolicy, AtTrigger, CronTrigger, IntervalTrigger
from avalan.trigger.codec import (
    TriggerRecord,
    decode_record,
    decode_schedule,
    encode_record,
    record_from_payload,
    record_payload,
    schedule_payload,
)
from avalan.trigger.error import TriggerError, TriggerErrorCode
from avalan.trigger.records import OccurrenceDisposition, TriggerStatus


@mark.parametrize(
    "record",
    [
        definition(),
        state(),
        occurrence(),
        span(),
        event(),
        replace(
            definition(),
            schedule=AtTrigger(at=NOW),
            policy=AtPolicy(),
            resolved_anchor=None,
        ),
        replace(
            definition(),
            schedule=CronTrigger(expression="* * * * *"),
            resolved_anchor=None,
        ),
        replace(
            definition(),
            schedule=IntervalTrigger(
                every_seconds=60, start_at=definition().resolved_anchor
            ),
        ),
        replace(
            state(),
            failure_count=1,
            retry_after=NOW,
            last_error_code=TriggerErrorCode.ADMISSION_RETRYABLE,
        ),
        replace(state(), status=TriggerStatus.EXHAUSTED, next_at=None),
        replace(
            occurrence(),
            disposition=OccurrenceDisposition.EXPIRED,
            run_id=None,
        ),
        replace(span(), exact_count=1),
        replace(event(), error_code=TriggerErrorCode.CONFLICT),
    ],
)
def test_record_roundtrip_is_canonical_and_opaque(
    record: TriggerRecord,
) -> None:
    encoded = encode_record(record)
    assert decode_record(encoded) == record
    assert encode_record(decode_record(encoded)) == encoded
    assert "encrypted-test-value" not in encoded
    assert '"version":1' in encoded


@mark.parametrize(
    "value",
    [
        None,
        [],
        {1: "invalid"},
        {},
        {
            "format": "avalan.trigger.state",
            "version": 1,
            "payload": {},
            "extra": None,
        },
    ],
)
def test_envelopes_reject_missing_and_unknown_fields(value: object) -> None:
    with raises(TriggerError):
        record_from_payload(value)


@mark.parametrize(
    "field,value",
    [("version", True), ("version", 2), ("format", "other"), ("format", [])],
)
def test_envelopes_reject_unknown_format_and_version(
    field: str, value: object
) -> None:
    payload = record_payload(state())
    payload[field] = value
    with raises(TriggerError) as error:
        record_from_payload(payload)
    assert error.value.code is TriggerErrorCode.UNSUPPORTED_VERSION


@mark.parametrize(
    "record", [definition(), state(), occurrence(), span(), event()]
)
def test_each_record_rejects_extra_payload_fields(
    record: TriggerRecord,
) -> None:
    payload = record_payload(record)
    data = cast(dict[str, object], payload["payload"])
    data["extra"] = "invalid"
    with raises(TriggerError):
        record_from_payload(payload)


@mark.parametrize(
    "field,value",
    [
        ("generation", True),
        ("owner_scope_id", 1),
        ("revision", 1.0),
        ("status", "unknown"),
        ("next_at", "2026-09-07T00:00:00Z"),
        ("next_at", 1),
        ("next_at", "invalid"),
    ],
)
def test_state_codec_validates_scalar_types(field: str, value: object) -> None:
    payload = record_payload(state())
    data = cast(dict[str, object], payload["payload"])
    data[field] = value
    with raises(TriggerError):
        record_from_payload(payload)


@mark.parametrize(
    "value",
    [
        {},
        {"type": "unknown"},
        {"type": "at", "at": "2026-09-07T00:00:00.000000Z", "extra": 1},
    ],
)
def test_schedule_codec_is_a_closed_union(value: object) -> None:
    with raises(TriggerError):
        decode_schedule(value)


def test_schedule_encoder_rejects_untyped_input() -> None:
    with raises(TriggerError):
        schedule_payload(cast(AtTrigger, object()))


@mark.parametrize(
    "field,value",
    [
        ("ciphertext", "!"),
        ("ciphertext", "é"),
        ("ciphertext", 1),
        ("metadata", {"key": 1}),
        ("metadata", []),
    ],
)
def test_definition_codec_rejects_malformed_ciphertext(
    field: str, value: object
) -> None:
    payload = record_payload(definition())
    data = cast(dict[str, object], payload["payload"])
    sealed = cast(dict[str, object], data["input"])
    sealed[field] = value
    with raises(TriggerError):
        record_from_payload(payload)


@mark.parametrize("value", ["", "{", '{"format":1,"format":2}', "NaN", "null"])
def test_decoder_rejects_invalid_json_and_duplicate_keys(value: str) -> None:
    with raises(TriggerError):
        decode_record(value)


@mark.parametrize("value", [None, [1]])
def test_event_codec_rejects_invalid_identifiers(value: object) -> None:
    payload = record_payload(event())
    data = cast(dict[str, object], payload["payload"])
    data["occurrence_ids"] = value
    with raises(TriggerError):
        decode_record(dumps(payload))


def test_decoder_rejects_missing_identity() -> None:
    payload = record_payload(state())
    data = cast(dict[str, object], payload["payload"])
    del data["revision"]
    with raises(TriggerError):
        record_from_payload(payload)
