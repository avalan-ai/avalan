"""Encode closed versioned trigger records without exposing plaintext input."""

from .definition import (
    AtPolicy,
    AtTrigger,
    CronTrigger,
    IntervalTrigger,
    MisfirePolicy,
    OverlapPolicy,
    RecurringPolicy,
    TriggerSpec,
    timestamp,
    utc_text,
)
from .error import TriggerError, TriggerErrorCode
from .records import (
    OccurrenceDisposition,
    OwnerScopeId,
    TriggerCoverageSpan,
    TriggerDefinition,
    TriggerEvent,
    TriggerEventKind,
    TriggerOccurrence,
    TriggerSealedInput,
    TriggerState,
    TriggerStatus,
)
from .schedule import ScheduleProvenance

from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from json import JSONDecodeError, dumps, loads
from typing import TypeAlias, TypeVar

TriggerRecord: TypeAlias = (
    TriggerDefinition
    | TriggerState
    | TriggerOccurrence
    | TriggerCoverageSpan
    | TriggerEvent
)
_Enum = TypeVar("_Enum", bound=StrEnum)
_IDENTITY = {"owner_scope_id", "trigger_id", "revision"}


def _mapping(
    value: object, keys: set[str] | None = None
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "record")
    if keys is not None and set(value) != keys:
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "record.fields")
    return value


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "record.string")
    return value


def _int(value: object) -> int:
    if type(value) is not int:
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "record.integer")
    return value


def _enum(enum: type[_Enum], value: object) -> _Enum:
    try:
        return enum(_str(value))
    except ValueError:
        raise TriggerError(
            TriggerErrorCode.INVALID_CONFIG, "record.enum"
        ) from None


def _time(value: object) -> datetime:
    text = _str(value)
    parsed = timestamp(text, "record.timestamp")
    if utc_text(parsed) != text:
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "record.timestamp")
    return parsed


def _optional_time(value: object) -> datetime | None:
    return None if value is None else _time(value)


def _identifiers(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise TriggerError(
            TriggerErrorCode.INVALID_CONFIG, "record.identifiers"
        )
    return tuple(_str(item) for item in value)


def _sealed(value: TriggerSealedInput) -> dict[str, object]:
    return {
        "ciphertext": b64encode(value.ciphertext).decode("ascii"),
        "key_id": value.key_id,
        "algorithm": value.algorithm,
        "metadata": dict(value.metadata),
    }


def _decode_sealed(value: object) -> TriggerSealedInput:
    data = _mapping(value, {"ciphertext", "key_id", "algorithm", "metadata"})
    try:
        ciphertext = b64decode(_str(data["ciphertext"]), validate=True)
    except (Base64Error, ValueError):
        raise TriggerError(
            TriggerErrorCode.INVALID_CONFIG, "record.input"
        ) from None
    metadata = _mapping(data["metadata"])
    return TriggerSealedInput(
        ciphertext=ciphertext,
        key_id=_str(data["key_id"]),
        algorithm=_str(data["algorithm"]),
        metadata={key: _str(item) for key, item in metadata.items()},
    )


def schedule_payload(schedule: TriggerSpec) -> dict[str, object]:
    match schedule:
        case CronTrigger():
            return {
                "type": "cron",
                "expression": schedule.expression,
                "timezone": schedule.timezone,
            }
        case IntervalTrigger():
            return {
                "type": "interval",
                "every_seconds": schedule.every_seconds,
                "start_at": (
                    utc_text(schedule.start_at) if schedule.start_at else None
                ),
            }
        case AtTrigger():
            return {"type": "at", "at": utc_text(schedule.at)}
    raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, "record.schedule")


def decode_schedule(value: object) -> TriggerSpec:
    data = _mapping(value)
    match data.get("type"):
        case "cron":
            data = _mapping(data, {"type", "expression", "timezone"})
            return CronTrigger(
                expression=_str(data["expression"]),
                timezone=_str(data["timezone"]),
            )
        case "interval":
            data = _mapping(data, {"type", "every_seconds", "start_at"})
            return IntervalTrigger(
                every_seconds=_int(data["every_seconds"]),
                start_at=_optional_time(data["start_at"]),
            )
        case "at":
            data = _mapping(data, {"type", "at"})
            return AtTrigger(at=_time(data["at"]))
    raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, "record.schedule")


def policy_payload(policy: AtPolicy) -> dict[str, object]:
    data: dict[str, object] = {
        "misfire_grace_seconds": policy.misfire_grace_seconds
    }
    if isinstance(policy, RecurringPolicy):
        data.update(misfire=policy.misfire.value, overlap=policy.overlap.value)
    return data


def decode_policy(value: object, schedule: TriggerSpec) -> AtPolicy:
    if isinstance(schedule, AtTrigger):
        data = _mapping(value, {"misfire_grace_seconds"})
        return AtPolicy(
            misfire_grace_seconds=_int(data["misfire_grace_seconds"])
        )
    data = _mapping(value, {"misfire_grace_seconds", "misfire", "overlap"})
    return RecurringPolicy(
        misfire_grace_seconds=_int(data["misfire_grace_seconds"]),
        misfire=_enum(MisfirePolicy, data["misfire"]),
        overlap=_enum(OverlapPolicy, data["overlap"]),
    )


def record_payload(record: TriggerRecord) -> dict[str, object]:
    """Return one tagged envelope with an opaque encrypted input payload."""
    data: dict[str, object] = {
        "owner_scope_id": record.owner_scope_id.value,
        "trigger_id": record.trigger_id,
        "revision": record.revision,
    }
    match record:
        case TriggerDefinition():
            tag = "definition"
            data.update(
                name=record.name,
                task_definition_id=record.task_definition_id,
                execution_deployment_id=record.execution_deployment_id,
                semantic_hash=record.semantic_hash,
                schedule=schedule_payload(record.schedule),
                input=_sealed(record.input),
                policy=policy_payload(record.policy),
                registered_at=utc_text(record.registered_at),
                resolved_anchor=(
                    utc_text(record.resolved_anchor)
                    if record.resolved_anchor
                    else None
                ),
                schema_version=record.schema_version,
                schedule_semantics_version=record.schedule_semantics_version,
                schedule_provenance={
                    "semantics_version": (
                        record.schedule_provenance.semantics_version
                    ),
                    "parser_version": (
                        record.schedule_provenance.parser_version
                    ),
                    "timezone_data": record.schedule_provenance.timezone_data,
                },
            )
        case TriggerState():
            tag = "state"
            data.update(
                generation=record.generation,
                status=record.status.value,
                next_at=utc_text(record.next_at) if record.next_at else None,
                last_processed_at=utc_text(record.last_processed_at),
                retry_after=(
                    utc_text(record.retry_after)
                    if record.retry_after
                    else None
                ),
                failure_count=record.failure_count,
                last_error_code=(
                    record.last_error_code.value
                    if record.last_error_code
                    else None
                ),
            )
        case TriggerOccurrence():
            tag = "occurrence"
            data.update(
                occurrence_id=record.occurrence_id,
                scheduled_at=utc_text(record.scheduled_at),
                decided_at=utc_text(record.decided_at),
                disposition=record.disposition.value,
                run_id=record.run_id,
            )
        case TriggerCoverageSpan():
            tag = "span"
            data.update(
                span_id=record.span_id,
                first_at=utc_text(record.first_at),
                until_at=utc_text(record.until_at),
                decided_at=utc_text(record.decided_at),
                disposition=record.disposition.value,
                exact_count=record.exact_count,
            )
        case TriggerEvent():
            tag = "event"
            data.update(
                generation=record.generation,
                event_id=record.event_id,
                kind=record.kind.value,
                recorded_at=utc_text(record.recorded_at),
                occurrence_ids=list(record.occurrence_ids),
                span_ids=list(record.span_ids),
                error_code=(
                    record.error_code.value if record.error_code else None
                ),
            )
    return {"format": "avalan.trigger." + tag, "version": 1, "payload": data}


def record_from_payload(value: object) -> TriggerRecord:
    """Reject unknown tags, versions and structural fields before decoding."""
    envelope = _mapping(value, {"format", "version", "payload"})
    if type(envelope["version"]) is not int or envelope["version"] != 1:
        raise TriggerError(
            TriggerErrorCode.UNSUPPORTED_VERSION, "record.version"
        )
    tag = envelope["format"]
    if not isinstance(tag, str) or tag not in {
        "avalan.trigger." + name
        for name in ("definition", "state", "occurrence", "span", "event")
    }:
        raise TriggerError(
            TriggerErrorCode.UNSUPPORTED_VERSION, "record.format"
        )
    data = _mapping(envelope["payload"])
    if not _IDENTITY <= data.keys():
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "record.identity")
    owner = OwnerScopeId(value=_str(data["owner_scope_id"]))
    trigger = _str(data["trigger_id"])
    revision = _int(data["revision"])
    if tag == "avalan.trigger.definition":
        _mapping(
            data,
            _IDENTITY
            | {
                "name",
                "task_definition_id",
                "execution_deployment_id",
                "semantic_hash",
                "schedule",
                "input",
                "policy",
                "registered_at",
                "resolved_anchor",
                "schema_version",
                "schedule_semantics_version",
                "schedule_provenance",
            },
        )
        schedule = decode_schedule(data["schedule"])
        provenance = _mapping(
            data["schedule_provenance"],
            {"semantics_version", "parser_version", "timezone_data"},
        )
        return TriggerDefinition(
            owner_scope_id=owner,
            trigger_id=trigger,
            revision=revision,
            name=_str(data["name"]),
            task_definition_id=_str(data["task_definition_id"]),
            execution_deployment_id=_str(data["execution_deployment_id"]),
            semantic_hash=_str(data["semantic_hash"]),
            schedule=schedule,
            input=_decode_sealed(data["input"]),
            policy=decode_policy(data["policy"], schedule),
            registered_at=_time(data["registered_at"]),
            resolved_anchor=_optional_time(data["resolved_anchor"]),
            schema_version=_int(data["schema_version"]),
            schedule_semantics_version=_int(
                data["schedule_semantics_version"]
            ),
            schedule_provenance=ScheduleProvenance(
                semantics_version=_int(provenance["semantics_version"]),
                parser_version=_str(provenance["parser_version"]),
                timezone_data=_str(provenance["timezone_data"]),
            ),
        )
    if tag == "avalan.trigger.state":
        _mapping(
            data,
            _IDENTITY
            | {
                "generation",
                "status",
                "next_at",
                "last_processed_at",
                "retry_after",
                "failure_count",
                "last_error_code",
            },
        )
        return TriggerState(
            owner_scope_id=owner,
            trigger_id=trigger,
            revision=revision,
            generation=_int(data["generation"]),
            status=_enum(TriggerStatus, data["status"]),
            next_at=_optional_time(data["next_at"]),
            last_processed_at=_time(data["last_processed_at"]),
            retry_after=_optional_time(data["retry_after"]),
            failure_count=_int(data["failure_count"]),
            last_error_code=(
                _enum(TriggerErrorCode, data["last_error_code"])
                if data["last_error_code"] is not None
                else None
            ),
        )
    if tag == "avalan.trigger.occurrence":
        _mapping(
            data,
            _IDENTITY
            | {
                "occurrence_id",
                "scheduled_at",
                "decided_at",
                "disposition",
                "run_id",
            },
        )
        return TriggerOccurrence(
            owner_scope_id=owner,
            trigger_id=trigger,
            revision=revision,
            occurrence_id=_str(data["occurrence_id"]),
            scheduled_at=_time(data["scheduled_at"]),
            decided_at=_time(data["decided_at"]),
            disposition=_enum(OccurrenceDisposition, data["disposition"]),
            run_id=(
                _str(data["run_id"]) if data["run_id"] is not None else None
            ),
        )
    if tag == "avalan.trigger.span":
        _mapping(
            data,
            _IDENTITY
            | {
                "span_id",
                "first_at",
                "until_at",
                "decided_at",
                "disposition",
                "exact_count",
            },
        )
        return TriggerCoverageSpan(
            owner_scope_id=owner,
            trigger_id=trigger,
            revision=revision,
            span_id=_str(data["span_id"]),
            first_at=_time(data["first_at"]),
            until_at=_time(data["until_at"]),
            decided_at=_time(data["decided_at"]),
            disposition=_enum(OccurrenceDisposition, data["disposition"]),
            exact_count=(
                _int(data["exact_count"])
                if data["exact_count"] is not None
                else None
            ),
        )
    _mapping(
        data,
        _IDENTITY
        | {
            "generation",
            "event_id",
            "kind",
            "recorded_at",
            "occurrence_ids",
            "span_ids",
            "error_code",
        },
    )
    return TriggerEvent(
        owner_scope_id=owner,
        trigger_id=trigger,
        revision=revision,
        generation=_int(data["generation"]),
        event_id=_str(data["event_id"]),
        kind=_enum(TriggerEventKind, data["kind"]),
        recorded_at=_time(data["recorded_at"]),
        occurrence_ids=_identifiers(data["occurrence_ids"]),
        span_ids=_identifiers(data["span_ids"]),
        error_code=(
            _enum(TriggerErrorCode, data["error_code"])
            if data["error_code"] is not None
            else None
        ),
    )


def encode_record(record: TriggerRecord) -> str:
    """Encode the canonical UTF-8 JSON representation."""
    return dumps(
        record_payload(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "record.fields"
            )
        result[key] = value
    return result


def decode_record(value: str) -> TriggerRecord:
    """Decode persisted JSON without accepting duplicate fields."""
    try:
        parsed: object = loads(value, object_pairs_hook=_unique_object)
    except (JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, TriggerError):
            raise
        raise TriggerError(
            TriggerErrorCode.INVALID_CONFIG, "record.json"
        ) from None
    return record_from_payload(parsed)
