from .dialect import parse_cron
from .error import TriggerError, TriggerErrorCode

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from pathlib import PurePosixPath
from re import fullmatch
from types import MappingProxyType
from typing import TypeAlias
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

JsonValue: TypeAlias = (
    str
    | bool
    | int
    | float
    | None
    | tuple["JsonValue", ...]
    | list["JsonValue"]
    | Mapping[str, "JsonValue"]
)

MAX_SECONDS = 315360000
SCHEMA_VERSION = 1
SCHEDULE_SEMANTICS_VERSION = 1


def integer(
    value: object,
    minimum: int,
    maximum: int,
    path: str,
    code: TriggerErrorCode = TriggerErrorCode.INVALID_CONFIG,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise TriggerError(code, path)
    return value


def timestamp(value: object, path: str = "schedule") -> datetime:
    """Parse an aware datetime without truncating fractional precision."""
    if isinstance(value, str):
        if not fullmatch(
            r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
            r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])",
            value,
        ):
            raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, path)
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise TriggerError(
                TriggerErrorCode.INVALID_SCHEDULE, path
            ) from None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, path)
    if value.utcoffset() is None:
        raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, path)
    try:
        return value.astimezone(UTC)
    except (ValueError, OverflowError):
        raise TriggerError(TriggerErrorCode.DATETIME_OVERFLOW, path) from None


def utc_text(value: datetime) -> str:
    """Encode a timestamp at fixed UTC microsecond precision."""
    return (
        timestamp(value)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CronTrigger:
    expression: str
    timezone: str = "UTC"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "expression", parse_cron(self.expression).expression
        )
        if (
            not isinstance(self.timezone, str)
            or not self.timezone
            or self.timezone in {"localtime", "posixrules"}
            or self.timezone.startswith(("posix/", "right/"))
        ):
            raise TriggerError(
                TriggerErrorCode.UNKNOWN_TIMEZONE, "schedule.timezone"
            )
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            raise TriggerError(
                TriggerErrorCode.UNKNOWN_TIMEZONE, "schedule.timezone"
            ) from None


@dataclass(frozen=True, slots=True, kw_only=True)
class IntervalTrigger:
    every_seconds: int
    start_at: datetime | None = None

    def __post_init__(self) -> None:
        integer(
            self.every_seconds,
            1,
            MAX_SECONDS,
            "schedule.every_seconds",
            TriggerErrorCode.INVALID_SCHEDULE,
        )
        if self.start_at is not None:
            object.__setattr__(
                self, "start_at", timestamp(self.start_at, "schedule.start_at")
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AtTrigger:
    at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", timestamp(self.at, "schedule.at"))


TriggerSpec: TypeAlias = CronTrigger | IntervalTrigger | AtTrigger


class MisfirePolicy(StrEnum):
    SKIP = "skip"
    LATEST = "latest"
    ALL = "all"


class OverlapPolicy(StrEnum):
    SKIP = "skip"
    ALLOW = "allow"


@dataclass(frozen=True, slots=True, kw_only=True)
class AtPolicy:
    misfire_grace_seconds: int = 3600

    def __post_init__(self) -> None:
        integer(
            self.misfire_grace_seconds,
            0,
            MAX_SECONDS,
            "policy.misfire_grace_seconds",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RecurringPolicy(AtPolicy):
    misfire: MisfirePolicy = MisfirePolicy.LATEST
    overlap: OverlapPolicy = OverlapPolicy.SKIP

    def __post_init__(self) -> None:
        AtPolicy.__post_init__(self)
        if not isinstance(self.misfire, MisfirePolicy) or not isinstance(
            self.overlap, OverlapPolicy
        ):
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "policy")


class BindingSource(StrEnum):
    SCHEDULED_AT = "scheduled_at"
    OCCURRENCE_ID = "occurrence_id"
    TRIGGER_ID = "trigger_id"
    TRIGGER_REVISION = "trigger_revision"


def pointer_parts(path: str) -> tuple[str, ...]:
    if not isinstance(path, str) or not path.startswith("/"):
        raise TriggerError(
            TriggerErrorCode.INVALID_BINDING, "input.bindings.path"
        )
    parts = path[1:].split("/")
    if any(fullmatch(r"(?:[^~]|~[01])*", part) is None for part in parts):
        raise TriggerError(
            TriggerErrorCode.INVALID_BINDING, "input.bindings.path"
        )
    return tuple(part.replace("~1", "/").replace("~0", "~") for part in parts)


@dataclass(frozen=True, slots=True, kw_only=True)
class InputBinding:
    path: str
    source: BindingSource

    def __post_init__(self) -> None:
        pointer_parts(self.path)
        if not isinstance(self.source, BindingSource):
            raise TriggerError(
                TriggerErrorCode.INVALID_BINDING, "input.bindings.source"
            )


def freeze_value(value: object) -> JsonValue:
    """Snapshot finite JSON values without retaining mutable containers."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float) and isfinite(value):
        return 0.0 if value == 0 else value
    if isinstance(value, Mapping) and all(
        isinstance(key, str) for key in value
    ):
        return MappingProxyType(
            {key: freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(freeze_value(item) for item in value)
    raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "input.value")


def pointer_value(value: object, parts: tuple[str, ...]) -> object:
    for part in parts:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        elif (
            isinstance(value, tuple)
            and fullmatch(r"0|[1-9][0-9]*", part)
            and len(part) <= len(str(len(value)))
            and int(part) < len(value)
        ):
            value = value[int(part)]
        else:
            raise TriggerError(
                TriggerErrorCode.INVALID_BINDING, "input.bindings.path"
            )
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerInput:
    value: object = None
    bindings: tuple[InputBinding, ...] = ()

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "value", freeze_value(self.value))
        except RecursionError:
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "input.value"
            ) from None
        if not isinstance(self.bindings, tuple) or not all(
            isinstance(binding, InputBinding) for binding in self.bindings
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_BINDING, "input.bindings"
            )
        paths: list[tuple[str, ...]] = []
        for binding in self.bindings:
            parts = pointer_parts(binding.path)
            if any(
                parts[: len(old)] == old or old[: len(parts)] == parts
                for old in paths
            ):
                raise TriggerError(
                    TriggerErrorCode.INVALID_BINDING, "input.bindings.path"
                )
            if isinstance(pointer_value(self.value, parts), Mapping | tuple):
                raise TriggerError(
                    TriggerErrorCode.INVALID_BINDING, "input.bindings.path"
                )
            paths.append(parts)


# Identity distinguishes omission from an explicit caller-supplied policy.
_DEFAULT_POLICY = AtPolicy()


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerConfiguration:
    name: str
    task_ref: str
    schedule: TriggerSpec
    schema_version: int = SCHEMA_VERSION
    desired_enabled: bool = True
    input: TriggerInput = field(default_factory=TriggerInput)
    policy: AtPolicy = _DEFAULT_POLICY

    def __post_init__(self) -> None:
        integer(
            self.schema_version,
            1,
            1,
            "trigger.schema_version",
            TriggerErrorCode.UNSUPPORTED_VERSION,
        )
        if not isinstance(self.name, str) or not fullmatch(
            r"[a-z][a-z0-9-]{0,62}", self.name
        ):
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "trigger.name")
        if (
            not isinstance(self.task_ref, str)
            or not self.task_ref
            or PurePosixPath(self.task_ref).is_absolute()
            or ".." in PurePosixPath(self.task_ref).parts
        ):
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "task.ref")
        if type(self.desired_enabled) is not bool or not isinstance(
            self.input, TriggerInput
        ):
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "trigger")
        if not isinstance(
            self.schedule, CronTrigger | IntervalTrigger | AtTrigger
        ):
            raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, "schedule")
        expected = (
            AtPolicy
            if isinstance(self.schedule, AtTrigger)
            else RecurringPolicy
        )
        if self.policy is _DEFAULT_POLICY:
            object.__setattr__(self, "policy", expected())
        if type(self.policy) is not expected:
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "policy")
