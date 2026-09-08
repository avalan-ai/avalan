from ..filesystem import read_text
from .definition import (
    AtPolicy,
    AtTrigger,
    BindingSource,
    CronTrigger,
    InputBinding,
    IntervalTrigger,
    MisfirePolicy,
    OverlapPolicy,
    RecurringPolicy,
    TriggerConfiguration,
    TriggerInput,
    TriggerSpec,
    integer,
    timestamp,
)
from .error import TriggerError, TriggerErrorCode

from collections.abc import Awaitable, Callable
from datetime import date, datetime, time
from math import isfinite
from pathlib import Path
from re import search
from tomllib import TOMLDecodeError, loads
from typing import TypeAlias

TomlValue: TypeAlias = (
    str
    | bool
    | int
    | float
    | date
    | datetime
    | time
    | list["TomlValue"]
    | dict[str, "TomlValue"]
)

TriggerTaskValidator = Callable[[Path, TriggerConfiguration], Awaitable[None]]


def _table(
    value: object, fields: set[str], required: set[str], path: str
) -> dict[str, TomlValue]:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - fields
    ):
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, path)
    return value


def _text(
    value: object,
    path: str,
    code: TriggerErrorCode = TriggerErrorCode.INVALID_CONFIG,
) -> str:
    if not isinstance(value, str):
        raise TriggerError(code, path)
    return value


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "trigger.enabled")
    return value


def parse_configuration(source: str) -> TriggerConfiguration:
    """Parse declarative TOML without granting task execution authority."""
    assert isinstance(source, str)
    try:
        data = loads(source, parse_float=_finite_float)
    except (TOMLDecodeError, ValueError):
        raise TriggerError(
            TriggerErrorCode.INVALID_CONFIG, "configuration"
        ) from None
    _reject_timestamp_precision(source)
    top = _table(
        data,
        {"trigger", "task", "input", "schedule", "policy"},
        {"trigger", "task", "schedule"},
        "configuration",
    )
    trigger = _table(
        top["trigger"],
        {"schema_version", "name", "enabled"},
        {"schema_version", "name"},
        "trigger",
    )
    task = _table(top["task"], {"ref"}, {"ref"}, "task")
    schedule = _table(
        top["schedule"],
        {"type", "expression", "timezone", "every_seconds", "start_at", "at"},
        {"type"},
        "schedule",
    )
    spec: TriggerSpec
    match schedule["type"]:
        case "cron":
            _table(
                schedule,
                {"type", "expression", "timezone"},
                {"type", "expression"},
                "schedule",
            )
            spec = CronTrigger(
                expression=_text(
                    schedule["expression"],
                    "schedule.expression",
                    TriggerErrorCode.INVALID_SCHEDULE,
                ),
                timezone=_text(
                    schedule.get("timezone", "UTC"),
                    "schedule.timezone",
                    TriggerErrorCode.UNKNOWN_TIMEZONE,
                ),
            )
        case "interval":
            _table(
                schedule,
                {"type", "every_seconds", "start_at"},
                {"type", "every_seconds"},
                "schedule",
            )
            spec = IntervalTrigger(
                every_seconds=integer(
                    schedule["every_seconds"],
                    1,
                    315360000,
                    "schedule.every_seconds",
                    TriggerErrorCode.INVALID_SCHEDULE,
                ),
                start_at=(
                    timestamp(schedule["start_at"], "schedule.start_at")
                    if "start_at" in schedule
                    else None
                ),
            )
        case "at":
            _table(schedule, {"type", "at"}, {"type", "at"}, "schedule")
            spec = AtTrigger(at=timestamp(schedule["at"], "schedule.at"))
        case _:
            raise TriggerError(
                TriggerErrorCode.INVALID_SCHEDULE, "schedule.type"
            )
    policy_fields = (
        {"misfire_grace_seconds"}
        if isinstance(spec, AtTrigger)
        else {"misfire", "overlap", "misfire_grace_seconds"}
    )
    policy = _table(top.get("policy", {}), policy_fields, set(), "policy")
    try:
        grace = integer(
            policy.get("misfire_grace_seconds", 3600),
            0,
            315360000,
            "policy.misfire_grace_seconds",
        )
        parsed_policy = (
            AtPolicy(misfire_grace_seconds=grace)
            if isinstance(spec, AtTrigger)
            else RecurringPolicy(
                misfire_grace_seconds=grace,
                misfire=MisfirePolicy(
                    _text(policy.get("misfire", "latest"), "policy.misfire")
                ),
                overlap=OverlapPolicy(
                    _text(policy.get("overlap", "skip"), "policy.overlap")
                ),
            )
        )
    except ValueError:
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "policy") from None
    input_table = _table(
        top.get("input", {}), {"value", "bindings"}, set(), "input"
    )
    raw_bindings = input_table.get("bindings", [])
    if not isinstance(raw_bindings, list):
        raise TriggerError(TriggerErrorCode.INVALID_BINDING, "input.bindings")
    bindings: list[InputBinding] = []
    for raw in raw_bindings:
        try:
            binding = _table(
                raw, {"path", "source"}, {"path", "source"}, "input.bindings"
            )
            bindings.append(
                InputBinding(
                    path=_text(binding["path"], "input.bindings.path"),
                    source=BindingSource(
                        _text(binding["source"], "input.bindings.source")
                    ),
                )
            )
        except ValueError:
            raise TriggerError(
                TriggerErrorCode.INVALID_BINDING, "input.bindings"
            ) from None
    return TriggerConfiguration(
        schema_version=integer(
            trigger["schema_version"],
            1,
            1,
            "trigger.schema_version",
            TriggerErrorCode.UNSUPPORTED_VERSION,
        ),
        name=_text(trigger["name"], "trigger.name"),
        desired_enabled=_boolean(trigger.get("enabled", True)),
        task_ref=_text(task["ref"], "task.ref"),
        schedule=spec,
        input=TriggerInput(
            value=input_table.get("value"), bindings=tuple(bindings)
        ),
        policy=parsed_policy,
    )


def _finite_float(value: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError("nonfinite input")
    return result


def _reject_timestamp_precision(source: str) -> None:
    """Inspect bare tokens; tomllib otherwise truncates native datetimes."""
    # Strings and comments are skipped lexically, so timestamp-looking
    # application strings are not reinterpreted as schedule values.
    index = 0
    while index < len(source):
        char = source[index]
        if char == "#":
            newline = source.find("\n", index)
            index = len(source) if newline < 0 else newline + 1
        elif char in ("'", '"'):
            delimiter = char * (3 if source.startswith(char * 3, index) else 1)
            index += len(delimiter)
            while index < len(source):
                if char == '"' and source[index] == "\\":
                    index += 2
                elif source.startswith(delimiter, index):
                    index += len(delimiter)
                    if len(delimiter) == 3:
                        # TOML permits one or two content quotes immediately
                        # before the closing triple delimiter (runs of 4/5).
                        while index < len(source) and source[index] == char:
                            index += 1
                    break
                else:
                    index += 1
        else:
            end = index + 1
            while end < len(source) and source[end] not in " \t\r\n,]}#\"'":
                end += 1
            if search(
                r"[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{7}", source[index:end]
            ):
                raise TriggerError(
                    TriggerErrorCode.INVALID_SCHEDULE, "schedule"
                )
            index = end


class TriggerLoader:
    """Load within trusted roots and require host task validation."""

    def __init__(
        self, *, roots: tuple[Path, ...], task_validator: TriggerTaskValidator
    ) -> None:
        assert roots and all(
            isinstance(root, Path) and root.is_absolute() for root in roots
        )
        assert callable(task_validator)
        self._roots = tuple(root.resolve() for root in roots)
        self._task_validator = task_validator

    def _path(self, value: Path, *, directory: bool = False) -> Path:
        try:
            resolved = value.resolve(strict=True)
        except (OSError, RuntimeError):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "task.ref"
            ) from None
        if not any(
            resolved.is_relative_to(root) for root in self._roots
        ) or not (resolved.is_dir() if directory else resolved.is_file()):
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "task.ref")
        return resolved

    async def load(self, path: Path) -> TriggerConfiguration:
        """Read a trigger file and validate its referenced queued task."""
        source_path = self._path(path)
        try:
            source = await read_text(source_path)
        except (OSError, UnicodeError):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "configuration"
            ) from None
        return await self.loads(source, source_path=source_path)

    async def loads(
        self, source: str, *, source_path: Path
    ) -> TriggerConfiguration:
        """Load text using a trusted source location for relative refs."""
        parent = self._path(source_path.parent, directory=True)
        configuration = parse_configuration(source)
        task_path = self._path(parent / configuration.task_ref)
        await self._task_validator(task_path, configuration)
        return configuration
