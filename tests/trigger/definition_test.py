from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from math import inf, nan
from typing import cast

from pytest import mark, raises

from avalan.trigger import (
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
    TriggerError,
    TriggerErrorCode,
    TriggerInput,
    TriggerSpec,
)
from avalan.trigger.definition import (
    freeze_value,
    pointer_parts,
    timestamp,
    utc_text,
)
from avalan.trigger.dialect import parse_cron


@mark.parametrize(
    "expression",
    [
        "",
        "* * * *",
        "* * * * * *",
        "@daily",
        "0/2 * * * *",
        "0 0 ? * *",
        "0 0 L * *",
        "0 0 * * MON#1",
        "0 0 * NOV-FEB *",
        "60 * * * *",
        "0 24 * * *",
        "0 0 0 * *",
        "0 0 * 13 *",
        "0 0 * * 8",
        "*/0 * * * *",
        "*/61 * * * *",
        "* * * * */8",
        "*/1/2 * * * *",
        "*/x * * * *",
        "0 0 1-2-3 * *",
        "0 0 JAN * *",
        "0,,1 * * * *",
        "*\u00a0* * * *",
        "* " * 513,
        1,
    ],
)
def test_reject_cron_outside_frozen_dialect(expression: object) -> None:
    with raises(TriggerError) as error:
        replace(
            CronTrigger(expression="* * * * *"),
            expression=cast(str, expression),
        )
    assert error.value.code == TriggerErrorCode.INVALID_SCHEDULE


@mark.parametrize("expression", ["0 0 31 2 *", "0 0 31 4,6,9,11 *"])
def test_prove_impossible_calendar(expression: str) -> None:
    with raises(TriggerError) as error:
        parse_cron(expression)
    assert error.value.code == TriggerErrorCode.IMPOSSIBLE_SCHEDULE


def test_cron_normalizes_names_and_preserves_literal_wildcard() -> None:
    cron = CronTrigger(expression=" 0\t9  */1  jan-mar/2 mon-fri ")
    assert cron.expression == "0 9 */1 JAN-MAR/2 MON-FRI"
    fields = parse_cron(cron.expression)
    assert fields.values[3] == (1, 3)
    assert len(fields.branches()) == 2
    assert parse_cron("0 0 31 2 MON").branches() == ("0 0 * 2 1",)
    assert parse_cron("0 0 * * 7").values[4] == (0,)


@mark.parametrize(
    "zone",
    [
        "",
        "Mars/Olympus",
        "../UTC",
        "/etc/localtime",
        "localtime",
        "posixrules",
        "posix/UTC",
        "right/UTC",
        1,
    ],
)
def test_reject_non_iana_timezone(zone: object) -> None:
    with raises(TriggerError) as error:
        replace(CronTrigger(expression="* * * * *"), timezone=cast(str, zone))
    assert error.value.code == TriggerErrorCode.UNKNOWN_TIMEZONE


@mark.parametrize(
    "value",
    [
        "2026-09-07",
        "2026-09-07T12:00:00",
        "2026-09-07T12:00:00.1234567Z",
        "2026-02-30T12:00:00Z",
        "2026-09-07T12:00:00+00:99",
        "2026-09-07T12:00:00+24:00",
        datetime(2026, 9, 7),
        1,
    ],
)
def test_timestamp_rejects_invalid_or_lossy_input(value: object) -> None:
    with raises(TriggerError) as error:
        timestamp(value)
    assert error.value.code == TriggerErrorCode.INVALID_SCHEDULE


class _NaiveZone(tzinfo):
    def dst(self, dt: datetime | None) -> None:
        return None

    def tzname(self, dt: datetime | None) -> None:
        return None

    def utcoffset(self, dt: datetime | None) -> None:
        return None


def test_timestamp_normalizes_and_checks_range() -> None:
    assert (
        utc_text(timestamp("2026-09-07T09:00:00.123456-03:00"))
        == "2026-09-07T12:00:00.123456Z"
    )
    with raises(TriggerError) as error:
        timestamp(datetime(1, 1, 1, tzinfo=timezone(timedelta(hours=1))))
    assert error.value.code == TriggerErrorCode.DATETIME_OVERFLOW
    with raises(TriggerError) as error:
        timestamp(datetime(2026, 1, 1, tzinfo=_NaiveZone()))
    assert error.value.code == TriggerErrorCode.INVALID_SCHEDULE


@mark.parametrize("seconds", [True, 0, -1, 1.5, 315360001])
def test_interval_requires_bounded_integer(seconds: object) -> None:
    with raises(TriggerError):
        replace(
            IntervalTrigger(every_seconds=1), every_seconds=cast(int, seconds)
        )


def test_interval_and_policy_types_are_frozen_and_strict() -> None:
    start = datetime(2026, 10, 1, tzinfo=timezone(timedelta(hours=3)))
    interval = IntervalTrigger(every_seconds=1, start_at=start)
    assert interval.start_at == start.astimezone(UTC)
    with raises(FrozenInstanceError):
        setattr(interval, "every_seconds", 2)
    assert AtPolicy(misfire_grace_seconds=0).misfire_grace_seconds == 0
    assert (
        RecurringPolicy(
            misfire=MisfirePolicy.ALL, overlap=OverlapPolicy.ALLOW
        ).misfire
        == MisfirePolicy.ALL
    )
    with raises(TriggerError):
        RecurringPolicy(misfire=cast(MisfirePolicy, "latest"))
    with raises(TriggerError):
        AtPolicy(misfire_grace_seconds=True)


@mark.parametrize(
    "value", [inf, -inf, nan, {1: "value"}, datetime.now(UTC), object()]
)
def test_input_accepts_only_finite_json(value: object) -> None:
    with raises(TriggerError):
        TriggerInput(value=value)


def test_input_snapshot_and_json_pointer_escapes() -> None:
    original = {"a/b": {"~": [0, -0.0, None, True, "é"]}}
    value = TriggerInput(
        value=original,
        bindings=(
            InputBinding(
                path="/a~1b/~0/0", source=BindingSource.TRIGGER_REVISION
            ),
        ),
    )
    original["a/b"]["~"][0] = 42
    assert value.value == freeze_value(
        {"a/b": {"~": [0, -0.0, None, True, "é"]}}
    )
    assert freeze_value(-0.0) == 0.0
    assert pointer_parts("/") == ("",)


@mark.parametrize("path", ["", "bad", "/~", "/~2", 1])
def test_reject_invalid_pointer(path: object) -> None:
    with raises(TriggerError):
        replace(
            InputBinding(path="/x", source=BindingSource.SCHEDULED_AT),
            path=cast(str, path),
        )


@mark.parametrize(
    "paths",
    [
        ["/missing"],
        ["/x/-"],
        ["/x/01"],
        ["/x/4"],
        ["/x/0/y"],
        ["/x"],
        ["/x/0", "/x/0"],
    ],
)
def test_binding_paths_must_be_unique_existing_leaves(
    paths: list[str],
) -> None:
    with raises(TriggerError):
        TriggerInput(
            value={"x": [1]},
            bindings=tuple(
                InputBinding(path=path, source=BindingSource.OCCURRENCE_ID)
                for path in paths
            ),
        )


def test_binding_source_and_collection_are_strict() -> None:
    with raises(TriggerError):
        InputBinding(path="/x", source=cast(BindingSource, "now"))
    with raises(TriggerError):
        TriggerInput(bindings=cast(tuple[InputBinding, ...], []))


@mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"name": "Upper"},
        {"name": 1},
        {"task_ref": "../x"},
        {"task_ref": "/x"},
        {"task_ref": ""},
        {"task_ref": 1},
        {"desired_enabled": 1},
        {"input": {}},
        {"schedule": "cron"},
        {"policy": AtPolicy()},
    ],
)
def test_configuration_rejects_invalid_contract(
    changes: dict[str, object],
) -> None:
    configuration = TriggerConfiguration(
        name="valid",
        task_ref="task.toml",
        schedule=IntervalTrigger(every_seconds=1),
    )
    with raises(TriggerError):
        TriggerConfiguration(
            name=cast(str, changes.get("name", configuration.name)),
            task_ref=cast(
                str, changes.get("task_ref", configuration.task_ref)
            ),
            schedule=cast(
                TriggerSpec, changes.get("schedule", configuration.schedule)
            ),
            schema_version=cast(int, changes.get("schema_version", 1)),
            desired_enabled=cast(bool, changes.get("desired_enabled", True)),
            input=cast(
                TriggerInput, changes.get("input", configuration.input)
            ),
            policy=cast(AtPolicy, changes.get("policy", configuration.policy)),
        )


def test_at_uses_only_at_policy() -> None:
    at = AtTrigger(at=datetime(2026, 10, 1, tzinfo=UTC))
    config = TriggerConfiguration(
        name="valid", task_ref="task.toml", schedule=at, policy=AtPolicy()
    )
    assert config.schedule == at
    omitted = TriggerConfiguration(
        name="valid", task_ref="task.toml", schedule=at
    )
    assert omitted.policy == AtPolicy(misfire_grace_seconds=3600)
    assert type(omitted.policy) is AtPolicy
    with raises(TriggerError):
        TriggerConfiguration(
            name="valid",
            task_ref="task.toml",
            schedule=at,
            policy=RecurringPolicy(),
        )


def test_cyclic_input_and_unbounded_array_index_are_safe_diagnostics() -> None:
    cycle: list[object] = []
    cycle.append(cycle)
    with raises(TriggerError) as error:
        TriggerInput(value=cycle)
    assert error.value.code == TriggerErrorCode.INVALID_CONFIG
    with raises(TriggerError) as error:
        TriggerInput(
            value={"x": [0]},
            bindings=(
                InputBinding(
                    path="/x/" + "9" * 5000, source=BindingSource.OCCURRENCE_ID
                ),
            ),
        )
    assert error.value.code == TriggerErrorCode.INVALID_BINDING
