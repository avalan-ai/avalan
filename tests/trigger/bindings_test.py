from .records_test import NOW, OWNER, definition

from datetime import timedelta, timezone

from pytest import raises

from avalan.trigger.bindings import bind_input
from avalan.trigger.definition import BindingSource, InputBinding, TriggerInput
from avalan.trigger.error import TriggerError
from avalan.trigger.records import occurrence_id


def test_materialize_all_explicit_sources_without_mutating_input() -> None:
    original = {
        "a/b": [{"~key": None}, False, 0, "placeholder"],
        "literal": "${scheduled_at} {{ trigger_id }}",
    }
    value = TriggerInput(
        value=original,
        bindings=tuple(
            InputBinding(path=path, source=source)
            for path, source in (
                ("/a~1b/0/~0key", BindingSource.SCHEDULED_AT),
                ("/a~1b/1", BindingSource.OCCURRENCE_ID),
                ("/a~1b/2", BindingSource.TRIGGER_REVISION),
                ("/a~1b/3", BindingSource.TRIGGER_ID),
            )
        ),
    )
    first = bind_input(value, definition(), NOW)
    assert first == {
        "a/b": (
            {"~key": "2026-09-07T00:00:00.000000Z"},
            occurrence_id(OWNER, "trigger", 1, NOW),
            1,
            "trigger",
        ),
        "literal": original["literal"],
    }
    assert original["a/b"] == [{"~key": None}, False, 0, "placeholder"]
    assert first == bind_input(
        value, definition(), NOW.astimezone(timezone(timedelta(hours=3)))
    )
    assert first != bind_input(
        value, definition(), NOW + timedelta(seconds=60)
    )


def test_unbound_root_scalars_and_invalid_timestamp() -> None:
    for value in (None, True, 12, 1.25, "${occurrence_id}"):
        assert (
            bind_input(TriggerInput(value=value), definition(), NOW) == value
        )
    with raises(TriggerError):
        bind_input(TriggerInput(), definition(), NOW.replace(tzinfo=None))
