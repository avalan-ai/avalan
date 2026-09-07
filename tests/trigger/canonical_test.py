from dataclasses import replace
from datetime import UTC, datetime
from json import loads
from os import environ
from pathlib import Path
from subprocess import run
from sys import executable

from avalan.trigger import (
    AtPolicy,
    AtTrigger,
    BindingSource,
    InputBinding,
    IntervalTrigger,
    TriggerConfiguration,
    TriggerInput,
)
from avalan.trigger.canonical import (
    canonical_configuration,
    configuration_hash,
    plain_value,
)
from avalan.trigger.loader import parse_configuration

_FIXTURES = (
    Path(__file__).resolve().parents[2] / "docs/contracts/triggers/fixtures"
)


def test_canonical_identity_excludes_operational_and_path_details() -> None:
    configuration = parse_configuration(
        (_FIXTURES / "bindings.trigger.toml").read_text()
    )
    changed = replace(
        configuration,
        name="another",
        desired_enabled=False,
        task_ref="other.toml",
        input=replace(
            configuration.input,
            bindings=tuple(reversed(configuration.input.bindings)),
        ),
    )
    identity = {
        "task_definition_id": "task-id",
        "execution_deployment_id": "deployment-id",
    }
    original = canonical_configuration(configuration, **identity)
    assert canonical_configuration(changed, **identity) == original
    assert configuration_hash(changed, **identity) == configuration_hash(
        configuration, **identity
    )
    assert len(configuration_hash(configuration, **identity)) == 64
    assert (
        canonical_configuration(
            configuration,
            task_definition_id="new-task",
            execution_deployment_id="deployment-id",
        )
        != original
    )
    payload = loads(original)
    assert payload["schedule_semantics_version"] == 1
    assert payload["input"]["bindings"] == sorted(
        payload["input"]["bindings"], key=lambda item: item["path"]
    )
    assert payload["policy"]["misfire"] == "latest"
    assert "enabled" not in original


def test_anchors_and_microsecond_datetimes_have_explicit_encoding() -> None:
    instant = datetime(2026, 10, 1, 12, 0, 0, 123456, tzinfo=UTC)
    config = TriggerConfiguration(
        name="example",
        task_ref="task.toml",
        schedule=IntervalTrigger(every_seconds=1),
    )
    identity = {
        "task_definition_id": "task",
        "execution_deployment_id": "deployment",
    }
    assert (
        loads(canonical_configuration(config, **identity))["schedule"][
            "start_at"
        ]
        is None
    )
    explicit = replace(
        config, schedule=IntervalTrigger(every_seconds=1, start_at=instant)
    )
    assert (
        loads(canonical_configuration(explicit, **identity))["schedule"][
            "start_at"
        ]
        == "2026-10-01T12:00:00.123456Z"
    )
    at = replace(config, schedule=AtTrigger(at=instant), policy=AtPolicy())
    assert loads(canonical_configuration(at, **identity))["schedule"] == {
        "type": "at",
        "at": "2026-10-01T12:00:00.123456Z",
    }
    assert loads(canonical_configuration(at, **identity))["policy"] == {
        "misfire_grace_seconds": 3600
    }


def test_json_snapshot_preserves_unicode_and_normalizes_negative_zero() -> (
    None
):
    configuration = TriggerConfiguration(
        name="test",
        task_ref="task.toml",
        schedule=IntervalTrigger(every_seconds=1),
        input=TriggerInput(
            value={"é": [-0.0, None, True, 1]},
            bindings=(
                InputBinding(path="/é/3", source=BindingSource.TRIGGER_ID),
            ),
        ),
    )
    result = canonical_configuration(
        configuration,
        task_definition_id="task",
        execution_deployment_id="deployment",
    )
    assert '"é":[0.0,null,true,1]' in result
    assert plain_value(configuration.input.value) == {
        "é": [0.0, None, True, 1]
    }


def test_core_and_interval_import_without_optional_dependencies() -> None:
    source = """
from importlib.abc import MetaPathFinder
from sys import meta_path, modules
class RejectOptional(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {'croniter','psycopg','psycopg_pool',
                     'prometheus_client','opentelemetry'}
        if fullname.split('.')[0] in forbidden:
            raise AssertionError(fullname)
meta_path.insert(0, RejectOptional())
from avalan.trigger import IntervalTrigger
from avalan.trigger.schedule import next_occurrence
from datetime import datetime, UTC
schedule=IntervalTrigger(every_seconds=1,start_at=datetime(2026,1,1,tzinfo=UTC))
assert next_occurrence(schedule,datetime(2025,1,1,tzinfo=UTC)).year==2026
assert 'croniter' not in modules
"""
    result = run(
        [executable, "-c", source],
        env={
            **environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
