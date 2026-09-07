from asyncio import run
from datetime import UTC, datetime
from json import loads
from pathlib import Path
from typing import cast
from unittest.mock import patch

from pytest import mark, raises

from avalan.agent.loader import OrchestratorLoader
from avalan.flow import FlowDefinition
from avalan.flow.loader import FlowDefinitionLoader
from avalan.task.context import TaskTargetContext
from avalan.task.definition import RunMode, TaskTargetType
from avalan.task.loader import load_task_definition
from avalan.task.target import TaskValidationContext
from avalan.task.targets.flow import (
    FlowTaskTargetRunner,
    task_flow_node_registry,
)
from avalan.task.validation import validate_task_input
from avalan.trigger import (
    AtTrigger,
    TriggerConfiguration,
    TriggerError,
    TriggerErrorCode,
)
from avalan.trigger.canonical import plain_value
from avalan.trigger.loader import TriggerLoader, parse_configuration
from avalan.trigger.schedule import validate_registration_time

_FIXTURES = (
    Path(__file__).resolve().parents[2] / "docs/contracts/triggers/fixtures"
)
_CASES = loads((_FIXTURES / "cases.json").read_text())


async def _flow(context: TaskTargetContext) -> FlowDefinition:
    return await FlowDefinitionLoader(
        registry=task_flow_node_registry(context)
    ).load(_FIXTURES / context.definition.execution.ref)


async def _validate_task(
    path: Path, configuration: TriggerConfiguration
) -> None:
    definition = await load_task_definition(path)
    if definition.run.mode != RunMode.QUEUE:
        raise TriggerError(TriggerErrorCode.QUEUE_REQUIRED, "task.ref")
    # The fixtures' binding placeholders already have their final types.
    # Production hosts must validate their binding-aware schema contract.
    assert not validate_task_input(
        definition, plain_value(configuration.input.value)
    )
    if definition.execution.type == TaskTargetType.FLOW:
        runner = FlowTaskTargetRunner(
            ref_base=path.parent,
            strict_resolver=_flow,
            execution_roots=(path.parent,),
        )
        assert not await runner.validate_definition(
            definition, TaskValidationContext(execution_roots=(path.parent,))
        )
    else:
        await OrchestratorLoader.validate_agent_file(
            str(path.parent / definition.execution.ref)
        )


@mark.parametrize("case", _CASES, ids=[case["file"] for case in _CASES])
def test_configuration_contract_corpus(case: dict[str, object]) -> None:
    loader = TriggerLoader(roots=(_FIXTURES,), task_validator=_validate_task)

    async def load_case() -> TriggerConfiguration:
        configuration = await loader.load(_FIXTURES / str(case["file"]))
        if case["boundary"] == "registration":
            validate_registration_time(
                configuration.schedule, datetime(2026, 9, 7, tzinfo=UTC)
            )
        return configuration

    if case["valid"]:
        assert run(load_case()).name == "contract-trigger"
    else:
        with raises(TriggerError) as error:
            run(load_case())
        assert error.value.code.value == case["error_code"]


@mark.parametrize(
    "change",
    [
        '[trigger]\nname="x"',
        'schedule = "bad"',
        "bad = [",
        '[trigger]\nschema_version=1\nname="x"\n[task]\nref="x"\n[schedule]\ntype="never"',
    ],
)
def test_reject_malformed_and_incomplete_configuration(change: str) -> None:
    with raises(TriggerError):
        parse_configuration(change)


@mark.parametrize(
    "addition",
    [
        '\n[input]\nbindings = "bad"',
        '\n[[input.bindings]]\nsource = "now"',
        '\n[[input.bindings]]\npath="/x"\nsource=123',
    ],
)
def test_binding_tables_are_closed_and_typed(addition: str) -> None:
    source = (
        '[trigger]\nschema_version=1\nname="x"\n[task]\nref="x"\n'
        '[schedule]\ntype="interval"\nevery_seconds=1\n'
    )
    with raises(TriggerError) as error:
        parse_configuration(source + addition)
    assert error.value.code == TriggerErrorCode.INVALID_BINDING


@mark.parametrize(
    "form",
    [
        '[schedule]\ntype="at"\nat=2026-10-01T12:00:00.1234567Z',
        'schedule = { type="at", at=2026-10-01T12:00:00.1234567Z }',
        'schedule.type="interval"\nschedule.every_seconds=1\nschedule.start_at=2026-10-01T12:00:00.1234567Z',
    ],
)
def test_native_toml_excess_precision_is_not_silently_truncated(
    form: str,
) -> None:
    # Keep inline/dotted forms at the document root.
    source = (
        form + '\n[trigger]\nschema_version=1\nname="x"\n[task]\nref="x"\n'
    )
    with raises(TriggerError) as error:
        parse_configuration(source)
    assert error.value.code == TriggerErrorCode.INVALID_SCHEDULE


def test_precision_scanner_does_not_parse_strings_or_comments() -> None:
    source = (
        '''[trigger]
schema_version=1
name="x"
[task]
ref="x"
[input]
value = [
"2026-10-01T12:00:00.1234567Z",
'2026-10-01T12:00:00.1234567Z',
"""date\n2026-10-01T12:00:00.1234567Z""", '''
        + "'''literal 12:00:00.1234567'''"
        + """]
# 2026-10-01T12:00:00.1234567Z
[schedule]
type="at"
at=2026-10-01T12:00:00.123456Z
"""
    )
    result = parse_configuration(source + "# comment without newline")
    assert isinstance(result.schedule, AtTrigger)
    assert result.schedule.at.microsecond == 123456
    escaped = source.replace(
        '"2026-10-01T12:00:00.1234567Z"',
        '"quoted \\"2026-10-01T12:00:00.1234567Z\\""',
    )
    assert parse_configuration(escaped).name == "x"


async def _accept(path: Path, configuration: TriggerConfiguration) -> None:
    assert path.is_file() and configuration.name


def test_loader_confines_source_and_task_to_trusted_roots(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = (_FIXTURES / "cron.trigger.toml").read_text()
    (root / "trigger.toml").write_text(source)
    (root / "queued.task.toml").write_text("placeholder")
    loader = TriggerLoader(roots=(root,), task_validator=_accept)
    assert run(loader.load(root / "trigger.toml")).name == "contract-trigger"
    assert (
        run(loader.loads(source, source_path=root / "unsaved.toml")).name
        == "contract-trigger"
    )
    for path in (root / "missing.toml", root, tmp_path / "outside.toml"):
        with raises(TriggerError):
            run(loader.load(path))
    outside = tmp_path / "outside.toml"
    outside.write_text("outside")
    (root / "queued.task.toml").unlink()
    (root / "queued.task.toml").symlink_to(outside)
    with raises(TriggerError):
        run(loader.load(root / "trigger.toml"))
    with (
        patch(
            "avalan.trigger.loader.read_text",
            side_effect=PermissionError("private"),
        ),
        raises(TriggerError) as error,
    ):
        run(loader.load(root / "trigger.toml"))
    assert "private" not in str(error.value)


def test_host_validation_failure_is_not_swallowed() -> None:
    async def reject(path: Path, configuration: TriggerConfiguration) -> None:
        raise TriggerError(TriggerErrorCode.CAPABILITY_UNAVAILABLE, "task.ref")

    loader = TriggerLoader(roots=(_FIXTURES,), task_validator=reject)
    with raises(TriggerError) as error:
        run(loader.load(_FIXTURES / "cron.trigger.toml"))
    assert error.value.code == TriggerErrorCode.CAPABILITY_UNAVAILABLE


def test_static_float_input_is_preserved() -> None:
    source = (_FIXTURES / "cron.trigger.toml").read_text()
    configuration = parse_configuration(
        source.replace("revision = 0", "revision = 1.25")
    )
    assert (
        cast(dict[str, object], plain_value(configuration.input.value))[
            "revision"
        ]
        == 1.25
    )


def test_enabled_requires_a_boolean() -> None:
    source = (_FIXTURES / "cron.trigger.toml").read_text()
    with raises(TriggerError) as error:
        parse_configuration(
            source.replace(
                "schema_version = 1", "schema_version = 1\nenabled = 1"
            )
        )
    assert error.value.code == TriggerErrorCode.INVALID_CONFIG


@mark.parametrize("quote", ['"', "'"])
@mark.parametrize("closing_count", [4, 5])
@mark.parametrize("fraction", ["123456", "1234567"])
def test_multiline_closing_quote_runs_preserve_precision_validation(
    quote: str, closing_count: int, fraction: str
) -> None:
    source = (
        '[trigger]\nschema_version=1\nname="x"\n[task]\nref="x"\n'
        "[input]\nvalue = "
        + quote * 3
        + "ending quote"
        + quote * closing_count
        + '\n[schedule]\ntype="at"\nat=2026-10-01T12:00:00.'
        + fraction
        + "Z\n"
    )
    if len(fraction) == 7:
        with raises(TriggerError) as error:
            parse_configuration(source)
        assert error.value.code == TriggerErrorCode.INVALID_SCHEDULE
    else:
        parsed = parse_configuration(source)
        assert parsed.input.value == "ending quote" + quote * (
            closing_count - 3
        )
        assert isinstance(parsed.schedule, AtTrigger)
        assert parsed.schedule.at.microsecond == 123456
