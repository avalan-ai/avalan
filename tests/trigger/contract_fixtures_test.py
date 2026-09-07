"""Check the Phase 0 fixture corpus without implying trigger support."""

from asyncio import run
from json import loads
from pathlib import Path
from tomllib import loads as load_toml

from pytest import mark

from avalan.flow import FlowDefinition
from avalan.flow.loader import FlowDefinitionLoader
from avalan.task.context import TaskTargetContext
from avalan.task.definition import RunMode
from avalan.task.loader import load_task_definition
from avalan.task.target import TaskValidationContext
from avalan.task.targets.flow import (
    FlowTaskTargetRunner,
    task_flow_node_registry,
)

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "contracts"
    / "triggers"
    / "fixtures"
)
_CASES = loads((_FIXTURES / "cases.json").read_text())


def test_fixture_inventory_has_all_cases_and_schedule_variants() -> None:
    """Require every fixture to have an explicit validity expectation."""
    names = [case["file"] for case in _CASES]
    assert len(names) == len(set(names))
    assert set(names) == {
        path.name for path in _FIXTURES.glob("*.trigger.toml")
    }
    valid_schedules = set()
    boundaries = set()
    for case in _CASES:
        assert set(case) == {"file", "valid", "error_code", "boundary"}
        assert isinstance(case["valid"], bool)
        value = load_toml((_FIXTURES / case["file"]).read_text())
        assert (_FIXTURES / value["task"]["ref"]).is_file()
        boundaries.add(case["boundary"])
        if case["valid"]:
            assert case["error_code"] is None
            valid_schedules.add(value["schedule"]["type"])
        else:
            assert case["error_code"].startswith("trigger.")
    assert valid_schedules == {"cron", "interval", "at"}
    assert boundaries == {"loader", "registration", "task"}


def test_fixture_policy_matrix_covers_every_recurring_combination() -> None:
    """Keep the all/skip combination explicitly valid."""
    policies = {
        (value["policy"]["misfire"], value["policy"]["overlap"])
        for path in _FIXTURES.glob("policy-*.trigger.toml")
        if (value := load_toml(path.read_text()))
    }
    assert policies == {
        (misfire, overlap)
        for misfire in ("skip", "latest", "all")
        for overlap in ("skip", "allow")
    }


@mark.parametrize(
    ("filename", "mode"),
    [
        ("queued.task.toml", RunMode.QUEUE),
        ("direct.task.toml", RunMode.DIRECT),
    ],
)
def test_reference_tasks_use_existing_loader(
    filename: str, mode: RunMode
) -> None:
    """Validate actual task fixtures through the current task loader."""
    definition = run(load_task_definition(_FIXTURES / filename))
    assert definition.run.mode == mode
    assert definition.task.name == "trigger-contract"


async def _resolve_fixture_flow(context: TaskTargetContext) -> FlowDefinition:
    return await FlowDefinitionLoader(
        registry=task_flow_node_registry(context)
    ).load(_FIXTURES / context.definition.execution.ref)


def test_flow_fixture_passes_existing_strict_target_validation() -> None:
    """Verify supported Flow capability with its actual file and registry."""
    definition = run(load_task_definition(_FIXTURES / "flow.task.toml"))
    runner = FlowTaskTargetRunner(
        ref_base=_FIXTURES,
        strict_resolver=_resolve_fixture_flow,
        execution_roots=(_FIXTURES,),
    )
    issues = run(
        runner.validate_definition(
            definition,
            TaskValidationContext(execution_roots=(_FIXTURES,)),
        )
    )
    assert issues == ()
    case = next(
        case for case in _CASES if case["file"] == "flow-task.trigger.toml"
    )
    assert case["valid"] is True
    assert case["error_code"] is None
