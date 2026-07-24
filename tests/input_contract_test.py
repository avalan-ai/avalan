"""Freeze the complete structured-input contract before implementation."""

from asyncio import gather, run
from copy import deepcopy
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from re import fullmatch
from types import SimpleNamespace
from typing import Any, cast

from input_contract_fixtures import (
    AsyncBarrier,
    LocalProtocolPeer,
    ManualClock,
    OpaqueIdFactory,
    ScriptedProvider,
    ScriptedProviderCall,
    TestCorrelation,
    TestPrincipal,
)
from jsonschema import Draft202012Validator

from avalan.flow import default_flow_node_registry
from avalan.model import ModelCapabilityCatalog
from avalan.server.a2a.router import _build_agent_card
from avalan.server.entities import (
    ChatCompletionRequest,
    EngineRequest,
    MCPToolRequest,
    ResponsesRequest,
)
from avalan.server.routers.mcp import _server_capabilities
from avalan.tool.manager import ToolManager

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "input"
_CURRENT_RUNTIME_FILES = (
    "tests/agent/durable_continuation_resume_test.py",
    "tests/agent/durable_runtime_test.py",
    "tests/agent/execution_coverage_regression_test.py",
    "tests/agent/execution_wrapper_input_required_test.py",
    "tests/agent/orchestrator_contract_coverage_test.py",
    "tests/agent/orchestrator_response_contract_coverage_test.py",
    "tests/input/broker_contract_test.py",
    "tests/input/failure_matrix_task_e2e_test.py",
    "tests/input/public_interaction_e2e_test.py",
    "tests/interaction/continuation_import_test.py",
    "tests/interaction/continuation_test.py",
    "tests/interaction/interaction_store_authority_precedence_regression_test.py",
    "tests/interaction/interaction_store_conformance_test.py",
    "tests/interaction/interaction_store_contract_test.py",
    "tests/interaction/interaction_store_validation_coverage_test.py",
    "tests/interaction/stores/interaction_pgsql_e2e.py",
    "tests/interaction/stores/interaction_pgsql_store_test.py",
    "tests/model/model_capability_test.py",
    "tests/model/nlp/vendor_openai_continuation_test.py",
    "tests/model/text_generation_response_additional_test.py",
    "tests/task/client_test.py",
    "tests/task/event_test.py",
    "tests/task/queue_test.py",
    "tests/task/queues/pgsql_protocol_test.py",
    "tests/task/runner_test.py",
    "tests/task/state_test.py",
    "tests/task/store_contract_test.py",
    "tests/task/stores/in_memory_task_store_test.py",
    "tests/task/stores/pgsql_migration_test.py",
    "tests/task/stores/pgsql_store_coverage_test.py",
    "tests/task/suspension_test.py",
    "tests/task/target_registry_test.py",
    "tests/task/targets/agent_target_test.py",
    "tests/task/targets/flow_target_test.py",
    "tests/task/worker_continuation_coverage_test.py",
    "tests/task/worker_test.py",
)
_CURRENT_RUNTIME_NODE_COUNT = 436


class _AgentCardMessage:
    """Capture the exact fields passed to one A2A protobuf constructor."""

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class _AgentCardProtocol:
    """Provide the protobuf constructors used by the live card builder."""

    AgentCapabilities = _AgentCardMessage
    AgentCard = _AgentCardMessage
    AgentInterface = _AgentCardMessage
    AgentSkill = _AgentCardMessage


class _AgentCardConstants:
    """Provide the A2A constants used by the live card builder."""

    PROTOCOL_VERSION_1_0 = "1.0"
    TransportProtocol = SimpleNamespace(JSONRPC="JSONRPC")


def _fixture(name: str) -> dict[str, Any]:
    """Return one decoded contract fixture."""
    value = loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _digest(value: object) -> str:
    """Return the canonical fixture digest."""
    encoded = dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256(encoded).hexdigest()


def _reserved_capability_absent(
    advertisement: dict[str, object],
    reserved_identifier: str,
    reserved_names: tuple[str, ...],
) -> bool:
    """Return whether a live advertisement omits every reserved capability."""
    encoded = dumps(advertisement, ensure_ascii=False, sort_keys=True)
    return reserved_identifier not in encoded and all(
        f'"{name}"' not in encoded for name in reserved_names
    )


def _resolved_schema(
    schema: dict[str, Any],
    root: dict[str, Any],
) -> dict[str, Any]:
    """Resolve one local JSON Schema reference."""
    reference = schema.get("$ref")
    if reference is None:
        return schema
    assert isinstance(reference, str) and reference.startswith("#/")
    value: object = root
    for raw in reference[2:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        assert isinstance(value, dict)
        value = value[key]
    assert isinstance(value, dict)
    return value


def _selected_object_schema(
    schema: dict[str, Any],
    instance: object,
) -> dict[str, Any]:
    """Return the object branch selected by one representative instance."""
    branches = schema.get("oneOf")
    if not isinstance(branches, list):
        return schema
    selected: list[dict[str, Any]] = []
    for branch in branches:
        assert isinstance(branch, dict)
        candidate = deepcopy(branch)
        if "$defs" in schema:
            candidate["$defs"] = schema["$defs"]
        if Draft202012Validator(candidate).is_valid(instance):
            selected.append(branch)
    assert len(selected) == 1
    return selected[0]


def _first_const_path(
    schema: dict[str, Any],
    instance: object,
    root: dict[str, Any],
    path: tuple[str | int, ...] = (),
) -> tuple[str | int, ...] | None:
    """Return the first reachable constant path in one schema instance."""
    resolved = _resolved_schema(schema, root)
    if "const" in resolved:
        return path
    branches = resolved.get("oneOf")
    if isinstance(branches, list):
        selected = _selected_object_schema(resolved, instance)
        return _first_const_path(selected, instance, root, path)
    if isinstance(instance, dict):
        properties = resolved.get("properties", {})
        assert isinstance(properties, dict)
        for name, value in instance.items():
            property_schema = properties.get(name)
            if not isinstance(property_schema, dict):
                continue
            found = _first_const_path(
                property_schema,
                value,
                root,
                path + (name,),
            )
            if found is not None:
                return found
    if isinstance(instance, list):
        item_schema = resolved.get("items")
        if isinstance(item_schema, dict):
            for index, value in enumerate(instance):
                found = _first_const_path(
                    item_schema,
                    value,
                    root,
                    path + (index,),
                )
                if found is not None:
                    return found
    return None


def _replace_path(
    value: object,
    path: list[str] | tuple[str | int, ...],
    replacement: object,
) -> None:
    """Replace one nested mapping or sequence value in place."""
    assert path
    target = value
    for name in path[:-1]:
        assert isinstance(target, (dict, list))
        target = target[name]
    assert isinstance(target, (dict, list))
    target[path[-1]] = replacement


def _value_at(value: object, path: list[str]) -> object:
    """Return one nested fixture value."""
    target = value
    for name in path:
        assert isinstance(target, dict)
        target = target[name]
    return target


def _assert_exact_schema_mutations_fail(
    schema: dict[str, Any],
    example: object,
) -> None:
    """Reject missing, extra, wrong-constant, and wrong-type variants."""
    validator = Draft202012Validator(schema)
    assert validator.is_valid(example)
    assert isinstance(example, dict)
    object_schema = _selected_object_schema(schema, example)
    required = object_schema["required"]
    assert isinstance(required, list) and required
    missing = deepcopy(example)
    missing.pop(required[0])
    assert not validator.is_valid(missing)
    extra = deepcopy(example)
    extra["unexpected_contract_field"] = True
    assert not validator.is_valid(extra)
    const_path = _first_const_path(schema, example, schema)
    assert const_path
    wrong_const = deepcopy(example)
    _replace_path(wrong_const, const_path, "__wrong_constant__")
    assert not validator.is_valid(wrong_const)
    assert not validator.is_valid([])


async def _exercise_async_fixtures() -> tuple[list[str], str, str]:
    """Exercise deterministic async coordination and local transport."""
    barrier = AsyncBarrier(parties=2)
    events: list[str] = []

    async def arrive(label: str) -> None:
        events.append(f"{label}:arrived")
        await barrier.arrive_and_wait()
        events.append(f"{label}:released")

    await gather(arrive("first"), arrive("second"))
    peer = LocalProtocolPeer()
    await peer.send_to_runtime("fixture-request")
    request = await peer.receive_from_client()
    await peer.send_to_client("fixture-resolution")
    resolution = await peer.receive_from_runtime()
    return events, request, resolution


def test_requirement_catalog_is_complete() -> None:
    """Require every normative, scenario, and delivery identifier once."""
    catalog = _fixture("requirements_traceability.json")
    requirements = catalog["requirements"]
    assert isinstance(requirements, list)
    expected = [f"INPUT-N-{index:03d}" for index in range(1, 108)]
    expected.extend(f"INPUT-26.{index}" for index in range(1, 13))
    expected.extend(f"INPUT-GATE-{index:03d}" for index in range(1, 13))
    assert [requirement["id"] for requirement in requirements] == expected
    assert all(requirement["test_node_ids"] for requirement in requirements)
    assert catalog["source_sections"] == [str(index) for index in range(7, 27)]
    assert catalog["catalog_sha256"] == _digest(requirements)


def test_acceptance_manifest_lifecycle_is_monotonic() -> None:
    """Require append-only activation snapshots and explicit replacements."""
    manifest = _fixture("acceptance_manifest.json")
    nodes = manifest["nodes"]
    snapshots = manifest["activation_snapshots"]
    history = manifest["activation_history"]
    assert isinstance(nodes, list)
    assert isinstance(snapshots, list)
    assert isinstance(history, list)
    active = [
        node["node_id"] for node in nodes if node["lifecycle"] == "active"
    ]
    assert manifest["current_phase"] == snapshots[-1]["phase"]
    assert manifest["current_phase"] == 6
    assert len(active) == 814
    phase5_behavioral = [
        node
        for node in nodes
        if node["active_from_phase"] == 5
        and any(
            requirement_id.startswith(("INPUT-N-", "INPUT-26."))
            for requirement_id in node["requirement_ids"]
        )
    ]
    assert len(phase5_behavioral) == _CURRENT_RUNTIME_NODE_COUNT
    assert {
        node["node_id"].split("::", 1)[0] for node in phase5_behavioral
    } == set(_CURRENT_RUNTIME_FILES)
    phase6_nodes = [
        node
        for node in nodes
        if node["lifecycle"] == "active" and node["active_from_phase"] == 6
    ]
    assert len(phase6_nodes) == 26
    assert sum(node["lifecycle"] == "planned" for node in nodes) == 130
    assert set(active) == set(snapshots[-1]["node_ids"])
    assert len(active) == len(snapshots[-1]["node_ids"])
    assert (
        snapshots[-1]["sha256"]
        == sha256("\n".join(snapshots[-1]["node_ids"]).encode()).hexdigest()
    )
    assert [item["phase"] for item in snapshots] == sorted(
        item["phase"] for item in snapshots
    )
    assert [item["phase"] for item in history] == [
        item["phase"] for item in snapshots
    ]
    assert set(history[-1]["node_ids"]) == {
        node["id"] for node in phase6_nodes
    }
    nodes_by_id = {node["id"]: node for node in nodes}
    assert (
        not {
            "durable-acceptance-437",
            "durable-acceptance-438",
            "durable-acceptance-440",
            "durable-acceptance-441",
        }
        & nodes_by_id.keys()
    )
    assert nodes_by_id["durable-acceptance-439"]["requirement_ids"] == [
        "INPUT-N-101",
        "INPUT-N-102",
    ]
    assert "current-runtime-201" in history[4]["node_ids"]
    assert "current-runtime-201" not in history[5]["node_ids"]
    phase3_expansions = {
        expansion["node_id"]: (
            len(expansion["instance_node_ids"]),
            expansion["sha256"],
        )
        for expansion in manifest["parameter_expansions"][:3]
    }
    capability_node_prefix = "tests/input/model_capability_contract_test.py::"
    assert phase3_expansions == {
        capability_node_prefix
        + "test_requirement_input_n_022": (
            40,
            "5696c322dab99245a0c97869edd4b56fc3118e6888d0614a01383b4896d8fb63",
        ),
        capability_node_prefix
        + "test_local_capability_support_matrix": (
            7,
            "c48006724c87f566ec34c1b6fdf6f241360befe37ba1bd408e659b4faa6963fa",
        ),
        capability_node_prefix
        + "test_reserved_schema_rejects_control_and_secret_injection": (
            5,
            "b8ea0d1aa75d1a51cc0ec6f0ab72c211d9ca0d9080f9dc61ca0de73f79fc83a9",
        ),
    }
    assert manifest["replacements"] == [
        {
            "phase": 1,
            "old_node_id": (
                "tests/input_contract_test.py::test_capability_remains_absent"
            ),
            "replacement_node_ids": [
                "tests/input_contract_test.py::test_capability_remains_dormant"
            ],
            "requirement_ids": ["INPUT-GATE-012"],
            "reviewed_by": "/root",
            "evidence": (
                "Canonical types exist while all production capability"
                " registries remain unadvertised."
            ),
        },
        {
            "phase": 5,
            "old_node_id": (
                "tests/agent/orchestrator_response_convergence_coverage_test.py"
                "::OrchestratorResponseInteractionCoverageTest::"
                "test_start_task_input_requires_attached_runtime"
            ),
            "replacement_node_ids": [
                "tests/agent/"
                "orchestrator_response_convergence_coverage_test.py::"
                "OrchestratorResponseInteractionCoverageTest::"
                "test_start_task_input_requires_interaction_runtime"
            ],
            "requirement_ids": [
                "INPUT-N-002",
                "INPUT-N-019",
                "INPUT-N-044",
                "INPUT-N-107",
            ],
            "reviewed_by": "/root",
            "evidence": (
                "The corrected interaction-runtime precondition test replaces"
                " the stale attached-runtime node name while preserving"
                " INPUT-N-002, INPUT-N-019, INPUT-N-044, and INPUT-N-107"
                " coverage."
            ),
        },
    ]


def test_failure_matrix_is_complete() -> None:
    """Require the full surface-by-condition matrix and exact outcomes."""
    matrix = _fixture("failure_matrix.json")
    decisions = _fixture("contract_decisions.json")
    surfaces = matrix["surfaces"]
    conditions = matrix["conditions"]
    cells = matrix["cells"]
    assert isinstance(surfaces, list)
    assert isinstance(conditions, list)
    assert isinstance(cells, list)
    surface_ids = [surface["id"] for surface in surfaces]
    condition_ids = [condition["id"] for condition in conditions]
    assert len(surface_ids) == 84
    assert condition_ids == [f"INPUT-F-{index:02d}" for index in range(1, 16)]
    expected_pairs = {
        (condition_id, surface_id)
        for condition_id in condition_ids
        for surface_id in surface_ids
    }
    assert {(cell["condition_id"], cell["surface_id"]) for cell in cells} == (
        expected_pairs
    )
    assert len(cells) == len(expected_pairs) == 1260
    corrections = matrix["activation_schedule_corrections"]
    assert len(corrections) == 5
    assert (
        sum(correction["corrected_cell_count"] for correction in corrections)
        == 62
    )
    assert all(
        correction["corrected_cell_count"]
        == len(correction["condition_ids"]) * len(correction["surface_ids"])
        for correction in corrections
    )
    assert [
        (
            correction["reviewed_in_phase"],
            correction["previous_active_from_phase"],
            correction["corrected_active_from_phase"],
        )
        for correction in corrections
    ] == [
        (5, 5, 7),
        (5, 5, 6),
        (5, 5, 9),
        (5, 5, 12),
        (6, 6, 9),
    ]
    corrected_cells = {
        (condition_id, surface_id): correction["corrected_active_from_phase"]
        for correction in corrections
        for condition_id in correction["condition_ids"]
        for surface_id in correction["surface_ids"]
    }
    assert len(corrected_cells) == 62
    assert all(
        cell["active_from_phase"]
        == corrected_cells.get(
            (cell["condition_id"], cell["surface_id"]),
            max(
                next(
                    surface["active_from_phase"]
                    for surface in surfaces
                    if surface["id"] == cell["surface_id"]
                ),
                next(
                    condition["active_from_phase"]
                    for condition in conditions
                    if condition["id"] == cell["condition_id"]
                ),
            ),
        )
        for cell in cells
    )
    active_cells = [
        cell
        for cell in cells
        if cell["applicable"] and cell["active_from_phase"] <= 5
    ]
    assert len(active_cells) == 19
    assert {
        (cell["condition_id"], cell["surface_id"]) for cell in active_cells
    } == {
        ("INPUT-F-01", "task-target-agent-direct"),
        ("INPUT-F-01", "task-target-agent-queue"),
        *{
            (f"INPUT-F-{condition:02d}", surface)
            for condition in range(4, 12)
            for surface in (
                "task-target-agent-direct",
                "task-target-agent-queue",
            )
        },
        ("INPUT-F-10", "task-client-cancel"),
    }
    decision_surfaces = decisions["capability_matrix"]["rows"]
    assert surface_ids == [
        row["public_failure_surface"]
        for row in decision_surfaces
        if row["public_failure_surface"] is not None
    ]
    envelopes = decisions["error_status"]["public_envelope_catalog"]
    assert all(
        cell["public_result"].removeprefix("envelope=") in envelopes
        for cell in cells
        if cell["applicable"]
    )
    assert all(
        fullmatch(
            r"[a-z][a-z0-9_]*=-?[A-Za-z0-9][A-Za-z0-9._-]*",
            cell["status_or_exit"],
        )
        for cell in cells
        if cell["applicable"]
    )
    replay = [
        cell
        for cell in cells
        if cell["condition_id"] == "INPUT-F-07" and cell["applicable"]
    ]
    conflict = [
        cell
        for cell in cells
        if cell["condition_id"] == "INPUT-F-08" and cell["applicable"]
    ]
    assert replay and {cell["expected_transition"] for cell in replay} == {
        "answered->answered"
    }
    assert conflict and {cell["expected_transition"] for cell in conflict} == {
        "answered->answered"
    }
    mcp_capability_absent = next(
        cell
        for cell in cells
        if cell["condition_id"] == "INPUT-F-15"
        and cell["surface_id"] == "mcp-inbound-task"
    )
    assert mcp_capability_absent == {
        "condition_id": "INPUT-F-15",
        "surface_id": "mcp-inbound-task",
        "applicable": True,
        "active_from_phase": 10,
        "expected_transition": "running->running",
        "public_result": "envelope=mcp.ordinary_result.v1",
        "status_or_exit": "result=ordinary",
        "provider_call_count": 1,
        "domain_side_effect_count": 0,
        "negative_e2e_node": (
            "tests/input/failure_matrix_mcp_e2e_test.py::test_input_f_15"
        ),
        "non_applicability_reason": None,
        "reviewed_by": None,
    }
    digest_value = {
        key: matrix[key]
        for key in (
            "observation_window",
            "domain_side_effect_scope",
            "surfaces",
            "conditions",
            "activation_schedule_corrections",
            "cells",
        )
    }
    assert matrix["matrix_sha256"] == _digest(digest_value)


def test_deterministic_fixtures_are_reproducible() -> None:
    """Require deterministic clocks, identifiers, and provider scripts."""
    fixture = _fixture("deterministic_fixtures.json")
    clock_fixture = fixture["clock"]
    clock = ManualClock(initial=clock_fixture["initial"])
    observed_clock = [clock.now()]
    for advance in clock_fixture["advances"]:
        clock.advance(advance)
        observed_clock.append(clock.now())
    assert observed_clock == clock_fixture["expected"]
    id_fixture = fixture["id_factory"]
    factory = OpaqueIdFactory(prefix=id_fixture["prefix"])
    run_id = factory.run_id()
    turn_id = factory.turn_id()
    task_id = factory.task_id()
    agent_id = factory.agent_id()
    branch_id = factory.branch_id()
    model_call_id = factory.model_call_id()
    request_id = factory.request_id()
    continuation_id = factory.continuation_id()
    stream_session_id = factory.stream_session_id()
    user_id = factory.user_id()
    tenant_id = factory.tenant_id()
    participant_id = factory.participant_id()
    session_id = factory.session_id()
    observed_ids = [
        run_id,
        turn_id,
        task_id,
        agent_id,
        branch_id,
        model_call_id,
        request_id,
        continuation_id,
        stream_session_id,
        user_id,
        tenant_id,
        participant_id,
        session_id,
    ]
    assert observed_ids == id_fixture["expected"]
    state_revision = factory.state_revision()
    assert state_revision == id_fixture["state_revision"]
    principal = TestPrincipal(
        user_id=user_id,
        tenant_id=tenant_id,
        participant_id=participant_id,
        session_id=session_id,
    )
    assert (
        len(
            {
                principal.user_id,
                principal.tenant_id,
                principal.participant_id,
                principal.session_id,
            }
        )
        == 4
    )
    correlation = TestCorrelation(
        run_id=run_id,
        turn_id=turn_id,
        task_id=task_id,
        agent_id=agent_id,
        branch_id=branch_id,
        parent_branch_id=None,
        model_call_id=model_call_id,
        request_id=request_id,
        continuation_id=continuation_id,
        stream_session_id=stream_session_id,
        state_revision=state_revision,
    )
    assert correlation.request_id == request_id
    assert correlation.state_revision == state_revision
    events, request, resolution = run(_exercise_async_fixtures())
    assert events[:2] == ["first:arrived", "second:arrived"]
    assert set(events[2:]) == {"first:released", "second:released"}
    assert request == "fixture-request"
    assert resolution == "fixture-resolution"
    provider_fixture = fixture["provider_calls"]
    provider = ScriptedProvider(
        ScriptedProviderCall(**call) for call in provider_fixture
    )
    assert [
        run(provider.generate(call["request"])) for call in provider_fixture
    ] == [call["response"] for call in provider_fixture]
    assert provider.call_count == len(provider_fixture)


def test_contract_decisions_are_frozen() -> None:
    """Require exact bounds, protocol projections, and public schemas."""
    decisions = _fixture("contract_decisions.json")
    bounds = decisions["request_bounds"]
    assert bounds["question_count"] == {"minimum": 1, "maximum": 3}
    assert bounds["inactivity_timeout_seconds"]["trusted_activity_event"]
    assert (
        "never"
        in bounds["inactivity_timeout_seconds"]["trusted_activity_event"]
    )
    a2a = decisions["protocol_projection"]["a2a"]
    assert a2a["task_states"] == [
        "TASK_STATE_UNSPECIFIED",
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
        "TASK_STATE_FAILED",
        "TASK_STATE_CANCELED",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_REJECTED",
        "TASK_STATE_AUTH_REQUIRED",
    ]
    mcp = decisions["protocol_projection"]["mcp"]
    multiple = mcp["elicitation"]["requestedSchema"]["properties"][
        "multiple_select"
    ]
    assert multiple["type"] == "array"
    assert multiple["items"]["type"] == "string"
    assert multiple["uniqueItems"] is True
    tasks = mcp["tasks"]
    ttl_schema = tasks["params_task_schema"]["properties"]["ttl"]
    assert ttl_schema == {
        "type": "number",
        "unit": "milliseconds",
    }
    assert (
        tasks["request_type_task_capability_absent"]
        == "receiver MUST process request normally and ignore params.task"
        " augmentation"
    )
    assert tasks["generic_receiver_task_requirement"] == {
        "omission_behavior": (
            "receiver MAY require task augmentation for a request type with"
            " declared support; omission MAY return -32600"
        ),
        "omission_error": -32600,
    }
    assert tasks["tool_execution_task_support"] == {
        "absent": (
            "defaults to forbidden; attempted params.task SHOULD return -32601"
        ),
        "forbidden": "attempted params.task SHOULD return -32601",
        "optional": "client MAY invoke normally or with params.task",
        "required": (
            "client MUST invoke with params.task; omission MUST return -32601"
        ),
    }
    task_schema = tasks["task_schema"]
    create_task_schema = tasks["CreateTaskResult"]
    assert "_meta" not in task_schema["properties"]
    assert "_meta" in create_task_schema["properties"]
    assert create_task_schema["required"] == ["task"]
    assert create_task_schema["additionalProperties"] is True
    assert create_task_schema["properties"]["_meta"] == {"type": "object"}
    for schema in (
        task_schema,
        create_task_schema["properties"]["task"],
    ):
        properties = schema["properties"]
        assert properties["statusMessage"] == {"type": "string"}
        assert properties["ttl"] == {"type": ["number", "null"]}
        assert properties["pollInterval"] == {"type": "number"}
    for name, schema_name in (
        ("params_task_with_ttl", "params_task_schema"),
        ("params_task_without_ttl", "params_task_schema"),
        ("Task", "task_schema"),
        ("CreateTaskResult", "CreateTaskResult"),
    ):
        schema = tasks[schema_name]
        Draft202012Validator.check_schema(schema)
        assert Draft202012Validator(schema).is_valid(
            tasks["schema_examples"][name]
        )
    params_task_validator = Draft202012Validator(tasks["params_task_schema"])
    assert params_task_validator.is_valid({})
    assert not params_task_validator.is_valid({"ttl": None})
    assert params_task_validator.is_valid({"ttl": 0})
    zero_duration_task = deepcopy(tasks["schema_examples"]["Task"])
    zero_duration_task["ttl"] = 0
    zero_duration_task["pollInterval"] = 0
    assert Draft202012Validator(task_schema).is_valid(zero_duration_task)
    extended_create_result = deepcopy(
        tasks["schema_examples"]["CreateTaskResult"]
    )
    extended_create_result["vendorExtension"] = {"accepted": True}
    assert Draft202012Validator(create_task_schema).is_valid(
        extended_create_result
    )
    assert tasks["initial_state"] == "working"
    assert tasks["legal_transitions"] == [
        ["working", "input_required"],
        ["input_required", "working"],
        ["input_required", "completed"],
        ["input_required", "failed"],
        ["working", "completed"],
        ["working", "failed"],
        ["working", "cancelled"],
        ["input_required", "cancelled"],
    ]
    task_with_meta = deepcopy(tasks["schema_examples"]["Task"])
    task_with_meta["_meta"] = {}
    assert not Draft202012Validator(task_schema).is_valid(task_with_meta)

    extension = a2a["extension"]
    metadata_schema = extension["message_metadata_schema"]
    metadata_examples = extension["message_metadata_examples"]
    Draft202012Validator.check_schema(metadata_schema)
    assert set(metadata_examples) == {"request", "accept", "decline", "cancel"}
    for example in metadata_examples.values():
        _assert_exact_schema_mutations_fail(metadata_schema, example)
    questions = metadata_examples["request"]["questions"]
    answers = metadata_examples["accept"]["answers"]
    assert {question["question_id"] for question in questions} == set(answers)
    invalid_question = deepcopy(metadata_examples["request"])
    invalid_question["questions"][0]["unexpected"] = True
    assert not Draft202012Validator(metadata_schema).is_valid(invalid_question)
    invalid_answer = deepcopy(metadata_examples["accept"])
    invalid_answer["answers"]["confirm"] = {"untyped": True}
    assert not Draft202012Validator(metadata_schema).is_valid(invalid_answer)
    assert len(extension["accepted_answer_invariants"]) == 7

    error_status = decisions["error_status"]
    assert error_status["mcp"] == {
        "invalid_params": -32602,
        "unavailable": -32001,
        "unauthorized": -32003,
        "conflict": -32009,
        "expired": -32010,
        "receiver_task_augmentation_required": -32600,
        "tool_task_augmentation_forbidden": -32601,
        "tool_task_augmentation_required": -32601,
    }
    catalog = error_status["public_envelope_catalog"]
    examples = error_status["public_envelope_examples"]
    contract = error_status["public_envelope_catalog_contract"]
    assert set(catalog) == set(examples)
    assert len(catalog) == 108
    assert "sdk.failed.v1" in catalog
    assert contract["mutation_requirements"] == [
        "missing_required_field",
        "extra_field",
        "wrong_const",
        "wrong_type",
        "cross_field_invariant",
    ]
    for envelope_id, schema in catalog.items():
        Draft202012Validator.check_schema(schema)
        _assert_exact_schema_mutations_fail(schema, examples[envelope_id])
    mcp_extension_error = catalog["mcp.extension_required_error.v1"]
    assert (
        mcp_extension_error["properties"]["error"]["properties"]["code"][
            "const"
        ]
        == -32601
    )
    for envelope_id in (
        "mcp.task_cancelled.v1",
        "mcp.task_input_required.v1",
        "mcp.task_working.v1",
    ):
        result_schema = catalog[envelope_id]["properties"]["result"]
        public_task_schema = result_schema["properties"]["task"]
        assert "_meta" in result_schema["required"]
        assert "_meta" in result_schema["properties"]
        assert "_meta" not in public_task_schema["required"]
        assert "_meta" not in public_task_schema["properties"]
        public_task_properties = public_task_schema["properties"]
        assert public_task_properties["statusMessage"] == {"type": "string"}
        assert public_task_properties["ttl"] == {"type": ["number", "null"]}
        assert public_task_properties["pollInterval"] == {"type": "number"}
        assert examples[envelope_id]["result"]["task"]["ttl"] == 0
        assert examples[envelope_id]["result"]["task"]["pollInterval"] == 0

    ordinary_schema = catalog["mcp.ordinary_result.v1"]
    assert ordinary_schema["properties"]["result"] == {
        "type": "object",
        "additionalProperties": True,
        "description": (
            "Exact result members are defined by the ordinary underlying MCP"
            " request method."
        ),
    }

    vectors = error_status["public_envelope_cross_field_mutations"]
    assert set(vectors) == {
        "a2a.task_input_required.v1",
        "a2a.task_working.v1",
        "mcp.task_cancelled.v1",
        "mcp.task_input_required.v1",
        "mcp.task_working.v1",
    }
    for envelope_id, mutations in vectors.items():
        schema = catalog[envelope_id]
        example = examples[envelope_id]
        for mutation in mutations:
            expected_keys = {
                "invariant_id",
                "path",
                "replacement",
                ("equals_path" if "equals_path" in mutation else "expected"),
            }
            assert set(mutation) == expected_keys
            if "equals_path" in mutation:
                assert _value_at(example, mutation["path"]) == _value_at(
                    example,
                    mutation["equals_path"],
                )
            else:
                assert (
                    _value_at(example, mutation["path"])
                    == mutation["expected"]
                )
            changed = deepcopy(example)
            _replace_path(changed, mutation["path"], mutation["replacement"])
            assert Draft202012Validator(schema).is_valid(changed)
            if "equals_path" in mutation:
                assert _value_at(changed, mutation["path"]) != _value_at(
                    changed,
                    mutation["equals_path"],
                )
            else:
                assert (
                    _value_at(changed, mutation["path"])
                    != mutation["expected"]
                )
    digest_value = {
        key: value
        for key, value in decisions.items()
        if key != "contract_sha256"
    }
    assert decisions["contract_sha256"] == _digest(digest_value)


def test_no_bc_removal_inventory_is_frozen() -> None:
    """Require the exact legacy symbol and assertion removal baseline."""
    inventory = _fixture("no_bc_removals.json")
    removals = inventory["removals"]
    assert isinstance(removals, list)
    expected_ids = {
        "tool-manager-provider-coupling",
        "completion-only-results",
        "engine-agent-reusable-mutable-fields",
        "orchestrator-reusable-mutable-fields",
        "orchestrator-response-continuation-state",
        "chat-sse-fake-turn",
        "responses-sse-fake-turn",
        "responses-non-stream-fake-turn",
        "mcp-fake-turn",
        "flow-run-equals-turn",
        "a2a-task-context-correlation",
        "model-single-stream-correlation",
        "model-non-stream-correlation",
        "legacy-chat-sse-correlation-assertion",
        "legacy-responses-sse-correlation-assertion",
        "legacy-mcp-correlation-assertion",
    }
    assert {removal["id"] for removal in removals} == expected_ids
    assert all(
        (_ROOT / removal["current_path"].split(":", maxsplit=1)[0]).exists()
        for removal in removals
    )
    assert inventory["inventory_sha256"] == _digest(removals)


def test_baseline_evidence_is_complete() -> None:
    """Require explicit pending or current-tree-bound gate evidence."""
    evidence = _fixture("baseline_evidence.json")
    manifest = _fixture("acceptance_manifest.json")
    current_active_test_node_ids = [
        node["node_id"]
        for node in manifest["nodes"]
        if node["lifecycle"] == "active"
    ]
    sealed_active_test_node_ids = next(
        snapshot["node_ids"]
        for snapshot in manifest["activation_snapshots"]
        if snapshot["phase"] == 5
    )
    phase6_test_node_ids = [
        node["node_id"]
        for node in manifest["nodes"]
        if node["lifecycle"] == "active" and node["active_from_phase"] == 6
    ]
    assert evidence["implementation_owner"] == "/root"
    assert evidence["independent_reviewer"] == "/root/input_contract_audit"
    assert evidence["implementation_owner"] != evidence["independent_reviewer"]
    assert evidence["recorded_at"] == "2026-07-24T11:25:16Z"
    assert evidence["review_history"][0] == {
        "sequence": 0,
        "phase": 0,
        "role": "baseline",
        "reviewer": "/root/input_contract_audit",
        "status": "approved",
        "recorded_at": "2026-07-21T03:59:00-03:00",
        "evidence": (
            "Preserve the independent phase-0 contract review identity"
            " without rewriting historical approval."
        ),
    }
    assert [record["role"] for record in evidence["review_history"][1:5]] == [
        "semantic",
        "gate",
        "semantic",
        "gate",
    ]
    assert all(
        record["status"] == "pending"
        for record in evidence["review_history"][1:3]
    )
    assert all(
        record["status"] == "approved"
        for record in evidence["review_history"][3:5]
    )
    assert evidence["review_history"][5:7] == [
        {
            "sequence": 5,
            "phase": 2,
            "role": "semantic",
            "reviewer": "/root/broker_review",
            "status": "pending",
            "recorded_at": "2026-07-21T14:06:16-03:00",
            "evidence": (
                "Independent broker lifecycle review is in progress against"
                " the current implementation and its focused tests."
            ),
        },
        {
            "sequence": 6,
            "phase": 2,
            "role": "gate",
            "reviewer": "/root/phase2_acceptance_review",
            "status": "pending",
            "recorded_at": "2026-07-21T14:06:16-03:00",
            "evidence": (
                "Independent acceptance and evidence review remains pending"
                " until the mixed-lifecycle ownership slice and live quality"
                " results are finalized."
            ),
        },
    ]
    assert evidence["review_history"][7:9] == [
        {
            "sequence": 7,
            "phase": 2,
            "role": "semantic",
            "reviewer": "/root/phase2_acceptance_review",
            "status": "approved",
            "recorded_at": "2026-07-21T17:29:44-03:00",
            "evidence": (
                "Independent broker lifecycle and liveness review completed"
                " without unresolved findings."
            ),
        },
        {
            "sequence": 8,
            "phase": 2,
            "role": "gate",
            "reviewer": "/root/phase2_metadata_review",
            "status": "approved",
            "recorded_at": "2026-07-21T17:29:45-03:00",
            "evidence": (
                "Independent terminal acceptance, coverage, and evidence"
                " review completed without unresolved findings."
            ),
        },
    ]
    assert evidence["review_history"][9:16] == [
        {
            "sequence": 9,
            "phase": 3,
            "role": "gate",
            "reviewer": "/root/terminal_review",
            "status": "pending",
            "recorded_at": "2026-07-22T00:01:07Z",
            "evidence": (
                "Independent terminal metadata review remains pending:"
                " phase-3 workflow pins and exact metadata assertions were"
                " stale at phase 2 and require correction and validation."
            ),
        },
        {
            "sequence": 10,
            "phase": 3,
            "role": "semantic",
            "reviewer": "/root/acceptance_review",
            "status": "approved",
            "recorded_at": "2026-07-22T00:03:18Z",
            "evidence": (
                "Independent phase-3 catalog, acceptance turn-3, and"
                " batch-boundary reviews by /root/catalog_review_retry,"
                " /root/acceptance_review, and /root/batch_review_final"
                " completed with zero unresolved P1, P2, or P3 findings."
            ),
        },
        {
            "sequence": 11,
            "phase": 3,
            "role": "closure",
            "reviewer": "/root/phase3_closure_audit",
            "status": "pending",
            "recorded_at": "2026-07-22T07:43:00Z",
            "evidence": (
                "Closure review turns four and five recorded the remaining"
                " async shutdown and ToolManager identity, inventory, and"
                " rollback findings; the bounded correction lanes completed"
                " and final independent verification began."
            ),
        },
        {
            "sequence": 12,
            "phase": 3,
            "role": "closure",
            "reviewer": "/root/phase3_closure_audit",
            "status": "approved",
            "recorded_at": "2026-07-22T07:44:17Z",
            "evidence": (
                "Independent turn-six async approval and independent turn-six"
                " ToolManager approval closed both stable correction lanes"
                " with zero P1, P2, or P3 findings across exact regressions,"
                " focused suites, full relevant test trees, and scoped static"
                " checks."
            ),
        },
        {
            "sequence": 13,
            "phase": 3,
            "role": "coverage-closure",
            "reviewer": (
                "/root/phase3_closure_audit/turn3_toolmanager_readonly"
            ),
            "status": "pending",
            "recorded_at": "2026-07-22T08:33:02Z",
            "evidence": (
                "Independent coverage-closure review began after the stream"
                " dead-code/no-backward-compatibility cleanup, redundant"
                " ToolManager guard removal, and model test-only coverage"
                " lanes completed."
            ),
        },
        {
            "sequence": 14,
            "phase": 3,
            "role": "coverage-closure",
            "reviewer": (
                "/root/phase3_closure_audit/turn3_toolmanager_readonly"
            ),
            "status": "approved",
            "recorded_at": "2026-07-22T08:33:12Z",
            "evidence": (
                "Independent coverage approvals by"
                " /root/phase3_closure_audit/turn3_toolmanager_identity_fix"
                " and"
                " /root/phase3_closure_audit/turn3_toolmanager_readonly"
                " completed with zero P1, P2, or P3 findings; they validated"
                " the stream dead-code/no-backward-compatibility removal,"
                " redundant ToolManager guard removal, and model test-only"
                " coverage additions without coverage tricks."
            ),
        },
        {
            "sequence": 15,
            "phase": 3,
            "role": "gate",
            "reviewer": "/root/terminal_review",
            "status": "approved",
            "recorded_at": "2026-07-22T09:37:14Z",
            "evidence": (
                "Independent terminal review approved the first frozen"
                " Phase 3 evidence with zero P1, P2, or P3 findings after"
                " validating the authoritative full, exact-coverage,"
                " acceptance, type, lint, diff, and focused results together"
                " with the live coverage-report and normalized-tree bindings."
            ),
        },
    ]
    assert evidence["review_history"][16:18] == [
        {
            "sequence": 16,
            "phase": 4,
            "role": "semantic",
            "reviewer": "/root/execution_runtime_review",
            "status": "approved",
            "recorded_at": "2026-07-23T05:34:16Z",
            "evidence": (
                "Independent execution-runtime semantic review and the"
                " follow-up execution, orchestrator-cleanup, response-cleanup,"
                " and text-response coverage-review loops completed with zero"
                " remaining P1, P2, or P3 findings across invocation"
                " isolation, attached suspension and resumption, cancellation,"
                " concurrency, and cleanup convergence."
            ),
        },
        {
            "sequence": 17,
            "phase": 4,
            "role": "gate",
            "reviewer": "/root/execution_gate_review",
            "status": "approved",
            "recorded_at": "2026-07-23T05:47:33Z",
            "evidence": (
                "Independent execution-state terminal review by"
                " /root/execution_gate_review completed with zero P1, P2, or"
                " P3 findings after validating the ordered authoritative"
                " results, live exact-coverage artifact and source inventory,"
                " fail-closed evidence schema, neutral execution naming,"
                " unchanged skip and deselection posture, and exclusion of"
                " preserved unrelated docs/examples/skills/code content;"
                " normalized final-tree bindings are sealed atomically with"
                " this approval."
            ),
        },
    ]
    assert [
        (
            record["phase"],
            record["role"],
            record["reviewer"],
            record["status"],
        )
        for record in evidence["review_history"][18:22]
    ] == [
        (
            5,
            "semantic",
            "/root/durable_lifecycle_corrections",
            "pending",
        ),
        (5, "gate", "/root/continuation_r5_review", "pending"),
        (
            5,
            "semantic",
            "/root/durable_lifecycle_corrections",
            "approved",
        ),
        (5, "gate", "/root/continuation_r5_review", "approved"),
    ]
    assert evidence["review_history"][22:24] == [
        {
            "sequence": 22,
            "phase": 6,
            "role": "semantic",
            "reviewer": "/root/public_sdk_surface",
            "status": "approved",
            "recorded_at": "2026-07-24T06:43:30Z",
            "evidence": (
                "Independent review 53 approved the public SDK annotation"
                " correction with zero P1, P2, or P3 findings after auditing"
                " 47 SDK symbols, 74 annotated targets, and 381 leaves with"
                " zero missing root exports; six focused tests, 34 strict type"
                " fixtures, Ruff, and git diff --check passed."
            ),
        },
        {
            "sequence": 23,
            "phase": 6,
            "role": "gate",
            "reviewer": "/root/public_sdk_gate_review",
            "status": "pending",
            "recorded_at": "2026-07-24T06:43:31Z",
            "evidence": (
                "The independent final gate review is pending until the"
                " ordered lint, strict type, provisioned PostgreSQL exact"
                " coverage and acceptance, and diff commands produce"
                " current-tree evidence."
            ),
        },
    ]
    assert evidence["review_history"][24:26] == [
        {
            "sequence": 24,
            "phase": 6,
            "role": "cold-import-closure",
            "reviewer": "/root/public_sdk_surface",
            "status": "pending",
            "recorded_at": "2026-07-24T07:43:02Z",
            "evidence": (
                "Independent review 54 found zero P1, one P2, and zero P3"
                " findings at disconnected copied patch seams that made the"
                " public SDK cold import pull optional service dependencies;"
                " a bounded canonical-ownership correction began."
            ),
        },
        {
            "sequence": 25,
            "phase": 6,
            "role": "cold-import-closure",
            "reviewer": "/root/public_sdk_surface",
            "status": "approved",
            "recorded_at": "2026-07-24T07:43:14Z",
            "evidence": (
                "Independent review 55 approved the correction with zero P1,"
                " P2, or P3 findings after copied bindings were removed and"
                " tests were canonicalized; nine seam, cold-import, and"
                " failure tests, six public tests, 34 strict type fixtures,"
                " mypy, Ruff, and git diff --check passed."
            ),
        },
    ]
    assert evidence["review_history"][26:28] == [
        {
            "sequence": 26,
            "phase": 6,
            "role": "independent_public_sdk_surface_reviewer",
            "reviewer": "/root/public_sdk_surface",
            "status": "approved",
            "recorded_at": "2026-07-24T08:17:34Z",
            "evidence": (
                "Independent review #56 approved both exact-gate corrections"
                " with P1=0, P2=0, and P3=0. The exact CLI-startup and"
                " store-authority command passed 2 tests in 2.36 seconds;"
                " focused suite A passed 31 tests and 3 subtests in 4.83"
                " seconds; focused suite B passed 88 tests in 8.55 seconds;"
                " the phase-6 type verifier passed 34 fixtures; the 47-symbol"
                " and 74-target annotation audit found no missing roots and"
                " canonical run_agent; scoped seven-file mypy, scoped Ruff,"
                " git diff --check, identity and runtime probes, and the"
                " scoped Jinja CLI probe passed. Exploratory broader mypy and"
                " SQLAlchemy-blocking probes exposed pre-existing CLI typing"
                " and database-settings import boundaries and are not claimed"
                " as gate successes. This focused review does not approve the"
                " common quality gate."
            ),
        },
        {
            "sequence": 27,
            "phase": 6,
            "role": "gate-correction-closure",
            "reviewer": "/root",
            "status": "approved",
            "recorded_at": "2026-07-24T08:17:34Z",
            "evidence": (
                "Root closed the two exact-gate correction lanes after make"
                " lint completed Ruff format, Black, Ruff check, full mypy"
                " for 440 files, and verifier-script mypy for 6 files, and"
                " make typecheck-input-contract INPUT_PHASE=6 passed 34"
                " fixtures. This is a pre-gate validation record, not an"
                " independent review or common-gate approval; the exact"
                " PostgreSQL coverage and acceptance gate plus final tree and"
                " coverage bindings remain pending."
            ),
        },
    ]
    assert evidence["review_history"][28] == {
        "sequence": 28,
        "phase": 6,
        "role": "gate",
        "reviewer": "/root/public_sdk_gate_review",
        "status": "approved",
        "recorded_at": "2026-07-24T11:25:16Z",
        "evidence": (
            "Independent review #57 returned verdict APPROVE with P1=0,"
            " P2=0, and P3=0 after validating the ordered final quality"
            " results, exact JSON and XML coverage artifacts, normalized"
            " final tree, focused evidence and verifier suite, and phase-6"
            " production verifier; this append-only record seals the"
            " already-reviewed gate without changing implementation"
            " behavior."
        ),
    }
    assert evidence["review_history_sha256"] == _digest(
        evidence["review_history"]
    )
    assert evidence["review_history_phase0_sha256"] == _digest(
        evidence["review_history"][:1]
    )
    assert evidence["review_history_phase1_sha256"] == _digest(
        evidence["review_history"][:5]
    )
    assert evidence["review_history_phase2_sha256"] == _digest(
        evidence["review_history"][:9]
    )
    assert evidence["review_history_prior_sha256"] == _digest(
        evidence["review_history"][:16]
    )
    assert (
        _digest(evidence["review_history"][:7])
        == "a83a4e9545ac72c99c23d6fd316c7661f5a6bfef86c8c39a5c209ee6185a852a"
    )
    assert evidence["quality_history"] == [
        {
            "phase": 1,
            "state": "complete",
            "quality_gate_sha256": (
                "f58bd16d9bf57bb3f2972982ff8bcf19a6125715a40194effecb8141c8ebd5ea"
            ),
            "evidence_sha256": (
                "a4c16a90cf2d451b423da22ba763b50742e47f583230ded87c9997d77e1b93b8"
            ),
        },
        {
            "phase": 2,
            "state": "complete",
            "quality_gate_sha256": (
                "d004e9f765e9167d31debb7642883e774e42a03503f32f8869eb6b4e084e3953"
            ),
            "evidence_sha256": (
                "d0e276493609d2e7254c576bf50552a933e4e54cb67c9ec6e6a71f94a17f0302"
            ),
        },
        {
            "phase": 3,
            "state": "complete",
            "quality_gate_sha256": (
                "62c94da810be0c995525580c19df034b35aee2700f6e9e8fa51c69ba778e0102"
            ),
            "evidence_sha256": (
                "59788e2441bec0bd34a61ff94f8b14459ca229a37fcf693ae6b94fb8106e8ab9"
            ),
        },
        {
            "phase": 4,
            "state": "complete",
            "quality_gate_sha256": (
                "07d5de78f45684af480d428d17ea8fef29565581e37c20fbd8e97a46c3fb30d0"
            ),
            "evidence_sha256": (
                "e3546c8702c933b8861db39a72e499f7d5bec80523eb9650c3f2bb7a52c0ecba"
            ),
        },
        {
            "phase": 5,
            "state": "complete",
            "quality_gate_sha256": (
                "6f02f22e439d6b1ec34a291f1b22c4fe26d429b8d01c0b1e3453a1730424c3c5"
            ),
            "evidence_sha256": (
                "97987b255baca02760ec1d3701db4ff7fcf92316abac6ab399b518cdb48f68f1"
            ),
        },
    ]
    assert evidence["quality_history_sha256"] == _digest(
        evidence["quality_history"]
    )
    assert evidence["active_test_node_ids"] == current_active_test_node_ids
    assert len(sealed_active_test_node_ids) == 788
    assert len(current_active_test_node_ids) == 814
    assert len(phase6_test_node_ids) == 26
    assert set(sealed_active_test_node_ids) < set(current_active_test_node_ids)
    assert set(current_active_test_node_ids) - set(
        sealed_active_test_node_ids
    ) == set(phase6_test_node_ids)
    prior_quality_gate = {
        "state": "complete",
        "required_commands": [
            "poetry run pytest --verbose -s",
            "make test-coverage -- -100 src/",
            "make test-coverage-exact no-install",
            (
                "poetry run python scripts/verify_input_acceptance.py "
                "--through-phase 3"
            ),
            "make typecheck-input-contract INPUT_PHASE=3",
            "make lint",
            "git diff --check",
            (
                "poetry run pytest --verbose -s "
                "tests/input_contract_test.py "
                "tests/input_acceptance_verifier_test.py "
                "tests/input_type_contract_test.py "
                "tests/input/canonical_contract_test.py "
                "tests/input/model_capability_contract_test.py "
                "tests/agent/orchestrator_response_capability_test.py "
                "tests/model/model_call_test.py "
                "tests/model/model_capability_test.py "
                "tests/model/nlp/vendor_google_capability_test.py "
                "tests/tool/tool_manager_test.py"
            ),
        ],
        "state_details": {
            "completed_at": "2026-07-22T09:25:17Z",
            "gate_run_id": "input-capability-authoritative-ecdd67ebe387",
        },
        "results": [
            {
                "command": "poetry run pytest --verbose -s",
                "exit_code": 0,
                "passed": 11013,
                "skipped": 66,
                "subtests_passed": 8129,
                "seconds": 172.6,
                "deselected": 0,
                "xfail": 0,
                "xpass": 0,
            },
            {
                "command": "make test-coverage -- -100 src/",
                "exit_code": 0,
                "output_lines": [],
            },
            {
                "command": "make test-coverage-exact no-install",
                "exit_code": 0,
                "covered_statements": 108402,
                "total_statements": 108402,
                "source_files": 424,
                "missing_lines": 0,
                "missing_files": 0,
                "passed": 11013,
                "skipped": 66,
                "subtests_passed": 8129,
                "seconds": 273.96,
            },
            {
                "command": (
                    "poetry run python "
                    "scripts/verify_input_acceptance.py --through-phase "
                    "3"
                ),
                "exit_code": 0,
                "active_nodes": 99,
                "active_instances": 187,
            },
            {
                "command": "make typecheck-input-contract INPUT_PHASE=3",
                "exit_code": 0,
                "active_fixtures": 18,
            },
            {
                "command": "make lint",
                "exit_code": 0,
                "source_files_typechecked": 424,
                "script_files_typechecked": 6,
            },
            {"command": "git diff --check", "exit_code": 0},
            {
                "command": (
                    "poetry run pytest --verbose -s "
                    "tests/input_contract_test.py "
                    "tests/input_acceptance_verifier_test.py "
                    "tests/input_type_contract_test.py "
                    "tests/input/canonical_contract_test.py "
                    "tests/input/model_capability_contract_test.py "
                    "tests/agent/orchestrator_response_capability_test.py "
                    "tests/model/model_call_test.py "
                    "tests/model/model_capability_test.py "
                    "tests/model/nlp/vendor_google_capability_test.py "
                    "tests/tool/tool_manager_test.py"
                ),
                "exit_code": 0,
                "passed": 488,
                "skipped": 0,
                "subtests_passed": 82,
                "seconds": 12.83,
                "deselected": 0,
                "xfail": 0,
                "xpass": 0,
            },
        ],
        "tree_binding": {
            "baseline_head": "609aa091c17756ab952cf5fe668ca3d867f0e311",
            "inventory_file_count": 1270,
            "inventory_sha256": (
                "59171e64a5f20f150114cd835e64faafe1b67a5430657a2889c92b1ed8a7ee81"
            ),
            "normalized_evidence_kind": "regular",
            "normalized_evidence_sha256": (
                "bd2369062e3ecd457051a462f9d4ecc27f74bb3934c335c98f37f984019b712e"
            ),
            "normalized_verifier_kind": "regular",
            "normalized_verifier_sha256": (
                "4f276c3f122571a73714a2c8c8b7149c83c0497a19ade038b56d56908a7afce0"
            ),
            "tree_sha256": (
                "60eb9628f98003ad6407e2506555db23c901be60082a5a45cc67c13becbc7baf"
            ),
        },
        "coverage_binding": {
            "report_sha256": (
                "ecdd67ebe3878439b41b9f9f04b9154f0772f6c3b5621d599ea18fc4d912a5e6"
            ),
            "source_inventory_sha256": (
                "32cd39d8285af3b782ca095bda1a80de5e991e98e4baf1ba1cf003c5d02a80ba"
            ),
            "source_file_count": 424,
            "statement_count": 108402,
            "excluded_line_count": 1327,
        },
    }
    assert (
        _digest(prior_quality_gate)
        == evidence["quality_history"][2]["quality_gate_sha256"]
    )
    quality_gate = prior_quality_gate
    commands = quality_gate["required_commands"]
    assert len(commands) == 8
    assert commands[:2] == [
        "poetry run pytest --verbose -s",
        "make test-coverage -- -100 src/",
    ]
    assert commands[2:] == [
        "make test-coverage-exact no-install",
        (
            "poetry run python scripts/verify_input_acceptance.py"
            " --through-phase 3"
        ),
        "make typecheck-input-contract INPUT_PHASE=3",
        "make lint",
        "git diff --check",
        commands[7],
    ]
    assert commands[7].startswith("poetry run pytest --verbose -s tests/")
    assert quality_gate["state"] == "complete"
    assert quality_gate["state_details"] == {
        "completed_at": "2026-07-22T09:25:17Z",
        "gate_run_id": "input-capability-authoritative-ecdd67ebe387",
    }
    assert quality_gate["results"] == [
        {
            "command": commands[0],
            "exit_code": 0,
            "passed": 11013,
            "skipped": 66,
            "subtests_passed": 8129,
            "seconds": 172.6,
            "deselected": 0,
            "xfail": 0,
            "xpass": 0,
        },
        {
            "command": commands[1],
            "exit_code": 0,
            "output_lines": [],
        },
        {
            "command": commands[2],
            "exit_code": 0,
            "covered_statements": 108402,
            "total_statements": 108402,
            "source_files": 424,
            "missing_lines": 0,
            "missing_files": 0,
            "passed": 11013,
            "skipped": 66,
            "subtests_passed": 8129,
            "seconds": 273.96,
        },
        {
            "command": commands[3],
            "exit_code": 0,
            "active_nodes": 99,
            "active_instances": 187,
        },
        {
            "command": commands[4],
            "exit_code": 0,
            "active_fixtures": 18,
        },
        {
            "command": commands[5],
            "exit_code": 0,
            "source_files_typechecked": 424,
            "script_files_typechecked": 6,
        },
        {"command": commands[6], "exit_code": 0},
        {
            "command": commands[7],
            "exit_code": 0,
            "passed": 488,
            "skipped": 0,
            "subtests_passed": 82,
            "seconds": 12.83,
            "deselected": 0,
            "xfail": 0,
            "xpass": 0,
        },
    ]
    tree_binding = quality_gate["tree_binding"]
    assert set(tree_binding) == {
        "baseline_head",
        "inventory_file_count",
        "inventory_sha256",
        "normalized_evidence_kind",
        "normalized_evidence_sha256",
        "normalized_verifier_kind",
        "normalized_verifier_sha256",
        "tree_sha256",
    }
    assert tree_binding["baseline_head"] == evidence["git"]["head"]
    assert tree_binding["inventory_file_count"] > 0
    assert tree_binding["normalized_evidence_kind"] == "regular"
    assert tree_binding["normalized_verifier_kind"] == "regular"
    for field in (
        "inventory_sha256",
        "normalized_evidence_sha256",
        "normalized_verifier_sha256",
        "tree_sha256",
    ):
        assert len(tree_binding[field]) == 64
        int(tree_binding[field], 16)
    assert quality_gate["coverage_binding"] == {
        "report_sha256": (
            "ecdd67ebe3878439b41b9f9f04b9154f0772f6c3b5621d599ea18fc4d912a5e6"
        ),
        "source_inventory_sha256": (
            "32cd39d8285af3b782ca095bda1a80de5e991e98e4baf1ba1cf003c5d02a80ba"
        ),
        "source_file_count": 424,
        "statement_count": 108402,
        "excluded_line_count": 1327,
    }
    assert evidence["pending_structural_inventory"] == {
        "source_inventory_sha256": (
            "924cbf18b65fb91c3b438a4cb7bec757ba7843e97b5ac0bd89b68072f7edc943"
        ),
        "source_file_count": 440,
        "statement_count": 119485,
        "excluded_line_count": 1834,
    }
    regression = evidence["current_regression_classification"]
    assert (
        regression["catalog_sha256"]
        == "e26844f278d2ba186211e137cecbb23100ee950dcaec3ded80a3c81b241b7814"
    )
    assert regression["catalog_sha256"] == _digest(
        {
            "mechanical_nodes": regression["mechanical_nodes"],
            "reviewed_nonsemantic_nodes": regression[
                "reviewed_nonsemantic_nodes"
            ],
            "support_surfaces": regression["support_surfaces"],
        }
    )
    dispositions = {
        disposition: [
            entry
            for entry in regression["reviewed_nonsemantic_nodes"]
            if entry["disposition"] == disposition
        ]
        for disposition in ("gate_support", "semantic_support")
    }
    assert len(dispositions["gate_support"]) == 4
    assert len(dispositions["semantic_support"]) == 56
    focused_coverage_node_ids = {
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_completed_json_rejects_every_non_finite_float"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_durable_handoff_rejects_untyped_and_mismatched_acknowledgements"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_durable_integration_rejects_unowned_components"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_legacy_async_controller_aliases_and_projection_failures"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_opaque_references_reject_type_shape_payload_and_pair_mismatches"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_owned_runtime_and_policy_factories_validate_callbacks"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_principal_clock_identifiers_authorization_and_classifier"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_public_controller_aliases_and_bridge_validation"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_public_handler_adapters_preserve_each_typed_outcome"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_public_persistence_values_reject_empty_payloads_and_bad_digests"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_public_state_errors_map_to_exact_exception_classes"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_public_value_objects_reject_each_invalid_member"
        ),
        (
            "tests/input/public_sdk_coverage_test.py::"
            "test_run_agent_maps_untyped_values_failures_and_input_handoff"
        ),
        (
            "tests/interaction/interaction_store_validation_coverage_test.py::"
            "test_record_resolution_validation_rejects_corrupt_authority"
        ),
        (
            "tests/interaction/interaction_store_validation_coverage_test.py::"
            "test_replay_branch_and_basic_identity_validators_fail_closed"
        ),
        (
            "tests/interaction/stores/interaction_pgsql_store_test.py::"
            "PgsqlInteractionStoreTest::"
            "test_advisory_timeout_rejects_wrong_scope_and_task_state"
        ),
    }
    focused_coverage_entries = [
        entry
        for entry in regression["reviewed_nonsemantic_nodes"]
        if entry["node_id"] in focused_coverage_node_ids
    ]
    assert {
        entry["node_id"] for entry in focused_coverage_entries
    } == focused_coverage_node_ids
    assert all(
        entry["disposition"] == "semantic_support"
        and entry["reviewed_by"] == "/root"
        and entry["baseline_ast_sha256"] != entry["current_ast_sha256"]
        and "coverage" in entry["evidence"].lower()
        for entry in focused_coverage_entries
    )
    sdk_coverage_entries = [
        entry
        for entry in focused_coverage_entries
        if entry["node_id"].startswith(
            "tests/input/public_sdk_coverage_test.py::"
        )
    ]
    assert len(sdk_coverage_entries) == 13
    assert all(
        entry["baseline_ast_sha256"]
        == "d6f5bc657cdeb0be6ee6c3f042458c9981e5bcb0a4dbe6a9f6d6c39f464f0479"
        for entry in sdk_coverage_entries
    )
    assert all(
        entry["evidence"]
        == "Exact live support definition for the public SDK boundary is"
        " classified outside the acceptance manifest."
        for entry in regression["reviewed_nonsemantic_nodes"]
        if entry["node_id"] not in focused_coverage_node_ids
    )
    assert len(regression["support_surfaces"]) == 14
    assert any(
        entry["path"] == "tests/input/public_sdk_coverage_test.py"
        and entry["baseline_ast_sha256"]
        == "6b21bdd337b5554cd17fe6cf861b9b1f457568a5a2e05a41e7ee744686ed0872"
        and entry["current_ast_sha256"]
        == "ea1eea6cdd3ba158c892192ae3677923b828f488685309405675dc5f5ffe3d9a"
        for entry in regression["support_surfaces"]
    )
    assert evidence["inventory"]["active_acceptance_nodes"] == 814
    assert evidence["inventory"]["active_pytest_instances"] == 988
    assert evidence["inventory"]["planned_acceptance_nodes"] == 130
    assert evidence["inventory"]["failure_surfaces"] == 84
    assert evidence["inventory"]["failure_cells"] == 1260
    assert evidence["typing_async_audit"]["strict_type_fixture_count"] == 34
    quality_gate = evidence["quality_gate"]
    assert quality_gate["state"] == "complete"
    assert quality_gate["required_commands"] == [
        "make lint",
        "make typecheck-input-contract INPUT_PHASE=6",
        "make test-pgsql-exact no-install INPUT_PHASE=6",
        "git diff --check",
    ]
    details = quality_gate["state_details"]
    assert details["completed_at"] == "2026-07-24T10:45:19Z"
    assert details["gate_run_id"] == "final-quality-evidence-91a820e283d1"
    assert details["final_review"] == {
        "reviewer": "/root/public_sdk_gate_review",
        "status": "approved",
        "approval_sealed": True,
    }
    assert [
        (
            attempt["attempt"],
            attempt["outcome"],
            attempt["passed"],
            attempt["skipped"],
            attempt["subtests_passed"],
            attempt["seconds"],
            attempt["failure_stage"],
        )
        for attempt in details["prior_failed_attempts"]
    ] == [
        (
            1,
            "failed",
            12000,
            59,
            8678,
            366.53,
            "coverage_exclusion_verification",
        ),
        (
            2,
            "failed",
            12000,
            59,
            8678,
            362.09,
            "coverage_exclusion_verification",
        ),
        (
            3,
            "failed",
            12000,
            59,
            8678,
            366.48,
            "exact_source_coverage",
        ),
    ]
    assert details["diagnostic_coverage"] == {
        "purpose": "diagnostic_only",
        "passed": 12000,
        "skipped": 59,
        "subtests_passed": 8678,
        "seconds": 368.56,
        "report_sha256": (
            "48b0587756849bc1c22fdf437f0f03643ad9e2c9788e4166168239e31be2cf7f"
        ),
    }
    assert details["hard_coverage_audit"] == {
        "command": "make test-coverage -- -100 src/",
        "exit_code": 0,
        "below_threshold_files": [],
    }
    assert quality_gate["results"] == [
        {
            "command": "make lint",
            "exit_code": 0,
            "source_files_typechecked": 440,
            "script_files_typechecked": 6,
        },
        {
            "command": "make typecheck-input-contract INPUT_PHASE=6",
            "exit_code": 0,
            "active_fixtures": 34,
        },
        {
            "command": "make test-pgsql-exact no-install INPUT_PHASE=6",
            "exit_code": 0,
            "active_nodes": 814,
            "active_instances": 988,
            "covered_statements": 119485,
            "total_statements": 119485,
            "source_files": 440,
            "missing_lines": 0,
            "missing_files": 0,
            "passed": 12013,
            "skipped": 59,
            "subtests_passed": 8678,
            "seconds": 380.35,
            "deselected": 0,
            "xfail": 0,
            "xpass": 0,
        },
        {"command": "git diff --check", "exit_code": 0},
    ]
    tree_binding = quality_gate["tree_binding"]
    assert set(tree_binding) == {
        "baseline_head",
        "inventory_file_count",
        "inventory_sha256",
        "normalized_evidence_kind",
        "normalized_evidence_sha256",
        "normalized_verifier_kind",
        "normalized_verifier_sha256",
        "source_tree_file_count",
        "source_tree_inventory_sha256",
        "test_tree_file_count",
        "test_tree_inventory_sha256",
        "script_tree_file_count",
        "script_tree_inventory_sha256",
        "support_tree_boundary",
        "support_tree_file_count",
        "support_tree_inventory_sha256",
        "tree_sha256",
    }
    assert tree_binding["inventory_file_count"] == 1352
    assert tree_binding["source_tree_file_count"] == 444
    assert tree_binding["test_tree_file_count"] == 749
    assert tree_binding["script_tree_file_count"] == 10
    assert tree_binding["support_tree_file_count"] == 149
    assert (
        tree_binding["source_tree_file_count"]
        + tree_binding["test_tree_file_count"]
        + tree_binding["script_tree_file_count"]
        + tree_binding["support_tree_file_count"]
        == tree_binding["inventory_file_count"]
    )
    assert (
        tree_binding["support_tree_boundary"]
        == "all normalized files outside src/, tests/, and scripts/"
    )
    for field in (
        "inventory_sha256",
        "normalized_evidence_sha256",
        "normalized_verifier_sha256",
        "source_tree_inventory_sha256",
        "test_tree_inventory_sha256",
        "script_tree_inventory_sha256",
        "support_tree_inventory_sha256",
        "tree_sha256",
    ):
        assert len(tree_binding[field]) == 64
        int(tree_binding[field], 16)
    assert quality_gate["coverage_binding"] == {
        "report_sha256": (
            "91a820e283d1dfa48073feaa008cf29c5730640767dd9f48c630c3d1157fba34"
        ),
        "xml_report_sha256": (
            "497870de64efb4b142db041e604d2cb9be8390561f1bda4480b44cedf6b5d170"
        ),
        "source_inventory_sha256": (
            "924cbf18b65fb91c3b438a4cb7bec757ba7843e97b5ac0bd89b68072f7edc943"
        ),
        "source_file_count": 440,
        "statement_count": 119485,
        "excluded_line_count": 1834,
    }
    assert evidence["unresolved_risks"] == [
        (
            "Database-backed exact target requires Docker or an admin DSN when"
            " exercised outside CI."
        ),
        (
            "Canonical interaction behavior remains dormant and unadvertised"
            " until atomic capability activation."
        ),
    ]


def test_capability_remains_dormant() -> None:
    """Require canonical production types without public advertisement."""
    decisions = _fixture("contract_decisions.json")
    evidence = _fixture("baseline_evidence.json")
    assert decisions["activation"]["production_default"] == "absent"
    assert (
        evidence["boundary"]["production_capability"] == "dormant_unadvertised"
    )
    assert evidence["boundary"]["production_capability_history"] == [
        {"phase": 0, "state": "absent"},
        {"phase": 1, "state": "dormant_unadvertised"},
        {"phase": 2, "state": "dormant_unadvertised"},
        {"phase": 3, "state": "dormant_unadvertised"},
        {"phase": 4, "state": "dormant_unadvertised"},
        {"phase": 5, "state": "dormant_unadvertised"},
        {"phase": 6, "state": "dormant_unadvertised"},
    ]
    assert evidence["boundary"]["production_source_changes"]
    assert all(
        not row["production_advertised"]
        for row in decisions["capability_matrix"]["rows"]
    )

    reserved_identifier = "https://avalan.ai/extensions/task-input/v1"
    reserved_names = (
        "request_input",
        "request_user_input",
        "structured_task_input",
        "task_input_request",
    )
    registry = default_flow_node_registry()
    assert all(not registry.supports(name) for name in reserved_names)

    schema_catalog = {
        model.__name__: model.model_json_schema()
        for model in (
            ChatCompletionRequest,
            EngineRequest,
            MCPToolRequest,
            ResponsesRequest,
        )
    }
    runtime_capabilities = _server_capabilities(cast(Any, None))
    agent_card = _build_agent_card(
        a2a_pb2=_AgentCardProtocol,
        constants=_AgentCardConstants,
        interface_url="/a2a",
        name="run",
        description="Run the test agent.",
    )
    agent_card_capabilities = vars(agent_card.capabilities)
    agent_card_extensions = getattr(
        agent_card.capabilities,
        "extensions",
        [],
    )
    assert agent_card_capabilities == {"streaming": True}
    assert agent_card_extensions == []
    manager = ToolManager.create_instance()
    model_catalog = ModelCapabilityCatalog.create(
        manager.export_model_capability_seed()
    )
    tool_catalog = {
        "descriptors": [
            descriptor.name for descriptor in manager.list_tools()
        ],
        "schemas": model_catalog.project().schemas,
        "provider_schemas": model_catalog.project().schemas,
    }
    live_advertisement: dict[str, object] = {
        "schemas": schema_catalog,
        "mcp_capabilities": runtime_capabilities,
        "a2a_agent_card": {
            "capabilities": agent_card_capabilities,
            "extensions": agent_card_extensions,
        },
        "tools": tool_catalog,
    }
    assert _reserved_capability_absent(
        live_advertisement,
        reserved_identifier,
        reserved_names,
    )
    mutated_advertisement = deepcopy(live_advertisement)
    mutated_a2a = cast(
        dict[str, object],
        mutated_advertisement["a2a_agent_card"],
    )
    mutated_capabilities = cast(
        dict[str, object],
        mutated_a2a["capabilities"],
    )
    mutated_capabilities["extensions"] = [{"uri": reserved_identifier}]
    assert not _reserved_capability_absent(
        mutated_advertisement,
        reserved_identifier,
        reserved_names,
    )
