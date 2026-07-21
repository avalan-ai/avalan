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
    assert active == snapshots[-1]["node_ids"]
    assert (
        snapshots[-1]["sha256"]
        == sha256("\n".join(active).encode()).hexdigest()
    )
    assert [item["phase"] for item in snapshots] == sorted(
        item["phase"] for item in snapshots
    )
    assert [item["phase"] for item in history] == [
        item["phase"] for item in snapshots
    ]
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
        }
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
    assert len(catalog) == 101
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
    active_test_node_ids = [
        node["node_id"]
        for node in manifest["nodes"]
        if node["lifecycle"] == "active"
    ]
    assert evidence["implementation_owner"] == "/root"
    assert evidence["independent_reviewer"] == "/root/input_contract_audit"
    assert evidence["implementation_owner"] != evidence["independent_reviewer"]
    assert evidence["recorded_at"] == "2026-07-21T17:29:45-03:00"
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
    assert evidence["review_history"][7:] == [
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
    assert evidence["review_history_sha256"] == _digest(
        evidence["review_history"]
    )
    assert evidence["review_history_phase0_sha256"] == _digest(
        evidence["review_history"][:1]
    )
    assert evidence["review_history_phase1_sha256"] == _digest(
        evidence["review_history"][:5]
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
        }
    ]
    assert evidence["quality_history_sha256"] == _digest(
        evidence["quality_history"]
    )
    assert evidence["active_test_node_ids"] == active_test_node_ids
    quality_gate = evidence["quality_gate"]
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
            " --through-phase 2"
        ),
        "make typecheck-input-contract INPUT_PHASE=2",
        "make lint",
        "git diff --check",
        commands[7],
    ]
    assert commands[7].startswith("poetry run pytest --verbose -s tests/")
    assert quality_gate["state"] == "complete"
    assert quality_gate["state_details"] == {
        "completed_at": "2026-07-21T17:14:59-03:00",
        "gate_run_id": "input-broker-authoritative-20260721T171459-0300",
    }
    assert quality_gate["results"] == [
        {
            "command": commands[0],
            "exit_code": 0,
            "passed": 10687,
            "skipped": 66,
            "subtests_passed": 7955,
            "seconds": 164.06,
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
            "covered_statements": 106507,
            "total_statements": 106507,
            "source_files": 422,
            "missing_lines": 0,
            "missing_files": 0,
            "passed": 10687,
            "skipped": 66,
            "subtests_passed": 7955,
            "seconds": 253.58,
        },
        {
            "command": commands[3],
            "exit_code": 0,
            "active_nodes": 86,
            "active_instances": 125,
        },
        {
            "command": commands[4],
            "exit_code": 0,
            "active_fixtures": 14,
        },
        {
            "command": commands[5],
            "exit_code": 0,
            "source_files_typechecked": 422,
            "script_files_typechecked": 6,
        },
        {
            "command": commands[6],
            "exit_code": 0,
        },
        {
            "command": commands[7],
            "exit_code": 0,
            "passed": 1609,
            "skipped": 0,
            "subtests_passed": 1141,
            "seconds": 30.39,
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
    for key in (
        "inventory_sha256",
        "normalized_evidence_sha256",
        "normalized_verifier_sha256",
        "tree_sha256",
    ):
        assert fullmatch(r"[0-9a-f]{64}", tree_binding[key])
    coverage_binding = quality_gate["coverage_binding"]
    assert set(coverage_binding) == {
        "report_sha256",
        "source_inventory_sha256",
        "source_file_count",
        "statement_count",
        "excluded_line_count",
    }
    assert coverage_binding["source_file_count"] == 422
    assert coverage_binding["statement_count"] == 106507
    assert coverage_binding["excluded_line_count"] == 1329
    assert fullmatch(r"[0-9a-f]{64}", coverage_binding["report_sha256"])
    assert fullmatch(
        r"[0-9a-f]{64}", coverage_binding["source_inventory_sha256"]
    )
    assert evidence["inventory"]["active_pytest_instances"] == 125
    assert evidence["inventory"]["failure_surfaces"] == 84
    assert evidence["inventory"]["failure_cells"] == 1260


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
    tool_catalog = {
        "descriptors": [
            descriptor.name for descriptor in manager.list_tools()
        ],
        "schemas": manager.json_schemas(),
        "provider_schemas": manager.provider_json_schemas(),
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
