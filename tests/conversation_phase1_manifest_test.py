"""Verify append-only Phase 1 conversation snapshot selection."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import loads
from pathlib import Path
from sys import modules
from sys import path as sys_path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests/fixtures/conversation"
_PHASE0_BYTES = {
    "acceptance_manifest.json": (
        "b046e73982c85dccb6f6d16ff09e036ec10e05cc4b91aca376b8c74118df3a05"
    ),
    "failure_matrix.json": (
        "2c8443c83c966f0f05cc0fd54956aae5d6865e173733eb8a33fe1de3ec1fa751"
    ),
    "threat_model.json": (
        "f1448630d12320eaf70e7394dbb063a7caaf65bbf22221a35a72e2c1b48c0120"
    ),
    "type_contract_manifest.json": (
        "87fc9af5ccc661b70010d6399b4a0235b05ff322a87306ed870b1dacfcd7416f"
    ),
}


def _load(name: str, relative: str) -> ModuleType:
    """Load one gate module from the tracked script directory."""
    scripts = str(_ROOT / "scripts")
    if scripts not in sys_path:
        sys_path.insert(0, scripts)
    spec = spec_from_file_location(name, _ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    modules[name] = module
    spec.loader.exec_module(module)
    return module


def _payload(name: str) -> dict[str, object]:
    """Return one strict object-shaped fixture payload."""
    value: object = loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {str(key): item for key, item in value.items()}


def test_phase7_defaults_select_the_complete_snapshot_family() -> None:
    """Select Phase 7 acceptance, type, failure, and threat snapshots."""
    type_verifier = _load(
        "_phase1_type_verifier",
        "scripts/verify_conversation_types.py",
    )
    acceptance = _load(
        "_phase1_acceptance_verifier",
        "scripts/verify_conversation_acceptance.py",
    )
    runner = _load(
        "_phase1_contract_runner",
        "scripts/run_conversation_contract_gate.py",
    )

    acceptance_path = acceptance.default_manifest_path()
    assert acceptance_path.name == "acceptance_manifest.phase7.json"
    assert (
        type_verifier.default_manifest_path().name
        == "type_contract_manifest.phase7.json"
    )
    assert (
        acceptance.companion_fixture_path(
            acceptance_path, "failure_matrix"
        ).name
        == "failure_matrix.phase7.json"
    )
    assert (
        acceptance.companion_fixture_path(acceptance_path, "threat_model").name
        == "threat_model.phase7.json"
    )
    assert (
        acceptance.companion_fixture_path(
            acceptance_path, "type_contract_manifest"
        ).name
        == "type_contract_manifest.phase7.json"
    )
    assert acceptance.load_manifest(acceptance_path).current_phase == 7
    assert (
        type_verifier.load_manifest(
            type_verifier.default_manifest_path()
        ).current_phase
        == 7
    )
    assert runner._CONVERSATION_CURRENT_PHASE == 7
    runner._validate_through_phase(_ROOT, 7)
    with pytest.raises(runner.ContractGateError):
        runner._validate_through_phase(_ROOT, 8)


def test_phase1_acceptance_validates_selected_companions() -> None:
    """Validate every selected Phase 1 companion before pytest execution."""
    acceptance = _load(
        "_phase1_acceptance_validation",
        "scripts/verify_conversation_acceptance.py",
    )
    manifest = acceptance.verify_acceptance(
        _FIXTURES / "acceptance_manifest.phase1.json",
        repo_root=_ROOT,
        through_phase=1,
        execute=False,
    )
    assert manifest.current_phase == 1
    assert len(manifest.active_nodes(1)) == 21


def test_phase3_snapshots_append_to_phase2_without_rewriting_history() -> None:
    """Preserve Phase 2 history while activating only Phase 3 evidence."""
    acceptance2 = _payload("acceptance_manifest.phase2.json")
    acceptance3 = _payload("acceptance_manifest.phase3.json")
    history2 = acceptance2["activation_history"]
    history3 = acceptance3["activation_history"]
    nodes2 = acceptance2["nodes"]
    nodes3 = acceptance3["nodes"]
    assert isinstance(history2, list) and isinstance(history3, list)
    assert isinstance(nodes2, list) and isinstance(nodes3, list)
    assert history3[: len(history2)] == history2
    assert len(history3) == len(history2) + 1
    assert len(nodes3) == len(nodes2) + 13
    for previous, current in zip(
        nodes2,
        nodes3[: len(nodes2)],
        strict=True,
    ):
        assert isinstance(previous, dict) and isinstance(current, dict)
        if previous.get("id") == "phase3-durable-pgsql":
            assert previous.get("lifecycle") == "planned"
            assert current == {**previous, "lifecycle": "replaced"}
        else:
            assert current == previous
    appended_nodes = nodes3[len(nodes2) :]
    assert all(
        isinstance(node, dict)
        and node.get("active_from_phase") == 3
        and node.get("lifecycle") == "active"
        for node in appended_nodes
    )
    replacements2 = acceptance2["replacements"]
    replacements3 = acceptance3["replacements"]
    assert isinstance(replacements2, list)
    assert isinstance(replacements3, list)
    assert replacements3[: len(replacements2)] == replacements2
    assert len(replacements3) == len(replacements2) + 1

    types2 = _payload("type_contract_manifest.phase2.json")
    types3 = _payload("type_contract_manifest.phase3.json")
    fixtures2 = types2["fixtures"]
    fixtures3 = types3["fixtures"]
    type_history2 = types2["activation_history"]
    type_history3 = types3["activation_history"]
    assert isinstance(fixtures2, list) and isinstance(fixtures3, list)
    assert isinstance(type_history2, list) and isinstance(type_history3, list)
    assert fixtures3[: len(fixtures2)] == fixtures2
    assert len(fixtures3) == len(fixtures2) + 2
    assert type_history3[: len(type_history2)] == type_history2

    failure2 = _payload("failure_matrix.phase2.json")
    failure3 = _payload("failure_matrix.phase3.json")
    for field, appended in (
        ("boundaries", 1),
        ("surfaces", 1),
        ("cells", 21),
    ):
        previous = failure2[field]
        current = failure3[field]
        assert isinstance(previous, list) and isinstance(current, list)
        assert current[: len(previous)] == previous
        assert len(current) == len(previous) + appended
    assert failure3["observation_window"] == failure2["observation_window"]
    assert failure3["tool_effect_scope"] == failure2["tool_effect_scope"]

    threats2 = _payload("threat_model.phase2.json")
    threats3 = _payload("threat_model.phase3.json")
    for field, appended in (
        ("assets", 4),
        ("trust_boundaries", 3),
        ("threats", 6),
    ):
        previous = threats2[field]
        current = threats3[field]
        assert isinstance(previous, list) and isinstance(current, list)
        assert current[: len(previous)] == previous
        assert len(current) == len(previous) + appended


def test_phase4_snapshots_append_to_phase3_without_rewriting_history() -> None:
    """Preserve Phase 3 bytes while activating direct SDK evidence."""
    acceptance3 = _payload("acceptance_manifest.phase3.json")
    acceptance4 = _payload("acceptance_manifest.phase4.json")
    history3 = acceptance3["activation_history"]
    history4 = acceptance4["activation_history"]
    nodes3 = acceptance3["nodes"]
    nodes4 = acceptance4["nodes"]
    assert isinstance(history3, list) and isinstance(history4, list)
    assert isinstance(nodes3, list) and isinstance(nodes4, list)
    assert history4[: len(history3)] == history3
    assert len(history4) == len(history3) + 1
    assert len(nodes4) == len(nodes3) + 6
    for previous, current in zip(
        nodes3,
        nodes4[: len(nodes3)],
        strict=True,
    ):
        assert isinstance(previous, dict) and isinstance(current, dict)
        if previous.get("id") == "phase4-public-sdk":
            assert previous.get("lifecycle") == "planned"
            assert current == {**previous, "lifecycle": "replaced"}
        else:
            assert current == previous
    appended_nodes = nodes4[len(nodes3) :]
    assert all(
        isinstance(node, dict) and node.get("active_from_phase") == 4
        for node in appended_nodes
    )
    assert isinstance(appended_nodes[0], dict)
    assert (
        appended_nodes[0]["node_id"]
        == "tests/conversation/direct_sdk_test.py::"
        "test_stream_create_continue_branch_and_compact"
    )
    assert appended_nodes[0]["lifecycle"] == "replaced"
    assert all(
        isinstance(node, dict) and node.get("lifecycle") == "active"
        for node in appended_nodes[1:]
    )
    replacements3 = acceptance3["replacements"]
    replacements4 = acceptance4["replacements"]
    assert isinstance(replacements3, list)
    assert isinstance(replacements4, list)
    assert replacements4[: len(replacements3)] == replacements3
    assert len(replacements4) == len(replacements3) + 2

    types3 = _payload("type_contract_manifest.phase3.json")
    types4 = _payload("type_contract_manifest.phase4.json")
    fixtures3 = types3["fixtures"]
    fixtures4 = types4["fixtures"]
    type_history3 = types3["activation_history"]
    type_history4 = types4["activation_history"]
    assert isinstance(fixtures3, list) and isinstance(fixtures4, list)
    assert isinstance(type_history3, list) and isinstance(type_history4, list)
    assert fixtures4[: len(fixtures3)] == fixtures3
    assert len(fixtures4) == len(fixtures3) + 2
    assert type_history4[: len(type_history3)] == type_history3

    failure3 = _payload("failure_matrix.phase3.json")
    failure4 = _payload("failure_matrix.phase4.json")
    assert failure4["boundaries"] == failure3["boundaries"]
    assert failure4["surfaces"] == failure3["surfaces"]
    cells3 = failure3["cells"]
    cells4 = failure4["cells"]
    assert isinstance(cells3, list) and isinstance(cells4, list)
    assert len(cells4) == len(cells3)
    for previous, current in zip(cells3, cells4, strict=True):
        assert isinstance(previous, dict) and isinstance(current, dict)
        if previous.get("id") == "durable_transaction_failure--direct_sdk":
            assert previous.get("lifecycle") == "planned"
            assert current == {
                **previous,
                "lifecycle": "active",
                "evidence_node_id": (
                    "tests/conversation/direct_sdk_pgsql_test.py::"
                    "test_public_pgsql_commit_failure_and_post_commit_recovery"
                ),
            }
        else:
            assert current == previous

    threats3 = _payload("threat_model.phase3.json")
    threats4 = _payload("threat_model.phase4.json")
    assert threats4["assets"] == threats3["assets"]
    assert threats4["trust_boundaries"] == threats3["trust_boundaries"]
    threats3_items = threats3["threats"]
    threats4_items = threats4["threats"]
    assert isinstance(threats3_items, list)
    assert isinstance(threats4_items, list)
    assert threats4_items[: len(threats3_items)] == threats3_items
    assert len(threats4_items) == len(threats3_items) + 1


def test_phase5_snapshots_activate_native_stateless_evidence_append_only() -> (
    None
):
    """Preserve Phase 4 history while activating native replay evidence."""
    acceptance4 = _payload("acceptance_manifest.phase4.json")
    acceptance5 = _payload("acceptance_manifest.phase5.json")
    history4 = acceptance4["activation_history"]
    history5 = acceptance5["activation_history"]
    nodes4 = acceptance4["nodes"]
    nodes5 = acceptance5["nodes"]
    assert isinstance(history4, list) and isinstance(history5, list)
    assert isinstance(nodes4, list) and isinstance(nodes5, list)
    assert history5[: len(history4)] == history4
    assert len(history5) == len(history4) + 1
    assert len(nodes5) == len(nodes4) + 16
    nodes5_by_id = {
        node["id"]: node for node in nodes5 if isinstance(node, dict)
    }
    for previous in nodes4:
        assert isinstance(previous, dict)
        current = nodes5_by_id[previous["id"]]
        if previous.get("id") == "phase5-stateless-wire":
            assert previous.get("lifecycle") == "planned"
            assert current.get("lifecycle") == "replaced"
            assert current.get("provider") == [
                "native_openai",
                "native_azure",
                "incapable_generic_compatible",
            ]
            assert current.get("provider_mode") == [
                "stateless_encrypted_replay"
            ]
            assert current.get("turn_topology") == [
                "first_turn",
                "ordinary_child",
            ]
        else:
            assert current == previous
    active_phase5 = [
        node
        for node in nodes5
        if isinstance(node, dict)
        and node.get("active_from_phase") == 5
        and node.get("lifecycle") == "active"
    ]
    assert len(active_phase5) == 16
    assert all(
        isinstance(node, dict)
        and node.get("active_from_phase") == 5
        and node.get("lifecycle") == "active"
        for node in active_phase5
    )
    assert {
        node["id"] for node in active_phase5 if isinstance(node, dict)
    } == {
        "phase5-openai-replay-private",
        "phase5-concurrent-branch-isolation",
        "phase5-openai-durable-fresh-process",
        "phase5-public-provider-projection",
        "phase5-reasoning-context-matrix",
        "phase5-reasoning-capability-rejection",
        "phase5-azure-exact-loopback-wire",
        "phase5-generic-compatible-rejection",
        "phase5-provider-item-limit",
        "phase5-provider-byte-limit",
        "phase5-provider-segment-validation",
        "phase5-native-close-cancellation",
        "phase5-legacy-facade-ownership",
        "phase5-sdk-failure-mapping",
        "phase5-stream-failure-boundaries",
        "phase5-stream-item-integrity",
    }
    active_requirements = [
        requirement_id
        for node in active_phase5
        if isinstance(node, dict)
        for requirement_id in node["requirement_ids"]
    ]
    assert len(active_requirements) == 18
    assert len(set(active_requirements)) == 18
    concurrent = nodes5_by_id["phase5-concurrent-branch-isolation"]
    assert concurrent.get("turn_topology") == ["explicit_branch"]
    assert concurrent.get("reasoning_context") == ["all_turns"]
    assert concurrent.get("limit") == ["branch_count", "concurrency"]
    assert concurrent.get("requirement_ids") == ["CONV-N-037"]

    types4 = _payload("type_contract_manifest.phase4.json")
    types5 = _payload("type_contract_manifest.phase5.json")
    fixtures4 = types4["fixtures"]
    fixtures5 = types5["fixtures"]
    type_history4 = types4["activation_history"]
    type_history5 = types5["activation_history"]
    assert isinstance(fixtures4, list) and isinstance(fixtures5, list)
    assert isinstance(type_history4, list) and isinstance(type_history5, list)
    assert fixtures5[: len(fixtures4)] == fixtures4
    assert len(fixtures5) == len(fixtures4) + 2
    assert type_history5[: len(type_history4)] == type_history4

    failure4 = _payload("failure_matrix.phase4.json")
    failure5 = _payload("failure_matrix.phase5.json")
    assert failure5["boundaries"] == failure4["boundaries"]
    assert failure5["surfaces"] == failure4["surfaces"]
    cells4 = failure4["cells"]
    cells5 = failure5["cells"]
    assert isinstance(cells4, list) and isinstance(cells5, list)
    assert len(cells5) == len(cells4)
    activated = {
        "durable_transaction_failure--provider_adapter",
        "durable_transaction_failure--stream",
    }
    for previous, current in zip(cells4, cells5, strict=True):
        assert isinstance(previous, dict) and isinstance(current, dict)
        if previous.get("id") in activated:
            assert previous.get("lifecycle") == "planned"
            assert current == {**previous, "lifecycle": "active"}
        else:
            assert current == previous

    threats4 = _payload("threat_model.phase4.json")
    threats5 = _payload("threat_model.phase5.json")
    assert threats5["assets"] == threats4["assets"]
    assert threats5["trust_boundaries"] == threats4["trust_boundaries"]
    threat_items4 = threats4["threats"]
    threat_items5 = threats5["threats"]
    assert isinstance(threat_items4, list) and isinstance(threat_items5, list)
    assert threat_items5[: len(threat_items4)] == threat_items4
    assert len(threat_items5) == len(threat_items4) + 1

    conformance0 = _payload("provider_conformance.json")
    conformance5 = _payload("provider_conformance.phase5.json")
    profiles0 = conformance0["profiles"]
    profiles5 = conformance5["profiles"]
    assert isinstance(profiles0, list) and isinstance(profiles5, list)
    assert profiles5[: len(profiles0)] == profiles0
    assert len(profiles5) == len(profiles0) + 4
    assert conformance5["activation_state"] == "test_only"


def test_phase0_manifests_remain_byte_pinned_and_history_is_retained() -> None:
    """Preserve Phase 0 bytes and append Phase 1 history monotonically."""
    for name, expected in _PHASE0_BYTES.items():
        assert sha256((_FIXTURES / name).read_bytes()).hexdigest() == expected

    acceptance0 = _payload("acceptance_manifest.json")
    acceptance1 = _payload("acceptance_manifest.phase1.json")
    history0 = acceptance0["activation_history"]
    history1 = acceptance1["activation_history"]
    nodes0 = acceptance0["nodes"]
    nodes1 = acceptance1["nodes"]
    assert isinstance(history0, list) and isinstance(history1, list)
    assert isinstance(nodes0, list) and isinstance(nodes1, list)
    assert history1[0] == history0[0]
    assert [
        node
        for node in nodes1
        if isinstance(node, dict) and node.get("active_from_phase") == 0
    ] == [
        node
        for node in nodes0
        if isinstance(node, dict) and node.get("active_from_phase") == 0
    ]

    types0 = _payload("type_contract_manifest.json")
    types1 = _payload("type_contract_manifest.phase1.json")
    type_history0 = types0["activation_history"]
    type_history1 = types1["activation_history"]
    fixtures0 = types0["fixtures"]
    fixtures1 = types1["fixtures"]
    assert isinstance(type_history0, list) and isinstance(type_history1, list)
    assert isinstance(fixtures0, list) and isinstance(fixtures1, list)
    assert type_history1[0] == type_history0[0]
    assert fixtures1[: len(fixtures0)] == fixtures0

    for stem, digest_key in (
        ("failure_matrix", "matrix_sha256"),
        ("threat_model", "threat_model_sha256"),
    ):
        base = _payload(f"{stem}.json")
        snapshot = _payload(f"{stem}.phase1.json")
        base.pop("current_phase")
        snapshot.pop("current_phase")
        base.pop(digest_key)
        snapshot.pop(digest_key)
        assert snapshot == base
