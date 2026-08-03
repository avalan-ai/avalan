"""Exercise the dormant conversation acceptance contract end to end."""

from ast import Constant, walk
from ast import parse as parse_python
from collections.abc import Callable
from dataclasses import fields
from hashlib import sha256
from json import dumps, loads
from pathlib import Path

from avalan.conversation import (
    CHECKPOINT_COMMIT_TRANSITIONS,
    CHECKPOINT_VISIBILITY,
    PUBLIC_RESPONSE_ID_TRANSITIONS,
    RESPONSE_OPERATION_POLICY,
    RESPONSE_RESOURCE_TRANSITIONS,
    CheckpointCommitState,
    CheckpointKind,
    PublicResponseIdState,
    ResponseOperation,
    ResponseResourceState,
)
from avalan.entities import GenerationSettings
from avalan.server.entities import (
    DORMANT_CONVERSATION_REQUEST_FIELDS,
    ResponsesRequest,
)

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURES = _ROOT / "tests" / "fixtures" / "conversation"
_GATE_SOURCES = (
    _ROOT / "scripts" / "contract_gate.py",
    _ROOT / "scripts" / "contract_startup" / "avalan_contract_gate_plugin.py",
    _ROOT / "scripts" / "contract_startup" / "sitecustomize.py",
    _ROOT / "scripts" / "run_conversation_contract_gate.py",
    _ROOT / "scripts" / "verify_conversation_acceptance.py",
    _ROOT / "scripts" / "verify_conversation_types.py",
    Path(__file__),
)
_MARKDOWN_SUFFIX = "." + "m" + "d"
_PHASE0_ACCEPTANCE_SHA256 = (
    "2df20048e56540b1c7534126a5005ff9c667bd0ee384a899bd100131e9c34da5"
)
_PHASE0_REQUIREMENTS_SHA256 = (
    "596f3f62b99be967aa09bdb1f543447d8f7580dfea533ddcaa3aaaa95e2994fe"
)
_PHASE0_FAILURE_MATRIX_SHA256 = (
    "df62a0d8898d930a48952d50d402c4b094878596a22ac4ff6bde87368afbc6f0"
)
_PHASE0_THREAT_MODEL_SHA256 = (
    "b0fd306dbda5202fa430a264650b0f323ef287cf70e39850d532d0ef51911118"
)
_PHASE0_PROVIDER_CANONICAL_SHA256 = (
    "f479bc544e1c3c41033cc5bc719428647f02552277f38912a99a85ec1c27c15f"
)
_PHASE0_PROVIDER_CONTRACT_SHA256 = (
    "7c97b7eaf359d91523828f93a5e5bea8475eb5f08c1db7616fd19c6512a08b61"
)
_PHASE0_PROVIDER_TEST_SHA256 = (
    "953066734fc2c292c26e1fa78b0a2f2ec26ad96035e01f3f0522493e94079ce8"
)
_PHASE0_PROVIDER_SOURCE_SHA256 = (
    "47d250ded5a4e0006fe3116ed51b9552f3a2b1caa313c73d77581e09e9ee5a0d"
)
_PHASE5_PROVIDER_TRANSITION = "provider_transition.phase5.json"


def _object(path: Path) -> dict[str, object]:
    value: object = loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {str(key): item for key, item in value.items()}


def _mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _canonical_digest(value: object) -> str:
    return sha256(
        dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _without_digest(
    payload: dict[str, object],
    field: str,
) -> dict[str, object]:
    result = dict(payload)
    result.pop(field)
    return result


def _validate_scoped_digest(payload: dict[str, object]) -> None:
    digest = _mapping(payload["canonical_digest"])
    assert digest["algorithm"] == "sha256"
    scope = _sequence(digest["scope"])
    assert scope and all(isinstance(field, str) for field in scope)
    scoped = {str(field): payload[str(field)] for field in scope}
    assert digest["value"] == _canonical_digest(scoped)


def _provider_transition() -> dict[str, dict[str, object]]:
    payload = _object(_FIXTURES / _PHASE5_PROVIDER_TRANSITION)
    assert payload["schema_version"] == 1
    assert payload["feature"] == "conversation_continuity"
    assert payload["phase"] == 5
    assert payload["kind"] == "reviewed_provider_source_transition"
    assert payload["reviewed_by"] == "phase5-native-provider-review"
    assert payload["canonical_sha256"] == _canonical_digest(
        _without_digest(payload, "canonical_sha256")
    )
    transitions = [
        _mapping(value) for value in _sequence(payload["transitions"])
    ]
    assert len(transitions) == 4
    return {str(item["path"]): item for item in transitions}


def test_phase0_contract_fixtures_are_frozen(
    record_property: Callable[[str, object], None],
) -> None:
    """Validate complete, digested Phase 0 acceptance inventories."""
    record_property("conversation_acceptance_evidence", "contract")
    requirements = _object(_FIXTURES / "requirements_traceability.json")
    raw_requirements = _sequence(requirements["requirements"])
    assert requirements["normative_occurrence_count"] == 144
    assert len(raw_requirements) == 144
    assert [_mapping(value)["id"] for value in raw_requirements] == [
        f"CONV-N-{ordinal:03d}" for ordinal in range(1, 145)
    ]
    assert requirements["catalog_sha256"] == _canonical_digest(
        raw_requirements
    )
    assert _canonical_digest(raw_requirements) == _PHASE0_REQUIREMENTS_SHA256

    acceptance = _object(_FIXTURES / "acceptance_manifest.json")
    nodes = [_mapping(value) for value in _sequence(acceptance["nodes"])]
    assert sum(node["lifecycle"] == "active" for node in nodes) == 9
    assert sum(node["lifecycle"] == "planned" for node in nodes) == 13
    acceptance_payload = _without_digest(acceptance, "manifest_sha256")
    assert acceptance["manifest_sha256"] == _canonical_digest(
        acceptance_payload
    )
    assert _canonical_digest(acceptance_payload) == _PHASE0_ACCEPTANCE_SHA256

    matrix = _object(_FIXTURES / "failure_matrix.json")
    boundaries = _sequence(matrix["boundaries"])
    surfaces = _sequence(matrix["surfaces"])
    cells = _sequence(matrix["cells"])
    assert len(boundaries) == 11
    assert len(surfaces) == 9
    assert len(cells) == len(boundaries) * len(surfaces) == 99
    matrix_payload = _without_digest(matrix, "matrix_sha256")
    assert matrix["matrix_sha256"] == _canonical_digest(matrix_payload)
    assert _canonical_digest(matrix_payload) == _PHASE0_FAILURE_MATRIX_SHA256


def test_all_production_capabilities_remain_dormant(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep every provider profile dormant or incapable in Phase 0."""
    record_property("conversation_acceptance_evidence", "negative")
    decisions = _object(_FIXTURES / "contract_decisions.json")
    conformance = _object(_FIXTURES / "provider_conformance.json")
    assert decisions["activation"] == "dormant"
    assert conformance["activation_state"] == "dormant"
    assert conformance["production_advertisement_enabled"] is False
    assert conformance["production_dispatch_enabled"] is False
    profiles = [
        _mapping(value) for value in _sequence(conformance["profiles"])
    ]
    assert profiles
    assert {profile["activation_state"] for profile in profiles} <= {
        "dormant",
        "incapable",
    }
    for profile in profiles:
        assert profile["identity_complete"] is False
        assert _sequence(profile["activation_evidence"]) == []
        assert set(_mapping(profile["capabilities"]).values()) <= {
            "dormant",
            "incapable",
        }


def test_phase0_threat_controls_are_complete(
    record_property: Callable[[str, object], None],
) -> None:
    """Cover every required Phase 0 threat with controls and evidence."""
    record_property("conversation_acceptance_evidence", "security")
    model = _object(_FIXTURES / "threat_model.json")
    threats = [_mapping(value) for value in _sequence(model["threats"])]
    assert {threat["id"] for threat in threats} == {
        "opaque-state-disclosure",
        "envelope-theft",
        "confused-deputy",
        "cross-tenant-equality",
        "replay-and-rollback",
        "decompression-size-bomb",
        "orphaned-upstream-state",
        "deletion-race",
    }
    assert all(_sequence(threat["controls"]) for threat in threats)
    assert all(_sequence(threat["evidence_node_ids"]) for threat in threats)
    threat_payload = _without_digest(model, "threat_model_sha256")
    assert model["threat_model_sha256"] == _canonical_digest(threat_payload)
    assert _canonical_digest(threat_payload) == _PHASE0_THREAT_MODEL_SHA256


def test_contract_state_tables_are_total(
    record_property: Callable[[str, object], None],
) -> None:
    """Require total immutable state tables for every Phase 0 enum."""
    record_property("conversation_acceptance_evidence", "contract")
    assert set(CHECKPOINT_VISIBILITY) == set(CheckpointKind)
    assert set(CHECKPOINT_COMMIT_TRANSITIONS) == set(CheckpointCommitState)
    assert set(PUBLIC_RESPONSE_ID_TRANSITIONS) == set(PublicResponseIdState)
    assert set(RESPONSE_RESOURCE_TRANSITIONS) == set(ResponseResourceState)
    assert set(RESPONSE_OPERATION_POLICY) == set(ResponseResourceState)
    assert all(
        set(operations) == set(ResponseOperation)
        for operations in RESPONSE_OPERATION_POLICY.values()
    )


def test_provider_contract_evidence_is_typed_and_dormant(
    record_property: Callable[[str, object], None],
) -> None:
    """Bind provider claims to typed SDK evidence without activation."""
    record_property("conversation_acceptance_evidence", "wire")
    contract_path = _FIXTURES / "provider_contract.json"
    contract = _object(contract_path)
    conformance = _object(_FIXTURES / "provider_conformance.json")
    transitions = _provider_transition()
    assert len(contract_path.read_bytes()) == 34_882
    assert sha256(contract_path.read_bytes()).hexdigest() == (
        _PHASE0_PROVIDER_CONTRACT_SHA256
    )
    expected_from = {
        "tests/model/nlp/vendor_openai_conversation_phase0_test.py": (
            96_247,
            _PHASE0_PROVIDER_TEST_SHA256,
        ),
        "src/avalan/model/nlp/text/vendor/openai.py": (
            336_124,
            _PHASE0_PROVIDER_SOURCE_SHA256,
        ),
    }
    expected_to = {
        "tests/model/nlp/vendor_openai_conversation_phase0_test.py": (
            97_796,
            "67614ab06d27b44fa49c8553b5f7a7a0cad2de3979dd3cd44ee8adf8c134e08b",
        ),
        "src/avalan/model/nlp/text/vendor/openai.py": (
            337_354,
            "7fcedb4274ecbe56134c7a921c0fa0b4adc1ee02477afb1406151ac135c6c0c5",
        ),
    }
    for relative in expected_from:
        transition = transitions[relative]
        size, digest = expected_from[relative]
        assert transition["from_size"] == size
        assert transition["from_sha256"] == digest
        to_size, to_digest = expected_to[relative]
        assert transition["to_size"] == to_size
        assert transition["to_sha256"] == to_digest
    assert contract["current_phase"] == 0
    assert contract["activation_state"] == "dormant"
    assert conformance["current_phase"] == 0
    assert conformance["activation_state"] == "dormant"
    _validate_scoped_digest(contract)
    canonical_digest = _mapping(contract["canonical_digest"])
    assert canonical_digest["value"] == _PHASE0_PROVIDER_CANONICAL_SHA256
    _validate_scoped_digest(conformance)
    sdk = _mapping(contract["sdk_boundary"])
    assert str(sdk["async_resource"]).endswith(".AsyncResponses")
    policy = _mapping(sdk["conversation_state_transport_policy"])
    assert policy["scope"] == "conversation_state_and_stateful_create_fields"
    assert policy["runtime_disposition"] == "dormant_fail_closed"
    assert policy["legacy_generic_request_kwargs_acknowledged"] is True
    assert _sequence(policy["prohibited_routes"]) == [
        "extra_body",
        "conversation_state_dict[str, A" + "ny]",
        "conversation_state_mapping_unpack",
        "untyped_generation_override",
        "caller_or_dynamic_store_control",
        "background_dispatch",
        "alternate_response_create_mapping_unpack",
        "responses_lifecycle_alias_or_getattr",
        "tracked_request_binding_rebind",
        "trusted_helper_shadow",
        "runtime_namespace_or_frame_reflection",
        "reasoning_mapping_alias_or_mutator",
        "phase0_provider_source_integrity_drift",
        "runtime_non_create_response_lifecycle",
    ]
    assert _sequence(policy["provider_wire_paths"]) == [
        "background",
        "previous_response_id",
        "conversation",
        "context_management",
        "context_management.compact_threshold",
        "reasoning.context",
        "store",
    ]
    assert _sequence(policy["public_request_fields"]) == [
        "background",
        "previous_response_id",
        "conversation",
        "context_management",
        "reasoning_context",
        "conversation_handle",
        "continuation_envelope",
        "store",
    ]
    reasoning = _mapping(policy["reasoning_mapping_policy"])
    assert reasoning == {
        "mapping_name": "reasoning",
        "allowed_static_keys": ["effort", "summary"],
        "forbidden_path": "reasoning.context",
        "dynamic_keys_allowed": False,
        "aliases_allowed": False,
        "mutator_calls_allowed": False,
    }
    stateful = _mapping(policy["stateful_create_field_policy"])
    assert _sequence(stateful["forbidden_provider_wire_roots"]) == [
        "background",
        "compact_threshold",
        "context_management",
        "conversation",
        "extra_body",
        "previous_response_id",
        "store",
    ]
    typed = _mapping(stateful["typed_sdk_create_fields"])
    assert set(typed) == {"background", "store"}
    background = _mapping(typed["background"])
    store = _mapping(typed["store"])
    assert background["provider_runtime_disposition"] == "prohibited"
    assert type(background["allowed_provider_write_count"]) is int
    assert background["allowed_provider_write_count"] == 0
    assert store["provider_runtime_disposition"] == "legacy_fixed_false_only"
    assert type(store["allowed_provider_write_count"]) is int
    assert store["allowed_provider_write_count"] == 1
    assert store["allowed_provider_value"] is False
    legacy_values = _mapping(stateful["legacy_fixed_provider_values"])
    assert set(legacy_values) == {"store"}
    assert legacy_values["store"] is False
    flow = _mapping(stateful["provider_mapping_flow"])
    assert flow["initial_request_mapping"] == "kwargs"
    assert flow["normalized_request_mapping"] == "request_kwargs"
    assert flow["attempt_request_mapping"] == "attempt_kwargs"
    assert flow["create_target"] == "request_client.responses.create"
    assert flow["create_unpack_source"] == "attempt_kwargs"
    assert type(flow["create_call_count"]) is int
    assert flow["create_call_count"] == 1
    assert type(flow["mapping_unpack_count"]) is int
    assert flow["mapping_unpack_count"] == 1
    closed_gate = _mapping(stateful["closed_ast_gate"])
    assert closed_gate == {
        "tracked_bindings": [
            "attempt_kwargs",
            "kwargs",
            "normalized_request_kwargs",
            "request_client",
            "request_kwargs",
        ],
        "trusted_helpers": ["_strict_replay_json_copy", "cast"],
        "forbidden_reflection_names": [
            "eval",
            "exec",
            "globals",
            "locals",
            "vars",
        ],
        "forbidden_frame_attributes": [
            "_getframe",
            "ag_frame",
            "cr_frame",
            "currentframe",
            "f_back",
            "f_globals",
            "f_locals",
            "gi_frame",
            "tb_frame",
        ],
        "phase0_source_integrity": {
            "phase": 0,
            "kind": "exact_source_sha256",
            "algorithm": "sha256",
            "encoding": "sha256 of exact UTF-8 provider module source bytes",
            "source_path": "src/avalan/model/nlp/text/vendor/openai.py",
            "covers": [
                "module_import_and_binding_topology",
                "_strict_replay_json_copy",
                "OpenAIClient.__call__",
                "OpenAIClient._reasoning_config",
            ],
            "rotation_policy": "reviewed_provider_phase_transition_only",
            "value": _PHASE0_PROVIDER_SOURCE_SHA256,
        },
    }
    assert stateful["public_runtime_disposition"] == "dormant_fail_closed"


def test_phase0_public_surfaces_fail_closed() -> None:
    """Keep every public surface dormant before provider dispatch."""
    decisions = _object(_FIXTURES / "contract_decisions.json")
    conformance = _object(_FIXTURES / "provider_conformance.json")
    surfaces = _mapping(decisions["surfaces"])
    phase_states = {
        _mapping(value)["phase_0"]
        for value in surfaces.values()
        if isinstance(value, dict)
    }
    assert phase_states == {"dormant"}
    assert conformance["production_advertisement_enabled"] is False
    assert conformance["production_dispatch_enabled"] is False


def test_one_shot_behavior_omits_conversation_state(
    record_property: Callable[[str, object], None],
) -> None:
    """Preserve stateless defaults after explicit server activation."""
    record_property("conversation_acceptance_evidence", "runtime")
    generation_fields = {field.name for field in fields(GenerationSettings)}
    response_fields = set(ResponsesRequest.model_fields)
    prohibited = set(DORMANT_CONVERSATION_REQUEST_FIELDS)
    assert generation_fields.isdisjoint(prohibited)
    assert response_fields & prohibited == {
        "background",
        "context_management",
        "include",
        "previous_response_id",
        "store",
    }
    assert ResponsesRequest.model_fields["background"].default is False
    assert ResponsesRequest.model_fields["store"].default is False
    for field_name in (
        "context_management",
        "include",
        "previous_response_id",
    ):
        assert ResponsesRequest.model_fields[field_name].default is None


def test_tracked_gate_sources_do_not_depend_on_ignored_material(
    record_property: Callable[[str, object], None],
) -> None:
    """Keep tracked acceptance code independent of Markdown inputs."""
    record_property("conversation_acceptance_evidence", "audit")
    assert len(_GATE_SOURCES) > 0
    for path in _GATE_SOURCES:
        assert path.is_file()
        tree = parse_python(path.read_text(encoding="utf-8"))
        assert not tuple(
            node.value
            for node in walk(tree)
            if isinstance(node, Constant)
            and isinstance(node.value, str)
            and node.value.casefold().endswith(_MARKDOWN_SUFFIX)
        )
