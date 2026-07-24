#!/usr/bin/env python
"""Validate and execute structured-input acceptance tests."""

from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from json import dumps
from os import environ
from pathlib import Path, PurePosixPath
from re import compile as compile_regex
from subprocess import CompletedProcess, TimeoutExpired, run
from sys import executable, stderr
from tempfile import TemporaryDirectory
from typing import Protocol, cast
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import parse as parse_xml

from input_contract_json import StrictJsonError, strict_json_path
from verify_input_types import (
    TypeContractManifest,
    TypeContractVerificationError,
)
from verify_input_types import (
    load_manifest as load_type_manifest,
)
from verify_src_coverage import (
    CoverageVerificationError,
    verify_report_freshness,
    verify_src_coverage,
)

_FEATURE = "structured_task_input"
_MIN_PHASE = 0
_MAX_PHASE = 12
_CURRENT_PHASE = 7
_CATEGORIES = frozenset(
    (
        "unit",
        "integration",
        "negative",
        "race",
        "security",
        "public_e2e",
    )
)
_TEST_NODE_PATTERN = compile_regex(r"^tests/[A-Za-z0-9_./-]+\.py::[^\s]+$")
_DYNAMIC_CODE_PATTERN = compile_regex(r"\b(?:exec|compile)\s*\(")
_NON_PASSING_SUMMARY_PATTERN = compile_regex(
    r"\b(?:skipped|xfailed|xpassed|deselected)\b"
)
_PUBLIC_RESULT_PATTERN = compile_regex(
    r"^envelope=([a-z][a-z0-9_.-]*\.v[1-9][0-9]*)$"
)
_STATUS_PATTERN = compile_regex(r"^[a-z][a-z_]*=[^\s=]+$")
_FAILURE_TRANSITIONS = {
    "INPUT-F-01": "created->unavailable",
    "INPUT-F-02": "pending->answered",
    "INPUT-F-03": "pending->cancelled",
    "INPUT-F-04": "pending->pending",
    "INPUT-F-05": "pending->pending",
    "INPUT-F-06": "pending->pending",
    "INPUT-F-07": "answered->answered",
    "INPUT-F-08": "answered->answered",
    "INPUT-F-09": "pending->expired",
    "INPUT-F-10": "pending->cancelled",
    "INPUT-F-11": "pending->superseded",
    "INPUT-F-12": "pending->pending",
    "INPUT-F-13": "pending->timed_out",
    "INPUT-F-14": "pending->answered",
    "INPUT-F-15": "created->unavailable",
}
_FAILURE_TRANSITION_OVERRIDES = {
    ("INPUT-F-15", "mcp-inbound-task"): "running->running",
}


class AcceptanceVerificationError(RuntimeError):
    """Report invalid or non-passing acceptance evidence."""


class _JsonSchemaValidator(Protocol):
    """Describe the JSON Schema operation used by the verifier."""

    def is_valid(self, instance: object) -> bool: ...


class _JsonSchemaValidatorFactory(Protocol):
    """Describe the dynamically loaded Draft 2020-12 validator."""

    def __call__(self, schema: dict[str, object]) -> _JsonSchemaValidator: ...

    def check_schema(self, schema: dict[str, object]) -> None: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceNode:
    """Store one lifecycle-aware acceptance node."""

    id: str
    category: str
    lifecycle: str
    active_from_phase: int
    requirement_ids: tuple[str, ...]
    node_id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceManifest:
    """Store the compact acceptance inventory."""

    path: Path
    current_phase: int
    nodes: tuple[AcceptanceNode, ...]

    def active_nodes(self, through_phase: int) -> tuple[AcceptanceNode, ...]:
        """Return active nodes introduced through one phase."""
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle == "active"
            and node.active_from_phase <= through_phase
        )

    def planned_nodes(self) -> tuple[AcceptanceNode, ...]:
        """Return nodes planned strictly after the current phase."""
        return tuple(
            node for node in self.nodes if node.lifecycle == "planned"
        )

    def current_phase_nodes(self) -> tuple[AcceptanceNode, ...]:
        """Return every node activated in the implemented phase."""
        return tuple(
            node
            for node in self.active_nodes(self.current_phase)
            if node.active_from_phase == self.current_phase
        )

    def activation_history(self) -> tuple[tuple[str, ...], ...]:
        """Derive every cumulative active snapshot from node metadata."""
        return tuple(
            tuple(node.node_id for node in self.active_nodes(phase))
            for phase in range(self.current_phase + 1)
        )

    def requirement_slice(
        self,
        requirement_id: str,
        through_phase: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Derive active and remaining nodes for one requirement."""
        assert _MIN_PHASE <= through_phase <= self.current_phase
        owned = tuple(
            node
            for node in self.nodes
            if requirement_id in node.requirement_ids
        )
        active = tuple(
            node.node_id
            for node in owned
            if node.lifecycle == "active"
            and node.active_from_phase <= through_phase
        )
        active_ids = frozenset(active)
        remaining = tuple(
            node.node_id for node in owned if node.node_id not in active_ids
        )
        return active, remaining


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureSurface:
    """Store one public failure surface."""

    id: str
    active_from_phase: int


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureCondition:
    """Store one failure condition."""

    id: str
    active_from_phase: int
    requirement_id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class ApplicabilityRule:
    """Store one compact rule covering equivalent failure cells."""

    condition_id: str
    surface_ids: tuple[str, ...]
    active_from_phase: int
    negative_e2e_node: str


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureMatrix:
    """Store compact failure surfaces, conditions, and applicability."""

    surfaces: tuple[FailureSurface, ...]
    conditions: tuple[FailureCondition, ...]
    rules: tuple[ApplicabilityRule, ...]

    def applicable_cells(self) -> frozenset[tuple[str, str]]:
        """Derive every applicable condition/surface pair."""
        return frozenset(
            (rule.condition_id, surface_id)
            for rule in self.rules
            for surface_id in rule.surface_ids
        )

    def all_cells(self) -> frozenset[tuple[str, str]]:
        """Derive the complete Cartesian failure matrix."""
        return frozenset(
            (condition.id, surface.id)
            for condition in self.conditions
            for surface in self.surfaces
        )

    def evidence_nodes(self, through_phase: int) -> tuple[str, ...]:
        """Return unique active E2E nodes required by applicable rules."""
        return tuple(
            dict.fromkeys(
                rule.negative_e2e_node
                for rule in self.rules
                if rule.active_from_phase <= through_phase
            )
        )


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def fixture_root() -> Path:
    """Return the tracked input-contract fixture directory."""
    return repository_root() / "tests" / "fixtures" / "input"


def default_manifest_path() -> Path:
    """Return the tracked acceptance-manifest path."""
    return fixture_root() / "acceptance_manifest.json"


def load_manifest(path: Path) -> AcceptanceManifest:
    """Load and validate the compact lifecycle-aware manifest."""
    payload = _strict_mapping(path, "acceptance manifest")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "categories",
            "replacements",
            "nodes",
        },
        "acceptance manifest",
    )
    _header(payload, "acceptance manifest", schema_version=2)
    current_phase = _phase(payload.get("current_phase"), "current_phase")
    categories = _string_list(payload.get("categories"), "categories")
    if frozenset(categories) != _CATEGORIES or len(categories) != len(
        _CATEGORIES
    ):
        raise AcceptanceVerificationError(
            "acceptance categories must be the exact required inventory"
        )
    raw_nodes = _list(payload.get("nodes"), "acceptance nodes")
    if not raw_nodes:
        raise AcceptanceVerificationError("acceptance nodes must be non-empty")
    nodes = tuple(_acceptance_node(raw, current_phase) for raw in raw_nodes)
    _unique((node.id for node in nodes), "acceptance node ID")
    _unique((node.node_id for node in nodes), "pytest node ID")
    if frozenset(node.category for node in nodes) != _CATEGORIES:
        raise AcceptanceVerificationError(
            "every acceptance category must own at least one node"
        )
    for phase in range(current_phase + 1):
        if not any(
            node.lifecycle == "active" and node.active_from_phase == phase
            for node in nodes
        ):
            raise AcceptanceVerificationError(
                f"active_from_phase inventory has a gap at phase {phase}"
            )
    if current_phase < _MAX_PHASE and not any(
        node.lifecycle == "planned" for node in nodes
    ):
        raise AcceptanceVerificationError(
            "future phases must remain explicitly planned"
        )
    _validate_replacements(payload.get("replacements"), nodes, current_phase)
    manifest = AcceptanceManifest(
        path=path,
        current_phase=current_phase,
        nodes=nodes,
    )
    history = manifest.activation_history()
    if any(
        not set(history[phase]).issubset(history[phase + 1])
        for phase in range(len(history) - 1)
    ):
        raise AcceptanceVerificationError(
            "derived activation history is not monotonic"
        )
    return manifest


def load_failure_matrix(
    path: Path,
    *,
    manifest: AcceptanceManifest | None = None,
    requirement_ids: frozenset[str] | None = None,
    decision_surface_ids: frozenset[str] | None = None,
    public_envelope_ids: frozenset[str] | None = None,
) -> FailureMatrix:
    """Load compact rules and derive the complete failure matrix."""
    payload = _strict_mapping(path, "failure matrix")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "observation_window",
            "domain_side_effect_scope",
            "surfaces",
            "conditions",
            "applicability_rules",
            "matrix_sha256",
        },
        "failure matrix",
    )
    _header(payload, "failure matrix", schema_version=2)
    for field in ("observation_window", "domain_side_effect_scope"):
        _nonempty_string(payload.get(field), f"failure {field}")
    surfaces = tuple(
        _failure_surface(raw)
        for raw in _list(payload.get("surfaces"), "failure surfaces")
    )
    conditions = tuple(
        _failure_condition(raw)
        for raw in _list(payload.get("conditions"), "failure conditions")
    )
    if not surfaces or not conditions:
        raise AcceptanceVerificationError(
            "failure surfaces and conditions must be non-empty"
        )
    _unique((surface.id for surface in surfaces), "failure surface ID")
    _unique((condition.id for condition in conditions), "failure condition ID")
    surface_by_id = {surface.id: surface for surface in surfaces}
    condition_by_id = {condition.id: condition for condition in conditions}
    if set(condition_by_id) != set(_FAILURE_TRANSITIONS):
        raise AcceptanceVerificationError(
            "failure conditions differ from transition semantics"
        )
    rules = tuple(
        _applicability_rule(raw, surface_by_id, condition_by_id)
        for raw in _list(
            payload.get("applicability_rules"), "applicability rules"
        )
    )
    if not rules:
        raise AcceptanceVerificationError(
            "applicability rules must be non-empty"
        )
    cells = [
        (rule.condition_id, surface_id)
        for rule in rules
        for surface_id in rule.surface_ids
    ]
    _unique(cells, "applicable failure cell")
    matrix = FailureMatrix(
        surfaces=surfaces,
        conditions=conditions,
        rules=rules,
    )
    if not matrix.applicable_cells() < matrix.all_cells():
        raise AcceptanceVerificationError(
            "failure rules must derive applicable and non-applicable cells"
        )
    applicable_conditions = {
        condition_id for condition_id, _ in matrix.applicable_cells()
    }
    if applicable_conditions != set(condition_by_id):
        raise AcceptanceVerificationError(
            "every failure condition must have an applicable surface"
        )
    if decision_surface_ids is not None and decision_surface_ids != frozenset(
        surface_by_id
    ):
        raise AcceptanceVerificationError(
            "failure surfaces differ from contract decisions"
        )
    if requirement_ids is not None and any(
        condition.requirement_id not in requirement_ids
        for condition in conditions
    ):
        raise AcceptanceVerificationError(
            "failure condition references an unknown requirement"
        )
    if public_envelope_ids is not None:
        for raw in _list(
            payload.get("applicability_rules"), "applicability rules"
        ):
            rule_payload = _mapping(raw, "applicability rule")
            match = _PUBLIC_RESULT_PATTERN.fullmatch(
                _nonempty_string(
                    rule_payload.get("public_result"),
                    "failure public_result",
                )
            )
            if match is None or match.group(1) not in public_envelope_ids:
                raise AcceptanceVerificationError(
                    "failure rule references an unknown public envelope"
                )
    if manifest is not None:
        node_by_id = {node.node_id: node for node in manifest.nodes}
        for rule in rules:
            node = node_by_id.get(rule.negative_e2e_node)
            if node is None or node.active_from_phase > rule.active_from_phase:
                raise AcceptanceVerificationError(
                    "failure evidence node is absent or activates too late: "
                    f"{rule.negative_e2e_node}"
                )
            if (
                rule.active_from_phase <= manifest.current_phase
                and node.lifecycle != "active"
            ):
                raise AcceptanceVerificationError(
                    "current failure evidence must be active"
                )
    canonical = {
        key: value for key, value in payload.items() if key != "matrix_sha256"
    }
    if payload.get("matrix_sha256") != _digest(canonical):
        raise AcceptanceVerificationError("failure matrix digest is invalid")
    return matrix


def verify_current_runtime(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> AcceptanceManifest:
    """Execute current-phase behavioral nodes only."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    if manifest.current_phase != _CURRENT_PHASE:
        raise AcceptanceVerificationError(
            "runtime-only verification requires the current phase"
        )
    _require_database_harness()
    _validate_contract_fixtures(manifest, path.parent, root)
    nodes = manifest.current_phase_nodes()
    _verify_nodes(nodes, root)
    return manifest


def verify_acceptance(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
    contract_fixture_root: Path | None = None,
) -> AcceptanceManifest:
    """Validate fixtures and execute every selected active test."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    if not _MIN_PHASE <= through_phase <= manifest.current_phase:
        raise AcceptanceVerificationError(
            "through-phase must be implemented by the current manifest"
        )
    if through_phase >= _CURRENT_PHASE:
        _require_database_harness()
    fixtures = contract_fixture_root or path.parent
    _validate_contract_fixtures(manifest, fixtures, root)
    if through_phase >= _CURRENT_PHASE:
        _validate_fresh_coverage(root)
    nodes = manifest.active_nodes(through_phase)
    if not nodes:
        raise AcceptanceVerificationError(
            "the selected acceptance inventory has no active nodes"
        )
    _verify_nodes(nodes, root)
    return manifest


def _validate_contract_fixtures(
    manifest: AcceptanceManifest,
    fixtures: Path,
    root: Path,
) -> None:
    requirements = _validate_requirements(
        fixtures / "requirements_traceability.json", manifest
    )
    decision_surfaces, envelopes = _validate_decisions(
        fixtures / "contract_decisions.json"
    )
    load_failure_matrix(
        fixtures / "failure_matrix.json",
        manifest=manifest,
        requirement_ids=requirements,
        decision_surface_ids=decision_surfaces,
        public_envelope_ids=envelopes,
    )
    _validate_deterministic_fixtures(fixtures / "deterministic_fixtures.json")
    _validate_no_bc(fixtures / "no_bc_removals.json")
    _validate_evidence(fixtures / "baseline_evidence.json", manifest)
    try:
        type_manifest = load_type_manifest(
            fixtures / "type_contract_manifest.json"
        )
    except TypeContractVerificationError as exc:
        raise AcceptanceVerificationError(str(exc)) from exc
    _validate_type_contract_phase(manifest, type_manifest)
    if not root.is_dir():
        raise AcceptanceVerificationError("repository root does not exist")


def _validate_requirements(
    path: Path,
    manifest: AcceptanceManifest,
) -> frozenset[str]:
    payload = _strict_mapping(path, "requirements traceability")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "source_sections",
            "requirements",
            "catalog_sha256",
        },
        "requirements traceability",
    )
    _header(payload, "requirements traceability", schema_version=2)
    source_sections = _string_list(
        payload.get("source_sections"), "source sections"
    )
    _unique(source_sections, "source section")
    raw_requirements = _list(payload.get("requirements"), "requirements")
    requirement_ids: list[str] = []
    allowed_sections = set(source_sections) | {"delivery_gate"}
    for raw in raw_requirements:
        item = _mapping(raw, "requirement")
        _exact_keys(
            item,
            {
                "id",
                "source_section",
                "normative_level",
                "paraphrase",
                "owner",
            },
            "requirement",
        )
        requirement_ids.append(
            _nonempty_string(item.get("id"), "requirement ID")
        )
        source_section = item.get("source_section")
        if not isinstance(source_section, str) or (
            source_section not in allowed_sections
            and source_section.split(".", 1)[0] not in allowed_sections
        ):
            raise AcceptanceVerificationError(
                "requirement references an unknown source section"
            )
        if item.get("normative_level") not in {
            "MUST",
            "SHOULD",
            "MAY",
            "SCENARIO",
        }:
            raise AcceptanceVerificationError(
                "requirement normative level is invalid"
            )
        _nonempty_string(item.get("paraphrase"), "requirement paraphrase")
        _nonempty_string(item.get("owner"), "requirement owner")
    _unique(requirement_ids, "requirement ID")
    node_requirements = {
        requirement_id
        for node in manifest.nodes
        for requirement_id in node.requirement_ids
    }
    if node_requirements != set(requirement_ids):
        raise AcceptanceVerificationError(
            "manifest and requirement catalog coverage differ"
        )
    if payload.get("catalog_sha256") != _digest(raw_requirements):
        raise AcceptanceVerificationError(
            "requirement catalog digest is invalid"
        )
    return frozenset(requirement_ids)


def _validate_decisions(
    path: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    payload = _strict_mapping(path, "contract decisions")
    required = {
        "schema_version",
        "feature",
        "identity",
        "request_bounds",
        "question_contracts",
        "state_transitions",
        "outcome_to_model",
        "execution",
        "capability_matrix",
        "protocol_projection",
        "privacy",
        "error_status",
        "repeated_requests",
        "activation",
        "capacity_budgets",
        "contract_sha256",
    }
    _exact_keys(payload, required, "contract decisions")
    _header(payload, "contract decisions", schema_version=1)
    for key in required - {"schema_version", "feature", "contract_sha256"}:
        if not isinstance(payload.get(key), (dict, list)) or not payload[key]:
            raise AcceptanceVerificationError(
                f"contract decision {key} must be populated"
            )
    activation = _mapping(payload.get("activation"), "activation")
    if activation.get("production_default") != "absent":
        raise AcceptanceVerificationError(
            "structured input must remain absent in production"
        )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "contract_sha256"
    }
    if payload.get("contract_sha256") != _digest(canonical):
        raise AcceptanceVerificationError(
            "contract decision digest is invalid"
        )
    capability = _mapping(
        payload.get("capability_matrix"), "capability matrix"
    )
    surfaces = _string_list(
        capability.get("public_failure_surface_ids"),
        "public failure surface IDs",
    )
    _unique(surfaces, "public failure surface ID")
    error_status = _mapping(payload.get("error_status"), "error status")
    catalog = _mapping(
        error_status.get("public_envelope_catalog"),
        "public envelope catalog",
    )
    examples = _mapping(
        error_status.get("public_envelope_examples"),
        "public envelope examples",
    )
    if set(catalog) != set(examples):
        raise AcceptanceVerificationError(
            "public envelope schemas and examples differ"
        )
    for envelope_id, raw_schema in catalog.items():
        schema = _mapping(raw_schema, f"public envelope {envelope_id}")
        _check_schema(schema, f"public envelope {envelope_id}")
        if not _draft_validator()(schema).is_valid(examples[envelope_id]):
            raise AcceptanceVerificationError(
                f"public envelope example is invalid: {envelope_id}"
            )
    _validate_known_schemas(payload)
    return frozenset(surfaces), frozenset(catalog)


def _validate_known_schemas(payload: dict[str, object]) -> None:
    paths = (
        ("identity", "state_revision", "wire_schema"),
        ("execution", "attached_result_schema"),
        ("execution", "detached_result_schema"),
        ("execution", "incapable_result_schema"),
        ("execution", "provider_snapshot", "schema"),
        (
            "protocol_projection",
            "openai_compatible",
            "extension",
            "request_body_field",
            "schema",
        ),
        (
            "protocol_projection",
            "openai_compatible",
            "semantic_request_retrieval",
            "response_schema",
        ),
        ("protocol_projection", "mcp", "tasks", "params_task_schema"),
        ("protocol_projection", "mcp", "tasks", "CreateTaskResult"),
        ("protocol_projection", "mcp", "tasks", "task_schema"),
        (
            "protocol_projection",
            "a2a",
            "extension",
            "message_metadata_schema",
        ),
        ("privacy", "submitted_secret_policy", "classification_schema"),
    )
    for path in paths:
        schema = _mapping(_at_path(payload, path), ".".join(path))
        _check_schema(schema, ".".join(path))
    streaming = _mapping(
        _at_path(
            payload,
            (
                "protocol_projection",
                "openai_compatible",
                "streaming_event_schemas",
            ),
        ),
        "streaming event schemas",
    )
    for name, raw_schema in streaming.items():
        _check_schema(_mapping(raw_schema, name), name)


def _validate_deterministic_fixtures(path: Path) -> None:
    payload = _strict_mapping(path, "deterministic fixtures")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "clock",
            "id_factory",
            "principal",
            "local_peer",
            "provider_calls",
            "barrier",
        },
        "deterministic fixtures",
    )
    _header(payload, "deterministic fixtures", schema_version=1)
    for key in set(payload) - {"schema_version", "feature"}:
        if not isinstance(payload[key], (dict, list)) or not payload[key]:
            raise AcceptanceVerificationError(
                f"deterministic fixture {key} must be populated"
            )


def _validate_no_bc(path: Path) -> None:
    payload = _strict_mapping(path, "no-BC removals")
    _exact_keys(
        payload,
        {"schema_version", "feature", "removals", "inventory_sha256"},
        "no-BC removals",
    )
    _header(payload, "no-BC removals", schema_version=1)
    raw_removals = _list(payload.get("removals"), "no-BC removals")
    if not raw_removals:
        raise AcceptanceVerificationError(
            "no-BC removal inventory must be non-empty"
        )
    ids: list[str] = []
    for raw in raw_removals:
        item = _mapping(raw, "no-BC removal")
        _exact_keys(
            item,
            {
                "id",
                "current_path",
                "remove_by_phase",
                "replacement",
                "evidence",
            },
            "no-BC removal",
        )
        ids.append(_nonempty_string(item.get("id"), "no-BC ID"))
        _nonempty_string(item.get("current_path"), "no-BC current path")
        _phase(item.get("remove_by_phase"), "no-BC removal phase")
        _nonempty_string(item.get("replacement"), "no-BC replacement")
        _nonempty_string(item.get("evidence"), "no-BC evidence")
    _unique(ids, "no-BC ID")
    if payload.get("inventory_sha256") != _digest(raw_removals):
        raise AcceptanceVerificationError(
            "no-BC removal inventory digest is invalid"
        )


def _validate_evidence(path: Path, manifest: AcceptanceManifest) -> None:
    payload = _strict_mapping(path, "acceptance evidence")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "authoritative_gate",
            "invariants",
            "evidence_sha256",
        },
        "acceptance evidence",
    )
    _header(payload, "acceptance evidence", schema_version=2)
    if payload.get("current_phase") != manifest.current_phase:
        raise AcceptanceVerificationError(
            "evidence and manifest phases differ"
        )
    gate = _mapping(payload.get("authoritative_gate"), "authoritative gate")
    expected_gate = {
        "command": "make test-pgsql-exact no-install INPUT_PHASE=7",
        "database_dsn_env": "AVALAN_TASK_TEST_POSTGRESQL_DSN",
        "coverage_report": "coverage.json",
        "coverage_scope": "src/",
        "fresh_report_required": True,
        "coverage_before_acceptance": True,
        "acceptance_before_database_teardown": True,
    }
    if gate != expected_gate:
        raise AcceptanceVerificationError(
            "authoritative gate evidence changed"
        )
    invariants = _mapping(payload.get("invariants"), "evidence invariants")
    expected_invariants = {
        "planned_nodes_are_not_evidence": True,
        "activation_is_derived_from_nodes": True,
        "failure_cells_are_derived_from_rules": True,
        "reject_skip_xfail_deselection": True,
        "reject_exec_compile_coverage_tricks": True,
        "exact_source_coverage": True,
        "fail_closed": True,
    }
    if invariants != expected_invariants:
        raise AcceptanceVerificationError(
            "acceptance evidence invariants changed"
        )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "evidence_sha256"
    }
    if payload.get("evidence_sha256") != _digest(canonical):
        raise AcceptanceVerificationError(
            "acceptance evidence digest is invalid"
        )


def _validate_fresh_coverage(root: Path) -> None:
    report = root / "coverage.json"
    try:
        verify_report_freshness(report, root)
        verify_src_coverage(report, repo_root=root)
    except (CoverageVerificationError, StrictJsonError) as exc:
        raise AcceptanceVerificationError(str(exc)) from exc


def _validate_type_contract_phase(
    manifest: AcceptanceManifest,
    type_manifest: TypeContractManifest,
) -> None:
    """Allow one acceptance-only phase with no new type obligations."""
    if type_manifest.current_phase == manifest.current_phase:
        return
    has_new_obligation = any(
        fixture.active_from_phase == manifest.current_phase
        for fixture in type_manifest.fixtures
    )
    if (
        manifest.current_phase != type_manifest.current_phase + 1
        or has_new_obligation
    ):
        raise AcceptanceVerificationError(
            "type and acceptance phases may differ only for one "
            "acceptance-only phase without new type obligations"
        )


def _verify_nodes(
    nodes: tuple[AcceptanceNode, ...],
    root: Path,
) -> tuple[str, ...]:
    node_ids = tuple(node.node_id for node in nodes)
    test_files = tuple(
        dict.fromkeys(node_id.split("::", 1)[0] for node_id in node_ids)
    )
    for relative in test_files:
        path = (root / relative).resolve()
        if not path.is_relative_to(root / "tests") or not path.is_file():
            raise AcceptanceVerificationError(
                f"active acceptance test does not exist: {relative}"
            )
        content = path.read_text(encoding="utf-8")
        match = _DYNAMIC_CODE_PATTERN.search(content)
        if match is not None:
            raise AcceptanceVerificationError(
                "active tests contain a prohibited coverage trick using "
                f"dynamic code: {relative}:{match.group(0)}"
            )
    collection = _pytest(
        root,
        ("--collect-only", "-q", *node_ids),
        timeout=180,
    )
    if collection.returncode != 0:
        raise AcceptanceVerificationError(
            "pytest collection failed:\n" + collection.stdout[-4000:]
        )
    collected = tuple(
        line.strip()
        for line in collection.stdout.splitlines()
        if line.startswith("tests/") and "::" in line
    )
    if not collected or len(collected) != len(set(collected)):
        raise AcceptanceVerificationError(
            "pytest collection is empty or duplicated"
        )
    for node_id in node_ids:
        if not any(
            collected_id == node_id or collected_id.startswith(f"{node_id}[")
            for collected_id in collected
        ):
            raise AcceptanceVerificationError(
                f"pytest did not collect active node: {node_id}"
            )
    with TemporaryDirectory(prefix="avalan-input-acceptance-") as temporary:
        junit = Path(temporary) / "pytest.xml"
        execution = _pytest(
            root,
            (
                "-q",
                "-r",
                "xXs",
                "-o",
                "junit_family=legacy",
                f"--junitxml={junit}",
                *node_ids,
            ),
            timeout=900,
        )
        if execution.returncode != 0:
            raise AcceptanceVerificationError(
                "pytest acceptance execution failed:\n"
                + execution.stdout[-8000:]
            )
        if _NON_PASSING_SUMMARY_PATTERN.search(execution.stdout):
            raise AcceptanceVerificationError(
                "acceptance execution skipped, xfailed, xpassed, or "
                "deselected tests"
            )
        if not junit.is_file():
            raise AcceptanceVerificationError(
                "pytest did not write execution evidence"
            )
        root_element = parse_xml(junit).getroot()
        suites = (
            tuple(root_element)
            if root_element.tag == "testsuites"
            else (root_element,)
        )
        totals = {
            key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        executed = tuple(
            _junit_testcase_id(testcase)
            for suite in suites
            for testcase in suite.iter("testcase")
        )
        if (
            totals["tests"] < len(collected)
            or len(executed) != len(set(executed))
            or set(executed) != set(collected)
            or any(totals[key] for key in ("failures", "errors", "skipped"))
        ):
            raise AcceptanceVerificationError(
                "pytest execution evidence does not match collected instance "
                f"IDs: {totals}"
            )
    return collected


def _junit_testcase_id(testcase: Element) -> str:
    """Return one exact pytest instance ID from legacy JUnit evidence."""
    relative = _nonempty_string(
        testcase.attrib.get("file"), "JUnit testcase file"
    )
    path = PurePosixPath(relative)
    if path.is_absolute() or path.suffix != ".py" or ".." in path.parts:
        raise AcceptanceVerificationError(
            f"JUnit testcase file is invalid: {relative}"
        )
    name = _nonempty_string(testcase.attrib.get("name"), "JUnit testcase name")
    classname = _nonempty_string(
        testcase.attrib.get("classname"), "JUnit testcase classname"
    )
    module_name = ".".join(path.with_suffix("").parts)
    if classname == module_name:
        class_parts: tuple[str, ...] = ()
    elif classname.startswith(f"{module_name}."):
        class_parts = tuple(classname[len(module_name) + 1 :].split("."))
    else:
        raise AcceptanceVerificationError(
            "JUnit testcase classname does not match its file"
        )
    if any(not part for part in class_parts):
        raise AcceptanceVerificationError(
            "JUnit testcase classname has an empty component"
        )
    return "::".join((relative, *class_parts, name))


def _pytest(
    root: Path,
    arguments: tuple[str, ...],
    *,
    timeout: int,
) -> CompletedProcess[str]:
    environment = {
        key: value
        for key, value in environ.items()
        if key.upper()
        not in {"PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
    }
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTHONPATH"] = str(root / "src")
    return run(
        (
            executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            *arguments,
        ),
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _acceptance_node(raw: object, current_phase: int) -> AcceptanceNode:
    item = _mapping(raw, "acceptance node")
    _exact_keys(
        item,
        {
            "id",
            "category",
            "lifecycle",
            "active_from_phase",
            "requirement_ids",
            "node_id",
        },
        "acceptance node",
    )
    category = _nonempty_string(item.get("category"), "node category")
    if category not in _CATEGORIES:
        raise AcceptanceVerificationError(
            f"acceptance node category is invalid: {category}"
        )
    lifecycle = _nonempty_string(item.get("lifecycle"), "node lifecycle")
    phase = _phase(item.get("active_from_phase"), "active_from_phase")
    if lifecycle not in {"active", "planned"}:
        raise AcceptanceVerificationError(
            "acceptance node lifecycle must be active or planned"
        )
    expected_lifecycle = "active" if phase <= current_phase else "planned"
    if lifecycle != expected_lifecycle:
        raise AcceptanceVerificationError(
            "node lifecycle disagrees with active_from_phase"
        )
    requirement_ids = _string_list(
        item.get("requirement_ids"), "node requirement_ids"
    )
    if not requirement_ids:
        raise AcceptanceVerificationError(
            "acceptance node must cover a requirement"
        )
    _unique(requirement_ids, "node requirement ID")
    node_id = _test_node(item.get("node_id"))
    return AcceptanceNode(
        id=_nonempty_string(item.get("id"), "acceptance node ID"),
        category=category,
        lifecycle=lifecycle,
        active_from_phase=phase,
        requirement_ids=requirement_ids,
        node_id=node_id,
    )


def _validate_replacements(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> None:
    replacements = _list(raw, "acceptance replacements")
    current_ids = {node.node_id for node in nodes}
    old_ids: list[str] = []
    replacement_ids: list[str] = []
    for value in replacements:
        item = _mapping(value, "acceptance replacement")
        _exact_keys(
            item,
            {
                "phase",
                "old_node_id",
                "replacement_node_ids",
                "requirement_ids",
                "reviewed_by",
                "evidence",
            },
            "acceptance replacement",
        )
        if _phase(item.get("phase"), "replacement phase") > current_phase:
            raise AcceptanceVerificationError(
                "implemented replacements cannot be future planned work"
            )
        old_id = _test_node(item.get("old_node_id"))
        replacements_for_item = _string_list(
            item.get("replacement_node_ids"), "replacement node IDs"
        )
        if (
            old_id in current_ids
            or not replacements_for_item
            or any(
                node_id not in current_ids for node_id in replacements_for_item
            )
        ):
            raise AcceptanceVerificationError(
                "replacement tombstone does not match the current inventory"
            )
        old_ids.append(old_id)
        replacement_ids.extend(replacements_for_item)
        _string_list(item.get("requirement_ids"), "replacement requirements")
        _nonempty_string(item.get("reviewed_by"), "replacement reviewer")
        _nonempty_string(item.get("evidence"), "replacement evidence")
    _unique(old_ids, "replaced node ID")
    if len(replacement_ids) != len(set(replacement_ids)):
        raise AcceptanceVerificationError(
            "a current node is claimed by multiple replacements"
        )


def _failure_surface(raw: object) -> FailureSurface:
    item = _mapping(raw, "failure surface")
    _exact_keys(
        item,
        {"id", "description", "active_from_phase"},
        "failure surface",
    )
    _nonempty_string(item.get("description"), "surface description")
    return FailureSurface(
        id=_nonempty_string(item.get("id"), "surface ID"),
        active_from_phase=_phase(
            item.get("active_from_phase"), "surface active_from_phase"
        ),
    )


def _failure_condition(raw: object) -> FailureCondition:
    item = _mapping(raw, "failure condition")
    _exact_keys(
        item,
        {"id", "description", "active_from_phase", "requirement_id"},
        "failure condition",
    )
    _nonempty_string(item.get("description"), "condition description")
    return FailureCondition(
        id=_nonempty_string(item.get("id"), "condition ID"),
        active_from_phase=_phase(
            item.get("active_from_phase"), "condition active_from_phase"
        ),
        requirement_id=_nonempty_string(
            item.get("requirement_id"), "condition requirement ID"
        ),
    )


def _applicability_rule(
    raw: object,
    surfaces: dict[str, FailureSurface],
    conditions: dict[str, FailureCondition],
) -> ApplicabilityRule:
    item = _mapping(raw, "applicability rule")
    _exact_keys(
        item,
        {
            "condition_id",
            "surface_ids",
            "active_from_phase",
            "expected_transition",
            "public_result",
            "status_or_exit",
            "provider_call_count",
            "domain_side_effect_count",
            "negative_e2e_node",
        },
        "applicability rule",
    )
    condition_id = _nonempty_string(
        item.get("condition_id"), "rule condition ID"
    )
    if condition_id not in conditions:
        raise AcceptanceVerificationError(
            f"applicability rule has unknown condition: {condition_id}"
        )
    surface_ids = _string_list(item.get("surface_ids"), "rule surface IDs")
    if not surface_ids:
        raise AcceptanceVerificationError(
            "applicability rule must cover a surface"
        )
    _unique(surface_ids, "rule surface ID")
    if any(surface_id not in surfaces for surface_id in surface_ids):
        raise AcceptanceVerificationError(
            "applicability rule has an unknown surface"
        )
    phase = _phase(item.get("active_from_phase"), "rule active_from_phase")
    minimum_phase = max(
        conditions[condition_id].active_from_phase,
        *(
            surfaces[surface_id].active_from_phase
            for surface_id in surface_ids
        ),
    )
    if phase < minimum_phase:
        raise AcceptanceVerificationError(
            "applicability rule activates before its condition or surface"
        )
    expected_transition = _nonempty_string(
        item.get("expected_transition"), "failure expected_transition"
    )
    for surface_id in surface_ids:
        semantic_transition = _FAILURE_TRANSITION_OVERRIDES.get(
            (condition_id, surface_id),
            _FAILURE_TRANSITIONS[condition_id],
        )
        if expected_transition != semantic_transition:
            raise AcceptanceVerificationError(
                "failure expected_transition does not match condition and "
                f"surface semantics: {condition_id}/{surface_id}"
            )
    if (
        _PUBLIC_RESULT_PATTERN.fullmatch(
            _nonempty_string(
                item.get("public_result"), "failure public_result"
            )
        )
        is None
    ):
        raise AcceptanceVerificationError(
            "failure public_result must name one envelope"
        )
    if (
        _STATUS_PATTERN.fullmatch(
            _nonempty_string(item.get("status_or_exit"), "status_or_exit")
        )
        is None
    ):
        raise AcceptanceVerificationError(
            "failure status_or_exit must be one machine literal"
        )
    _nonnegative_int(item.get("provider_call_count"), "provider call count")
    _nonnegative_int(
        item.get("domain_side_effect_count"), "domain side-effect count"
    )
    return ApplicabilityRule(
        condition_id=condition_id,
        surface_ids=surface_ids,
        active_from_phase=phase,
        negative_e2e_node=_test_node(item.get("negative_e2e_node")),
    )


def _check_schema(schema: dict[str, object], label: str) -> None:
    try:
        _draft_validator().check_schema(schema)
    except Exception as exc:
        raise AcceptanceVerificationError(
            f"invalid JSON Schema for {label}: {exc}"
        ) from exc


def _draft_validator() -> _JsonSchemaValidatorFactory:
    module = import_module("jsonschema.validators")
    factory = getattr(module, "Draft202012Validator", None)
    if not callable(factory):
        raise AcceptanceVerificationError(
            "jsonschema Draft202012Validator is unavailable"
        )
    return cast(_JsonSchemaValidatorFactory, factory)


def _at_path(
    value: object,
    path: tuple[str, ...],
) -> object:
    current = value
    for part in path:
        current = _mapping(current, ".".join(path)).get(part)
    return current


def _require_database_harness() -> None:
    if not environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN"):
        raise AcceptanceVerificationError(
            "current acceptance inventory requires the real PostgreSQL harness"
        )


def _strict_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        return _mapping(strict_json_path(path), label)
    except StrictJsonError as exc:
        raise AcceptanceVerificationError(
            f"cannot read {label}: {exc}"
        ) from exc


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AcceptanceVerificationError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise AcceptanceVerificationError(f"{label} must be a list")
    return value


def _header(
    payload: dict[str, object],
    label: str,
    *,
    schema_version: int,
) -> None:
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != schema_version
    ):
        raise AcceptanceVerificationError(
            f"{label} schema_version must be {schema_version}"
        )
    if payload.get("feature") != _FEATURE:
        raise AcceptanceVerificationError(
            f"{label} feature must be {_FEATURE}"
        )


def _phase(value: object, label: str) -> int:
    if type(value) is not int or not _MIN_PHASE <= value <= _MAX_PHASE:
        raise AcceptanceVerificationError(
            f"{label} must be an integer from {_MIN_PHASE} through"
            f" {_MAX_PHASE}"
        )
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AcceptanceVerificationError(
            f"{label} must be a non-negative integer"
        )
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise AcceptanceVerificationError(
            f"{label} must be a non-empty string"
        )
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    raw = _list(value, label)
    return tuple(_nonempty_string(item, label) for item in raw)


def _test_node(value: object) -> str:
    node_id = _nonempty_string(value, "pytest node ID")
    if (
        _TEST_NODE_PATTERN.fullmatch(node_id) is None
        or "\\" in node_id
        or ".." in PurePosixPath(node_id.split("::", 1)[0]).parts
    ):
        raise AcceptanceVerificationError(f"invalid pytest node ID: {node_id}")
    return node_id


def _unique(values: Iterable[object], label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise AcceptanceVerificationError(f"duplicate {label}")


def _exact_keys(
    value: dict[str, object],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise AcceptanceVerificationError(
            f"{label} has invalid keys: {sorted(set(value) ^ expected)}"
        )


def _digest(value: object) -> str:
    return sha256(
        dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Collect and execute active structured-input acceptance tests "
            "without skips, xfails, deselection, or synthetic coverage."
        )
    )
    parser.add_argument("--through-phase", required=True, type=int)
    parser.add_argument(
        "--manifest", type=Path, default=default_manifest_path()
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--runtime-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run acceptance verification from the command line."""
    args = _parse_args()
    try:
        if args.runtime_only:
            if args.through_phase != _CURRENT_PHASE:
                raise AcceptanceVerificationError(
                    "--runtime-only requires the current implemented phase"
                )
            manifest = verify_current_runtime(
                args.manifest, repo_root=args.repo_root
            )
            node_count = len(manifest.current_phase_nodes())
        else:
            manifest = verify_acceptance(
                args.manifest,
                repo_root=args.repo_root,
                through_phase=args.through_phase,
            )
            node_count = len(manifest.active_nodes(args.through_phase))
    except (
        AcceptanceVerificationError,
        CoverageVerificationError,
        StrictJsonError,
        TimeoutExpired,
    ) as exc:
        print(f"structured-input acceptance failed: {exc}", file=stderr)
        return 1
    print(
        "structured-input acceptance passed: "
        f"through_phase={args.through_phase} nodes={node_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
