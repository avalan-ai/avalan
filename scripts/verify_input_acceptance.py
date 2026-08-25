#!/usr/bin/env python
"""Validate and execute structured-input acceptance tests."""

from argparse import ArgumentParser, Namespace
from ast import AST, AsyncFunctionDef, ClassDef, FunctionDef
from ast import parse as parse_python
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from json import JSONDecodeError, dumps, loads
from os import environ
from pathlib import Path, PurePosixPath
from re import compile as compile_regex
from subprocess import CompletedProcess, TimeoutExpired
from sys import stderr
from tempfile import TemporaryDirectory
from typing import Protocol, cast
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import parse as parse_xml

from contract_gate import (
    POSTGRESQL_TEST_DSN_ENV,
    ContractGateError,
    junit_testcase_id,
    run_pytest,
)
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
_CURRENT_PHASE = _MAX_PHASE
_POSTGRESQL_ACCEPTANCE_PREFIX = (
    "tests/interaction/stores/interaction_pgsql_e2e.py::"
)
_CATEGORY_VALUES = (
    "unit",
    "integration",
    "negative",
    "race",
    "security",
    "public_e2e",
)
_CATEGORIES = frozenset(_CATEGORY_VALUES)
_TEST_NODE_PATTERN = compile_regex(r"^tests/[A-Za-z0-9_./-]+\.py::[^\s]+$")
_DYNAMIC_CODE_PATTERN = compile_regex(r"\b(?:exec|compile)\s*\(")
_NON_PASSING_SUMMARY_PATTERN = compile_regex(
    r"\b(?:skipped|xfailed|xpassed|deselected)\b"
)
_PUBLIC_RESULT_PATTERN = compile_regex(
    r"^envelope=([a-z][a-z0-9_.-]*\.v[1-9][0-9]*)$"
)
_STATUS_PATTERN = compile_regex(r"^[a-z][a-z_]*=[^\s=]+$")
_ACTIVE_CAPABILITY_EVIDENCE_PREFIX = "active:"
_CAPABILITY_PROVIDER_KINDS = frozenset(("local_model", "provider_adapter"))
_CAPABILITY_ROW_FIELD_VALUES = (
    "active_from_phase",
    "advertisement_rule",
    "attached_prerequisite",
    "durable_prerequisite",
    "evidence",
    "fallback",
    "id",
    "kind",
    "path",
    "production_advertised",
    "public_failure_surface",
    "snapshot_resumability",
)
_CAPABILITY_ROW_FIELDS = frozenset(_CAPABILITY_ROW_FIELD_VALUES)
_NATIVE_PRODUCTION_PROVIDER_ID = "provider-openai"
_FAILURE_MATCH_FIELDS = (
    "condition_id transition_from transition_to public_result_id status_key "
    "status_value provider_call_count domain_side_effect_count"
).split()
_FAILURE_EVIDENCE_FIELD_VALUES = (
    *_FAILURE_MATCH_FIELDS,
    "surface_id",
    "public_result",
)
_FAILURE_EVIDENCE_FIELDS = frozenset(_FAILURE_EVIDENCE_FIELD_VALUES)
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
    ("INPUT-F-03", "cli-agent-run-piped-with-tty"): "pending->unavailable",
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

    def postgresql_nodes(
        self,
        through_phase: int,
    ) -> tuple[AcceptanceNode, ...]:
        """Return selected nodes that require the real PostgreSQL harness."""
        return tuple(
            node
            for node in self.active_nodes(through_phase)
            if node.node_id.startswith(_POSTGRESQL_ACCEPTANCE_PREFIX)
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
    evidence_claim: tuple[str | int, ...]
    negative_e2e_node: str


@dataclass(frozen=True, kw_only=True, slots=True)
class NonApplicabilityRule:
    """Store one explicit fail-closed non-applicability rectangle."""

    condition_ids: tuple[str, ...]
    surface_ids: tuple[str, ...]
    active_from_phase: int
    reason: str
    owner: str
    evidence: str


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureMatrix:
    """Store compact failure surfaces, conditions, and applicability."""

    surfaces: tuple[FailureSurface, ...]
    conditions: tuple[FailureCondition, ...]
    rules: tuple[ApplicabilityRule, ...]
    non_applicability_rules: tuple[NonApplicabilityRule, ...] = ()

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

    def non_applicable_cells(self) -> frozenset[tuple[str, str]]:
        """Derive every explicitly non-applicable failure cell."""
        return frozenset(
            (condition_id, surface_id)
            for rule in self.non_applicability_rules
            for condition_id in rule.condition_ids
            for surface_id in rule.surface_ids
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
    has_planned_nodes = any(node.lifecycle == "planned" for node in nodes)
    if has_planned_nodes != (current_phase < _MAX_PHASE):
        raise AcceptanceVerificationError(
            "planned-node inventory must match remaining future phases"
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
            "non_applicability_rules",
            "matrix_sha256",
        },
        "failure matrix",
    )
    _header(payload, "failure matrix", schema_version=3)
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
    non_applicability_rules = tuple(
        _non_applicability_rule(raw, surface_by_id, condition_by_id)
        for raw in _list(
            payload.get("non_applicability_rules"),
            "non-applicability rules",
        )
    )
    if not non_applicability_rules:
        raise AcceptanceVerificationError(
            "non-applicability rules must be non-empty"
        )
    cells = [
        (rule.condition_id, surface_id)
        for rule in rules
        for surface_id in rule.surface_ids
    ]
    _unique(cells, "applicable failure cell")
    non_applicable_cells = [
        (condition_id, surface_id)
        for rule in non_applicability_rules
        for condition_id in rule.condition_ids
        for surface_id in rule.surface_ids
    ]
    _unique(non_applicable_cells, "non-applicable failure cell")
    matrix = FailureMatrix(
        surfaces=surfaces,
        conditions=conditions,
        rules=rules,
        non_applicability_rules=non_applicability_rules,
    )
    applicable = matrix.applicable_cells()
    non_applicable = matrix.non_applicable_cells()
    overlap = applicable & non_applicable
    if overlap:
        raise AcceptanceVerificationError(
            "applicable and non-applicable failure cells overlap: "
            f"{sorted(overlap)[:3]}"
        )
    unexplained = matrix.all_cells() - applicable - non_applicable
    if unexplained:
        raise AcceptanceVerificationError(
            f"failure matrix has unexplained cells: {sorted(unexplained)[:3]}"
        )
    if applicable | non_applicable != matrix.all_cells():
        raise AcceptanceVerificationError(
            "failure matrix contains cells outside its declared inventory"
        )
    applicable_conditions = {condition_id for condition_id, _ in applicable}
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
    current_phase = (
        manifest.current_phase if manifest is not None else _CURRENT_PHASE
    )
    if any(
        rule.active_from_phase <= current_phase
        and rule.evidence.casefold().startswith("planned:")
        for rule in non_applicability_rules
    ):
        raise AcceptanceVerificationError(
            "active non-applicability evidence cannot be planned"
        )
    for non_applicability_rule in non_applicability_rules:
        _validate_non_applicability_evidence(non_applicability_rule.evidence)
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
    if manifest.postgresql_nodes(through_phase):
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
    if activation.get("production_default") != "capability_gated":
        raise AcceptanceVerificationError(
            "structured input production activation must remain "
            "capability-gated"
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
    _validate_capability_rows(capability, frozenset(surfaces))
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


def _validate_capability_rows(
    capability: dict[str, object],
    public_failure_surfaces: frozenset[str],
) -> None:
    """Validate production activation against exact consumer evidence."""
    row_ids: list[str] = []
    for raw in _list(capability.get("rows"), "capability rows"):
        row = _mapping(raw, "capability row")
        kind = _nonempty_string(row.get("kind"), "capability row kind")
        provider_surface = kind in _CAPABILITY_PROVIDER_KINDS
        expected_fields = set(_CAPABILITY_ROW_FIELDS)
        if not provider_surface:
            expected_fields.add("interaction_mode")
        _exact_keys(row, expected_fields, "capability row")

        row_id = _nonempty_string(row.get("id"), "capability row ID")
        row_ids.append(row_id)
        for field in (
            "advertisement_rule",
            "attached_prerequisite",
            "durable_prerequisite",
            "evidence",
            "fallback",
            "path",
            "snapshot_resumability",
        ):
            _nonempty_string(row.get(field), f"capability row {field}")
        if not provider_surface:
            _nonempty_string(
                row.get("interaction_mode"),
                "capability row interaction mode",
            )
        _phase(row.get("active_from_phase"), "capability row phase")

        advertised = row.get("production_advertised")
        if type(advertised) is not bool:
            raise AcceptanceVerificationError(
                "capability production advertisement must be boolean"
            )
        public_surface = row.get("public_failure_surface")
        if provider_surface:
            if public_surface is not None:
                raise AcceptanceVerificationError(
                    "provider capability rows cannot claim a public surface"
                )
        elif (
            not isinstance(public_surface, str)
            or public_surface not in public_failure_surfaces
        ):
            raise AcceptanceVerificationError(
                "consumer capability row has an unknown public surface"
            )

        rule = cast(str, row["advertisement_rule"])
        gated = rule.startswith("advertise only when ")
        never = rule.startswith("never advertise")
        if gated == never:
            raise AcceptanceVerificationError(
                "capability advertisement rule must be gated or never"
            )
        evidence = cast(str, row["evidence"])
        if provider_surface and not gated:
            raise AcceptanceVerificationError(
                "provider capability rows must retain prerequisite gating"
            )
        if advertised:
            if not gated:
                raise AcceptanceVerificationError(
                    "advertised capability row must retain prerequisite gating"
                )
            if provider_surface and row_id != _NATIVE_PRODUCTION_PROVIDER_ID:
                raise AcceptanceVerificationError(
                    "unsupported provider or local capability cannot activate"
                )
            if not evidence.startswith(_ACTIVE_CAPABILITY_EVIDENCE_PREFIX):
                raise AcceptanceVerificationError(
                    "enabled capability evidence must use the active prefix"
                )
            _validate_evidence_references(
                evidence.removeprefix(_ACTIVE_CAPABILITY_EVIDENCE_PREFIX),
                "enabled capability",
            )
        elif provider_surface:
            if row_id == _NATIVE_PRODUCTION_PROVIDER_ID:
                raise AcceptanceVerificationError(
                    "native OpenAI capability must remain production enabled"
                )
        elif gated:
            raise AcceptanceVerificationError(
                "capability-gated consumer must remain production enabled"
            )
        elif not never:
            raise AcceptanceVerificationError(
                "unadvertised consumer must explicitly never advertise"
            )
        if not advertised and not evidence.startswith("planned:"):
            raise AcceptanceVerificationError(
                "unadvertised capability evidence must remain planned"
            )
    _unique(row_ids, "capability row ID")


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
        "command": (
            "make test-pgsql-exact no-install "
            f"INPUT_PHASE={manifest.current_phase}"
        ),
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
            "pytest collection failed:"
            f"\nstdout:\n{collection.stdout[-4000:]}"
            f"\nstderr:\n{collection.stderr[-4000:]}"
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
                "-s",
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
        testcases = tuple(
            testcase for suite in suites for testcase in suite.iter("testcase")
        )
        executed = tuple(map(_junit_testcase_id, testcases))
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
        matrix_path = root / "tests/fixtures/input/failure_matrix.json"
        decisions_path = root / "tests/fixtures/input/contract_decisions.json"
        if matrix_path.is_file() and decisions_path.is_file():
            _verify_failure_matrix_evidence(
                testcases,
                load_failure_matrix(matrix_path),
                _failure_envelope_schemas(decisions_path),
            )
    return collected


def _failure_envelope_schemas(
    path: Path,
) -> dict[str, dict[str, object]]:
    """Return frozen public failure envelope schemas by identifier."""
    payload = _strict_mapping(path, "contract decisions")
    error_status = _mapping(payload.get("error_status"), "error status")
    catalog = _mapping(
        error_status.get("public_envelope_catalog"),
        "public envelope catalog",
    )
    return {
        envelope_id: _mapping(schema, f"public envelope {envelope_id}")
        for envelope_id, schema in catalog.items()
    }


def _verify_failure_matrix_evidence(
    testcases: tuple[Element, ...],
    matrix: FailureMatrix,
    schemas: dict[str, dict[str, object]],
) -> None:
    """Reject missing, aliased, duplicated, or inaccurate matrix evidence."""
    expected_by_node: dict[str, dict[tuple[str, str], ApplicabilityRule]] = {}
    for rule in matrix.rules:
        expected = expected_by_node.setdefault(rule.negative_e2e_node, {})
        for surface_id in rule.surface_ids:
            expected[(rule.condition_id, surface_id)] = rule
    observed_by_node: dict[str, set[tuple[str, str]]] = {}
    for testcase in testcases:
        node_id = _junit_testcase_id(testcase)
        base_node = node_id.split("[", 1)[0]
        properties = tuple(
            property_element
            for property_element in testcase.findall("./properties/property")
            if property_element.attrib.get("name") == "failure_matrix_evidence"
        )
        expected_rules = expected_by_node.get(base_node)
        if expected_rules is None:
            if properties:
                raise AcceptanceVerificationError(
                    f"unassigned failure evidence: {node_id}"
                )
            continue
        if len(properties) != 1:
            raise AcceptanceVerificationError(
                f"failure node needs one evidence property: {node_id}"
            )
        raw_value = properties[0].attrib.get("value")
        try:
            raw_observations = loads(
                _nonempty_string(raw_value, "failure evidence property")
            )
        except JSONDecodeError as exc:
            raise AcceptanceVerificationError(
                f"failure evidence is not strict JSON: {node_id}"
            ) from exc
        observations = _list(raw_observations, "failure evidence")
        if len(observations) != 1:
            raise AcceptanceVerificationError(
                f"failure evidence must own one surface: {node_id}"
            )
        observation = _mapping(observations[0], "failure observation")
        _exact_keys(
            observation, _FAILURE_EVIDENCE_FIELDS, "failure observation"
        )
        for field in _FAILURE_EVIDENCE_FIELDS - {
            "public_result",
            "provider_call_count",
            "domain_side_effect_count",
        }:
            _nonempty_string(observation.get(field), f"failure {field}")
        public_result_id = cast(str, observation["public_result_id"])
        schema = schemas.get(public_result_id)
        if schema is None or not _draft_validator()(schema).is_valid(
            _mapping(observation.get("public_result"), "failure public result")
        ):
            raise AcceptanceVerificationError(
                f"failure result violates frozen schema: {public_result_id}"
            )
        for field in ("provider_call_count", "domain_side_effect_count"):
            _nonnegative_int(observation.get(field), f"failure {field}")
        surface_id = cast(str, observation["surface_id"])
        instance_ids = (
            surface_id,
            f"{observation['condition_id']}|{surface_id}",
        )
        if not node_id.endswith(tuple(f"[{value}]" for value in instance_ids)):
            raise AcceptanceVerificationError(
                f"failure evidence surface differs from instance: {node_id}"
            )
        key = (cast(str, observation["condition_id"]), surface_id)
        matched_rule = expected_rules.get(key)
        values = (
            matched_rule.evidence_claim if matched_rule is not None else ()
        )
        if (
            tuple(observation[name] for name in _FAILURE_MATCH_FIELDS)
            != values
        ):
            raise AcceptanceVerificationError(
                f"failure evidence differs from rule: {key}"
            )
        observed = observed_by_node.setdefault(base_node, set())
        if key in observed:
            raise AcceptanceVerificationError(
                f"duplicate dynamic failure evidence: {key}"
            )
        observed.add(key)
    for node_id, observed in observed_by_node.items():
        if observed != set(expected_by_node[node_id]):
            raise AcceptanceVerificationError(
                f"failure evidence cells differ from rules: {node_id}"
            )


def _junit_testcase_id(testcase: Element) -> str:
    """Return one exact pytest instance ID from legacy JUnit evidence."""
    try:
        return junit_testcase_id(testcase)
    except ContractGateError as exc:
        raise AcceptanceVerificationError(str(exc)) from exc


def _pytest(
    root: Path,
    arguments: tuple[str, ...],
    *,
    timeout: int,
) -> CompletedProcess[str]:
    return run_pytest(
        root,
        arguments,
        timeout=timeout,
        inherited_names=(POSTGRESQL_TEST_DSN_ENV,),
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
    public_result = _nonempty_string(
        item.get("public_result"), "failure public_result"
    )
    public_result_match = _PUBLIC_RESULT_PATTERN.fullmatch(public_result)
    if public_result_match is None:
        raise AcceptanceVerificationError(
            "failure public_result must name one envelope"
        )
    status_or_exit = _nonempty_string(
        item.get("status_or_exit"), "status_or_exit"
    )
    if _STATUS_PATTERN.fullmatch(status_or_exit) is None:
        raise AcceptanceVerificationError(
            "failure status_or_exit must be one machine literal"
        )
    status_key, status_value = status_or_exit.split("=", 1)
    provider_call_count = _nonnegative_int(
        item.get("provider_call_count"), "provider call count"
    )
    domain_side_effect_count = _nonnegative_int(
        item.get("domain_side_effect_count"), "domain side-effect count"
    )
    return ApplicabilityRule(
        condition_id=condition_id,
        surface_ids=surface_ids,
        active_from_phase=phase,
        evidence_claim=(
            condition_id,
            *expected_transition.split("->", 1),
            public_result_match.group(1),
            status_key,
            status_value,
            provider_call_count,
            domain_side_effect_count,
        ),
        negative_e2e_node=_test_node(item.get("negative_e2e_node")),
    )


def _non_applicability_rule(
    raw: object,
    surfaces: dict[str, FailureSurface],
    conditions: dict[str, FailureCondition],
) -> NonApplicabilityRule:
    item = _mapping(raw, "non-applicability rule")
    _exact_keys(
        item,
        {
            "condition_ids",
            "surface_ids",
            "reason",
            "owner",
            "evidence",
        },
        "non-applicability rule",
    )
    condition_ids = _string_list(
        item.get("condition_ids"), "non-applicable condition IDs"
    )
    surface_ids = _string_list(
        item.get("surface_ids"), "non-applicable surface IDs"
    )
    if not condition_ids or not surface_ids:
        raise AcceptanceVerificationError(
            "non-applicability rule must cover conditions and surfaces"
        )
    _unique(condition_ids, "non-applicable condition ID")
    _unique(surface_ids, "non-applicable surface ID")
    if any(condition_id not in conditions for condition_id in condition_ids):
        raise AcceptanceVerificationError(
            "non-applicability rule has an unknown condition"
        )
    if any(surface_id not in surfaces for surface_id in surface_ids):
        raise AcceptanceVerificationError(
            "non-applicability rule has an unknown surface"
        )
    active_from_phase = max(
        *(
            conditions[condition_id].active_from_phase
            for condition_id in condition_ids
        ),
        *(
            surfaces[surface_id].active_from_phase
            for surface_id in surface_ids
        ),
    )
    return NonApplicabilityRule(
        condition_ids=condition_ids,
        surface_ids=surface_ids,
        active_from_phase=active_from_phase,
        reason=_nonempty_string(
            item.get("reason"), "non-applicability reason"
        ),
        owner=_nonempty_string(item.get("owner"), "non-applicability owner"),
        evidence=_nonempty_string(
            item.get("evidence"), "non-applicability evidence"
        ),
    )


def _validate_non_applicability_evidence(evidence: str) -> None:
    """Require every non-applicability evidence reference to resolve."""
    _validate_evidence_references(evidence, "non-applicability")


def _validate_evidence_references(evidence: str, label: str) -> None:
    """Require every semicolon-delimited evidence reference to resolve."""
    references = tuple(reference.strip() for reference in evidence.split(";"))
    if not references or any(not reference for reference in references):
        raise AcceptanceVerificationError(
            f"{label} evidence has an empty reference"
        )
    root = repository_root()
    for reference in references:
        if "::" in reference:
            relative, symbol = reference.split("::", 1)
            separator = "::"
        elif "#" in reference:
            relative, symbol = reference.split("#", 1)
            separator = "#"
        else:
            relative, symbol, separator = reference, "", ""
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or "\\" in relative
            or ".." in path.parts
            or not relative
        ):
            raise AcceptanceVerificationError(
                f"invalid {label} evidence path: {relative}"
            )
        resolved = root.joinpath(*path.parts)
        if not resolved.is_file():
            raise AcceptanceVerificationError(
                f"{label} evidence path is missing: {relative}"
            )
        if not separator:
            continue
        if not symbol:
            raise AcceptanceVerificationError(
                f"{label} evidence symbol is empty: {reference}"
            )
        if separator == "::":
            if resolved.suffix != ".py" or not _python_symbol_exists(
                resolved, tuple(symbol.split("::"))
            ):
                raise AcceptanceVerificationError(
                    f"{label} evidence test is missing: {reference}"
                )
        elif resolved.suffix == ".json":
            payload = strict_json_path(resolved)
            if not _json_fragment_exists(payload, symbol):
                raise AcceptanceVerificationError(
                    f"{label} evidence JSON fragment is missing: {reference}"
                )
        elif resolved.suffix != ".py" or not _python_symbol_exists(
            resolved, tuple(symbol.split("."))
        ):
            raise AcceptanceVerificationError(
                f"{label} evidence symbol is missing: {reference}"
            )


def _python_symbol_exists(path: Path, parts: tuple[str, ...]) -> bool:
    """Return whether a dotted Python definition path exists."""
    if not parts or any(not part for part in parts):
        return False
    try:
        tree = parse_python(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return False
    children: tuple[AST, ...] = tuple(tree.body)
    for part in parts:
        match = next(
            (
                node
                for node in children
                if isinstance(
                    node,
                    (AsyncFunctionDef, ClassDef, FunctionDef),
                )
                and node.name == part
            ),
            None,
        )
        if match is None:
            return False
        children = tuple(match.body)
    return True


def _json_fragment_exists(value: object, fragment: str) -> bool:
    """Return whether a dot-delimited fragment resolves through JSON keys."""
    if not fragment or not isinstance(value, dict):
        return False
    mapping = cast(dict[str, object], value)
    if fragment in mapping:
        return True
    return any(
        fragment.startswith(f"{key}.")
        and _json_fragment_exists(
            child,
            fragment[len(key) + 1 :],
        )
        for key, child in mapping.items()
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
    if not environ.get(POSTGRESQL_TEST_DSN_ENV):
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
    expected: Iterable[str],
    label: str,
) -> None:
    expected_keys = set(expected)
    if set(value) != expected_keys:
        raise AcceptanceVerificationError(
            f"{label} has invalid keys: {sorted(set(value) ^ expected_keys)}"
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
