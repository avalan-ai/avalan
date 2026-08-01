#!/usr/bin/env python
"""Validate and execute conversation-continuity acceptance evidence."""

from argparse import ArgumentParser, Namespace
from ast import Constant, walk
from ast import parse as parse_python
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from re import compile as compile_regex
from sys import stderr
from tempfile import TemporaryDirectory

from contract_gate import (
    ContractGateError,
    StrictJsonError,
    canonical_sha256,
    execute_pytest_nodes,
    mapping,
    object_list,
    strict_json_path,
)
from verify_conversation_types import (
    ConversationTypeContractError,
    validate_type_source_phase_anchors,
)
from verify_conversation_types import (
    load_manifest as load_type_contract_manifest,
)


def _frozen(*values: str) -> frozenset[str]:
    return frozenset(values)


_FEATURE = "conversation_continuity"
_MARKDOWN_SUFFIX = "." + "m" + "d"
_MIN_PHASE = 0
_MAX_PHASE = 12
_NORMATIVE_OCCURRENCES = 144
_CATEGORIES = _frozen(
    "positive",
    "negative",
    "race",
    "security",
    "persistence",
    "wire",
    "integration",
    "public_e2e",
)
_DIMENSIONS = {
    "provider": _frozen(
        "native_openai",
        "native_azure",
        "incapable_generic_compatible",
    ),
    "provider_mode": _frozen(
        "off",
        "stateless_encrypted_replay",
        "provider_stored_chain",
    ),
    "local_retention": _frozen(
        "direct_process_local",
        "durable_local",
        "served_store_false",
        "served_store_true",
    ),
    "transport": _frozen("streaming", "non_streaming"),
    "execution": _frozen(
        "no_tool",
        "one_tool",
        "multiple_tool_cycles",
        "structured_input_suspension",
        "multiple_agents_lanes",
    ),
    "turn_topology": _frozen(
        "first_turn",
        "ordinary_child",
        "explicit_branch",
        "named_head_conflict",
        "retry",
        "reset",
    ),
    "reasoning_context": _frozen(
        "auto_omitted",
        "current_turn",
        "all_turns",
        "unsupported",
    ),
    "compaction": _frozen(
        "none",
        "inline_no_boundary",
        "inline_boundary",
        "repeated_boundary",
        "standalone",
    ),
    "lifecycle": _frozen(
        "same_process",
        "fresh_process",
        "expiry",
        "deletion",
        "tombstone",
        "key_rotation",
    ),
    "failure": _frozen(
        "validation",
        "known_no_dispatch",
        "ambiguous_dispatch",
        "before_output",
        "after_visible_output",
        "malformed_item",
        "commit_failure",
        "publication_failure",
    ),
    "authority": _frozen(
        "correct_principal",
        "wrong_tenant",
        "wrong_principal",
        "wrong_agent",
        "no_authenticated_authority",
    ),
    "limit": _frozen(
        "item_count",
        "bytes",
        "depth",
        "branch_count",
        "concurrency",
        "envelope_size",
        "ttl",
    ),
}
_NODE_PATTERN = compile_regex(r"^tests/[A-Za-z0-9_./-]+\.py::[^\s]+$")
_REQUIREMENT_PATTERN = compile_regex(r"^CONV-N-[0-9]{3}$")
_ACTIVE_INTEGRATED_FIXTURES = (
    "contract_decisions.json",
    "deterministic_fixtures.json",
    "provider_contract.json",
    "provider_conformance.json",
)
_THREAT_IDS = _frozen(
    "opaque-state-disclosure",
    "envelope-theft",
    "confused-deputy",
    "cross-tenant-equality",
    "replay-and-rollback",
    "decompression-size-bomb",
    "orphaned-upstream-state",
    "deletion-race",
)
_EVIDENCE_CLASSES = _frozen(
    "audit",
    "contract",
    "database",
    "live",
    "matrix",
    "negative",
    "pre_dispatch_rejection",
    "public",
    "runtime",
    "security",
    "wire",
)
_PHASE0_NODE_PAYLOAD_SHA256 = (
    "0440d0f24548c5b9ddcead0ad6f4e238416f3e0dc6683414f1eeb16dd92d046b"
)
_PHASE0_REQUIREMENTS_SHA256 = (
    "596f3f62b99be967aa09bdb1f543447d8f7580dfea533ddcaa3aaaa95e2994fe"
)
_PHASE0_FAILURE_STRUCTURE_SHA256 = (
    "773df38b02279bec169e6dc6f71e801ea78457a700c2de7ecf1f51912b4e5e80"
)
_PHASE0_THREAT_STRUCTURE_SHA256 = (
    "7d3e7470e5d978da1c5bfaba2c734c15de169f97045f33188633abc77266f239"
)
_PHASE0_PROVIDER_CANONICAL_SHA256 = (
    "f479bc544e1c3c41033cc5bc719428647f02552277f38912a99a85ec1c27c15f"
)
_PHASE0_PROVIDER_SOURCE_SHA256 = (
    "47d250ded5a4e0006fe3116ed51b9552f3a2b1caa313c73d77581e09e9ee5a0d"
)
_PHASE0_PROVIDER_BYTE_ANCHORS = {
    "tests/fixtures/conversation/provider_contract.json": (
        34_882,
        "7c97b7eaf359d91523828f93a5e5bea8475eb5f08c1db7616fd19c6512a08b61",
    ),
    "tests/model/nlp/vendor_openai_conversation_phase0_test.py": (
        96_247,
        "953066734fc2c292c26e1fa78b0a2f2ec26ad96035e01f3f0522493e94079ce8",
    ),
    "src/avalan/model/nlp/text/vendor/openai.py": (
        336_124,
        _PHASE0_PROVIDER_SOURCE_SHA256,
    ),
}
_PHASE0_ACTIVE_SOURCE_SHA256 = {
    "tests/conversation_contract_gate_test.py": (
        "c014a962f1e0384370bc70113acc7189de48bbd0e7ecba54c041054eee4de349"
    ),
    "tests/conversation_phase0_contract_test.py": (
        "1b3c5fd038c7e8436a42e6456afbc7189465cc80be9ea06c6a4e5f23334a10fc"
    ),
    "tests/conversation_response_dormancy_test.py": (
        "a528fbcff6e706a83bb5560deb14f23c7fc27c56cd4145796b99ac485e396458"
    ),
}
_ACTIVE_SOURCE_SHA256_BY_PHASE = {
    0: _PHASE0_ACTIVE_SOURCE_SHA256,
    1: {
        "tests/conversation/domain_contract_test.py": (
            "e74903247c01622bd420fa4e3d444b9990977957161f40d0d25a36bc64deb310"
        )
    },
}
_NODE_PAYLOAD_SHA256_BY_PHASE = {
    0: _PHASE0_NODE_PAYLOAD_SHA256,
    1: "9a85447f5de838051a3801b66eccd865ecc62b6e72ecfd9d3084603468ff8663",
}
_ACTIVATION_HISTORY_BY_PHASE = {
    0: "b8385b1c2ee8c56e7118ccd6c27a25d746974378808e92699953e5c846567f74",
    1: "cc98a83a046019ac7bb1f2c16469cc3a67fa6408885e87ff1fb6b265c6aa6161",
}
_REPLACEMENT_HISTORY_BY_PHASE = {
    0: (
        0,
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    ),
    1: (
        1,
        "c8982b4da6b6603a382d3319688e73d9a495ecee29d2301b2c4962cdb62b1e8b",
    ),
}
_FAILURE_STRUCTURE_BY_PHASE = {
    0: (11, 9, 99, _PHASE0_FAILURE_STRUCTURE_SHA256),
    1: (11, 9, 99, _PHASE0_FAILURE_STRUCTURE_SHA256),
}
_THREAT_STRUCTURE_BY_PHASE = {
    0: (5, 5, 8, _PHASE0_THREAT_STRUCTURE_SHA256),
    1: (5, 5, 8, _PHASE0_THREAT_STRUCTURE_SHA256),
}
_PHASE0_NODE_INVENTORY = (
    (
        "phase0-positive-fixtures",
        "tests/conversation_phase0_contract_test.py::test_phase0_contract_fixtures_are_frozen",
        0,
        "contract",
    ),
    (
        "phase0-negative-dormancy",
        "tests/conversation_phase0_contract_test.py::test_all_production_capabilities_remain_dormant",
        0,
        "negative",
    ),
    (
        "phase0-race-sealed-inventory",
        "tests/conversation_contract_gate_test.py::test_sealed_inventory_rejects_mid_run_mutation",
        0,
        "runtime",
    ),
    (
        "phase0-security-threats",
        "tests/conversation_phase0_contract_test.py::test_phase0_threat_controls_are_complete",
        0,
        "security",
    ),
    (
        "phase0-persistence-state-table",
        "tests/conversation_phase0_contract_test.py::test_contract_state_tables_are_total",
        0,
        "contract",
    ),
    (
        "phase0-wire-provider-evidence",
        "tests/conversation_phase0_contract_test.py::test_provider_contract_evidence_is_typed_and_dormant",
        0,
        "wire",
    ),
    (
        "phase0-public-fail-closed",
        "tests/conversation_response_dormancy_test.py::test_responses_reject_dormant_conversation_fields_before_dispatch",
        0,
        "pre_dispatch_rejection",
    ),
    (
        "phase0-one-shot-regression",
        "tests/conversation_phase0_contract_test.py::test_one_shot_behavior_omits_conversation_state",
        0,
        "runtime",
    ),
    (
        "phase0-source-isolation",
        "tests/conversation_phase0_contract_test.py::test_tracked_gate_sources_do_not_depend_on_ignored_material",
        0,
        "audit",
    ),
)
_PHASE0_FAILURE_BOUNDARIES = (
    ("validation_before_dispatch", 11),
    ("provider_rejection", 11),
    ("known_no_dispatch", 11),
    ("ambiguous_dispatch", 11),
    ("before_visible_output", 11),
    ("after_visible_output", 11),
    ("malformed_stream_item", 11),
    ("tool_effect", 8),
    ("structured_input_suspension", 8),
    ("checkpoint_commit_failure", 11),
    ("outward_publication_failure", 11),
)
_PHASE0_FAILURE_SURFACES = (
    ("direct_sdk", 4),
    ("provider_adapter", 5),
    ("agent_sdk", 8),
    ("served_responses", 9),
    ("compact", 7),
    ("retrieve", 9),
    ("delete", 9),
    ("stream", 5),
    ("structured_input", 8),
)
_FAILURE_COUNT_VALUES = frozenset((0, 1))
_PUBLIC_MAPPING_VALUES = _frozen(
    "absent",
    "committed_unpublished",
    "input_required",
    "not_applicable",
)
_RETRY_DECISION_VALUES = _frozen(
    "bounded_if_proven_safe",
    "fenced",
    "never",
    "not_applicable",
    "reconcile_only",
    "resume_only",
)
_PARENT_STATE_VALUES = _frozen("not_applicable", "unchanged")
_PUBLIC_ERROR_VALUES = _frozen(
    "conversation_dispatch_ambiguous",
    "conversation_effect_boundary",
    "conversation_failed_after_output",
    "conversation_input_required",
    "conversation_provider_failed",
    "conversation_provider_rejected",
    "conversation_publication_failed",
    "conversation_state_commit_failed",
    "conversation_stream_item_invalid",
    "conversation_transport_no_dispatch",
    "conversation_validation_failed",
    "not_applicable",
)
_RECONCILIATION_STATE_VALUES = _frozen(
    "none",
    "not_applicable",
    "pending",
    "quarantined",
    "required",
    "suspended",
)


class ConversationAcceptanceError(RuntimeError):
    """Report invalid or non-passing conversation evidence."""


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceNode:
    """Store one lifecycle-aware acceptance node."""

    id: str
    category: str
    lifecycle: str
    active_from_phase: int
    requirement_ids: tuple[str, ...]
    node_id: str
    surface: str
    dimensions: dict[str, tuple[str, ...]]
    evidence_class: str


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceReplacement:
    """Store one reviewed append-only acceptance evidence replacement."""

    phase: int
    old_node_id: str
    replacement_node_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceManifest:
    """Store the validated conversation acceptance inventory."""

    path: Path
    current_phase: int
    nodes: tuple[AcceptanceNode, ...]
    replacements: tuple[AcceptanceReplacement, ...]

    def active_nodes(self, through_phase: int) -> tuple[AcceptanceNode, ...]:
        """Return active nodes introduced through one phase."""
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle == "active"
            and node.active_from_phase <= through_phase
        )

    def planned_nodes(self) -> tuple[AcceptanceNode, ...]:
        """Return all future planned nodes."""
        return tuple(
            node for node in self.nodes if node.lifecycle == "planned"
        )

    def ever_activated_nodes(
        self,
        through_phase: int,
    ) -> tuple[AcceptanceNode, ...]:
        """Return retained active and replaced records through one phase."""
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle in {"active", "replaced"}
            and node.active_from_phase <= through_phase
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class Requirement:
    """Store one ordinal normative requirement."""

    id: str
    normative_ordinal: int
    source_section: str
    normative_level: str
    paraphrase: str
    owner_phase: int
    production_artifact: str
    test_node_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureBoundary:
    """Store one failure boundary and its owning requirements."""

    id: str
    owner_phase: int
    requirement_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureSurface:
    """Store one public or runtime failure surface."""

    id: str
    owner_phase: int


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureCell:
    """Store one explicit failure-boundary and surface intersection."""

    id: str
    boundary_id: str
    surface_id: str
    applicability: str
    lifecycle: str
    active_from_phase: int
    evidence_node_id: str


@dataclass(frozen=True, kw_only=True, slots=True)
class FailureMatrix:
    """Store the complete failure-boundary Cartesian matrix."""

    boundaries: tuple[FailureBoundary, ...]
    surfaces: tuple[FailureSurface, ...]
    cells: tuple[FailureCell, ...]


def repository_root() -> Path:
    """Return the repository root containing this script."""
    return Path(__file__).resolve().parents[1]


def fixture_root() -> Path:
    """Return the tracked conversation fixture directory."""
    return repository_root() / "tests" / "fixtures" / "conversation"


def default_manifest_path() -> Path:
    """Return the tracked acceptance manifest path."""
    return fixture_root() / "acceptance_manifest.phase1.json"


def companion_fixture_path(manifest_path: Path, stem: str) -> Path:
    """Return a phase-qualified companion beside an acceptance manifest."""
    name = manifest_path.name
    prefix = "acceptance_manifest"
    qualifier = ""
    if name.startswith(prefix) and name.endswith(".json"):
        qualifier = name[len(prefix) : -len(".json")]
    return manifest_path.parent / f"{stem}{qualifier}.json"


def load_manifest(path: Path) -> AcceptanceManifest:
    """Load and validate the lifecycle-aware acceptance manifest."""
    payload = _strict_mapping(path, "acceptance manifest")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "categories",
            "required_dimensions",
            "replacements",
            "activation_history",
            "nodes",
            "manifest_sha256",
        },
        "acceptance manifest",
    )
    _header(payload, "acceptance manifest")
    current_phase = _phase(payload.get("current_phase"), "current_phase")
    categories = _string_list(payload.get("categories"), "categories")
    if frozenset(categories) != _CATEGORIES or len(categories) != len(
        _CATEGORIES
    ):
        raise ConversationAcceptanceError(
            "acceptance categories differ from the required inventory"
        )
    required_dimensions = _required_dimensions(
        payload.get("required_dimensions")
    )
    raw_nodes = object_list(payload.get("nodes"), "acceptance nodes")
    if not raw_nodes:
        raise ConversationAcceptanceError("acceptance nodes must be non-empty")
    nodes = tuple(_acceptance_node(raw, current_phase) for raw in raw_nodes)
    _unique((node.id for node in nodes), "acceptance node ID")
    _unique((node.node_id for node in nodes), "pytest node ID")
    if frozenset(node.category for node in nodes) != _CATEGORIES:
        raise ConversationAcceptanceError(
            "every acceptance category must own a node"
        )
    active_categories = frozenset(
        node.category for node in nodes if node.lifecycle == "active"
    )
    if active_categories != _CATEGORIES:
        raise ConversationAcceptanceError(
            "every acceptance category must have active Phase 0 evidence"
        )
    if current_phase < _MAX_PHASE and not any(
        node.lifecycle == "planned" for node in nodes
    ):
        raise ConversationAcceptanceError(
            "future acceptance nodes must remain explicitly planned"
        )
    for phase in range(current_phase + 1):
        if not any(
            node.lifecycle in {"active", "replaced"}
            and node.active_from_phase == phase
            for node in nodes
        ):
            raise ConversationAcceptanceError(
                f"implemented acceptance inventory has a gap at phase {phase}"
            )
    replacements = _validate_replacements(
        payload.get("replacements"), nodes, current_phase
    )
    activation_history = _validate_activation_history(
        payload.get("activation_history"), nodes, current_phase
    )
    _validate_replacement_transitions(
        replacements,
        nodes,
        activation_history,
    )
    observed_dimensions = {
        name: frozenset(
            value for node in nodes for value in node.dimensions[name]
        )
        for name in _DIMENSIONS
    }
    if observed_dimensions != required_dimensions:
        raise ConversationAcceptanceError(
            "acceptance nodes do not cover every mandatory dimension"
        )
    active_dimensions = {
        name: frozenset(
            value
            for node in nodes
            if node.lifecycle == "active"
            for value in node.dimensions[name]
        )
        for name in _DIMENSIONS
    }
    if active_dimensions != required_dimensions:
        raise ConversationAcceptanceError(
            "active evidence lacks an explicit disposition for mandatory "
            "dimensions"
        )
    if not any(
        node.lifecycle == "active"
        and node.evidence_class == "pre_dispatch_rejection"
        and node.surface == "served_responses"
        for node in nodes
    ):
        raise ConversationAcceptanceError(
            "active served dimensions require executable pre-dispatch "
            "rejection evidence"
        )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if payload.get("manifest_sha256") != canonical_sha256(canonical):
        raise ConversationAcceptanceError(
            "acceptance manifest digest is invalid"
        )
    observed_phase0_nodes = tuple(
        (
            node.id,
            node.node_id,
            node.active_from_phase,
            node.evidence_class,
        )
        for node in nodes
        if node.active_from_phase == 0
    )
    if observed_phase0_nodes != _PHASE0_NODE_INVENTORY:
        raise ConversationAcceptanceError(
            "Phase 0 acceptance node tuple inventory drifted"
        )
    _validate_node_phase_anchors(raw_nodes, nodes, current_phase)
    return AcceptanceManifest(
        path=path,
        current_phase=current_phase,
        nodes=nodes,
        replacements=replacements,
    )


def load_requirements(
    path: Path,
    manifest: AcceptanceManifest,
    *,
    repo_root: Path,
) -> tuple[Requirement, ...]:
    """Load and validate all ordinal normative requirements."""
    payload = _strict_mapping(path, "requirements traceability")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "source_sections",
            "normative_occurrence_count",
            "requirements",
            "catalog_sha256",
        },
        "requirements traceability",
    )
    _header(payload, "requirements traceability")
    sections = _string_list(payload.get("source_sections"), "source sections")
    if sections != tuple(str(section) for section in range(9, 27)):
        raise ConversationAcceptanceError(
            "source sections must be the contiguous 9 through 26 inventory"
        )
    if payload.get("normative_occurrence_count") != _NORMATIVE_OCCURRENCES:
        raise ConversationAcceptanceError(
            "normative occurrence count must be exactly 144"
        )
    raw_requirements = object_list(payload.get("requirements"), "requirements")
    if len(raw_requirements) != _NORMATIVE_OCCURRENCES:
        raise ConversationAcceptanceError(
            "requirement catalog does not contain every normative occurrence"
        )
    requirements = tuple(
        _requirement(raw, repo_root=repo_root) for raw in raw_requirements
    )
    expected_ids = tuple(
        f"CONV-N-{ordinal:03d}"
        for ordinal in range(1, _NORMATIVE_OCCURRENCES + 1)
    )
    if tuple(requirement.id for requirement in requirements) != expected_ids:
        raise ConversationAcceptanceError(
            "requirement IDs must be stable and ordinal"
        )
    if tuple(
        requirement.normative_ordinal for requirement in requirements
    ) != tuple(range(1, _NORMATIVE_OCCURRENCES + 1)):
        raise ConversationAcceptanceError(
            "normative occurrence ordinals must be contiguous"
        )
    node_by_id = {node.node_id: node for node in manifest.nodes}
    reverse: dict[str, set[str]] = {}
    for node in manifest.nodes:
        for requirement_id in node.requirement_ids:
            reverse.setdefault(requirement_id, set()).add(node.node_id)
    replacement_by_old = {
        replacement.old_node_id: replacement.replacement_node_ids
        for replacement in manifest.replacements
    }
    for requirement in requirements:
        for node_id in requirement.test_node_ids:
            owner_node = node_by_id.get(node_id)
            if owner_node is None:
                raise ConversationAcceptanceError(
                    f"requirement references an unknown node: {node_id}"
                )
            if owner_node.active_from_phase != requirement.owner_phase:
                raise ConversationAcceptanceError(
                    "requirement owner phase differs from its exact nodes: "
                    f"{requirement.id}"
                )
        allowed = _replacement_closure(
            requirement.test_node_ids,
            replacement_by_old,
            node_by_id,
            requirement.id,
        )
        if allowed != reverse.get(requirement.id, set()):
            raise ConversationAcceptanceError(
                "requirement ownership is outside its reviewed replacement "
                f"chain: {requirement.id}"
            )
        if requirement.owner_phase <= manifest.current_phase:
            leaves = allowed - set(replacement_by_old)
            if not leaves or any(
                node_by_id[node_id].lifecycle != "active" for node_id in leaves
            ):
                raise ConversationAcceptanceError(
                    "implemented requirement lacks active replacement-chain "
                    f"evidence: {requirement.id}"
                )
    if payload.get("catalog_sha256") != canonical_sha256(raw_requirements):
        raise ConversationAcceptanceError(
            "requirement catalog digest is invalid"
        )
    if canonical_sha256(raw_requirements) != _PHASE0_REQUIREMENTS_SHA256:
        raise ConversationAcceptanceError(
            "requirement catalog differs from the independent Phase 0 anchor"
        )
    return requirements


def _replacement_closure(
    roots: tuple[str, ...],
    replacement_by_old: dict[str, tuple[str, ...]],
    node_by_id: dict[str, AcceptanceNode],
    requirement_id: str,
) -> set[str]:
    """Return roots and reviewed descendants owning one requirement."""
    observed = set(roots)
    pending = list(roots)
    while pending:
        current = pending.pop()
        for target in replacement_by_old.get(current, ()):
            if (
                requirement_id in node_by_id[target].requirement_ids
                and target not in observed
            ):
                observed.add(target)
                pending.append(target)
    return observed


def load_failure_matrix(
    path: Path,
    *,
    manifest: AcceptanceManifest,
    requirement_ids: frozenset[str],
) -> FailureMatrix:
    """Load and validate the complete explicit failure matrix."""
    payload = _strict_mapping(path, "failure matrix")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "observation_window",
            "tool_effect_scope",
            "boundaries",
            "surfaces",
            "cells",
            "matrix_sha256",
        },
        "failure matrix",
    )
    _header(payload, "failure matrix")
    if payload.get("current_phase") != manifest.current_phase:
        raise ConversationAcceptanceError(
            "failure matrix and acceptance phases differ"
        )
    _nonempty_string(payload.get("observation_window"), "observation window")
    _nonempty_string(payload.get("tool_effect_scope"), "tool-effect scope")
    raw_boundaries = object_list(
        payload.get("boundaries"), "failure boundaries"
    )
    boundaries = tuple(
        _failure_boundary(raw, requirement_ids) for raw in raw_boundaries
    )
    raw_surfaces = object_list(payload.get("surfaces"), "failure surfaces")
    surfaces = tuple(_failure_surface(raw) for raw in raw_surfaces)
    if not boundaries or not surfaces:
        raise ConversationAcceptanceError(
            "failure boundaries and surfaces must be non-empty"
        )
    _unique((item.id for item in boundaries), "failure boundary ID")
    _unique((item.id for item in surfaces), "failure surface ID")
    phase0_boundary_count = len(_PHASE0_FAILURE_BOUNDARIES)
    if (
        tuple(
            (item.id, item.owner_phase)
            for item in boundaries[:phase0_boundary_count]
        )
        != _PHASE0_FAILURE_BOUNDARIES
    ):
        raise ConversationAcceptanceError(
            "failure boundary inventory differs from the Phase 0 anchor"
        )
    phase0_surface_count = len(_PHASE0_FAILURE_SURFACES)
    if (
        tuple(
            (item.id, item.owner_phase)
            for item in surfaces[:phase0_surface_count]
        )
        != _PHASE0_FAILURE_SURFACES
    ):
        raise ConversationAcceptanceError(
            "failure surface inventory differs from the Phase 0 anchor"
        )
    boundary_by_id = {item.id: item for item in boundaries}
    surface_by_id = {item.id: item for item in surfaces}
    node_by_id = {node.node_id: node for node in manifest.nodes}
    raw_cells = object_list(payload.get("cells"), "failure cells")
    cells = tuple(
        _failure_cell(
            raw,
            boundary_by_id=boundary_by_id,
            surface_by_id=surface_by_id,
            node_by_id=node_by_id,
            current_phase=manifest.current_phase,
        )
        for raw in raw_cells
    )
    _unique((cell.id for cell in cells), "failure cell ID")
    observed = {(cell.boundary_id, cell.surface_id) for cell in cells}
    expected = {
        (boundary.id, surface.id)
        for boundary in boundaries
        for surface in surfaces
    }
    if len(cells) != len(expected) or observed != expected:
        raise ConversationAcceptanceError(
            "failure matrix must cover the complete Cartesian inventory"
        )
    if not any(
        cell.applicability == "applicable" for cell in cells
    ) or not any(cell.applicability == "not_applicable" for cell in cells):
        raise ConversationAcceptanceError(
            "failure matrix needs applicable and explicit non-applicable cells"
        )
    canonical = {
        key: value for key, value in payload.items() if key != "matrix_sha256"
    }
    if payload.get("matrix_sha256") != canonical_sha256(canonical):
        raise ConversationAcceptanceError("failure matrix digest is invalid")
    _validate_failure_structure_anchors(
        payload,
        raw_boundaries,
        raw_surfaces,
        raw_cells,
        manifest.current_phase,
    )
    return FailureMatrix(
        boundaries=boundaries,
        surfaces=surfaces,
        cells=cells,
    )


def _validate_failure_structure_anchors(
    payload: dict[str, object],
    raw_boundaries: list[object],
    raw_surfaces: list[object],
    raw_cells: list[object],
    current_phase: int,
) -> None:
    """Validate append-only failure structure while ignoring lifecycle."""
    _require_phase_anchor_keys(
        _FAILURE_STRUCTURE_BY_PHASE,
        current_phase,
        "failure structure",
    )
    previous = (0, 0, 0)
    for phase in range(current_phase + 1):
        boundary_count, surface_count, cell_count, expected_sha256 = (
            _FAILURE_STRUCTURE_BY_PHASE[phase]
        )
        counts = (boundary_count, surface_count, cell_count)
        available = (
            len(raw_boundaries),
            len(raw_surfaces),
            len(raw_cells),
        )
        if any(
            before > after for before, after in zip(previous, counts)
        ) or any(count > maximum for count, maximum in zip(counts, available)):
            raise ConversationAcceptanceError(
                "failure structure phase anchors are not append-only"
            )
        normalized_cells = [
            {
                key: value
                for key, value in mapping(raw, "failure cell").items()
                if key != "lifecycle"
            }
            for raw in raw_cells[:cell_count]
        ]
        structure = {
            "observation_window": payload.get("observation_window"),
            "tool_effect_scope": payload.get("tool_effect_scope"),
            "boundaries": raw_boundaries[:boundary_count],
            "surfaces": raw_surfaces[:surface_count],
            "cells": normalized_cells,
        }
        if canonical_sha256(structure) != expected_sha256:
            raise ConversationAcceptanceError(
                "failure structure differs from its immutable phase "
                f"anchor at phase {phase}"
            )
        previous = counts
    if previous != (
        len(raw_boundaries),
        len(raw_surfaces),
        len(raw_cells),
    ):
        raise ConversationAcceptanceError(
            "failure structure has unanchored appended payload"
        )


def verify_acceptance(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
    execute: bool = True,
) -> AcceptanceManifest:
    """Validate all fixtures and execute selected active nodes."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    if not _MIN_PHASE <= through_phase <= manifest.current_phase:
        raise ConversationAcceptanceError(
            "through-phase must be implemented by the current manifest"
        )
    fixtures = path.parent
    requirements = load_requirements(
        fixtures / "requirements_traceability.json",
        manifest,
        repo_root=root,
    )
    requirement_ids = frozenset(item.id for item in requirements)
    load_failure_matrix(
        companion_fixture_path(path, "failure_matrix"),
        manifest=manifest,
        requirement_ids=requirement_ids,
    )
    _validate_threat_model(
        companion_fixture_path(path, "threat_model"),
        manifest=manifest,
        requirement_ids=requirement_ids,
    )
    _validate_integrated_fixtures(fixtures)
    _validate_type_manifest(
        fixtures,
        manifest.current_phase,
        root,
        acceptance_path=path,
    )
    verify_gate_source_isolation(root, manifest)
    nodes = manifest.active_nodes(through_phase)
    if not nodes:
        raise ConversationAcceptanceError(
            "the selected acceptance inventory has no active nodes"
        )
    if execute:
        with TemporaryDirectory(
            prefix="avalan-conversation-acceptance-"
        ) as temporary:
            try:
                execute_pytest_nodes(
                    root,
                    tuple(node.node_id for node in nodes),
                    junit_path=Path(temporary) / "pytest.xml",
                    expected_evidence={
                        node.node_id: node.evidence_class for node in nodes
                    },
                )
            except ContractGateError as exc:
                raise ConversationAcceptanceError(str(exc)) from exc
    return manifest


def verify_gate_source_isolation(
    root: Path,
    manifest: AcceptanceManifest,
) -> None:
    """Reject Markdown dependencies from tracked gate and active tests."""
    _require_phase_anchor_keys(
        _ACTIVE_SOURCE_SHA256_BY_PHASE,
        manifest.current_phase,
        "active source",
    )
    for phase in range(manifest.current_phase + 1):
        observed = {
            node.node_id.split("::", 1)[0]
            for node in manifest.nodes
            if node.lifecycle in {"active", "replaced"}
            and node.active_from_phase == phase
        }
        expected = _ACTIVE_SOURCE_SHA256_BY_PHASE[phase]
        if observed != set(expected):
            raise ConversationAcceptanceError(
                "active acceptance source inventory differs from its "
                f"phase anchor at phase {phase}"
            )
        for relative, expected_sha256 in expected.items():
            source = root / relative
            if not source.is_file():
                raise ConversationAcceptanceError(
                    f"active acceptance source is missing: {relative}"
                )
            if sha256(source.read_bytes()).hexdigest() != expected_sha256:
                raise ConversationAcceptanceError(
                    f"active acceptance source digest changed: {relative}"
                )
    candidates = {
        root
        / "scripts"
        / "contract_startup"
        / "avalan_contract_gate_plugin.py",
        root / "scripts" / "contract_startup" / "sitecustomize.py",
        root / "scripts" / "verify_conversation_acceptance.py",
        root / "scripts" / "verify_conversation_types.py",
        root / "scripts" / "run_conversation_contract_gate.py",
        root / "scripts" / "contract_gate.py",
        *(
            root / node.node_id.split("::", 1)[0]
            for node in manifest.nodes
            if node.lifecycle in {"active", "replaced"}
        ),
    }
    for path in candidates:
        if not path.is_file():
            raise ConversationAcceptanceError(
                f"tracked gate source is missing: {path.relative_to(root)}"
            )
        try:
            tree = parse_python(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise ConversationAcceptanceError(
                f"cannot audit tracked gate source: {path.relative_to(root)}"
            ) from exc
        markdown_literals = tuple(
            node.value
            for node in walk(tree)
            if isinstance(node, Constant)
            and isinstance(node.value, str)
            and node.value.casefold().endswith(_MARKDOWN_SUFFIX)
        )
        if markdown_literals:
            raise ConversationAcceptanceError(
                "tracked gate sources must not depend on Markdown inputs: "
                f"{path.relative_to(root)}"
            )


def _acceptance_node(raw: object, current_phase: int) -> AcceptanceNode:
    item = mapping(raw, "acceptance node")
    expected = {
        "id",
        "category",
        "lifecycle",
        "active_from_phase",
        "requirement_ids",
        "node_id",
        "surface",
        "provider",
        "provider_mode",
        "local_retention",
        "transport",
        "execution",
        "turn_topology",
        "reasoning_context",
        "compaction",
        "scenario_lifecycle",
        "failure",
        "authority",
        "limit",
        "evidence_class",
    }
    _exact_keys(item, expected, "acceptance node")
    category = _nonempty_string(item.get("category"), "node category")
    if category not in _CATEGORIES:
        raise ConversationAcceptanceError(f"invalid node category: {category}")
    phase = _phase(item.get("active_from_phase"), "active_from_phase")
    lifecycle = _nonempty_string(item.get("lifecycle"), "node lifecycle")
    if lifecycle not in {"active", "planned", "replaced"} or (
        (phase > current_phase) != (lifecycle == "planned")
    ):
        raise ConversationAcceptanceError(
            "node lifecycle disagrees with active_from_phase"
        )
    requirement_ids = _string_list(
        item.get("requirement_ids"), "node requirement IDs"
    )
    if not requirement_ids:
        raise ConversationAcceptanceError(
            "acceptance node must cover at least one requirement"
        )
    _unique(requirement_ids, "node requirement ID")
    for requirement_id in requirement_ids:
        if _REQUIREMENT_PATTERN.fullmatch(requirement_id) is None:
            raise ConversationAcceptanceError(
                f"invalid requirement ID: {requirement_id}"
            )
    dimensions = {
        name: _dimension_values(
            item.get("scenario_lifecycle" if name == "lifecycle" else name),
            name,
        )
        for name in _DIMENSIONS
    }
    evidence_class = _nonempty_string(
        item.get("evidence_class"), "evidence class"
    )
    if evidence_class not in _EVIDENCE_CLASSES:
        raise ConversationAcceptanceError(
            f"invalid evidence class: {evidence_class}"
        )
    return AcceptanceNode(
        id=_nonempty_string(item.get("id"), "node ID"),
        category=category,
        lifecycle=lifecycle,
        active_from_phase=phase,
        requirement_ids=requirement_ids,
        node_id=_test_node(item.get("node_id")),
        surface=_nonempty_string(item.get("surface"), "node surface"),
        dimensions=dimensions,
        evidence_class=evidence_class,
    )


def _required_dimensions(raw: object) -> dict[str, frozenset[str]]:
    item = mapping(raw, "required dimensions")
    _exact_keys(item, set(_DIMENSIONS), "required dimensions")
    observed = {
        name: frozenset(_string_list(item.get(name), f"{name} dimension"))
        for name in _DIMENSIONS
    }
    if observed != _DIMENSIONS:
        raise ConversationAcceptanceError(
            "mandatory dimension inventory changed"
        )
    return observed


def _dimension_values(raw: object, name: str) -> tuple[str, ...]:
    values = _string_list(raw, f"{name} dimension values")
    if not values or not set(values) <= _DIMENSIONS[name]:
        raise ConversationAcceptanceError(
            f"node has invalid or empty {name} dimensions"
        )
    _unique(values, f"node {name} dimension")
    return values


def _requirement(raw: object, *, repo_root: Path) -> Requirement:
    item = mapping(raw, "requirement")
    _exact_keys(
        item,
        {
            "id",
            "normative_ordinal",
            "source_section",
            "normative_level",
            "paraphrase",
            "owner_phase",
            "production_artifact",
            "test_node_ids",
        },
        "requirement",
    )
    identifier = _nonempty_string(item.get("id"), "requirement ID")
    ordinal = _positive_int(item.get("normative_ordinal"), "normative ordinal")
    if identifier != f"CONV-N-{ordinal:03d}":
        raise ConversationAcceptanceError("requirement ID and ordinal differ")
    section = _nonempty_string(item.get("source_section"), "source section")
    try:
        major = int(section.split(".", 1)[0])
    except ValueError as exc:
        raise ConversationAcceptanceError(
            f"invalid source section: {section}"
        ) from exc
    if not 9 <= major <= 26:
        raise ConversationAcceptanceError(f"invalid source section: {section}")
    level = _nonempty_string(item.get("normative_level"), "normative level")
    if level not in {"MUST", "MUST NOT"}:
        raise ConversationAcceptanceError(f"invalid normative level: {level}")
    paraphrase = _nonempty_string(
        item.get("paraphrase"), "requirement paraphrase"
    )
    phase = _phase(item.get("owner_phase"), "requirement owner phase")
    artifact = _relative_path(
        item.get("production_artifact"), "production artifact"
    )
    if phase == 0 and not (repo_root / artifact).is_file():
        raise ConversationAcceptanceError(
            f"active production artifact is missing: {artifact}"
        )
    nodes = _string_list(item.get("test_node_ids"), "requirement test nodes")
    if not nodes:
        raise ConversationAcceptanceError(
            f"requirement has no exact test nodes: {identifier}"
        )
    for node in nodes:
        _test_node(node)
    _unique(nodes, "requirement test node")
    return Requirement(
        id=identifier,
        normative_ordinal=ordinal,
        source_section=section,
        normative_level=level,
        paraphrase=paraphrase,
        owner_phase=phase,
        production_artifact=artifact,
        test_node_ids=nodes,
    )


def _failure_boundary(
    raw: object,
    requirement_ids: frozenset[str],
) -> FailureBoundary:
    item = mapping(raw, "failure boundary")
    _exact_keys(
        item,
        {"id", "description", "owner_phase", "requirement_ids"},
        "failure boundary",
    )
    _nonempty_string(item.get("description"), "boundary description")
    owned = _string_list(item.get("requirement_ids"), "boundary requirements")
    if not owned or not set(owned) <= requirement_ids:
        raise ConversationAcceptanceError(
            "failure boundary references unknown requirements"
        )
    return FailureBoundary(
        id=_nonempty_string(item.get("id"), "boundary ID"),
        owner_phase=_phase(item.get("owner_phase"), "boundary owner phase"),
        requirement_ids=owned,
    )


def _failure_surface(raw: object) -> FailureSurface:
    item = mapping(raw, "failure surface")
    _exact_keys(
        item,
        {"id", "description", "owner_phase"},
        "failure surface",
    )
    _nonempty_string(item.get("description"), "surface description")
    return FailureSurface(
        id=_nonempty_string(item.get("id"), "surface ID"),
        owner_phase=_phase(item.get("owner_phase"), "surface owner phase"),
    )


def _failure_cell(
    raw: object,
    *,
    boundary_by_id: dict[str, FailureBoundary],
    surface_by_id: dict[str, FailureSurface],
    node_by_id: dict[str, AcceptanceNode],
    current_phase: int,
) -> FailureCell:
    item = mapping(raw, "failure cell")
    _exact_keys(
        item,
        {
            "id",
            "boundary_id",
            "surface_id",
            "applicability",
            "lifecycle",
            "active_from_phase",
            "evidence_node_id",
            "expected_dispatch_count",
            "visible_output_count",
            "tool_effect_count",
            "checkpoint_commit_count",
            "public_mapping",
            "retry_decision",
            "parent_state",
            "public_error",
            "reconciliation_state",
            "rationale",
        },
        "failure cell",
    )
    boundary_id = _nonempty_string(item.get("boundary_id"), "boundary ID")
    surface_id = _nonempty_string(item.get("surface_id"), "surface ID")
    boundary = boundary_by_id.get(boundary_id)
    surface = surface_by_id.get(surface_id)
    if boundary is None or surface is None:
        raise ConversationAcceptanceError(
            "failure cell references an unknown boundary or surface"
        )
    identifier = _nonempty_string(item.get("id"), "failure cell ID")
    if identifier != f"{boundary_id}--{surface_id}":
        raise ConversationAcceptanceError(
            "failure cell ID differs from its coordinates"
        )
    applicability = _nonempty_string(
        item.get("applicability"), "cell applicability"
    )
    if applicability not in {"applicable", "not_applicable"}:
        raise ConversationAcceptanceError(
            "failure cell applicability is invalid"
        )
    phase = _phase(item.get("active_from_phase"), "cell active phase")
    lifecycle = _nonempty_string(item.get("lifecycle"), "cell lifecycle")
    expected_lifecycle = "active" if phase <= current_phase else "planned"
    if lifecycle != expected_lifecycle:
        raise ConversationAcceptanceError(
            "failure cell lifecycle disagrees with activation"
        )
    evidence_node_id = _test_node(item.get("evidence_node_id"))
    evidence = node_by_id.get(evidence_node_id)
    if evidence is None or evidence.active_from_phase > phase:
        raise ConversationAcceptanceError(
            "failure cell evidence is missing or activates too late"
        )
    counts = tuple(
        _nonnegative_int(item.get(field), f"failure cell {field}")
        for field in (
            "expected_dispatch_count",
            "visible_output_count",
            "tool_effect_count",
            "checkpoint_commit_count",
        )
    )
    if any(value not in _FAILURE_COUNT_VALUES for value in counts):
        raise ConversationAcceptanceError(
            "failure cell counts must use the closed zero-or-one inventory"
        )
    states = tuple(
        _nonempty_string(item.get(field), f"failure cell {field}")
        for field in (
            "public_mapping",
            "retry_decision",
            "parent_state",
            "public_error",
            "reconciliation_state",
        )
    )
    allowed_states = (
        _PUBLIC_MAPPING_VALUES,
        _RETRY_DECISION_VALUES,
        _PARENT_STATE_VALUES,
        _PUBLIC_ERROR_VALUES,
        _RECONCILIATION_STATE_VALUES,
    )
    if any(
        value not in allowed
        for value, allowed in zip(states, allowed_states, strict=True)
    ):
        raise ConversationAcceptanceError(
            "failure cell uses a state outside the closed Phase 0 inventory"
        )
    _nonempty_string(item.get("rationale"), "failure cell rationale")
    if applicability == "not_applicable":
        if phase != 0 or any(counts) or set(states) != {"not_applicable"}:
            raise ConversationAcceptanceError(
                "non-applicable failure cells need exact Phase 0 evidence"
            )
    elif phase < max(boundary.owner_phase, surface.owner_phase):
        raise ConversationAcceptanceError(
            "applicable failure cell activates before its owners"
        )
    return FailureCell(
        id=identifier,
        boundary_id=boundary_id,
        surface_id=surface_id,
        applicability=applicability,
        lifecycle=lifecycle,
        active_from_phase=phase,
        evidence_node_id=evidence_node_id,
    )


def _validate_threat_model(
    path: Path,
    *,
    manifest: AcceptanceManifest,
    requirement_ids: frozenset[str],
) -> None:
    payload = _strict_mapping(path, "threat model")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "assets",
            "trust_boundaries",
            "threats",
            "threat_model_sha256",
        },
        "threat model",
    )
    _header(payload, "threat model")
    if payload.get("current_phase") != manifest.current_phase:
        raise ConversationAcceptanceError(
            "threat model and acceptance phases differ"
        )
    assets = _string_list(payload.get("assets"), "threat assets")
    trust_boundaries = _string_list(
        payload.get("trust_boundaries"), "trust boundaries"
    )
    if not assets or not trust_boundaries:
        raise ConversationAcceptanceError(
            "threat assets and trust boundaries must be non-empty"
        )
    node_ids = {
        node.node_id
        for node in manifest.nodes
        if node.lifecycle in {"active", "replaced"}
    }
    observed: list[str] = []
    raw_threats = object_list(payload.get("threats"), "threats")
    for raw in raw_threats:
        item = mapping(raw, "threat")
        _exact_keys(
            item,
            {
                "id",
                "asset",
                "actor",
                "boundary",
                "attack",
                "controls",
                "requirement_ids",
                "owner_phase",
                "lifecycle",
                "evidence_node_ids",
            },
            "threat",
        )
        observed.append(_nonempty_string(item.get("id"), "threat ID"))
        for field in ("asset", "actor", "boundary", "attack"):
            _nonempty_string(item.get(field), f"threat {field}")
        if not _string_list(item.get("controls"), "threat controls"):
            raise ConversationAcceptanceError(
                "threat controls must be non-empty"
            )
        owned = _string_list(
            item.get("requirement_ids"), "threat requirements"
        )
        if not owned or not set(owned) <= requirement_ids:
            raise ConversationAcceptanceError(
                "threat references unknown requirements"
            )
        _phase(item.get("owner_phase"), "threat owner phase")
        if item.get("lifecycle") != "active":
            raise ConversationAcceptanceError(
                "Phase 0 threat entries must be active"
            )
        evidence = _string_list(
            item.get("evidence_node_ids"), "threat evidence nodes"
        )
        if not evidence or not set(evidence) <= node_ids:
            raise ConversationAcceptanceError(
                "threat evidence must use active exact nodes"
            )
    phase0_threat_count = len(_THREAT_IDS)
    _unique(observed, "threat ID")
    if (
        len(observed) < phase0_threat_count
        or frozenset(observed[:phase0_threat_count]) != _THREAT_IDS
    ):
        raise ConversationAcceptanceError(
            "threat inventory is incomplete or duplicated"
        )
    canonical = {
        key: value
        for key, value in payload.items()
        if key != "threat_model_sha256"
    }
    if payload.get("threat_model_sha256") != canonical_sha256(canonical):
        raise ConversationAcceptanceError("threat model digest is invalid")
    _validate_threat_structure_anchors(
        assets,
        trust_boundaries,
        raw_threats,
        manifest.current_phase,
    )


def _validate_threat_structure_anchors(
    assets: tuple[str, ...],
    trust_boundaries: tuple[str, ...],
    raw_threats: list[object],
    current_phase: int,
) -> None:
    """Validate cumulative append-only threat structure snapshots."""
    _require_phase_anchor_keys(
        _THREAT_STRUCTURE_BY_PHASE,
        current_phase,
        "threat structure",
    )
    previous = (0, 0, 0)
    for phase in range(current_phase + 1):
        asset_count, boundary_count, threat_count, expected_sha256 = (
            _THREAT_STRUCTURE_BY_PHASE[phase]
        )
        counts = (asset_count, boundary_count, threat_count)
        available = (len(assets), len(trust_boundaries), len(raw_threats))
        if any(
            before > after for before, after in zip(previous, counts)
        ) or any(count > maximum for count, maximum in zip(counts, available)):
            raise ConversationAcceptanceError(
                "threat structure phase anchors are not append-only"
            )
        structure = {
            "assets": assets[:asset_count],
            "trust_boundaries": trust_boundaries[:boundary_count],
            "threats": raw_threats[:threat_count],
        }
        if canonical_sha256(structure) != expected_sha256:
            raise ConversationAcceptanceError(
                "threat structure differs from its immutable phase "
                f"anchor at phase {phase}"
            )
        previous = counts
    if previous != (len(assets), len(trust_boundaries), len(raw_threats)):
        raise ConversationAcceptanceError(
            "threat structure has unanchored appended payload"
        )


def _validate_integrated_fixtures(fixtures: Path) -> None:
    """Validate integrated contract/provider fixtures once they arrive."""
    authoritative = fixtures.resolve() == fixture_root().resolve()
    if authoritative:
        missing = tuple(
            name
            for name in _ACTIVE_INTEGRATED_FIXTURES
            if not (fixtures / name).is_file()
        )
        if missing:
            raise ConversationAcceptanceError(
                f"integrated Phase 0 fixtures are missing: {missing}"
            )
        _validate_phase0_provider_byte_anchors(fixtures.parents[2])
    for name in _ACTIVE_INTEGRATED_FIXTURES:
        path = fixtures / name
        if not path.exists():
            continue
        payload = _strict_mapping(path, f"integrated fixture {name}")
        if name == "contract_decisions.json":
            _validate_contract_decisions(payload)
        elif name == "deterministic_fixtures.json":
            _validate_deterministic_fixtures(payload)
        elif name == "provider_contract.json":
            _validate_provider_contract(payload)
        else:
            _validate_provider_conformance(payload)


def _validate_phase0_provider_byte_anchors(root: Path) -> None:
    """Validate exact provider fixture, test, and production source bytes."""
    for relative, (
        expected_size,
        expected_sha256,
    ) in _PHASE0_PROVIDER_BYTE_ANCHORS.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ConversationAcceptanceError(
                f"anchored Phase 0 provider source is missing: {relative}"
            )
        payload = path.read_bytes()
        if (
            len(payload) != expected_size
            or sha256(payload).hexdigest() != expected_sha256
        ):
            raise ConversationAcceptanceError(
                f"anchored Phase 0 provider source changed: {relative}"
            )


def _validate_contract_decisions(payload: dict[str, object]) -> None:
    expected = {
        "activation",
        "atomic_boundaries",
        "authority",
        "branching",
        "checkpoint",
        "configuration",
        "contract_version",
        "deletion",
        "descendants",
        "failure_fence_tuple_fields",
        "failure_fences",
        "feature",
        "idempotency",
        "identity",
        "migration",
        "owner",
        "provider_lane_binding",
        "public_response_id",
        "response_resource",
        "retention",
        "schema_version",
        "storage",
        "surfaces",
    }
    _exact_keys(payload, expected, "contract decisions")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("activation") != "dormant"
    ):
        raise ConversationAcceptanceError(
            "contract decisions are not the dormant version 1 contract"
        )
    for field in expected - {
        "activation",
        "contract_version",
        "feature",
        "owner",
        "schema_version",
    }:
        if (
            not isinstance(payload.get(field), (dict, list))
            or not payload[field]
        ):
            raise ConversationAcceptanceError(
                f"contract decision group is empty: {field}"
            )


def _validate_deterministic_fixtures(payload: dict[str, object]) -> None:
    expected = {
        "async_barrier",
        "clock",
        "contract_version",
        "fault_injection",
        "fixture_sha256",
        "id_factory",
        "keys",
        "named_head_cases",
        "principal",
        "provider_capability",
        "provider_item_trace",
        "public_response_resources",
        "retention_cases",
        "schema_version",
    }
    _exact_keys(payload, expected, "deterministic fixtures")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_version") != 1
    ):
        raise ConversationAcceptanceError(
            "deterministic fixtures are not the version 1 inventory"
        )
    for field in expected - {"contract_version", "schema_version"}:
        if field == "fixture_sha256":
            continue
        if (
            not isinstance(payload.get(field), (dict, list))
            or not payload[field]
        ):
            raise ConversationAcceptanceError(
                f"deterministic fixture group is empty: {field}"
            )
    canonical = {
        key: value for key, value in payload.items() if key != "fixture_sha256"
    }
    if payload.get("fixture_sha256") != canonical_sha256(canonical):
        raise ConversationAcceptanceError(
            "deterministic fixture digest is invalid"
        )


def _validate_provider_contract(payload: dict[str, object]) -> None:
    expected = {
        "activation_state",
        "canonical_digest",
        "conformance_digest",
        "contract_version",
        "current_phase",
        "feature",
        "owner",
        "retrieved_date",
        "schema_version",
        "sdk_boundary",
        "snapshot_id",
        "sources",
    }
    _exact_keys(payload, expected, "provider contract")
    _validate_provider_header(payload, "provider contract")
    sdk = mapping(payload.get("sdk_boundary"), "provider SDK boundary")
    policy = mapping(
        sdk.get("conversation_state_transport_policy"),
        "conversation state transport policy",
    )
    _exact_keys(
        policy,
        {
            "scope",
            "runtime_disposition",
            "legacy_generic_request_kwargs_acknowledged",
            "legacy_generic_request_kwargs_description",
            "prohibited_routes",
            "provider_wire_paths",
            "public_request_fields",
            "reasoning_mapping_policy",
            "stateful_create_field_policy",
        },
        "conversation state transport policy",
    )
    if (
        policy.get("scope") != "conversation_state_and_stateful_create_fields"
        or policy.get("runtime_disposition") != "dormant_fail_closed"
        or policy.get("legacy_generic_request_kwargs_acknowledged") is not True
    ):
        raise ConversationAcceptanceError(
            "provider conversation-state transport policy is not fail closed"
        )
    _nonempty_string(
        policy.get("legacy_generic_request_kwargs_description"),
        "legacy generic request kwargs description",
    )
    if _string_list(
        policy.get("prohibited_routes"), "prohibited provider routes"
    ) != (
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
    ):
        raise ConversationAcceptanceError(
            "provider prohibited transport routes changed"
        )
    if _string_list(
        policy.get("provider_wire_paths"), "provider wire paths"
    ) != (
        "background",
        "previous_response_id",
        "conversation",
        "context_management",
        "context_management.compact_threshold",
        "reasoning.context",
        "store",
    ):
        raise ConversationAcceptanceError(
            "provider wire path inventory changed"
        )
    if _string_list(
        policy.get("public_request_fields"), "public request fields"
    ) != (
        "background",
        "previous_response_id",
        "conversation",
        "context_management",
        "reasoning_context",
        "conversation_handle",
        "continuation_envelope",
        "store",
    ):
        raise ConversationAcceptanceError(
            "provider public conversation request field inventory changed"
        )
    _validate_reasoning_mapping_policy(policy.get("reasoning_mapping_policy"))
    _validate_stateful_create_field_policy(
        policy.get("stateful_create_field_policy")
    )
    _validate_scoped_digest(payload, "provider contract")
    digest = mapping(
        payload.get("canonical_digest"), "provider contract digest"
    )
    if digest.get("value") != _PHASE0_PROVIDER_CANONICAL_SHA256:
        raise ConversationAcceptanceError(
            "provider contract differs from its independent Phase 0 anchor"
        )


def _validate_reasoning_mapping_policy(raw: object) -> None:
    """Validate the closed static reasoning mapping policy."""
    policy = mapping(raw, "reasoning mapping policy")
    if policy != {
        "mapping_name": "reasoning",
        "allowed_static_keys": ["effort", "summary"],
        "forbidden_path": "reasoning.context",
        "dynamic_keys_allowed": False,
        "aliases_allowed": False,
        "mutator_calls_allowed": False,
    }:
        raise ConversationAcceptanceError(
            "provider reasoning mapping policy changed"
        )


def _validate_stateful_create_field_policy(raw: object) -> None:
    """Validate fixed Phase 0 provider retention and dispatch policy."""
    policy = mapping(raw, "stateful create field policy")
    _exact_keys(
        policy,
        {
            "forbidden_provider_wire_roots",
            "typed_sdk_create_fields",
            "legacy_fixed_provider_values",
            "provider_mapping_flow",
            "closed_ast_gate",
            "public_runtime_disposition",
        },
        "stateful create field policy",
    )
    typed = mapping(
        policy.get("typed_sdk_create_fields"),
        "typed stateful create fields",
    )
    _exact_keys(typed, {"background", "store"}, "typed stateful create fields")
    annotation_sha256 = (
        "80624365ea5db072b2ea31b2a3bf9d483b05fd5f828c7fc0ed7da554518892a5"
    )
    background = mapping(typed.get("background"), "background field policy")
    _exact_keys(
        background,
        {
            "sdk_parameter_kind",
            "sdk_default_contract",
            "sdk_resolved_annotation_sha256",
            "provider_runtime_disposition",
            "allowed_provider_write_count",
            "public_runtime_disposition",
        },
        "background field policy",
    )
    store = mapping(typed.get("store"), "store field policy")
    _exact_keys(
        store,
        {
            "sdk_parameter_kind",
            "sdk_default_contract",
            "sdk_resolved_annotation_sha256",
            "provider_runtime_disposition",
            "allowed_provider_write_count",
            "allowed_provider_value",
            "public_runtime_disposition",
        },
        "store field policy",
    )
    shared = {
        "sdk_parameter_kind": "KEYWORD_ONLY",
        "sdk_default_contract": "singleton:openai.Omit",
        "sdk_resolved_annotation_sha256": annotation_sha256,
        "public_runtime_disposition": "dormant_fail_closed",
    }
    if (
        any(background.get(key) != value for key, value in shared.items())
        or background.get("provider_runtime_disposition") != "prohibited"
        or type(background.get("allowed_provider_write_count")) is not int
        or background.get("allowed_provider_write_count") != 0
    ):
        raise ConversationAcceptanceError(
            "background provider policy is not fail closed"
        )
    if (
        any(store.get(key) != value for key, value in shared.items())
        or store.get("provider_runtime_disposition")
        != "legacy_fixed_false_only"
        or type(store.get("allowed_provider_write_count")) is not int
        or store.get("allowed_provider_write_count") != 1
        or store.get("allowed_provider_value") is not False
    ):
        raise ConversationAcceptanceError(
            "store provider policy is not fixed to false"
        )
    legacy_values = mapping(
        policy.get("legacy_fixed_provider_values"),
        "legacy fixed provider values",
    )
    if (
        set(legacy_values) != {"store"}
        or legacy_values.get("store") is not False
    ):
        raise ConversationAcceptanceError(
            "legacy provider values are not fixed to store=false"
        )
    mapping_flow = mapping(
        policy.get("provider_mapping_flow"),
        "provider mapping flow",
    )
    if mapping_flow != {
        "initial_request_mapping": "kwargs",
        "normalization_temporary": "normalized_request_kwargs",
        "normalized_request_mapping": "request_kwargs",
        "attempt_request_mapping": "attempt_kwargs",
        "copy_function": "_strict_replay_json_copy",
        "create_target": "request_client.responses.create",
        "create_unpack_source": "attempt_kwargs",
        "create_call_count": 1,
        "mapping_unpack_count": 1,
    } or any(
        type(mapping_flow.get(field)) is not int
        for field in ("create_call_count", "mapping_unpack_count")
    ):
        raise ConversationAcceptanceError("provider mapping flow changed")
    if _string_list(
        policy.get("forbidden_provider_wire_roots"),
        "forbidden provider wire roots",
    ) != (
        "background",
        "compact_threshold",
        "context_management",
        "conversation",
        "extra_body",
        "previous_response_id",
        "store",
    ):
        raise ConversationAcceptanceError(
            "forbidden provider wire roots changed"
        )
    closed_gate = mapping(policy.get("closed_ast_gate"), "closed AST gate")
    if closed_gate != {
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
    }:
        raise ConversationAcceptanceError("provider closed AST gate changed")
    source_integrity = mapping(
        closed_gate.get("phase0_source_integrity"),
        "Phase 0 provider source integrity",
    )
    if type(source_integrity.get("phase")) is not int:
        raise ConversationAcceptanceError(
            "provider source-integrity phase must be an integer"
        )
    if policy.get("public_runtime_disposition") != "dormant_fail_closed":
        raise ConversationAcceptanceError(
            "stateful public create fields are not fail closed"
        )


def _validate_provider_conformance(payload: dict[str, object]) -> None:
    expected = {
        "activation_state",
        "canonical_digest",
        "capability_names",
        "capability_states",
        "current_phase",
        "feature",
        "identity_dimensions",
        "inference_policy",
        "owner",
        "production_advertisement_enabled",
        "production_dispatch_enabled",
        "profile_schema_version",
        "profiles",
        "rejected_inference_cases",
        "schema_version",
    }
    _exact_keys(payload, expected, "provider conformance")
    _validate_provider_header(payload, "provider conformance")
    if (
        payload.get("production_advertisement_enabled") is not False
        or payload.get("production_dispatch_enabled") is not False
    ):
        raise ConversationAcceptanceError(
            "Phase 0 provider capabilities must not advertise or dispatch"
        )
    profiles = object_list(payload.get("profiles"), "provider profiles")
    if not profiles:
        raise ConversationAcceptanceError(
            "provider profiles must be non-empty"
        )
    for raw in profiles:
        profile = mapping(raw, "provider profile")
        activation = profile.get("activation_state")
        lifecycle = profile.get("lifecycle")
        if activation not in {"dormant", "incapable"} or lifecycle not in {
            "planned",
            "incapable",
        }:
            raise ConversationAcceptanceError(
                "provider profiles must remain planned/dormant or incapable"
            )
        if profile.get("identity_complete") is not False:
            raise ConversationAcceptanceError(
                "Phase 0 provider profiles cannot claim complete identity"
            )
        capabilities = mapping(
            profile.get("capabilities"), "provider capabilities"
        )
        if not capabilities or any(
            value not in {"dormant", "incapable"}
            for value in capabilities.values()
        ):
            raise ConversationAcceptanceError(
                "provider capability state is prematurely active"
            )
        if object_list(
            profile.get("activation_evidence"), "activation evidence"
        ):
            raise ConversationAcceptanceError(
                "Phase 0 profiles cannot contain activation evidence"
            )
    _validate_scoped_digest(payload, "provider conformance")


def _validate_provider_header(
    payload: dict[str, object],
    label: str,
) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("feature") != _FEATURE
        or payload.get("current_phase") != 0
        or payload.get("activation_state") != "dormant"
    ):
        raise ConversationAcceptanceError(
            f"{label} is not the dormant Phase 0 version"
        )


def _validate_scoped_digest(
    payload: dict[str, object],
    label: str,
) -> None:
    digest = mapping(payload.get("canonical_digest"), f"{label} digest")
    _exact_keys(
        digest,
        {"algorithm", "encoding", "scope", "value"},
        f"{label} digest",
    )
    if digest.get("algorithm") != "sha256":
        raise ConversationAcceptanceError(
            f"{label} digest algorithm must be sha256"
        )
    scope = _string_list(digest.get("scope"), f"{label} digest scope")
    expected_scope = tuple(
        field for field in payload if field != "canonical_digest"
    )
    if scope != expected_scope:
        raise ConversationAcceptanceError(f"{label} digest scope is invalid")
    scoped = {field: payload[field] for field in scope}
    if digest.get("value") != canonical_sha256(scoped):
        raise ConversationAcceptanceError(f"{label} digest is invalid")


def _validate_type_manifest(
    fixtures: Path,
    current_phase: int,
    root: Path,
    *,
    acceptance_path: Path | None = None,
) -> None:
    """Reuse complete type-manifest and source-anchor validation."""
    path = (
        companion_fixture_path(acceptance_path, "type_contract_manifest")
        if acceptance_path is not None
        else fixtures / "type_contract_manifest.json"
    )
    if not path.is_file():
        raise ConversationAcceptanceError(
            "conversation type-contract manifest is missing"
        )
    try:
        manifest = load_type_contract_manifest(path)
        if manifest.current_phase != current_phase:
            raise ConversationAcceptanceError(
                "type and acceptance manifest phases differ"
            )
        validate_type_source_phase_anchors(manifest, root)
    except (
        ContractGateError,
        ConversationTypeContractError,
        StrictJsonError,
    ) as exc:
        raise ConversationAcceptanceError(
            f"type-contract validation failed: {exc}"
        ) from exc


def _validate_activation_history(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> tuple[tuple[str, ...], ...]:
    history = object_list(raw, "activation history")
    _require_phase_anchor_keys(
        _ACTIVATION_HISTORY_BY_PHASE,
        current_phase,
        "activation history",
    )
    if len(history) != current_phase + 1:
        raise ConversationAcceptanceError(
            "activation history must preserve every implemented phase"
        )
    previous: set[str] = set()
    snapshots: list[tuple[str, ...]] = []
    for expected_phase, raw_entry in enumerate(history):
        entry = mapping(raw_entry, "activation history entry")
        _exact_keys(entry, {"phase", "node_ids", "sha256"}, "activation entry")
        if _phase(entry.get("phase"), "activation phase") != expected_phase:
            raise ConversationAcceptanceError(
                "activation history phases must be contiguous"
            )
        node_ids = _string_list(entry.get("node_ids"), "activation node IDs")
        _unique(node_ids, "activation node ID")
        expected_ids = tuple(
            node.node_id
            for node in nodes
            if node.lifecycle in {"active", "replaced"}
            and node.active_from_phase <= expected_phase
        )
        if node_ids != expected_ids or not previous <= set(node_ids):
            raise ConversationAcceptanceError(
                "activation history is not monotonic at phase"
                f" {expected_phase}"
            )
        if entry.get("sha256") != _text_digest(node_ids):
            raise ConversationAcceptanceError(
                "activation history digest is invalid at phase"
                f" {expected_phase}"
            )
        if entry.get("sha256") != _ACTIVATION_HISTORY_BY_PHASE[expected_phase]:
            raise ConversationAcceptanceError(
                "activation history differs from its immutable phase "
                f"anchor at phase {expected_phase}"
            )
        previous = set(node_ids)
        snapshots.append(node_ids)
    return tuple(snapshots)


def _validate_node_phase_anchors(
    raw_nodes: list[object],
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> None:
    """Validate independently anchored node payloads by activation phase."""
    _require_phase_anchor_keys(
        _NODE_PAYLOAD_SHA256_BY_PHASE,
        current_phase,
        "acceptance node payload",
    )
    for phase in range(current_phase + 1):
        payload = [
            {
                key: value
                for key, value in mapping(raw, "acceptance node").items()
                if key != "lifecycle"
            }
            for raw, node in zip(raw_nodes, nodes, strict=True)
            if node.active_from_phase == phase
        ]
        if canonical_sha256(payload) != _NODE_PAYLOAD_SHA256_BY_PHASE[phase]:
            raise ConversationAcceptanceError(
                "acceptance node payload differs from its independent "
                f"phase anchor at phase {phase}"
            )


def _require_phase_anchor_keys(
    anchors: Mapping[int, object],
    current_phase: int,
    label: str,
) -> None:
    """Require one append-only independent anchor per implemented phase."""
    expected = set(range(current_phase + 1))
    if not expected <= set(anchors):
        raise ConversationAcceptanceError(
            f"{label} anchors must cover every implemented phase"
        )


def _validate_replacements(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> tuple[AcceptanceReplacement, ...]:
    replacements = object_list(raw, "acceptance replacements")
    current_ids = {node.node_id for node in nodes}
    parsed: list[AcceptanceReplacement] = []
    old_ids: list[str] = []
    targets: list[str] = []
    phases: list[int] = []
    for raw_entry in replacements:
        entry = mapping(raw_entry, "acceptance replacement")
        _exact_keys(
            entry,
            {
                "phase",
                "old_node_id",
                "replacement_node_ids",
                "reviewed_by",
                "evidence",
            },
            "acceptance replacement",
        )
        phase = _phase(entry.get("phase"), "replacement phase")
        if phase > current_phase:
            raise ConversationAcceptanceError(
                "future replacements cannot alter activation history"
            )
        old = _test_node(entry.get("old_node_id"))
        replacements_for_old = _string_list(
            entry.get("replacement_node_ids"), "replacement nodes"
        )
        if (
            old not in current_ids
            or not replacements_for_old
            or not set(replacements_for_old) <= current_ids
            or old in replacements_for_old
        ):
            raise ConversationAcceptanceError(
                "replacement tombstone differs from current inventory"
            )
        old_ids.append(old)
        targets.extend(replacements_for_old)
        phases.append(phase)
        parsed.append(
            AcceptanceReplacement(
                phase=phase,
                old_node_id=old,
                replacement_node_ids=replacements_for_old,
            )
        )
        _nonempty_string(entry.get("reviewed_by"), "replacement reviewer")
        _nonempty_string(entry.get("evidence"), "replacement evidence")
    _unique(old_ids, "replaced node ID")
    _unique(targets, "replacement target")
    _validate_replacement_phase_anchors(
        replacements,
        tuple(phases),
        current_phase,
    )
    return tuple(parsed)


def _validate_replacement_transitions(
    replacements: tuple[AcceptanceReplacement, ...],
    nodes: tuple[AcceptanceNode, ...],
    activation_history: tuple[tuple[str, ...], ...],
) -> None:
    """Validate retained tombstones against adjacent activation snapshots."""
    node_by_id = {node.node_id: node for node in nodes}
    replacement_by_old = {
        replacement.old_node_id: replacement for replacement in replacements
    }
    replaced_ids = {
        node.node_id for node in nodes if node.lifecycle == "replaced"
    }
    if replaced_ids != set(replacement_by_old):
        raise ConversationAcceptanceError(
            "replaced acceptance records and reviewed ledger entries differ"
        )
    for replacement in replacements:
        if replacement.phase == 0:
            raise ConversationAcceptanceError(
                "acceptance replacements require a preceding phase snapshot"
            )
        old = node_by_id[replacement.old_node_id]
        previous = set(activation_history[replacement.phase - 1])
        current = set(activation_history[replacement.phase])
        additions = current - previous
        same_phase_split = old.active_from_phase == replacement.phase
        retained_prior = (
            old.active_from_phase < replacement.phase
            and replacement.old_node_id in previous
        )
        introduced_split = (
            same_phase_split
            and replacement.old_node_id not in previous
            and replacement.old_node_id in additions
        )
        if old.lifecycle != "replaced" or not (
            retained_prior or introduced_split
        ):
            raise ConversationAcceptanceError(
                "acceptance replacement old record is neither a retained "
                "prior member nor a reviewed same-phase split"
            )
        target_requirement_sets: list[set[str]] = []
        for target_id in replacement.replacement_node_ids:
            target = node_by_id[target_id]
            if (
                target.active_from_phase != replacement.phase
                or target.lifecycle not in {"active", "replaced"}
                or target_id not in additions
            ):
                raise ConversationAcceptanceError(
                    "acceptance replacement targets must be new same-phase "
                    "records"
                )
            target_requirement_sets.append(set(target.requirement_ids))
        old_requirements = set(old.requirement_ids)
        replicated = all(
            requirements == old_requirements
            for requirements in target_requirement_sets
        )
        partitioned = (
            all(target_requirement_sets)
            and set().union(*target_requirement_sets) == old_requirements
            and sum(len(value) for value in target_requirement_sets)
            == len(old_requirements)
        )
        if not (replicated or partitioned):
            raise ConversationAcceptanceError(
                "acceptance replacement targets must replicate or exactly "
                "partition preserved requirement ownership"
            )


def _validate_replacement_phase_anchors(
    replacements: list[object],
    phases: tuple[int, ...],
    current_phase: int,
) -> None:
    """Validate cumulative append-only acceptance replacement history."""
    _require_phase_anchor_keys(
        _REPLACEMENT_HISTORY_BY_PHASE,
        current_phase,
        "acceptance replacement history",
    )
    previous_count = 0
    for phase in range(current_phase + 1):
        count, expected_sha256 = _REPLACEMENT_HISTORY_BY_PHASE[phase]
        if (
            count < previous_count
            or count > len(replacements)
            or any(value > phase for value in phases[:count])
            or any(value <= phase for value in phases[count:])
        ):
            raise ConversationAcceptanceError(
                "acceptance replacement history anchors are not append-only"
            )
        if canonical_sha256(replacements[:count]) != expected_sha256:
            raise ConversationAcceptanceError(
                "acceptance replacement history differs from its immutable "
                f"phase anchor at phase {phase}"
            )
        previous_count = count
    if previous_count != len(replacements):
        raise ConversationAcceptanceError(
            "acceptance replacement history has unanchored appended payload"
        )


def _strict_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        return mapping(strict_json_path(path), label)
    except (ContractGateError, StrictJsonError) as exc:
        raise ConversationAcceptanceError(
            f"cannot read {label}: {exc}"
        ) from exc


def _header(payload: dict[str, object], label: str) -> None:
    if payload.get("schema_version") != 1:
        raise ConversationAcceptanceError(f"{label} schema_version must be 1")
    if payload.get("feature") != _FEATURE:
        raise ConversationAcceptanceError(
            f"{label} feature must be {_FEATURE}"
        )


def _phase(value: object, label: str) -> int:
    if type(value) is not int or not _MIN_PHASE <= value <= _MAX_PHASE:
        raise ConversationAcceptanceError(
            f"{label} must be an integer from {_MIN_PHASE} through"
            f" {_MAX_PHASE}"
        )
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ConversationAcceptanceError(
            f"{label} must be a non-negative integer"
        )
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ConversationAcceptanceError(
            f"{label} must be a positive integer"
        )
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationAcceptanceError(
            f"{label} must be a non-empty string"
        )
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    return tuple(
        _nonempty_string(item, label) for item in object_list(value, label)
    )


def _relative_path(value: object, label: str) -> str:
    raw = _nonempty_string(value, label)
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw:
        raise ConversationAcceptanceError(f"{label} escapes the repository")
    return raw


def _test_node(value: object) -> str:
    node_id = _nonempty_string(value, "pytest node ID")
    relative = node_id.split("::", 1)[0]
    if (
        _NODE_PATTERN.fullmatch(node_id) is None
        or "\\" in node_id
        or ".." in PurePosixPath(relative).parts
    ):
        raise ConversationAcceptanceError(f"invalid pytest node ID: {node_id}")
    return node_id


def _unique(values: Iterable[object], label: str) -> None:
    items = tuple(values)
    if len(items) != len(set(items)):
        raise ConversationAcceptanceError(f"duplicate {label}")


def _exact_keys(
    value: dict[str, object],
    expected: Iterable[str],
    label: str,
) -> None:
    expected_keys = set(expected)
    if set(value) != expected_keys:
        raise ConversationAcceptanceError(
            f"{label} has invalid keys: {sorted(set(value) ^ expected_keys)}"
        )


def _text_digest(values: tuple[str, ...]) -> str:
    return sha256("\n".join(values).encode("utf-8")).hexdigest()


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Collect and execute active conversation acceptance nodes without "
            "skips, xfails, deselection, or placeholder evidence."
        )
    )
    parser.add_argument("--through-phase", required=True, type=int)
    parser.add_argument(
        "--manifest", type=Path, default=default_manifest_path()
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Run conversation acceptance verification from the command line."""
    args = _parse_args()
    try:
        manifest = verify_acceptance(
            args.manifest,
            repo_root=args.repo_root,
            through_phase=args.through_phase,
            execute=not args.validate_only,
        )
    except (
        ContractGateError,
        ConversationAcceptanceError,
        StrictJsonError,
    ) as exc:
        print(f"conversation acceptance failed: {exc}", file=stderr)
        return 1
    active = len(manifest.active_nodes(args.through_phase))
    planned = len(manifest.planned_nodes())
    print(
        "conversation acceptance passed: "
        f"through_phase={args.through_phase} active={active} planned={planned}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
