#!/usr/bin/env python
"""Collect and execute the exact structured-input acceptance inventory."""

from argparse import ArgumentParser, Namespace
from ast import (
    AST,
    Assert,
    AsyncFor,
    AsyncFunctionDef,
    AsyncWith,
    Attribute,
    Break,
    Call,
    ClassDef,
    Compare,
    Constant,
    Continue,
    Eq,
    Expr,
    For,
    FunctionDef,
    If,
    ImportFrom,
    Is,
    Load,
    Match,
    MatchAs,
    Name,
    Pass,
    Raise,
    Return,
    Subscript,
    Try,
    While,
    With,
    dump,
    parse,
    walk,
)
from ast import (
    Dict as AstDict,
)
from ast import (
    List as AstList,
)
from ast import (
    Set as AstSet,
)
from ast import (
    Tuple as AstTuple,
)
from collections.abc import Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from json import dumps
from os import environ
from pathlib import Path, PurePosixPath
from re import IGNORECASE
from re import compile as compile_regex
from subprocess import CompletedProcess, TimeoutExpired, run
from sys import executable, stderr
from typing import Protocol, cast

from input_contract_json import (
    StrictJsonError,
    strict_json_loads,
    strict_json_path,
)

_FEATURE = "structured_task_input"
_MIN_PHASE = 0
_MAX_PHASE = 12
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
_DISALLOWED_MARKERS = frozenset(("skip", "skipif", "xfail"))
_PROHIBITED_TEST_CONTROLS = frozenset(
    (
        "skip",
        "skipIf",
        "skipUnless",
        "importorskip",
        "xfail",
    )
)
_PROHIBITED_EXECUTION_NAMES = frozenset(("exec", "compile"))
_PROHIBITED_TEST_SYMBOLS = (
    _PROHIBITED_TEST_CONTROLS | _PROHIBITED_EXECUTION_NAMES
)
_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_PUBLIC_SCHEMA_MUTATIONS = (
    "missing_required_field",
    "extra_field",
    "wrong_const",
    "wrong_type",
    "cross_field_invariant",
)
_EXPECTED_PROTOCOL_SCHEMA_SHA256 = {
    "a2a_message_metadata": (
        "f2226918b7d610aeabd987434cd7e186902486add3fb4e4ad2abebe6ca9fbeb3"
    ),
    "mcp_params_task": (
        "ecd4f03f32ca078f6e47650385ff7abe968f5f1bbaf4ba2c32b41e52fd08c834"
    ),
    "mcp_task": (
        "f1fb83bcc7c798c59985258b55df13883efa1f314f1d329ede4995dc657efa0e"
    ),
    "mcp_create_task_result": (
        "185f091128f71fc8fd3ad202b2625a0b2bf8a70b2d6bbdc8c7ea2304d7041f86"
    ),
}
_PUBLIC_CROSS_FIELD_INVARIANTS = {
    "a2a.task_working.v1": frozenset(
        {
            "a2a_resolution_task_id",
            "a2a_resolution_context_id",
            "a2a_resolution_request_id",
        }
    ),
    "a2a.task_input_required.v1": frozenset(
        {
            "a2a_task_message_task_id",
            "a2a_task_message_context_id",
            "a2a_task_message_request_id",
        }
    ),
    "mcp.task_cancelled.v1": frozenset(
        {
            "mcp_related_task_id",
            "mcp_canonical_request_id",
        }
    ),
    "mcp.task_input_required.v1": frozenset(
        {
            "mcp_related_task_id",
            "mcp_canonical_request_id",
        }
    ),
    "mcp.task_working.v1": frozenset(
        {
            "mcp_related_task_id",
            "mcp_canonical_request_id",
        }
    ),
}
_COLLECT_SENTINEL = "__INPUT_ACCEPTANCE_COLLECT__"
_EXECUTE_SENTINEL = "__INPUT_ACCEPTANCE_EXECUTE__"
_PROCESS_TIMEOUT_SECONDS = 300
_COVERAGE_EXCLUSION_PATTERN = compile_regex(
    r"#\s*(?:pragma\s*:?\s*no\s*cover|coverage\s*:?\s*ignore)",
    IGNORECASE,
)
_TRANSITION_PATTERN = compile_regex(r"([a-z][a-z0-9_]*)->([a-z][a-z0-9_]*)")
_PUBLIC_RESULT_PATTERN = compile_regex(r"envelope=([a-z][a-z0-9._-]*)")
_STATUS_OR_EXIT_PATTERN = compile_regex(
    r"([a-z][a-z0-9_]*)=(-?[A-Za-z0-9][A-Za-z0-9._-]*)"
)
_INTERACTION_STATES = frozenset(
    {
        "created",
        "pending",
        "running",
        "answered",
        "declined",
        "cancelled",
        "timed_out",
        "unavailable",
        "expired",
        "superseded",
    }
)
_STATUS_OR_EXIT_KEYS = frozenset(
    {
        "exit",
        "interaction_state",
        "result",
        "exception",
        "http",
        "jsonrpc_error",
        "task_status",
        "task_state",
        "flow_state",
        "branch_state",
        "capability",
        "client_result",
    }
)
_BEHAVIOR_REQUIREMENT_IDS = frozenset(
    tuple(f"INPUT-N-{index:03d}" for index in range(1, 108))
    + tuple(f"INPUT-26.{index}" for index in range(1, 13))
)
_GATE_REQUIREMENT_IDS = frozenset(
    f"INPUT-GATE-{index:03d}" for index in range(1, 13)
)
_EXPECTED_REQUIREMENT_IDS = _BEHAVIOR_REQUIREMENT_IDS | _GATE_REQUIREMENT_IDS
_EXPECTED_FAILURE_CONDITIONS = frozenset(
    f"INPUT-F-{index:02d}" for index in range(1, 16)
)
_EXPECTED_NO_BC_IDS = frozenset(
    {
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
)
_EXPECTED_REQUIREMENTS_SHA256 = (
    "4871c2cee6bab371387d03dcfac8f81c5d4a3583bd6d6c510e0314a37468f06a"
)
_EXPECTED_FAILURE_MATRIX_SHA256 = (
    "e5ce3aac0d441897b80a09d6a693853c65d4a446ed7e4c0184b3e3bc0b212c08"
)
_EXPECTED_DECISIONS_SHA256 = (
    "ef8702b71d737e16a72d182169b1d23cfefe995f8450f79c4a7bee906ddd649a"
)
_EXPECTED_NO_BC_SHA256 = (
    "4a0140865a8ba58d2590fbc75245326c3d791f3dc541c52e8d3657b987d563b6"
)
_EXPECTED_ACCEPTANCE_LEDGER_SHA256 = (
    "d84ff52b1cc1c6b6dbba78aa92c309d671d801df966343450256c9b2f2066dbd"
)
_EXPECTED_EVIDENCE_SHA256 = (
    "75fab61eece213c6b968b6c75e15ba2905cdaeefc775349bd71312d2cc1dbccb"
)
_EXPECTED_IMPLEMENTATION_OWNER = "/root"
_EXPECTED_INDEPENDENT_REVIEWER = "/root/input_contract_audit"
_EXPECTED_BASELINE_HEAD = "609aa091c17756ab952cf5fe668ca3d867f0e311"
_EXPECTED_BASELINE_SUBJECT = "Bump version to v1.5.8 (#1067)"
_EXPECTED_BOUNDARY_PATHS = frozenset(
    {
        ".github/workflows/test.yml",
        "Makefile",
        "scripts/input_contract_json.py",
        "scripts/run_input_contract_gate.py",
        "scripts/task_pgsql_test_database.py",
        "scripts/verify_input_acceptance.py",
        "scripts/verify_input_types.py",
        "scripts/verify_src_coverage.py",
        "tests/fixtures/input/",
        "tests/input_acceptance_verifier_test.py",
        "tests/input_contract_fixtures.py",
        "tests/input_contract_harness_test.py",
        "tests/input_contract_metadata_test.py",
        "tests/input_contract_test.py",
        "tests/input_type_contract_test.py",
        "tests/input_type_contracts/",
        "tests/model/full_coverage_gap_model_test.py",
        "tests/project_metadata_test.py",
        "tests/reasoning_summary_phase1_test.py",
        "tests/src_coverage_verifier_test.py",
    }
)
_EXPECTED_COMMON_GATE_COMMANDS = frozenset(
    {
        "poetry run pytest --verbose -s",
        "make test-coverage -- -100 src/",
        "make test-coverage-exact no-install",
        (
            "poetry run python scripts/verify_input_acceptance.py"
            " --through-phase 0"
        ),
        "make typecheck-input-contract INPUT_PHASE=0",
        "make lint",
        "git diff --check",
    }
)

_COLLECT_DRIVER = f"""
from json import dumps
from sys import argv
from pytest import main

class Probe:
    def __init__(self):
        self.items = []
        self.deselected = []
        self.collection_reports = []

    def pytest_collection_finish(self, session):
        self.items = [
            {{
                "nodeid": item.nodeid,
                "markers": sorted(
                    marker.name for marker in item.iter_markers()
                ),
            }}
            for item in session.items
        ]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection_reports.append(
                {{
                    "nodeid": report.nodeid,
                    "outcome": report.outcome,
                    "detail": str(report.longrepr),
                }}
            )

probe = Probe()
exit_code = main(
    [
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        "-p",
        "anyio.pytest_plugin",
        *argv[1:],
    ],
    plugins=[probe],
)
print("{_COLLECT_SENTINEL}" + dumps({{
    "exit_code": int(exit_code),
    "items": probe.items,
    "deselected": probe.deselected,
    "collection_reports": probe.collection_reports,
}}, sort_keys=True))
"""

_EXECUTE_DRIVER = f"""
from json import dumps
from sys import argv
from pytest import main

class Probe:
    def __init__(self):
        self.items = []
        self.deselected = []
        self.collection_reports = []
        self.reports = []

    def pytest_collection_finish(self, session):
        self.items = [item.nodeid for item in session.items]

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report):
        if report.failed or report.skipped:
            self.collection_reports.append({{
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "detail": str(report.longrepr),
            }})

    def pytest_runtest_logreport(self, report):
        self.reports.append({{
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "wasxfail": str(getattr(report, "wasxfail", "")),
            "detail": (
                str(report.longrepr)
                if report.failed or report.skipped
                else ""
            ),
        }})

probe = Probe()
exit_code = main(
    ["-q", "-p", "no:cacheprovider", "-p", "anyio.pytest_plugin", *argv[1:]],
    plugins=[probe],
)
print("{_EXECUTE_SENTINEL}" + dumps({{
    "exit_code": int(exit_code),
    "items": probe.items,
    "deselected": probe.deselected,
    "collection_reports": probe.collection_reports,
    "reports": probe.reports,
}}, sort_keys=True))
"""


class AcceptanceVerificationError(RuntimeError):
    """Report an invalid or non-passing acceptance inventory."""


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
    """Store the validated acceptance inventory."""

    path: Path
    current_phase: int
    nodes: tuple[AcceptanceNode, ...]

    def active_nodes(self, through_phase: int) -> tuple[AcceptanceNode, ...]:
        """Return active nodes introduced no later than the requested gate."""
        assert _MIN_PHASE <= through_phase <= self.current_phase
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle == "active"
            and node.active_from_phase <= through_phase
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class _CheckPaths:
    """Store check state for every reachable control-flow outcome."""

    next_states: frozenset[bool] = frozenset()
    return_states: frozenset[bool] = frozenset()
    break_states: frozenset[bool] = frozenset()
    continue_states: frozenset[bool] = frozenset()


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
    """Load and validate the lifecycle-aware acceptance manifest."""
    payload = _strict_mapping(path, "acceptance manifest")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "categories",
            "activation_history",
            "activation_snapshots",
            "replacements",
            "nodes",
        },
        "acceptance manifest",
    )
    _header(payload, "acceptance manifest")
    current_phase = _phase(payload.get("current_phase"), "current_phase")
    categories = _string_list(payload.get("categories"), "categories")
    if frozenset(categories) != _CATEGORIES or len(categories) != len(
        _CATEGORIES
    ):
        raise AcceptanceVerificationError(
            "acceptance categories must contain the exact required inventory"
        )
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise AcceptanceVerificationError(
            "acceptance nodes must be a non-empty list"
        )
    nodes = tuple(_acceptance_node(item, current_phase) for item in raw_nodes)
    _unique((node.id for node in nodes), "acceptance node ID")
    _unique((node.node_id for node in nodes), "pytest node ID")
    if frozenset(node.category for node in nodes) != _CATEGORIES:
        raise AcceptanceVerificationError(
            "every acceptance category must own at least one node"
        )
    _activation_history(
        payload.get("activation_history"),
        nodes,
        current_phase,
    )
    _activation_snapshots(
        payload.get("activation_snapshots"),
        payload.get("replacements"),
        nodes,
        current_phase,
    )
    return AcceptanceManifest(
        path=path,
        current_phase=current_phase,
        nodes=nodes,
    )


def verify_acceptance(
    manifest_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    through_phase: int,
    contract_fixture_root: Path | None = None,
) -> AcceptanceManifest:
    """Validate fixtures and require exact passing node execution."""
    root = (repo_root or repository_root()).resolve()
    path = manifest_path or default_manifest_path()
    manifest = load_manifest(path)
    if through_phase < _MIN_PHASE or through_phase > manifest.current_phase:
        raise AcceptanceVerificationError(
            "through-phase must be implemented by the current manifest: "
            f"requested={through_phase}, current={manifest.current_phase}"
        )
    fixtures = contract_fixture_root or path.parent
    _validate_contract_fixtures(manifest, fixtures, root)
    active = manifest.active_nodes(through_phase)
    if not active:
        raise AcceptanceVerificationError(
            "the selected acceptance inventory has no active nodes"
        )
    node_ids = tuple(node.node_id for node in active)
    _validate_execution_scope(path, node_ids, root)
    for node_id in node_ids:
        _validate_test_implementation(node_id, root)
    collection = _run_probe(
        _COLLECT_DRIVER,
        _COLLECT_SENTINEL,
        node_ids,
        root,
    )
    _verify_collection(node_ids, collection)
    execution = _run_probe(
        _EXECUTE_DRIVER,
        _EXECUTE_SENTINEL,
        node_ids,
        root,
    )
    _verify_execution(node_ids, execution)
    return manifest


def _strict_mapping(path: Path, label: str) -> dict[str, object]:
    try:
        value = strict_json_path(path)
    except StrictJsonError as exc:
        raise AcceptanceVerificationError(
            f"cannot read {label}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AcceptanceVerificationError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _header(payload: dict[str, object], label: str) -> None:
    if (
        type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
    ):
        raise AcceptanceVerificationError(
            f"{label} schema_version must be the integer 1"
        )
    if payload.get("feature") != _FEATURE:
        raise AcceptanceVerificationError(
            f"{label} feature must be {_FEATURE}"
        )


def _acceptance_node(raw: object, current_phase: int) -> AcceptanceNode:
    if not isinstance(raw, dict):
        raise AcceptanceVerificationError("acceptance node must be an object")
    item = cast(dict[str, object], raw)
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
    identifier = _nonempty_string(item.get("id"), "acceptance node id")
    category = _nonempty_string(item.get("category"), "acceptance category")
    if category not in _CATEGORIES:
        raise AcceptanceVerificationError(
            f"unknown acceptance category: {category}"
        )
    lifecycle = _nonempty_string(item.get("lifecycle"), "node lifecycle")
    if lifecycle not in {"planned", "active"}:
        raise AcceptanceVerificationError(
            f"invalid acceptance lifecycle: {lifecycle}"
        )
    active_from_phase = _phase(
        item.get("active_from_phase"), "active_from_phase"
    )
    expected_lifecycle = (
        "active" if active_from_phase <= current_phase else "planned"
    )
    if lifecycle != expected_lifecycle:
        raise AcceptanceVerificationError(
            f"acceptance lifecycle regression for {identifier}: "
            f"expected {expected_lifecycle}, observed {lifecycle}"
        )
    requirement_ids = _string_list(
        item.get("requirement_ids"), "requirement_ids"
    )
    _unique(requirement_ids, f"requirement ID on {identifier}")
    node_id = _node_id(item.get("node_id"))
    return AcceptanceNode(
        id=identifier,
        category=category,
        lifecycle=lifecycle,
        active_from_phase=active_from_phase,
        requirement_ids=requirement_ids,
        node_id=node_id,
    )


def _activation_history(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> None:
    if not isinstance(raw, list) or len(raw) != current_phase + 1:
        raise AcceptanceVerificationError(
            "activation history must contain every implemented phase"
        )
    observed: list[str] = []
    for expected_phase, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise AcceptanceVerificationError(
                "activation history entries must be objects"
            )
        item = cast(dict[str, object], entry)
        _exact_keys(item, {"phase", "node_ids"}, "activation history entry")
        phase = _phase(item.get("phase"), "activation history phase")
        if phase != expected_phase:
            raise AcceptanceVerificationError(
                "activation history phases must be contiguous"
            )
        node_ids = _string_list(item.get("node_ids"), "activation node_ids")
        _unique(node_ids, f"activation nodes at phase {phase}")
        expected = tuple(
            node.id for node in nodes if node.active_from_phase == phase
        )
        if set(node_ids) != set(expected) or len(node_ids) != len(expected):
            raise AcceptanceVerificationError(
                f"activation history mismatch at phase {phase}"
            )
        observed.extend(node_ids)
    active_ids = [node.id for node in nodes if node.lifecycle == "active"]
    if set(observed) != set(active_ids) or len(observed) != len(active_ids):
        raise AcceptanceVerificationError(
            "activation history does not exactly preserve active nodes"
        )


def _activation_snapshots(
    raw_snapshots: object,
    raw_replacements: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> None:
    if (
        not isinstance(raw_snapshots, list)
        or len(raw_snapshots) != current_phase + 1
    ):
        raise AcceptanceVerificationError(
            "activation snapshots must preserve every implemented phase"
        )
    if not isinstance(raw_replacements, list):
        raise AcceptanceVerificationError("replacements must be a list")
    replacements: dict[str, tuple[str, ...]] = {}
    replacement_phases: dict[str, int] = {}
    replacement_requirements: dict[str, frozenset[str]] = {}
    replacement_targets: set[str] = set()
    node_by_id = {node.node_id: node for node in nodes}
    for raw in raw_replacements:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "acceptance replacement must be an object"
            )
        replacement = cast(dict[str, object], raw)
        _exact_keys(
            replacement,
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
        phase = _phase(replacement.get("phase"), "replacement phase")
        if phase > current_phase:
            raise AcceptanceVerificationError(
                "replacement phase is not implemented"
            )
        old_node_id = _node_id(replacement.get("old_node_id"))
        if old_node_id in replacements:
            raise AcceptanceVerificationError(
                f"acceptance node is replaced more than once: {old_node_id}"
            )
        replacement_ids = _string_list(
            replacement.get("replacement_node_ids"),
            "replacement node_ids",
        )
        _unique(replacement_ids, "replacement node ID")
        for node_id in replacement_ids:
            _node_id(node_id)
            if node_id in replacement_targets:
                raise AcceptanceVerificationError(
                    f"replacement target is reused: {node_id}"
                )
            replacement_targets.add(node_id)
        requirement_ids = _string_list(
            replacement.get("requirement_ids"),
            "replacement requirement_ids",
        )
        _unique(requirement_ids, "replacement requirement ID")
        if not set(requirement_ids) <= _EXPECTED_REQUIREMENT_IDS:
            raise AcceptanceVerificationError(
                f"replacement owns unknown requirements: {old_node_id}"
            )
        _nonempty_string(
            replacement.get("reviewed_by"), "replacement reviewed_by"
        )
        _nonempty_string(replacement.get("evidence"), "replacement evidence")
        replacements[old_node_id] = replacement_ids
        replacement_phases[old_node_id] = phase
        replacement_requirements[old_node_id] = frozenset(requirement_ids)
    ledger_digest = sha256(
        dumps(
            {
                "activation_snapshots": raw_snapshots,
                "replacements": raw_replacements,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if ledger_digest != _EXPECTED_ACCEPTANCE_LEDGER_SHA256:
        raise AcceptanceVerificationError(
            "acceptance activation ledger changed without verifier review"
        )
    snapshots: list[tuple[str, ...]] = []
    for expected_phase, raw in enumerate(raw_snapshots):
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "activation snapshot must be an object"
            )
        snapshot = cast(dict[str, object], raw)
        _exact_keys(
            snapshot, {"phase", "node_ids", "sha256"}, "activation snapshot"
        )
        if _phase(snapshot.get("phase"), "snapshot phase") != expected_phase:
            raise AcceptanceVerificationError(
                "activation snapshot phases must be contiguous"
            )
        node_ids = _string_list(snapshot.get("node_ids"), "snapshot node_ids")
        _unique(node_ids, f"snapshot node ID at phase {expected_phase}")
        for node_id in node_ids:
            _node_id(node_id)
        digest = _nonempty_string(snapshot.get("sha256"), "snapshot SHA-256")
        calculated = sha256("\n".join(node_ids).encode("utf-8")).hexdigest()
        if digest != calculated:
            raise AcceptanceVerificationError(
                "activation snapshot digest mismatch at phase"
                f" {expected_phase}"
            )
        snapshots.append(node_ids)

    replacements_by_phase = {
        phase: {
            old_node_id
            for old_node_id, replacement_phase in replacement_phases.items()
            if replacement_phase == phase
        }
        for phase in range(current_phase + 1)
    }
    targets_by_phase = {
        phase: {
            target
            for old_node_id in replacements_by_phase[phase]
            for target in replacements[old_node_id]
        }
        for phase in range(current_phase + 1)
    }
    previous: set[str] = set()
    all_snapshot_ids: set[str] = set()
    for phase, snapshot_ids in enumerate(snapshots):
        current = set(snapshot_ids)
        added = current - previous
        removed = previous - current
        expected_removed = replacements_by_phase[phase]
        if removed != expected_removed:
            raise AcceptanceVerificationError(
                "activation snapshot removals lack exact reviewed tombstones"
                f" at phase {phase}: expected={sorted(expected_removed)},"
                f" observed={sorted(removed)}"
            )
        missing_targets = targets_by_phase[phase] - added
        if missing_targets:
            raise AcceptanceVerificationError(
                "replacement targets are not same-phase snapshot additions:"
                f" phase={phase}, missing={sorted(missing_targets)}"
            )
        expected_current_additions = {
            node.node_id
            for node in nodes
            if node.lifecycle == "active" and node.active_from_phase == phase
        }
        missing_current = expected_current_additions - added
        if missing_current:
            raise AcceptanceVerificationError(
                "active nodes are absent from their activation snapshot:"
                f" phase={phase}, missing={sorted(missing_current)}"
            )
        for node_id in added:
            node = node_by_id.get(node_id)
            if node is not None:
                if (
                    node.lifecycle != "active"
                    or node.active_from_phase != phase
                ):
                    raise AcceptanceVerificationError(
                        "snapshot node was added outside its activation phase:"
                        f" {node_id}"
                    )
            elif node_id not in replacements:
                raise AcceptanceVerificationError(
                    "historical snapshot node lacks a later tombstone:"
                    f" {node_id}"
                )
        previous = current
        all_snapshot_ids.update(current)

    historical_only = all_snapshot_ids - set(node_by_id)
    if not historical_only <= set(replacements):
        raise AcceptanceVerificationError(
            "historical acceptance nodes lack reviewed tombstones"
        )
    for old_node_id, replacement_ids in replacements.items():
        phase = replacement_phases[old_node_id]
        if phase == 0 or old_node_id not in set(snapshots[phase - 1]):
            raise AcceptanceVerificationError(
                "replacement old node was not active immediately before its"
                f" tombstone: {old_node_id}"
            )
        if old_node_id in set(snapshots[phase]):
            raise AcceptanceVerificationError(
                f"replacement old node remains active: {old_node_id}"
            )
        target_requirements: set[str] = set()
        for target_id in replacement_ids:
            target = node_by_id.get(target_id)
            if target is not None:
                if target.active_from_phase != phase:
                    raise AcceptanceVerificationError(
                        "replacement target activated in another phase:"
                        f" {target_id}"
                    )
                target_requirements.update(target.requirement_ids)
                continue
            target_phase = replacement_phases.get(target_id)
            if target_phase is None or target_phase <= phase:
                raise AcceptanceVerificationError(
                    "replacement chain is cyclic or lacks a later tombstone:"
                    f" {target_id}"
                )
            target_requirements.update(replacement_requirements[target_id])
        if target_requirements != set(replacement_requirements[old_node_id]):
            raise AcceptanceVerificationError(
                "replacement does not exactly preserve requirements:"
                f" {old_node_id}"
            )

    active = tuple(
        node.node_id for node in nodes if node.lifecycle == "active"
    )
    if snapshots[-1] != active:
        raise AcceptanceVerificationError(
            "latest activation snapshot differs from active inventory"
        )


def _validate_contract_fixtures(
    manifest: AcceptanceManifest,
    fixtures: Path,
    root: Path,
) -> None:
    decision_surfaces, public_envelopes = _validate_decisions(
        fixtures / "contract_decisions.json"
    )
    requirements = _validate_requirements(
        fixtures / "requirements_traceability.json", manifest
    )
    _validate_failure_matrix(
        fixtures / "failure_matrix.json",
        manifest,
        requirements,
        decision_surfaces,
        public_envelopes,
    )
    _validate_type_manifest(
        fixtures / "type_contract_manifest.json",
        manifest.current_phase,
        root,
    )
    _validate_no_bc(fixtures / "no_bc_removals.json")
    _validate_evidence(fixtures / "baseline_evidence.json", manifest, root)


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
            "catalog_sha256",
            "requirements",
        },
        "requirements traceability",
    )
    _header(payload, "requirements traceability")
    sections = _string_list(payload.get("source_sections"), "source_sections")
    expected_sections = tuple(str(value) for value in range(7, 27))
    if sections != expected_sections:
        raise AcceptanceVerificationError(
            "requirements source sections must be the frozen 7 through 26"
            " inventory"
        )
    raw_requirements = payload.get("requirements")
    if not isinstance(raw_requirements, list):
        raise AcceptanceVerificationError("requirements must be a list")
    manifest_by_node = {node.node_id: node for node in manifest.nodes}
    requirement_nodes: dict[str, tuple[str, ...]] = {}
    observed_ids: list[str] = []
    mapped_nodes: set[str] = set()
    for raw in raw_requirements:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError("requirement must be an object")
        item = cast(dict[str, object], raw)
        _exact_keys(
            item,
            {
                "id",
                "source_section",
                "normative_level",
                "paraphrase",
                "owner",
                "implementation_artifacts",
                "test_node_ids",
            },
            "requirement",
        )
        requirement_id = _nonempty_string(item.get("id"), "requirement id")
        observed_ids.append(requirement_id)
        _nonempty_string(
            item.get("source_section"), "requirement source_section"
        )
        level = _nonempty_string(
            item.get("normative_level"), "normative_level"
        )
        if level not in {"MUST", "SHOULD", "MAY", "SCENARIO"}:
            raise AcceptanceVerificationError(
                f"invalid normative level for {requirement_id}: {level}"
            )
        _nonempty_string(item.get("paraphrase"), "requirement paraphrase")
        _nonempty_string(item.get("owner"), "requirement owner")
        _string_list(
            item.get("implementation_artifacts"), "implementation_artifacts"
        )
        node_ids = _string_list(item.get("test_node_ids"), "test_node_ids")
        _unique(node_ids, f"test node for {requirement_id}")
        requirement_nodes[requirement_id] = node_ids
        for node_id in node_ids:
            node = manifest_by_node.get(node_id)
            if node is None:
                raise AcceptanceVerificationError(
                    f"unmapped requirement node {node_id} for {requirement_id}"
                )
            if requirement_id not in node.requirement_ids:
                raise AcceptanceVerificationError(
                    f"non-reciprocal requirement mapping: {requirement_id},"
                    f" {node_id}"
                )
            mapped_nodes.add(node_id)
    _unique(observed_ids, "requirement ID")
    _verify_digest(
        raw_requirements,
        payload.get("catalog_sha256"),
        _EXPECTED_REQUIREMENTS_SHA256,
        "requirements catalog",
    )
    if frozenset(observed_ids) != _EXPECTED_REQUIREMENT_IDS:
        missing = sorted(_EXPECTED_REQUIREMENT_IDS - frozenset(observed_ids))
        unexpected = sorted(
            frozenset(observed_ids) - _EXPECTED_REQUIREMENT_IDS
        )
        raise AcceptanceVerificationError(
            f"requirements inventory mismatch: missing={missing},"
            f" unexpected={unexpected}"
        )
    for node in manifest.nodes:
        reciprocal = tuple(
            requirement_id
            for requirement_id, node_ids in requirement_nodes.items()
            if node.node_id in node_ids
        )
        if set(reciprocal) != set(node.requirement_ids) or len(
            reciprocal
        ) != len(node.requirement_ids):
            raise AcceptanceVerificationError(
                "acceptance node requirement mapping is not exact:"
                f" {node.node_id}"
            )
        for requirement_id in node.requirement_ids:
            if requirement_id not in _EXPECTED_REQUIREMENT_IDS:
                raise AcceptanceVerificationError(
                    "acceptance node owns unknown requirement:"
                    f" {requirement_id}"
                )
        if node.node_id not in mapped_nodes:
            raise AcceptanceVerificationError(
                "acceptance node has no reciprocal requirement:"
                f" {node.node_id}"
            )
    return frozenset(observed_ids)


def _validate_failure_matrix(
    path: Path,
    manifest: AcceptanceManifest,
    requirements: frozenset[str],
    decision_surfaces: frozenset[str],
    public_envelopes: frozenset[str],
) -> None:
    payload = _strict_mapping(path, "failure matrix")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "matrix_sha256",
            "observation_window",
            "domain_side_effect_scope",
            "surfaces",
            "conditions",
            "cells",
        },
        "failure matrix",
    )
    _header(payload, "failure matrix")
    _nonempty_string(
        payload.get("observation_window"),
        "failure observation_window",
    )
    _nonempty_string(
        payload.get("domain_side_effect_scope"),
        "failure domain_side_effect_scope",
    )
    surfaces = _record_ids(
        payload.get("surfaces"),
        {"id", "description", "active_from_phase"},
        "failure surface",
    )
    if frozenset(surfaces) != decision_surfaces:
        raise AcceptanceVerificationError(
            "failure surfaces differ from the public capability inventory"
        )
    conditions = _record_ids(
        payload.get("conditions"),
        {"id", "description", "requirement_id", "active_from_phase"},
        "failure condition",
    )
    if frozenset(conditions) != _EXPECTED_FAILURE_CONDITIONS:
        raise AcceptanceVerificationError(
            "failure condition inventory must contain all fifteen conditions"
        )
    raw_conditions = cast(list[dict[str, object]], payload["conditions"])
    condition_phases: dict[str, int] = {}
    condition_requirements: dict[str, str] = {}
    for condition in raw_conditions:
        requirement_id = _nonempty_string(
            condition.get("requirement_id"), "failure requirement_id"
        )
        if requirement_id not in requirements:
            raise AcceptanceVerificationError(
                f"failure condition owns unknown requirement: {requirement_id}"
            )
        condition_id = _nonempty_string(condition.get("id"), "condition id")
        condition_phases[condition_id] = _phase(
            condition.get("active_from_phase"),
            "condition active_from_phase",
        )
        condition_requirements[condition_id] = requirement_id
    raw_surfaces = cast(list[dict[str, object]], payload["surfaces"])
    surface_phases = {
        _nonempty_string(item.get("id"), "surface id"): _phase(
            item.get("active_from_phase"), "surface active_from_phase"
        )
        for item in raw_surfaces
    }
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        raise AcceptanceVerificationError("failure cells must be a list")
    expected_cells = {
        (condition, surface)
        for condition in conditions
        for surface in surfaces
    }
    observed_cells: set[tuple[str, str]] = set()
    manifest_nodes = {node.node_id: node for node in manifest.nodes}
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError("failure cell must be an object")
        cell = cast(dict[str, object], raw)
        _exact_keys(
            cell,
            {
                "condition_id",
                "surface_id",
                "applicable",
                "active_from_phase",
                "expected_transition",
                "public_result",
                "status_or_exit",
                "provider_call_count",
                "domain_side_effect_count",
                "negative_e2e_node",
                "non_applicability_reason",
                "reviewed_by",
            },
            "failure cell",
        )
        condition_id = _nonempty_string(
            cell.get("condition_id"), "condition_id"
        )
        surface_id = _nonempty_string(cell.get("surface_id"), "surface_id")
        key = (condition_id, surface_id)
        if key in observed_cells:
            raise AcceptanceVerificationError(f"duplicate failure cell: {key}")
        observed_cells.add(key)
        if key not in expected_cells:
            raise AcceptanceVerificationError(f"unknown failure cell: {key}")
        active_from_phase = _phase(
            cell.get("active_from_phase"), "failure active_from_phase"
        )
        expected_phase = max(
            surface_phases[surface_id],
            condition_phases[condition_id],
        )
        if active_from_phase != expected_phase:
            raise AcceptanceVerificationError(
                f"failure cell activation differs from its surface: {key}"
            )
        applicable = cell.get("applicable")
        if type(applicable) is not bool:
            raise AcceptanceVerificationError(
                f"failure applicability must be boolean: {key}"
            )
        transition = _nonempty_string(
            cell.get("expected_transition"), "expected_transition"
        )
        public_result = _nonempty_string(
            cell.get("public_result"), "public_result"
        )
        status_or_exit = _nonempty_string(
            cell.get("status_or_exit"), "status_or_exit"
        )
        provider_calls = _nonnegative_int(
            cell.get("provider_call_count"), "provider_call_count"
        )
        side_effects = _nonnegative_int(
            cell.get("domain_side_effect_count"), "domain_side_effect_count"
        )
        node_id = cell.get("negative_e2e_node")
        reason = cell.get("non_applicability_reason")
        reviewed_by = cell.get("reviewed_by")
        if key == ("INPUT-F-15", "mcp-inbound-task") and cell != {
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
        }:
            raise AcceptanceVerificationError(
                "MCP capability-absent fallback must preserve ordinary"
                " execution"
            )
        if applicable:
            for label, value in (
                ("expected_transition", transition),
                ("public_result", public_result),
                ("status_or_exit", status_or_exit),
            ):
                _unambiguous_outcome(value, label, key)
            transition_match = _TRANSITION_PATTERN.fullmatch(transition)
            if (
                transition_match is None
                or not set(transition_match.groups()) <= _INTERACTION_STATES
            ):
                raise AcceptanceVerificationError(
                    "failure transition is not a canonical interaction-state"
                    f" transition: {key}"
                )
            public_result_match = _PUBLIC_RESULT_PATTERN.fullmatch(
                public_result
            )
            if (
                public_result_match is None
                or public_result_match.group(1) not in public_envelopes
            ):
                raise AcceptanceVerificationError(
                    "failure public result is not a cataloged literal"
                    f" envelope: {key}"
                )
            status_match = _STATUS_OR_EXIT_PATTERN.fullmatch(status_or_exit)
            if (
                status_match is None
                or status_match.group(1) not in _STATUS_OR_EXIT_KEYS
            ):
                raise AcceptanceVerificationError(
                    f"failure status or exit is not one machine literal: {key}"
                )
            if side_effects != 0:
                raise AcceptanceVerificationError(
                    f"failure cell permits a domain side effect: {key}"
                )
            exact_node = _node_id(node_id)
            manifest_node = manifest_nodes.get(exact_node)
            if manifest_node is None or manifest_node.category != "public_e2e":
                raise AcceptanceVerificationError(
                    f"failure cell lacks an owned negative E2E node: {key}"
                )
            if (
                condition_requirements[condition_id]
                not in manifest_node.requirement_ids
            ):
                raise AcceptanceVerificationError(
                    f"failure cell node does not own its requirement: {key}"
                )
            if manifest_node.active_from_phase != active_from_phase:
                raise AcceptanceVerificationError(
                    "failure cell node activation does not match its cell:"
                    f" {key}"
                )
            if reason is not None:
                raise AcceptanceVerificationError(
                    "applicable failure cell has a non-applicability reason:"
                    f" {key}"
                )
            if reviewed_by is not None:
                raise AcceptanceVerificationError(
                    f"applicable failure cell has an N/A reviewer: {key}"
                )
            if (
                transition == "not_applicable"
                or public_result == "none"
                or status_or_exit == "not_applicable"
            ):
                raise AcceptanceVerificationError(
                    f"applicable failure cell has placeholder behavior: {key}"
                )
        else:
            if (
                node_id is not None
                or transition != "not_applicable"
                or public_result != "none"
                or status_or_exit != "not_applicable"
                or provider_calls != 0
                or side_effects != 0
            ):
                raise AcceptanceVerificationError(
                    "non-applicable failure cell has executable behavior:"
                    f" {key}"
                )
            concrete = _nonempty_string(reason, "non_applicability_reason")
            normalized = concrete.lower().replace("_", " ").strip(" .")
            if (
                len(concrete) < 80
                or surface_id not in concrete
                or condition_id not in concrete
                or normalized in {"n/a", "na", "not applicable"}
                or "another lifecycle owner" in normalized
                or "declared ownership cannot exercise" in normalized
            ):
                raise AcceptanceVerificationError(
                    "non-applicable failure cell lacks a concrete reviewed"
                    f" reason: {key}"
                )
            reviewer = _nonempty_string(reviewed_by, "reviewed_by")
            reviewer_label = reviewer.lower().replace("_", "-")
            if any(
                claim in reviewer_label
                for claim in ("audit", "auditor", "independent")
            ):
                raise AcceptanceVerificationError(
                    "N/A reviewer claims unrecorded independent approval:"
                    f" {key}"
                )
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        raise AcceptanceVerificationError(
            f"failure matrix is incomplete: missing={missing}"
        )
    _verify_digest(
        {
            "observation_window": payload["observation_window"],
            "domain_side_effect_scope": payload["domain_side_effect_scope"],
            "surfaces": payload["surfaces"],
            "conditions": payload["conditions"],
            "cells": payload["cells"],
        },
        payload.get("matrix_sha256"),
        _EXPECTED_FAILURE_MATRIX_SHA256,
        "failure matrix",
    )


def _validate_type_manifest(
    path: Path,
    acceptance_phase: int,
    root: Path,
) -> None:
    payload = _strict_mapping(path, "type-contract manifest")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "current_phase",
            "activation_history",
            "activation_snapshots",
            "replacements",
            "fixtures",
        },
        "type-contract manifest",
    )
    _header(payload, "type-contract manifest")
    current_phase = _phase(payload.get("current_phase"), "type current_phase")
    if current_phase != acceptance_phase:
        raise AcceptanceVerificationError(
            "type and acceptance manifests must implement the same phase"
        )
    raw_fixtures = payload.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise AcceptanceVerificationError(
            "type fixtures must be a non-empty list"
        )
    identifiers: list[str] = []
    paths: list[str] = []
    active_ids: list[str] = []
    for raw in raw_fixtures:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError("type fixture must be an object")
        item = cast(dict[str, object], raw)
        _exact_keys(
            item,
            {
                "id",
                "kind",
                "lifecycle",
                "active_from_phase",
                "path",
                "expected_diagnostics",
            },
            "type fixture",
        )
        identifier = _nonempty_string(item.get("id"), "type fixture id")
        identifiers.append(identifier)
        kind = _nonempty_string(item.get("kind"), "type fixture kind")
        if kind not in {"positive", "negative"}:
            raise AcceptanceVerificationError(
                f"invalid type fixture kind: {kind}"
            )
        active_from = _phase(
            item.get("active_from_phase"), "type active_from_phase"
        )
        lifecycle = _nonempty_string(item.get("lifecycle"), "type lifecycle")
        expected = "active" if active_from <= current_phase else "planned"
        if lifecycle != expected:
            raise AcceptanceVerificationError(
                f"type fixture lifecycle regression: {item.get('id')}"
            )
        if lifecycle == "active":
            active_ids.append(identifier)
        raw_path = _nonempty_string(item.get("path"), "type fixture path")
        paths.append(raw_path)
        _type_fixture_path(raw_path, root)
        diagnostics = item.get("expected_diagnostics")
        if not isinstance(diagnostics, list) or not all(
            isinstance(value, str) and value for value in diagnostics
        ):
            raise AcceptanceVerificationError(
                "type expected_diagnostics must be a string list"
            )
        if (kind == "positive") is bool(diagnostics):
            raise AcceptanceVerificationError(
                f"type diagnostics do not match fixture kind: {item.get('id')}"
            )
    _unique(identifiers, "type fixture ID")
    _unique(paths, "type fixture path")
    history = payload.get("activation_history")
    if not isinstance(history, list) or len(history) != current_phase + 1:
        raise AcceptanceVerificationError(
            "type activation history must contain every implemented phase"
        )
    observed_active: list[str] = []
    for expected_phase, raw in enumerate(history):
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "type activation history entry must be an object"
            )
        entry = cast(dict[str, object], raw)
        _exact_keys(entry, {"phase", "fixture_ids"}, "type activation entry")
        if (
            _phase(entry.get("phase"), "type activation phase")
            != expected_phase
        ):
            raise AcceptanceVerificationError(
                "type activation history phases must be contiguous"
            )
        phase_ids = _string_list(entry.get("fixture_ids"), "type fixture_ids")
        expected_ids = tuple(
            cast(str, item["id"])
            for item in cast(list[dict[str, object]], raw_fixtures)
            if item.get("active_from_phase") == expected_phase
        )
        if set(phase_ids) != set(expected_ids) or len(phase_ids) != len(
            expected_ids
        ):
            raise AcceptanceVerificationError(
                f"type activation history mismatch at phase {expected_phase}"
            )
        observed_active.extend(phase_ids)
    if set(observed_active) != set(active_ids) or len(observed_active) != len(
        active_ids
    ):
        raise AcceptanceVerificationError(
            "type activation history does not preserve active fixtures"
        )


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
    _header(payload, "contract decisions")
    for key in required - {"schema_version", "feature", "contract_sha256"}:
        value = payload.get(key)
        if not isinstance(value, (dict, list)) or not value:
            raise AcceptanceVerificationError(
                f"contract decision {key} must be populated"
            )
    activation = cast(dict[str, object], payload["activation"])
    if activation.get("production_default") != "absent":
        raise AcceptanceVerificationError(
            "structured input must remain absent in production"
        )
    content = {
        key: value
        for key, value in payload.items()
        if key != "contract_sha256"
    }
    _verify_digest(
        content,
        payload.get("contract_sha256"),
        _EXPECTED_DECISIONS_SHA256,
        "contract decisions",
    )
    _validate_protocol_decision_shapes(payload)
    capability_matrix = cast(dict[str, object], payload["capability_matrix"])
    surface_ids = _string_list(
        capability_matrix.get("public_failure_surface_ids"),
        "public failure surface IDs",
    )
    _unique(surface_ids, "public failure surface ID")
    error_status = cast(dict[str, object], payload["error_status"])
    raw_catalog = error_status.get("public_envelope_catalog")
    if not isinstance(raw_catalog, dict) or not raw_catalog:
        raise AcceptanceVerificationError(
            "public envelope catalog must be a non-empty object"
        )
    envelope_ids: list[str] = []
    for raw_id, envelope in raw_catalog.items():
        if (
            not isinstance(raw_id, str)
            or _PUBLIC_RESULT_PATTERN.fullmatch(f"envelope={raw_id}") is None
            or not isinstance(envelope, dict)
            or not envelope
        ):
            raise AcceptanceVerificationError(
                "public envelope catalog entry has an invalid literal shape"
            )
        envelope_ids.append(raw_id)
    _unique(envelope_ids, "public envelope catalog ID")
    return frozenset(surface_ids), frozenset(envelope_ids)


def _validate_protocol_decision_shapes(payload: dict[str, object]) -> None:
    projection = _decision_mapping(
        payload.get("protocol_projection"), "protocol"
    )
    a2a = _decision_mapping(projection.get("a2a"), "A2A")
    expected_task_states = [
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
    if a2a.get("task_states") != expected_task_states:
        raise AcceptanceVerificationError(
            "A2A task states must contain the complete ordered 1.0 enum"
        )
    error_status = _decision_mapping(
        payload.get("error_status"), "error status"
    )
    a2a_errors = _decision_mapping(error_status.get("a2a"), "A2A errors")
    core_errors = _decision_mapping(a2a_errors.get("core"), "A2A core errors")
    if (
        core_errors.get("push_notification_not_supported") != -32003
        or core_errors.get("version_not_supported") != -32009
    ):
        raise AcceptanceVerificationError(
            "A2A reserved core error codes are incorrect"
        )

    mcp = _decision_mapping(projection.get("mcp"), "MCP")
    elicitation = _decision_mapping(mcp.get("elicitation"), "MCP elicitation")
    requested = _decision_mapping(
        elicitation.get("requestedSchema"), "MCP requestedSchema"
    )
    _exact_keys(
        requested,
        {"allowed_top_level_keys", "type", "properties", "required"},
        "MCP requestedSchema",
    )
    if requested.get("allowed_top_level_keys") != [
        "$schema",
        "type",
        "properties",
        "required",
    ]:
        raise AcceptanceVerificationError(
            "MCP requestedSchema top-level keys are incorrect"
        )
    requested_type = _decision_mapping(
        requested.get("type"), "MCP requestedSchema type"
    )
    if (
        set(requested_type) != {"const"}
        or requested_type.get("const") != "object"
    ):
        raise AcceptanceVerificationError(
            "MCP requestedSchema type must be the object literal"
        )
    properties = _decision_mapping(
        requested.get("properties"), "MCP requestedSchema properties"
    )
    _exact_keys(
        properties,
        {"shape", "primitive_types", "single_select", "multiple_select"},
        "MCP requestedSchema properties",
    )
    if (
        properties.get("primitive_types")
        != [
            "string",
            "number",
            "integer",
            "boolean",
        ]
        or requested.get("required")
        != "optional array of unique property names"
    ):
        raise AcceptanceVerificationError(
            "MCP requestedSchema primitive or required fields are incorrect"
        )
    single = _decision_mapping(
        properties.get("single_select"), "MCP single-select schema"
    )
    _exact_keys(single, {"type", "enum"}, "MCP single-select schema")
    if single != {
        "type": "string",
        "enum": "non-empty unique stable string values",
    }:
        raise AcceptanceVerificationError(
            "MCP single-select schema must use a string enum"
        )
    multiple = _decision_mapping(
        properties.get("multiple_select"), "MCP multiple-select schema"
    )
    _exact_keys(
        multiple,
        {"type", "items", "uniqueItems"},
        "MCP multiple-select schema",
    )
    multiple_items = _decision_mapping(
        multiple.get("items"), "MCP multiple-select items"
    )
    _exact_keys(
        multiple_items,
        {"type", "enum"},
        "MCP multiple-select items",
    )
    if (
        multiple.get("type") != "array"
        or multiple.get("uniqueItems") is not True
        or multiple_items.get("type") != "string"
        or multiple_items.get("enum")
        != "non-empty unique stable string values"
    ):
        raise AcceptanceVerificationError(
            "MCP multiple-select schema must use a unique enum array"
        )

    tasks = _decision_mapping(mcp.get("tasks"), "MCP tasks")
    params = _decision_mapping(
        tasks.get("params_task_schema"), "MCP task params"
    )
    params_properties = _decision_mapping(
        params.get("properties"), "MCP task params properties"
    )
    _exact_keys(
        params,
        {"$schema", "type", "additionalProperties", "properties"},
        "MCP task params",
    )
    _exact_keys(params_properties, {"ttl"}, "MCP task params properties")
    ttl = _decision_mapping(params_properties.get("ttl"), "MCP task TTL")
    if (
        set(ttl) != {"type", "unit"}
        or params.get("type") != "object"
        or params.get("additionalProperties") is not False
        or ttl.get("unit") != "milliseconds"
        or ttl.get("type") != "number"
        or tasks.get("ttl_mapping")
        != "canonical continuation TTL seconds multiplied by 1000 without"
        " rounding"
    ):
        raise AcceptanceVerificationError(
            "MCP task TTL must use exact milliseconds"
        )
    create_result = _decision_mapping(
        tasks.get("CreateTaskResult"), "MCP CreateTaskResult"
    )
    result_properties = _decision_mapping(
        create_result.get("properties"), "MCP CreateTaskResult properties"
    )
    _exact_keys(
        create_result,
        {
            "$schema",
            "type",
            "additionalProperties",
            "required",
            "properties",
        },
        "MCP CreateTaskResult",
    )
    _exact_keys(
        result_properties,
        {"task", "_meta"},
        "MCP CreateTaskResult properties",
    )
    required = create_result.get("required")
    if (
        create_result.get("type") != "object"
        or required != ["task"]
        or "_meta" in cast(list[object], required)
        or create_result.get("additionalProperties") is not True
        or result_properties.get("_meta") != {"type": "object"}
    ):
        raise AcceptanceVerificationError(
            "MCP CreateTaskResult must permit optional _meta and extensions"
        )
    task_schema = _decision_mapping(tasks.get("task_schema"), "MCP Task")
    task_properties = _decision_mapping(
        task_schema.get("properties"), "MCP Task properties"
    )
    _exact_keys(
        task_schema,
        {
            "$schema",
            "type",
            "additionalProperties",
            "required",
            "properties",
        },
        "MCP Task",
    )
    expected_task_properties = {
        "taskId",
        "status",
        "statusMessage",
        "createdAt",
        "lastUpdatedAt",
        "ttl",
        "pollInterval",
    }
    task_id = _decision_mapping(task_properties.get("taskId"), "MCP taskId")
    task_status = _decision_mapping(
        task_properties.get("status"), "MCP task status"
    )
    status_message = _decision_mapping(
        task_properties.get("statusMessage"), "MCP task statusMessage"
    )
    created_at = _decision_mapping(
        task_properties.get("createdAt"), "MCP task createdAt"
    )
    last_updated_at = _decision_mapping(
        task_properties.get("lastUpdatedAt"), "MCP task lastUpdatedAt"
    )
    task_ttl = _decision_mapping(task_properties.get("ttl"), "MCP Task ttl")
    poll_interval = _decision_mapping(
        task_properties.get("pollInterval"), "MCP Task pollInterval"
    )
    if (
        set(task_properties) != expected_task_properties
        or task_schema.get("type") != "object"
        or task_schema.get("additionalProperties") is not False
        or task_schema.get("required")
        != ["taskId", "status", "createdAt", "lastUpdatedAt", "ttl"]
        or task_id != {"type": "string", "minLength": 1}
        or task_status
        != {
            "enum": [
                "working",
                "input_required",
                "completed",
                "failed",
                "cancelled",
            ]
        }
        or status_message != {"type": "string"}
        or created_at != {"type": "string", "format": "date-time"}
        or last_updated_at != {"type": "string", "format": "date-time"}
        or task_ttl != {"type": ["number", "null"]}
        or poll_interval != {"type": "number"}
        or result_properties.get("task")
        != {
            key: value
            for key, value in task_schema.items()
            if key != "$schema"
        }
    ):
        raise AcceptanceVerificationError(
            "MCP Task schema differs from the complete protocol contract"
        )
    if (
        tasks.get("request_type_task_capability_absent")
        != "receiver MUST process request normally and ignore params.task"
        " augmentation"
    ):
        raise AcceptanceVerificationError(
            "MCP request-type task capability fallback is incorrect"
        )
    generic_requirement = _decision_mapping(
        tasks.get("generic_receiver_task_requirement"),
        "MCP generic receiver task requirement",
    )
    if generic_requirement != {
        "omission_behavior": (
            "receiver MAY require task augmentation for a request type with"
            " declared support; omission MAY return -32600"
        ),
        "omission_error": -32600,
    }:
        raise AcceptanceVerificationError(
            "MCP generic receiver task requirement is incorrect"
        )
    tool_support = _decision_mapping(
        tasks.get("tool_execution_task_support"),
        "MCP tool execution task support",
    )
    if tool_support != {
        "absent": (
            "defaults to forbidden; attempted params.task SHOULD return -32601"
        ),
        "forbidden": "attempted params.task SHOULD return -32601",
        "optional": "client MAY invoke normally or with params.task",
        "required": (
            "client MUST invoke with params.task; omission MUST return -32601"
        ),
    }:
        raise AcceptanceVerificationError(
            "MCP tool execution task-support behavior is incorrect"
        )
    if tasks.get("initial_state") != "working":
        raise AcceptanceVerificationError(
            "MCP tasks must begin in the working state"
        )
    if tasks.get("legal_transitions") != [
        ["working", "input_required"],
        ["input_required", "working"],
        ["input_required", "completed"],
        ["input_required", "failed"],
        ["working", "completed"],
        ["working", "failed"],
        ["working", "cancelled"],
        ["input_required", "cancelled"],
    ]:
        raise AcceptanceVerificationError(
            "MCP task transitions differ from the frozen state graph"
        )
    mcp_errors = _decision_mapping(error_status.get("mcp"), "MCP errors")
    if mcp_errors != {
        "invalid_params": -32602,
        "unavailable": -32001,
        "unauthorized": -32003,
        "conflict": -32009,
        "expired": -32010,
        "receiver_task_augmentation_required": -32600,
        "tool_task_augmentation_forbidden": -32601,
        "tool_task_augmentation_required": -32601,
    }:
        raise AcceptanceVerificationError(
            "MCP receiver and tool task errors are conflated"
        )

    _validate_a2a_message_metadata(a2a)
    _validate_mcp_schema_examples(tasks)
    _validate_public_envelope_contract(error_status)


def _validate_a2a_message_metadata(a2a: dict[str, object]) -> None:
    extension = _decision_mapping(a2a.get("extension"), "A2A extension")
    schema = _decision_mapping(
        extension.get("message_metadata_schema"),
        "A2A message metadata schema",
    )
    schema_digest = _EXPECTED_PROTOCOL_SCHEMA_SHA256["a2a_message_metadata"]
    _verify_digest(
        schema,
        schema_digest,
        schema_digest,
        "A2A message metadata schema",
    )
    examples = _decision_mapping(
        extension.get("message_metadata_examples"),
        "A2A message metadata examples",
    )
    if set(examples) != {"request", "accept", "decline", "cancel"}:
        raise AcceptanceVerificationError(
            "A2A message metadata examples must cover every action"
        )
    _validate_schema_examples(
        schema,
        examples,
        "A2A message metadata",
        exercise_mutations=True,
    )


def _validate_mcp_schema_examples(tasks: dict[str, object]) -> None:
    params_schema = _decision_mapping(
        tasks.get("params_task_schema"), "MCP task params"
    )
    task_schema = _decision_mapping(tasks.get("task_schema"), "MCP Task")
    create_result_schema = _decision_mapping(
        tasks.get("CreateTaskResult"), "MCP CreateTaskResult"
    )
    for schema_name, schema in (
        ("mcp_params_task", params_schema),
        ("mcp_task", task_schema),
        ("mcp_create_task_result", create_result_schema),
    ):
        expected_digest = _EXPECTED_PROTOCOL_SCHEMA_SHA256[schema_name]
        _verify_digest(
            schema,
            expected_digest,
            expected_digest,
            f"protocol schema {schema_name}",
        )
    schemas = {
        "params_task_with_ttl": params_schema,
        "params_task_without_ttl": params_schema,
        "Task": task_schema,
        "CreateTaskResult": create_result_schema,
    }
    examples = _decision_mapping(
        tasks.get("schema_examples"), "MCP schema examples"
    )
    if set(examples) != set(schemas):
        raise AcceptanceVerificationError(
            "MCP schema examples must cover every frozen schema variant"
        )
    for name, schema in schemas.items():
        _validate_schema_examples(
            schema,
            {name: examples[name]},
            f"MCP {name}",
            exercise_mutations=False,
            allow_empty=name == "params_task_without_ttl",
        )
    _validate_mcp_schema_probes(
        params_schema,
        task_schema,
        create_result_schema,
    )


def _validate_mcp_schema_probes(
    params_schema: dict[str, object],
    task_schema: dict[str, object],
    create_result_schema: dict[str, object],
) -> None:
    validator_factory = _draft202012_validator()
    params_validator = validator_factory(params_schema)
    task_validator = validator_factory(task_schema)
    create_result_validator = validator_factory(create_result_schema)
    task = {
        "taskId": "task-0001",
        "status": "working",
        "statusMessage": "",
        "createdAt": "2026-07-20T00:00:00Z",
        "lastUpdatedAt": "2026-07-20T00:00:00Z",
        "ttl": 0,
        "pollInterval": -1,
    }
    if not all(
        params_validator.is_valid(probe)
        for probe in ({}, {"ttl": 0}, {"ttl": -1})
    ):
        raise AcceptanceVerificationError(
            "MCP task params reject a protocol-valid numeric TTL"
        )
    if any(
        params_validator.is_valid(probe)
        for probe in ({"ttl": None}, {"ttl": "1"}, {"extra": True})
    ):
        raise AcceptanceVerificationError(
            "MCP task params accept an invalid task augmentation"
        )
    if not task_validator.is_valid(task):
        raise AcceptanceVerificationError(
            "MCP Task rejects optional fields or protocol-valid bounds"
        )
    minimal_task = {
        key: value
        for key, value in task.items()
        if key not in {"statusMessage", "pollInterval"}
    }
    minimal_task["ttl"] = None
    if not task_validator.is_valid(minimal_task):
        raise AcceptanceVerificationError(
            "MCP Task rejects omitted optional fields or unlimited TTL"
        )
    invalid_tasks = []
    for key, value in (
        ("statusMessage", 1),
        ("pollInterval", None),
        ("ttl", "0"),
    ):
        invalid = deepcopy(task)
        invalid[key] = value
        invalid_tasks.append(invalid)
    missing_required = deepcopy(task)
    del missing_required["taskId"]
    invalid_tasks.append(missing_required)
    task_with_meta = deepcopy(task)
    task_with_meta["_meta"] = {}
    invalid_tasks.append(task_with_meta)
    if any(task_validator.is_valid(probe) for probe in invalid_tasks):
        raise AcceptanceVerificationError(
            "MCP Task accepts an invalid field, type, or omission"
        )
    valid_results = (
        {"task": task},
        {"task": task, "_meta": {}},
        {
            "task": task,
            "_meta": {"vendor": True},
            "vendor_extension": {"enabled": True},
        },
    )
    if not all(
        create_result_validator.is_valid(probe) for probe in valid_results
    ):
        raise AcceptanceVerificationError(
            "MCP CreateTaskResult rejects optional metadata or extensions"
        )
    if any(
        create_result_validator.is_valid(probe)
        for probe in ({}, {"task": task, "_meta": []})
    ):
        raise AcceptanceVerificationError(
            "MCP CreateTaskResult accepts an invalid required field or _meta"
        )


def _validate_public_envelope_contract(
    error_status: dict[str, object],
) -> None:
    catalog = _decision_mapping(
        error_status.get("public_envelope_catalog"),
        "public envelope catalog",
    )
    examples = _decision_mapping(
        error_status.get("public_envelope_examples"),
        "public envelope examples",
    )
    contract = _decision_mapping(
        error_status.get("public_envelope_catalog_contract"),
        "public envelope catalog contract",
    )
    _exact_keys(
        contract,
        {
            "dialect",
            "each_entry",
            "validation",
            "cross_field_invariants",
            "mutation_requirements",
        },
        "public envelope catalog contract",
    )
    if contract.get("dialect") != _JSON_SCHEMA_DIALECT or contract.get(
        "mutation_requirements"
    ) != list(_PUBLIC_SCHEMA_MUTATIONS):
        raise AcceptanceVerificationError(
            "public envelope schema proof requirements are incomplete"
        )
    if set(catalog) != set(examples):
        raise AcceptanceVerificationError(
            "public envelope schemas and representative examples differ"
        )
    for envelope_id, raw_schema in catalog.items():
        if not isinstance(raw_schema, dict):
            raise AcceptanceVerificationError(
                f"public envelope schema is invalid: {envelope_id}"
            )
        _validate_schema_examples(
            cast(dict[str, object], raw_schema),
            {envelope_id: examples[envelope_id]},
            f"public envelope {envelope_id}",
            exercise_mutations=True,
        )
    _validate_public_cross_field_mutations(
        catalog,
        examples,
        error_status.get("public_envelope_cross_field_mutations"),
    )


def _draft202012_validator() -> _JsonSchemaValidatorFactory:
    module = import_module("jsonschema")
    return cast(
        _JsonSchemaValidatorFactory,
        getattr(module, "Draft202012Validator"),
    )


def _validate_schema_examples(
    schema: dict[str, object],
    examples: dict[str, object],
    label: str,
    *,
    exercise_mutations: bool,
    allow_empty: bool = False,
) -> None:
    if schema.get("$schema") != _JSON_SCHEMA_DIALECT:
        raise AcceptanceVerificationError(
            f"{label} must declare the frozen JSON Schema dialect"
        )
    validator_factory = _draft202012_validator()
    try:
        validator_factory.check_schema(schema)
    except Exception as exc:
        raise AcceptanceVerificationError(
            f"{label} is not a valid JSON Schema: {exc}"
        ) from exc
    validator = validator_factory(schema)
    for name, example in examples.items():
        if not isinstance(example, dict) or (not example and not allow_empty):
            raise AcceptanceVerificationError(
                f"{label} example must be a populated object: {name}"
            )
        if not validator.is_valid(example):
            raise AcceptanceVerificationError(
                f"{label} representative example does not validate: {name}"
            )
        if exercise_mutations:
            _exercise_schema_mutations(
                schema,
                cast(dict[str, object], example),
                validator,
                f"{label} example {name}",
            )


def _exercise_schema_mutations(
    schema: dict[str, object],
    example: dict[str, object],
    validator: _JsonSchemaValidator,
    label: str,
) -> None:
    object_schema = _matching_object_schema(schema, example, label)
    required = object_schema.get("required")
    properties = object_schema.get("properties")
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(item, str) and item for item in required)
        or not isinstance(properties, dict)
        or object_schema.get("additionalProperties") is not False
    ):
        raise AcceptanceVerificationError(
            f"{label} lacks a closed required object contract"
        )
    missing = deepcopy(example)
    missing.pop(cast(str, required[0]), None)
    extra = deepcopy(example)
    extra["__unexpected_contract_field__"] = True
    constant_name = next(
        (
            name
            for name, raw_property in properties.items()
            if isinstance(name, str)
            and isinstance(raw_property, dict)
            and "const" in raw_property
        ),
        None,
    )
    if constant_name is None or constant_name not in example:
        raise AcceptanceVerificationError(
            f"{label} lacks a representative constant field"
        )
    raw_constant_schema = properties[constant_name]
    assert isinstance(raw_constant_schema, dict)
    wrong_constant = deepcopy(example)
    wrong_constant[constant_name] = _different_json_value(
        raw_constant_schema["const"]
    )
    mutations = {
        "missing_required_field": missing,
        "extra_field": extra,
        "wrong_const": wrong_constant,
        "wrong_type": [],
    }
    for mutation_name, mutation in mutations.items():
        if validator.is_valid(mutation):
            raise AcceptanceVerificationError(
                f"{label} accepts prohibited {mutation_name} mutation"
            )


def _matching_object_schema(
    schema: dict[str, object],
    example: dict[str, object],
    label: str,
) -> dict[str, object]:
    if schema.get("type") == "object":
        return schema
    raw_branches = schema.get("oneOf")
    if not isinstance(raw_branches, list):
        raise AcceptanceVerificationError(
            f"{label} must be an object schema or an object union"
        )
    matches: list[dict[str, object]] = []
    for raw_branch in raw_branches:
        if not isinstance(raw_branch, dict):
            continue
        branch = deepcopy(cast(dict[str, object], raw_branch))
        branch["$schema"] = _JSON_SCHEMA_DIALECT
        if "$defs" in schema:
            branch["$defs"] = schema["$defs"]
        if _draft202012_validator()(branch).is_valid(example):
            matches.append(cast(dict[str, object], raw_branch))
    if len(matches) != 1 or matches[0].get("type") != "object":
        raise AcceptanceVerificationError(
            f"{label} does not select exactly one object branch"
        )
    return matches[0]


def _different_json_value(value: object) -> object:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str):
        return value + "-invalid"
    if type(value) in {int, float}:
        return cast(int | float, value) + 1
    if value is None:
        return "invalid"
    return {"invalid": True}


def _validate_public_cross_field_mutations(
    catalog: dict[str, object],
    examples: dict[str, object],
    raw_mutations: object,
) -> None:
    if not isinstance(raw_mutations, dict) or set(raw_mutations) != set(
        _PUBLIC_CROSS_FIELD_INVARIANTS
    ):
        raise AcceptanceVerificationError(
            "public cross-field mutation inventory is incomplete"
        )
    for envelope_id, expected_ids in _PUBLIC_CROSS_FIELD_INVARIANTS.items():
        raw_vectors = raw_mutations.get(envelope_id)
        if not isinstance(raw_vectors, list) or not raw_vectors:
            raise AcceptanceVerificationError(
                f"public cross-field mutations are empty: {envelope_id}"
            )
        schema = cast(dict[str, object], catalog[envelope_id])
        example = cast(dict[str, object], examples[envelope_id])
        validator = _draft202012_validator()(schema)
        observed_ids: list[str] = []
        for raw_vector in raw_vectors:
            if not isinstance(raw_vector, dict):
                raise AcceptanceVerificationError(
                    f"public cross-field mutation is invalid: {envelope_id}"
                )
            vector = cast(dict[str, object], raw_vector)
            invariant_id = _nonempty_string(
                vector.get("invariant_id"), "cross-field invariant ID"
            )
            observed_ids.append(invariant_id)
            equals_path: tuple[str, ...] | None = None
            if "expected" in vector:
                _exact_keys(
                    vector,
                    {"invariant_id", "path", "expected", "replacement"},
                    "public expected-value cross-field mutation",
                )
                comparison = vector.get("expected")
            else:
                _exact_keys(
                    vector,
                    {
                        "invariant_id",
                        "path",
                        "equals_path",
                        "replacement",
                    },
                    "public equality cross-field mutation",
                )
                equals_path = _json_path_parts(
                    vector.get("equals_path"), "cross-field equals_path"
                )
                comparison = _json_path(
                    example,
                    equals_path,
                )
            path = _json_path_parts(vector.get("path"), "cross-field path")
            if equals_path is not None and path == equals_path:
                raise AcceptanceVerificationError(
                    f"cross-field mutation compares one field: {invariant_id}"
                )
            original = _json_path(example, path)
            if original != comparison:
                raise AcceptanceVerificationError(
                    f"public cross-field invariant is false: {invariant_id}"
                )
            replacement = vector.get("replacement")
            if replacement == original:
                raise AcceptanceVerificationError(
                    f"public cross-field mutation is a no-op: {invariant_id}"
                )
            mutated = deepcopy(example)
            _replace_json_path(mutated, path, replacement)
            if not validator.is_valid(mutated):
                raise AcceptanceVerificationError(
                    "cross-field mutation must remain schema-valid:"
                    f" {invariant_id}"
                )
            mutated_value = _json_path(mutated, path)
            if mutated_value == comparison:
                raise AcceptanceVerificationError(
                    f"cross-field mutation did not violate: {invariant_id}"
                )
        if set(observed_ids) != expected_ids or len(observed_ids) != len(
            expected_ids
        ):
            raise AcceptanceVerificationError(
                f"public cross-field invariant IDs differ: {envelope_id}"
            )


def _json_path_parts(value: object, label: str) -> tuple[str, ...]:
    parts = _string_list(value, label)
    if not parts:
        raise AcceptanceVerificationError(f"{label} must not be empty")
    return parts


def _json_path(value: object, parts: Sequence[str]) -> object:
    current = value
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise AcceptanceVerificationError(
                f"cross-field JSON path does not exist: {'.'.join(parts)}"
            )
        current = current[part]
    return current


def _replace_json_path(
    value: dict[str, object],
    parts: Sequence[str],
    replacement: object,
) -> None:
    parent = _json_path(value, parts[:-1]) if len(parts) > 1 else value
    if not isinstance(parent, dict):
        raise AcceptanceVerificationError(
            f"cross-field JSON path parent is invalid: {'.'.join(parts)}"
        )
    parent[parts[-1]] = replacement


def _decision_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise AcceptanceVerificationError(
            f"contract decision {label} must be a populated object"
        )
    return cast(dict[str, object], value)


def _validate_no_bc(path: Path) -> None:
    payload = _strict_mapping(path, "no-BC removal inventory")
    _exact_keys(
        payload,
        {"schema_version", "feature", "inventory_sha256", "removals"},
        "no-BC removal inventory",
    )
    _header(payload, "no-BC removal inventory")
    raw = payload.get("removals")
    if not isinstance(raw, list) or len(raw) < 5:
        raise AcceptanceVerificationError(
            "no-BC removal inventory must contain all known replacement paths"
        )
    ids: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise AcceptanceVerificationError(
                "no-BC removal must be an object"
            )
        item = cast(dict[str, object], entry)
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
        ids.append(_nonempty_string(item.get("id"), "no-BC id"))
        _nonempty_string(item.get("current_path"), "no-BC current_path")
        _phase(item.get("remove_by_phase"), "no-BC remove_by_phase")
        _nonempty_string(item.get("replacement"), "no-BC replacement")
        _nonempty_string(item.get("evidence"), "no-BC evidence")
    _unique(ids, "no-BC removal ID")
    if frozenset(ids) != _EXPECTED_NO_BC_IDS:
        raise AcceptanceVerificationError(
            "no-BC removal inventory does not contain the frozen paths"
        )
    _verify_digest(
        raw,
        payload.get("inventory_sha256"),
        _EXPECTED_NO_BC_SHA256,
        "no-BC removal inventory",
    )


def _validate_evidence(
    path: Path,
    manifest: AcceptanceManifest,
    root: Path,
) -> None:
    payload = _strict_mapping(path, "implementation evidence")
    _exact_keys(
        payload,
        {
            "schema_version",
            "feature",
            "recorded_at",
            "implementation_owner",
            "independent_reviewer",
            "active_test_node_ids",
            "git",
            "baseline",
            "boundary",
            "inventory",
            "quality_gate",
            "typing_async_audit",
            "unresolved_risks",
        },
        "implementation evidence",
    )
    _header(payload, "implementation evidence")
    _nonempty_string(payload.get("recorded_at"), "evidence recorded_at")
    implementation_owner = _nonempty_string(
        payload.get("implementation_owner"), "implementation_owner"
    )
    independent_reviewer = _nonempty_string(
        payload.get("independent_reviewer"), "independent_reviewer"
    )
    if (
        implementation_owner != _EXPECTED_IMPLEMENTATION_OWNER
        or independent_reviewer != _EXPECTED_INDEPENDENT_REVIEWER
        or implementation_owner == independent_reviewer
    ):
        raise AcceptanceVerificationError(
            "implementation evidence ownership identities changed"
        )
    active_test_node_ids = _string_list(
        payload.get("active_test_node_ids"), "active_test_node_ids"
    )
    _unique(active_test_node_ids, "active evidence pytest node ID")
    expected_active_test_node_ids = tuple(
        node.node_id for node in manifest.nodes if node.lifecycle == "active"
    )
    if active_test_node_ids != expected_active_test_node_ids:
        raise AcceptanceVerificationError(
            "implementation evidence active pytest nodes differ from the"
            " manifest"
        )

    git = _evidence_mapping(payload.get("git"), "git")
    _exact_keys(
        git,
        {
            "branch",
            "head",
            "head_subject",
            "production_changes_before_baseline",
            "preserved_untracked",
        },
        "implementation evidence git",
    )
    preserved_untracked = _string_list(
        git.get("preserved_untracked"), "preserved untracked paths"
    )
    if (
        git.get("branch") != "input"
        or git.get("head") != _EXPECTED_BASELINE_HEAD
        or git.get("head_subject") != _EXPECTED_BASELINE_SUBJECT
        or git.get("production_changes_before_baseline") != []
        or preserved_untracked != ("docs/examples/skills/code/",)
    ):
        raise AcceptanceVerificationError(
            "implementation evidence git baseline changed"
        )
    if (
        _git_output(
            root,
            "log",
            "-1",
            "--format=%s",
            _EXPECTED_BASELINE_HEAD,
        )
        != _EXPECTED_BASELINE_SUBJECT
        or _git_returncode(
            root,
            "merge-base",
            "--is-ancestor",
            _EXPECTED_BASELINE_HEAD,
            "HEAD",
        )
        != 0
    ):
        raise AcceptanceVerificationError(
            "live git baseline differs from implementation evidence"
        )

    baseline = _evidence_mapping(payload.get("baseline"), "baseline")
    _exact_keys(
        baseline,
        {
            "command",
            "exit_code",
            "collected",
            "passed",
            "skipped",
            "subtests_passed",
            "seconds",
        },
        "implementation evidence baseline",
    )
    baseline_counts = {
        name: _nonnegative_int(baseline.get(name), f"baseline {name}")
        for name in (
            "exit_code",
            "collected",
            "passed",
            "skipped",
            "subtests_passed",
        )
    }
    seconds = baseline.get("seconds")
    if (
        baseline.get("command") != "poetry run pytest --verbose -s"
        or baseline_counts["exit_code"] != 0
        or baseline_counts["collected"]
        != baseline_counts["passed"] + baseline_counts["skipped"]
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or seconds <= 0
    ):
        raise AcceptanceVerificationError(
            "implementation evidence baseline result is inconsistent"
        )

    boundary = _evidence_mapping(payload.get("boundary"), "boundary")
    _exact_keys(
        boundary,
        {
            "production_capability",
            "production_source_changes",
            "changed_paths",
        },
        "implementation evidence boundary",
    )
    changed_paths = _string_list(
        boundary.get("changed_paths"), "boundary changed_paths"
    )
    _unique(changed_paths, "boundary changed path")
    if (
        boundary.get("production_capability") != "absent"
        or boundary.get("production_source_changes") != []
        or frozenset(changed_paths) != _EXPECTED_BOUNDARY_PATHS
        or len(changed_paths) != len(_EXPECTED_BOUNDARY_PATHS)
        or any(
            path == "src" or path.startswith("src/") for path in changed_paths
        )
    ):
        raise AcceptanceVerificationError(
            "implementation evidence crossed the production boundary"
        )
    _validate_live_boundary(
        root,
        changed_paths,
        preserved_untracked,
    )

    failure = _strict_mapping(path.with_name("failure_matrix.json"), "failure")
    raw_surfaces = failure.get("surfaces")
    raw_conditions = failure.get("conditions")
    raw_cells = failure.get("cells")
    if not all(
        isinstance(value, list)
        for value in (raw_surfaces, raw_conditions, raw_cells)
    ):
        raise AcceptanceVerificationError(
            "implementation evidence cannot derive failure counts"
        )
    surfaces = cast(list[object], raw_surfaces)
    conditions = cast(list[object], raw_conditions)
    cells = cast(list[object], raw_cells)
    applicable_cells = len(
        [
            cell
            for cell in cells
            if isinstance(cell, dict) and cell.get("applicable") is True
        ]
    )
    inventory = _evidence_mapping(payload.get("inventory"), "inventory")
    _exact_keys(
        inventory,
        {
            "behavior_requirements",
            "public_scenarios",
            "delivery_requirements",
            "active_acceptance_nodes",
            "planned_acceptance_nodes",
            "failure_conditions",
            "failure_surfaces",
            "failure_cells",
            "applicable_failure_cells",
            "non_applicable_failure_cells",
        },
        "implementation evidence inventory",
    )
    risks = payload.get("unresolved_risks")
    if not isinstance(risks, list) or not all(
        isinstance(item, str) and item for item in risks
    ):
        raise AcceptanceVerificationError(
            "implementation evidence unresolved_risks must be a string list"
        )
    active = len(
        [node for node in manifest.nodes if node.lifecycle == "active"]
    )
    planned = len(
        [node for node in manifest.nodes if node.lifecycle == "planned"]
    )
    expected_inventory = {
        "behavior_requirements": 107,
        "public_scenarios": 12,
        "delivery_requirements": 12,
        "active_acceptance_nodes": active,
        "planned_acceptance_nodes": planned,
        "failure_conditions": len(conditions),
        "failure_surfaces": len(surfaces),
        "failure_cells": len(cells),
        "applicable_failure_cells": applicable_cells,
        "non_applicable_failure_cells": len(cells) - applicable_cells,
    }
    if inventory != expected_inventory or len(cells) != len(surfaces) * len(
        conditions
    ):
        raise AcceptanceVerificationError(
            "implementation evidence inventory counts are stale"
        )

    type_manifest = _strict_mapping(
        path.with_name("type_contract_manifest.json"),
        "type manifest evidence",
    )
    raw_type_fixtures = type_manifest.get("fixtures")
    if not isinstance(raw_type_fixtures, list):
        raise AcceptanceVerificationError(
            "implementation evidence cannot derive type counts"
        )
    type_fixtures = [
        fixture for fixture in raw_type_fixtures if isinstance(fixture, dict)
    ]
    active_type_fixtures = len(
        [
            fixture
            for fixture in type_fixtures
            if fixture.get("lifecycle") == "active"
        ]
    )
    planned_negative_fixtures = len(
        [
            fixture
            for fixture in type_fixtures
            if fixture.get("lifecycle") == "planned"
            and fixture.get("kind") == "negative"
        ]
    )
    typing_audit = _evidence_mapping(
        payload.get("typing_async_audit"), "typing_async_audit"
    )
    _exact_keys(
        typing_audit,
        {
            "strict_type_fixture_count",
            "negative_type_fixtures_planned",
            "effect_interfaces_async",
            "blocking_waits",
            "timing_sleeps",
            "network_dependencies",
        },
        "implementation evidence typing audit",
    )
    if (
        typing_audit.get("strict_type_fixture_count") != active_type_fixtures
        or typing_audit.get("negative_type_fixtures_planned")
        != planned_negative_fixtures
        or typing_audit.get("effect_interfaces_async")
        != ["scripted provider", "async barrier", "local protocol peer"]
        or typing_audit.get("blocking_waits") != 0
        or typing_audit.get("timing_sleeps") != 0
        or typing_audit.get("network_dependencies") != 0
    ):
        raise AcceptanceVerificationError(
            "implementation evidence type or async audit is stale"
        )

    _validate_quality_gate_evidence(
        payload.get("quality_gate"),
        active_acceptance_nodes=active,
        active_type_fixtures=active_type_fixtures,
    )
    _verify_digest(
        payload,
        _EXPECTED_EVIDENCE_SHA256,
        _EXPECTED_EVIDENCE_SHA256,
        "implementation evidence",
    )


def _validate_live_boundary(
    root: Path,
    declared_paths: Sequence[str],
    preserved_untracked: Sequence[str],
) -> None:
    tracked = set(
        _git_lines(
            root,
            "diff",
            "--name-only",
            f"{_EXPECTED_BASELINE_HEAD}..HEAD",
            "--",
        )
    )
    tracked.update(_git_lines(root, "diff", "--name-only", "HEAD", "--"))
    untracked = set(
        _git_lines(root, "ls-files", "--others", "--exclude-standard", "--")
    )
    for prefix in preserved_untracked:
        if not prefix.endswith("/"):
            raise AcceptanceVerificationError(
                f"preserved untracked path must be a directory: {prefix}"
            )
        if any(path.startswith(prefix) for path in tracked):
            raise AcceptanceVerificationError(
                f"preserved untracked path became tracked: {prefix}"
            )
    live_files = tracked | {
        path
        for path in untracked
        if not any(path.startswith(prefix) for prefix in preserved_untracked)
    }
    source_changes = sorted(
        path for path in live_files if path == "src" or path.startswith("src/")
    )
    if source_changes:
        raise AcceptanceVerificationError(
            f"live production source changes are prohibited: {source_changes}"
        )
    directory_claims = tuple(
        path for path in declared_paths if path.endswith("/")
    )
    normalized = {
        next(
            (prefix for prefix in directory_claims if path.startswith(prefix)),
            path,
        )
        for path in live_files
    }
    if normalized != set(declared_paths):
        raise AcceptanceVerificationError(
            "live changed paths differ from implementation evidence:"
            f" declared={sorted(declared_paths)}, live={sorted(normalized)}"
        )


def _git_lines(root: Path, *arguments: str) -> tuple[str, ...]:
    completed = run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AcceptanceVerificationError(
            "cannot verify live git evidence: "
            f"git {' '.join(arguments)}: {completed.stderr.strip()}"
        )
    return tuple(line for line in completed.stdout.splitlines() if line)


def _git_returncode(root: Path, *arguments: str) -> int:
    completed = run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )
    return completed.returncode


def _git_output(root: Path, *arguments: str) -> str:
    lines = _git_lines(root, *arguments)
    if len(lines) != 1:
        raise AcceptanceVerificationError(
            f"live git evidence returned {len(lines)} lines:"
            f" {' '.join(arguments)}"
        )
    return lines[0]


def _evidence_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise AcceptanceVerificationError(
            f"implementation evidence {label} must be a populated object"
        )
    return cast(dict[str, object], value)


def _validate_quality_gate_evidence(
    raw: object,
    *,
    active_acceptance_nodes: int,
    active_type_fixtures: int,
) -> None:
    if not isinstance(raw, list) or len(raw) != 8:
        raise AcceptanceVerificationError(
            "implementation evidence must contain eight exact gate results"
        )
    results: dict[str, dict[str, object]] = {}
    for value in raw:
        result = _evidence_mapping(value, "quality gate result")
        command = _nonempty_string(result.get("command"), "quality command")
        if (
            command in results
            or type(result.get("exit_code")) is not int
            or result.get("exit_code") != 0
        ):
            raise AcceptanceVerificationError(
                "quality gate commands must be unique successful results"
            )
        if _contains_none(result):
            raise AcceptanceVerificationError(
                f"quality gate result contains null evidence: {command}"
            )
        results[command] = result
    commands = frozenset(results)
    if not _EXPECTED_COMMON_GATE_COMMANDS <= commands:
        raise AcceptanceVerificationError(
            "implementation evidence omits a common gate command"
        )
    focused = commands - _EXPECTED_COMMON_GATE_COMMANDS
    if len(focused) != 1 or not next(iter(focused)).startswith(
        "poetry run pytest --verbose -s tests/"
    ):
        raise AcceptanceVerificationError(
            "implementation evidence lacks one exact focused pytest command"
        )
    focused_command = next(iter(focused))
    for command in (
        "poetry run pytest --verbose -s",
        focused_command,
    ):
        result = results[command]
        _exact_keys(
            result,
            {
                "command",
                "exit_code",
                "passed",
                "skipped",
                "deselected",
                "xfail",
                "xpass",
            },
            "test quality-gate evidence",
        )
        counts = {
            name: _nonnegative_int(result.get(name), f"quality {name}")
            for name in (
                "passed",
                "skipped",
                "deselected",
                "xfail",
                "xpass",
            )
        }
        if (
            counts["passed"] == 0
            or counts["deselected"] != 0
            or counts["xfail"] != 0
            or counts["xpass"] != 0
        ):
            raise AcceptanceVerificationError(
                f"test quality-gate evidence is incomplete: {command}"
            )
    legacy_coverage = results["make test-coverage -- -100 src/"]
    _exact_keys(
        legacy_coverage,
        {"command", "exit_code", "output_lines"},
        "legacy coverage evidence",
    )
    if legacy_coverage.get("output_lines") != []:
        raise AcceptanceVerificationError(
            "the legacy exact-coverage audit must have zero output lines"
        )
    exact_coverage = results["make test-coverage-exact no-install"]
    _exact_keys(
        exact_coverage,
        {
            "command",
            "exit_code",
            "covered_statements",
            "total_statements",
            "missing_lines",
            "missing_files",
        },
        "exact coverage evidence",
    )
    covered = exact_coverage.get("covered_statements")
    if (
        type(covered) is not int
        or covered <= 0
        or exact_coverage.get("total_statements") != covered
        or exact_coverage.get("missing_lines") != 0
        or exact_coverage.get("missing_files") != 0
    ):
        raise AcceptanceVerificationError(
            "exact source-coverage evidence is incomplete"
        )
    acceptance = results[
        "poetry run python scripts/verify_input_acceptance.py"
        " --through-phase 0"
    ]
    _exact_keys(
        acceptance,
        {"command", "exit_code", "active_nodes"},
        "acceptance evidence",
    )
    if acceptance.get("active_nodes") != active_acceptance_nodes:
        raise AcceptanceVerificationError(
            "acceptance gate evidence has a stale node count"
        )
    type_result = results["make typecheck-input-contract INPUT_PHASE=0"]
    _exact_keys(
        type_result,
        {"command", "exit_code", "active_fixtures"},
        "type evidence",
    )
    if type_result.get("active_fixtures") != active_type_fixtures:
        raise AcceptanceVerificationError(
            "type gate evidence has a stale fixture count"
        )
    for command in ("make lint", "git diff --check"):
        _exact_keys(
            results[command],
            {"command", "exit_code"},
            "quality command evidence",
        )


def _contains_none(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_none(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_none(item) for item in value)
    return False


def _record_ids(
    raw: object,
    keys: set[str],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise AcceptanceVerificationError(
            f"{label} inventory must be non-empty"
        )
    values: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise AcceptanceVerificationError(f"{label} must be an object")
        item = cast(dict[str, object], entry)
        _exact_keys(item, keys, label)
        values.append(_nonempty_string(item.get("id"), f"{label} id"))
        _nonempty_string(item.get("description"), f"{label} description")
    _unique(values, f"{label} id")
    return tuple(values)


def _run_probe(
    driver: str,
    sentinel: str,
    node_ids: tuple[str, ...],
    root: Path,
) -> dict[str, object]:
    environment = dict(environ)
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.pop("PYTEST_PLUGINS", None)
    environment.pop("PYTHONPATH", None)
    for key in tuple(environment):
        if key.startswith("COV_CORE_") or key == "COVERAGE_PROCESS_START":
            del environment[key]
    completed = run(
        [executable, "-c", driver, *node_ids],
        cwd=root,
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=_PROCESS_TIMEOUT_SECONDS,
    )
    return _probe_payload(completed, sentinel)


def _probe_payload(
    completed: CompletedProcess[str], sentinel: str
) -> dict[str, object]:
    if type(completed.returncode) is not int or completed.returncode != 0:
        raise AcceptanceVerificationError(
            "acceptance probe process exited with code "
            f"{completed.returncode}:\n{completed.stdout}{completed.stderr}"
        )
    lines = [
        line.removeprefix(sentinel)
        for line in completed.stdout.splitlines()
        if line.startswith(sentinel)
    ]
    if len(lines) != 1:
        raise AcceptanceVerificationError(
            "acceptance probe did not return one result payload"
        )
    try:
        payload = strict_json_loads(lines[0])
    except (StrictJsonError, ValueError) as exc:
        raise AcceptanceVerificationError(
            f"acceptance probe returned invalid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise AcceptanceVerificationError(
            "acceptance probe payload must be an object"
        )
    result = cast(dict[str, object], payload)
    result["probe_stdout"] = completed.stdout[-4000:]
    result["probe_stderr"] = completed.stderr[-4000:]
    return result


def _verify_collection(
    expected: tuple[str, ...], payload: dict[str, object]
) -> None:
    _exact_keys(
        payload,
        {
            "exit_code",
            "items",
            "deselected",
            "collection_reports",
            "probe_stdout",
            "probe_stderr",
        },
        "collection payload",
    )
    _verify_probe_common(payload)
    if (
        payload.get("exit_code") != 0
        or type(payload.get("exit_code")) is not int
    ):
        raise AcceptanceVerificationError(
            "acceptance collection exited with code"
            f" {payload.get('exit_code')}"
        )
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise AcceptanceVerificationError("collection items must be a list")
    observed: list[str] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "collection item must be an object"
            )
        item = cast(dict[str, object], raw)
        _exact_keys(item, {"nodeid", "markers"}, "collection item")
        node_id = _nonempty_string(item.get("nodeid"), "collected nodeid")
        markers = _string_list_allow_empty(item.get("markers"), "markers")
        disallowed = sorted(set(markers) & _DISALLOWED_MARKERS)
        if disallowed:
            raise AcceptanceVerificationError(
                f"{node_id} has disallowed markers: {disallowed}"
            )
        observed.append(node_id)
    _verify_exact_nodes(expected, tuple(observed), "collected")


def _verify_execution(
    expected: tuple[str, ...], payload: dict[str, object]
) -> None:
    _exact_keys(
        payload,
        {
            "exit_code",
            "items",
            "deselected",
            "collection_reports",
            "reports",
            "probe_stdout",
            "probe_stderr",
        },
        "execution payload",
    )
    _verify_probe_common(payload)
    raw_items = _string_list(payload.get("items"), "execution items")
    _verify_exact_nodes(expected, raw_items, "executed")
    raw_reports = payload.get("reports")
    if not isinstance(raw_reports, list):
        raise AcceptanceVerificationError("execution reports must be a list")
    by_node: dict[str, list[dict[str, object]]] = {
        node: [] for node in expected
    }
    for raw in raw_reports:
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                "execution report must be an object"
            )
        report = cast(dict[str, object], raw)
        _exact_keys(
            report,
            {"nodeid", "when", "outcome", "wasxfail", "detail"},
            "execution report",
        )
        node_id = report.get("nodeid")
        if not isinstance(node_id, str) or node_id not in by_node:
            raise AcceptanceVerificationError(
                f"unexpected acceptance execution report: {node_id!r}"
            )
        by_node[node_id].append(report)
    for node_id, reports in by_node.items():
        phases = [report.get("when") for report in reports]
        if len(reports) != 3 or set(phases) != {"setup", "call", "teardown"}:
            raise AcceptanceVerificationError(
                f"{node_id} was not exactly once fully executed: {phases}"
            )
        for report in reports:
            if not isinstance(report.get("wasxfail"), str):
                raise AcceptanceVerificationError("wasxfail must be a string")
            if report.get("wasxfail"):
                raise AcceptanceVerificationError(
                    f"{node_id} produced an xfail/xpass outcome"
                )
            if report.get("outcome") != "passed":
                raise AcceptanceVerificationError(
                    f"{node_id} {report.get('when')} outcome was "
                    f"{report.get('outcome')}: {report.get('detail')}"
                )
    if (
        payload.get("exit_code") != 0
        or type(payload.get("exit_code")) is not int
    ):
        raise AcceptanceVerificationError(
            f"acceptance execution exited with code {payload.get('exit_code')}"
        )


def _verify_probe_common(payload: dict[str, object]) -> None:
    if not isinstance(payload.get("probe_stdout"), str) or not isinstance(
        payload.get("probe_stderr"), str
    ):
        raise AcceptanceVerificationError("probe diagnostics must be strings")
    deselected = _string_list_allow_empty(
        payload.get("deselected"), "deselected"
    )
    if deselected:
        raise AcceptanceVerificationError(
            f"acceptance nodes were deselected: {deselected}"
        )
    reports = payload.get("collection_reports")
    if not isinstance(reports, list):
        raise AcceptanceVerificationError("collection reports must be a list")
    if reports:
        raise AcceptanceVerificationError(
            f"acceptance collection was skipped or failed: {reports}"
        )


def _verify_exact_nodes(
    expected: tuple[str, ...], observed: tuple[str, ...], label: str
) -> None:
    missing = sorted(set(expected) - set(observed))
    unexpected = sorted(set(observed) - set(expected))
    duplicates = sorted(
        node_id for node_id in set(observed) if observed.count(node_id) > 1
    )
    if missing or unexpected or duplicates or len(expected) != len(observed):
        raise AcceptanceVerificationError(
            f"acceptance nodes were not exactly {label}: missing={missing},"
            f" unexpected={unexpected}, duplicates={duplicates}"
        )


def _validate_execution_scope(
    manifest_path: Path,
    node_ids: tuple[str, ...],
    root: Path,
) -> None:
    if not root.is_dir():
        raise AcceptanceVerificationError(
            f"acceptance repository root is not a directory: {root}"
        )
    try:
        manifest_path.resolve().relative_to(root)
    except ValueError as exc:
        raise AcceptanceVerificationError(
            "acceptance manifest must be inside the repository root"
        ) from exc
    for node_id in node_ids:
        raw_path = node_id.split("::", 1)[0]
        posix = PurePosixPath(raw_path)
        if posix.is_absolute() or ".." in posix.parts or "\\" in raw_path:
            raise AcceptanceVerificationError(
                f"acceptance node path escapes repository root: {raw_path}"
            )
        if not posix.parts or posix.parts[0] != "tests":
            raise AcceptanceVerificationError(
                f"active acceptance node must be under tests/: {raw_path}"
            )
        path = (root / Path(*posix.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AcceptanceVerificationError(
                f"acceptance node path escapes repository root: {raw_path}"
            ) from exc


def _validate_test_implementation(node_id: str, root: Path) -> None:
    raw_path, *parts = node_id.split("::")
    if not parts:
        raise AcceptanceVerificationError(f"invalid pytest node ID: {node_id}")
    names = [part.split("[", 1)[0] for part in parts]
    function_name = names[-1]
    path = root / Path(*PurePosixPath(raw_path).parts)
    try:
        source = path.read_text(encoding="utf-8")
        tree = parse(source, filename=str(path))
    except (OSError, SyntaxError, UnicodeError) as exc:
        raise AcceptanceVerificationError(
            f"cannot inspect acceptance test {node_id}: {exc}"
        ) from exc
    scope: object = tree
    for class_name in names[:-1]:
        body = getattr(scope, "body", ())
        classes = [
            node
            for node in body
            if isinstance(node, ClassDef) and node.name == class_name
        ]
        if len(classes) != 1:
            raise AcceptanceVerificationError(
                f"acceptance test class is missing or ambiguous: {node_id}"
            )
        scope = classes[0]
    functions = [
        node
        for node in getattr(scope, "body", ())
        if isinstance(node, (FunctionDef, AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(functions) != 1:
        raise AcceptanceVerificationError(
            f"acceptance test function is missing or ambiguous: {node_id}"
        )
    function = functions[0]
    body = list(function.body)
    if (
        body
        and isinstance(body[0], Expr)
        and isinstance(body[0].value, Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    meaningful = [
        statement
        for statement in body
        if not _placeholder_statement(statement)
    ]
    if not meaningful:
        raise AcceptanceVerificationError(
            "acceptance test is a placeholder or unconditional pass:"
            f" {node_id}"
        )
    check_paths = _check_sequence(body, frozenset({False}))
    successful_paths = check_paths.next_states | check_paths.return_states
    if not successful_paths or False in successful_paths:
        raise AcceptanceVerificationError(
            "acceptance test has a reachable successful path without a"
            f" meaningful check: {node_id}"
        )
    _validate_prohibited_test_constructs(tree, node_id)
    segment = "\n".join(
        source.splitlines()[function.lineno - 1 : function.end_lineno]
    )
    if _COVERAGE_EXCLUSION_PATTERN.search(segment):
        raise AcceptanceVerificationError(
            "acceptance test uses a feature-specific coverage exclusion:"
            f" {node_id}"
        )


def _placeholder_statement(statement: AST) -> bool:
    if isinstance(statement, Pass):
        return True
    if isinstance(statement, Expr) and isinstance(statement.value, Constant):
        return statement.value.value in {True, Ellipsis, None}
    if isinstance(statement, Assert) and isinstance(statement.test, Constant):
        return True
    if isinstance(statement, Expr) and isinstance(statement.value, Call):
        call = statement.value
        if (
            isinstance(call.func, Attribute)
            and call.func.attr in {"assertTrue", "assertEqual"}
            and call.args
            and all(isinstance(argument, Constant) for argument in call.args)
        ):
            values = tuple(
                cast(Constant, argument).value for argument in call.args
            )
            if call.func.attr == "assertTrue" and bool(values[0]):
                return True
            if (
                call.func.attr == "assertEqual"
                and len(values) >= 2
                and values[0] == values[1]
            ):
                return True
    return isinstance(statement, Return) and statement.value is None


def _is_check(node: AST) -> bool:
    if isinstance(node, Assert):
        return _expression_is_dynamic(node.test)
    if not isinstance(node, Call):
        return False
    if isinstance(node.func, Attribute):
        if node.func.attr in {"raises", "warns"}:
            return bool(node.args) and any(
                _expression_is_dynamic(argument) for argument in node.args
            )
        if not node.func.attr.startswith("assert"):
            return False
        if (
            node.func.attr
            in {"assertEqual", "assertIs", "assertSequenceEqual"}
            and len(node.args) >= 2
            and _same_expression(node.args[0], node.args[1])
        ):
            return False
        return not node.args or any(
            _expression_is_dynamic(argument) for argument in node.args
        )
    return False


def _validate_prohibited_test_constructs(tree: AST, node_id: str) -> None:
    for node in walk(tree):
        prohibited: str | None = None
        if isinstance(node, ImportFrom):
            prohibited = next(
                (
                    item.name.rsplit(".", 1)[-1]
                    for item in node.names
                    if item.name.rsplit(".", 1)[-1] in _PROHIBITED_TEST_SYMBOLS
                ),
                None,
            )
        elif (
            isinstance(node, Name)
            and isinstance(node.ctx, Load)
            and node.id in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.id
        elif (
            isinstance(node, Attribute)
            and node.attr in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.attr
        elif (
            isinstance(node, Subscript)
            and isinstance(node.slice, Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.slice.value
        elif (
            isinstance(node, Call)
            and isinstance(node.func, Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.args[1].value
        elif (
            isinstance(node, Call)
            and isinstance(node.func, Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value in _PROHIBITED_TEST_SYMBOLS
        ):
            prohibited = node.args[0].value
        if prohibited is None:
            continue
        category = (
            "pytest control"
            if prohibited in _PROHIBITED_TEST_CONTROLS
            else "execution trick"
        )
        raise AcceptanceVerificationError(
            f"acceptance test uses a prohibited {category} ({prohibited}):"
            f" {node_id}"
        )


def _check_sequence(
    statements: Sequence[AST],
    initial_states: frozenset[bool],
) -> _CheckPaths:
    next_states = set(initial_states)
    return_states: set[bool] = set()
    break_states: set[bool] = set()
    continue_states: set[bool] = set()
    for statement in statements:
        if not next_states:
            break
        statement_paths = _merge_check_paths(
            _check_statement(statement, state) for state in next_states
        )
        next_states = set(statement_paths.next_states)
        return_states.update(statement_paths.return_states)
        break_states.update(statement_paths.break_states)
        continue_states.update(statement_paths.continue_states)
    return _CheckPaths(
        next_states=frozenset(next_states),
        return_states=frozenset(return_states),
        break_states=frozenset(break_states),
        continue_states=frozenset(continue_states),
    )


def _check_statement(statement: AST, checked: bool) -> _CheckPaths:
    if isinstance(statement, (FunctionDef, AsyncFunctionDef, ClassDef)):
        return _CheckPaths(next_states=frozenset({checked}))
    if isinstance(statement, Assert):
        if isinstance(statement.test, Constant):
            if bool(statement.test.value):
                return _CheckPaths(next_states=frozenset({checked}))
            return _CheckPaths()
        state = checked or _is_check(statement)
        return _CheckPaths(next_states=frozenset({state}))
    if isinstance(statement, Return):
        return _CheckPaths(return_states=frozenset({checked}))
    if isinstance(statement, Raise):
        return _CheckPaths()
    if isinstance(statement, Break):
        return _CheckPaths(break_states=frozenset({checked}))
    if isinstance(statement, Continue):
        return _CheckPaths(continue_states=frozenset({checked}))
    if isinstance(statement, If):
        branches: tuple[Sequence[AST], ...]
        if isinstance(statement.test, Constant):
            branches = (
                (
                    statement.body
                    if bool(statement.test.value)
                    else statement.orelse
                ),
            )
        else:
            branches = (statement.body, statement.orelse)
        return _merge_check_paths(
            _check_sequence(branch, frozenset({checked}))
            for branch in branches
        )
    if isinstance(statement, (For, AsyncFor)):
        body = _check_sequence(statement.body, frozenset({checked}))
        natural_states = body.next_states | body.continue_states
        if not _statically_nonempty_iter(statement.iter):
            natural_states |= frozenset({checked})
        orelse = _check_sequence(statement.orelse, natural_states)
        return _CheckPaths(
            next_states=orelse.next_states | body.break_states,
            return_states=body.return_states | orelse.return_states,
            break_states=orelse.break_states,
            continue_states=orelse.continue_states,
        )
    if isinstance(statement, While):
        if isinstance(statement.test, Constant) and not bool(
            statement.test.value
        ):
            return _check_sequence(statement.orelse, frozenset({checked}))
        body = _check_sequence(statement.body, frozenset({checked}))
        if isinstance(statement.test, Constant):
            natural_states = frozenset()
        else:
            natural_states = body.next_states | body.continue_states
            natural_states |= frozenset({checked})
        orelse = _check_sequence(statement.orelse, natural_states)
        return _CheckPaths(
            next_states=orelse.next_states | body.break_states,
            return_states=body.return_states | orelse.return_states,
            break_states=orelse.break_states,
            continue_states=orelse.continue_states,
        )
    if isinstance(statement, (With, AsyncWith)):
        state = checked or any(
            _is_check(item.context_expr) for item in statement.items
        )
        return _check_sequence(statement.body, frozenset({state}))
    if isinstance(statement, Try):
        body = _check_sequence(statement.body, frozenset({checked}))
        normal = _check_sequence(statement.orelse, body.next_states)
        handlers = _merge_check_paths(
            _check_sequence(handler.body, frozenset({checked}))
            for handler in statement.handlers
        )
        combined = _merge_check_paths(
            (
                _CheckPaths(
                    next_states=normal.next_states,
                    return_states=body.return_states | normal.return_states,
                    break_states=body.break_states | normal.break_states,
                    continue_states=(
                        body.continue_states | normal.continue_states
                    ),
                ),
                handlers,
            )
        )
        return _apply_finally(combined, statement.finalbody)
    if isinstance(statement, Match):
        paths = [
            _check_sequence(case.body, frozenset({checked}))
            for case in statement.cases
        ]
        if not any(
            isinstance(case.pattern, MatchAs)
            and case.pattern.pattern is None
            and case.guard is None
            for case in statement.cases
        ):
            paths.append(_CheckPaths(next_states=frozenset({checked})))
        return _merge_check_paths(paths)
    state = checked or _statement_executes_check(statement)
    return _CheckPaths(next_states=frozenset({state}))


def _statement_executes_check(statement: AST) -> bool:
    if isinstance(statement, Expr):
        return _is_check(statement.value)
    return _is_check(statement)


def _merge_check_paths(paths: Iterable[_CheckPaths]) -> _CheckPaths:
    next_states: set[bool] = set()
    return_states: set[bool] = set()
    break_states: set[bool] = set()
    continue_states: set[bool] = set()
    for path in paths:
        next_states.update(path.next_states)
        return_states.update(path.return_states)
        break_states.update(path.break_states)
        continue_states.update(path.continue_states)
    return _CheckPaths(
        next_states=frozenset(next_states),
        return_states=frozenset(return_states),
        break_states=frozenset(break_states),
        continue_states=frozenset(continue_states),
    )


def _apply_finally(
    paths: _CheckPaths, statements: Sequence[AST]
) -> _CheckPaths:
    if not statements:
        return paths
    transformed: list[_CheckPaths] = []
    for outcome, states in (
        ("next", paths.next_states),
        ("return", paths.return_states),
        ("break", paths.break_states),
        ("continue", paths.continue_states),
    ):
        for state in states:
            final = _check_sequence(statements, frozenset({state}))
            if outcome == "next":
                preserved = _CheckPaths(next_states=final.next_states)
            elif outcome == "return":
                preserved = _CheckPaths(return_states=final.next_states)
            elif outcome == "break":
                preserved = _CheckPaths(break_states=final.next_states)
            else:
                preserved = _CheckPaths(continue_states=final.next_states)
            transformed.extend(
                (
                    preserved,
                    _CheckPaths(
                        return_states=final.return_states,
                        break_states=final.break_states,
                        continue_states=final.continue_states,
                    ),
                )
            )
    return _merge_check_paths(transformed)


def _expression_is_dynamic(expression: AST) -> bool:
    if (
        isinstance(expression, Compare)
        and len(expression.ops) == 1
        and isinstance(expression.ops[0], (Eq, Is))
        and len(expression.comparators) == 1
        and _same_expression(expression.left, expression.comparators[0])
    ):
        return False
    return any(
        isinstance(node, (Name, Call, Attribute, Subscript))
        for node in walk(expression)
    )


def _same_expression(left: AST, right: AST) -> bool:
    return dump(left, include_attributes=False) == dump(
        right,
        include_attributes=False,
    )


def _statically_nonempty_iter(expression: AST) -> bool:
    if isinstance(expression, (AstList, AstSet, AstTuple)):
        return bool(expression.elts)
    if isinstance(expression, AstDict):
        return bool(expression.keys)
    return (
        isinstance(expression, Constant)
        and isinstance(expression.value, (str, tuple, frozenset))
        and bool(expression.value)
    )


def _type_fixture_path(raw: str, root: Path) -> Path:
    posix = PurePosixPath(raw)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or "\\" in raw
        or len(posix.parts) < 3
        or posix.parts[:2] != ("tests", "input_type_contracts")
        or not raw.endswith(".py")
    ):
        raise AcceptanceVerificationError(
            f"type fixture path is outside its tracked directory: {raw}"
        )
    path = (root / Path(*posix.parts)).resolve()
    try:
        path.relative_to((root / "tests" / "input_type_contracts").resolve())
    except ValueError as exc:
        raise AcceptanceVerificationError(
            f"type fixture path escapes its tracked directory: {raw}"
        ) from exc
    return path


def _verify_digest(
    value: object,
    raw_digest: object,
    expected_digest: str,
    label: str,
) -> None:
    digest = _nonempty_string(raw_digest, f"{label} SHA-256")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise AcceptanceVerificationError(
            f"{label} SHA-256 must be lowercase hexadecimal"
        )
    canonical = dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    calculated = sha256(canonical).hexdigest()
    if digest != calculated or digest != expected_digest:
        raise AcceptanceVerificationError(
            f"{label} digest mismatch: declared={digest},"
            f" calculated={calculated}"
        )


def _exact_keys(
    payload: dict[str, object], expected: set[str], label: str
) -> None:
    if set(payload) != expected:
        raise AcceptanceVerificationError(
            f"{label} has invalid keys: expected={sorted(expected)}, "
            f"observed={sorted(payload)}"
        )


def _phase(value: object, label: str) -> int:
    if type(value) is not int or value < _MIN_PHASE or value > _MAX_PHASE:
        raise AcceptanceVerificationError(
            f"{label} must be an integer from {_MIN_PHASE} to {_MAX_PHASE}"
        )
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise AcceptanceVerificationError(
            f"{label} must be a non-negative integer"
        )
    return value


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcceptanceVerificationError(
            f"{label} must be a non-empty string"
        )
    return value


def _unambiguous_outcome(
    value: str,
    label: str,
    key: tuple[str, str],
) -> None:
    normalized = f" {value.lower()} "
    if " or " in normalized or " unless " in normalized or "|" in value:
        raise AcceptanceVerificationError(
            f"failure {label} is ambiguous for {key}: {value}"
        )


def _string_list(value: object, label: str) -> tuple[str, ...]:
    result = _string_list_allow_empty(value, label)
    if not result:
        raise AcceptanceVerificationError(f"{label} must not be empty")
    return result


def _string_list_allow_empty(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise AcceptanceVerificationError(f"{label} must be a string list")
    return tuple(cast(list[str], value))


def _node_id(value: object) -> str:
    node_id = _nonempty_string(value, "pytest node ID")
    raw_path, separator, test_name = node_id.partition("::")
    if not separator or not raw_path.endswith(".py") or not test_name:
        raise AcceptanceVerificationError(f"invalid pytest node ID: {node_id}")
    posix = PurePosixPath(raw_path)
    if (
        posix.is_absolute()
        or ".." in posix.parts
        or "\\" in raw_path
        or not posix.parts
        or posix.parts[0] != "tests"
    ):
        raise AcceptanceVerificationError(
            f"pytest node path must be a safe tracked test path: {node_id}"
        )
    return node_id


def _unique(values: Iterable[str], label: str) -> None:
    items = tuple(values)
    duplicates = sorted(item for item in set(items) if items.count(item) > 1)
    if duplicates:
        raise AcceptanceVerificationError(f"duplicate {label}: {duplicates}")


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Collect and execute every active structured-input acceptance "
            "node without skips, xfails, or deselection."
        )
    )
    parser.add_argument("--through-phase", required=True, type=int)
    parser.add_argument(
        "--manifest", type=Path, default=default_manifest_path()
    )
    parser.add_argument("--repo-root", type=Path, default=repository_root())
    return parser.parse_args()


def main() -> int:
    """Run acceptance verification from the command line."""
    args = _parse_args()
    try:
        manifest = verify_acceptance(
            args.manifest,
            repo_root=args.repo_root,
            through_phase=args.through_phase,
        )
    except (AcceptanceVerificationError, TimeoutExpired) as exc:
        print(f"structured-input acceptance failed: {exc}", file=stderr)
        return 1
    active_count = len(manifest.active_nodes(args.through_phase))
    print(
        "structured-input acceptance passed: "
        f"through_phase={args.through_phase} nodes={active_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
