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
from stat import S_IXGRP, S_IXOTH, S_IXUSR
from subprocess import CompletedProcess, TimeoutExpired, run
from sys import executable, stderr
from typing import Protocol, cast

from coverage import Coverage
from input_contract_json import (
    StrictJsonError,
    strict_json_loads,
    strict_json_path,
)
from verify_src_coverage import (
    CoverageVerificationError,
    verify_src_coverage,
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
    "5f73b948ccf2f817fad54952b1b4864b6403a2f4928dd82d1d38e28d54b57ea1"
)
_EXPECTED_FAILURE_MATRIX_SHA256 = (
    "e5ce3aac0d441897b80a09d6a693853c65d4a446ed7e4c0184b3e3bc0b212c08"
)
_EXPECTED_DECISIONS_SHA256 = (
    "c13bcff64c0b28905c64c8e92b040d56e2312a99b45303b4e3a5d4d4490c882d"
)
_EXPECTED_NO_BC_SHA256 = (
    "c75145467fe15a1cd55b6bb10e7dd16fc5ff8e4b25b530c2d7f147ab3c641887"
)
_EXPECTED_ACCEPTANCE_LEDGER_SHA256 = (
    "e9b5258995b4b0769c9276141e0bb89161c03099dc9db08c5afa7190fb179bbb"
)
_EXPECTED_EVIDENCE_SHA256 = (
    "59788e2441bec0bd34a61ff94f8b14459ca229a37fcf693ae6b94fb8106e8ab9"
)
_EXPECTED_REVIEW_HISTORY_SHA256 = (
    "f59a5cb66ee765407b15134bfe8e2a2c19600b807dcca550a89ef68b2caaee1c"
)
_EXPECTED_PHASE0_REVIEW_SHA256 = (
    "573625598e6f7501e5d3cbc158be7b630427143e1cdd7658814a52b6374d8f6b"
)
_EXPECTED_PHASE1_REVIEW_SHA256 = (
    "42ee51f1041cc975bcdd750247d3e61a08fe453f1f332d76f9dd47e18b8e4a85"
)
_EXPECTED_PHASE2_REVIEW_SHA256 = (
    "7c94eb4806501ecb3ae82f1447fd94ed95e31d185d41e9fbcba2f31ce448a408"
)
_EXPECTED_PHASE2_PENDING_REVIEW_SHA256 = (
    "a83a4e9545ac72c99c23d6fd316c7661f5a6bfef86c8c39a5c209ee6185a852a"
)
_EXPECTED_PHASE1_QUALITY_SHA256 = (
    "f58bd16d9bf57bb3f2972982ff8bcf19a6125715a40194effecb8141c8ebd5ea"
)
_EXPECTED_PHASE1_EVIDENCE_SHA256 = (
    "a4c16a90cf2d451b423da22ba763b50742e47f583230ded87c9997d77e1b93b8"
)
_EXPECTED_PHASE2_QUALITY_SHA256 = (
    "d004e9f765e9167d31debb7642883e774e42a03503f32f8869eb6b4e084e3953"
)
_EXPECTED_PHASE2_EVIDENCE_SHA256 = (
    "d0e276493609d2e7254c576bf50552a933e4e54cb67c9ec6e6a71f94a17f0302"
)
_EXPECTED_QUALITY_HISTORY_SHA256 = (
    "0bc69e337549c1308468fd095e26a5a680440e8d60181527469b52b13497710f"
)
_EXPECTED_IMPLEMENTATION_OWNER = "/root"
_EXPECTED_INDEPENDENT_REVIEWER = "/root/input_contract_audit"
_EXPECTED_REVIEW_OCCURRENCES = (
    (0, "baseline", "/root/input_contract_audit", "approved"),
    (1, "semantic", "/root/interaction_round4_semantic", "pending"),
    (1, "gate", "/root/interaction_round4_gates", "pending"),
    (1, "semantic", "/root/interaction_round4_semantic", "approved"),
    (1, "gate", "/root/interaction_round4_gates", "approved"),
    (2, "semantic", "/root/broker_review", "pending"),
    (2, "gate", "/root/phase2_acceptance_review", "pending"),
    (2, "semantic", "/root/phase2_acceptance_review", "approved"),
    (2, "gate", "/root/phase2_metadata_review", "approved"),
    (3, "gate", "/root/terminal_review", "pending"),
    (3, "semantic", "/root/acceptance_review", "approved"),
    (3, "closure", "/root/phase3_closure_audit", "pending"),
    (3, "closure", "/root/phase3_closure_audit", "approved"),
    (
        3,
        "coverage-closure",
        "/root/phase3_closure_audit/turn3_toolmanager_readonly",
        "pending",
    ),
    (
        3,
        "coverage-closure",
        "/root/phase3_closure_audit/turn3_toolmanager_readonly",
        "approved",
    ),
    (3, "gate", "/root/terminal_review", "approved"),
)
_EXPECTED_CURRENT_SEMANTIC_REVIEW_STATUS = "approved"
_EXPECTED_CURRENT_GATE_REVIEW_STATUS = "approved"
_EXPECTED_BASELINE_HEAD = "609aa091c17756ab952cf5fe668ca3d867f0e311"
_EXPECTED_BASELINE_SUBJECT = "Bump version to v1.5.8 (#1067)"
_EXPECTED_PENDING_SOURCE_INVENTORY = (
    "32cd39d8285af3b782ca095bda1a80de5e991e98e4baf1ba1cf003c5d02a80ba",
    424,
    108402,
    1327,
)
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
        "src/avalan/agent/",
        "src/avalan/cli/",
        "src/avalan/event/__init__.py",
        "src/avalan/event/manager.py",
        "src/avalan/flow/registry.py",
        "src/avalan/interaction/",
        "src/avalan/model/",
        "src/avalan/server/a2a/router.py",
        "src/avalan/server/routers/chat.py",
        "src/avalan/server/routers/mcp.py",
        "src/avalan/server/routers/responses.py",
        "src/avalan/task/event.py",
        "src/avalan/tool/",
        "tests/agent/",
        "tests/cli/",
        "tests/event/interaction_lifecycle_test.py",
        "tests/fixtures/input/",
        "tests/flow/",
        "tests/input/",
        "tests/input_acceptance_verifier_test.py",
        "tests/input_contract_fixtures.py",
        "tests/input_contract_harness_test.py",
        "tests/input_contract_metadata_test.py",
        "tests/input_contract_test.py",
        "tests/input_type_contract_test.py",
        "tests/input_type_contracts/",
        "tests/interaction/",
        "tests/interaction_type_contracts/",
        "tests/model/",
        "tests/project_metadata_test.py",
        "tests/reasoning_summary_phase1_test.py",
        "tests/server/",
        "tests/src_coverage_verifier_test.py",
        "tests/task/",
        "tests/tool/",
    }
)
_EXPECTED_PRODUCTION_SOURCE_PATHS = frozenset(
    {
        "src/avalan/agent/",
        "src/avalan/cli/",
        "src/avalan/event/__init__.py",
        "src/avalan/event/manager.py",
        "src/avalan/flow/registry.py",
        "src/avalan/interaction/",
        "src/avalan/model/",
        "src/avalan/server/a2a/router.py",
        "src/avalan/server/routers/chat.py",
        "src/avalan/server/routers/mcp.py",
        "src/avalan/server/routers/responses.py",
        "src/avalan/task/event.py",
        "src/avalan/tool/",
    }
)
_EXPECTED_ORDERED_COMMON_GATE_COMMANDS = (
    "poetry run pytest --verbose -s",
    "make test-coverage -- -100 src/",
    "make test-coverage-exact no-install",
    (
        "poetry run python scripts/verify_input_acceptance.py"
        + " --through-phase 3"
    ),
    "make typecheck-input-contract INPUT_PHASE=3",
    "make lint",
    "git diff --check",
)
_EXPECTED_COMMON_GATE_COMMANDS = frozenset(
    _EXPECTED_ORDERED_COMMON_GATE_COMMANDS
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
class RequirementActivationSlice:
    """Store reviewed ownership for one partially active requirement."""

    requirement_id: str
    phase: int
    active_owner: str
    active_scope: str
    active_node_ids: tuple[str, ...]
    remaining_owner: str
    remaining_scope: str
    planned_node_ids: tuple[str, ...]
    reviewed_by: str
    evidence: str


@dataclass(frozen=True, kw_only=True, slots=True)
class ParameterExpansion:
    """Store the exact pytest instances owned by one parametrized node."""

    node_id: str
    instance_node_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True, slots=True)
class AcceptanceManifest:
    """Store the validated acceptance inventory."""

    path: Path
    current_phase: int
    nodes: tuple[AcceptanceNode, ...]
    requirement_activation_slices: tuple[RequirementActivationSlice, ...]
    parameter_expansions: tuple[ParameterExpansion, ...]

    def active_nodes(self, through_phase: int) -> tuple[AcceptanceNode, ...]:
        """Return active nodes introduced no later than the requested gate."""
        assert _MIN_PHASE <= through_phase <= self.current_phase
        return tuple(
            node
            for node in self.nodes
            if node.lifecycle == "active"
            and node.active_from_phase <= through_phase
        )

    def active_pytest_instances(self, through_phase: int) -> tuple[str, ...]:
        """Return the exact pytest instances required by the selected gate."""
        expansions = {
            expansion.node_id: expansion.instance_node_ids
            for expansion in self.parameter_expansions
        }
        return tuple(
            instance
            for node in self.active_nodes(through_phase)
            for instance in expansions.get(node.node_id, (node.node_id,))
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
            "requirement_activation_slices",
            "parameter_expansions",
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
    requirement_activation_slices = _requirement_activation_slices(
        payload.get("requirement_activation_slices"),
        nodes,
        current_phase,
    )
    parameter_expansions = _parameter_expansions(
        payload.get("parameter_expansions"),
        nodes,
    )
    _activation_history(
        payload.get("activation_history"),
        nodes,
        current_phase,
    )
    _activation_snapshots(
        payload.get("activation_snapshots"),
        payload.get("replacements"),
        payload.get("requirement_activation_slices"),
        payload.get("parameter_expansions"),
        nodes,
        current_phase,
    )
    return AcceptanceManifest(
        path=path,
        current_phase=current_phase,
        nodes=nodes,
        requirement_activation_slices=requirement_activation_slices,
        parameter_expansions=parameter_expansions,
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
    instance_node_ids = manifest.active_pytest_instances(through_phase)
    _validate_execution_scope(path, node_ids, root)
    for node_id in node_ids:
        _validate_test_implementation(node_id, root)
    collection = _run_probe(
        _COLLECT_DRIVER,
        _COLLECT_SENTINEL,
        node_ids,
        root,
    )
    collected_node_ids = _verify_collection(instance_node_ids, collection)
    execution = _run_probe(
        _EXECUTE_DRIVER,
        _EXECUTE_SENTINEL,
        node_ids,
        root,
    )
    _verify_execution(instance_node_ids, execution, collected_node_ids)
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


def _parameter_expansions(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
) -> tuple[ParameterExpansion, ...]:
    if not isinstance(raw, list):
        raise AcceptanceVerificationError(
            "parameter expansions must be a list"
        )
    by_node_id = {node.node_id: node for node in nodes}
    manifest_node_ids = frozenset(by_node_id)
    expansions: list[ParameterExpansion] = []
    all_instances: list[str] = []
    for value in raw:
        if not isinstance(value, dict):
            raise AcceptanceVerificationError(
                "parameter expansion must be an object"
            )
        item = cast(dict[str, object], value)
        _exact_keys(
            item,
            {"node_id", "instance_node_ids", "sha256"},
            "parameter expansion",
        )
        node_id = _node_id(item.get("node_id"))
        node = by_node_id.get(node_id)
        if node is None:
            raise AcceptanceVerificationError(
                "parameter expansion must own one exact manifest node:"
                f" {node_id}"
            )
        if "[" in node_id.rsplit("::", 1)[-1]:
            raise AcceptanceVerificationError(
                "explicit parameter instance must remain exact-only:"
                f" {node_id}"
            )
        instances = _string_list(
            item.get("instance_node_ids"),
            "parameter instance node IDs",
        )
        _unique(instances, f"parameter instance for {node_id}")
        for instance in instances:
            _node_id(instance)
            if not instance.startswith(f"{node_id}[") or not instance.endswith(
                "]"
            ):
                raise AcceptanceVerificationError(
                    "parameter instance does not belong to its base node:"
                    f" {instance}"
                )
            if instance in manifest_node_ids:
                raise AcceptanceVerificationError(
                    "parameter instance duplicates a manifest node:"
                    f" {instance}"
                )
        digest = _nonempty_string(
            item.get("sha256"),
            "parameter expansion SHA-256",
        )
        calculated = sha256("\n".join(instances).encode("utf-8")).hexdigest()
        if digest != calculated:
            raise AcceptanceVerificationError(
                f"parameter expansion digest mismatch: {node_id}"
            )
        expansions.append(
            ParameterExpansion(
                node_id=node_id,
                instance_node_ids=instances,
            )
        )
        all_instances.extend(instances)
    _unique(
        (expansion.node_id for expansion in expansions),
        "parameter expansion base node",
    )
    _unique(all_instances, "parameter instance across expansions")
    node_order = {node.node_id: index for index, node in enumerate(nodes)}
    expansion_order = [
        node_order[expansion.node_id] for expansion in expansions
    ]
    if expansion_order != sorted(expansion_order):
        raise AcceptanceVerificationError(
            "parameter expansions must preserve manifest node order"
        )
    return tuple(expansions)


def _requirement_activation_slices(
    raw: object,
    nodes: tuple[AcceptanceNode, ...],
    current_phase: int,
) -> tuple[RequirementActivationSlice, ...]:
    if not isinstance(raw, list):
        raise AcceptanceVerificationError(
            "requirement activation slices must be a list"
        )
    active_by_requirement: dict[str, tuple[AcceptanceNode, ...]] = {}
    planned_by_requirement: dict[str, tuple[AcceptanceNode, ...]] = {}
    for requirement_id in _EXPECTED_REQUIREMENT_IDS:
        active = tuple(
            node
            for node in nodes
            if requirement_id in node.requirement_ids
            and node.lifecycle == "active"
        )
        planned = tuple(
            node
            for node in nodes
            if requirement_id in node.requirement_ids
            and node.lifecycle == "planned"
        )
        if active:
            active_by_requirement[requirement_id] = active
        if planned:
            planned_by_requirement[requirement_id] = planned
    mixed_requirements = set(active_by_requirement) & set(
        planned_by_requirement
    )
    slices: list[RequirementActivationSlice] = []
    expected_keys = {
        "requirement_id",
        "phase",
        "active_owner",
        "active_scope",
        "active_node_ids",
        "remaining_owner",
        "remaining_scope",
        "planned_node_ids",
        "reviewed_by",
        "evidence",
    }
    for value in raw:
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise AcceptanceVerificationError(
                "requirement activation slice has invalid shape"
            )
        item = cast(dict[str, object], value)
        requirement_id = _nonempty_string(
            item.get("requirement_id"),
            "slice requirement_id",
        )
        if requirement_id not in mixed_requirements:
            raise AcceptanceVerificationError(
                "requirement activation slice is not mixed-lifecycle:"
                f" {requirement_id}"
            )
        phase = _phase(item.get("phase"), "slice phase")
        if phase > current_phase:
            raise AcceptanceVerificationError(
                "requirement activation slice phase is not implemented"
            )
        active_nodes = active_by_requirement[requirement_id]
        planned_nodes = planned_by_requirement[requirement_id]
        expected_phase = min(node.active_from_phase for node in active_nodes)
        if phase != expected_phase:
            raise AcceptanceVerificationError(
                "requirement activation slice phase differs from its first"
                f" active node: {requirement_id}"
            )
        active_node_ids = _string_list(
            item.get("active_node_ids"),
            "slice active_node_ids",
        )
        planned_node_ids = _string_list(
            item.get("planned_node_ids"),
            "slice planned_node_ids",
        )
        _unique(active_node_ids, "slice active node ID")
        _unique(planned_node_ids, "slice planned node ID")
        if active_node_ids != tuple(node.node_id for node in active_nodes):
            raise AcceptanceVerificationError(
                "requirement activation slice active inventory changed:"
                f" {requirement_id}"
            )
        if planned_node_ids != tuple(node.node_id for node in planned_nodes):
            raise AcceptanceVerificationError(
                "requirement activation slice planned inventory changed:"
                f" {requirement_id}"
            )
        active_owner = _slice_detail(item, "active_owner")
        active_scope = _slice_detail(item, "active_scope")
        remaining_owner = _slice_detail(item, "remaining_owner")
        remaining_scope = _slice_detail(item, "remaining_scope")
        reviewed_by = _nonempty_string(
            item.get("reviewed_by"),
            "slice reviewed_by",
        )
        if reviewed_by != _EXPECTED_IMPLEMENTATION_OWNER:
            raise AcceptanceVerificationError(
                "requirement activation slice lacks implementation review"
            )
        evidence = _slice_detail(item, "evidence")
        slices.append(
            RequirementActivationSlice(
                requirement_id=requirement_id,
                phase=phase,
                active_owner=active_owner,
                active_scope=active_scope,
                active_node_ids=active_node_ids,
                remaining_owner=remaining_owner,
                remaining_scope=remaining_scope,
                planned_node_ids=planned_node_ids,
                reviewed_by=reviewed_by,
                evidence=evidence,
            )
        )
    _unique(
        (item.requirement_id for item in slices),
        "requirement activation slice ID",
    )
    observed = {item.requirement_id for item in slices}
    if observed != mixed_requirements:
        raise AcceptanceVerificationError(
            "mixed-lifecycle requirements lack exact activation slices:"
            f" expected={sorted(mixed_requirements)},"
            f" observed={sorted(observed)}"
        )
    return tuple(slices)


def _slice_detail(item: dict[str, object], field: str) -> str:
    value = _nonempty_string(item.get(field), f"slice {field}").strip()
    if len(value) < 20 or value.lower() in {
        "pending",
        "placeholder",
        "tbd",
        "todo",
    }:
        raise AcceptanceVerificationError(
            f"requirement activation slice {field} is not concrete"
        )
    return value


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
    raw_requirement_slices: object,
    raw_parameter_expansions: object,
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
    if not isinstance(raw_parameter_expansions, list):
        raise AcceptanceVerificationError(
            "parameter expansions must be a list"
        )
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
                "requirement_activation_slices": raw_requirement_slices,
                "parameter_expansions": raw_parameter_expansions,
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
            "planned_replacements",
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
    active_paths: list[str] = []
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
        if lifecycle not in {"active", "planned", "replaced"}:
            raise AcceptanceVerificationError(
                f"invalid type fixture lifecycle: {lifecycle}"
            )
        expected = "active" if active_from <= current_phase else "planned"
        if lifecycle != "replaced" and lifecycle != expected:
            raise AcceptanceVerificationError(
                f"type fixture lifecycle regression: {item.get('id')}"
            )
        if lifecycle == "replaced" and active_from > current_phase:
            raise AcceptanceVerificationError(
                f"unimplemented type fixture replacement: {item.get('id')}"
            )
        if lifecycle == "active":
            active_ids.append(identifier)
        raw_path = _nonempty_string(item.get("path"), "type fixture path")
        if lifecycle != "replaced":
            active_paths.append(raw_path)
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
    _unique(active_paths, "type fixture path")
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
            and item.get("lifecycle") == "active"
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
            "review_history_sha256",
            "review_history_phase0_sha256",
            "review_history_phase1_sha256",
            "review_history_phase2_sha256",
            "review_history",
            "quality_history_sha256",
            "quality_history",
            "active_test_node_ids",
            "git",
            "baseline",
            "boundary",
            "pending_structural_inventory",
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
    _validate_review_history(
        payload.get("review_history"),
        payload.get("review_history_sha256"),
        payload.get("review_history_phase0_sha256"),
        payload.get("review_history_phase1_sha256"),
        payload.get("review_history_phase2_sha256"),
        manifest.current_phase,
        implementation_owner,
    )
    _validate_quality_history(
        payload.get("quality_history"),
        payload.get("quality_history_sha256"),
        manifest.current_phase,
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
            "production_capability_history",
            "production_source_changes",
            "changed_paths",
        },
        "implementation evidence boundary",
    )
    changed_paths = _string_list(
        boundary.get("changed_paths"), "boundary changed_paths"
    )
    _unique(changed_paths, "boundary changed path")
    production_source_changes = _string_list(
        boundary.get("production_source_changes"),
        "production source changes",
    )
    _unique(production_source_changes, "production source change")
    capability_history = _production_capability_history(
        boundary.get("production_capability_history"),
        manifest.current_phase,
    )
    if (
        boundary.get("production_capability") != capability_history[-1]
        or frozenset(production_source_changes)
        != _EXPECTED_PRODUCTION_SOURCE_PATHS
        or len(production_source_changes)
        != len(_EXPECTED_PRODUCTION_SOURCE_PATHS)
        or frozenset(changed_paths) != _EXPECTED_BOUNDARY_PATHS
        or len(changed_paths) != len(_EXPECTED_BOUNDARY_PATHS)
    ):
        raise AcceptanceVerificationError(
            "implementation evidence production boundary is stale"
        )
    _validate_live_boundary(
        root,
        changed_paths,
        production_source_changes,
        preserved_untracked,
    )

    _validate_pending_structural_inventory(
        payload.get("pending_structural_inventory"), root
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
            "active_pytest_instances",
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
        "active_pytest_instances": len(
            manifest.active_pytest_instances(manifest.current_phase)
        ),
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
        active_pytest_instances=len(
            manifest.active_pytest_instances(manifest.current_phase)
        ),
        active_type_fixtures=active_type_fixtures,
        root=root,
        preserved_untracked=preserved_untracked,
        evidence_payload=payload,
    )
    _verify_digest(
        payload,
        _EXPECTED_EVIDENCE_SHA256,
        _EXPECTED_EVIDENCE_SHA256,
        "implementation evidence",
    )


def _production_capability_history(
    raw: object,
    current_phase: int,
) -> tuple[str, ...]:
    """Return the exact production-capability state through this phase."""
    if not isinstance(raw, list) or len(raw) != current_phase + 1:
        raise AcceptanceVerificationError(
            "production capability history must contain every implemented"
            " phase"
        )
    states: list[str] = []
    for expected_phase, value in enumerate(raw):
        if not isinstance(value, dict):
            raise AcceptanceVerificationError(
                "production capability history entries must be objects"
            )
        entry = cast(dict[str, object], value)
        _exact_keys(
            entry,
            {"phase", "state"},
            "production capability history entry",
        )
        phase = _phase(entry.get("phase"), "production capability phase")
        if phase != expected_phase:
            raise AcceptanceVerificationError(
                "production capability phases must be contiguous"
            )
        state = _nonempty_string(
            entry.get("state"), "production capability state"
        )
        expected_state = (
            "absent"
            if phase == 0
            else "active" if phase == _MAX_PHASE else "dormant_unadvertised"
        )
        if state != expected_state:
            raise AcceptanceVerificationError(
                "production capability activated outside the atomic"
                f" boundary: phase={phase}, expected={expected_state},"
                f" observed={state}"
            )
        states.append(state)
    return tuple(states)


def _validate_live_boundary(
    root: Path,
    declared_paths: Sequence[str],
    declared_source_paths: Sequence[str],
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
    source_directory_claims = tuple(
        path for path in declared_source_paths if path.endswith("/")
    )
    source_changes = {
        next(
            (
                prefix
                for prefix in source_directory_claims
                if path.startswith(prefix)
            ),
            path,
        )
        for path in live_files
        if path == "src" or path.startswith("src/")
    }
    if source_changes != set(declared_source_paths):
        raise AcceptanceVerificationError(
            "live production source changes differ from implementation"
            f" evidence: declared={sorted(declared_source_paths)},"
            f" live={sorted(source_changes)}"
        )


def _validate_pending_structural_inventory(raw: object, root: Path) -> None:
    """Validate pending source inventory without relying on key order."""
    inventory = _evidence_mapping(raw, "pending structural inventory")
    _exact_keys(
        inventory,
        {
            "source_inventory_sha256",
            "source_file_count",
            "statement_count",
            "excluded_line_count",
        },
        "pending structural inventory",
    )
    observed = (
        _sha256_string(
            inventory.get("source_inventory_sha256"),
            "pending source inventory SHA-256",
        ),
        _nonnegative_int(
            inventory.get("source_file_count"),
            "pending source file count",
        ),
        _nonnegative_int(
            inventory.get("statement_count"),
            "pending statement count",
        ),
        _nonnegative_int(
            inventory.get("excluded_line_count"),
            "pending excluded line count",
        ),
    )
    if (
        observed != _EXPECTED_PENDING_SOURCE_INVENTORY
        or _source_statement_inventory(root)
        != _EXPECTED_PENDING_SOURCE_INVENTORY
    ):
        raise AcceptanceVerificationError(
            "pending structural inventory differs from the live source tree"
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


def _validate_review_history(
    raw: object,
    raw_digest: object,
    raw_phase0_digest: object,
    raw_phase1_digest: object,
    raw_phase2_digest: object,
    current_phase: int,
    implementation_owner: str,
) -> None:
    if not isinstance(raw, list) or not raw:
        raise AcceptanceVerificationError(
            "implementation evidence review history must be non-empty"
        )
    _verify_digest(
        raw[:1],
        raw_phase0_digest,
        _EXPECTED_PHASE0_REVIEW_SHA256,
        "phase-0 review prefix",
    )
    if len(raw) < 5:
        raise AcceptanceVerificationError(
            "review history lost its phase-1 prefix"
        )
    _verify_digest(
        raw[:5],
        raw_phase1_digest,
        _EXPECTED_PHASE1_REVIEW_SHA256,
        "phase-1 review prefix",
    )
    if len(raw) < 7:
        raise AcceptanceVerificationError(
            "review history lost its phase-2 pending prefix"
        )
    _verify_digest(
        raw[:7],
        _EXPECTED_PHASE2_PENDING_REVIEW_SHA256,
        _EXPECTED_PHASE2_PENDING_REVIEW_SHA256,
        "phase-2 pending review prefix",
    )
    if len(raw) < 9:
        raise AcceptanceVerificationError(
            "review history lost its phase-2 prefix"
        )
    _verify_digest(
        raw[:9],
        raw_phase2_digest,
        _EXPECTED_PHASE2_REVIEW_SHA256,
        "phase-2 review prefix",
    )
    latest_status: dict[tuple[int, str], str] = {}
    recorded_times: list[str] = []
    for expected_sequence, value in enumerate(raw):
        record = _evidence_mapping(value, "review history record")
        _exact_keys(
            record,
            {
                "sequence",
                "phase",
                "role",
                "reviewer",
                "status",
                "recorded_at",
                "evidence",
            },
            "review history record",
        )
        if record.get("sequence") != expected_sequence:
            raise AcceptanceVerificationError(
                "review history sequences must be contiguous and append-only"
            )
        phase = _phase(record.get("phase"), "review history phase")
        if phase > current_phase:
            raise AcceptanceVerificationError(
                "review history phase is not implemented"
            )
        role = _nonempty_string(record.get("role"), "review role")
        reviewer = _nonempty_string(record.get("reviewer"), "reviewer")
        status = _nonempty_string(record.get("status"), "review status")
        recorded_at = _nonempty_string(
            record.get("recorded_at"),
            "review recorded_at",
        )
        evidence = _nonempty_string(
            record.get("evidence"),
            "review evidence",
        )
        if len(evidence) < 20:
            raise AcceptanceVerificationError(
                "review history evidence must be concrete"
            )
        if reviewer == implementation_owner:
            raise AcceptanceVerificationError(
                "implementation owner cannot review its own evidence"
            )
        if expected_sequence >= len(_EXPECTED_REVIEW_OCCURRENCES):
            raise AcceptanceVerificationError(
                "review history contains an unexpected occurrence"
            )
        expected_occurrence = _EXPECTED_REVIEW_OCCURRENCES[expected_sequence]
        if (phase, role, reviewer, status) != expected_occurrence:
            raise AcceptanceVerificationError(
                "review history occurrence identity or status changed"
            )
        if status not in {"pending", "approved", "rejected"}:
            raise AcceptanceVerificationError(
                f"invalid review status: {status}"
            )
        key = (phase, role)
        previous = latest_status.get(key)
        if previous is None:
            if phase == 0 and status != "approved":
                raise AcceptanceVerificationError(
                    "phase-0 review must preserve its approval"
                )
            direct_current_semantic_approval = (
                phase == current_phase
                and role == "semantic"
                and status == "approved"
            )
            if (
                phase > 0
                and status != "pending"
                and not direct_current_semantic_approval
            ):
                raise AcceptanceVerificationError(
                    "new review roles must begin pending"
                )
        elif previous != "pending" or status not in {"approved", "rejected"}:
            raise AcceptanceVerificationError(
                "review history rewrites or extends a terminal decision"
            )
        latest_status[key] = status
        recorded_times.append(recorded_at)
    if recorded_times != sorted(recorded_times):
        raise AcceptanceVerificationError(
            "review history timestamps must be monotonic"
        )
    if latest_status.get((0, "baseline")) != "approved":
        raise AcceptanceVerificationError("phase-0 review approval is missing")
    expected_current_statuses = {
        "semantic": _EXPECTED_CURRENT_SEMANTIC_REVIEW_STATUS,
        "gate": _EXPECTED_CURRENT_GATE_REVIEW_STATUS,
    }
    for role, expected_status in expected_current_statuses.items():
        if latest_status.get((current_phase, role)) != expected_status:
            raise AcceptanceVerificationError(
                f"current {role} review status is not {expected_status}"
            )
    _verify_digest(
        raw,
        raw_digest,
        _EXPECTED_REVIEW_HISTORY_SHA256,
        "review history",
    )


def _validate_quality_history(
    raw: object,
    raw_digest: object,
    current_phase: int,
) -> None:
    """Validate append-only digests for prior completed quality records."""
    if not isinstance(raw, list) or len(raw) != max(0, current_phase - 1):
        raise AcceptanceVerificationError(
            "quality history must preserve every prior completed gate"
        )
    for expected_phase, value in enumerate(raw, start=1):
        record = _evidence_mapping(value, "quality history record")
        _exact_keys(
            record,
            {
                "phase",
                "state",
                "quality_gate_sha256",
                "evidence_sha256",
            },
            "quality history record",
        )
        phase = _phase(record.get("phase"), "quality history phase")
        if phase != expected_phase or record.get("state") != "complete":
            raise AcceptanceVerificationError(
                "quality history phases must be contiguous completed gates"
            )
        quality_digest = _sha256_string(
            record.get("quality_gate_sha256"),
            "historical quality gate SHA-256",
        )
        evidence_digest = _sha256_string(
            record.get("evidence_sha256"),
            "historical evidence SHA-256",
        )
        if quality_digest == evidence_digest:
            raise AcceptanceVerificationError(
                "quality history cannot reuse its evidence digest"
            )
        expected_historical_digests = {
            1: (
                _EXPECTED_PHASE1_QUALITY_SHA256,
                _EXPECTED_PHASE1_EVIDENCE_SHA256,
            ),
            2: (
                _EXPECTED_PHASE2_QUALITY_SHA256,
                _EXPECTED_PHASE2_EVIDENCE_SHA256,
            ),
        }
        expected_digests = expected_historical_digests.get(phase)
        if (
            expected_digests is not None
            and (
                quality_digest,
                evidence_digest,
            )
            != expected_digests
        ):
            raise AcceptanceVerificationError(
                f"quality history lost its phase-{phase} record"
            )
    _verify_digest(
        raw,
        raw_digest,
        _EXPECTED_QUALITY_HISTORY_SHA256,
        "quality history",
    )


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
    active_pytest_instances: int,
    active_type_fixtures: int,
    root: Path,
    preserved_untracked: tuple[str, ...],
    evidence_payload: dict[str, object],
) -> None:
    quality_gate = _evidence_mapping(raw, "quality gate")
    _exact_keys(
        quality_gate,
        {
            "state",
            "required_commands",
            "state_details",
            "results",
            "tree_binding",
            "coverage_binding",
        },
        "quality gate",
    )
    state = _nonempty_string(quality_gate.get("state"), "quality state")
    if state not in {"pending", "complete"}:
        raise AcceptanceVerificationError(
            f"invalid quality gate evidence state: {state}"
        )
    required_commands = _string_list(
        quality_gate.get("required_commands"),
        "required quality commands",
    )
    _unique(required_commands, "required quality command")
    if len(required_commands) != 8:
        raise AcceptanceVerificationError(
            "implementation evidence must require eight exact gate commands"
        )
    if required_commands[:7] != _EXPECTED_ORDERED_COMMON_GATE_COMMANDS:
        raise AcceptanceVerificationError(
            "required common quality commands changed order or identity"
        )
    required = frozenset(required_commands)
    if not _EXPECTED_COMMON_GATE_COMMANDS <= required:
        raise AcceptanceVerificationError(
            "implementation evidence omits a common gate command"
        )
    focused = required - _EXPECTED_COMMON_GATE_COMMANDS
    if len(focused) != 1 or not next(iter(focused)).startswith(
        "poetry run pytest --verbose -s tests/"
    ):
        raise AcceptanceVerificationError(
            "implementation evidence lacks one exact focused pytest command"
        )
    raw_results = quality_gate.get("results")
    if not isinstance(raw_results, list):
        raise AcceptanceVerificationError(
            "implementation evidence quality results must be a list"
        )
    state_details = quality_gate.get("state_details")
    tree_binding = quality_gate.get("tree_binding")
    coverage_binding = quality_gate.get("coverage_binding")
    if state == "pending":
        details = _evidence_mapping(state_details, "pending quality state")
        _exact_keys(
            details,
            {"requested_at", "reason"},
            "pending quality state",
        )
        _nonempty_string(details.get("requested_at"), "quality requested_at")
        reason = _nonempty_string(details.get("reason"), "quality reason")
        if len(reason) < 20:
            raise AcceptanceVerificationError(
                "pending quality evidence requires a concrete reason"
            )
        if raw_results or tree_binding != {} or coverage_binding != {}:
            raise AcceptanceVerificationError(
                "pending quality evidence cannot claim completed results or"
                " bindings"
            )
        return

    details = _evidence_mapping(state_details, "complete quality state")
    _exact_keys(
        details,
        {"completed_at", "gate_run_id"},
        "complete quality state",
    )
    _nonempty_string(details.get("completed_at"), "quality completed_at")
    gate_run_id = _nonempty_string(
        details.get("gate_run_id"),
        "quality gate_run_id",
    )
    if len(gate_run_id) < 12:
        raise AcceptanceVerificationError(
            "completed quality evidence requires a concrete gate run ID"
        )
    if len(raw_results) != len(required_commands):
        raise AcceptanceVerificationError(
            "completed quality evidence lacks exact gate results"
        )
    results: dict[str, dict[str, object]] = {}
    observed_commands: list[str] = []
    for value in raw_results:
        result = _evidence_mapping(value, "quality gate result")
        command = _nonempty_string(result.get("command"), "quality command")
        observed_commands.append(command)
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
    if tuple(observed_commands) != required_commands:
        raise AcceptanceVerificationError(
            "completed quality results must preserve required command order"
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
                "subtests_passed",
                "seconds",
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
                "subtests_passed",
                "deselected",
                "xfail",
                "xpass",
            )
        }
        _positive_number(result.get("seconds"), f"quality seconds: {command}")
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
            "source_files",
            "missing_lines",
            "missing_files",
            "passed",
            "skipped",
            "subtests_passed",
            "seconds",
        },
        "exact coverage evidence",
    )
    exact_test_counts = {
        name: _nonnegative_int(
            exact_coverage.get(name),
            f"exact coverage {name}",
        )
        for name in ("passed", "skipped", "subtests_passed")
    }
    _positive_number(
        exact_coverage.get("seconds"),
        "exact coverage seconds",
    )
    if exact_test_counts["passed"] == 0:
        raise AcceptanceVerificationError(
            "exact coverage evidence has no passing tests"
        )
    try:
        verified_coverage = verify_src_coverage(
            report_path=root / "coverage.json",
            repo_root=root,
        )
    except CoverageVerificationError as exc:
        raise AcceptanceVerificationError(
            f"live exact source coverage is invalid: {exc}"
        ) from exc
    derived_coverage = {
        "covered_statements": verified_coverage.summary.covered_lines,
        "total_statements": verified_coverage.summary.num_statements,
        "source_files": len(verified_coverage.files),
        "missing_lines": verified_coverage.summary.missing_lines,
        "missing_files": 0,
    }
    observed_coverage_result = {
        key: exact_coverage.get(key) for key in derived_coverage
    }
    if observed_coverage_result != derived_coverage:
        raise AcceptanceVerificationError(
            "exact source-coverage evidence differs from the validated live"
            " report"
        )
    acceptance = results[
        "poetry run python scripts/verify_input_acceptance.py"
        " --through-phase 3"
    ]
    _exact_keys(
        acceptance,
        {"command", "exit_code", "active_nodes", "active_instances"},
        "acceptance evidence",
    )
    if (
        acceptance.get("active_nodes") != active_acceptance_nodes
        or acceptance.get("active_instances") != active_pytest_instances
    ):
        raise AcceptanceVerificationError(
            "acceptance gate evidence has stale node or instance counts"
        )
    type_result = results["make typecheck-input-contract INPUT_PHASE=3"]
    _exact_keys(
        type_result,
        {"command", "exit_code", "active_fixtures"},
        "type evidence",
    )
    if type_result.get("active_fixtures") != active_type_fixtures:
        raise AcceptanceVerificationError(
            "type gate evidence has a stale fixture count"
        )
    lint = results["make lint"]
    _exact_keys(
        lint,
        {
            "command",
            "exit_code",
            "source_files_typechecked",
            "script_files_typechecked",
        },
        "lint quality evidence",
    )
    lint_source_files = _nonnegative_int(
        lint.get("source_files_typechecked"),
        "lint source files typechecked",
    )
    lint_script_files = _nonnegative_int(
        lint.get("script_files_typechecked"),
        "lint script files typechecked",
    )
    if lint_source_files == 0 or lint_script_files == 0:
        raise AcceptanceVerificationError(
            "lint quality evidence has empty typechecked inventories"
        )
    _exact_keys(
        results["git diff --check"],
        {"command", "exit_code"},
        "quality command evidence",
    )
    live_tree_binding = _current_tree_binding(
        root,
        preserved_untracked,
        evidence_payload,
    )
    if tree_binding != live_tree_binding:
        raise AcceptanceVerificationError(
            "completed quality evidence does not match the live git tree"
        )
    coverage = _evidence_mapping(
        coverage_binding,
        "coverage binding",
    )
    _exact_keys(
        coverage,
        {
            "report_sha256",
            "source_inventory_sha256",
            "source_file_count",
            "statement_count",
            "excluded_line_count",
        },
        "coverage binding",
    )
    report_digest = _sha256_string(
        coverage.get("report_sha256"),
        "coverage report SHA-256",
    )
    inventory_digest = _sha256_string(
        coverage.get("source_inventory_sha256"),
        "coverage source inventory SHA-256",
    )
    if report_digest == "0" * 64 or report_digest == inventory_digest:
        raise AcceptanceVerificationError(
            "coverage report digest is missing or reused"
        )
    live_inventory = _source_statement_inventory(root)
    live_report = _coverage_report_binding(root)
    if report_digest != live_report[0]:
        raise AcceptanceVerificationError(
            "coverage report digest does not match the live report"
        )
    if live_report[1:] != live_inventory:
        raise AcceptanceVerificationError(
            "coverage report source inventory differs from live source"
        )
    if (
        len(verified_coverage.files) != live_inventory[1]
        or verified_coverage.summary.num_statements != live_inventory[2]
        or verified_coverage.summary.excluded_lines != live_inventory[3]
    ):
        raise AcceptanceVerificationError(
            "validated exact coverage inventory differs from live source"
        )
    expected_coverage = {
        "source_inventory_sha256": live_inventory[0],
        "source_file_count": live_inventory[1],
        "statement_count": live_inventory[2],
        "excluded_line_count": live_inventory[3],
    }
    observed_coverage = {key: coverage.get(key) for key in expected_coverage}
    if observed_coverage != expected_coverage:
        raise AcceptanceVerificationError(
            "coverage source inventory does not match the live source tree"
        )
    if exact_coverage.get("total_statements") != live_inventory[2]:
        raise AcceptanceVerificationError(
            "exact coverage statement count differs from its source inventory"
        )
    if lint_source_files != live_inventory[1]:
        raise AcceptanceVerificationError(
            "lint source-file count differs from the live source inventory"
        )


def _current_tree_binding(
    root: Path,
    preserved_untracked: tuple[str, ...],
    evidence_payload: dict[str, object],
) -> dict[str, object]:
    evidence_path = "tests/fixtures/input/baseline_evidence.json"
    verifier_path = "scripts/verify_input_acceptance.py"
    ignored_paths = frozenset((evidence_path, verifier_path))
    tracked_modes = _git_stage_modes(root)
    untracked = _git_null_paths(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
    )
    tracked_paths = set(tracked_modes)
    untracked_paths = set(untracked)
    if tracked_paths & untracked_paths:
        raise AcceptanceVerificationError(
            "git tree inventory classifies one path twice"
        )
    included_untracked = {
        relative
        for relative in untracked_paths
        if not any(
            relative.startswith(prefix) for prefix in preserved_untracked
        )
    }
    staged_changed = set(
        _git_null_paths(
            root,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--",
        )
    )
    worktree_changed = set(
        _git_null_paths(
            root,
            "diff",
            "--name-only",
            "-z",
            "--",
        )
    )
    ambiguous_paths = staged_changed & worktree_changed
    if ambiguous_paths:
        raise AcceptanceVerificationError(
            "git tree inventory has staged and unstaged changes for: "
            + ", ".join(sorted(ambiguous_paths))
        )
    inventory: list[dict[str, object]] = []
    resolved_root = root.resolve()
    for relative in sorted(
        (tracked_paths | included_untracked) - ignored_paths
    ):
        pure_path = PurePosixPath(relative)
        if (
            not relative
            or pure_path.is_absolute()
            or "." in pure_path.parts
            or ".." in pure_path.parts
        ):
            raise AcceptanceVerificationError(
                f"unsafe git tree inventory path: {relative}"
            )
        path = resolved_root.joinpath(*pure_path.parts)
        if path.is_symlink():
            raise AcceptanceVerificationError(
                f"git tree inventory entry is a symlink: {relative}"
            )
        if not path.exists():
            if relative in tracked_paths:
                continue
            raise AcceptanceVerificationError(
                f"untracked git tree entry disappeared: {relative}"
            )
        kind = _tree_entry_kind(path, resolved_root, relative)
        if (
            relative in tracked_modes
            and kind != tracked_modes[relative]
            and relative not in worktree_changed
        ):
            raise AcceptanceVerificationError(
                f"git index and working-tree modes differ for: {relative}"
            )
        inventory.append(
            {
                "path": relative,
                "kind": kind,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    normalized_evidence = deepcopy(evidence_payload)
    normalized_quality = normalized_evidence.get("quality_gate")
    if not isinstance(normalized_quality, dict):
        raise AcceptanceVerificationError(
            "cannot normalize quality evidence tree binding"
        )
    cast(dict[str, object], normalized_quality)["tree_binding"] = {}
    normalized_evidence_kind = _bound_tree_entry_kind(
        resolved_root / evidence_path,
        resolved_root,
        evidence_path,
        tracked_modes,
        worktree_changed,
    )
    evidence_digest = sha256(
        dumps(
            normalized_evidence,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    verifier_file = resolved_root / verifier_path
    normalized_verifier_kind = _bound_tree_entry_kind(
        verifier_file,
        resolved_root,
        verifier_path,
        tracked_modes,
        worktree_changed,
    )
    verifier_source = verifier_file.read_text(encoding="utf-8")
    normalized_verifier = verifier_source.replace(
        _EXPECTED_EVIDENCE_SHA256,
        "0" * 64,
    )
    verifier_digest = sha256(normalized_verifier.encode("utf-8")).hexdigest()
    inventory_digest = sha256(
        dumps(
            inventory,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    values: dict[str, object] = {
        "baseline_head": _EXPECTED_BASELINE_HEAD,
        "inventory_file_count": len(inventory),
        "inventory_sha256": inventory_digest,
        "normalized_evidence_kind": normalized_evidence_kind,
        "normalized_evidence_sha256": evidence_digest,
        "normalized_verifier_kind": normalized_verifier_kind,
        "normalized_verifier_sha256": verifier_digest,
    }
    values["tree_sha256"] = sha256(
        dumps(
            values,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return values


def _tree_entry_kind(path: Path, root: Path, relative: str) -> str:
    if path.is_symlink():
        raise AcceptanceVerificationError(
            f"git tree inventory entry is a symlink: {relative}"
        )
    if not path.exists():
        raise AcceptanceVerificationError(
            f"git tree inventory entry is missing: {relative}"
        )
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise AcceptanceVerificationError(
            f"git tree inventory path escapes repository: {relative}"
        ) from exc
    if resolved_path != path or not path.is_file():
        raise AcceptanceVerificationError(
            f"git tree inventory entry is not a regular file: {relative}"
        )
    mode = path.stat().st_mode
    if mode & (S_IXUSR | S_IXGRP | S_IXOTH):
        return "executable"
    return "regular"


def _bound_tree_entry_kind(
    path: Path,
    root: Path,
    relative: str,
    tracked_modes: dict[str, str],
    worktree_changed: set[str],
) -> str:
    kind = _tree_entry_kind(path, root, relative)
    if (
        relative in tracked_modes
        and kind != tracked_modes[relative]
        and relative not in worktree_changed
    ):
        raise AcceptanceVerificationError(
            f"git index and working-tree modes differ for: {relative}"
        )
    return kind


def _git_stage_modes(root: Path) -> dict[str, str]:
    raw = _git_bytes(root, "ls-files", "--stage", "-z", "--")
    if not raw:
        return {}
    if not raw.endswith(b"\0"):
        raise AcceptanceVerificationError(
            "git index inventory is not NUL terminated"
        )
    values: dict[str, str] = {}
    for raw_entry in raw.removesuffix(b"\0").split(b"\0"):
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_id, stage = metadata.split(b" ")
            relative = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise AcceptanceVerificationError(
                "git index inventory contains an invalid entry"
            ) from exc
        if (
            stage != b"0"
            or len(object_id) not in {40, 64}
            or any(value not in b"0123456789abcdef" for value in object_id)
        ):
            raise AcceptanceVerificationError(
                f"git index inventory contains an unresolved entry: {relative}"
            )
        match mode:
            case b"100644":
                kind = "regular"
            case b"100755":
                kind = "executable"
            case _:
                raise AcceptanceVerificationError(
                    f"git index inventory contains an unsafe mode: {relative}"
                )
        if relative in values:
            raise AcceptanceVerificationError(
                f"git index inventory contains a duplicate path: {relative}"
            )
        values[relative] = kind
    return values


def _git_null_paths(root: Path, *arguments: str) -> tuple[str, ...]:
    raw = _git_bytes(root, *arguments)
    if not raw:
        return ()
    if not raw.endswith(b"\0"):
        raise AcceptanceVerificationError(
            "git tree inventory is not NUL terminated"
        )
    try:
        values = tuple(
            value.decode("utf-8")
            for value in raw.removesuffix(b"\0").split(b"\0")
        )
    except UnicodeDecodeError as exc:
        raise AcceptanceVerificationError(
            "git tree inventory contains a non-UTF-8 path"
        ) from exc
    if len(set(values)) != len(values):
        raise AcceptanceVerificationError(
            "git tree inventory contains a duplicate path"
        )
    return values


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = run(
        ("git", *arguments),
        cwd=root,
        capture_output=True,
        check=False,
        text=False,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise AcceptanceVerificationError(
            "cannot verify live git tree binding:"
            f" git {' '.join(arguments)}: {detail}"
        )
    return completed.stdout


def _source_statement_inventory(root: Path) -> tuple[str, int, int, int]:
    source_root = root / "src"
    if not source_root.is_dir():
        raise AcceptanceVerificationError(
            "coverage source inventory root is missing"
        )
    analyzer = Coverage(config_file=False, data_file=None)
    inventory: list[dict[str, object]] = []
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            _, statements, excluded, _, _ = analyzer.analysis2(str(path))
        except Exception as exc:
            raise AcceptanceVerificationError(
                f"cannot analyze coverage source inventory: {relative}: {exc}"
            ) from exc
        inventory.append(
            {
                "path": relative,
                "statements": len(statements),
                "excluded_lines": len(excluded),
            }
        )
    digest = sha256(
        dumps(
            inventory,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return (
        digest,
        len(inventory),
        sum(cast(int, entry["statements"]) for entry in inventory),
        sum(cast(int, entry["excluded_lines"]) for entry in inventory),
    )


def _coverage_report_binding(
    root: Path,
) -> tuple[str, str, int, int, int]:
    report_path = root / "coverage.json"
    if not report_path.is_file():
        raise AcceptanceVerificationError(
            "completed quality evidence requires the live coverage report"
        )
    report_digest = sha256(report_path.read_bytes()).hexdigest()
    report = _strict_mapping(report_path, "coverage report evidence")
    raw_files = report.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise AcceptanceVerificationError(
            "coverage report evidence has no source files"
        )
    inventory: list[dict[str, object]] = []
    for relative, raw in sorted(raw_files.items()):
        if not isinstance(relative, str) or not relative.startswith("src/"):
            raise AcceptanceVerificationError(
                f"coverage report contains a non-source path: {relative!r}"
            )
        if not isinstance(raw, dict):
            raise AcceptanceVerificationError(
                f"coverage report file entry must be an object: {relative}"
            )
        summary = cast(dict[str, object], raw).get("summary")
        if not isinstance(summary, dict):
            raise AcceptanceVerificationError(
                f"coverage report file summary is missing: {relative}"
            )
        statements = summary.get("num_statements")
        excluded = summary.get("excluded_lines")
        if (
            type(statements) is not int
            or statements < 0
            or type(excluded) is not int
            or excluded < 0
        ):
            raise AcceptanceVerificationError(
                f"coverage report file counts are invalid: {relative}"
            )
        inventory.append(
            {
                "path": relative,
                "statements": statements,
                "excluded_lines": excluded,
            }
        )
    inventory_digest = sha256(
        dumps(
            inventory,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return (
        report_digest,
        inventory_digest,
        len(inventory),
        sum(cast(int, entry["statements"]) for entry in inventory),
        sum(cast(int, entry["excluded_lines"]) for entry in inventory),
    )


def _sha256_string(value: object, label: str) -> str:
    digest = _nonempty_string(value, label)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise AcceptanceVerificationError(
            f"{label} must be lowercase hexadecimal"
        )
    return digest


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
) -> tuple[str, ...]:
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
    collected = tuple(observed)
    _verify_identical_nodes(expected, collected, "collected")
    return collected


def _verify_execution(
    expected: tuple[str, ...],
    payload: dict[str, object],
    collected: tuple[str, ...],
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
    _verify_identical_nodes(expected, collected, "collected")
    raw_items = _string_list(payload.get("items"), "execution items")
    _verify_identical_nodes(collected, raw_items, "executed")
    raw_reports = payload.get("reports")
    if not isinstance(raw_reports, list):
        raise AcceptanceVerificationError("execution reports must be a list")
    by_node: dict[str, list[dict[str, object]]] = {
        node: [] for node in raw_items
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
        if (
            phases.count("setup") != 1
            or phases.count("call") < 1
            or phases.count("teardown") != 1
            or set(phases) != {"setup", "call", "teardown"}
        ):
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


def _verify_identical_nodes(
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


def _positive_number(value: object, label: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
    ):
        raise AcceptanceVerificationError(f"{label} must be positive")
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
    instance_count = len(manifest.active_pytest_instances(args.through_phase))
    print(
        "structured-input acceptance passed: "
        f"through_phase={args.through_phase} nodes={active_count}"
        f" instances={instance_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
